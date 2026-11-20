<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Sparse attention in KVCacheManagerV2

Target: the **decode instance of a disaggregated deployment**.

**Implemented**: the resolve, copy, index and flush-list kernels; `Batch`; the per-request
reservation and device state; batch-wide write-back through `Batch::flushSparseUnits()`
or `Batch::advanceSparseHistory()`; `fetch_sparse_pages` and
`prepare_sparse_write_indices`; sparse buffers across more than one pool; sparsity as a pool
attribute, so one layer may hold both sparse and dense buffers.

Compression is a separate feature, configured through
`KvCacheCompressionConfig` rather than through anything here, and needs nothing
sparse-specific -- see *Compression*.

A separate instance serves prefill. The GPU cache holds one resident set per
`(request, layer)`, which excludes per-KV-head selection. See *Device-side cache*. Kernel fusion
remains future work; so do multi-beam requests. `ISparseUnitGather` is retired — see
*Extraction interface*.

Read `AGENTS.md` in this directory first for the storage, locking and concurrency model
this builds on.

## Problem

Sparse-attention decoding reads only a small, per-step, per-layer subset of a sequence's
history. Keeping all of that history in GPU memory wastes the resource the sparse
algorithm exists to conserve: a request's memory bill still grows with its full context,
so decoding hits a capacity wall long before it runs out of compute.

The goal is to hold history on the host tier and let GPU residency be bounded by the
selection width rather than by context length.

Two existing properties make this cheap:

- The host tier is registered `CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP`
  (`utils/hostMem.cpp:305`), so a kernel can read a host page directly.
  `isGpuAccessibleMemory` (`storageManager.cpp:138`) already relies on this for the
  cold-page codec, which encodes and decodes against host memory with no staging buffer.
- Selection output is a device array. If everything the fetch path needs is also on the
  device, the host never enters the loop, and the operation is stream-ordered and
  capturable in a CUDA graph.

## Vocabulary

| term | meaning |
| --- | --- |
| page | the KVCM2 allocation, locking and migration unit: `tokens_per_block` tokens |
| buffer | one layer's slice of a page; several coalesce into one page |
| selection unit | what the backend selects; a factor of `tokens_per_block`, never larger |
| in-flight | units in the published span from device history (inclusive) to capacity (exclusive), both rounded down to units |
| units per page | selection units a page holds, `tokens_per_block / sparse_selection_unit_tokens` |
| pure history | a unit whose ordinal is below the row's device history length divided by `unit_tokens` (rounded down) |

## Scope

Sparse-attention families differ in what they select, and the difference is load-bearing:

| family | selection unit | models |
| --- | --- | --- |
| MSA | block, 16 × 128 tokens (`minimax_m3/common.py:70`) | MiniMax-M3 |
| NSA, MoBA | block | DeepSeek-V4-Flash, Kimi |
| Quest | page | training-free, any model |
| DSA | **token**, k = 2048 | DeepSeek-V3.2, V4.1, GLM-5.x |

Block-level families work with the selection unit equal to one page and need nothing
special. DSA needs a sub-page unit, which this design supports through the selection-unit
mechanism below, at the cost of a finer cache entry. First implementation should target
the block-level families; DSA follows without an API change.

DeepSeek-V4's compressed pools are worth noting as a middle case: per-layer
`compress_ratios` with `compressed_block_sizes = tokens_per_block // ratio`
(`deepseek_v4/cache_manager.py:370`), so at ratio 128 a compressed page holds exactly one
entry and the selection unit is already a page.

## Shape of the design

1. **No new tier or medium.** What varies is the *memory kind of a pool group*: a sparse
   life cycle's level-0 pool group is backed by host memory, a dense one by GPU memory.
   Level 1 stays an ordinary codec target for both.
2. **`BufferConfig.is_sparse` is a per-buffer flag** and selects that backing. A sparse
   buffer requires a lifecycle of its own. Sparse buffers of the same size still coalesce
   with each other.
3. **Every page of a sparse buffer locks at level 0**, like every other page. Sparse pages
   are ordinary hot pages — coalesced, addressed by page index, reachable through
   `get_mem_pool_base_address`, and visible to the transceiver — that happen to live in
   host memory.
4. **All GPU residency is the per-request cache**, which is a dedicated arena rather than a
   storage tier. A sparse buffer has no separately managed GPU pages: both the block being
   written and the history a step selects occupy cache cells, drawn from the same lease and
   replaced by the same eviction.
5. **A completed unit can be fetched from its host page.** The batch queues write-back
   of units crossed by a history advance before the next fetch. Until device history
   passes a unit, its GPU cell holds the request's live KV and stays pinned.
6. **Extraction is pluggable**, because KVCM2 buffers are opaque bytes and a sub-page unit
   cannot be located without layout knowledge.

## Locking and flush

`acquirePageIndex` already writes the page's *current* slot id with no level assumption
(`page.cpp:401`):

```cpp
int old = kvc->updateBasePageIndex(
    mUser.beamIndex, mUser.ordinal, mUser.lifeCycle, slotIdToPageIndexValue(pg.slotId()));
```

so `getBasePageIndices()` reports hot slot ids with no change, and because every page of a
sparse buffer locks there, the table is **uniform**: every entry is a level-0 slot id, as
for a dense buffer. Nothing resolves against the cache arena, and no caller has to know
which rule applies to which ordinal.

A block's data reaches its host page by a **flush**: a copy on the Batch's stream that does
**not** evict the GPU entry, which stays cached and becomes evictable once device history passes its unit.

### Who flushes

The batch orders write-back of newly completed units before the next fetch on its CUDA
stream. Both paths use the page indices from the last `publish()`.

- **Host-managed history.** The caller passes the new per-row lengths to
  `Batch::flushSparseUnits(newHistoryLengths)` before `resize()`. When a seated row's history
  crosses a unit boundary, `resize()` checks that the recorded flush covers the requested
  history (`newHistory <= recorded`). Several resizes can use the same flush, including
  a retry that advances history by fewer tokens.
- **Device-managed history.** `Batch::advanceSparseHistory()` takes the kept-token counts
  from the GPU, advances each row's device watermark, then queues write-back of the units
  it crossed. `publish()` preserves that watermark. Once the host reads the count,
  `resize()` records the accepted history in the member's `KvCache` without a second flush.

**Related page operations**

- `_unlockStaleBlocks` exists for SWA blocks outside the window, whose content is *dead*;
  sparse history is live and will be read again. The trigger and the range arithmetic match,
  the disposition does not — routing sparse layers through it would drop the data rather
  than write it back. It also gates on `windowSize.has_value()`, so for a full-attention
  sparse layer it fires on nothing.
- `_shortcutSetHistoryLength` returns "unchanged" when no SWA stale range moved, and for a
  full-attention layer `windowSize.has_value()` is false, so nothing is ever reported as
  moved: it assigns `mHistoryLength`, returns true, and `resize()` returns before reaching
  its body. Work placed in that body does not run for a sparse layer.
- `commit()` can share a completed page because `num_committed_tokens <= history_length`.
  When it raises the member's host history length, it does so through `resize()`.

### How much a step may cross

Each row supplies a range of completed units. Each layer has one compact write-back list
shared by all batch rows. A row advancing `t` tokens can cross at most
`floor(t / unitTokens) + 1` unit boundaries, including a unit partly filled in the
previous step. With a batch token budget, the list capacity is therefore
`floor(maxTotalInputTokens / unitTokens) + maxBatchSize`. Without that budget the batch
reserves `maxBatchSize * maxInFlightUnits` entries per layer.

`flushSparseUnits()` checks the total units crossed across all rows against the list
capacity. For a device-managed row, the advance kernel checks
`0 <= advance <= published_capacity - device_history` and computes the row's range on
the GPU. Both paths use the same write-back list.

### Ordering

`flushSparseUnits` stages its per-row ranges on a helper stream and launches the write-back
on the Batch's stream:

```
helper: waitEvent(flushRead)          // the previous step finished reading the ranges
        cuMemcpyBatchAsync(ranges -> device, CU_MEMCPY_SRC_ACCESS_ORDER_DURING_API_CALL)
        staged = record(helper)
main:   waitEvent(staged)
        scan the flush lists, then one write-back copy per pool
        flushRead = record(main)
```

The helper stream buys both things, dominated by the first. `DURING_API_CALL` consumes the
host array *before the call returns*, so issuing the copy on the Batch's stream would hold
the host until the previous step's graph replay reached it; on a helper stream the source is
captured at once and the call returns promptly. The device write then overlaps that replay
too, since the main stream reaches its wait on `staged` only after draining what it already
had queued — though with two ints per row that part is worth a microsecond, where the
host-side decoupling is worth whatever the replay has left.

The kernel stays on the Batch's stream because the cells it reads are written there. The
wait is a stream wait, not a host sync: the hazard is the previous step's kernel still
reading the same destination, which is ordinary stream ordering. (Do not set
`CU_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE` on that copy — it breaks `DURING_API_CALL`
semantics; driver bug 6718200, see `batchedPageCopy.cu`.)

The write-back reads the index table as of the **last publish**. That is current enough
because the move is clamped to the capacity the step started from, so the blocks it touches
were allocated by an earlier resize and published then.

### Advancing sparse KV history on the GPU between MTP steps

The native `Batch` can advance a request's history from MTP verification results on the
GPU. With overlap scheduling, the next forward can be enqueued before the host reads
those results. For example, if a member starts with 100 tokens of history and verification
accepts two drafts, the current input token and those two drafts extend its history to
103. The next forward starts writing at position 103. The Python sparse-offload managers
take one-model MTP-Eagle drafting a linear chain, whose step writes its tokens at
consecutive positions and keeps a prefix of them, and reject other speculative decoding
at construction.

`Batch::advanceSparseHistory()` takes the number of input tokens each row kept, including
the current token and the accepted drafts (three tokens in this example). On the batch's
CUDA stream, a kernel adds that count to the row's device watermark. The batch then
queues write-back of each newly completed sparse unit to its host page.

`publish()` preserves the updated watermark for the next forward, so the executor can
enqueue that forward without waiting for the host to read the count and send back a new
watermark. The host updates the member's `KvCache` history length when it processes the
sampling results.

The advance kernel checks that each advance is nonnegative and fits within the row's
published capacity. The batch queues the write-back scans and copies on every step because
only the GPU knows which rows completed a sparse unit. The scan skips rows with an empty range.

**Joining the batch.** A sparse member's host pages hold its completed KV units.
When it joins, the batch assigns it GPU cache cells. Call `publish()` before its first
decode step so the GPU has the lease, page table, and starting history length.
After the first call to `advanceSparseHistory()`, `publish()` uses `kKeepSparseHistory`
to preserve the row's device watermark. `move()` rejects such a row because it would
initialize the destination from the host's potentially stale history length.

**Recording an accepted step.** In the example above, the GPU uses position 103 for the
next write even while the host still records 100. When the host reads the same step's
sampling results, it calls `resize()` to record 103 in the member's `KvCache`.
`advanceSparseHistory()` has already queued copies of any newly completed units to their
host pages. If the batch calls `flushSparseUnits()` for other rows, it must pass this
member's current host history length so the call skips write-back for this row.

**Leaving and rejoining.** `removeKvCache()` releases the member's GPU cells after the
queued write-back on the same CUDA stream. The member can leave while the host still
records 100. Before it rejoins, the host must apply all outstanding acceptance results.
Publishing 100 again would put the next write at position 100 and could cause a sparse
fetch to treat a completed unit in host memory as an in-flight unit in an empty GPU cell.

**GPU cache capacity.** The host reserves space for every position MTP may write, including
drafts that verification may reject. With overlap, it can reserve the next step's positions
before it knows how many drafts the previous step kept. `sparseMaxInFlightTokens` bounds
the difference between reserved capacity and the history length the host currently records.
The minimum cache holds one query's selection plus the maximum number of in-flight
units. See [Sizing](#sizing) for requests with several queries.

### Unflushed entries carry no state

The resolve kernel derives pinning from a resident entry's ordinal and the row's
device history.

```
pinned  <=>  entry_ordinal >= floor(device_history / unit_tokens)
```

A pinned cell holds this request's KV until device history passes the unit. A history
advance queues write-back for each newly completed unit, so a subsequent fetch can evict
its cell and later refill it from the host page. Only finalized pages enter the radix tree,
so the request owns its unflushed units exclusively.

If verification accepts only a prefix, capacity can shrink below a unit the model already
wrote. Its cell stays pinned even while the write-index output omits it. When capacity
covers that ordinal again, the model overwrites the same cell. These cells count toward
the in-flight bound. Each was allocated within `max_input_length` tokens of history, and
history only advances.

## Per-request reservation

Two separate reservations, easily confused.

**Hot pages.** Every block of the sequence has a locked hot page, so the count is
`divUp(capacity, tokens_per_block)` — the existing per-sequence locking logic, entirely
unchanged. Only the backing medium differs. This is the payoff of uniform locking: nothing
new to size, and no notion of "history pages" distinct from the rest.

**Cache cells.** `max_cached_units` is the whole cache, counting both populations:

```
cells = max_cached_units                          one buffer's cells
lease = num_buffers * divUp(cells, expansion)     contiguous cells a request takes
```

Each selected history unit and each pinned write occupy a cell. The minimum reservation
covers one query's selection and all pinned writes. See [Sizing](#sizing) for the
multi-query bound.

### `max_input_length`

`KVCacheManagerConfig.sparse_max_in_flight_tokens` sets a sparse request's read-only
`max_input_length`. `resize()` checks `capacity - history_length` using the **host**
history length. With overlap, this gap includes capacity reserved for the next step
before the host processes the previous step's accepted count.

Fetch and write-index preparation use the **device** history watermark. It can already
include the accepted tokens while the host still counts their positions in the
reservation.

The lease size depends on configuration alone. It is taken when the cache is seated in
a `Batch` and returned when it leaves, so a request holds cells exactly while it is being
stepped: one still receiving its disaggregated history, or sitting a step out, holds none.
Seating cannot fail for want of a cell, because the arena is sized from the same
`constraints` the Batch takes its rows from -- a free row therefore has a free lease, and
the fetch path never has to handle exhaustion. Because every request of a life cycle needs
exactly the same number of cells, the arena is a fixed-size-block free list rather than a
general allocator.

Leaving returns the lease with every entry free. Write-back preserves completed units in
their host pages, so a request that rejoins starts with an empty GPU cache.

### Why the cache is a flat array of cells

A lease is one contiguous run of buffer-sized cells, and an entry resolves to one by
arithmetic alone:

```
cell    = buffer * divUp(max_cached_units, expansion) + entry / expansion
address = lease_base + cell * buffer_bytes
unit    = entry % expansion                       which unit within the cell
```

No page table, no slot indirection, no coalescing. Coalescing earns nothing inside the
cache because different layers select different units, so a slot's cells would never be
filled together; the cache took whole coalesced slots only while it drew them from the
storage allocator, which deals in slots. A dedicated arena has no such constraint, and
dropping it removes a page lookup from both the copy and the index kernel.

The consequence is that **a cache cell is not a page**. It holds one buffer's
unit, identified by `(buffer, entry index)` rather than by a block ordinal. This breaks an
assumption that runs through KVCM2 — that a page is both the unit of allocation *and* the
unit of identity — which is precisely why the cache is not a storage tier: it would have to
answer "which block does this page hold", and it has no answer.

## Device-side cache

Per `(request, layer)`, managed entirely by kernels: one 32-bit word per entry, holding
the unit resident there and when that unit was last used. A free entry is a value of the
same word, so there is no separate free list.

**Scope is per `(request, layer)`, never per KV head.** Every head of a layer shares one
resident set, because a selection unit is the tokens it spans across all of them. A layer
that names a `selection_id` widens the scope rather than narrowing it: the members of a
selection share one resident set between them, one per `(request, selection)` — see
*Selection groups*. Their cells stay per layer and mirror. An
algorithm that lets different heads select different units is not expressible here, and this
is a real exclusion rather than a hypothetical: Rocket's selection indices are
`[num_kv_heads, total_tokens]` and MiniMax-M3's are `[tokens, num_kv_heads, topk]` — both
choose per head. Serving them would need a resident set per `(request, layer, head)`, which
multiplies the metadata and the reservation by the head count.

**Scope is per request.** Two requests sharing a prefix do not share a cached copy. That
closes a soundness hole by construction: entries are reachable only through their own
request's page table, so a host slot can never be recycled underneath a surviving entry.
A global, provenance-keyed cache would give cross-request sharing for free via the radix
tree, but needs generation counters or an invalidation pass; not supported.

**Replacement is by age, not "evict whatever was not selected this step".** Retention
across steps is the point: holding hot entries beyond the current top-k gives a large
miss-rate reduction (HiSparse reports the measurement), and that is only available when
the cache is larger than the selection width.

An entry's word carries a 7-bit step timestamp and one history bit saying whether its last
use found it already resident. Eviction takes the highest `2 * age - history`, so age
decides and, within one age, a unit fetched that step leaves before one that was hit — which
is what makes a repeatedly selected unit outrank a first-time selection. The order is
therefore approximate where an exact recency permutation would be exact: two entries of the
same age and the same history bit are interchangeable, and which of them leaves is whichever
thread reaches the oversubscribed bucket first.

The clock wraps, so a timestamp left untouched for longer than its range would read as a
recent one. Every resolve streams every entry anyway, so one resolve in every
`kRebaseStep` pulls the timestamps it passes up to the oldest age the field still
distinguishes. That is idempotent, which is what lets the write-index path apply it too
without the two having to agree on which of them does.

**Residency is tracked by a reverse map** — a dense array indexed by slot `0..N-1` holding
the resident unit ordinal, N entries per `(request, layer)` — not a forward map from
ordinal to slot. The two scale with different things:

| | size | at 1 M context, 256 requests, 64 layers |
| --- | --- | --- |
| forward, ordinal → slot | batch × layers × **max_ordinals** | hundreds of MB, and unbounded in context |
| reverse, slot → ordinal | batch × layers × **N** | proportional to cache already bought |

The forward map is proportional to context length, the quantity this feature exists to
stop paying for. The dense reverse form is also what the eviction pass wants: it reads every
entry once either way.

### Probe index

Answering "is this ordinal resident, and where" against a dense reverse map would be O(N)
per selection. The fetch kernel therefore builds a hash in shared memory at the top of each
launch and discards it at the end. Two properties follow from it being ephemeral: there are
no deletions, so no tombstones and none of the maintenance that made a persistent device
hash map unattractive; and nothing has to stay consistent across steps or graph replays.

**The hash is over the required ordinals** -- the in-flight units followed by the selection
-- and the resident set is then streamed past it. So the table is sized by `k` rather than
by the cache's capacity `N`:

```cpp
// Index the required ordinals, then match the resident set against that index.
for (int32_t i = threadIdx.x; i < required; i += kThreads) {
    int32_t ord = i < inFlight ? historyUnits + i : selected[i - inFlight];
    if (ord < 0) continue;
    table.insert(ord, mask);                       // one CAS per probe step
}
__syncthreads();

for (int32_t s = threadIdx.x; s < entries; s += kThreads) {   // the resident set
    uint32_t word = entry[s];
    entryCache[s] = word;
    int32_t ord = SparseEntry::ordinal(word);
    bool matched = ord >= 0 && table.publish(ord, s, mask);   // claims its query
    flag[s] = matched ? kFlagMatched : (ord >= historyUnits ? kFlagHeld : kFlagEvictable);
    if (flag[s] == kFlagEvictable) atomicAdd(&hist[staleness(word)], 1);
}
```

Open addressing with linear probing, capacity `next_pow2(2 * required)` so the load factor
stays at or below one half, which also guarantees an unsuccessful probe terminates on an
empty entry. A position is a single 32-bit word holding both the ordinal and the entry that
claimed it: a `__syncthreads()` separates build from match, so no thread reads a position
mid-insert and the entry index needs no atomicity of its own. Packing them halves the
shared-memory traffic of the publish pass, which walks the whole resident set and is where
most of the kernel's time goes. The split is 14 bits of entry index against 18 of ordinal,
and both bounds are checked rather than truncated into.

The largest unit ordinal is 262,142, exposed as `SparseCacheConfig.max_unit_ordinal`.
Offload startup checks `max_seq_len // unit_tokens` against it for every fetched layer
before sizing or allocating the pools. This includes the history watermark that names
the next in-flight unit, not just the selected history ordinals. The check is synchronized
across ranks, so a rank with an unsupported layer stops its peers before serving starts.

Multiplicative hashing rather than identity, because ordinals are small and dense, and dense
runs make an unsuccessful probe walk the whole run.

**A miss is recorded by its table position alone.** The position's word already holds the
ordinal, so the miss list keeps one 16-bit position per miss and reads the ordinal back
from the table when it assigns the victim. That bounds the table at 65,536 positions, a row
of at most 32,768 required units, which the shared-memory check enforces. At GLM-5.2 MTP3
(a row of `4 * 2048` units beside 13 in flight, a cache of 16,256 entries) the CTA takes
about 224 KiB: the 128 KiB table, 64 KiB of entry words, 16 KiB of flags and 16 KiB of
miss positions.

The table stays at `next_pow2(2 * required)` rather than being right-sized to a load factor
of one half. Right-sizing measured *slower*: the publish pass is dominated by *unsuccessful*
probes, whose expected cost is `0.5 * (1 + 1/(1-a)^2)` and so rises much faster in the load
factor `a` than a successful probe's does.

The resident sweep is also where each entry is classified and its staleness binned, so the
pass the match needs is the pass the eviction ranking needs; see "Capacity and eviction".

**The CTA width is chosen per life cycle.** Every pass but the probe is a stride loop over
the resident set, so doubling the CTA halves what a thread walks — but only pays while
shared memory already holds a multiprocessor to one or two blocks of the kernel. Past that
the wider CTA buys nothing per block and costs residency. The kernel is therefore
instantiated at 256 and 512 threads, and the launcher picks between them by asking the
device for the narrow instantiation's occupancy at this shape's shared-memory size. The
three extents are the life cycle's, so the answer is computed once and held.

**The index is keyed by the selection, not by the resident set.** The alternative — keying
by ordinal and valuing by slot, sized `next_pow2(2N)` — returns a hit's slot straight from
the probe and so saves one pass. It does not make the miss list available any sooner, as
might be supposed: that build has to walk every slot and cross a `__syncthreads()` before
its first probe, so the misses are known no earlier either way.

What keying by the selection costs, and what it buys:

- **Shared memory is O(k), not O(N).** At the DSA geometry the table is 2048 buckets rather
  than 8192. This is the point of the choice: `N` costs linear per-entry scratch rather than
  a `next_pow2`-rounded table, so the quantity bounding how large a cache one launch can
  serve is the selection width, not the capacity.
- **A third pass of size `k`** resolves hits from the table. Shared-memory only, and small
  against the resident sweep.
- **The index build cannot precede the dependency wait.** A resident-keyed build reads only
  the layer's own residency table and could run while the producing kernel was still
  draining; hashing the selection cannot, because the selection is what the producer
  produces. Only shared-memory initialisation sits above the wait.

On DSA at the geometry above, the resolve kernel takes 0.833 ms per decode step summed
across 21 sparse layers, against 1.468 ms for the resident-keyed form.

When the configured in-flight span is one unit, `prepare_sparse_write_indices` scans the
entries for that ordinal and reuses its cell or takes the stalest eligible entry. For a
wider span it runs the resolve with an empty selection. See
[Writing in-flight units](#writing-in-flight-units).

## Extraction interface

KVCM2 buffers are raw bytes with no layout information, so locating a selection unit inside
one requires the owner's knowledge. Two mechanisms, chosen per buffer, serving two
different situations.

### Declaring where a unit sits

A unit is the tokens it spans across *every* KV head, so where its bytes sit follows from
two numbers on `BufferConfig`, both consumed once at configuration:

```cpp
std::optional<int> sparseSelectionUnitTokens;  // unset: a unit is a whole page
int sparseUnitChunks = 1;                      // pieces a unit is gathered from
```

`sparseUnitChunks` is the page's dimension order expressed as a count:

- **1** for a sequence-major page (`[token][head][dim]`), where a unit's tokens carry their
  heads with them and the whole unit is one contiguous range. Also 1 for any single-head
  page — a compressed or pooled latent — where the two orders are the same bytes.
- **numHeads** for a head-major page (`[head][token][dim]`), where each head contributes
  its own run and a unit is that many strided pieces.

Everything else follows, because a page is densely packed:

```
unitsPerPage = tokens_per_block / sparseSelectionUnitTokens
chunkStride  = size / sparseUnitChunks
chunkBytes   = size / (sparseUnitChunks * unitsPerPage)
unitStride   = chunkBytes
```

The four-field `SelectionUnitLayout` the copy kernel reads is derived from these at
configuration, so the kernel is unchanged and the config surface is two integers. The
derivation assumes dense packing, so a page with an alignment gap — between heads, or
between units — is not expressible; no model in this repository has one.

**Why this is not `tokens_per_block_override`.** That knob reinterprets a buffer's page for
*storage*, and stretches its public page index space by the same factor. Sparse selection
needs only the cache's own granularity, so it gets its own field and leaves the buffer's
pages, page indices and index space on the manager's `tokens_per_block`. Setting both on
one buffer is rejected. `units per page` and `PageIndexConverter`'s `expansion` are
distinct quantities: the sparse code says `unitsPerPage`, and `expansion` belongs to the
index converter alone.

**What `sparseUnitChunks > 1` is for.** Every sparse model in the repository today passes 1:
DeepSeek V4 asserts `num_kv_heads == 1`, and GLM's kpool, QSA's index-K and MiniMax's
index-K are all single-head and sequence-major. The field earns its place on QSA, whose
selection is shared across heads (its index-K has one head) while the buffer attention then
gathers from is head-major with ordinary GQA heads — the one combination that needs strided
units. Rocket and MiniMax-M3 under MSA are also head-major with several heads, but both
select *per head*, which this cache excludes for a different reason (see *Device-side
cache*); strided units would not rescue them.

### The gather interface, and why there isn't one

Earlier revisions planned an `ISparseUnitGather` — a per-buffer, stream-taking device call
that would extract a unit from a representation arithmetic could not address. It existed for
exactly one reason: the fetch read cold pages, so a codec that compressed or permuted them
put the unit somewhere a layout value could not describe.

Reading hot pages removes the reason. The hot representation is KVCM2's own, always in
ordinary slot layout, and a codec by construction only transforms data on its way *out* of
that tier. A compressed level 1 is therefore invisible to the fetch, and no plugin boundary
is needed on this path at all.

This also retires the LTO-IR/nvJitLink fusion work that existed only to inline such a
gatherer back into the resolve. The extension point is gone rather than deferred, which is
worth more than the code it would have taken to build.

### Kernel structure

The descriptor path crosses **no plugin boundary**. KVCM2 resolves selections, updates the
eviction state, allocates cells and performs the copies itself, because a layout is data it
can interpret rather than code it must call.

It is two launches — a resolve and a copy — but that split is internal and chosen, not
forced: the resolve writes a miss list the copy consumes, which is what lets the PCIe
copies start while the resolve is still finishing. Fusing them would be a local
optimisation, not an interface change.

There is no second path to fuse with, so the structure is final rather than a first stage.

## Compression

Compressing history is attractive — it is the bulk of the footprint, written once and read
rarely — and with sparse history on the hot tier it needs **nothing sparse-specific**. A
codec already transforms pages crossing the hot boundary; a sparse life cycle's pages now
cross it like any other, so level 1 can hold a compressed form and the existing
`IKvCacheColdPageCodec` does the work.

The fetch is unaffected: it reads hot pages, which are never the compressed form.

`SparseHistoryPassesThroughTheCodec` pins this: a sparse life cycle's page is encoded on
demotion and decoded on promotion, with the test first asserting that the life cycle's hot
pool group really is `HOST_MEM` so that it cannot pass by accidentally describing a dense
one. Nothing in the fetch path is involved, which is the point -- it reads hot pages, which
are never the compressed form.

**The two are decoupled deliberately.** Compression is configured by
`KvCacheCompressionConfig`, not by `SparseAttentionConfig`, on the stated grounds that
compression changes *which KV is stored* while sparsity changes *which KV is read*. A
storage-bound method provides a native cold-page codec that KVCM2 retains and invokes
(`_util.py`, `cold_page_codec_provider`); an iteration-driven one runs on the
resource-manager cycle instead. Neither knows anything about this package.

Two algorithms ship: `quantization_for_cold_page`, which stores host and disk pages as
NVFP4 while leaving the active GPU cache alone, and `triattention`, which evicts tokens
periodically during decode.

So "can sparse history be compressed" reduces to "does a sparse page reach the codec", and
it does. Which pages, precisely, because two true sentences sound like one here:
sparse history is *hot* -- host-backed, but hot -- and a cold-page codec compresses pages
on their way to a **cold** tier. Sparse history is therefore compressed once demoted, at
which point the codec applies with nothing sparse-specific, and not before.

What remains is the question the codec has always posed and this design does not answer:
whether a lossy codec is acceptable for KV history, and under what accuracy budget.

## Fetch

One call per **layer** per step, on the caller's stream — not per layer group. A layer's
selection is normally computed from the previous layer's output, so a step's selections are
not all known at its start; the call has to be the unit the caller can actually fill.

Within a layer, every sparse buffer shares the call. They select the same units, so they
share one metadata row, and one resolve yields the entry list that addresses each buffer's
own cell. Their agreement is structural rather than a consequence of running identical
logic on identical input.

### Selection groups

A model that runs one indexer across a run of layers hands them all the *same* selection
every step. Resolving it per layer computes the same answer as many times as there are
layers, each against its own resident set — and they cannot diverge,
because `initialiseLease` clears every row alike and identical input then evolves
identically. So the copies are of one answer, paid for in kernel launches and in
`entriesPerSelection` of metadata per layer.

Those layers instead name one `selection_id`. Members of a selection share **one** resident
set, one in-flight reservation and one miss list. The cells stay per
layer and are *mirrored*, exactly as a layer's K and k-scale mirror today: the entry index
is shared, the bytes behind it are not. `entriesPerSelection` is the metadata's first
dimension, and `layerSelection[layer]` is where a layer reads it — the identity where every
layer selects for itself.

The id is opaque, not a `BufferId`. KVCM2 never uses *which* buffer produced a selection, so
naming one would state a claim it does not rely on, and that claim goes stale for a model
with several indexer buffers per layer or one that computes a selection without caching
anything. The validations worth having are expressible over an opaque id: members must
agree on `num_selected_blocks`, `max_cached_units` and `unit_tokens`, must share a life
cycle (the metadata they share is the life cycle's), and an explicit id may not name a layer
that set none of its own — silence is never shared.

The call then splits in two. `resolveSparseSelection()` runs **once** per selection, and
`fetchResolvedPages()` runs **per layer**, each copying the same misses into its own cells
and writing its own index list. `fetchSparsePages()` remains as the fused form and serves a
layer whose selection is its own; it rejects a shared one rather than mis-serving it, since
a second resolve would count the first's admissions as hits, issue no copies, and leave the
second layer's cells unfilled.

The copy kernel needed no change for any of this. It consumes a work list plus a buffer
range, and nothing in it depended on which layer produced the list. Neither did the
write-back: `sparseResolveKernel` is the only writer of the lease's entry words, and
`sparseFlushListKernel` only reads them — so collapsing the resolve is sufficient, and a
per-layer flush against shared metadata stays correct.

Layers that are *not* tied keep a call each, which is what starts the first layer's copies
as early as possible instead of waiting for the whole run.

The tied layers stay separate layers. The alternative — declaring their buffers as one
`AttentionLayerConfig` with distinct roles — also yields one metadata row and one call, but
it merges the copies too, so nothing can be fetched until every member's bytes are ready.
It also misdescribes the model: GLM-5.2's four indexer-sharing layers are four layers, and
only the full-indexer one declares an `INDEX_KEY` buffer, the other three being wholly
sparse. It is the **life cycle** that needs a dense buffer, not each layer. A selection
group keeps the layers apart for the copy and shares only what is actually shared.

So a call covers one layer's buffers and the copy needs no way to name a subset of them.
A caller wanting a layer's copies split further would be asking for the inverse of the
configuration choice above, having already declared those buffers as one layer.

Each call then:

1. read the selection: unit ordinals, uniform width, `-1` padded;
2. build the probe index over the required ordinals — in-flight units, then the selection;
3. stream the resident set past it: a match is a hit with its slot, the rest are misses;
4. allocate a slot per miss, taking the stalest entries first;
5. extract and copy each miss from the device-mapped host page;
6. stamp every entry the step used with the current step, flagged by whether it was already
   holding its unit;
7. write the output index list: one index per selection slot, in the selection's own
   order, `kBadPageIndex` where a slot is padded.

The output is compact rather than a table indexed by unit ordinal, because that is the
shape every sparse attention consumer in this repository takes. A token-sparse kernel is
handed the index list in place of its page table — `attentionOp.cpp` assigns
`kvPageIdxPtr = sparse_attn_indices`, and the trtllm-gen K/V TMA descriptors are rebuilt
as a flat token array so one "page" is one token — while a block-sparse kernel takes the
list beside a page table it still uses. None indexes a table by ordinal, so producing one
would only oblige the caller to compact it again.

One index per buffer, though. A layer's buffers share the resolve but not their cells, so
the list carries a buffer dimension: `[row][buffer][selection]`, which is also the single
tensor covering K and V that attention kernels tend to want.

Steps 4 and 6 are one pass: an entry's new word is written where its fate is decided, so
nothing is left to do after the work list is complete. The copies — the only step that
touches PCIe, and so the long pole — are released as soon as that pass ends.

No host involvement, so the call enqueues and returns. Every shape is static across steps.

The selection is not the only thing that needs a cell. The block being written needs one
too, and needs it before the selection takes any victim. Both are resolved by the same
probe, over one virtual list: the in-flight ordinals `[history_units, capacity_units)`
first, then the selection. An in-flight unit differs in exactly one way — a miss allocates
its cell but never joins the copy list, because its host page holds nothing to fetch until
the watermark passes it.

### Writing in-flight units

`prepare_sparse_write_indices` gives the model a destination for each in-flight unit.
It can run before the indexer produces its selections because it needs only the row's
published capacity and device history. A later fetch reuses these cells. Only misses
below the device watermark read from host pages.

The output shape is `[batch, buffers, max_in_flight_units]`. Set
`history_units = floor(device_history / unit_tokens)` and
`capacity_units = floor(published_capacity / unit_tokens)`. For each request and buffer,
slot `i` holds the arena index for unit `history_units + i` if that ordinal is below
`capacity_units`. All remaining slots hold `BAD_PAGE_INDEX`. With four-token units and
a one-token step, a new unit becomes writable once every four steps.

### One fetch row per request, one selection per query

Each decode query selects up to `num_selected_blocks` history units. `fetch_sparse_pages`
takes one row per request. Lay out the queries' selections in query order and pad unused
positions with `-1`. The row width is `sparse_queries_per_request * num_selected_blocks`.

The manager resolves the whole row together so every index stays valid while attention
reads it. If two queries select `[3, 7]` and `[7, 9]`, pass `[3, 7, 7, 9]`.
The four output positions reference three cells per buffer. Both occurrences of unit 7
return the same index, and a miss for that unit requires one copy per buffer.

The native fetch serves the decode instance in a disaggregated deployment. The prefill
instance computes the prompt and hands over its KV cache.

### Arriving with a history: the bulk watermark jump

A resumed request is the ordinary case under disaggregated serving, and it moves history from
zero to the whole prompt in one `resize()`. Nothing is written back, because nothing was ever
in flight: those units reached their host pages directly rather than through a cell.

The write-back is therefore bounded by the capacity the resize started from, not by how far
history travelled:

```
flush range = [old_history, min(new_history, old_capacity))    in units
```

Only a unit below the old capacity can have been in the cache at all. Without that clamp the
range is the whole prompt, which no reservation could cover, and the resize fails on a
write-back that had nothing to do.

That jump has to happen **before the request joins a Batch**. `Batch::resize` refuses a
history beyond the capacity its row already had, because capacity granted by a call makes
room for tokens written after it — so history can only cover tokens whose space was already
there. A resumed request sets capacity and history together from nothing, which is the one
shape that rule excludes. Seeding it on the way in costs nothing: nothing was in flight, so
there is nothing to write back and no converted table is needed yet.

### Capacity and eviction

A row fits when its distinct misses fit in the free or evictable cells. A selected unit
already resident needs no additional cell. The resolve protects every cell the row names
and every cell pinned by device history. It uses free cells and evicts the stalest
eligible entries to satisfy misses. If too few cells are available, the kernel traps
and the process cannot continue. See [Sizing](#sizing) for the guaranteed capacity.

### Preconditions the caller owns

Selections must name only pure-history units; `num_blocks` carries each request's eligible
count.

**Host-managed row**

1. `flushSparseUnits(newHistory)` — writes back what the watermark is about to pass. It
   reads the index table as of the last publish, which covers the blocks in range.
2. `resize(...)` per row, or `Batch::resize` — refused if it would take history past what
   was flushed.
3. `publish()` — converts this step's indices for the fetch to read.
4. the fetch, inside the captured graph.

**Device-managed row**

1. After verification, `advanceSparseHistory(advances)` moves each row's device watermark
   and queues write-back of the newly completed units.
2. `publish()` — for the next step. Leaves the advanced rows' histories alone.
3. the fetch, inside the captured graph.
4. `resize(...)` per row, once the host has read the count — records that step's accepted
   tokens in the member's host history and returns its unused capacity. The next step may
   already be in flight.

`publish()` uploads the staged rows asynchronously, so anything that writes what it staged
has to be separated from it by `waitPublished()` — a direct `resize()` on a member, a
`commit()`, a beam fork, an SWA recycle, anything that locks, unlocks or recycles a page.
The Batch's own mutators do that wait themselves: `Batch::resize`, `move`, `addKvCache` and
`removeKvCache` each call it before touching a row. `publish()` refuses rather than waits
when the previous one is still outstanding, because a wait there would cover only its own
staging and would quietly hide a member mutated before the call, which had already raced.

Naming an ineligible one is no longer a memory-safety problem. The table is uniform, so the
ordinal resolves to a real host page of the same request — but an unflushed block's host
page has no valid contents yet, so the result is stale data rather than unrelated memory.
A debug check against the flushed state can catch it; silence would be a wrong answer, not
corruption.

### Measuring the hit rate

A hit is a selected unit the cache already held; a miss is one the fetch copied from its
host page. The ratio is what a deployment tunes the cache size against, so it is collected
rather than inferred.

**Both figures come from the resolve, because only the resolve can produce either.** The
obvious denominator — rows times selection width, which the host knows for free — counts
things that are neither. A selection entry may be padded, and padding is negative and lives
on the device, chosen by whatever selected the units. An in-flight unit is resolved in the
same pass but is not a fetch: its host page is not written until the watermark passes it,
so there is nothing to copy and nothing to miss. Counting either as a hit inflates the rate,
and does so worst where histories are short — early decode, short prompts — which is
exactly the regime someone reads the rate to understand.

So the resolve counts the selection entries that named real history, in the pass that
already classifies them. The miss half is `claimed`, the run the row takes in the compacted
list, which it computes anyway. The two are produced together and agree by construction on
what they count; hits are their difference and are never stored.

**Each row adds its own contribution, rather than anything reading a total afterwards.**
This is the part that is easy to get wrong. The step's counters — `work.total` and anything
beside it — are cleared once per `publish()`, not once per fetch, which is what lets one
clear cover every layer without a non-kernel node between each resolve and the work before
it. A consumer downstream that read those counters and folded them into a cumulative pair
would therefore be correct only if every layer were fetched exactly once per publish. That
is what a step does, but the interface permits more, and a second fetch of a layer would
silently re-add the first one's totals. Accumulating per CTA from the values the row holds
in registers has no such precondition, costs two atomics per row, and needs no ordering
against anything.

It also keeps the whole mechanism inside one kernel: nothing in the copy, no programmatic
dependency to reason about, and no need to exclude the write-back — which shares the copy
kernel and whose moved units are due to be written rather than units the cache failed to
hold. The in-flight resolve contributes nothing on its own, since it has no selection and
its ordinals sit above the watermark.

64-bit counters: a step at B = 256 with a 2048-wide selection resolves millions of units per
layer, so a 32-bit pair would wrap within a few hundred steps — and a wrapped miss count
makes the hit rate *improve*, which is the failure a reader would trust.

**There is no enable switch.** The cost is two atomics per row per fetched layer. A switch
would cost nothing to run either, but it would bake into a captured graph, so turning
statistics on mid-run would keep reporting zeros until the graph was captured again — a
disabled feature indistinguishable from a broken one.

`peekHitMissCounter()` reads the pair on a temporary stream and synchronises only that, so
it never stalls the step. The pair may be a few launches stale, which does not move a ratio;
a caller wanting the two to agree exactly synchronises the device first. Because the read
races a running fetch and the counters are separate atomics, a copy can land between them —
the result is clamped rather than checked, since a diagnostic must not be able to fail the
cache. Per-row detail is already available separately in `missCount`.

### Measuring how full the cache gets

`peek_sparse_cache_peak_use(layer_id)` returns the peak number of cells required by one
fetch row in the layer's life cycle and the number of cells available per row. The peak
covers fetches since batch construction or the last `reset_sparse_cache_peak_use()`.

The resolve kernel computes `entries - evictable + distinct_misses` for each row and
updates the peak with an atomic maximum. The first term (`entries - evictable`) counts
cells already occupied by selected or pinned units. Each distinct miss requires one
additional cell. Compare the observed peak with capacity when sizing for workloads
whose queries select overlapping history.

## One arena, one index space

The cache arena holds nothing but cells — selected history and the unit being written alike.
That is what lets the fetch emit a **single** index list covering both, and therefore why
existing attention kernels need no change: a kernel still indexes one list against one base.

Indices are unit-granular, and over the buffer's **arena region** — one per hot pool, see
*Sparse buffers across more than one pool*:

```
index = cell * expansion + unit
      = (lease * cells_per_lease + slot_in_pool * cells_per_buffer + entry / expansion)
        * expansion + entry % expansion
```

where `slot_in_pool` is the buffer's index among those of the life cycle sharing its pool,
and `cells_per_lease` is that pool's buffers times `cells_per_buffer`.

A sparse buffer therefore has **two** base addresses, answering different questions:

| accessor | names | consumer |
| --- | --- | --- |
| `get_mem_pool_base_address` | the buffer's pages, hot tier | transceiver, KV transfer, anything addressing blocks |
| `get_sparse_cache_base_address` | the cache arena region of the buffer's pool | the attention kernel, against the fetch's index list |

`get_mem_pool_base_address` keeps meaning "this buffer's pages" for every buffer, sparse or
dense, which is what lets the transceiver work unchanged on hot, coalesced pages. The arena
is a second allocation holding the only GPU-resident copy a kernel can read, so it gets a
second accessor rather than overloading the first. The consumer's side is otherwise
unchanged: one base, one index list, the expanded index space `PageIndexConverter` already
produces.

`tokens_per_block_override` plays no part in this. It is the index converter's own
mechanism: it scales `BufferConfig.size` and stretches the buffer's public page index space
by `expansion`, while allocation, locking and migration stay on the global
`tokens_per_block`. It is rejected on a sparse buffer, which says what it means with
`sparse_selection_unit_tokens` and leaves its index space alone. `unitsPerPage` and
`expansion` are distinct quantities and the sparse code never uses the latter.

## Batch

Every device array here is indexed by batch row, and every call must agree on which row a
request occupies. `Batch` owns that: membership, row order, the page table, `num_blocks`,
and the staging mirror. It is global, not per layer group, and carries one page table per
layer group.

It also owns three things the sparse path depends on:

- **the stream.** Taken at construction; joining binds the member to it, and moving a member
  off it is refused. A member's own work and the Batch's transfers have to be ordered
  against each other, and a stream per call left two sources of truth for which one that is.
  It is also what makes the write-back's ordering free rather than checked, since a member
  records its events on the stream the flush is launched on.
- **the converted index tables** (`INDEX_CONVERSION_DESIGN.md`), which both the fetch and
  the write-back read.
- **the write-back** through `flushSparseUnits()` or `advanceSparseHistory()`. See
  [Who flushes](#who-flushes).

**The upload is whole-table by decision.** `publish()` stages every row and sends the table
in one `cuMemcpyHtoDAsync` on the upload stream, which overlaps compute. Sending only the
rows whose locks moved would replace that single driver call with one per row, and the cost
of those launches outweighs the bytes they save. The table is large -- four layer groups,
about 32 MiB a step at a 1M `max_seq_len` and B = 256 -- but size is not the figure that
decides it.

**Dirty rows — planned, not built.** What dirty tracking would still save is the host-side
restaging, not the transfer: a `_KVCache` knows its `Batch` and row (it does, for detach),
any lock-changing operation could mark that row dirty, and `Batch` could restage only those,
following the existing `mark_stats_dirty` / `clear_stats_dirty` /
`get_dirty_stats_kv_cache_ids` pattern. Until then a step restages the whole table, which is
a cost rather than a wrong answer.

Direct mutation of a seated member is permitted rather than refused. Refusing it would need
a batched twin for every mutator — `commit` changes locks and runs every decode step — and
the failure it would prevent is a restaged table, which costs a step rather than giving a
wrong answer. A debug-mode check can still flag direct mutation while batched.

**Batched resize** resizes every member and publishes once at the end. The contract is row
order and publish-on-completion, which holds however the resize itself is implemented. The
result is per-request rather than a single flag, since `_KVCache.resize()` rolls back per
cache and a table published after a partial failure must reflect the post-rollback state.

**Rows are slots.** Removing a request leaves a hole to be reused rather than shifting later
rows, which would invalidate the whole table and destabilise the shapes graph capture needs.
The `_torch` layer already works this way with `index_mapper_capacity` and a `copy_idx`
gather. Membership is exclusive: a row already occupied is refused rather than rebound,
because nothing downstream would notice the substitution.

**A row owns its member.** `mRows` holds a `shared_ptr`, so a seated cache cannot be
destroyed under the Batch and a caller may keep only the Batch; `removeKvCache()` hands the
reference back, which makes "the Batch kept it alive" and "here it is back" one operation.
The binding pinned every cache ever added for the Batch's whole life and released none of
them, which was the wrong shape as well as a leak: what a row needs is a reference for
exactly as long as it is seated. `KvCache` points back with a raw `mBatch`, so there is no
cycle, and `_detach()` -- the path for a member destroyed while seated -- is now unreachable
by construction and says so with a fatal check rather than carrying teardown that could
never run.

## Sizing

`N = max_cached_units` is the per-request cache capacity for a sparse selection. Let `k` be
`num_selected_blocks` per query, `S` be `sparse_queries_per_request`, and
`H = ceil(sparse_max_in_flight_tokens / unit_tokens)` be the maximum pinned units.
Both manager settings, `sparse_queries_per_request` and `sparse_max_in_flight_tokens`,
default to one.

- **Configuration minimum** — `k + H` cells guarantee room for one full query and every
  pinned write.
- **Worst-case multi-query row** — `S * k + H` cells guarantee room even when all `S`
  queries select disjoint history. A smaller cache can serve a row when its distinct
  misses fit the free or evictable cells.
- **Retention** — extra cells can retain history for later steps. For single-query decode,
  `2k`–`4k` is the starting range reported by HiSparse, subject to the configuration minimum.

The Python managers size `N` from `SparseKvCacheOffloadConfig.gpu_cache_factor`, in
selections of one query. Left at its default, the factor is multiplied by `S`, so a step
always has room for the worst-case row and each query keeps the retention a single query
has; set explicitly, it must be at least `S`. The result is capped at the largest whole-page
cache the probe can name (16,256 token units at 128 per page), but never below `S * k + H`.
A step therefore never meets a row it cannot resolve, which would stop the process.

The batch reserves `max_batch_size` leases of `N` units per buffer. For measured demand,
see [Measuring how full the cache gets](#measuring-how-full-the-cache-gets).

## Python surface

Declared in `tensorrt_llm/runtime/kv_cache_manager_v2/__init__.pyi`:

- `BufferConfig.is_sparse: bool` and `BufferConfig.sparse_unit_chunks: int` —
  construction-time, per buffer.
- `AttentionLayerConfig.sparse_cache: SparseCacheConfig | None` — the layer's fetch path:
  `num_selected_blocks` (the selection width `k`), `max_cached_units` (the reservation
  bound), `unit_tokens` and `selection_id`; construction-time, per layer.
- `KVCacheManagerConfig.sparse_max_in_flight_tokens: int` — maximum gap in tokens between
  reserved capacity and the request's host history length.
- `KVCacheManagerConfig.sparse_queries_per_request: int` — query tokens a request may feed
  one step; a fetch's row is that many selections wide.
- `KVCacheManager.is_sparse(layer_id, data_role) -> bool`.
- `Batch` — membership, row order, device arrays, `resize()` and `publish()`, which
  transfers the whole table; dirty-row tracking is planned, not declared.
- `Batch.fetch_sparse_pages(buffers, selected, out, stream)`.
- `Batch.sparse_out_slots(buffers) -> list[int]` — the slice of `out` each buffer's
  indices land in, for both the fetch and the write.
- `Batch.flush_sparse_units(new_history_lengths)` and
  `Batch.advance_sparse_history(advances)` — the write-back, with the history given by the
  host or advanced on the device; `Batch.history_on_device(row)` tells which a row uses.

Publishing the page table is `Batch`'s job. It must not reuse the dense path's block
offsets: those are scaled and fold `BAD_PAGE_INDEX` to slot zero, whereas a sparse table
numbers history and input-token entries in different levels' pools and needs the sentinel
to distinguish an absent block from slot zero.

Device arrays are typed by a `DeviceArray` protocol over DLPack, so callers pass torch
tensors without this package depending on torch.

## Lock levels

Slot release is level-aware: `Page::~Page()` releases with the page's own `cacheLevel`, and
the `kHotLevel` in `ScratchSlotLock::unlock()` holds because SWA prefill scratch is GPU-only
by definition.

`_lockHeldBlocks` carries no level hazard. The OOM rollback for `resize` re-locks at the hot
level, which is where every buffer locks, so a page table's entries and the level its pages
lock at cannot disagree. No per-target lock level exists: there is no `queryLockLevel`, no
`BatchedLockTarget::cacheLevel`, no `StaleBackup::cacheLevel` and no `residencyLevel`.

No attention kernel addresses a host page. A sparse layer's kernel reads its units from the
arena, which is GPU memory bounded by the lease count and the cells a lease holds, and a
non-sparse layer reads the device pool through its block table as it always has. The host
pool is reached only by this manager's own copies, which compute their offsets in `size_t`.

## Planned: a selection unit size per layer

A layer declares its fetch path with `sparse_cache`, and layers sharing a life cycle must
currently declare the same one. The arena is per life cycle and its metadata is a single
stride — a selection's rows live at `entry + slot * entriesPerSelection` — so one
entry count serves every layer, and `cellsPerBuffer` follows from `max_cached_units` and the
unit size together.

That is an implementation limit, not a property of the design. Lifting it:

- **Size a cell from the largest unit of the life cycle**, or simply from a whole page.
- **A layer with a smaller unit subdivides that cell.** This is the mechanism a cell already
  uses: it holds `unitsPerPage` units, addressed as `cell * unitsPerPage + unit`. A layer
  whose unit is half the cell's just has twice the divisor.
- **Require unit sizes to be powers of two**, so every such division is exact and a buffer's
  bytes split without remainder at any level.
- The entry words are **already per selection**, so they need no change; only the entry
  count and the divisor become per layer rather than per life cycle.

Until then a mismatch is rejected at construction. It was previously taken as a `max` across
the life cycle, which quietly served one layer's shape to another.

## The output list merges across mirrored pools

The arena mirrors the hot pool group, and a lease takes the same cell index in every pool
of it. So **mirrored pools** -- equal `buffersInPool`, hence equal `cellsPerLease`, with
`cellsPerBuffer` one value for the whole shape -- resolve a selection to the same index.
`out` carries one slice per distinct index table entry rather than one per buffer, with
the caller applying each pool's own base.

For a life cycle whose pool group holds a dense indexer and two sparse pools -- `{K,V}`
and `{kscale,vscale}` -- `sparseIndexTableSpecs()` emits one spec per `(pool, layerRow)`
and `Batch::_buildIndexTables` folds the two into **one** table, because their `scale` and
`concatOffsets` agree. The layer's four buffers therefore take the slices `[0, 1, 0, 1]`:
`K` with its scale, `V` with its scale. A scale shared by `K` and `V` together is a pool
of one buffer, whose `cellsPerLease` differs, so it keeps a slice of its own: `[0, 1, 2]`.

`Batch::sparseOutSlots` is the query, and `fetchSparsePages`,
`fetchResolvedPages` and `prepareSparseWriteIndices` all follow it. They have to agree, or a caller would need one
rule for reading a fetch and another for a write. A run whose slices an earlier one
already wrote is not launched at all -- the saved work and the saved writes are the point
-- and the entries of `out` no buffer maps to are left untouched rather than packed out,
so `out`'s shape stays `len(buffers)` however the buffers happen to merge.

The grouping follows the caller's `IndexTableSpec`, so two buffers merge only where one
table was asked for over both. The merge is not read off the dense tables alone, though:
those merge on the hot pool's `scale` and `concatOffsets`, which is not the same quantity
as the arena's geometry -- a pool carrying only some of the group's layers has the same
scale and a different `cellsPerLease`. `sparseOutSlots` checks the arena geometry agrees
before sharing a slice, and raises rather than aliasing two different cells onto one.

Rests on a property only a second lease can observe:
`TestSparseFetch::test_mirrored_pools_resolve_a_selection_to_one_index` uses one, because
at lease zero the `leaseIndex * cellsPerLease` term vanishes and pools of unequal width
are indistinguishable.

The rest of the arrangement is what makes one base per slice enough:

- **A base per pool, in both spaces.** `getSparseCacheBaseAddress` returns
  `arena.base(poolIndex)` -- "per pool, not per buffer" -- so `{K,V}` share one arena base
  and `{kscale,vscale}` another, exactly as the two pools share one hot-pool base each.
- **The within-pool offset rides in the index**, as `slotInPool * cellsPerBuffer`. That is
  the concat-offset arrangement, so one base does serve a pool's buffers.
- **The output order is the plan's**, because `_sparseLayerSpan` rejects any other:
  "buffers must be exactly this layer's sparse buffers, in the cache's order".

## Sparse buffers across more than one pool

A life cycle's sparse buffers need not agree on size. The **quantised shape** — block scales
as a separate, smaller `DataRole` — works, which matters because it is a real configuration
rather than an exotic one.

Pools partition a life cycle's buffers by size, and the copy addresses one pool with one set
of strides: `bufferBytes`, the hot slot stride and the unit layout are all per pool. The
design follows from making the arena **mirror the hot pool group** rather than inventing a
layout of its own:

- **One region per pool**, sized `maxLeases * buffersInPool * cellsPerBuffer * cellBytes`.
  A lease is an index rather than a cell offset, because a pool's cells-per-lease is its own.
  Each nonempty region also reserves a zeroed tail of 128 selection units before its
  alignment padding. For token-sparse MLA, one unit is one stored KV row: this covers
  FMHA's one-tile over-read past the last indexed row. The tail is per pool, not per
  buffer or lease, and is excluded from usable cells, entry counts and index arithmetic.
  Its bytes are `128 * cellBytes / unitsPerCell`, independent of the page's unit count.
  Shared buffer/layer mappings and lease metadata follow all the padded pool regions.
  A pool index no sparse buffer of the life cycle uses keeps its place in the list and sizes
  to nothing: the arena is indexed by the storage pool index rather than by a dense numbering
  of its own, and an empty region costs only the entry that says where the next one starts.
- **Buffers ordered `(layer, pool)`.** A layer's buffers stay consecutive, which the fetch
  needs to name them as a range; and within a layer a pool's buffers are consecutive too,
  which the copy needs for the same reason. `bufferLayer` stays non-decreasing.
- **One launch per maximal run of consecutive buffers sharing a pool.** When the buffers are
  alike that is a single run spanning the whole life cycle, so the uniform case pays nothing
  for the general one; the quantised shape yields one run per layer and pool. The fetch
  clips each run to the layer it resolved; the write-back takes them whole, because its work
  list is per layer and a run already spans as many layers as share its pool.
- **`get_sparse_cache_base_address` is role-dependent**, naming the region of the buffer's
  pool. It already took a role.

Every per-pool quantity is a launch scalar — `deviceBase`, `bufferBytes`, `cellsPerLease`,
`layout` — so the kernels keep their shape. What they gained is `bufferSlotInPool`, a
per-buffer map alongside `bufferLayer`, replacing the assumption that a buffer's position
among *all* the life cycle's buffers is also its position in its pool. A per-buffer offset
within the cold page sat here too, until the converted index absorbed it; see
`INDEX_CONVERSION_DESIGN.md`.

Buffers sharing a pool must still share a selection unit layout. A pool fixes a cell's size
but not how its units are strided within it, and the copy walks a run with one set of
strides; the check is per run rather than per life cycle.

**The runs share one launch.** Nothing orders one pool's copy against another's — their
cells are disjoint and the converted page table they share is read-only — while each is
small and latency-bound, so issuing them back to back on one stream would expose each one's
latency in turn. So one grid runs them all, and what differs between them lives in a
**device-resident descriptor**.

`gridDim.y` walks the buffers of every pool as **one flat range**, with a constant map from
buffer to pool. The alternative — pools on one grid axis, buffers-per-pool on another —
has to size the buffer axis for the widest pool, so every buffer a narrower pool lacks
costs an idle block. Pools of one life cycle are often alike, and then the two are the same
shape; they stop being alike as soon as a pool holds more buffers per layer than another,
which a key-and-value pool beside a scale pool already does.

The descriptors are built once, with the Batch. Every field comes from the buffer plan and
the index tables, both fixed at construction, so nothing is written per step and a captured
graph replays a copy with no host involvement at all. The fetch needs one row per layer — a
layer has its own index-table entry and its own slice of each pool's buffers — while the
write-back spans every layer at once and needs only the one.

The vector width is part of the descriptor, chosen from the layout's **sizes alone**. The
two base pointers are checked against it where the descriptor is built, because a pool whose
base is misaligned is a configuration error worth naming rather than a case to carry into
the kernel. A block switches on the width once and calls a device function templated on it;
`blockIdx.z` is block-uniform, so the switch costs no divergence.

One grid rather than several is also what lets **every** pool carry the launch dependency
below. A per-pool fan-out across streams could not: a programmatic dependency follows a
stream, not an event.

### Launching before the producer finishes

All three kernels are enrolled in programmatic dependent launch when the environment enables
it, and in each the order of the prologue is the point: a block does everything that is not
its predecessor's output *above* `cudaGridDependencySynchronize()`, so that by the time the
predecessor drains the block is resident, addressed, and as far along as it can be.

How far along differs by kernel. The resolve gets least out of it: its probe index is keyed
by the required ordinals, and the selection is exactly what the producer produces, so only
shared-memory initialisation sits above the wait. That was not always so -- an index built
over the resident set read only the layer's own residency table, which nothing before a
resolve touches, so the whole build could run while the predecessor drained. Keying the
table by `k` rather than by the cache's capacity was worth giving that up; see
"Probe index". The copy resolves its pool, its
buffer and its work list from descriptors and block coordinates; the flush-list scan
addresses its row from the staged view. Each then waits for the one thing that is genuinely
produced upstream.

**The work-list counters are cleared once a step, in `publish()`**, not once per producer.
That is not a micro-optimisation: a clear is a memset, and a memset is not a kernel, so one
standing between every resolve and whatever ran before it would sever the launch dependency
at each of a model's layers -- for DeepSeek-V4, sixty-one times a step.

What makes one clear enough is that the counters are per layer and per direction, so no two
producers in a step share one. The fetch keeps a *single* list, reused from layer to layer,
because a layer's copy has consumed it before the next layer's resolve overwrites it; only
the counter has to read zero when a resolve starts. Per-layer lists would have cost the
layer count times the list, which is megabytes, against a counter's four bytes.

A step that somehow skipped `publish()` would have its second resolve add on top of the
first's count, and the claim's bound check would fail rather than quietly overrun the list.

Global **writes** stay below the wait in all three, even where the datum looks private. A
block that has not yet waited may still be running beside a grid that could read what it
would overwrite, and the cost of keeping the rule is a few stores.

The producers release as early as they correctly can. The resolve's work list is final at
the end of its eviction loop, which is also where the entry words are written, so it calls
`cudaTriggerProgrammaticLaunchCompletion()` there and only its miss count follows. The
flush-list scan is the same shape — the list is the last thing it writes — so in both cases
the release brings the copy's launch forward rather than overlapping a tail.

The copy releases immediately after its own wait. That looks wrong and is not: the kernel
that follows it on the stream builds the attention index table from the resolve's slots and
never reads a cell, so it must not be ordered behind a copy it does not depend on.

## Where sparse data lives

A sparse buffer's pages are ordinary hot pages that happen to be host memory, device-mapped
so the fetch reads them without a host round trip. Cells come from a per-life-cycle arena,
leased per request; the fetch reads hot pages at `BufferAttr::offset`.

A sparse deployment places no requirement on its cold tiers. History is hot, so level 1 may
be host, disk, or absent.

Memory kind is a property of the **pool**, not of the level and not of the life cycle. The
GPU tier allocates a sparse pool from host memory and a dense one from device memory, in the
same group; `AttnLifeCycle` carries no `isSparse`.

A layer may therefore hold both a sparse buffer and a dense one, which is what a model
wants: an indexer and the data it selects over are one attention computation and belong in
one layer. Sparsity constrains pool formation and nothing above it, so two layers differing
only in sparsity share a life cycle — one layer group, one page table in `Batch`.

**The codec is the hot↔cold transform**, not the level 0 → level 1 one.
`StorageManager::migrate` computes `srcIsHot = srcLevel == kHotLevel` and performs a plain
byte copy when source and destination are on the same side, invoking the codec only across
that boundary. Sparse history is hot, so it crosses that boundary on its way to a cold tier
like any other page: compression needs nothing sparse-specific.

The arena is sized from `constraints`, which already declares the batch shape `Batch` derives
its rows from, so the cells are a static reservation rather than an on-demand draw on the
shared pool.

### Sizing a host-backed pool

Sparsity is a property of a **pool**, not of a life cycle or of a pool group. A pool is
uniformly sparse or uniformly dense, because buffer grouping keys on `(size, isSparse)`;
the GPU tier allocates the sparse ones from host memory, device-mapped, and reports
`GPU_MEM` for the group either way.

That it reports GPU is sound because nothing reads the hot side's tier. The codec path keeps
only `coldTier = srcIsHot ? dstTier : srcTier`, and the copy-engine path needs both ends
hot, which never happens; host memory is device-accessible, so a mixed group needs no second
tier.

**A sparse pool's bytes are excluded from the device quota.** A slot is one row spanning
every pool of its group, so once a group holds at least one device-backed pool, the quota
sizes that pool and the sparse pools inherit the same row count. The quota therefore charges
only the dense pools.

This is why a pool group must hold **at least one dense pool**: a wholly sparse group would
have no device bytes per slot and so no slot count to derive. That is a validation, not a
heuristic.

**But the row count is not free.** Inheriting it gave the sparse pools no bound of their own:
their host memory grew with whatever the device quota bought, and the only way to hold it
down was to shrink the device cache that everything else shares.
`KVCacheManagerConfig::hostSparsePoolQuota` is that bound, and the allocator takes it as a
second budget over a disjoint set of the same pools rather than as a clamp applied afterwards.

The two budgets **partition** a group's pools — dense charged to the device quota, sparse to
the host budget, each in its own allocation grain — so one slot count settles both at once.
`ratioToSlotCountList` therefore takes every pool's size once, plus an index list naming the
pools each budget answers for. Three properties follow, and the first is the reason this is
one pass rather than two:

- **A group bound by one budget releases what it does not use of the other.** Clamping the
  slot count afterwards could only lower it; the device bytes a host-bound group then stopped
  using went to nobody, and a group with no sparse pool at all -- which can never be
  host-bound -- was left short for no reason.
- **Both budgets and `min_slots` are kept exactly; the pool ratio is what gives way.** The
  floors come off the top before anything is shared, so no division of the rest can break one.
- **Their only conflict is a configuration error, and it is caught once.** The floors can
  exceed a budget, but that names neither the ratio nor the device quota -- the two things a
  rebalance changes. So it is rejected the first time the allocator runs and cannot begin to
  fail later.

The split belongs to the hot level alone and is not a property a level can report about
itself: a cold group is a single pool holding dense and sparse buffers together, and a GPU
tier may itself be a cold level. The pools a quota pays for are therefore derived as the
complement of the ones the host budget names -- never passed alongside them -- so the two
cannot overlap, and no pool can be left for neither to pay.

#### The quota excludes sparse pools; the ratio counts them

These are deliberately different currencies, and the asymmetry looks like a bug until the
third party is in view.

- **The quota** (`roundedNumBytes(cacheTier())`) counts only the pools on the level's own
  medium. Host-backed bytes consume none of the device memory it measures.
- **The ratio** (`ratioList()`, and the profiler's) weighs *every* pool, host-backed
  included. It has to: it is projected onto cold levels, where a single pool holds dense and
  sparse buffers together, and it is compared against the achieved layout, which is measured
  the same way.

The consequence is easy to miss. `KvCacheManager::_needAdjustment` compares the layout's
current ratio against the target `constrainPoolGroupRatio` produces, and rebalances when they
differ. `constrainPoolGroupRatio` derives that target by *running the allocator* and
normalising the bytes it would produce -- so it must run it with the **same budgets**, the
sparse host quota included. Omit the budget there and the target describes a layout the
allocator will not produce: every comparison mismatches, and a rebalance re-triggers on every
check even though the profiler's answer never moved. The ratio, the measurement and the
constraint all have to speak all-bytes for that loop to settle.

What the asymmetry *does* require is that the division be driven by all-bytes demand, not
that the ratio be re-expressed per budget. Dividing the device quota alone by an all-bytes
ratio does not realise that ratio: a sparse-heavy group takes a device share its device cost
per slot does not warrant, and converts it into more slots than asked for.

The release step does not rescue this. It returns the device grains a group cannot use
*because the host budget held it* -- so where the host budget is generous, nothing is held,
nothing is released, and the over-share stands. The fix belongs in how the budgets are
divided: demand is computed from each group's all-bytes cost per slot, so the achieved
all-bytes shares track the ratio, and the per-medium budgets act as caps on that demand.
Re-expressing the ratio itself in a single medium's currency would be wrong for the reasons
above -- it has to stay comparable with what `ratioList()` measures and with what a cold
level inherits.

#### How the division runs

The allocator is a search over one scalar, and all-bytes is what that scalar buys.

Write `A(pg, n)` for what `n` slots of group `pg` cost in all-bytes — every pool rounded up
to the grain of the medium backing it, which is exactly `CacheLevelStorage::allMediaBytes()`. For
a *level* of all-bytes spending `L`, each group's demand is

    demand(pg, L) = max( minSlots[pg], maxSlotsForAllBytes(L · r[pg]) )

i.e. the most slots whose all-bytes cost fits that group's share of `L`, never below its
floor. `demand` is monotonic in `L`, so each budget admits a largest feasible level, found by
bisection: `levelQuota` for the device quota, `levelSecond` for the host budget. Both must
hold, so the search is run against both and the binding one wins.

Three details make the result exact rather than approximately right.

- **Floors come off the top.** Each group starts at `minSlots`, snapped up by `freeRun` to
  the largest count costing no more grains in any medium — a count short of what it has
  already paid for is not the count a re-derivation from the same bytes returns, and that
  disagreement is the grains↔slots round-trip bug in its original form. Host-backed pools
  count here even when no host quota is set: a pool nothing is bounding is still real memory,
  and letting the snap ignore it grows the floor to whatever the device grain absorbs.
- **A group frozen by one budget releases the other.** When the host budget pins a group, its
  slot count stops rising, and the device grains its demand would have claimed are returned to
  the search rather than reserved. This is what makes one pass strictly better than a quota
  followed by a clamp.
- **A bounded greedy finish spends the remainder.** Bisection lands on a level, not on an
  integral allocation; leftover grains in either budget are then handed out one group at a
  time, each step choosing the group whose resulting `A(pg, n)/r[pg]` is lowest. That is a
  leximin step on achieved all-bytes shares: equal shares mean the achieved ratio *is* `r`,
  so raising the currently worst-served group is the move that gets closest to it.

The fixed-point property falls out of measuring demand and achievement in the same currency:
feed the allocator the normalised all-bytes ratio of its own output and it returns the same
slot counts, which is what stops `_needAdjustment` from re-firing on an unchanged profile.
`test_feeding_back_the_layouts_own_ratio_changes_nothing` pins it, including the case where a
host-backed pool is most of a group's bytes and none of its quota.

A level's quota measures one medium, which is why host-backed bytes cannot be drawn from it:
they consume none of the device bytes it counts, so charging them would cap sparse history at
a GPU-sized number — the opposite of the point — while leaving those device bytes
unallocated. No host tier's quota holds them either: `cacheTiers` is a positional list, so
"the host tier" may be several or none, and a hot pool's size would then change when someone
added a cold tier that has nothing to do with it.

**Coverage of the data path.** `TestSparseDataPath` asserts the bytes themselves on both
sides of the copy: a pattern written to a hot page must appear in the cell the fetch names,
and a pattern written to a cell must appear in its hot page once the watermark passes it.

It is the only coverage that constrains *which* allocation the copy reads. Offsetting the
page base by one slot leaves 57 other sparse tests passing, because residency, index shape
and counts are all insensitive to that address — so nothing else in the suite would catch a
pool addressed at the wrong base.

**No cell is ever read from, or written to, an encoded page.** Sparse pages sit at the hot
level, so they are eligible for demotion, and demotion is what invokes the codec. A page
cannot be encoded while the cache still holds a cell copied from it:

- An `ACTIVE` request locks the pages it requires (`AGENTS.md`, *Per-request cache*), and
  `LOCKED` forbids eviction outright (*Page status*).
- Eviction is the only route to migration, and migration across the hot boundary is the only
  thing that invokes the codec.
- The lease is held inside the locks' lifetime, not alongside it: seating requires a resumed
  cache, which has already locked the pages it requires, and leaving returns the lease. A
  cell therefore exists inside its page's lock, with room to spare at both ends, in every
  configuration but the sliding-window one below.

`commit()` does downgrade a lock to a holder, but only for a page it is discarding in favour
of an existing tree page -- not one the request goes on to read.

The one configuration where a live request holds unlocked pages is a **sparse layer with a
sliding window**, where `_unlockStaleBlocks` releases out-of-window blocks. It is safe, and
not by caller discipline: the indexer that produces the selection applies the same window, so
a block the window has made stale is never named. Nothing reads an unlocked page, so nothing
can read one that has been encoded or recycled.

A cell fetched before the window moved does outlive its page's lock there. It is harmless:
nothing names it again, so it ages out within a few steps. The write-back is
unaffected in either case — the units it crosses lie in `[history, capacity)`, above the
watermark `_unlockStaleBlocks` works from, so their pages are locked for as long as the
request is active.

It is the *selector*, not this package, that makes that true.
`fetch_sparse_pages` cannot detect an ineligible ordinal — the page table is uniform, so a
stale ordinal still resolves to a real page of the same request. The combination does not
arise today in any case: a sliding window already bounds a step's reads to a narrow fixed
span, which is the span a selection would otherwise have to discover.

## Standing limits and invariants

Properties the rest of the design is entitled to assume.

- **Decode only.** A separate prefill instance computes and transfers the prompt KV.
- **History is monotonic for a request.** `resize()` rejects a decrease in host history.
  `advanceSparseHistory()` accepts only nonnegative advances within published capacity,
  and `publish()` preserves device-managed rows' watermarks.
- **A selection unit never exceeds `tokens_per_block`.** Sub-page units are supported
  through `expansion`; a unit spanning several pages is rejected, which is what keeps slot
  contiguity and k-at-a-time eviction out of the design.
- **`N` is a uniform upper bound**, not per-request. Ragged device arrays would cost the
  static shapes graph capture needs.
- **`num_blocks` lives on the device, in `Batch`.** Deriving it per call from the watermark
  would produce a host value the kernel cannot use, costing an H2D per layer.
- **Eviction never copies.** An entry is evictable only once flushed, so evicting is a
  free-list push. Flushes are triggered by the watermark alone, never by eviction pressure.
- **Pinning covers unflushed units only.** They are private because unflushed blocks are
  not in the radix tree, and bounded by `max_input_length`. A unit stays pinned until
  device history passes it, including while capacity does not cover it. Staleness orders
  the eligible entries.
- **No stale-key invalidation.** Per-request cache scope makes it impossible.
- **`tokens_per_block_override` is not the sparse granularity lever**, and is rejected on a
  sparse buffer. `sparse_selection_unit_tokens` is — see *Extraction interface*.

## References

- HiSparse: [arXiv:2608.07009](https://arxiv.org/abs/2608.07009),
  [SGLang docs](https://docs.sglang.io/docs/advanced_features/hisparse_guide),
  [LMSYS blog](https://www.lmsys.org/blog/2026-04-10-sglang-hisparse/)
- MiniMax Sparse Attention: [arXiv:2606.13392](https://arxiv.org/abs/2606.13392)
- DeepSeek-V4: [arXiv:2606.19348](https://arxiv.org/pdf/2606.19348)
