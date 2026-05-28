# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Reproducer for MAX_UTILIZATION scheduler vs KVCacheManager admission-gap divergence.

Background
----------
Under MAX_UTILIZATION + high concurrency + short ISL/OSL (e.g. bielik_11b_v2.2 bench
at maxbs=512, ISL/OSL=128/128, L40S), the C++ KVCacheManager crashes in
WindowBlockManager::allocateBlock with "No free blocks left" at
cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp:2365.

Hypotheses under test
---------------------
B. Admission gap: the scheduler's snapshot of free blocks (set in start_scheduling,
   mutated by scheduling_remove_sequence on pause) does not see admission-time
   reservations made by the same pass. The cumulative `num_scheduled_blocks`
   accumulator in `MaxUtilizationScheduledBlocksManager` (scheduler.py:1234-1298)
   should plug this gap — but only if `get_needed_blocks_one_step` returns the
   same number of blocks that `addSequenceBatch` + `addToken` actually consume.

C. No per-step revalidation: between scheduler decision and `add_token`, no
   defensive check exists in resource_manager.py:1025-1054. Any off-by-one in the
   predictor is unrecoverable — `TLLM_CHECK_WITH_INFO` at kvCacheManager.cpp:2365
   aborts the process.

Approach
--------
Drive the **real C++ KVCacheManager** (via the nanobind binding) and the **real
PyCapacityScheduler** with MAX_UTILIZATION through scenarios that match the failing
benchmark's per-step admission + decode pattern. No model, no forward pass — just
the KV pool, the scheduler, and `LlmRequest` instances. Runs in seconds on any
single GPU.

The mock variant of this test was deliberately rejected: faithfully mirroring
`schedulingReleaseBlocks` / `mSchedulingNumFreeBlocks` semantics in Python risks
papering over the very C++/scheduler interaction we are trying to stress. Using the
real binding means any divergence between `get_needed_blocks_one_step` and the
actual `addSequenceBatch` / `addToken` allocation surfaces directly.

Outcomes
--------
- If a scenario asserts "No free blocks left" (C++ TLLM_CHECK abort surfacing as a
  Python exception, or the test's invariant check failing): hypotheses B and/or C
  are reproduced — the scheduler over-admitted relative to what the C++ allocator
  can actually serve. Whichever scenario fails localises the trigger.
- If all scenarios pass: the C++ KVCacheManager + scheduler pair is internally
  consistent at this geometry, redirecting the search to higher-order effects in
  the bench path: chunked-prefill non-first-chunk allocation (line 3426 returns 0),
  VSWA cross-pool contention, draft tokens, or the actual benchmark's request mix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import pytest

# This test requires a CUDA device because KVCacheManagerCpp.allocate_pools(False)
# allocates real GPU memory for the block pool.
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("requires CUDA device for KVCacheManager pool allocation",
                allow_module_level=True)

import tensorrt_llm.bindings  # noqa: E402
from tensorrt_llm._torch.pyexecutor.llm_request import (  # noqa: E402
    LlmRequest, LlmRequestState, SamplingConfig)
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import (  # noqa: E402
    PyCapacityScheduler)
from tensorrt_llm.llmapi.llm_args import CapacitySchedulerPolicy  # noqa: E402

KVCacheManagerCpp = tensorrt_llm.bindings.internal.batch_manager.KVCacheManager
CacheTypeCpp = tensorrt_llm.bindings.internal.batch_manager.CacheType
DataType = tensorrt_llm.bindings.DataType


# ---------------------------------------------------------------------------
# Request factory
# ---------------------------------------------------------------------------

def _make_request(request_id: int,
                  prompt_len: int = 128,
                  max_new_tokens: int = 128,
                  beam_width: int = 1,
                  state: LlmRequestState = LlmRequestState.CONTEXT_INIT) -> LlmRequest:
    req = LlmRequest(
        request_id=request_id,
        max_new_tokens=max_new_tokens,
        input_tokens=list(range(prompt_len)),
        sampling_config=SamplingConfig(beam_width),
        is_streaming=False,
        draft_tokens=None,
    )
    req.state = state
    return req


# ---------------------------------------------------------------------------
# Thin adapter around the real C++ KVCacheManager
# ---------------------------------------------------------------------------

class _SchedulerKVAdapter:
    """Thin shim around the C++ KVCacheManager binding for the scheduler interface.

    Forwards every behaviorally-relevant call to the real binding. Adds the
    Python-side attributes the scheduler reads but the binding does not expose
    (`max_attention_window_vec`, `cross_kv_cache_manager`).

    Crucially, this is NOT a mock — it does no accounting of its own. All
    block tracking, free-block snapshots, pause/restore semantics, and
    boundary-crossing allocations happen inside the real C++ KVCacheManager.
    """

    def __init__(self, impl, max_attention_window_vec: list[int]):
        self.impl = impl
        self.max_attention_window_vec = list(max_attention_window_vec)
        self.cross_kv_cache_manager = None

    @property
    def is_variable_window(self) -> bool:
        return self.impl.is_variable_window

    @property
    def enable_block_reuse(self) -> bool:
        return self.impl.enable_block_reuse

    @property
    def tokens_per_block(self) -> int:
        return self.impl.tokens_per_block

    def start_scheduling(self):
        return self.impl.start_scheduling()

    def scheduling_has_free_blocks(self, num_required: int, window_size: int) -> bool:
        return self.impl.scheduling_has_free_blocks(num_required, window_size)

    def scheduling_remove_sequence(self, request_id: int):
        return self.impl.scheduling_remove_sequence(request_id)

    def get_needed_blocks_one_step(self, req, two_step_lookahead: bool,
                                   window_size: int, cached_summary=None) -> int:
        return self.impl.get_needed_blocks_one_step(req, two_step_lookahead,
                                                    window_size, cached_summary)

    def get_remaining_blocks_to_completion(self, req, window_size: int,
                                           cached_summary=None) -> int:
        return self.impl.get_remaining_blocks_to_completion(req, window_size,
                                                            cached_summary)

    def get_kv_cache_stats(self):
        return self.impl.get_kv_cache_stats()

    def analyze_prefix_reuse(self, unique_tokens, req):
        return self.impl.analyze_prefix_reuse(unique_tokens, req)

    def add_sequence_batch(self, request_infos, llm_requests):
        return self.impl.add_sequence_batch(request_infos, llm_requests)

    def add_token(self, request_id: int):
        return self.impl.add_token(request_id)

    def remove_sequence(self, request_id: int, llm_request,
                        pin_on_release: bool = False):
        # nanobind does not propagate C++ default arguments, so all three
        # positional args must be supplied (kvCacheManager.cpp:451).
        return self.impl.remove_sequence(request_id, llm_request, pin_on_release)

    # === Convenience accessors used by the scenarios below ====================

    def num_free_blocks(self, window_size: int) -> int:
        return self.get_kv_cache_stats().num_free_blocks_per_window_size[window_size]


# ---------------------------------------------------------------------------
# Geometry + fixture
# ---------------------------------------------------------------------------

@dataclass
class _Geom:
    tokens_per_block: int
    window_size: int
    max_num_sequences: int
    max_sequence_length: int
    chunk_size: int
    primary_blocks: int


def _make_kv_manager(geom: _Geom) -> _SchedulerKVAdapter:
    """Construct a real C++ KVCacheManager with the requested geometry.

    Minimal model surface (1 layer, 1 KV head, head_dim=64, fp16) — only the
    block-management surface matters for this test, so KV memory shape is kept
    tiny to keep allocation fast.
    """
    stream = torch.cuda.Stream()
    impl = KVCacheManagerCpp(
        num_kv_heads_per_layer=[1],
        size_per_head=64,
        tokens_per_block=geom.tokens_per_block,
        blocks_per_window={geom.window_size: (geom.primary_blocks, 0)},
        max_num_sequences=geom.max_num_sequences,
        max_beam_width=1,
        max_attention_window_vec=[geom.window_size],
        dtype=DataType.HALF,
        sink_token_length=0,
        stream=stream.cuda_stream,
        max_sequence_length=geom.max_sequence_length,
        chunk_size=geom.chunk_size,
        enable_block_reuse=False,
        cache_type=CacheTypeCpp.SELF,
    )
    impl.allocate_pools(False)
    return _SchedulerKVAdapter(impl, max_attention_window_vec=[geom.window_size])


# ---------------------------------------------------------------------------
# Scenario A — admission burst against a tight pool
# ---------------------------------------------------------------------------

def test_admission_burst_does_not_overflow_pool():
    """Many CONTEXT_INIT requests admitted in a single scheduling pass.

    Hypothesis B: with the pool sized to fit J requests, if the scheduler
    over-admits, the post-schedule `add_sequence_batch` will crash inside
    `WindowBlockManager::allocateBlock` (kvCacheManager.cpp:2365).
    """
    TPB = 64
    PROMPT_LEN = 128                          # 2 blocks per request
    BLOCKS_PER_REQ = math.ceil(PROMPT_LEN / TPB)
    POOL = 8                                  # holds exactly 4 requests
    EXPECTED_FIT = POOL // BLOCKS_PER_REQ
    NUM_REQS = 10

    geom = _Geom(tokens_per_block=TPB,
                 window_size=4096,
                 max_num_sequences=NUM_REQS,
                 max_sequence_length=4096,
                 chunk_size=4096,
                 primary_blocks=POOL)
    kv = _make_kv_manager(geom)

    scheduler = PyCapacityScheduler(
        max_num_requests=NUM_REQS,
        kv_cache_manager=kv,
        scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
    )
    requests = [
        _make_request(i, prompt_len=PROMPT_LEN, max_new_tokens=128)
        for i in range(NUM_REQS)
    ]

    fitting, _, paused = scheduler.schedule_request(requests)

    # Apply admissions exactly as PyExecutor.prepare_resources does
    # (resource_manager.py:929). The tuple is (request_id, input_length, beam_width).
    # C++ computes numContextBlocks itself: ceilDiv(inputLength, tokensPerBlock)
    # at kvCacheManager.cpp:3682.
    request_infos = [(req.py_request_id, req.prompt_len, req.py_beam_width)
                     for req in fitting]
    try:
        kv.add_sequence_batch(request_infos, list(fitting))
    except Exception as e:  # noqa: BLE001 — any C++ crash counts as reproduction
        pytest.fail(
            f"REPRODUCED at add_sequence_batch: {e}\n"
            f"  fit={len(fitting)} expected_fit={EXPECTED_FIT} "
            f"pool={POOL} blocks/req={BLOCKS_PER_REQ}")

    assert len(fitting) == EXPECTED_FIT, (
        f"OVER-ADMISSION (hypothesis B confirmed): "
        f"scheduler admitted {len(fitting)} requests, pool fits only {EXPECTED_FIT}. "
        f"free_after={kv.num_free_blocks(geom.window_size)}")


# ---------------------------------------------------------------------------
# Scenario B — simultaneous decode boundary crossings
# ---------------------------------------------------------------------------

def test_decode_boundary_burst_does_not_overflow_pool():
    """N active decode sequences all at a block boundary, tight free margin.

    Each scheduled sequence's next `add_token` triggers a 1-block allocation
    (numTokens crosses tokensPerBlock). If the scheduler over-admits across
    the boundary, `add_token` crashes in `WindowBlockManager::allocateBlock`.
    """
    TPB = 64
    NUM_ACTIVE = 8
    INITIAL_PROMPT_LEN = TPB              # exactly one full block per sequence
    INITIAL_BLOCKS_PER_REQ = 1            # held by each active req after add_sequence_batch
    USED = NUM_ACTIVE * INITIAL_BLOCKS_PER_REQ
    FREE_MARGIN = 3                       # only 3 of 8 boundary-crossings can succeed
    POOL = USED + FREE_MARGIN

    geom = _Geom(tokens_per_block=TPB,
                 window_size=4096,
                 max_num_sequences=NUM_ACTIVE,
                 max_sequence_length=4096,
                 chunk_size=4096,
                 primary_blocks=POOL)
    kv = _make_kv_manager(geom)

    # Seed: admit NUM_ACTIVE requests in CONTEXT_INIT, transition to GENERATION_IN_PROGRESS.
    # After this, each sequence holds exactly one block and getNumTokens()==TPB,
    # so the next add_token call crosses the boundary into block #2.
    seed_reqs = [
        _make_request(i,
                      prompt_len=INITIAL_PROMPT_LEN,
                      max_new_tokens=128,
                      state=LlmRequestState.CONTEXT_INIT)
        for i in range(NUM_ACTIVE)
    ]
    seed_infos = [(r.py_request_id, r.prompt_len, r.py_beam_width)
                  for r in seed_reqs]
    kv.add_sequence_batch(seed_infos, seed_reqs)
    for r in seed_reqs:
        r.state = LlmRequestState.GENERATION_IN_PROGRESS

    # Sanity: after seeding, free should be FREE_MARGIN.
    assert kv.num_free_blocks(geom.window_size) == FREE_MARGIN, (
        f"seed precondition violated: free={kv.num_free_blocks(geom.window_size)} "
        f"expected={FREE_MARGIN}")

    scheduler = PyCapacityScheduler(
        max_num_requests=NUM_ACTIVE,
        kv_cache_manager=kv,
        scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
    )
    fitting, _, paused = scheduler.schedule_request(seed_reqs)

    # Each fitting request gets one add_token (the next decode token crosses TPB).
    try:
        for req in fitting:
            kv.add_token(req.py_request_id)
    except Exception as e:  # noqa: BLE001
        pytest.fail(
            f"REPRODUCED at add_token: {e}\n"
            f"  fit={len(fitting)} paused={len(paused)} free_margin={FREE_MARGIN}\n"
            f"  fitting_ids={[r.request_id for r in fitting]}\n"
            f"  paused_ids={[r.request_id for r in paused]}")


# ---------------------------------------------------------------------------
# Scenario C — bielik-shaped steady-state churn
# ---------------------------------------------------------------------------

def test_steady_state_admission_churn_bielik_shape():
    """Realistic shape: ISL=OSL=128, TPB=64, tight pool, multi-step admission churn.

    Mirrors the pressure points of the failing bench scaled down for unit-test
    latency. Each iteration:
      1. Top up `active` from `pending` to max_batch.
      2. Run the real scheduler (MAX_UTILIZATION).
      3. Apply the schedule: `add_sequence_batch` for new admissions,
         `add_token` for active decode (incl. prefill-chunk tokens via
         per-token loop, matching resource_manager.py:1052 behavior).
      4. Remove completed sequences; re-queue paused.

    If `add_sequence_batch` or `add_token` raises, hypotheses B/C are
    reproduced and pytest.fail surfaces the precise step + state.
    """
    TPB = 64
    PROMPT_LEN = 128
    MAX_NEW = 128
    MAX_BATCH = 16
    BLOCKS_PER_SEQ_AT_PEAK = math.ceil((PROMPT_LEN + MAX_NEW) / TPB)  # 4
    POOL = MAX_BATCH * BLOCKS_PER_SEQ_AT_PEAK
    NUM_REQS = 4 * MAX_BATCH                                          # ~4 turns
    MAX_STEPS = 500

    geom = _Geom(tokens_per_block=TPB,
                 window_size=4096,
                 max_num_sequences=MAX_BATCH,
                 max_sequence_length=PROMPT_LEN + MAX_NEW + 4,
                 chunk_size=4096,
                 primary_blocks=POOL)
    kv = _make_kv_manager(geom)

    scheduler = PyCapacityScheduler(
        max_num_requests=MAX_BATCH,
        kv_cache_manager=kv,
        scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
    )

    pending = [
        _make_request(i, prompt_len=PROMPT_LEN, max_new_tokens=MAX_NEW)
        for i in range(NUM_REQS)
    ]
    active: list[LlmRequest] = []
    completed: list[LlmRequest] = []
    num_tokens_seen: dict[int, int] = {}

    for step in range(MAX_STEPS):
        while len(active) < MAX_BATCH and pending:
            active.append(pending.pop(0))
        if not active:
            break

        fitting, _, paused = scheduler.schedule_request(active)

        # Apply schedule.
        try:
            new_infos = []
            new_reqs = []
            for req in fitting:
                if req.is_context_init_state and req.is_first_context_chunk:
                    new_infos.append(
                        (req.py_request_id, req.prompt_len, req.py_beam_width))
                    new_reqs.append(req)
            if new_infos:
                kv.add_sequence_batch(new_infos, new_reqs)
            for req in new_reqs:
                req.state = LlmRequestState.GENERATION_IN_PROGRESS
                num_tokens_seen[req.py_request_id] = req.prompt_len

            # One decode token for every fitting request that's now in generation.
            for req in fitting:
                if req.is_generation_in_progress_state:
                    kv.add_token(req.py_request_id)
                    num_tokens_seen[req.py_request_id] = (
                        num_tokens_seen.get(req.py_request_id, 0) + 1)
        except Exception as e:  # noqa: BLE001
            pytest.fail(
                f"REPRODUCED at step={step}: {e}\n"
                f"  fitting_ids={[r.request_id for r in fitting]}\n"
                f"  paused_ids={[r.request_id for r in paused]}\n"
                f"  pool_total={POOL} "
                f"free_at_fail={kv.num_free_blocks(geom.window_size)}")

        # Re-queue paused at head of pending.
        for req in paused:
            if req in active:
                active.remove(req)
        pending = list(paused) + pending

        # Drop completed.
        still_active = []
        for req in fitting:
            n = num_tokens_seen.get(req.py_request_id, 0)
            if n >= PROMPT_LEN + MAX_NEW:
                kv.remove_sequence(req.py_request_id, req, False)
                completed.append(req)
                num_tokens_seen.pop(req.py_request_id, None)
            else:
                still_active.append(req)
        retained = [r for r in active if r not in fitting and r not in paused]
        active = still_active + retained

    assert len(completed) >= NUM_REQS // 2, (
        f"insufficient progress: completed={len(completed)}/{NUM_REQS} "
        f"in {MAX_STEPS} steps")
