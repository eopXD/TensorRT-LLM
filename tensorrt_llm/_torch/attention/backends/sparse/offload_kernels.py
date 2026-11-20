# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernels shared by sparse attention models that offload KV history.

A decode step's selection names units of a request's history; the KV cache
manager fetches them into per-request arena rows. These kernels mask a
selection to the history the Batch published, and assemble the arena rows the
attention reads from the fetch output and the in-flight unit's write row.

Model inputs follow the requests' input positions, while the native Batch keeps
each request in a fixed row, its slot, for as long as the request is scheduled.
The kernels read input-ordered tensors by position and Batch-owned tensors by
row, through a per-step map from each request's position to its Batch row.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Message of the opt-in device assertion on a merge's coverage flag.
SPARSE_ARENA_COVERAGE_MESSAGE = (
    "Sparse offload: a live selection of a fetched or in-flight unit has no arena row"
)


def check_sparse_offload_tensor(
    tensor: torch.Tensor, name: str, ndim: int, device: torch.device
) -> None:
    """Require a CUDA int32 tensor of ``ndim`` dimensions on ``device``."""
    if tensor.dtype != torch.int32 or tensor.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}D int32 tensor")
    if not tensor.is_cuda or tensor.device != device:
        raise ValueError(f"{name} must be on CUDA device {device}")


def check_sparse_arena_merge_inputs(
    topk_indices: torch.Tensor,
    unit_lengths: torch.Tensor,
    history_lengths: torch.Tensor,
    batch_rows: torch.Tensor,
    fetched_indices: torch.Tensor,
    write_indices: torch.Tensor,
    active_request_count: torch.Tensor,
    coverage: torch.Tensor | None,
    *,
    queries_per_request: int = 1,
) -> None:
    """Check the inputs every merge of fetched arena rows into a selection reads.

    Selections are [N, K] in query order, a request's queries consecutive and at
    most ``queries_per_request`` (Q) of them. The fetch output [B, Q*K], the write
    destinations [B, U], the history lengths and the Batch row map [B] are by Batch
    row. Unit lengths cover the requests the queries name and fit the Batch. The
    active count and the coverage flag are [1].
    """
    device = fetched_indices.device
    for tensor, name, ndim in (
        (topk_indices, "topk_indices", 2),
        (unit_lengths, "unit_lengths", 1),
        (history_lengths, "history_lengths", 1),
        (batch_rows, "batch_rows", 1),
        (fetched_indices, "fetched_indices", 2),
        (write_indices, "write_indices", 2),
        (active_request_count, "active_request_count", 1),
    ):
        check_sparse_offload_tensor(tensor, name, ndim, device)
    if queries_per_request <= 0:
        raise ValueError(f"queries_per_request must be positive, got {queries_per_request}")
    num_rows, width = fetched_indices.shape
    num_queries, topk = topk_indices.shape
    if topk * queries_per_request != width:
        raise ValueError("The fetch output must hold top-k for each query of a request")
    if active_request_count.shape != (1,):
        raise ValueError("active_request_count must be [1]")
    if (
        history_lengths.shape != (num_rows,)
        or write_indices.shape[0] != num_rows
        or batch_rows.shape != (num_rows,)
    ):
        raise ValueError(
            "history_lengths, write_indices and batch_rows must cover the native Batch"
        )
    if unit_lengths.numel() > num_rows or num_queries > unit_lengths.numel() * queries_per_request:
        raise ValueError("Query and unit-length rows must fit the native Batch capacity")
    if coverage is not None:
        check_sparse_offload_tensor(coverage, "coverage", 1, device)
        if coverage.shape != (1,):
            raise ValueError("coverage must be [1]")


@triton.jit
def query_segment(
    query, request, request_indices_ptr, request_indices_stride, QUERIES: tl.constexpr
):
    """Return ``query``'s place among the queries of its request, counted from zero.

    A request's queries are consecutive in input order, so this counts the queries
    just before ``query`` that name the same request, up to ``QUERIES - 1`` of them.
    """
    segment = 0
    for back in tl.static_range(1, QUERIES):
        previous = tl.load(
            request_indices_ptr + (query - back) * request_indices_stride,
            mask=query >= back,
            other=-1,
        )
        segment += (previous == request).to(tl.int32)
    return segment


@triton.jit(do_not_specialize=["num_lengths"])
def _select_sparse_history_units_kernel(
    topk_ptr,
    request_indices_ptr,
    batch_rows_ptr,
    active_request_count_ptr,
    unit_lengths_ptr,
    history_lengths_ptr,
    out_ptr,
    topk_stride0,
    topk_stride1,
    request_indices_stride,
    batch_rows_stride,
    unit_lengths_stride,
    history_lengths_stride,
    out_stride0,
    out_stride1,
    NUM_ROWS: tl.constexpr,
    num_lengths,
    TOPK: tl.constexpr,
    QUERIES: tl.constexpr,
    UNIT_TOKENS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    query = tl.program_id(0)
    active_count = tl.load(active_request_count_ptr)
    request = tl.load(request_indices_ptr + query * request_indices_stride)
    # Live queries precede graph padding, and each live request feeds at most QUERIES.
    active = (query < active_count * QUERIES) & (request >= 0) & (request < active_count)
    active = active & (request < NUM_ROWS) & (request < num_lengths)
    segment = query_segment(query, request, request_indices_ptr, request_indices_stride, QUERIES)
    active = active & (segment < QUERIES)
    row = tl.load(batch_rows_ptr + request * batch_rows_stride, active, other=-1)
    active = active & (row >= 0) & (row < NUM_ROWS)
    length = tl.load(unit_lengths_ptr + request * unit_lengths_stride, active, other=0)
    history = tl.load(history_lengths_ptr + row * history_lengths_stride, active, other=0)
    k = tl.arange(0, BLOCK_K)
    unit = tl.load(topk_ptr + query * topk_stride0 + k * topk_stride1, k < TOPK, other=-1)
    valid = (unit >= 0) & (unit < length) & (unit < history // UNIT_TOKENS)
    # Each live query owns its segment of its request's Batch row. Keep top-k
    # positions, including holes left by in-flight units, so fetch output has the
    # same slot meaning.
    tl.store(
        out_ptr + row * out_stride0 + (segment * TOPK + k) * out_stride1,
        tl.where(valid, unit, -1),
        mask=active & (k < TOPK),
    )


def select_sparse_history_units(
    topk_indices: torch.Tensor,
    request_indices: torch.Tensor,
    batch_rows: torch.Tensor,
    active_request_count: torch.Tensor,
    unit_lengths: torch.Tensor,
    history_lengths: torch.Tensor,
    out: torch.Tensor,
    *,
    unit_tokens: int,
    queries_per_request: int = 1,
) -> None:
    """Mask non-history selections and scatter them into native Batch rows.

    All tensors are CUDA int32 on one device. ``topk_indices`` is [Q, K] in
    selection units, with distinct nonnegative selections per live row.
    ``request_indices`` [Q] gives each query's request by input position; a
    request's queries are consecutive, at most ``queries_per_request`` of them.
    ``batch_rows`` [B] gives each request's native Batch row, or -1 for a
    position without a live request; live requests have distinct
    rows, in any order and with free rows between them. ``active_request_count``
    [1] excludes graph padding. ``unit_lengths`` [R] counts each request's
    selectable units by input position, and may cover fewer requests than the
    native Batch capacity B. ``history_lengths`` [B] is the Batch-owned,
    published history watermark in original tokens.

    ``out`` is caller-owned [B, queries_per_request * K], disjoint from the
    inputs. The i-th query of a request fills segment i of its row, columns
    [i * K, (i + 1) * K); a request with fewer queries leaves the rest -1. Only
    ordinals below both the unit length and history_lengths // unit_tokens are
    fetched.
    Top-k order and padding positions are preserved; in-flight units and invalid
    selections become -1. Batch rows without a live query are overwritten with
    -1 on every call, including when a graph's Q is smaller than B. No tensor
    allocation, host read, or synchronization is performed.
    """
    device = out.device
    for tensor, name, ndim in (
        (topk_indices, "topk_indices", 2),
        (request_indices, "request_indices", 1),
        (batch_rows, "batch_rows", 1),
        (active_request_count, "active_request_count", 1),
        (unit_lengths, "unit_lengths", 1),
        (history_lengths, "history_lengths", 1),
        (out, "out", 2),
    ):
        check_sparse_offload_tensor(tensor, name, ndim, device)
    num_rows, width = out.shape
    num_queries, topk = topk_indices.shape
    if unit_tokens <= 0 or queries_per_request <= 0:
        raise ValueError("unit_tokens and queries_per_request must be positive")
    if topk * queries_per_request != width:
        raise ValueError(
            "The selection row must hold top-k for each query a request may feed a step"
        )
    if request_indices.shape != (num_queries,) or active_request_count.shape != (1,):
        raise ValueError("request_indices must be [Q] and active_request_count must be [1]")
    if history_lengths.shape != (num_rows,) or batch_rows.shape != (num_rows,):
        raise ValueError("history_lengths and batch_rows must cover the native Batch capacity")
    if not num_queries <= unit_lengths.numel() * queries_per_request:
        raise ValueError("Every query must name a request with a unit length")
    if unit_lengths.numel() > num_rows:
        raise ValueError("Unit-length rows must fit the native Batch capacity")

    with torch.cuda.device(device):
        # Clear separately: padded queries can carry stale or invalid request IDs,
        # and a smaller graph may leave native Batch rows without any query.
        out.fill_(-1)
        if num_queries == 0 or topk == 0:
            return
        _select_sparse_history_units_kernel[(num_queries,)](
            topk_indices,
            request_indices,
            batch_rows,
            active_request_count,
            unit_lengths,
            history_lengths,
            out,
            *topk_indices.stride(),
            request_indices.stride(0),
            batch_rows.stride(0),
            unit_lengths.stride(0),
            history_lengths.stride(0),
            *out.stride(),
            NUM_ROWS=num_rows,
            num_lengths=unit_lengths.numel(),
            TOPK=topk,
            QUERIES=queries_per_request,
            UNIT_TOKENS=unit_tokens,
            BLOCK_K=triton.next_power_of_2(topk),
        )


@triton.jit
def sparse_arena_rows_in_flight(
    unit,
    in_width,
    live,
    length,
    history_units,
    fetched_row_ptr,
    fetched_stride,
    lane,
    write_row_ptr,
    write_stride,
    IN_FLIGHT: tl.constexpr,
    coverage_ptr,
    CHECK_COVERAGE: tl.constexpr,
):
    """Return the arena row each lane's selected unit is read from.

    ``unit`` holds a query's selected units in its lanes, -1 for none, and
    ``in_width`` the lanes that hold a selection. A unit below the request's
    ``history_units`` takes its fetched row, from ``fetched_row_ptr`` at ``lane``.
    A request has up to ``IN_FLIGHT`` units in flight: unit ``history_units + i``
    takes entry ``i`` of the row's write destinations at ``write_row_ptr``. Padding
    queries (``live`` false), invalid selections and units past ``length`` take -1.

    With ``CHECK_COVERAGE``, clears the int32 flag at ``coverage_ptr`` when a live,
    valid selection of a history or in-flight unit gets no row.
    """
    valid = live & (unit >= 0) & (unit < length)
    fetched = tl.load(
        fetched_row_ptr + lane * fetched_stride,
        mask=valid & in_width & (unit < history_units),
        other=-1,
    )
    flight = unit - history_units
    in_flight = valid & in_width & (flight >= 0) & (flight < IN_FLIGHT)
    write = tl.load(write_row_ptr + flight * write_stride, mask=in_flight, other=-1)
    rows = tl.where(in_flight, write, fetched)
    if CHECK_COVERAGE:
        required = valid & in_width & (flight < IN_FLIGHT)
        covered = tl.min((~required | (rows >= 0)).to(tl.int32), axis=0)
        tl.atomic_and(coverage_ptr, covered, sem="relaxed")
    return rows


@triton.jit(do_not_specialize=["num_lengths"])
def _merge_sparse_arena_indices_kernel(
    topk_ptr,
    request_indices_ptr,
    batch_rows_ptr,
    active_request_count_ptr,
    unit_lengths_ptr,
    history_lengths_ptr,
    fetched_ptr,
    write_ptr,
    out_ptr,
    topk_stride0,
    topk_stride1,
    request_indices_stride,
    batch_rows_stride,
    unit_lengths_stride,
    history_lengths_stride,
    fetched_stride0,
    fetched_stride1,
    write_stride0,
    write_stride1,
    out_stride0,
    out_stride1,
    coverage_ptr,
    NUM_ROWS: tl.constexpr,
    num_lengths,
    TOPK: tl.constexpr,
    QUERIES: tl.constexpr,
    IN_FLIGHT: tl.constexpr,
    UNIT_TOKENS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CHECK_COVERAGE: tl.constexpr,
):
    query = tl.program_id(0)
    active_count = tl.load(active_request_count_ptr)
    request = tl.load(request_indices_ptr + query * request_indices_stride)
    live = (query < active_count * QUERIES) & (request >= 0) & (request < active_count)
    live = live & (request < NUM_ROWS) & (request < num_lengths)
    segment = query_segment(query, request, request_indices_ptr, request_indices_stride, QUERIES)
    live = live & (segment < QUERIES)
    row = tl.load(batch_rows_ptr + request * batch_rows_stride, live, other=-1)
    live = live & (row >= 0) & (row < NUM_ROWS)
    length = tl.load(unit_lengths_ptr + request * unit_lengths_stride, live, other=0)
    history_units = (
        tl.load(history_lengths_ptr + row * history_lengths_stride, live, other=0) // UNIT_TOKENS
    )
    k = tl.arange(0, BLOCK_K)
    in_width = k < TOPK
    unit = tl.load(topk_ptr + query * topk_stride0 + k * topk_stride1, in_width, other=-1)
    rows = sparse_arena_rows_in_flight(
        unit,
        in_width,
        live,
        length,
        history_units,
        fetched_ptr + row * fetched_stride0 + segment * TOPK * fetched_stride1,
        fetched_stride1,
        k,
        write_ptr + row * write_stride0,
        write_stride1,
        IN_FLIGHT,
        coverage_ptr,
        CHECK_COVERAGE,
    )
    tl.store(out_ptr + query * out_stride0 + k * out_stride1, rows, mask=in_width)


def merge_sparse_arena_indices(
    topk_indices: torch.Tensor,
    request_indices: torch.Tensor,
    batch_rows: torch.Tensor,
    active_request_count: torch.Tensor,
    unit_lengths: torch.Tensor,
    history_lengths: torch.Tensor,
    fetched_indices: torch.Tensor,
    write_indices: torch.Tensor,
    out: torch.Tensor,
    *,
    unit_tokens: int,
    queries_per_request: int = 1,
    coverage: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return each query's selection as rows of one sparse buffer's arena.

    Inputs follow :func:`select_sparse_history_units`. ``fetched_indices`` is
    the fetch output of one buffer, [B, queries_per_request * K] in Batch row
    order, each query's top-k in its own segment; ``write_indices`` [B, U] is
    that buffer's write destinations by Batch row, entry ``i`` the arena row of
    unit ``history_units + i``, written this step. A [B] tensor is one unit in
    flight per row. Both may be strided views of
    the per-buffer outputs.

    ``out`` is [Q, K] in query order. A history unit takes its fetched row, an
    in-flight unit its write row, and padding, invalid selections and units past
    the request's length take -1. No tensor allocation, host read, or
    synchronization is performed. With ``coverage``, an int32 [1] diagnostic
    flag, the merge also resets the flag, clears it when a live selection gets
    no row, and asserts on it on the device. A failed assertion makes the CUDA
    context unusable, so pass it only for debugging.
    """
    if write_indices.ndim == 1:
        write_indices = write_indices.unsqueeze(1)
    check_sparse_arena_merge_inputs(
        topk_indices,
        unit_lengths,
        history_lengths,
        batch_rows,
        fetched_indices,
        write_indices,
        active_request_count,
        coverage,
        queries_per_request=queries_per_request,
    )
    device = fetched_indices.device
    check_sparse_offload_tensor(request_indices, "request_indices", 1, device)
    check_sparse_offload_tensor(out, "out", 2, device)
    num_rows = fetched_indices.shape[0]
    num_queries, topk = topk_indices.shape
    if unit_tokens <= 0:
        raise ValueError(f"unit_tokens must be positive, got {unit_tokens}")
    if out.shape != (num_queries, topk) or request_indices.shape != (num_queries,):
        raise ValueError("out must be [Q, K] and request_indices [Q]")
    if num_queries == 0 or topk == 0:
        return out
    with torch.cuda.device(device):
        if coverage is not None:
            coverage.fill_(1)
        _merge_sparse_arena_indices_kernel[(num_queries,)](
            topk_indices,
            request_indices,
            batch_rows,
            active_request_count,
            unit_lengths,
            history_lengths,
            fetched_indices,
            write_indices,
            out,
            *topk_indices.stride(),
            request_indices.stride(0),
            batch_rows.stride(0),
            unit_lengths.stride(0),
            history_lengths.stride(0),
            *fetched_indices.stride(),
            *write_indices.stride(),
            *out.stride(),
            coverage if coverage is not None else out,
            NUM_ROWS=num_rows,
            num_lengths=unit_lengths.numel(),
            TOPK=topk,
            QUERIES=queries_per_request,
            IN_FLIGHT=write_indices.shape[1],
            UNIT_TOKENS=unit_tokens,
            BLOCK_K=triton.next_power_of_2(topk),
            CHECK_COVERAGE=coverage is not None,
        )
        if coverage is not None:
            torch._assert_async(coverage, SPARSE_ARENA_COVERAGE_MESSAGE)
    return out


@triton.jit(do_not_specialize=["num_rows", "num_out"])
def _gather_sparse_batch_rows_kernel(
    values_ptr,
    batch_rows_ptr,
    out_ptr,
    values_stride0,
    values_stride1,
    batch_rows_stride,
    out_stride0,
    out_stride1,
    num_rows,
    num_out,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
):
    position = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_range = position < num_out
    row = tl.load(batch_rows_ptr + position * batch_rows_stride, in_range, other=-1)
    valid = in_range & (row >= 0) & (row < num_rows)
    column = tl.arange(0, BLOCK_WIDTH)
    in_width = column < WIDTH
    value = tl.load(
        values_ptr + row[:, None] * values_stride0 + column[None, :] * values_stride1,
        valid[:, None] & in_width[None, :],
        other=-1,
    )
    tl.store(
        out_ptr + position[:, None] * out_stride0 + column[None, :] * out_stride1,
        value,
        mask=in_range[:, None] & in_width[None, :],
    )


def gather_sparse_batch_rows(
    values: torch.Tensor,
    batch_rows: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Read a per-Batch-row value, or row of values, for each request, in input order.

    ``values`` [B] or [B, W] holds int32 values per native Batch row, such as one
    buffer's write rows; ``batch_rows`` [N] gives each request's Batch row, or -1 for
    a position without a live request, which reads -1. Both may be strided. ``out``
    [N] or [N, W] receives the result in place, also possibly strided, and is
    allocated when None. One kernel launch; no host read or synchronization.
    """
    device = values.device
    if values.ndim not in (1, 2):
        raise ValueError("values must be a 1D or 2D int32 tensor")
    shape = (*batch_rows.shape, *values.shape[1:])
    if out is None:
        out = torch.empty(shape, dtype=torch.int32, device=device)
    check_sparse_offload_tensor(values, "values", values.ndim, device)
    check_sparse_offload_tensor(batch_rows, "batch_rows", 1, device)
    check_sparse_offload_tensor(out, "out", values.ndim, device)
    if out.shape != shape:
        raise ValueError("out must have one entry, or one row of entries, per request")
    num_out = batch_rows.numel()
    if out.numel() == 0:
        return out
    # A [B] input is one column of values.
    values_2d = values if values.ndim == 2 else values.unsqueeze(1)
    out_2d = out if out.ndim == 2 else out.unsqueeze(1)
    width = values_2d.shape[1]
    block = 256
    with torch.cuda.device(device):
        _gather_sparse_batch_rows_kernel[(triton.cdiv(num_out, block),)](
            values_2d,
            batch_rows,
            out_2d,
            *values_2d.stride(),
            batch_rows.stride(0),
            *out_2d.stride(),
            values_2d.shape[0],
            num_out,
            WIDTH=width,
            BLOCK=block,
            BLOCK_WIDTH=triton.next_power_of_2(width),
        )
    return out


@triton.jit
def _sparse_token_write_rows_kernel(
    write_ptr,
    batch_rows_ptr,
    request_indices_ptr,
    active_request_count_ptr,
    kv_lens_ptr,
    history_lengths_ptr,
    block_offsets_ptr,
    write_slots_ptr,
    write_stride0,
    write_stride1,
    batch_rows_stride,
    request_indices_stride,
    kv_lens_stride,
    history_lengths_stride,
    block_offsets_stride0,
    block_offsets_stride1,
    write_slots_stride,
    num_tokens,
    num_requests,
    NUM_ROWS: tl.constexpr,
    QUERIES: tl.constexpr,
    IN_FLIGHT: tl.constexpr,
    UNIT_TOKENS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_range = token < num_tokens
    active_count = tl.load(active_request_count_ptr)
    request = tl.load(request_indices_ptr + token * request_indices_stride, in_range, other=-1)
    live = in_range & (token < active_count * QUERIES) & (request >= 0)
    live = live & (request < active_count) & (request < num_requests)
    # The request's tokens are consecutive: those just before this one give its
    # place among them, and those around it how many the request feeds this step.
    segment = tl.zeros([BLOCK], dtype=tl.int32)
    seq_len = tl.full([BLOCK], 1, dtype=tl.int32)
    for back in tl.static_range(1, QUERIES):
        previous = tl.load(
            request_indices_ptr + (token - back) * request_indices_stride,
            mask=in_range & (token >= back),
            other=-1,
        )
        following = tl.load(
            request_indices_ptr + (token + back) * request_indices_stride,
            mask=in_range & (token + back < num_tokens),
            other=-1,
        )
        segment += (previous == request).to(tl.int32)
        seq_len += (previous == request).to(tl.int32) + (following == request).to(tl.int32)
    live = live & (seq_len <= QUERIES)
    row = tl.load(batch_rows_ptr + request * batch_rows_stride, live, other=-1)
    live = live & (row >= 0) & (row < NUM_ROWS)
    kv_len = tl.load(kv_lens_ptr + request * kv_lens_stride, live, other=0)
    history = tl.load(history_lengths_ptr + row * history_lengths_stride, live, other=0)
    # The token's position, and the in-flight unit it completes, counted from the
    # unit at the history watermark.
    position = kv_len - seq_len + segment
    flight = position // UNIT_TOKENS - history // UNIT_TOKENS
    live = live & (flight >= 0) & (flight < IN_FLIGHT)
    arena_row = tl.load(write_ptr + row * write_stride0 + flight * write_stride1, live, other=-1)
    live = live & (arena_row >= 0)
    tl.store(
        block_offsets_ptr + request * block_offsets_stride0 + segment * block_offsets_stride1,
        arena_row,
        mask=live,
    )
    tl.store(
        write_slots_ptr + token * write_slots_stride, tl.where(live, segment, -1), mask=in_range
    )


def sparse_token_write_rows(
    write_indices: torch.Tensor,
    batch_rows: torch.Tensor,
    request_indices: torch.Tensor,
    active_request_count: torch.Tensor,
    kv_lens: torch.Tensor,
    history_lengths: torch.Tensor,
    block_offsets: torch.Tensor,
    write_slots: torch.Tensor,
    *,
    unit_tokens: int,
    queries_per_request: int,
) -> None:
    """Address each query token's new KV row in one sparse buffer's arena.

    A request's tokens are written to the in-flight units at their positions: the
    i-th of the n tokens request r feeds this step sits at ``kv_lens[r] - n + i``,
    n counted from ``request_indices``, and the
    unit holding that position is entry ``unit - history_units`` of the row's write
    destinations ``write_indices`` [B, U]. A writer that addresses a request through
    one-token pages reads the arena row from ``block_offsets`` [R, Q] at the token's
    page, and the page from ``write_slots`` [N], which holds -1 for a token not to
    write: graph padding, a token without a request or with no in-flight cell.

    ``request_indices`` [N] gives each token's request by input position, a
    request's tokens consecutive, at most ``queries_per_request`` of them;
    ``kv_lens`` [R] counts each request's tokens with this step's, by input
    position; ``batch_rows`` and ``history_lengths`` [B] by Batch row, as in
    :func:`select_sparse_history_units`. Pages no token writes keep their values.
    No allocation, host read or synchronization.
    """
    device = write_slots.device
    for tensor, name, ndim in (
        (write_indices, "write_indices", 2),
        (batch_rows, "batch_rows", 1),
        (request_indices, "request_indices", 1),
        (active_request_count, "active_request_count", 1),
        (kv_lens, "kv_lens", 1),
        (history_lengths, "history_lengths", 1),
        (block_offsets, "block_offsets", 2),
        (write_slots, "write_slots", 1),
    ):
        check_sparse_offload_tensor(tensor, name, ndim, device)
    num_rows = write_indices.shape[0]
    num_tokens = request_indices.numel()
    num_requests = kv_lens.numel()
    if unit_tokens <= 0 or queries_per_request <= 0:
        raise ValueError("unit_tokens and queries_per_request must be positive")
    if batch_rows.shape != (num_rows,) or history_lengths.shape != (num_rows,):
        raise ValueError("batch_rows and history_lengths must cover the native Batch")
    if block_offsets.shape[1] < queries_per_request or block_offsets.shape[0] < num_requests:
        raise ValueError("block_offsets must hold a page for each query of each request")
    if write_slots.shape != (num_tokens,):
        raise ValueError("write_slots must hold one entry per token")
    if num_tokens == 0:
        return
    block = 128
    with torch.cuda.device(device):
        _sparse_token_write_rows_kernel[(triton.cdiv(num_tokens, block),)](
            write_indices,
            batch_rows,
            request_indices,
            active_request_count,
            kv_lens,
            history_lengths,
            block_offsets,
            write_slots,
            *write_indices.stride(),
            batch_rows.stride(0),
            request_indices.stride(0),
            kv_lens.stride(0),
            history_lengths.stride(0),
            *block_offsets.stride(),
            write_slots.stride(0),
            num_tokens,
            num_requests,
            NUM_ROWS=num_rows,
            QUERIES=queries_per_request,
            IN_FLIGHT=write_indices.shape[1],
            UNIT_TOKENS=unit_tokens,
            BLOCK=block,
        )
