# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for sparse offload with several query tokens per request, as MTP feeds a step."""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.offload_kernels import (
    merge_sparse_arena_indices,
    select_sparse_history_units,
    sparse_token_write_rows,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.sparse_offload_memory import (
    SPARSE_MAX_CACHE_ENTRIES,
    resolve_sparse_cache_factor,
    sparse_cache_config,
)

_requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _cuda(value) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.int32, device="cuda")


def _segments(requests: list[int]) -> list[int]:
    """Each query's place among the queries of its request."""
    seen: dict[int, int] = {}
    out = []
    for request in requests:
        out.append(seen.get(request, 0))
        seen[request] = out[-1] + 1
    return out


# Two live requests feeding three and one query tokens, then one padding token.
# Request 0 sits in Batch row 2 with history 10, request 1 in row 0 with history 4.
_REQUESTS = [0, 0, 0, 1, 2]
_BATCH_ROWS = [2, 0, -1, -1]
_HISTORY = [4, 0, 10, 0]
_ACTIVE = 2
_QUERIES = 3
_TOPK = 4
# Request 0's verify tokens sit at 10, 11 and 12; request 1's one token at 4.
_KV_LENS = [13, 5, 0]
_TOPK_INDICES = [
    [9, 3, 10, -1],  # history 9 and 3, the token itself at 10
    [3, 11, 10, 0],  # shares 3 and 10 with the first query
    [12, 2, 3, 11],
    [4, 1, 0, -1],
    [0, 1, 2, 3],  # padding: no live request
]


def _select() -> torch.Tensor:
    out = torch.full((4, _QUERIES * _TOPK), 999, dtype=torch.int32, device="cuda")
    select_sparse_history_units(
        _cuda(_TOPK_INDICES),
        _cuda(_REQUESTS),
        _cuda(_BATCH_ROWS),
        _cuda([_ACTIVE]),
        _cuda(_KV_LENS),
        _cuda(_HISTORY),
        out,
        unit_tokens=1,
        queries_per_request=_QUERIES,
    )
    return out


@_requires_cuda
def test_select_gives_each_query_its_own_segment_of_the_request_row():
    out = _select().cpu().tolist()
    # Row 2 holds request 0: its queries fill segments 0, 1 and 2, history only.
    assert out[2] == [9, 3, -1, -1, 3, -1, -1, 0, -1, 2, 3, -1]
    # Row 0 holds request 1's one query; its other segments stay empty.
    assert out[0] == [-1, 1, 0, -1] + [-1] * 8
    # Free rows select nothing, and padding writes no row.
    assert out[1] == [-1] * 12 and out[3] == [-1] * 12


@_requires_cuda
def test_merge_reads_each_querys_segment_and_in_flight_units_by_offset():
    selected = _select()
    # A fetch names an arena row for each selected history unit: 100 + unit.
    fetched = torch.where(selected >= 0, selected + 100, -1)
    # Write destinations by row: entry i is unit history + i. Row 2 has three in
    # flight, row 0 one; -1 past the reservation.
    writes = _cuda([[50, -1, -1], [-1, -1, -1], [60, 61, 62], [-1, -1, -1]])
    out = torch.empty((5, _TOPK), dtype=torch.int32, device="cuda")
    merge_sparse_arena_indices(
        _cuda(_TOPK_INDICES),
        _cuda(_REQUESTS),
        _cuda(_BATCH_ROWS),
        _cuda([_ACTIVE]),
        _cuda(_KV_LENS),
        _cuda(_HISTORY),
        fetched,
        writes,
        out,
        unit_tokens=1,
        queries_per_request=_QUERIES,
    )
    assert out.cpu().tolist() == [
        [109, 103, 60, -1],
        [103, 61, 60, 100],
        [62, 102, 103, 61],
        [50, 101, 100, -1],
        [-1, -1, -1, -1],
    ]


@_requires_cuda
def test_merge_matches_a_reference_for_random_rows():
    generator = torch.Generator().manual_seed(3)
    queries, topk, in_flight = 4, 64, 7
    requests = [0] * 4 + [1] * 2 + [2] + [3] * 3
    rows = [5, 1, 3, 0, -1, -1, -1, -1]
    history = torch.tensor([300, 7, 0, 41, 0, 900, 0, 0], dtype=torch.int32)
    kv_lens = [905, 9, 42, 303]
    topk_indices = torch.stack(
        [
            torch.randperm(kv_lens[r], generator=generator)[:topk]
            if kv_lens[r] >= topk
            else torch.cat(
                [
                    torch.randperm(kv_lens[r], generator=generator),
                    torch.full((topk - kv_lens[r],), -1),
                ]
            )
            for r in requests
        ]
    ).to(torch.int32)
    selected = torch.empty((8, queries * topk), dtype=torch.int32, device="cuda")
    common = (
        _cuda(topk_indices),
        _cuda(requests),
        _cuda(rows),
        _cuda([4]),
        _cuda(kv_lens),
        _cuda(history),
    )
    select_sparse_history_units(*common, selected, unit_tokens=1, queries_per_request=queries)
    fetched = torch.where(selected >= 0, selected * 2 + 1, -1)
    writes = torch.arange(8 * in_flight, dtype=torch.int32, device="cuda").reshape(8, in_flight)
    out = torch.empty((len(requests), topk), dtype=torch.int32, device="cuda")
    merge_sparse_arena_indices(
        *common, fetched, writes, out, unit_tokens=1, queries_per_request=queries
    )

    expected = torch.full((len(requests), topk), -1, dtype=torch.int32)
    for query, (request, segment) in enumerate(zip(requests, _segments(requests))):
        row = rows[request]
        base = history[row].item()
        for k, unit in enumerate(topk_indices[query].tolist()):
            if not 0 <= unit < kv_lens[request]:
                continue
            if unit < base:
                expected[query, k] = unit * 2 + 1
                assert selected[row, segment * topk + k].item() == unit
            elif unit - base < in_flight:
                expected[query, k] = writes[row, unit - base].item()
    torch.testing.assert_close(out.cpu(), expected)


def _write_rows(requests, kv_lens, history, writes, queries):
    block_offsets = torch.full((len(kv_lens), queries), 777, dtype=torch.int32, device="cuda")
    slots = torch.full((len(requests),), 777, dtype=torch.int32, device="cuda")
    sparse_token_write_rows(
        _cuda(writes),
        _cuda(_BATCH_ROWS),
        _cuda(requests),
        _cuda([_ACTIVE]),
        _cuda(kv_lens),
        _cuda(history),
        block_offsets,
        slots,
        unit_tokens=1,
        queries_per_request=queries,
    )
    return block_offsets.cpu().tolist(), slots.cpu().tolist()


@_requires_cuda
def test_write_rows_address_every_verify_token_at_its_position():
    writes = [[50, 51, -1], [-1, -1, -1], [60, 61, 62], [-1, -1, -1]]
    pages, slots = _write_rows(_REQUESTS, _KV_LENS, _HISTORY, writes, _QUERIES)
    # Request 0's tokens at 10, 11 and 12 take entries 0, 1 and 2 of row 2.
    assert pages[0] == [60, 61, 62]
    # Request 1's one token at 4 takes entry 0 of row 0; its other pages are untouched.
    assert pages[1] == [50, 777, 777]
    # The padding token writes nothing.
    assert slots == [0, 1, 2, 0, -1]


@_requires_cuda
def test_write_rows_follow_a_draft_step_past_the_verify_tokens():
    # A draft step feeds one token per request, past the verify positions: request 0
    # at 13 (entry 3 of 5), request 1 at 6 (entry 2), still above the watermarks.
    writes = [[50, 51, 52, -1, -1], [-1] * 5, [60, 61, 62, 63, 64], [-1] * 5]
    pages, slots = _write_rows([0, 1], [14, 7], _HISTORY, writes, _QUERIES)
    assert pages[0][0] == 63 and pages[1][0] == 52
    assert slots == [0, 0]
    # A token with no in-flight cell is skipped rather than written anywhere.
    pages, slots = _write_rows([0, 1], [14, 9], _HISTORY, writes, _QUERIES)
    assert slots == [0, -1]


def test_default_cache_factor_scales_with_the_queries():
    assert resolve_sparse_cache_factor(2.0, explicit=False, queries_per_request=4) == 8.0
    assert resolve_sparse_cache_factor(2.0, explicit=False, queries_per_request=1) == 2.0
    assert resolve_sparse_cache_factor(5.0, explicit=True, queries_per_request=4) == 5.0
    with pytest.raises(ValueError, match="at least 4"):
        resolve_sparse_cache_factor(3.0, explicit=True, queries_per_request=4)


@pytest.mark.parametrize(
    "queries, in_flight_tokens, expected_units",
    [
        (1, 1, 4224),  # today: two selections and the token being written
        (2, 7, 8320),  # MTP1: four selections
        (3, 9, 12416),  # MTP2: six selections
        (4, 13, 16256),  # MTP3: eight selections, capped where the probe can name entries
    ],
)
def test_glm_cache_holds_every_query_selection(queries, in_flight_tokens, expected_units):
    config = sparse_cache_config(
        num_selected_units=2048,
        unit_tokens=1,
        tokens_per_block=128,
        gpu_cache_factor=resolve_sparse_cache_factor(
            2.0, explicit=False, queries_per_request=queries
        ),
        max_in_flight_tokens=in_flight_tokens,
        queries_per_request=queries,
    )
    assert config.max_cached_units == expected_units
    assert config.max_cached_units <= SPARSE_MAX_CACHE_ENTRIES
    assert config.max_cached_units >= queries * 2048 + in_flight_tokens


def test_cache_rejects_queries_whose_selections_exceed_the_probe():
    with pytest.raises(ValueError, match="more than the"):
        sparse_cache_config(
            num_selected_units=2048,
            unit_tokens=1,
            tokens_per_block=128,
            gpu_cache_factor=16.0,
            max_in_flight_tokens=29,
            queries_per_request=8,
        )


def test_cache_shrinks_to_the_resolve_the_device_holds():
    # GLM-5.2 MTP4: the default factor asks for ten selections, which the probe caps at
    # 16256 units. A device whose resolve holds 15968 entries beside the row gets the
    # most whole 128-unit cells below that, still above all five queries' selections.
    asked = []

    def device_limit(width: int, in_flight_units: int) -> int:
        asked.append((width, in_flight_units))
        return 15968

    shape = dict(
        num_selected_units=2048,
        unit_tokens=1,
        tokens_per_block=128,
        max_in_flight_tokens=11,
        queries_per_request=5,
    )
    config = sparse_cache_config(
        gpu_cache_factor=resolve_sparse_cache_factor(2.0, explicit=False, queries_per_request=5),
        max_resolvable_entries=device_limit,
        **shape,
    )
    assert asked == [(5 * 2048, 11)]
    assert config.max_cached_units == 124 * 128
    # The device bounds an explicit factor the same way.
    config = sparse_cache_config(gpu_cache_factor=8.0, max_resolvable_entries=device_limit, **shape)
    assert config.max_cached_units == 124 * 128
    # A device that cannot hold every query's selection beside the units in flight
    # cannot run the step at all.
    with pytest.raises(ValueError, match="name and hold on this device"):
        sparse_cache_config(
            gpu_cache_factor=5.0,
            max_resolvable_entries=lambda width, in_flight_units: width + in_flight_units - 1,
            **shape,
        )
