# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from _torch.executor.multimodal_utils import make_llm_request, make_mm_request

from tensorrt_llm._torch.peft.lora.adapter_slot_manager import AdapterSlotManager
from tensorrt_llm._torch.peft.lora.cuda_graph_lora_manager import CudaGraphLoraManager
from tensorrt_llm._torch.peft.lora.cuda_graph_lora_params import (
    CudaGraphLoraParams,
    LoraLayerParams,
)
from tensorrt_llm._torch.pyexecutor.engine.lora import LoraParamBuilder
from tensorrt_llm._torch.pyexecutor.resource_manager import PeftCacheManager
from tensorrt_llm._torch.pyexecutor.scheduler import MultimodalScheduler
from tensorrt_llm._torch.tensor_lru_cache import TensorLRUCache
from tensorrt_llm.bindings import DataType

pytestmark = pytest.mark.cpu_only

_QK_KEY = CudaGraphLoraParams.LoraLayerKey(7, (1, 2))
_OUTPUT_KEY = CudaGraphLoraParams.LoraLayerKey(8, (4,))


@pytest.fixture
def graph_params() -> CudaGraphLoraParams:
    """Use CPU buffers for updates whose constructor normally allocates CUDA tables."""
    params = object.__new__(CudaGraphLoraParams)
    params.max_lora_size = 2
    params.max_rank = 8
    params.component_module_ids = frozenset((26, 27))
    params.layer_info = {
        key: CudaGraphLoraParams.LoraLayerInfo(
            module_num=len(key.module_ids), output_sizes=[8] * len(key.module_ids)
        )
        for key in (_QK_KEY, _OUTPUT_KEY)
    }
    params.layer_module2key = params._calculate_layer_module2key()
    params.slot_ranks_host = torch.zeros(2, dtype=torch.int32)
    params.slot_ranks = torch.zeros_like(params.slot_ranks_host)
    params.layer_params = {}
    for key in params.layer_info:
        shape = (len(key.module_ids), params.max_lora_size)
        sizes = torch.full((len(key.module_ids),), 8, dtype=torch.int32)
        offsets = torch.arange(len(key.module_ids), dtype=torch.int64) * 8
        params.layer_params[key] = LoraLayerParams(
            d_b_ptrs=torch.zeros(shape, dtype=torch.int64),
            d_b_prime_ptrs=torch.zeros(shape, dtype=torch.int64),
            h_b_ptrs=torch.zeros(shape, dtype=torch.int64),
            h_b_prime_ptrs=torch.zeros(shape, dtype=torch.int64),
            d_output_sizes=sizes.clone(),
            d_output_sizes_offset=offsets.clone(),
            h_output_sizes=sizes,
            h_output_sizes_offset=offsets,
        )
    params._moe_slot_cache = {}
    return params


def _entry(layer: int, module: int, rank: int, pointer: int) -> SimpleNamespace:
    return SimpleNamespace(
        layer_id=layer,
        module_id=module,
        adapter_size=rank,
        weights_in_pointer=pointer,
        weights_out_pointer=pointer + 100,
        scaling_vec_pointer=None,
    )


def _decoder_entries(rank: int, pointer: int) -> list[SimpleNamespace]:
    return [
        _entry(7, 1, rank, pointer),
        _entry(7, 2, rank, pointer + 1),
        _entry(8, 4, rank, pointer + 2),
    ]


def _pointer_buffers(params: CudaGraphLoraParams) -> list[torch.Tensor]:
    return [
        buffer
        for layer in params.layer_params.values()
        for buffer in (layer.h_b_ptrs, layer.h_b_prime_ptrs, layer.d_b_ptrs, layer.d_b_prime_ptrs)
    ]


def test_composite_tasks_route_only_decoder_weights(graph_params: CudaGraphLoraParams) -> None:
    graph_params.update_weight_pointers(
        {
            10: [
                _entry(7, 26, 4, 900),
                *_decoder_entries(4, 101),
                _entry(0, 27, 4, 901),
            ],
            11: _decoder_entries(2, 301),
        },
        (10, 11),
    )

    assert graph_params.slot_ranks_host.tolist() == [4, 2]
    assert graph_params.slot_ranks.tolist() == [4, 2]
    for key, expected in (
        (_QK_KEY, [[101, 301], [102, 302]]),
        (_OUTPUT_KEY, [[103, 303]]),
    ):
        layer = graph_params.layer_params[key]
        expected_in = torch.tensor(expected, dtype=torch.int64)
        for buffer in (layer.h_b_ptrs, layer.d_b_ptrs):
            torch.testing.assert_close(buffer, expected_in)
        for buffer in (layer.h_b_prime_ptrs, layer.d_b_prime_ptrs):
            torch.testing.assert_close(buffer, expected_in + 100)


def test_component_only_slot_reuse_clears_decoder_ranks_and_all_pointers(
    graph_params: CudaGraphLoraParams,
) -> None:
    graph_params.update_weight_pointers(
        {10: _decoder_entries(4, 101), 11: _decoder_entries(2, 301)}, (10, 11)
    )
    buffers = _pointer_buffers(graph_params)
    addresses = [buffer.data_ptr() for buffer in buffers]
    rank_addresses = (graph_params.slot_ranks_host.data_ptr(), graph_params.slot_ranks.data_ptr())
    other_slot = [buffer[:, 1].clone() for buffer in buffers]
    assert all(torch.count_nonzero(buffer[:, 0]) == buffer.shape[0] for buffer in buffers)

    graph_params.update_weight_pointers(
        {20: [_entry(7, 26, 4, 900), _entry(0, 27, 4, 901)]}, (20, 11)
    )

    assert graph_params.slot_ranks_host.tolist() == [0, 2]
    assert graph_params.slot_ranks.tolist() == [0, 2]
    for buffer, retained in zip(buffers, other_slot, strict=True):
        assert torch.count_nonzero(buffer[:, 0]) == 0
        torch.testing.assert_close(buffer[:, 1], retained)
    assert [buffer.data_ptr() for buffer in _pointer_buffers(graph_params)] == addresses
    assert (
        graph_params.slot_ranks_host.data_ptr(),
        graph_params.slot_ranks.data_ptr(),
    ) == rank_addresses


@pytest.mark.parametrize("module_id", [3, 28])
def test_unknown_module_ids_are_not_silently_filtered(
    graph_params: CudaGraphLoraParams, module_id: int
) -> None:
    with pytest.raises(KeyError) as error:
        graph_params.update_weight_pointers(
            {10: [_entry(7, 26, 4, 900), _entry(7, module_id, 4, 101)]}, (10, None)
        )

    assert error.value.args == ((7, module_id),)


def test_default_decoder_scope_does_not_ignore_component_sized_ids(
    graph_params: CudaGraphLoraParams,
) -> None:
    graph_params.component_module_ids = frozenset()

    with pytest.raises(KeyError) as error:
        graph_params.update_weight_pointers({10: [_entry(7, 26, 4, 900)]}, (10, None))

    assert error.value.args == ((7, 26),)


@pytest.fixture
def graph_manager(graph_params: CudaGraphLoraParams) -> CudaGraphLoraManager:
    graph_params.max_batch_size = 2
    graph_params.persistent_sorted_ids = torch.zeros(2, dtype=torch.int32)
    graph_params.sorted_ids_host = torch.zeros(2, dtype=torch.int32)
    graph_params.token_to_slot_host = torch.zeros(2, dtype=torch.int32)
    graph_params.slot_counts_host = torch.zeros(2, dtype=torch.int32)
    graph_params.slot_counts = torch.zeros(2, dtype=torch.int32)
    graph_params.slot_offsets_full_host = torch.zeros(3, dtype=torch.int64)
    graph_params.slot_offsets_full = torch.zeros(3, dtype=torch.int64)
    manager = object.__new__(CudaGraphLoraManager)
    manager.adapter_slot_manager = AdapterSlotManager(2)
    manager.cuda_graph_lora_params = graph_params
    manager.lora_aux_stream = None
    return manager


def _cache(resident: set[int]) -> PeftCacheManager:
    cache = object.__new__(PeftCacheManager)
    cache.impl = SimpleNamespace(
        is_task_cached_device=lambda task: task in resident, data_type=DataType.HALF
    )
    return cache


def _batch(tasks: list[int], *, context: bool = False) -> tuple[SimpleNamespace, SimpleNamespace]:
    requests = [
        SimpleNamespace(lora_task_id=task, py_request_id=index) for index, task in enumerate(tasks)
    ]
    scheduled = SimpleNamespace(
        num_context_requests=len(requests) if context else 0,
        generation_requests=[] if context else requests,
        all_requests=lambda: requests,
    )
    metadata = SimpleNamespace(
        host_request_types=torch.zeros(len(requests), dtype=torch.int32),
        prompt_lens_cpu=torch.ones(len(requests), dtype=torch.int32),
        num_seqs=len(requests),
        num_contexts=len(requests) if context else 0,
        num_generations=0 if context else len(requests),
    )
    return scheduled, metadata


@pytest.mark.parametrize(
    "observe_eviction", [False, True], ids=["encoder-only-gap", "observed-eviction"]
)
def test_same_task_reload_refreshes_graph_pointers_after_eager_prefill(
    graph_manager: CudaGraphLoraManager, observe_eviction: bool
) -> None:
    resident = {10}
    cache = _cache(resident)
    cache._batch_peft_table = {10: _decoder_entries(4, 101)}
    graph_manager.prepare_cuda_graph_lora_params(*_batch([10]), cache)
    params = graph_manager.cuda_graph_lora_params
    buffers = _pointer_buffers(params)
    addresses = [buffer.data_ptr() for buffer in buffers]
    assert params.layer_params[_QK_KEY].h_b_ptrs[:, 0].tolist() == [101, 102]
    assert not graph_manager.adapter_slot_manager.has_slots_changed()

    resident.remove(10)
    if observe_eviction:
        cache._batch_peft_table = {}
        graph_manager.prepare_base_only_batch(cache)
    resident.add(10)
    cache._batch_peft_table = {10: _decoder_entries(4, 201)}
    builder = LoraParamBuilder(attn_backend=object, cuda_graph_manager=graph_manager)
    builder.build(*_batch([10], context=True), peft_cache_manager=cache, maybe_graph=False)
    cache._batch_peft_table = {10: _decoder_entries(4, 201)}
    graph_manager.prepare_cuda_graph_lora_params(*_batch([10]), cache)

    assert graph_manager.adapter_slot_manager.get_slot_to_task_mapping() == (10, None)
    for key, expected in ((_QK_KEY, [201, 202]), (_OUTPUT_KEY, [203])):
        layer = params.layer_params[key]
        for buffer in (layer.h_b_ptrs, layer.d_b_ptrs):
            assert buffer[:, 0].tolist() == expected
        for buffer in (layer.h_b_prime_ptrs, layer.d_b_prime_ptrs):
            assert buffer[:, 0].tolist() == [pointer + 100 for pointer in expected]
    assert [buffer.data_ptr() for buffer in _pointer_buffers(params)] == addresses


def test_fresh_peft_table_retains_untouched_occupied_slot(
    graph_manager: CudaGraphLoraManager,
) -> None:
    cache = _cache({10, 11})
    cache._batch_peft_table = {10: _decoder_entries(4, 101), 11: _decoder_entries(2, 301)}
    graph_manager.prepare_cuda_graph_lora_params(*_batch([10, 11]), cache)
    params = graph_manager.cuda_graph_lora_params
    retained = [buffer[:, 1].clone() for buffer in _pointer_buffers(params)]

    cache._batch_peft_table = {10: _decoder_entries(4, 201)}
    graph_manager.prepare_cuda_graph_lora_params(*_batch([10]), cache)

    assert params.layer_params[_QK_KEY].h_b_ptrs[:, 0].tolist() == [201, 202]
    assert params.slot_ranks_host.tolist() == [4, 2]
    for buffer, expected in zip(_pointer_buffers(params), retained, strict=True):
        torch.testing.assert_close(buffer[:, 1], expected)


def test_missing_peft_table_clears_evicted_slot_and_retains_occupied_slot(
    graph_manager: CudaGraphLoraManager,
) -> None:
    resident = {10, 11}
    cache = _cache(resident)
    cache._batch_peft_table = {10: _decoder_entries(4, 101), 11: _decoder_entries(2, 301)}
    graph_manager.prepare_cuda_graph_lora_params(*_batch([10, 11]), cache)
    params = graph_manager.cuda_graph_lora_params
    retained = [buffer[:, 1].clone() for buffer in _pointer_buffers(params)]

    resident.remove(10)
    cache._batch_peft_table = None
    graph_manager.prepare_cuda_graph_lora_params(*_batch([11]), cache)

    assert params.slot_ranks_host.tolist() == [0, 2]
    for buffer, expected in zip(_pointer_buffers(params), retained, strict=True):
        assert torch.count_nonzero(buffer[:, 0]) == 0
        torch.testing.assert_close(buffer[:, 1], expected)


def test_default_scheduler_can_skip_decoder_before_admitting_a_ready_request() -> None:
    base_scheduler = SimpleNamespace(
        capacity_scheduler=SimpleNamespace(
            schedule_request=lambda requests: (list(requests), [], [])
        ),
        micro_batch_scheduler=SimpleNamespace(
            schedule=lambda requests, _: ([], list(requests), [])
        ),
    )
    scheduler = MultimodalScheduler(
        base_scheduler,
        max_batch_size=1,
        max_num_tokens=1,
        encoder_cache=TensorLRUCache(1024),
        get_item_cache_keys=lambda request: [
            (request.request_id, item) for item in range(request.py_mm_encoder_state.num_items)
        ],
        bytes_per_encoder_embedding=4,
        retain_cache_entries=False,
    )
    partial = make_mm_request(30, [1, 1, 1])
    first = scheduler.schedule_request([partial], set())
    assert first.context_requests == []
    assert first.generation_requests == []
    assert first.scheduled_mm_encoder_items == {30: [0]}

    state = partial.py_mm_encoder_state
    key = state.item_cache_keys[0]
    scheduler.encoder_cache.commit(key, torch.zeros(1, 1))
    state.mark_cache_key_ready(key)
    ready = make_llm_request(31)
    second = scheduler.schedule_request([partial, ready], set())
    assert second.context_requests == [ready]
    assert second.scheduled_mm_encoder_items == {30: [1]}
