# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import pickle
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import cloudpickle
import pytest
import torch
from cuda.bindings import runtime as cudart
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from safetensors.torch import save_file

import tensorrt_llm.bindings as bindings
from tensorrt_llm._torch.distributed import AllReduceStrategy
from tensorrt_llm._torch.models.modeling_nemotron_h_multimodal import _NemotronVisionProjection
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
from tensorrt_llm._torch.peft.lora.adapter_revision import AdapterRevisionRegistry
from tensorrt_llm._torch.peft.lora.component_runtime import (
    ComponentLoraRuntime,
    bind_component_lora,
)
from tensorrt_llm._torch.peft.lora.config import LoraConfig
from tensorrt_llm._torch.peft.lora.layer import LoraLayer, LoraModuleType
from tensorrt_llm._torch.peft.lora.manager import LoraManager, LoraModelConfig
from tensorrt_llm._torch.peft.lora.ownership import ComponentLoraOwner, ComponentLoraTarget
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import PeftCacheManager
from tensorrt_llm._utils import mpi_comm, mpi_rank, mpi_world_size
from tensorrt_llm.llmapi.llm_args import PeftCacheConfig
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

pytestmark = pytest.mark.threadleak(enabled=False)

_TP = 4
_HIDDEN = 1280
_RANK = 4
_LAYER = 7


def _model(case: str, rank: int) -> tuple[torch.nn.Module, tuple[ComponentLoraOwner, ...]]:
    model = torch.nn.Module()
    model.vision = torch.nn.Module()
    local = _HIDDEN // _TP
    targets = []
    if case in ("split", "partial"):
        model.vision.projection = LoraLayer(
            [LoraModuleType.ATTENTION_Q, LoraModuleType.ATTENTION_K, LoraModuleType.ATTENTION_V],
            [local] * 3,
        )
        specs = [(f"attn_{name}", _HIDDEN, ()) for name in ("q", "k", "v")]
    elif case == "packed":
        model.vision.projection = LoraLayer([LoraModuleType.ATTENTION_QKV], [local * 3])
        specs = [("attn_qkv", _HIDDEN * 3, (_HIDDEN,) * 3)]
    elif case == "out":
        model.vision.projection = LoraLayer([LoraModuleType.ATTENTION_DENSE], [_HIDDEN])
        model.vision.linear = Linear(
            _HIDDEN,
            _HIDDEN,
            bias=False,
            dtype=torch.bfloat16,
            mapping=Mapping(world_size=_TP, rank=rank, tp_size=_TP, gpus_per_node=_TP),
            tensor_parallel_mode=TensorParallelMode.ROW,
            lora=model.vision.projection,
            allreduce_strategy=AllReduceStrategy.NCCL,
        ).cuda()
        model.vision.linear.weight.data.zero_()
        specs = [("attn_dense", _HIDDEN, ())]
    else:
        model.connector = _NemotronVisionProjection(
            torch.nn.Sequential(
                torch.nn.Identity(),
                torch.nn.Linear(128, 512, bias=False),
                torch.nn.GELU(),
                torch.nn.Linear(512, 256, bias=False),
            )
        ).to(device="cuda", dtype=torch.bfloat16)
        for parameter in model.connector.parameters():
            parameter.data.zero_()
        for index, module_type, input_size, output_size in (
            (1, "mlp_h_to_4h", 128, 512),
            (2, "mlp_4h_to_h", 512, 256),
        ):
            targets.append(
                ComponentLoraTarget(
                    f"linear{index}_lora",
                    (f"connector.linear{index}",),
                    0,
                    module_type,
                    -1,
                    input_size,
                    output_size,
                )
            )
        return model, bind_component_lora(
            model,
            (ComponentLoraOwner("vision_connector", "connector", tuple(targets), "bfloat16", 4),),
            tp_size=_TP,
        )

    for module_type, output_size, fused_sizes in specs:
        targets.append(
            ComponentLoraTarget(
                "projection",
                (f"vision.layers.{_LAYER}.{module_type}",),
                _LAYER,
                module_type,
                -1,
                _HIDDEN,
                output_size,
                fused_output_sizes=fused_sizes,
                tp_size=_TP,
                in_tp_split_dim=1 if case == "out" else -1,
                out_tp_split_dim=-1 if case == "out" else 0,
            )
        )
    return model, bind_component_lora(
        model,
        (ComponentLoraOwner("vision", "vision", tuple(targets), "bfloat16", 4),),
        tp_size=_TP,
    )


def _weights(target: ComponentLoraTarget, adapter: int) -> tuple[torch.Tensor, torch.Tensor]:
    # Sparse dyadic values keep both GEMMs and the four-rank sum exactly representable.
    phase = LoraManager.LORA_MODULE_IDS[target.module_type] % 4
    a = torch.zeros(_RANK, target.input_size, dtype=torch.bfloat16)
    for index in range(_RANK):
        a[index, index * (target.input_size // _RANK) + index + phase] = 1 / 8
    rows = torch.arange(target.output_size)
    b = torch.zeros(target.output_size, _RANK, dtype=torch.bfloat16)
    b[rows, rows % _RANK] = (rows // 64 + phase + 1).to(torch.bfloat16) / 16
    if adapter == 1:
        b *= -2
    return a, b


def _checkpoint(
    directory: Path, owner: ComponentLoraOwner, adapter: int, case: str
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    directory.mkdir()
    tensors = {}
    raw = {}
    for target in owner.targets:
        if case == "partial" and target.module_type != ("attn_q" if adapter == 0 else "attn_v"):
            continue
        a, b = _weights(target, adapter)
        prefix = f"base_model.model.{target.checkpoint_modules[0]}"
        tensors[f"{prefix}.lora_A.weight"] = a
        tensors[f"{prefix}.lora_B.weight"] = b
        raw[target.execution_id] = (a, b)
    save_file(tensors, str(directory / "adapter_model.safetensors"))
    (directory / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": _RANK, "lora_alpha": 8, "target_modules": []})
    )
    return raw


def _cache(owners: tuple[ComponentLoraOwner, ...], rank: int) -> PeftCacheManager:
    config = bindings.ModelConfig(
        vocab_size=32,
        num_layers=8,
        num_attention_layers=8,
        num_rnn_layers=0,
        num_heads=4,
        hidden_size=_HIDDEN // _TP,
        data_type=bindings.DataType.BF16,
    )
    config.set_num_lora_layers(8)
    config.use_lora_plugin = True
    config.max_lora_rank = _RANK
    config.lora_modules = [
        bindings.LoraModule(
            bindings.LoraModuleType(LoraManager.LORA_MODULE_IDS[target.module_type]),
            target.input_size,
            target.output_size,
            False,
            True,
            target.in_tp_split_dim,
            target.out_tp_split_dim,
            execution_id=target.execution_id,
        )
        for owner in owners
        for target in owner.targets
    ]
    return PeftCacheManager(
        peft_cache_config=PeftCacheConfig(
            num_host_module_layer=512,
            num_device_module_layer=256,
            optimal_adapter_size=_RANK,
            max_adapter_size=_RANK,
            num_put_workers=1,
            num_ensure_workers=1,
        ),
        lora_config=LoraConfig(enable_multimodal_lora=True, max_lora_rank=_RANK),
        model_config=config,
        lora_model_config=LoraModelConfig([], {}, _HIDDEN, "bfloat16", components=owners),
        world_config=bindings.WorldConfig(tensor_parallelism=_TP, rank=rank, gpus_per_node=_TP),
        initial_data_type=torch.bfloat16,
    )


def _read_pointer(pointer: int, shape: tuple[int, ...]) -> torch.Tensor:
    result = torch.empty(shape, dtype=torch.bfloat16)
    (status,) = cudart.cudaMemcpy(
        result.data_ptr(),
        pointer,
        result.numel() * result.element_size(),
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
    )
    assert status == cudart.cudaError_t.cudaSuccess
    return result


def _local_weights(
    target: ComponentLoraTarget, raw: tuple[torch.Tensor, torch.Tensor], rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    a, b = raw
    if target.in_tp_split_dim == 1:
        a = a.chunk(_TP, dim=1)[rank]
    if target.out_tp_split_dim == 0:
        parts = b.split(target.fused_output_sizes or (target.output_size,), dim=0)
        b = torch.cat([part.chunk(_TP, dim=0)[rank] for part in parts], dim=0)
    return a.contiguous(), (b * 2).contiguous()


def _delta(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    middle = (x.float() @ a.cuda().float().T).to(torch.bfloat16)
    return (middle.float() @ b.cuda().float().T).to(torch.bfloat16)


@torch.inference_mode()
def _run_rank(case: str) -> bool:
    assert mpi_world_size() == _TP
    rank = mpi_rank()
    torch.cuda.set_device(rank)
    model, owners = _model(case, rank)
    owner = owners[0]
    cache = _cache(owners, rank)
    runtime = ComponentLoraRuntime(owners)
    with tempfile.TemporaryDirectory() as directory, AdapterRevisionRegistry() as registry:
        paths = [Path(directory) / str(index) for index in range(2)]
        raw = [_checkpoint(path, owner, index, case) for index, path in enumerate(paths)]
        snapshots = [registry.resolve(index + 1, str(path)) for index, path in enumerate(paths)]
        requests = [
            LlmRequest(
                request_id=index + 1,
                max_new_tokens=1,
                input_tokens=[1],
                sampling_config=bindings.SamplingConfig(1),
                is_streaming=False,
                lora_task_id=snapshot.task_id,
                py_lora_path=snapshot.path,
            )
            for index, snapshot in enumerate(snapshots)
        ]
        table = cache.prepare_component_peft(requests)
        torch.cuda.synchronize()
        runtime.bind_batch(table, torch.bfloat16)
        for snapshot, matrices in zip(snapshots, raw, strict=True):
            entries = {entry.module_id: entry for entry in table[snapshot.task_id]}
            assert entries.keys() == matrices.keys()
            for target in owner.targets:
                if target.execution_id not in matrices:
                    continue
                a, b = _local_weights(target, matrices[target.execution_id], rank)
                entry = entries[target.execution_id]
                assert entry.in_size == a.numel() and entry.out_size == b.numel()
                torch.testing.assert_close(
                    _read_pointer(entry.weights_in_pointer, tuple(a.shape)), a, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    _read_pointer(entry.weights_out_pointer, tuple(b.shape)), b, rtol=0, atol=0
                )

        for order in ((None, 0, 1), (1, None, 0)):
            counts = (1, 2, 3)
            segments = [
                (None if index is None else snapshots[index].task_id, count)
                for index, count in zip(order, counts, strict=True)
            ]
            identity = (
                tuple(target.execution_id for target in owner.targets),
                tuple((snapshot.task_id, snapshot.revision) for snapshot in snapshots),
                segments,
            )
            assert all(value == identity for value in mpi_comm().allgather(identity))
            params = runtime.parameters(owner.name, segments)
            input_size = 128 if case == "projector" else _HIDDEN
            x = (torch.arange(sum(counts) * input_size).reshape(-1, input_size) % 5).cuda()
            x = x.to(torch.bfloat16) / 8
            if case == "out":
                linear = model.vision.linear
                with patch.object(
                    linear.all_reduce, "forward", wraps=linear.all_reduce.forward
                ) as reduce:
                    actual = linear(
                        x.chunk(_TP, dim=1)[rank].contiguous(), lora_params=params, layer_idx=_LAYER
                    )
                assert reduce.call_count == 1
            elif case == "projector":
                actual = model.connector(x, lora_params=params)
            else:
                actual = model.vision.projection(x, params, _LAYER)
            expected = torch.zeros_like(actual)
            start = 0
            for index, count in zip(order, counts, strict=True):
                rows = slice(start, start + count)
                start += count
                if index is None:
                    continue
                if case == "projector":
                    first, second = owner.targets
                    a, b = raw[index][first.execution_id]
                    hidden = torch.nn.functional.gelu(_delta(x[rows], a, b * 2))
                    a, b = raw[index][second.execution_id]
                    expected[rows] = _delta(hidden, a, b * 2)
                elif case == "out":
                    target = owner.targets[0]
                    a, b = raw[index][target.execution_id]
                    expected[rows] = _delta(x[rows], a, b * 2)
                else:
                    offset = 0
                    for target in owner.targets:
                        width = target.output_size // _TP
                        if target.execution_id in raw[index]:
                            a, b = _local_weights(target, raw[index][target.execution_id], rank)
                            expected[rows, offset : offset + width] = _delta(x[rows], a, b)
                        offset += width
            assert expected.count_nonzero() > 0
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if case in ("out", "projector"):
                outputs = mpi_comm().allgather(actual.cpu())
                assert all(torch.equal(value, outputs[0]) for value in outputs)
        for request in requests:
            cache.free_resources(request)
        torch.cuda.synchronize()
    return True


def _init_worker(paths: list[str]) -> None:
    sys.path[:] = paths


@pytest.mark.parametrize("case", ["split", "partial", "packed", "out", "projector"])
def test_component_lora_tp4_native_cache(case: str) -> None:
    """Native cache shards and projection collectives preserve each request's adapter."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < _TP:
        pytest.skip("Requires four local CUDA GPUs")
    if mpi_world_size() > 1:
        assert _run_rank(case)
        return
    with MPIPoolExecutor(
        max_workers=_TP, initializer=_init_worker, initargs=(sys.path,)
    ) as executor:
        assert all(executor.map(_run_rank, [case] * _TP))
