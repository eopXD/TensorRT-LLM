# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import load_file, save_file

from tensorrt_llm._torch.peft.lora.component_runtime import (
    ComponentLoraRuntime,
    bind_component_lora,
)
from tensorrt_llm._torch.peft.lora.config import LoraConfig
from tensorrt_llm._torch.peft.lora.layer import LoraLayer, LoraModuleType
from tensorrt_llm._torch.peft.lora.manager import LoraManager, LoraModelConfig, load_torch_lora
from tensorrt_llm._torch.peft.lora.ownership import (
    ComponentLoraOwner,
    ComponentLoraTarget,
    DecoderLoraOwner,
)
from tensorrt_llm.executor.base_worker import BaseWorker
from tensorrt_llm.executor.request import LoRARequest
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def components() -> tuple[ComponentLoraOwner, ...]:
    declarations = [
        ("vision", 7, 64, 6, 8, "vision.encoder.layers.7.self_attn.o_proj"),
        ("audio", 7, 65, 8, 6, "audio.encoder.layers.7.self_attn.o_proj"),
        ("vision_connector", 0, 66, 8, 4, "vision_connector.linear"),
        ("audio_connector", 0, 67, 6, 4, "audio_connector.linear"),
    ]
    return tuple(
        ComponentLoraOwner(
            name=name,
            module_path=name,
            dtype="float16",
            max_rank=4,
            hf_prefixes=(name,),
            targets=(
                ComponentLoraTarget(
                    module_path="projection",
                    checkpoint_modules=(checkpoint_module,),
                    layer_idx=layer,
                    module_type="attn_dense",
                    execution_id=execution_id,
                    input_size=input_size,
                    output_size=output_size,
                ),
            ),
        )
        for name, layer, execution_id, input_size, output_size, checkpoint_module in declarations
    )


@pytest.fixture
def model_config(components: tuple[ComponentLoraOwner, ...]) -> LoraModelConfig:
    return LoraModelConfig(
        lora_target_modules=["attn_dense"],
        trtllm_modules_to_hf_modules={"attn_dense": "o_proj"},
        hidden_size=4,
        dtype="float16",
        owner=DecoderLoraOwner(module_path="language_model"),
        components=components,
    )


@pytest.fixture
def cpu_transfers(monkeypatch: pytest.MonkeyPatch) -> Mock:
    transfer = Mock(side_effect=lambda tensor: tensor)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor, *args, **kwargs: transfer(tensor))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (8, 0))
    return transfer


def _weights(
    model_config: LoraModelConfig, names: tuple[str, ...], *, marker: float = 1.0
) -> dict[str, torch.Tensor]:
    shapes = {"decoder": ("model.layers.7.self_attn.o_proj", 4, 4)}
    for component in model_config.components:
        target = component.targets[0]
        shapes[component.name] = (
            target.checkpoint_modules[0],
            target.input_size,
            target.output_size,
        )
    weights = {}
    for index, name in enumerate(names):
        module, input_size, output_size = shapes[name]
        prefix = f"base_model.model.{module}"
        weights[f"{prefix}.lora_A.weight"] = torch.full(
            (2, input_size), marker + index, dtype=torch.float16
        )
        weights[f"{prefix}.lora_B.weight"] = torch.full(
            (output_size, 2), marker + index + 0.5, dtype=torch.float16
        )
    return weights


def _write_adapter(
    path: Path,
    weights: dict[str, torch.Tensor],
    *,
    use_bin: bool = False,
    extra_config: dict | None = None,
) -> Path:
    path.mkdir()
    if use_bin:
        torch.save(weights, path / "adapter_model.bin")
    else:
        save_file(weights, path / "adapter_model.safetensors")
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 2,
                "lora_alpha": 2,
                "target_modules": ["o_proj", "linear"],
                **(extra_config or {}),
            }
        )
    )
    return path


def _manager(model_config: LoraModelConfig) -> LoraManager:
    return LoraManager(mapping=Mapping(), model_config=model_config, cpp_peft_cache_manager=Mock())


@pytest.mark.parametrize(
    "names",
    [
        ("decoder",),
        ("vision",),
        ("audio",),
        ("vision_connector",),
        ("audio_connector",),
        ("decoder", "vision", "audio", "vision_connector", "audio_connector"),
    ],
)
def test_component_subsets_preserve_distinct_execution_ids_dimensions_and_weights(
    model_config: LoraModelConfig, cpu_transfers: Mock, names: tuple[str, ...], tmp_path: Path
) -> None:
    weights = _weights(model_config, names)
    path = _write_adapter(tmp_path / "adapter", weights)
    config = LoraConfig(
        lora_dir=[str(path)],
        enable_multimodal_lora=True,
        lora_target_modules=["attn_dense"],
        trtllm_modules_to_hf_modules={"attn_dense": "o_proj"},
    )
    load_torch_lora(config, owner=model_config.owner, components=model_config.components)
    manager = _manager(model_config)

    assert manager.load_from_ckpt([str(path)], model_config, uids=["7"]) == ["7"]

    entries = manager.cpp_lora_config["7"].tolist()
    assert len(entries) == len(names)
    actual = {
        (entry[0], entry[1]): packed
        for entry, packed in zip(entries, manager.cpp_lora_weights["7"])
    }
    expected_keys = set()
    for name in names:
        if name == "decoder":
            identity, module = (4, 7), "model.layers.7.self_attn.o_proj"
        else:
            target = next(owner for owner in model_config.components if owner.name == name).targets[
                0
            ]
            identity, module = (target.execution_id, target.layer_idx), target.checkpoint_modules[0]
        expected_keys.add(identity)
        prefix = f"base_model.model.{module}"
        expected = torch.cat(
            [weights[f"{prefix}.lora_{direction}.weight"].flatten() for direction in ("A", "B")]
        )
        torch.testing.assert_close(actual[identity][: expected.numel()], expected)
        assert torch.count_nonzero(actual[identity][expected.numel() :]) == 0
    assert set(actual) == expected_keys
    assert all(entry[2:] == [2, 0] for entry in entries)
    assert set(manager.component_fingerprints["7"]) == set(names) - {"decoder"}
    assert cpu_transfers.called


def test_failed_component_load_keeps_cache_and_fingerprints_atomic_and_uid_reusable(
    model_config: LoraModelConfig, cpu_transfers: Mock, tmp_path: Path
) -> None:
    manager = _manager(model_config)
    names = ("decoder", "vision", "audio", "vision_connector")
    weights = _weights(model_config, names)
    good = _write_adapter(tmp_path / "good", weights)
    manager.load_from_ckpt([str(good)], model_config, uids=["7"])
    old_weights = manager.cpp_lora_weights["7"].clone()
    old_config = manager.cpp_lora_config["7"].clone()
    old_fingerprints = dict(manager.component_fingerprints["7"])
    bad_key = "base_model.model.audio.encoder.layers.7.self_attn.o_proj.lora_A.weight"
    invalid = dict(weights)
    invalid[bad_key] = torch.ones(2, 9, dtype=torch.float16)
    bad = _write_adapter(tmp_path / "bad", invalid)
    cpu_transfers.reset_mock()

    with pytest.raises(ValueError) as error:
        manager.load_from_ckpt([str(bad)], model_config, uids=["8"])

    assert "audio" in str(error.value)
    for table in (
        manager.cpp_lora_weights,
        manager.cpp_lora_config,
        manager._lora_uid_to_low_ranks,
        manager._lora_weights_pointers_list,
        manager.component_fingerprints,
    ):
        assert "8" not in table
    torch.testing.assert_close(manager.cpp_lora_weights["7"], old_weights)
    torch.testing.assert_close(manager.cpp_lora_config["7"], old_config)
    assert manager.component_fingerprints["7"] == old_fingerprints
    cpu_transfers.assert_not_called()
    assert manager.load_from_ckpt([str(good)], model_config, uids=["8"]) == ["8"]
    assert manager.component_fingerprints["8"] == old_fingerprints


def test_component_fingerprint_ignores_decoder_weights_and_tracks_component_contents(
    model_config: LoraModelConfig, cpu_transfers: Mock, tmp_path: Path
) -> None:
    manager = _manager(model_config)
    original = _weights(model_config, ("decoder", "vision"))
    variants = [original, dict(original), dict(original)]
    variants[1]["base_model.model.model.layers.7.self_attn.o_proj.lora_B.weight"] = torch.full(
        (4, 2), 7.0, dtype=torch.float16
    )
    variants[2]["base_model.model.vision.encoder.layers.7.self_attn.o_proj.lora_B.weight"] = (
        torch.full((8, 2), 9.0, dtype=torch.float16)
    )
    for uid, weights in enumerate(variants):
        path = _write_adapter(tmp_path / str(uid), weights)
        manager.load_from_ckpt([str(path)], model_config, uids=[str(uid)])

    assert manager.component_fingerprints["0"] == manager.component_fingerprints["1"]
    assert (
        manager.component_fingerprints["0"]["vision"]
        != manager.component_fingerprints["2"]["vision"]
    )


def test_component_runtime_keeps_request_ownership_and_base_padding(
    components: tuple[ComponentLoraOwner, ...],
) -> None:
    runtime = ComponentLoraRuntime(components)

    def module(execution_id: int, rank: int, pointer: int) -> SimpleNamespace:
        return SimpleNamespace(
            layer_id=7,
            module_id=execution_id,
            adapter_size=rank,
            weights_in_pointer=pointer,
            weights_out_pointer=pointer + 1,
            scaling_vec_pointer=0,
        )

    runtime.bind_batch(
        {7: [module(64, 2, 10), module(65, 2, 20)], 8: [module(64, 4, 30)]},
        torch.float16,
    )
    params = runtime.parameters("vision", [(8, 1), (None, 2), (7, 3)])
    assert params[7][64]["adapter_size"].tolist() == [4, 0, 2]
    assert params[7][64]["weight_pointers"].tolist() == [30, 31, 0, 0, 0, 0, 10, 11, 0]
    assert set(params[7]) == {64}
    assert params["host_request_types"].tolist() == [0, 0, 0]
    assert params["prompt_lens_cpu"].tolist() == [1, 2, 3]
    assert params["num_seqs"] == 3
    assert params["data_type"] is torch.float16
    audio = runtime.parameters("audio", [(7, 5), (8, 2)])
    assert audio[7][65]["adapter_size"].tolist() == [2, 0]
    assert audio["prompt_lens_cpu"].tolist() == [5, 2]
    assert runtime.parameters("vision_connector", [(7, 2)]) is None
    runtime.bind_batch({}, torch.float16)
    assert runtime.parameters("vision", [(None, 2)]) is None
    with pytest.raises(RuntimeError, match="was not prepared"):
        runtime.parameters("vision", [(7, 3), (None, 2)])


def _component_model(components: tuple[ComponentLoraOwner, ...]) -> torch.nn.Module:
    model = torch.nn.Module()
    for component in components:
        module = torch.nn.Module()
        module.projection = LoraLayer(
            [LoraModuleType.ATTENTION_DENSE], [component.targets[0].output_size]
        )
        model.add_module(component.name, module)
    return model


def test_binding_assigns_distinct_component_ids_to_same_local_projection(
    components: tuple[ComponentLoraOwner, ...],
) -> None:
    model = _component_model(components)

    bound = bind_component_lora(model, components)

    execution_ids = [owner.targets[0].execution_id for owner in bound]
    assert len(set(execution_ids)) == len(components)
    assert set(execution_ids).isdisjoint(LoraManager.LORA_MODULE_IDS.values())
    for declared, owner in zip(components, bound):
        target = owner.targets[0]
        assert (target.input_size, target.output_size) == (
            declared.targets[0].input_size,
            declared.targets[0].output_size,
        )
        assert model.get_submodule(owner.module_path).projection.execution_module_ids == (
            target.execution_id,
        )


def test_binding_rejects_shared_layer_claimed_by_distinct_components(
    components: tuple[ComponentLoraOwner, ...],
) -> None:
    components = (
        components[0],
        replace(
            components[1],
            targets=(replace(components[1].targets[0], output_size=8),),
        ),
        *components[2:],
    )
    model = _component_model(components)
    model.audio.projection = model.vision.projection

    with pytest.raises(ValueError, match="shared LoRA layer"):
        bind_component_lora(model, components)

    assert model.vision.projection.execution_module_ids is None


def test_binding_rejects_incomplete_fused_projection_declaration(
    components: tuple[ComponentLoraOwner, ...],
) -> None:
    model = _component_model(components)
    model.vision.projection = LoraLayer(
        [LoraModuleType.ATTENTION_Q, LoraModuleType.ATTENTION_K, LoraModuleType.ATTENTION_V],
        [8, 8, 8],
    )
    target = replace(components[0].targets[0], module_type="attn_q")
    owner = replace(components[0], targets=(target,))

    with pytest.raises(ValueError, match="Every projection"):
        bind_component_lora(model, (owner,))

    assert model.vision.projection.execution_module_ids is None


@pytest.mark.parametrize("active_projection", [0, 1], ids=["query-only", "key-only"])
def test_partial_qkv_passes_only_active_width_and_restores_other_outputs(
    monkeypatch: pytest.MonkeyPatch, active_projection: int
) -> None:
    output_sizes = [8, 4, 4]
    layer = LoraLayer(
        [LoraModuleType.ATTENTION_Q, LoraModuleType.ATTENTION_K, LoraModuleType.ATTENTION_V],
        output_sizes,
    )
    layer.execution_module_ids = (64, 65, 66)
    x = torch.arange(18, dtype=torch.float32).reshape(3, 6) / 20
    a = torch.arange(12, dtype=torch.float32).reshape(2, 6) / 20
    b = torch.arange(output_sizes[active_projection] * 2, dtype=torch.float32).reshape(-1, 2) / 20
    expected_delta = x @ a.T @ b.T
    kernel = Mock(return_value=expected_delta)
    monkeypatch.setattr(torch.ops.trtllm, "lora_grouped_gemm", kernel)
    params = {
        7: {
            layer.execution_module_ids[active_projection]: {
                "adapter_size": torch.tensor([2], dtype=torch.int32),
                "weight_pointers": torch.tensor([11, 22, 0], dtype=torch.int64),
            }
        },
        "num_seqs": 1,
        "host_request_types": torch.tensor([0], dtype=torch.int32),
        "prompt_lens_cpu": torch.tensor([3], dtype=torch.int32),
    }

    result = layer(x, params, layer_idx=7)

    kernel.assert_called_once()
    assert kernel.call_args.args[5] == [output_sizes[active_projection]]
    assert len(kernel.call_args.args[2]) == len(kernel.call_args.args[3]) == 1
    expected = torch.zeros(3, sum(output_sizes))
    offset = sum(output_sizes[:active_projection])
    expected[:, offset : offset + output_sizes[active_projection]] = expected_delta
    torch.testing.assert_close(result, expected)


def _assert_unpublished(manager: LoraManager, uid: str) -> None:
    for table in (
        manager.cpp_lora_weights,
        manager.cpp_lora_config,
        manager._lora_uid_to_low_ranks,
        manager._lora_weights_pointers_list,
        manager.component_fingerprints,
    ):
        assert uid not in table
    assert manager.lora_weights == []


@pytest.mark.parametrize(
    "failure", ["duplicate-alias", "undeclared-owner", "undeclared-projection", "disabled-owner"]
)
def test_invalid_component_destinations_publish_no_adapter_state(
    model_config: LoraModelConfig, cpu_transfers: Mock, tmp_path: Path, failure: str
) -> None:
    weights = _weights(model_config, ("vision",))
    if failure == "duplicate-alias":
        owner = model_config.components[0]
        target = owner.targets[0]
        alias = "vision.alias.layers.7.self_attn.o_proj"
        target = replace(target, checkpoint_modules=(*target.checkpoint_modules, alias))
        model_config = replace(
            model_config,
            components=(replace(owner, targets=(target,)), *model_config.components[1:]),
        )
        weights.update(
            {
                key.replace("vision.encoder.layers.7.self_attn.o_proj", alias): value.clone()
                for key, value in list(weights.items())
            }
        )
        match = "duplicate destination"
    elif failure in ("undeclared-owner", "undeclared-projection"):
        replacement = (
            "unregistered.encoder.layers.7.self_attn.o_proj"
            if failure == "undeclared-owner"
            else "vision.encoder.layers.7.self_attn.q_proj"
        )
        weights = {
            key.replace("vision.encoder.layers.7.self_attn.o_proj", replacement): value
            for key, value in weights.items()
        }
        match = "outside selected owner|Unsupported LoRA key"
    else:
        model_config = replace(model_config, components=())
        match = "outside selected owner"
    path = _write_adapter(tmp_path / "invalid", weights)
    manager = _manager(model_config)

    with pytest.raises(ValueError, match=match):
        manager.load_from_ckpt([str(path)], model_config, uids=["7"])

    _assert_unpublished(manager, "7")
    cpu_transfers.assert_not_called()


@pytest.mark.parametrize("packed_first", [False, True])
def test_component_packed_and_split_qkv_conflict_before_cache_publication(
    model_config: LoraModelConfig, cpu_transfers: Mock, tmp_path: Path, packed_first: bool
) -> None:
    owner = model_config.components[0]
    split = replace(
        owner.targets[0],
        module_path="attention.splitted_qkv_lora",
        module_type="attn_q",
        checkpoint_modules=("vision.layers.7.attn.q_proj",),
    )
    packed = replace(
        split,
        module_path="attention.fused_qkv_lora",
        module_type="attn_qkv",
        execution_id=68,
        checkpoint_modules=("vision.layers.7.attn.qkv_proj",),
        output_size=24,
        fused_output_sizes=(8, 8, 8),
    )
    model_config = replace(model_config, components=(replace(owner, targets=(split, packed)),))
    weights = {}
    for target in (packed, split) if packed_first else (split, packed):
        prefix = f"base_model.model.{target.checkpoint_modules[0]}"
        weights[f"{prefix}.lora_A.weight"] = torch.ones(2, 6, dtype=torch.float16)
        weights[f"{prefix}.lora_B.weight"] = torch.ones(target.output_size, 2, dtype=torch.float16)
    path = _write_adapter(tmp_path / "invalid", weights, use_bin=True)
    manager = _manager(model_config)

    with pytest.raises(ValueError, match="mix packed and split attention") as error:
        manager.load_from_ckpt([str(path)], model_config, uids=["7"])

    assert "q_proj" in str(error.value) and "qkv_proj" in str(error.value)
    _assert_unpublished(manager, "7")
    cpu_transfers.assert_not_called()


@pytest.mark.parametrize(
    "unsupported", [{"use_dora": True}, {"alpha_pattern": {"vision": 4}}, {"fan_in_fan_out": True}]
)
def test_unsupported_component_adapter_modes_fail_at_startup_and_load(
    model_config: LoraModelConfig, cpu_transfers: Mock, tmp_path: Path, unsupported: dict
) -> None:
    path = _write_adapter(
        tmp_path / "invalid", _weights(model_config, ("vision",)), extra_config=unsupported
    )
    config = LoraConfig(
        lora_dir=[str(path)], lora_target_modules=["attn_dense"], enable_multimodal_lora=True
    )
    with pytest.raises(ValueError, match="does not support"):
        load_torch_lora(config, owner=model_config.owner, components=model_config.components)
    manager = _manager(model_config)

    with pytest.raises(ValueError, match="does not support"):
        manager.load_from_ckpt([str(path)], model_config, uids=["7"])

    _assert_unpublished(manager, "7")
    cpu_transfers.assert_not_called()


def _worker(model_config: LoraModelConfig) -> BaseWorker:
    worker = object.__new__(BaseWorker)
    worker.doing_shutdown = True
    worker._runtime_model_config = None
    worker._lora_model_config = model_config
    worker._lora_manager = _manager(model_config)
    worker._component_lora_registry = None
    worker._component_lora_lock = threading.Lock()
    return worker


def test_worker_adapter_reload_preserves_prior_snapshot_and_component_fingerprints(
    model_config: LoraModelConfig, cpu_transfers: Mock, tmp_path: Path
) -> None:
    original = _weights(model_config, ("decoder", "vision"))
    path = _write_adapter(tmp_path / "adapter", original)
    request = LoRARequest("adapter", 7, str(path))
    worker = _worker(model_config)
    try:
        first, first_path = worker._load_component_lora_adapter(request)
        first_packed = worker._lora_manager.cpp_lora_weights[str(first.task_id)].clone()
        changed = _weights(model_config, ("decoder", "vision"), marker=9)
        save_file(changed, path / "adapter_model.safetensors")

        second, second_path = worker._load_component_lora_adapter(request)

        assert first.task_id != second.task_id
        assert first.revision != second.revision
        assert first_path != second_path
        assert first.component_fingerprints != second.component_fingerprints
        for key, tensor in load_file(str(Path(first_path) / "adapter_model.safetensors")).items():
            torch.testing.assert_close(tensor, original[key])
        torch.testing.assert_close(
            worker._lora_manager.cpp_lora_weights[str(first.task_id)], first_packed
        )
        assert (
            tuple(sorted(worker._lora_manager.component_fingerprints[str(first.task_id)].items()))
            == first.component_fingerprints
        )
        save_file(original, path / "adapter_model.safetensors")
        restored, restored_path = worker._load_component_lora_adapter(request)
        assert restored == first
        assert restored_path == first_path
    finally:
        worker._close_component_lora_registry()


def test_decoder_only_worker_keeps_external_adapter_id_behavior(
    model_config: LoraModelConfig, cpu_transfers: Mock, tmp_path: Path
) -> None:
    model_config = replace(model_config, components=())
    path = _write_adapter(tmp_path / "adapter", _weights(model_config, ("decoder",)))
    worker = _worker(model_config)
    request = LoRARequest("adapter", 7, str(path))
    assert worker._load_lora_adapter(request)
    original = worker._lora_manager.cpp_lora_weights["7"].clone()
    save_file(_weights(model_config, ("decoder",), marker=9), path / "adapter_model.safetensors")

    assert not worker._load_lora_adapter(request)

    assert set(worker._lora_manager.cpp_lora_weights) == {"7"}
    torch.testing.assert_close(worker._lora_manager.cpp_lora_weights["7"], original)
    assert worker._component_lora_registry is None


@pytest.mark.parametrize("include_vision", [False, True], ids=["decoder-only", "composite"])
def test_multimodal_policy_accepts_an_explicit_decoder_target_subset(
    model_config: LoraModelConfig, cpu_transfers: Mock, tmp_path: Path, include_vision: bool
) -> None:
    module_mapping = {
        **model_config.trtllm_modules_to_hf_modules,
        "mlp_h_to_4h": "gate_proj",
        "attn_qkv": "qkv_proj",
    }
    model_config = replace(model_config, trtllm_modules_to_hf_modules=module_mapping)
    names = ("decoder", "vision") if include_vision else ("decoder",)
    path = _write_adapter(tmp_path / "adapter", _weights(model_config, names))
    config = LoraConfig(
        lora_dir=[str(path)],
        enable_multimodal_lora=True,
        lora_target_modules=["attn_dense"],
        trtllm_modules_to_hf_modules=module_mapping,
    )

    load_torch_lora(config, owner=model_config.owner, components=model_config.components)
    manager = _manager(model_config)
    assert manager.load_from_ckpt([str(path)], model_config, uids=["7"]) == ["7"]

    assert config.lora_target_modules == ["attn_dense"]
    assert {row[0] for row in manager.cpp_lora_config["7"].tolist()} == (
        {4, 64} if include_vision else {4}
    )


@pytest.mark.parametrize(
    "module_path",
    [
        "model.layers.7.mlp.gate_proj",
        "model.layers.7.self_attn.qkv_proj",
        "model.layers.7.self_attn.unknown_proj",
        "model.lm_head",
        "model.embed_tokens",
    ],
    ids=["disabled-mlp", "disabled-fused-qkv", "unknown-projection", "lm-head", "embedding"],
)
def test_multimodal_policy_rejects_unselected_decoder_weights_before_partial_loading(
    model_config: LoraModelConfig, cpu_transfers: Mock, tmp_path: Path, module_path: str
) -> None:
    module_mapping = {
        **model_config.trtllm_modules_to_hf_modules,
        "mlp_h_to_4h": "gate_proj",
        "attn_qkv": "qkv_proj",
    }
    model_config = replace(model_config, trtllm_modules_to_hf_modules=module_mapping)
    weights = _weights(model_config, ("decoder", "vision"))
    rejected_keys = [
        f"base_model.model.{module_path}.lora_{direction}.weight" for direction in ("A", "B")
    ]
    weights[rejected_keys[0]] = torch.ones(2, 4, dtype=torch.float16)
    weights[rejected_keys[1]] = torch.ones(4, 2, dtype=torch.float16)
    path = _write_adapter(tmp_path / "invalid", weights)
    config = LoraConfig(
        lora_dir=[str(path)],
        enable_multimodal_lora=True,
        lora_target_modules=["attn_dense"],
        trtllm_modules_to_hf_modules=module_mapping,
    )

    with pytest.raises(ValueError, match="not enabled") as startup_error:
        load_torch_lora(config, owner=model_config.owner, components=model_config.components)
    manager = _manager(model_config)
    with pytest.raises(ValueError, match="not enabled") as runtime_error:
        manager.load_from_ckpt([str(path)], model_config, uids=["7"])

    for error in (startup_error, runtime_error):
        assert any(key in str(error.value) for key in rejected_keys)
        assert "selected owner 'decoder'" in str(error.value)
        assert "Enabled decoder targets: ['attn_dense']" in str(error.value)
    _assert_unpublished(manager, "7")
    cpu_transfers.assert_not_called()
