# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file

from tensorrt_llm._torch.models.modeling_nemotron_h_multimodal import NemotronHMultimodalModel
from tensorrt_llm._torch.peft.lora.component_runtime import bind_component_lora
from tensorrt_llm._torch.peft.lora.layer import LoraLayer, LoraModuleType
from tensorrt_llm._torch.peft.lora.manager import LoraManager, LoraModelConfig
from tensorrt_llm._torch.peft.lora.ownership import ComponentLoraOwner, ComponentLoraTarget
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


def _components() -> tuple[ComponentLoraOwner, ...]:
    column = ComponentLoraTarget(
        module_path="qkv",
        checkpoint_modules=("vision.layers.0.attn.qkv",),
        layer_idx=0,
        module_type="attn_qkv",
        execution_id=-1,
        input_size=12,
        output_size=16,
        fused_output_sizes=(8, 4, 4),
        tp_size=4,
        out_tp_split_dim=0,
    )
    row = ComponentLoraTarget(
        module_path="output",
        checkpoint_modules=("vision.layers.0.attn.o_proj",),
        layer_idx=0,
        module_type="attn_dense",
        execution_id=-1,
        input_size=12,
        output_size=8,
        tp_size=4,
        in_tp_split_dim=1,
    )
    connector = ComponentLoraTarget(
        module_path="projection",
        checkpoint_modules=("vision_connector.linear",),
        layer_idx=0,
        module_type="attn_dense",
        execution_id=-1,
        input_size=8,
        output_size=6,
    )
    return (
        ComponentLoraOwner("vision", "vision", (column, row), "bfloat16", 4, ("vision",)),
        ComponentLoraOwner(
            "vision_connector",
            "vision_connector",
            (connector,),
            "bfloat16",
            4,
            ("vision_connector",),
        ),
    )


def _model() -> torch.nn.Module:
    model = torch.nn.Module()
    model.vision = torch.nn.Module()
    model.vision.qkv = LoraLayer([LoraModuleType.ATTENTION_QKV], [4])
    model.vision.output = LoraLayer([LoraModuleType.ATTENTION_DENSE], [8])
    model.vision_connector = torch.nn.Module()
    model.vision_connector.projection = LoraLayer([LoraModuleType.ATTENTION_DENSE], [6])
    return model


def test_tp4_binding_preserves_global_dimensions_and_replicated_connector() -> None:
    model = _model()
    bound = bind_component_lora(model, _components(), tp_size=4)
    qkv, output = bound[0].targets
    (connector,) = bound[1].targets

    assert (qkv.input_size, qkv.output_size, qkv.fused_output_sizes) == (12, 16, (8, 4, 4))
    assert (qkv.tp_size, qkv.in_tp_split_dim, qkv.out_tp_split_dim) == (4, -1, 0)
    assert (output.tp_size, output.in_tp_split_dim, output.out_tp_split_dim) == (4, 1, -1)
    assert (connector.tp_size, connector.in_tp_split_dim, connector.out_tp_split_dim) == (1, -1, -1)
    ids = {target.execution_id for owner in bound for target in owner.targets}
    assert len(ids) == 3 and ids.isdisjoint(LoraManager.LORA_MODULE_IDS.values())
    assert model.vision.qkv.execution_module_ids == (qkv.execution_id,)
    assert model.vision.output.execution_module_ids == (output.execution_id,)
    assert model.vision_connector.projection.execution_module_ids == (connector.execution_id,)


@pytest.mark.parametrize("tp_size", [0, -1])
def test_binding_requires_a_positive_executor_tp_size(tp_size: int) -> None:
    model = _model()

    with pytest.raises(ValueError, match="positive executor TP size"):
        bind_component_lora(model, _components(), tp_size=tp_size)

    assert model.vision.qkv.execution_module_ids is None


@pytest.mark.parametrize(
    ("updates", "expected_error"),
    [
        ({"tp_size": 2}, "executor TP group"),
        ({"tp_size": 1}, "executor TP group"),
        ({"in_tp_split_dim": 1}, "one split dimension"),
        ({"out_tp_split_dim": -1}, "one split dimension"),
        ({"out_tp_split_dim": 1}, "one split dimension"),
        ({"output_size": 15, "fused_output_sizes": ()}, "divisible"),
        ({"input_size": 0}, "Invalid dimensions"),
        ({"fused_output_sizes": (8, 4)}, "fused projection sizes"),
        ({"fused_output_sizes": (10, 3, 3)}, "fused projection sizes"),
        ({"fused_output_sizes": (16, 0, 0)}, "fused projection sizes"),
        ({"output_size": 20, "fused_output_sizes": (12, 4, 4)}, "local output size"),
    ],
)
def test_invalid_tp_target_does_not_publish_any_execution_ids(
    updates: dict, expected_error: str
) -> None:
    model = _model()
    vision, connector = _components()
    qkv, output = vision.targets
    vision = replace(vision, targets=(output, replace(qkv, **updates)))

    with pytest.raises(ValueError, match=expected_error):
        bind_component_lora(model, (connector, vision), tp_size=4)

    assert all(
        layer.execution_module_ids is None
        for layer in model.modules()
        if isinstance(layer, LoraLayer)
    )


def test_row_parallel_target_validates_its_global_input_partition() -> None:
    model = _model()
    vision, connector = _components()
    qkv, output = vision.targets
    vision = replace(vision, targets=(qkv, replace(output, input_size=10)))

    with pytest.raises(ValueError, match="divisible"):
        bind_component_lora(model, (vision, connector), tp_size=4)

    assert model.vision.qkv.execution_module_ids is None


def test_row_sharded_fused_target_is_rejected_before_binding() -> None:
    model = _model()
    vision, connector = _components()
    qkv, output = vision.targets
    vision = replace(vision, targets=(qkv, replace(output, fused_output_sizes=(4, 4))))

    with pytest.raises(ValueError, match="Fused component LoRA targets require output TP sharding"):
        bind_component_lora(model, (connector, vision), tp_size=4)

    assert all(
        layer.execution_module_ids is None
        for layer in model.modules()
        if isinstance(layer, LoraLayer)
    )


def test_split_qkv_binding_uses_each_projection_local_width() -> None:
    model = _model()
    model.vision.qkv = LoraLayer(
        [LoraModuleType.ATTENTION_Q, LoraModuleType.ATTENTION_K, LoraModuleType.ATTENTION_V],
        [2, 1, 1],
    )
    (vision,) = _components()[:1]
    packed = vision.targets[0]
    targets = tuple(
        replace(
            packed,
            module_type=name,
            output_size=width,
            fused_output_sizes=(),
            checkpoint_modules=(f"vision.layers.0.attn.{name}",),
        )
        for name, width in (("attn_q", 8), ("attn_k", 4), ("attn_v", 4))
    )

    bound = bind_component_lora(model, (replace(vision, targets=targets),), tp_size=4)

    assert model.vision.qkv.execution_module_ids == tuple(
        target.execution_id for target in bound[0].targets
    )
    assert len(set(model.vision.qkv.execution_module_ids)) == 3


@pytest.mark.parametrize("rank", range(4))
def test_tp4_host_packing_places_each_qkv_rank_in_one_contiguous_region(
    rank: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor, *args, **kwargs: tensor)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (8, 0))
    mapping = Mapping(world_size=4, tp_size=4, rank=rank)
    bound = bind_component_lora(_model(), _components(), tp_size=mapping.tp_size)
    targets = [target for owner in bound for target in owner.targets]
    model_config = LoraModelConfig(
        lora_target_modules=["attn_dense"],
        trtllm_modules_to_hf_modules={"attn_dense": "o_proj"},
        hidden_size=8,
        dtype="bfloat16",
        components=bound,
    )
    weights = {}
    originals = {}
    for index, target in enumerate(targets):
        a = (torch.arange(2 * target.input_size).reshape(2, target.input_size) + index * 40).to(
            torch.bfloat16
        )
        b = (torch.arange(2 * target.output_size).reshape(target.output_size, 2) + index * 40).to(
            torch.bfloat16
        )
        prefix = "base_model.model." + target.checkpoint_modules[0]
        weights[prefix + ".lora_A.weight"], weights[prefix + ".lora_B.weight"] = a, b
        originals[target.execution_id] = (a, b)
    decoder_a = torch.arange(16, dtype=torch.bfloat16).reshape(2, 8)
    decoder_b = torch.arange(16, dtype=torch.bfloat16).reshape(8, 2)
    decoder_prefix = "base_model.model.model.layers.0.self_attn.o_proj"
    weights[decoder_prefix + ".lora_A.weight"] = decoder_a
    weights[decoder_prefix + ".lora_B.weight"] = decoder_b
    save_file(weights, tmp_path / "adapter_model.safetensors")
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"r": 2, "lora_alpha": 4, "target_modules": ["qkv", "o_proj", "linear"]})
    )
    manager = LoraManager(mapping=mapping, model_config=model_config, cpp_peft_cache_manager=Mock())

    manager.load_from_ckpt([str(tmp_path)], model_config, uids=["7"])

    packed = {
        int(config[0]): tensor
        for config, tensor in zip(
            manager.cpp_lora_config["7"], manager.cpp_lora_weights["7"], strict=True
        )
    }
    assert set(packed) == {LoraManager.LORA_MODULE_IDS["attn_dense"], *originals}
    assert manager.cpp_lora_config["7"][:, 2:].tolist() == [[2, 0]] * 4
    qkv = targets[0]
    a, b = originals[qkv.execution_id]
    actual_b = packed[qkv.execution_id][a.numel() : a.numel() + b.numel()].reshape(4, 4, 2)
    expected_rows = torch.tensor([rank * 2, rank * 2 + 1, 8 + rank, 12 + rank])
    torch.testing.assert_close(actual_b[rank], b[expected_rows] * 2, rtol=0, atol=0)
    for target in targets:
        a, b = originals[target.execution_id]
        torch.testing.assert_close(
            packed[target.execution_id][: a.numel()], a.flatten(), rtol=0, atol=0
        )
        if not target.fused_output_sizes:
            torch.testing.assert_close(
                packed[target.execution_id][a.numel() : a.numel() + b.numel()],
                (b * 2).flatten(),
                rtol=0,
                atol=0,
            )
    expected_decoder = torch.cat((decoder_a.flatten(), (decoder_b * 2).flatten()))
    torch.testing.assert_close(
        packed[LoraManager.LORA_MODULE_IDS["attn_dense"]][: expected_decoder.numel()],
        expected_decoder,
        rtol=0,
        atol=0,
    )


def _attention(mapping: Mapping, hidden_size: int, num_heads: int) -> torch.nn.Module:
    attention = torch.nn.Module()
    attention.mapping = mapping
    attention.tp_size = mapping.tp_size
    attention.hidden_size = hidden_size
    attention.num_heads = attention.num_key_value_heads = num_heads // mapping.tp_size
    attention.head_dim = hidden_size // num_heads
    attention.q_size = attention.kv_size = hidden_size // mapping.tp_size
    attention.qkv_proj = torch.nn.Linear(
        hidden_size, 3 * attention.q_size, bias=False, device="meta", dtype=torch.bfloat16
    )
    attention.o_proj = torch.nn.Linear(
        attention.q_size, hidden_size, bias=False, device="meta", dtype=torch.bfloat16
    )
    attention.splitted_qkv_lora = LoraLayer(
        [LoraModuleType.ATTENTION_Q, LoraModuleType.ATTENTION_K, LoraModuleType.ATTENTION_V],
        [attention.q_size] * 3,
    )
    attention.fused_qkv_lora = LoraLayer([LoraModuleType.ATTENTION_QKV], [3 * attention.q_size])
    attention.o_lora = LoraLayer([LoraModuleType.ATTENTION_DENSE], [hidden_size])
    return attention


def _nemotron(*, super_model: bool, rank: int = 0) -> NemotronHMultimodalModel:
    tp_size, hidden_size, num_layers = (4, 4096, 88) if super_model else (1, 2688, 52)
    mapping = Mapping(world_size=tp_size, tp_size=tp_size, rank=rank)
    model = NemotronHMultimodalModel.__new__(NemotronHMultimodalModel)
    torch.nn.Module.__init__(model)
    config = SimpleNamespace(
        architectures=[
            "NemotronH_Omni_Reasoning_V3" if super_model else "NemotronH_Nano_Omni_Reasoning_V3"
        ],
        torch_dtype=torch.bfloat16,
        llm_config=SimpleNamespace(hidden_size=hidden_size, num_hidden_layers=num_layers),
        vit_hidden_size=1280,
        projector_hidden_size=20480,
        patch_size=16,
        downsample_ratio=0.5,
        vision_config=SimpleNamespace(args={"model": "vit_huge_patch16_224"}),
        sound_config=None
        if super_model
        else SimpleNamespace(
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=8,
            projection_hidden_size=4096,
        ),
    )
    model._mm_model_config = SimpleNamespace(
        pretrained_config=config, mapping=mapping, quant_config=SimpleNamespace(quant_algo=None)
    )
    vision = torch.nn.Module()
    vision.embed_dim = 1280
    vision.blocks = torch.nn.ModuleList()
    for _ in range(32):
        block = torch.nn.Module()
        block.attn = _attention(mapping, 1280, 16)
        vision.blocks.append(block)
    model.vision_encoder = torch.nn.Module()
    model.vision_encoder.vision_model = torch.nn.Module()
    model.vision_encoder.vision_model.radio_model = torch.nn.Module()
    model.vision_encoder.vision_model.radio_model.model = vision
    model.vision_encoder.mlp1 = torch.nn.Sequential(
        torch.nn.Identity(),
        torch.nn.Linear(5120, 20480, device="meta", dtype=torch.bfloat16),
        torch.nn.Identity(),
        torch.nn.Linear(20480, hidden_size, device="meta", dtype=torch.bfloat16),
    )
    model.sound_encoder = None
    if not super_model:
        model.sound_encoder = torch.nn.Module()
        model.sound_encoder.encoder = torch.nn.Module()
        model.sound_encoder.encoder.layers = torch.nn.ModuleList()
        for _ in range(24):
            block = torch.nn.Module()
            block.self_attn = _attention(mapping, 1024, 8)
            model.sound_encoder.encoder.layers.append(block)
        model.sound_encoder.projection = torch.nn.Module()
        for name, size_in, size_out in (("linear1", 1024, 4096), ("linear2", 4096, hidden_size)):
            linear = torch.nn.Linear(size_in, size_out, device="meta", dtype=torch.bfloat16)
            linear.tp_size = 1
            setattr(model.sound_encoder.projection, name, linear)
    return model


@pytest.mark.parametrize("super_model", [False, True])
def test_nemotron_enrollment_preserves_global_vision_and_replicated_projector_dimensions(
    super_model: bool,
) -> None:
    model = _nemotron(super_model=super_model)
    tp_size = 4 if super_model else 1

    declarations = model.configure_component_lora(4)
    bound = bind_component_lora(model, declarations, tp_size=tp_size)

    assert model.component_lora_tp_sizes == (tp_size,)
    assert {owner.name for owner in bound} == (
        {"vision", "vision_connector"}
        if super_model
        else {"vision", "vision_connector", "audio", "audio_connector"}
    )
    vision = next(owner for owner in bound if owner.name == "vision")
    assert len(vision.targets) == 32 * 5
    for target in vision.targets:
        assert target.tp_size == tp_size and target.input_size == 1280
        assert target.output_size == (3840 if target.module_type == "attn_qkv" else 1280)
        expected_split = (-1, -1)
        if super_model:
            expected_split = (1, -1) if target.module_type == "attn_dense" else (-1, 0)
        assert (target.in_tp_split_dim, target.out_tp_split_dim) == expected_split
    connector = next(owner for owner in bound if owner.name == "vision_connector")
    assert [(target.input_size, target.output_size) for target in connector.targets] == [
        (5120, 20480),
        (20480, 4096 if super_model else 2688),
    ]
    assert all(
        (target.tp_size, target.in_tp_split_dim, target.out_tp_split_dim) == (1, -1, -1)
        for target in connector.targets
    )
    assert model.configure_component_lora(4) is declarations


def test_super_execution_ids_are_identical_across_all_four_rank_local_models() -> None:
    signatures = []
    for rank in range(4):
        model = _nemotron(super_model=True, rank=rank)
        bound = bind_component_lora(model, model.configure_component_lora(4), tp_size=4)
        signatures.append(
            {
                (owner.name, target.layer_idx, target.module_type): target.execution_id
                for owner in bound
                for target in owner.targets
            }
        )
    assert all(signature == signatures[0] for signature in signatures)
    assert len(set(signatures[0].values())) == 7
    assert set(signatures[0].values()).isdisjoint(LoraManager.LORA_MODULE_IDS.values())


@pytest.mark.parametrize(
    "invalid",
    [
        "architecture",
        "dtype",
        "quantization",
        "audio_config",
        "audio_module",
        "decoder_width",
        "vision_tp",
        "projector_width",
        "vision_model",
    ],
)
def test_super_rejects_other_configurations_before_projector_wrapping(invalid: str) -> None:
    model = _nemotron(super_model=True)
    config = model._mm_model_config.pretrained_config
    projection = model.vision_encoder.mlp1
    if invalid == "architecture":
        config.architectures = ["NemotronH_Omni_Reasoning_V3_Extra"]
    elif invalid == "dtype":
        config.torch_dtype = torch.float16
    elif invalid == "quantization":
        model._mm_model_config.quant_config.quant_algo = "FP8"
    elif invalid == "audio_config":
        config.sound_config = SimpleNamespace()
    elif invalid == "audio_module":
        model.sound_encoder = torch.nn.Module()
    elif invalid == "decoder_width":
        config.llm_config.hidden_size = 2688
    elif invalid == "vision_tp":
        model.vision_encoder.vision_model.radio_model.model.blocks[-1].attn.tp_size = 2
    elif invalid == "projector_width":
        projection[3] = torch.nn.Linear(20480, 2688, device="meta", dtype=torch.bfloat16)
    else:
        config.vision_config.args["model"] = "other_vit"

    with pytest.raises(ValueError):
        model.configure_component_lora(4)

    assert model.vision_encoder.mlp1 is projection
    assert not hasattr(model, "_component_lora_owners")
    assert all(
        layer.execution_module_ids is None
        for layer in model.modules()
        if isinstance(layer, LoraLayer)
    )


@pytest.mark.parametrize("missing", ["config", "module"])
def test_nano_requires_its_audio_tower_before_projector_wrapping(missing: str) -> None:
    model = _nemotron(super_model=False)
    projection = model.vision_encoder.mlp1
    if missing == "config":
        model._mm_model_config.pretrained_config.sound_config = None
    else:
        model.sound_encoder = None

    with pytest.raises(ValueError, match="native Parakeet audio tower"):
        model.configure_component_lora(4)

    assert model.vision_encoder.mlp1 is projection
    assert not hasattr(model, "_component_lora_owners")


@pytest.fixture
def super_loader(monkeypatch: pytest.MonkeyPatch) -> tuple[LoraManager, LoraModelConfig, Mock]:
    transfer = Mock(side_effect=lambda tensor: tensor)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor, *args, **kwargs: transfer(tensor))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (8, 0))
    model = _nemotron(super_model=True)
    components = bind_component_lora(model, model.configure_component_lora(4), tp_size=4)
    config = LoraModelConfig(
        lora_target_modules=["attn_dense"],
        trtllm_modules_to_hf_modules={"attn_dense": "o_proj"},
        hidden_size=4096,
        dtype="bfloat16",
        components=components,
    )
    manager = LoraManager(
        mapping=Mapping(world_size=4, tp_size=4),
        model_config=config,
        cpp_peft_cache_manager=Mock(),
    )
    return manager, config, transfer


def _projection_weights(module: str) -> dict[str, torch.Tensor]:
    prefix = f"base_model.model.{module}"
    return {
        f"{prefix}.lora_A.weight": torch.ones(2, 1280, dtype=torch.bfloat16),
        f"{prefix}.lora_B.weight": torch.full((1280, 2), 2, dtype=torch.bfloat16),
    }


def _write_projection_adapter(path: Path, weights: dict[str, torch.Tensor]) -> Path:
    path.mkdir()
    save_file(weights, path / "adapter_model.safetensors")
    (path / "adapter_config.json").write_text(
        json.dumps({"r": 2, "lora_alpha": 2, "target_modules": ["query", "key", "value", "o_proj"]})
    )
    return path


def _assert_uid_unpublished(manager: LoraManager, uid: str) -> None:
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
    "audio_module", ["sound_encoder.encoder.layers.0.self_attn.o_proj", "sound_projection.linear1"]
)
def test_super_rejects_audio_weights_before_publication_and_accepts_same_uid_retry(
    super_loader: tuple[LoraManager, LoraModelConfig, Mock], tmp_path: Path, audio_module: str
) -> None:
    manager, config, transfer = super_loader
    vision = _projection_weights("vision_model.encoder.layer.0.attention.attention.query")
    invalid = _write_projection_adapter(
        tmp_path / "invalid", vision | _projection_weights(audio_module)
    )

    with pytest.raises(ValueError, match="outside selected owner") as error:
        manager.load_from_ckpt([str(invalid)], config, uids=["7"])

    assert audio_module in str(error.value)
    _assert_uid_unpublished(manager, "7")
    transfer.assert_not_called()
    valid = _write_projection_adapter(tmp_path / "valid", vision)
    assert manager.load_from_ckpt([str(valid)], config, uids=["7"]) == ["7"]
    assert set(manager.component_fingerprints["7"]) == {"vision"}


@pytest.mark.parametrize(
    ("projection", "module_type"), [("query", "attn_q"), ("key", "attn_k"), ("value", "attn_v")]
)
def test_super_nested_modelopt_alias_loads_the_declared_projection(
    super_loader: tuple[LoraManager, LoraModelConfig, Mock],
    tmp_path: Path,
    projection: str,
    module_type: str,
) -> None:
    manager, config, _ = super_loader
    module = f"vision_model.encoder.layer.0.attention.attention.{projection}"
    path = _write_projection_adapter(tmp_path / "adapter", _projection_weights(module))

    assert manager.load_from_ckpt([str(path)], config, uids=["7"]) == ["7"]

    vision = next(owner for owner in config.components if owner.name == "vision")
    target = next(
        target
        for target in vision.targets
        if target.layer_idx == 0 and target.module_type == module_type
    )
    assert manager.cpp_lora_config["7"].tolist() == [[target.execution_id, 0, 2, 0]]
    expected = torch.cat([tensor.flatten() for tensor in _projection_weights(module).values()])
    torch.testing.assert_close(
        manager.cpp_lora_weights["7"][0, : expected.numel()], expected, rtol=0, atol=0
    )


@pytest.mark.parametrize("projection", ["query", "key", "value"])
def test_super_short_and_nested_modelopt_aliases_reject_duplicate_destinations(
    super_loader: tuple[LoraManager, LoraModelConfig, Mock], tmp_path: Path, projection: str
) -> None:
    manager, config, transfer = super_loader
    short = f"vision_model.encoder.layer.0.attention.{projection}"
    nested = f"vision_model.encoder.layer.0.attention.attention.{projection}"
    weights = _projection_weights(short) | _projection_weights(nested)
    path = _write_projection_adapter(tmp_path / "adapter", weights)

    with pytest.raises(ValueError, match="duplicate destination") as error:
        manager.load_from_ckpt([str(path)], config, uids=["7"])

    assert short in str(error.value) and nested in str(error.value)
    assert "selected owner 'vision'" in str(error.value)
    _assert_uid_unpublished(manager, "7")
    transfer.assert_not_called()
