# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from _torch.modeling.test_modeling_parakeet import _make_sound_config
from _torch.modeling.test_modeling_radio import _TINY_VIT, _make_bf16_model_config
from transformers import Mistral3Config, MistralConfig, PixtralVisionConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models import modeling_radio
from tensorrt_llm._torch.models.modeling_mistral import Mistral3MultiModalProjector
from tensorrt_llm._torch.models.modeling_multimodal_encoder import MultimodalEncoderMixin
from tensorrt_llm._torch.models.modeling_nemotron_h_multimodal import _NemotronVisionProjection
from tensorrt_llm._torch.models.modeling_parakeet import ParakeetProjection, ProjectedParakeet
from tensorrt_llm._torch.models.modeling_pixtral import PixtralVisionModel
from tensorrt_llm._torch.peft.lora.component_runtime import (
    ComponentLoraRuntime,
    bind_component_lora,
)
from tensorrt_llm._torch.peft.lora.layer import LoraLayer, LoraModuleType
from tensorrt_llm._torch.peft.lora.ownership import ComponentLoraOwner, ComponentLoraTarget

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _initialize(model: torch.nn.Module) -> None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.ndim >= 2:
                parameter.normal_(std=0.05)
            elif name.endswith("weight"):
                parameter.fill_(1)
            else:
                parameter.zero_()


def _attention_targets(
    model: torch.nn.Module, attention_path: str, projection: str
) -> tuple[tuple[ComponentLoraTarget, ...], list[tuple[torch.Tensor, slice]]]:
    attention = model.get_submodule(attention_path)
    if projection == "out":
        specs = [("attn_dense", "o_lora", attention.hidden_size, attention.o_proj.weight, 0)]
    else:
        specs = [
            ("attn_q", "splitted_qkv_lora", attention.q_size, attention.qkv_proj.weight, 0),
            (
                "attn_k",
                "splitted_qkv_lora",
                attention.kv_size,
                attention.qkv_proj.weight,
                attention.q_size,
            ),
            (
                "attn_v",
                "splitted_qkv_lora",
                attention.kv_size,
                attention.qkv_proj.weight,
                attention.q_size + attention.kv_size,
            ),
        ]
    targets = tuple(
        ComponentLoraTarget(
            module_path=f"{attention_path}.{lora_name}",
            checkpoint_modules=(f"tower.layers.0.{module_type}",),
            layer_idx=0,
            module_type=module_type,
            execution_id=-1,
            input_size=attention.hidden_size,
            output_size=output_size,
        )
        for module_type, lora_name, output_size, _, _ in specs
    )
    return targets, [(weight, slice(start, start + size)) for _, _, size, weight, start in specs]


def _assert_mixed_adapters_match_merged_weights(
    model: torch.nn.Module,
    targets: tuple[ComponentLoraTarget, ...],
    weight_slices: list[tuple[torch.Tensor, slice]],
    forward: Callable[[dict | None], torch.Tensor],
    physical_rows: list[int],
    output_rows: list[int],
) -> None:
    owner = ComponentLoraOwner("component", "", targets, "bfloat16", 4)
    owners = bind_component_lora(model, (owner,))
    runtime = ComponentLoraRuntime(owners)
    # Adapter tensors must outlive the pointer-based component forwards.
    buffers = []
    peft_table = {7: [], 8: []}
    deltas = []
    for target in owners[0].targets:
        a = torch.randn(4, target.input_size, device="cuda", dtype=torch.bfloat16) * 0.15
        b = torch.randn(target.output_size, 4, device="cuda", dtype=torch.bfloat16) * 0.15
        zero_b = torch.zeros_like(b)
        buffers.extend((a, b, zero_b))
        deltas.append(b.float() @ a.float())
        for task_id, output_weight in ((7, b), (8, zero_b)):
            peft_table[task_id].append(
                SimpleNamespace(
                    layer_id=target.layer_idx,
                    module_id=target.execution_id,
                    adapter_size=4,
                    weights_in_pointer=a.data_ptr(),
                    weights_out_pointer=output_weight.data_ptr(),
                    scaling_vec_pointer=0,
                )
            )
    runtime.bind_batch(peft_table, torch.bfloat16)
    with torch.inference_mode():
        base = forward(None).flatten(0, -2).clone()
        originals = [weight[rows].clone() for weight, rows in weight_slices]
        try:
            for (weight, rows), original, delta in zip(weight_slices, originals, deltas):
                weight[rows].copy_((original.float() + delta).to(weight.dtype))
            merged = forward(None).flatten(0, -2).clone()
        finally:
            for (weight, rows), original in zip(weight_slices, originals):
                weight[rows].copy_(original)

        assert torch.isfinite(base).all() and torch.isfinite(merged).all()
        assert (merged.float() - base.float()).abs().max().item() > 1e-3
        for task_ids in ((7, 8, None), (None, 7, 8)):
            params = runtime.parameters("component", list(zip(task_ids, physical_rows)))
            actual = forward(params).flatten(0, -2)
            expected_chunks = []
            start = 0
            for task_id, count in zip(task_ids, output_rows):
                reference = merged if task_id == 7 else base
                expected_chunks.append(reference[start : start + count])
                if task_id == 7:
                    reference_delta = (
                        (
                            merged[start : start + count].float()
                            - base[start : start + count].float()
                        )
                        .abs()
                        .max()
                    )
                    actual_delta = (
                        (
                            actual[start : start + count].float()
                            - base[start : start + count].float()
                        )
                        .abs()
                        .max()
                    )
                    assert reference_delta.item() > 1e-3
                    assert actual_delta > reference_delta / 4
                else:
                    torch.testing.assert_close(
                        actual[start : start + count], base[start : start + count], rtol=0, atol=0
                    )
                start += count
            expected = torch.cat(expected_chunks)
            torch.testing.assert_close(actual, expected, rtol=0.06, atol=0.02)


@pytest.mark.parametrize("projection", ["qkv", "out"])
@pytest.mark.parametrize("tower", ["radio", "pixtral", "parakeet"])
def test_tower_lora_matches_independently_merged_weights(tower: str, projection: str) -> None:
    torch.manual_seed(123)
    dtype = torch.bfloat16
    if tower == "radio":
        with patch.dict(modeling_radio.VIT_TIMM_CONFIG_BY_NAME, {"vit_tiny_test": _TINY_VIT}):
            model = modeling_radio.RADIOVisionModel(
                _make_bf16_model_config(), disable_quantization=True
            )
        model = model.to(device="cuda", dtype=dtype).eval()
        for module in model.modules():
            if isinstance(module, MultimodalEncoderMixin):
                module.setup_attn_metadata(max_num_tokens=128)
        values = torch.randn(3, 3, 32, 32, device="cuda", dtype=dtype)
        attention_path = "radio_model.model.blocks.0.attn"
        physical_rows = [4 + model.radio_model.model.patch_generator.num_skip] * 3
        output_rows = [4] * 3

        def forward(params: dict | None) -> torch.Tensor:
            return model(values, lora_params=params)
    elif tower == "pixtral":
        config = PixtralVisionConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=2,
            head_dim=32,
            image_size=32,
            patch_size=8,
            hidden_act="silu",
            torch_dtype=dtype,
        )
        model = PixtralVisionModel(ModelConfig(pretrained_config=config, attn_backend="VANILLA"))
        model = model.to(device="cuda", dtype=dtype).eval()
        model.setup_attn_metadata(max_num_tokens=128)
        values = torch.randn(3, 3, 32, 32, device="cuda", dtype=dtype)
        sizes = torch.tensor([[32, 32], [16, 16], [16, 16]], device="cuda")
        attention_path = "transformer.layers.0.attention"
        physical_rows = output_rows = [16, 4, 4]

        def forward(params: dict | None) -> torch.Tensor:
            return model(values, sizes, lora_params=params)
    else:
        model = ProjectedParakeet(_make_sound_config(), llm_hidden_size=128, dtype=dtype)
        model = model.to(device="cuda", dtype=dtype).eval()
        values = torch.randn(3, 200, 80, device="cuda", dtype=dtype)
        mask = (
            torch.arange(200, device="cuda")[None, :]
            < torch.tensor([200, 160, 120], device="cuda")[:, None]
        )
        attention_path = "encoder.layers.0.self_attn"
        rows = int(model.encoder._get_subsampling_output_length(torch.tensor([200])).item())
        physical_rows = output_rows = [rows] * 3

        def forward(params: dict | None) -> torch.Tensor:
            return model(values, mask, lora_params=params)

    _initialize(model)
    targets, weight_slices = _attention_targets(model, attention_path, projection)
    _assert_mixed_adapters_match_merged_weights(
        model, targets, weight_slices, forward, physical_rows, output_rows
    )


@pytest.mark.parametrize("connector", ["nemotron_vision", "parakeet_audio", "mistral_vision"])
def test_connector_lora_matches_independently_merged_weights(connector: str) -> None:
    torch.manual_seed(321)
    dtype = torch.bfloat16
    if connector == "nemotron_vision":
        model = _NemotronVisionProjection(
            torch.nn.Sequential(
                torch.nn.LayerNorm(32),
                torch.nn.Linear(32, 64),
                torch.nn.GELU(),
                torch.nn.Linear(64, 16),
            )
        ).to(device="cuda", dtype=dtype)
        linears = [(model[1], "linear1_lora"), (model[3], "linear2_lora")]
        values = torch.randn(3, 5, 32, device="cuda", dtype=dtype)
        rows = [5, 5, 5]

        def forward(params: dict | None) -> torch.Tensor:
            return model(values, lora_params=params)
    elif connector == "parakeet_audio":
        model = ParakeetProjection(32, 64, 16, dtype=dtype).to("cuda")
        linears = [(model.linear1, "linear1.lora"), (model.linear2, "linear2.lora")]
        values = torch.randn(3, 5, 32, device="cuda", dtype=dtype)
        rows = [5, 5, 5]

        def forward(params: dict | None) -> torch.Tensor:
            return model(values, lora_params=params)
    else:
        config = Mistral3Config(
            vision_config=PixtralVisionConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=32,
                patch_size=8,
            ),
            text_config=MistralConfig(
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
            ),
            spatial_merge_size=2,
            torch_dtype=dtype,
        )
        model = Mistral3MultiModalProjector(ModelConfig(pretrained_config=config)).to(
            device="cuda", dtype=dtype
        )
        linears = [(model.linear_1, "linear_1.lora"), (model.linear_2, "linear_2.lora")]
        values = torch.randn(24, 32, device="cuda", dtype=dtype)
        sizes = torch.tensor([[32, 32], [16, 16], [16, 16]], device="cuda")
        rows = [4, 1, 1]

        def forward(params: dict | None) -> torch.Tensor:
            return model(values, sizes, lora_params=params)

    model.eval()
    _initialize(model)
    targets = []
    weight_slices = []
    for (linear, lora_path), module_type, semantic_id in zip(
        linears,
        ("mlp_h_to_4h", "mlp_4h_to_h"),
        (LoraModuleType.MLP_H_TO_4H, LoraModuleType.MLP_4H_TO_H),
    ):
        if lora_path.endswith(".lora"):
            linear.lora = LoraLayer([semantic_id], [linear.out_features])
        targets.append(
            ComponentLoraTarget(
                module_path=lora_path,
                checkpoint_modules=(f"connector.{lora_path}",),
                layer_idx=0,
                module_type=module_type,
                execution_id=-1,
                input_size=linear.in_features,
                output_size=linear.out_features,
            )
        )
        weight_slices.append((linear.weight, slice(None)))
    _assert_mixed_adapters_match_merged_weights(
        model, tuple(targets), weight_slices, forward, rows, rows
    )


@pytest.mark.parametrize("active_projection", [0, 1], ids=["query-only", "key-only"])
def test_partial_qkv_component_lora_matches_dense_reference(active_projection: int) -> None:
    torch.manual_seed(456)
    output_sizes = [64, 32, 32]
    model = torch.nn.Module()
    model.projection = LoraLayer(
        [LoraModuleType.ATTENTION_Q, LoraModuleType.ATTENTION_K, LoraModuleType.ATTENTION_V],
        output_sizes,
    )
    targets = tuple(
        ComponentLoraTarget(
            module_path="projection",
            checkpoint_modules=(f"vision.{module_type}",),
            layer_idx=7,
            module_type=module_type,
            execution_id=-1,
            input_size=64,
            output_size=output_size,
        )
        for module_type, output_size in zip(("attn_q", "attn_k", "attn_v"), output_sizes)
    )
    owners = bind_component_lora(model, (ComponentLoraOwner("vision", "", targets, "bfloat16", 4),))
    target = owners[0].targets[active_projection]
    x = torch.randn(6, 64, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16) * 0.1
    b = torch.randn(target.output_size, 4, device="cuda", dtype=torch.bfloat16) * 0.1
    zero_b = torch.zeros_like(b)
    runtime = ComponentLoraRuntime(owners)
    runtime.bind_batch(
        {
            task_id: [
                SimpleNamespace(
                    layer_id=7,
                    module_id=target.execution_id,
                    adapter_size=4,
                    weights_in_pointer=a.data_ptr(),
                    weights_out_pointer=out_weight.data_ptr(),
                    scaling_vec_pointer=0,
                )
            ]
            for task_id, out_weight in ((7, b), (8, zero_b))
        },
        torch.bfloat16,
    )
    params = runtime.parameters("vision", [(None, 1), (7, 2), (8, 3)])

    with torch.inference_mode():
        actual = model.projection(x, params, layer_idx=7)

    expected = torch.zeros(6, sum(output_sizes), device="cuda", dtype=torch.bfloat16)
    offset = sum(output_sizes[:active_projection])
    expected[1:3, offset : offset + target.output_size] = (
        x[1:3].float() @ a.float().T @ b.float().T
    ).to(torch.bfloat16)
    assert torch.count_nonzero(expected) > 0
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.005)
    assert torch.count_nonzero(actual[0]) == 0
    assert torch.count_nonzero(actual[3:]) == 0
