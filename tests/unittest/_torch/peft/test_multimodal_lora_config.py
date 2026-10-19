# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.peft.lora.config import (
    LoraConfig,
    get_default_trtllm_modules_to_hf_modules,
)
from tensorrt_llm._torch.peft.lora.layer import LoraLayer, LoraModuleType
from tensorrt_llm._torch.peft.lora.ownership import ComponentLoraOwner, ComponentLoraTarget
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
from tensorrt_llm.llmapi.llm_args import MultimodalConfig, PrefillCudaGraphBackend, TorchLlmArgs
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


def _engine(multimodal_config: MultimodalConfig | None = None) -> PyTorchModelEngine:
    engine = object.__new__(PyTorchModelEngine)
    engine._cleanup_done = True
    engine._lora_components = None
    engine.mapping = Mapping()
    engine.llm_args = SimpleNamespace(
        multimodal_config=multimodal_config or MultimodalConfig(),
        mm_encoder_only=False,
        disable_mm_encoder=False,
        encode_only=False,
        torch_compile_config=None,
    )
    engine.prefill_cuda_graph_backend = PrefillCudaGraphBackend.DISABLED
    engine.model = torch.nn.Module()
    engine.model.projection = LoraLayer([LoraModuleType.ATTENTION_DENSE], [8])
    owner = ComponentLoraOwner(
        name="vision",
        module_path="",
        dtype="bfloat16",
        max_rank=8,
        targets=(
            ComponentLoraTarget(
                module_path="projection",
                checkpoint_modules=("vision.output",),
                layer_idx=0,
                module_type="attn_dense",
                execution_id=-1,
                input_size=8,
                output_size=8,
            ),
        ),
    )
    engine.model.configure_component_lora = Mock(return_value=(owner,))
    return engine


@pytest.mark.parametrize(
    ("setting", "expected_error"),
    [
        ("topology", "TP1/PP1"),
        ("encoder_graph", "eager encoders"),
        ("side_stream", "side-stream prefetch"),
        ("eager_scheduler", "EAGER encoder scheduling"),
        ("positive_evs", "EVS video pruning"),
        ("encoder_only", "local encoder and decoder"),
        ("prefill_graph", "prefill CUDA graphs"),
        ("compile", "torch.compile"),
    ],
)
def test_component_lora_rejects_incompatible_execution_before_model_enrollment(
    setting: str, expected_error: str
) -> None:
    engine = _engine()
    mm_updates = {
        "encoder_graph": {"encoder_cuda_graph": {"vision": {"buckets": [(16, 1)]}}},
        "side_stream": {"encoder_side_stream_max_ahead": 1},
        "eager_scheduler": {"encoder_scheduling_policy": "EAGER"},
        "positive_evs": {"video_pruning_rate": 0.5},
    }
    if setting in mm_updates:
        engine.llm_args.multimodal_config = MultimodalConfig(**mm_updates[setting])
    elif setting == "topology":
        engine.mapping = Mapping(world_size=2, tp_size=2)
    elif setting == "encoder_only":
        engine.llm_args.mm_encoder_only = True
    elif setting == "prefill_graph":
        engine.prefill_cuda_graph_backend = PrefillCudaGraphBackend.PIECEWISE
    else:
        engine.llm_args.torch_compile_config = object()

    with pytest.raises(ValueError, match=expected_error):
        engine.initialize_lora_components(LoraConfig(enable_multimodal_lora=True))

    engine.model.configure_component_lora.assert_not_called()
    assert engine._lora_components is None
    assert engine.model.projection.execution_module_ids is None


@pytest.mark.parametrize("scheduling_policy", ["DEFAULT", "DISABLED"])
@pytest.mark.parametrize("pruning_rate", [None, 0.0])
def test_component_lora_accepts_inline_or_admitted_item_scheduling_without_pruning(
    scheduling_policy: str, pruning_rate: float | None
) -> None:
    engine = _engine(
        MultimodalConfig(
            encoder_scheduling_policy=scheduling_policy, video_pruning_rate=pruning_rate
        )
    )
    config = LoraConfig(enable_multimodal_lora=True, max_lora_rank=8)

    components = engine.initialize_lora_components(config)

    engine.model.configure_component_lora.assert_called_once_with(8)
    assert engine.model.projection.execution_module_ids == (components[0].targets[0].execution_id,)
    assert engine._component_lora_runtime is engine.model._component_lora_runtime
    assert engine.initialize_lora_components(config) is components


def test_default_decoder_only_policy_does_not_apply_component_execution_guards() -> None:
    engine = _engine(MultimodalConfig(encoder_scheduling_policy="EAGER"))
    engine.mapping = Mapping(world_size=2, tp_size=2)

    assert engine.initialize_lora_components(LoraConfig()) == ()

    engine.model.configure_component_lora.assert_not_called()


def test_empty_multimodal_targets_wait_for_model_capability_resolution() -> None:
    config = LoraConfig(enable_multimodal_lora=True)
    args = TorchLlmArgs.model_construct(lora_config=config)

    assert args.validate_lora_config_consistency() is args

    assert args.lora_config is config
    assert config.lora_target_modules == []


def test_empty_decoder_only_targets_keep_existing_defaults() -> None:
    config = LoraConfig()
    args = TorchLlmArgs.model_construct(lora_config=config)

    assert args.validate_lora_config_consistency() is args

    assert args.lora_config is config
    assert config.lora_target_modules == list(get_default_trtllm_modules_to_hf_modules())
