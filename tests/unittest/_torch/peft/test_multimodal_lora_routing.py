# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_multimodal_mixin import MultimodalModelMixin
from tensorrt_llm._torch.peft.lora.component_runtime import ComponentLoraRuntime
from tensorrt_llm._torch.peft.lora.ownership import (
    ComponentLoraOwner,
    ComponentLoraRequest,
    ComponentLoraTarget,
)
from tensorrt_llm.inputs.multimodal import MultimodalInput, MultimodalParams, MultimodalRuntimeData
from tensorrt_llm.llmapi.llm_args import MultimodalConfig

pytestmark = pytest.mark.cpu_only


class _EncoderCacheModel(MultimodalModelMixin):
    supports_encoder_cache = True
    embedding_dim = 4
    embedding_dtype = torch.float32

    def __init__(self) -> None:
        self.model_config = ModelConfig(
            multimodal_config=MultimodalConfig(encoder_cache_max_bytes=4096)
        )
        self.encode_calls = 0
        self._initialize_multimodal_encoder_cache(4096)

    def encode_multimodal_inputs(self, params: list[MultimodalParams], **kwargs) -> torch.Tensor:
        self.encode_calls += 1
        rows = sum(sum(param.multimodal_data["multimodal_embedding_lengths"]) for param in params)
        return torch.full((rows, self.embedding_dim), float(self.encode_calls))


def _param(request: ComponentLoraRequest | None, *, items: int = 1) -> MultimodalParams:
    return MultimodalParams(
        multimodal_input=MultimodalInput(
            multimodal_hashes=[[index + 1] * 8 for index in range(items)],
            multimodal_positions=[0] * items,
            multimodal_lengths=[2] * items,
        ),
        multimodal_data={
            "image": {
                "pixel_values": torch.arange(items * 12).reshape(items, 3, 2, 2),
                "image_sizes": [[2, 2]] * items,
            },
            "multimodal_embedding_lengths": [2] * items,
            "mm_processor_kwargs_hash": "same-processor",
        },
        multimodal_runtime=MultimodalRuntimeData(
            embed_mask_cumsum=torch.arange(1, 2 * items + 1),
            past_seen_token_num=0,
            chunk_end_pos=2 * items,
        ),
        component_lora_request=request,
    )


@pytest.mark.parametrize("component", ["vision", "vision_connector"])
def test_encoder_cache_separates_component_versions_and_reuses_original(component: str) -> None:
    model = _EncoderCacheModel()
    first = ComponentLoraRequest(7, "revision-a", ((component, "weights-a"),))
    second = ComponentLoraRequest(8, "revision-b", ((component, "weights-b"),))
    reloaded = ComponentLoraRequest(7, "revision-c", ((component, "weights-c"),))

    outputs = [
        model._get_or_encode_multimodal_embeddings([_param(request)])
        for request in (first, second, first, reloaded, second)
    ]

    assert model.encode_calls == 3
    for output, expected_marker in zip(outputs, [1, 2, 1, 3, 2]):
        torch.testing.assert_close(output, torch.full((2, 4), float(expected_marker)))


def test_encoder_cache_reuses_identical_components_across_decoder_changes() -> None:
    model = _EncoderCacheModel()
    decoder_a = ComponentLoraRequest(7, "decoder-a")
    decoder_b = ComponentLoraRequest(8, "decoder-b")
    outputs = [
        model._get_or_encode_multimodal_embeddings([_param(request)])
        for request in (None, decoder_a, decoder_b, None)
    ]

    assert model.encode_calls == 1
    for output in outputs:
        torch.testing.assert_close(output, torch.ones(2, 4))


def test_encoder_cache_uses_relevant_component_content_instead_of_request_revision() -> None:
    model = _EncoderCacheModel()
    request = ComponentLoraRequest(7, "revision-a", (("vision", "unchanged"),))
    first = model._get_or_encode_multimodal_embeddings([_param(request)])
    same_component = replace(request, task_id=8, revision="decoder-changed")
    second = model._get_or_encode_multimodal_embeddings([_param(same_component)])

    assert model.encode_calls == 1
    torch.testing.assert_close(second, first)


def test_audio_only_adapter_keeps_image_and_silent_video_cache_keys() -> None:
    request = ComponentLoraRequest(7, "audio-version", (("audio", "audio-a"),))
    refs = [("image", 0), ("video", 0), ("video", 1)]
    data = {"video": {"audio": {"has_audio": [False, True]}}}
    kwargs = dict(
        multimodal_hashes=[[1] * 8, [2] * 8, [3] * 8],
        item_refs=refs,
        embedding_lengths=[2, 3, 4],
        kwargs_hash="same-processor",
        multimodal_data=data,
    )

    base = MultimodalModelMixin.build_encoder_cache_item_keys(**kwargs)
    adapted = MultimodalModelMixin.build_encoder_cache_item_keys(
        **kwargs, component_lora_request=request
    )

    assert adapted[:2] == base[:2]
    assert adapted[2] != base[2]


def test_sliced_media_preserves_adapter_selection_and_item_order() -> None:
    request = ComponentLoraRequest(7, "version-a", (("vision", "weights-a"),))
    param = _param(request, items=3)
    model = _EncoderCacheModel()

    sliced = model.build_multimodal_encoder_input(param, [2, 0])
    model._apply_metadata_slice(sliced, param, [2, 0])

    assert sliced.component_lora_request is request
    assert sliced.multimodal_input.multimodal_hashes == [[3] * 8, [1] * 8]
    torch.testing.assert_close(
        sliced.multimodal_data["image"]["pixel_values"],
        param.multimodal_data["image"]["pixel_values"][[2, 0]],
    )


def test_attached_embeddings_require_matching_component_provenance() -> None:
    request = ComponentLoraRequest(7, "version-a", (("vision_connector", "weights-a"),))
    param = _param(request)
    param.multimodal_data["multimodal_embedding"] = torch.ones(2, 4)

    with pytest.raises(ValueError, match="provenance"):
        param.validate_embedding_lora_provenance()
    param.mark_embedding_lora_provenance()
    param.validate_embedding_lora_provenance()
    param.component_lora_request = replace(request, revision="decoder-only-reload")
    param.validate_embedding_lora_provenance()
    param.component_lora_request = replace(
        request, revision="version-b", component_fingerprints=(("vision_connector", "weights-b"),)
    )
    with pytest.raises(ValueError, match="different encoder or connector"):
        param.validate_embedding_lora_provenance()


def _routing_runtime(names: tuple[str, ...]) -> ComponentLoraRuntime:
    owners = tuple(
        ComponentLoraOwner(
            name=name,
            module_path=name,
            dtype="float32",
            max_rank=4,
            targets=(
                ComponentLoraTarget(
                    module_path="projection",
                    checkpoint_modules=(f"{name}.projection",),
                    layer_idx=0,
                    module_type="attn_dense",
                    execution_id=64 + index,
                    input_size=4,
                    output_size=4,
                ),
            ),
        )
        for index, name in enumerate(names)
    )
    runtime = ComponentLoraRuntime(owners)
    runtime.bind_batch(
        {
            task_id: [
                SimpleNamespace(
                    layer_id=0,
                    module_id=owner.targets[0].execution_id,
                    adapter_size=rank,
                    weights_in_pointer=10,
                    weights_out_pointer=20,
                    scaling_vec_pointer=0,
                )
                for owner in owners
            ]
            for task_id, rank in ((7, 2), (8, 4))
        },
        torch.float32,
    )
    return runtime


def test_mistral_builds_distinct_patch_and_connector_row_mappings() -> None:
    from tensorrt_llm._torch.models.modeling_mistral import Mistral3VLM

    model = Mistral3VLM.__new__(Mistral3VLM)
    torch.nn.Module.__init__(model)
    model._vision_tower = SimpleNamespace(config=SimpleNamespace(patch_size=2))
    model._multi_modal_projector = SimpleNamespace(
        patch_merger=SimpleNamespace(_spatial_merge_size=2)
    )
    runtime = _routing_runtime(("vision", "vision_connector"))
    model._component_lora_runtime = runtime
    model._get_image_features = Mock(return_value=torch.ones(6, 4))
    params = []
    for task_id, sizes in ((7, [[4, 4], [4, 8]]), (None, [[8, 4]]), (8, [[4, 4]])):
        params.append(
            MultimodalParams(
                multimodal_data={
                    "image": {
                        "pixel_values": torch.ones(
                            len(sizes),
                            3,
                            max(size[0] for size in sizes),
                            max(size[1] for size in sizes),
                        ),
                        "image_sizes": sizes,
                    }
                },
                component_lora_request=ComponentLoraRequest(task_id, "version")
                if task_id is not None
                else None,
            )
        )

    model._vision_forward(params)

    kwargs = model._get_image_features.call_args.kwargs
    vision = kwargs["vision_lora_params"]
    connector = kwargs["connector_lora_params"]
    assert vision["prompt_lens_cpu"].tolist() == [4, 8, 8, 4]
    assert connector["prompt_lens_cpu"].tolist() == [1, 2, 2, 1]
    assert vision[0][64]["adapter_size"].tolist() == [2, 2, 0, 4]
    assert connector[0][65]["adapter_size"].tolist() == [2, 2, 0, 4]


def test_nemotron_audio_maps_valid_subsampled_rows_and_base_padding() -> None:
    from tensorrt_llm._torch.models.modeling_nemotron_h_multimodal import NemotronHMultimodalModel

    class SoundEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = SimpleNamespace(
                _get_subsampling_output_length=lambda lengths: lengths // 2
            )
            self.forward_kwargs = None

        def forward(self, features: torch.Tensor, mask: torch.Tensor, **kwargs) -> torch.Tensor:
            self.forward_kwargs = kwargs
            return (
                torch.arange(features.shape[0] * (features.shape[1] // 2))
                .reshape(features.shape[0], -1, 1)
                .expand(-1, -1, 4)
                .float()
            )

    model = NemotronHMultimodalModel.__new__(NemotronHMultimodalModel)
    torch.nn.Module.__init__(model)
    model.model_dtype = torch.float32
    model.sound_encoder = SoundEncoder()
    model._component_lora_runtime = _routing_runtime(("audio", "audio_connector"))
    inputs = []
    for frames, valid_lengths in ((10, [10, 6]), (14, [12]), (8, [8])):
        inputs.append(
            {
                "input_audio_features": torch.ones(len(valid_lengths), frames, 4),
                "feature_attention_mask": torch.arange(frames)[None, :]
                < torch.tensor(valid_lengths)[:, None],
            }
        )

    outputs = model._encode_audio(inputs, task_ids=[7, None, 8])

    kwargs = model.sound_encoder.forward_kwargs
    for name, execution_id in (("lora_params", 64), ("connector_lora_params", 65)):
        assert kwargs[name]["prompt_lens_cpu"].tolist() == [5, 2, 3, 4, 6, 1, 4, 3]
        assert kwargs[name][0][execution_id]["adapter_size"].tolist() == [2, 0, 2, 0, 0, 0, 4, 0]
    assert [counts for _, counts in outputs] == [[5, 3], [6], [4]]
    assert [embedding.shape[0] for embedding, _ in outputs] == [8, 6, 4]
