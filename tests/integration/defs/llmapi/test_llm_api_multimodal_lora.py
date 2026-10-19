# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native component-LoRA serving on one B200 with decoder graphs off/on.

Requires Mistral-Small-3.1-24B-Instruct-2503 and
NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 under LLM_MODELS_ROOT.
Adapters, images, and audio are generated locally; missing checkpoints fail.
"""

import copy
import hashlib
import json
import wave
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file
from test_common.llm_data import get_checkpoint

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.peft.lora.adapter_revision import AdapterRevisionRegistry
from tensorrt_llm._torch.peft.lora.config import LoraConfig
from tensorrt_llm.executor.request import LoRARequest
from tensorrt_llm.inputs.utils import default_multimodal_input_loader
from tensorrt_llm.llmapi.llm_args import CudaGraphConfig, KvCacheConfig, MultimodalConfig

pytestmark = pytest.mark.threadleak(enabled=False)

_CHECKPOINTS = {
    "mistral": "Mistral-Small-3.1-24B-Instruct-2503",
    "omni": "NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16",
}


def _projection_shapes(family: str, config: dict) -> list[tuple[str, int, int]]:
    if family == "mistral":
        decoder, vision = config["text_config"], config["vision_config"]
        hidden, vision_hidden = decoder["hidden_size"], vision["hidden_size"]
        return [
            (
                "language_model.model.layers.0.self_attn.o_proj",
                decoder["num_attention_heads"] * decoder["head_dim"],
                hidden,
            ),
            ("vision_tower.transformer.layers.0.attention.o_proj", vision_hidden, vision_hidden),
            ("multi_modal_projector.linear_1", vision_hidden, hidden),
            ("multi_modal_projector.linear_2", hidden, hidden),
        ]
    decoder, sound = config["llm_config"], config["sound_config"]
    layer = decoder["hybrid_override_pattern"].index("*")
    hidden, vision_hidden = decoder["hidden_size"], config["vit_hidden_size"]
    return [
        (
            f"language_model.backbone.layers.{layer}.mixer.o_proj",
            decoder["num_attention_heads"] * decoder["head_dim"],
            hidden,
        ),
        (f"vision_model.radio_model.model.blocks.{layer}.attn.proj", vision_hidden, vision_hidden),
        (
            f"sound_encoder.encoder.layers.{layer}.self_attn.o_proj",
            sound["hidden_size"],
            sound["hidden_size"],
        ),
        (
            "mlp1.1",
            vision_hidden * int(1 / config["downsample_ratio"]) ** 2,
            config["projector_hidden_size"],
        ),
        ("mlp1.3", config["projector_hidden_size"], hidden),
        ("sound_projection.linear1", sound["hidden_size"], sound["projection_hidden_size"]),
        ("sound_projection.linear2", sound["projection_hidden_size"], hidden),
    ]


def _write_adapters(path: Path, specs: list[tuple[str, int, int]]) -> tuple[Path, Path]:
    generator = torch.Generator().manual_seed(701)
    weights = {}
    for module, input_size, output_size in specs:
        prefix = f"base_model.model.{module}"
        weights[f"{prefix}.lora_A.weight"] = (
            torch.randn(4, input_size, generator=generator) * 0.01
        ).to(torch.bfloat16)
        weights[f"{prefix}.lora_B.weight"] = (
            torch.randn(output_size, 4, generator=generator) * 0.01
        ).to(torch.bfloat16)
    for name, zero in (("active", False), ("zero", True)):
        directory = path / name
        directory.mkdir()
        save_file(
            {
                key: torch.zeros_like(value) if zero and ".lora_B." in key else value
                for key, value in weights.items()
            },
            directory / "adapter_model.safetensors",
        )
        (directory / "adapter_config.json").write_text(
            json.dumps(
                {
                    "peft_type": "LORA",
                    "r": 4,
                    "lora_alpha": 8,
                    "target_modules": sorted({module.rsplit(".", 1)[1] for module, _, _ in specs}),
                }
            )
        )
    return path / "active", path / "zero"


def _ordered_adapter_requests(active: Path, zero: Path) -> tuple[list[LoRARequest | None], dict]:
    # Contexts sort by task ID; keep the active peer last in both compared batches.
    with AdapterRevisionRegistry() as registry:
        for index in range(1, 65):
            zero_id, active_id = 2 * index - 1, 2 * index
            zero_snapshot = registry.resolve(zero_id, str(zero))
            active_snapshot = registry.resolve(active_id, str(active))
            if zero_snapshot.task_id < active_snapshot.task_id:
                return [
                    None,
                    LoRARequest("zero", zero_id, str(zero)),
                    LoRARequest("active", active_id, str(active)),
                ], {
                    "zero": {"external_id": zero_id, "task_id": zero_snapshot.task_id},
                    "active": {"external_id": active_id, "task_id": active_snapshot.task_id},
                }
    raise AssertionError("Could not select adapter IDs with zero before active")


def _write_media(path: Path, audio: bool) -> list[str]:
    image_path = path / "image.png"
    pixels = torch.arange(112 * 112 * 3).reshape(112, 112, 3).remainder(251).byte()
    Image.fromarray(pixels.numpy()).save(image_path)
    media = [str(image_path)]
    if audio:
        audio_path = path / "audio.wav"
        time = torch.arange(16000, dtype=torch.float32) / 16000
        signal = (torch.sin(2 * torch.pi * 300 * time) * 6000).to(torch.int16)
        with wave.open(str(audio_path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(signal.numpy().tobytes())
        media.append(str(audio_path))
    return media


def _tensor_signature(value: torch.Tensor | None) -> tuple | None:
    if value is None:
        return None
    tensor = value.detach().cpu().contiguous()
    return (
        tuple(tensor.shape),
        str(tensor.dtype),
        hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
    )


def _align_prefill_reference(
    reference: list[torch.Tensor],
    reference_packing: list[dict],
    observed_packing: list[dict],
    *,
    fixed_rows: tuple[int, ...] = (),
) -> list[torch.Tensor]:
    assert len(reference_packing) == len(observed_packing) == 1, (
        "Expected one complete prefill batch"
    )
    source, observed = reference_packing[0], observed_packing[0]
    expected_rows = list(range(len(reference)))
    for batch in (source, observed):
        assert batch["num_contexts"] == len(reference) and batch["num_generations"] == 0
        assert sorted(batch["request_order"]) == expected_rows
        assert batch["input_ids"] is not None and batch["position_ids"] is not None
    assert {key: value for key, value in source.items() if key != "request_order"} == {
        key: value for key, value in observed.items() if key != "request_order"
    }, "Decoder input tensors, positions, or sequence lengths changed between compared batches"
    physical_position = {row: position for position, row in enumerate(observed["request_order"])}
    for row in fixed_rows:
        assert source["request_order"].index(row) == physical_position[row], (
            f"Submitted row {row} changed physical position between compared batches"
        )
    return [reference[source["request_order"][physical_position[row]]] for row in expected_rows]


@pytest.mark.cpu_only
def test_prefill_reference_alignment_uses_observed_physical_positions() -> None:
    common = {
        "num_contexts": 3,
        "num_generations": 0,
        "seq_lens": (4, 4, 4),
        "input_ids": "same-packed-tokens",
        "position_ids": "same-packed-positions",
    }
    mixed = [{**common, "request_order": (0, 2, 1)}]
    base = _align_prefill_reference(
        [torch.tensor(10), torch.tensor(20), torch.tensor(30)],
        [{**common, "request_order": (0, 1, 2)}],
        mixed,
    )
    repeated = _align_prefill_reference(
        [torch.tensor(20), torch.tensor(30), torch.tensor(10)],
        [{**common, "request_order": (2, 0, 1)}],
        mixed,
    )
    assert [value.item() for value in base] == [10, 30, 20]
    assert [value.item() for value in repeated] == [10, 30, 20]
    with pytest.raises(AssertionError, match="Submitted row 2 changed physical position"):
        _align_prefill_reference(
            base, [{**common, "request_order": (0, 1, 2)}], mixed, fixed_rows=(2,)
        )
    with pytest.raises(AssertionError, match="Decoder input tensors"):
        _align_prefill_reference(base, mixed, [{**mixed[0], "input_ids": "different-tokens"}])


@pytest.mark.parametrize("graphs", [False, True], ids=["eager", "graphs"])
@pytest.mark.parametrize("family", ["mistral", "omni"])
def test_native_multimodal_composite_lora(
    family: str, graphs: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    model_path = get_checkpoint(_CHECKPOINTS[family])
    config = json.loads((Path(model_path) / "config.json").read_text())
    specs = _projection_shapes(family, config)
    active, zero = _write_adapters(tmp_path, specs)
    media = _write_media(tmp_path, audio=family == "omni")
    adapter_requests, adapter_ids = _ordered_adapter_requests(active, zero)
    with LLM(
        model=model_path,
        backend="pytorch",
        dtype="bfloat16",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        trust_remote_code=True,
        max_batch_size=3,
        max_num_tokens=8192,
        max_seq_len=12288,
        encoder_max_batch_size=6,
        encoder_max_num_tokens=8192,
        # Submission includes checkpoint loading; measured packing below must still match.
        batch_wait_timeout_ms=5000,
        enable_chunked_prefill=False,
        kv_cache_config=KvCacheConfig(
            free_gpu_memory_fraction=0.3, enable_block_reuse=False, mamba_ssm_cache_dtype="float32"
        ),
        cuda_graph_config=CudaGraphConfig(batch_sizes=[1, 2, 3], enable_padding=True)
        if graphs
        else None,
        multimodal_config=MultimodalConfig(
            video_pruning_rate=0.0 if family == "omni" else None,
            encoder_cache_max_bytes=0,
        ),
        lora_config=LoraConfig(
            enable_multimodal_lora=True,
            lora_dir=[str(active)],
            lora_target_modules=["attn_q", "attn_k", "attn_v", "attn_dense"],
            max_lora_rank=4,
            max_loras=2,
            max_cpu_loras=2,
        ),
    ) as llm:
        prompt = default_multimodal_input_loader(
            tokenizer=llm.tokenizer,
            model_dir=model_path,
            model_type=config["model_type"],
            modality="image_audio" if family == "omni" else "image",
            prompts=["Describe the supplied media briefly. /no_think"],
            media=[media],
            image_data_format="pt",
            device="cpu",
        )[0]
        executor = llm._executor.engine
        engine = executor.model_engine
        expected_components = {"vision", "vision_connector"}
        if family == "omni":
            expected_components.update(("audio", "audio_connector"))
        assert {owner.name for owner in engine._lora_components} == expected_components
        active_components = set()
        handles = []

        def observe_component(
            name: str,
        ) -> Callable[[torch.nn.Module, tuple, torch.Tensor | None], None]:
            def hook(module: torch.nn.Module, inputs: tuple, output: torch.Tensor | None) -> None:
                if isinstance(output, torch.Tensor) and torch.count_nonzero(output).item():
                    active_components.add(name)

            return hook

        for owner in engine._lora_components:
            component = engine.model.get_submodule(owner.module_path)
            for path in {target.module_path for target in owner.targets}:
                handles.append(
                    component.get_submodule(path).register_forward_hook(
                        observe_component(owner.name)
                    )
                )

        def context_logits(
            requests: list[LoRARequest | None],
        ) -> tuple[list[torch.Tensor], list[dict]]:
            packing = []
            model_forward = engine._runner.model_forward

            def record_forward(*, is_dummy: bool, **kwargs) -> torch.Tensor | dict:
                metadata = kwargs["attn_metadata"]
                if not is_dummy and metadata.num_contexts:
                    clients = {
                        request.py_request_id: request.py_client_id
                        for request in executor.active_requests
                    }
                    packing.append(
                        {
                            "request_order": tuple(
                                clients[request_id] for request_id in metadata.request_ids
                            ),
                            "num_contexts": metadata.num_contexts,
                            "num_generations": metadata.num_generations,
                            "seq_lens": tuple(metadata.seq_lens.tolist()),
                            "input_ids": _tensor_signature(kwargs.get("input_ids")),
                            "position_ids": _tensor_signature(kwargs.get("position_ids")),
                        }
                    )
                return model_forward(is_dummy=is_dummy, **kwargs)

            with patch.object(engine._runner, "model_forward", side_effect=record_forward):
                outputs = llm.generate(
                    [copy.deepcopy(prompt) for _ in requests],
                    SamplingParams(
                        max_tokens=4, min_tokens=4, temperature=0.0, return_context_logits=True
                    ),
                    lora_request=requests,
                    use_tqdm=False,
                )
            assert all(len(output.outputs[0].token_ids) == 4 for output in outputs)
            assert all(output.context_logits is not None for output in outputs)
            assert packing, "No decoder prefill packing was observed"
            row_by_client = {output.request_id: index for index, output in enumerate(outputs)}
            for batch in packing:
                batch["request_order"] = tuple(
                    row_by_client[client] for client in batch["request_order"]
                )
            return [
                output.context_logits[-1].detach().float().cpu().clone() for output in outputs
            ], packing

        try:
            with patch.object(
                engine.cuda_graph_runner, "replay", wraps=engine.cuda_graph_runner.replay
            ) as replay:
                context_logits(adapter_requests[1:])
                all_base, all_base_packing = context_logits([None] * 3)
                reference_requests = [None, None, adapter_requests[2]]
                base, base_packing = context_logits(reference_requests)
                repeated, repeated_packing = context_logits(reference_requests)
                active_components.clear()
                replay.reset_mock()
                mixed, mixed_packing = context_logits(adapter_requests)
                raw_base, raw_repeated = base, repeated
                base = _align_prefill_reference(base, base_packing, mixed_packing, fixed_rows=(2,))
                repeated = _align_prefill_reference(
                    repeated, repeated_packing, mixed_packing, fixed_rows=(2,)
                )
                all_base = _align_prefill_reference(all_base, all_base_packing, mixed_packing)
                assert mixed_packing[0]["request_order"] == (0, 1, 2)
                noise = max(
                    (first - second).abs().max().item() for first, second in zip(base, repeated)
                )
                diagnostics = {
                    "family": family,
                    "graphs": graphs,
                    "submitted_adapters": ["base", "zero", "active"],
                    "reference_adapters": ["base", "base", "active"],
                    "adapter_ids": adapter_ids,
                    "all_base_packing": all_base_packing,
                    "base_packing": base_packing,
                    "repeat_packing": repeated_packing,
                    "mixed_packing": mixed_packing,
                    "noise": noise,
                    "base_zero_atol": max(0.002, 4 * noise),
                    "active_components": sorted(active_components),
                    "decoder_replays": replay.call_count,
                    "rows": [
                        {
                            "logical_row": index,
                            "max_abs_delta": (actual - reference).abs().max().item(),
                            "rms_delta": (actual - reference).square().mean().sqrt().item(),
                            "repeat_noise": (reference - repeated[index]).abs().max().item(),
                        }
                        for index, (actual, reference) in enumerate(zip(mixed, base))
                    ],
                }
                receipt = tmp_path / "numerical-diagnostics.json"
                receipt.write_text(json.dumps(diagnostics, indent=2))
                torch.save(
                    {
                        "raw_base": raw_base,
                        "raw_repeated": raw_repeated,
                        "aligned_base": base,
                        "aligned_repeated": repeated,
                        "aligned_all_base": all_base,
                        "mixed": mixed,
                    },
                    tmp_path / "prefill-logits.pt",
                )
                print(f"Numerical diagnostics: {receipt}")
                assert active_components == expected_components
                for index in range(3):
                    row = diagnostics["submitted_adapters"][index]
                    label = f"Submitted row {index} ({row}), noise={noise}; {receipt}"
                    torch.testing.assert_close(
                        mixed[index],
                        base[index],
                        rtol=0,
                        atol=max(0.002, 4 * noise),
                        msg=lambda message: f"{label}\n{message}",
                    )
                delta = (mixed[2] - all_base[2]).abs().max().item()
                assert delta > max(0.005, 10 * noise), (
                    f"adapter delta={delta}, repeated-reference noise={noise}"
                )
                assert bool(replay.call_count) is graphs
        finally:
            for handle in handles:
                handle.remove()
