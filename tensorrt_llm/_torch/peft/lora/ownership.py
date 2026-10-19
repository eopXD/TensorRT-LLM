# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass


@dataclass(frozen=True)
class DecoderLoraOwner:
    """Decoder module path and checkpoint prefixes before the layer index.

    HF prefixes exclude the optional ``base_model.model`` transport prefix.
    An empty module path selects the text model itself.
    """

    module_path: str = ""
    hf_prefixes: tuple[str, ...] = (
        "layers",
        "model.layers",
        "backbone.layers",
        "language_model.layers",
        "language_model.model.layers",
        "language_model.backbone.layers",
        "transformer.layers",
        "transformer.h",
        "gpt_neox.layers",
        "decoder.layers",
        "model.decoder.layers",
    )
    nemo_prefixes: tuple[str, ...] = (
        "model.layers",
        "model.language_model.encoder.layers",
        "model.decoder.layers",
        "language_model.encoder.layers",
        "decoder.layers",
    )


@dataclass(frozen=True)
class ComponentLoraTarget:
    """One bound projection and its accepted HF checkpoint module names.

    Checkpoint names exclude ``base_model.model`` and the LoRA tensor suffix.
    Dimensions describe the unsharded weights; replicated targets use TP size one.
    """

    module_path: str
    checkpoint_modules: tuple[str, ...]
    layer_idx: int
    module_type: str
    execution_id: int
    input_size: int
    output_size: int
    fused_output_sizes: tuple[int, ...] = ()
    tp_size: int = 1
    in_tp_split_dim: int = -1
    out_tp_split_dim: int = -1


@dataclass(frozen=True)
class ComponentLoraOwner:
    """An enrolled encoder or connector with concrete projection dimensions."""

    name: str
    module_path: str
    targets: tuple[ComponentLoraTarget, ...]
    dtype: str
    max_rank: int
    hf_prefixes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ComponentLoraRequest:
    """Immutable adapter version and component contents selected by a request."""

    task_id: int | None
    revision: str
    component_fingerprints: tuple[tuple[str, str], ...] = ()
