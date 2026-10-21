# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Component execution identities and request mappings for multimodal LoRA."""

from collections.abc import Sequence
from dataclasses import replace

import torch
from torch import nn

from tensorrt_llm.bindings.internal.runtime import TaskLayerModuleConfig

from .layer import LoraLayer
from .manager import LoraManager
from .ownership import ComponentLoraOwner


def bind_component_lora(
    model: nn.Module,
    components: tuple[ComponentLoraOwner, ...],
    *,
    tp_size: int = 1,
) -> tuple[ComponentLoraOwner, ...]:
    """Assign execution IDs and bind each declared projection to its LoRA layer."""
    if tp_size <= 0:
        raise ValueError("Component LoRA requires a positive executor TP size")
    next_id = max(LoraManager.LORA_MODULE_IDS.values()) + 1
    descriptors: dict[tuple, int] = {}
    bound_layers: dict[LoraLayer, tuple[str, int, dict[int, int]]] = {}
    owner_names: set[str] = set()
    bound_components = []
    for owner in components:
        if owner.name in owner_names or not owner.targets:
            raise ValueError(
                f"LoRA component {owner.name!r} must have unique ownership and targets"
            )
        owner_names.add(owner.name)
        component = model.get_submodule(owner.module_path)
        targets = []
        for target in owner.targets:
            if target.module_type not in LoraManager.LORA_MODULE_IDS:
                raise ValueError(f"Unsupported LoRA projection {owner.name}.{target.module_type}")
            if target.layer_idx < 0 or min(target.input_size, target.output_size) <= 0:
                raise ValueError(
                    f"Invalid dimensions or layer index for {owner.name}.{target.module_path}"
                )
            split_dims = (target.in_tp_split_dim, target.out_tp_split_dim)
            replicated = target.tp_size == 1 and split_dims == (-1, -1)
            sharded = target.tp_size == tp_size and tp_size > 1 and split_dims in ((-1, 0), (1, -1))
            if not replicated and not sharded:
                raise ValueError(
                    "Component LoRA targets must be replicated TP1 or sharded across "
                    "the executor TP group with one split dimension"
                )
            if sharded and target.fused_output_sizes and target.out_tp_split_dim != 0:
                raise ValueError("Fused component LoRA targets require output TP sharding")
            if (target.in_tp_split_dim == 1 and target.input_size % tp_size) or (
                target.out_tp_split_dim == 0 and target.output_size % tp_size
            ):
                raise ValueError("Component LoRA split dimensions must be divisible by TP size")
            if target.fused_output_sizes and (
                sum(target.fused_output_sizes) != target.output_size
                or any(size <= 0 for size in target.fused_output_sizes)
                or (
                    target.out_tp_split_dim == 0
                    and any(size % tp_size for size in target.fused_output_sizes)
                )
            ):
                raise ValueError("Invalid component LoRA fused projection sizes for TP layout")
            signature = (
                owner.name,
                target.module_type,
                target.input_size,
                target.output_size,
                target.fused_output_sizes,
                target.tp_size,
                target.in_tp_split_dim,
                target.out_tp_split_dim,
                owner.dtype,
            )
            if signature not in descriptors:
                descriptors[signature] = next_id
                next_id += 1
            execution_id = descriptors[signature]
            layer = component.get_submodule(target.module_path)
            if not isinstance(layer, LoraLayer):
                raise ValueError(f"{owner.name}.{target.module_path} must select a LoraLayer")
            semantic_id = LoraManager.LORA_MODULE_IDS[target.module_type]
            if semantic_id not in layer.lora_module_types:
                raise ValueError(
                    f"{owner.name}.{target.module_path} does not implement {target.module_type}"
                )
            local_output_size = target.output_size
            if target.out_tp_split_dim == 0:
                local_output_size //= tp_size
            projection_index = layer.lora_module_types.index(semantic_id)
            if layer.output_hidden_sizes[projection_index] != local_output_size:
                raise ValueError(
                    f"Component LoRA target {owner.name}.{target.module_path} declares local "
                    f"output size {local_output_size}, but the bound LoRA layer requires "
                    f"{layer.output_hidden_sizes[projection_index]}"
                )
            layer_owner, layer_idx, module_ids = bound_layers.setdefault(
                layer, (owner.name, target.layer_idx, {})
            )
            if (layer_owner, layer_idx) != (owner.name, target.layer_idx):
                raise ValueError(
                    "A shared LoRA layer cannot belong to different component destinations"
                )
            previous_id = module_ids.setdefault(semantic_id, execution_id)
            if previous_id != execution_id:
                raise ValueError(f"Conflicting dimensions for {owner.name}.{target.module_path}")
            targets.append(replace(target, execution_id=execution_id))
        bound_components.append(replace(owner, targets=tuple(targets)))

    for layer, (_, _, module_ids) in bound_layers.items():
        if set(module_ids) != {int(module_type) for module_type in layer.lora_module_types}:
            raise ValueError("Every projection in an enrolled fused LoRA layer must be declared")
    for layer, (_, _, module_ids) in bound_layers.items():
        layer.execution_module_ids = tuple(module_ids[int(t)] for t in layer.lora_module_types)
    return tuple(bound_components)


class ComponentLoraRuntime:
    """Build eager encoder mappings from the physical row order of each component."""

    def __init__(self, components: tuple[ComponentLoraOwner, ...]) -> None:
        self.components = components
        self._destinations = {
            owner.name: {(target.layer_idx, target.execution_id) for target in owner.targets}
            for owner in components
        }
        self._peft_table: dict[int, list[TaskLayerModuleConfig]] = {}
        self._data_type: torch.dtype | None = None

    def bind_batch(
        self, peft_table: dict[int, list[TaskLayerModuleConfig]], data_type: torch.dtype
    ) -> None:
        """Retain a separate view of cache pointers until component execution finishes."""
        self._peft_table = peft_table
        self._data_type = data_type

    def parameters(self, component: str, segments: Sequence[tuple[int | None, int]]) -> dict | None:
        """Build parameters for packed rows; padding segments use a None task ID."""
        if component not in self._destinations:
            raise ValueError(f"Undeclared LoRA component {component!r}")
        if any(length < 0 for _, length in segments):
            raise ValueError("Component LoRA row counts must be nonnegative")
        segments = [(task_id, length) for task_id, length in segments if length]
        permitted = self._destinations[component]
        entries = {}
        active = set()
        for task_id, _ in segments:
            if task_id is None:
                continue
            if task_id not in self._peft_table:
                raise RuntimeError(
                    f"LoRA adapter {task_id} was not prepared before {component} execution"
                )
            for config in self._peft_table[task_id]:
                destination = (config.layer_id, config.module_id)
                if destination in permitted:
                    entries[(task_id, *destination)] = config
                    active.add(destination)
        if not active:
            return None

        params = {}
        for layer_id, module_id in sorted(active):
            ranks = []
            pointers = []
            for task_id, _ in segments:
                config = entries.get((task_id, layer_id, module_id))
                if config is None:
                    ranks.append(0)
                    pointers.extend((0, 0, 0))
                else:
                    ranks.append(config.adapter_size)
                    pointers.extend(
                        (
                            config.weights_in_pointer,
                            config.weights_out_pointer,
                            config.scaling_vec_pointer or 0,
                        )
                    )
            params.setdefault(layer_id, {})[module_id] = {
                "adapter_size": torch.tensor(ranks, dtype=torch.int32),
                "weight_pointers": torch.tensor(pointers, dtype=torch.int64),
            }
        params.update(
            host_request_types=torch.zeros(len(segments), dtype=torch.int32),
            prompt_lens_cpu=torch.tensor([length for _, length in segments], dtype=torch.int32),
            num_seqs=len(segments),
            num_contexts=len(segments),
            num_generations=0,
            data_type=self._data_type,
            use_cuda_graph_mode=False,
        )
        return params
