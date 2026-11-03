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
"""The FP4 MLA cache layout declared as persistent-state regions.

* The FP4 latent keeps K, K block scales and the V sidecars derived from them
  for the whole history. A request resumes from another's latent only on a
  complete 16-token tile, which the block size makes a block boundary.
* The FP8 latent of the fallback layers needs a pool of its own; a window that
  never evicts gives it a distinct life cycle while it is sized as full history.
* The HP ring keeps the BF16 tokens of the incomplete tile plus the speculative
  rewind of each FP4 layer. A token's ring slot follows from its position, so
  the history length must follow every chunk.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, List, Optional, Sequence, Tuple

from tensorrt_llm._torch.disaggregation.resource.page import MapperKind
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import (
    _get_local_mamba_cache_layout,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.regions import (
    AllocationUnit,
    HandOff,
    Plane,
    Region,
    RegionContext,
    RegionLayer,
    RegionLayout,
    RegionLifeCycle,
    RegionProvider,
    RegionTransfer,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import LayerId

from .cache_manager import Fp4MlaKVCacheManagerV2, Fp4MlaV2CacheLayoutPolicy

if TYPE_CHECKING:
    import torch

    from tensorrt_llm.llmapi.llm_args import KvCacheConfig

FP4_LATENT = "fp4_mla_latent"
FP8_LATENT = "fp8_mla_latent"
HP_RING = "fp4_mla_hp_ring"


def uses_fp4_mla_layout(dtype: DataType, kv_cache_type: CacheTypeCpp, pretrained_config) -> bool:
    """Whether an MLA latent cache of this dtype is stored in the FP4 MLA layout."""
    return (
        dtype == DataType.NVFP4
        and kv_cache_type == CacheTypeCpp.SELFKONLY
        and getattr(pretrained_config, "kv_lora_rank", None) is not None
        and getattr(pretrained_config, "qk_rope_head_dim", None) is not None
    )


def model_uses_fp4_mla_layout(model_config) -> bool:
    """Whether a model config's quantization selects the FP4 MLA layout."""
    config = getattr(model_config, "pretrained_config", None)
    quant_config = getattr(model_config, "quant_config", None)
    return (
        getattr(config, "kv_lora_rank", None) is not None
        and getattr(config, "qk_rope_head_dim", None) is not None
        and quant_config is not None
        and quant_config.quant_mode.has_fp4_kv_cache()
    )


def _token_blocks(window: Optional[int]) -> RegionLifeCycle:
    return RegionLifeCycle(AllocationUnit.TOKEN_BLOCK, HandOff.BLOCK_BOUNDARIES, window=window)


def _fp4_latent(manager, layers: List[int]) -> Region:
    windows = {manager.max_attention_window_vec[layer] for layer in layers}
    if len(windows) != 1:
        raise ValueError(f"FP4 MLA layers must share one attention window, got {sorted(windows)}.")
    tokens_per_block = manager.tokens_per_block

    def planes(layer: int) -> Tuple[Plane, ...]:
        result = [
            Plane(role, manager.get_layer_bytes_per_token(layer, role) * tokens_per_block)
            for role in manager._get_buffer_roles_for_layer(layer)
        ]
        if not manager.fp4_mla_context_only_cache:
            result.append(Plane(Role.MLA_V_SCALE, manager._v_scale_bytes_per_page()))
        if manager.mla_v_head_dim is not None:
            result.append(Plane(Role.MLA_V_PACKED, manager._v_packed_bytes_per_page()))
        return tuple(result)

    policy = Fp4MlaV2CacheLayoutPolicy
    return Region(
        FP4_LATENT,
        _token_blocks(windows.pop()),
        tuple(
            RegionLayer(
                layer,
                planes(layer),
                bytes_per_token=manager.get_layer_bytes_per_token(layer, Role.ALL),
            )
            for layer in layers
        ),
        transfer=RegionTransfer(
            mapper_kinds={
                role: kind
                for role, kind in policy.DISAGG_ROLE_MAPPER_KINDS.items()
                if role != Role.MLA_HP_TAIL
            },
            rebuilt_roles=policy.DISAGG_IGNORED_ROLES,
        ),
        resumes_at_block_boundaries=True,
    )


def _fp8_latent(manager, layers: List[int]) -> Region:
    never_evicting = (
        manager.max_seq_len + manager.num_extra_kv_tokens + manager._kv_reserve_draft_tokens + 1
    )
    return Region(
        FP8_LATENT,
        _token_blocks(never_evicting),
        tuple(
            RegionLayer(
                layer,
                (
                    Plane(
                        Role.KEY,
                        manager._fp8_mla_bytes_per_token(layer) * manager.tokens_per_block,
                    ),
                ),
                bytes_per_token=manager.get_layer_bytes_per_token(layer, Role.ALL),
            )
            for layer in layers
        ),
        transfer=RegionTransfer(mapper_kinds={Role.KEY: MapperKind.NHD}),
    )


def _hp_ring(manager, layers: List[int]) -> Region:
    ring_size = manager._fp4_mla_hp_pool_size
    return Region(
        HP_RING,
        _token_blocks(ring_size),
        tuple(
            RegionLayer(
                layer,
                (Plane(Role.MLA_HP_TAIL, manager._hp_bytes_per_page(layer)),),
                bytes_per_token=manager.get_layer_bytes_per_token(layer, Role.MLA_HP_TAIL),
            )
            for layer in layers
        ),
        storage_index=1,
        transfer=RegionTransfer(
            mapper_kinds={
                Role.MLA_HP_TAIL: Fp4MlaV2CacheLayoutPolicy.DISAGG_ROLE_MAPPER_KINDS[
                    Role.MLA_HP_TAIL
                ]
            },
            role_layouts=manager._fp4_mla_disagg_role_layouts(),
        ),
        positioned_by_history=True,
    )


def fp4_mla_regions(manager, *, include_latent: bool = True) -> List[Region]:
    """Declare the FP4 MLA regions of ``manager``'s rank-local MLA layers.

    ``manager`` must have been configured by
    ``Fp4MlaV2CacheLayoutPolicy.configure_manager``. The FP4 and FP8 layer
    bookkeeping the policy's accessors read is recorded on it here. Without
    ``include_latent``, the FP4 latent stays in the manager's own layer
    configs and only the FP8 latent and the HP ring are declared.
    """
    fp8_layers = manager._fp8_mla_local_layer_indices()
    fp4_layers = manager._fp4_mla_local_layer_indices()
    if not fp4_layers:
        raise ValueError("FP4 MLA V2 layout requires at least one local attention layer.")
    manager._fp8_mla_manager_layer_ids = [LayerId(layer) for layer in fp8_layers]
    manager._fp4_mla_local_to_compact = {layer: compact for compact, layer in enumerate(fp4_layers)}
    manager._fp4_mla_compact_to_local = fp4_layers
    manager._cache_manager_layer_ids = [LayerId(layer) for layer in fp4_layers]

    regions = []
    if include_latent:
        regions.append(_fp4_latent(manager, fp4_layers))
    if fp8_layers:
        regions.append(_fp8_latent(manager, fp8_layers))
    regions.append(_hp_ring(manager, fp4_layers))
    return regions


class Fp4MlaRegionProvider(RegionProvider):
    """Stores an MLA latent cache in the FP4 MLA layout.

    Args:
        context: The construction arguments of the manager it serves.
    """

    def __init__(self, context: RegionContext) -> None:
        self._context = context
        self._manager = None

    def configure(self, manager, kv_cache_config: "KvCacheConfig") -> "KvCacheConfig":
        self._manager = manager
        context = self._context
        return Fp4MlaV2CacheLayoutPolicy.configure_manager(
            manager,
            kv_cache_config,
            context.kv_cache_type,
            dtype=context.dtype,
            tokens_per_block=context.tokens_per_block,
            head_dim=context.head_dim,
            pretrained_config=context.pretrained_config,
            spec_config=context.spec_config,
            is_disagg=context.is_disagg,
            mapping=context.mapping,
        )

    def declare_regions(self) -> Sequence[Region]:
        return fp4_mla_regions(self._manager)

    def on_layout(self, layout: RegionLayout) -> None:
        self._manager._hp_manager_layer_ids = list(layout.layer_ids(HP_RING))

    def finalize(self, manager) -> None:
        Fp4MlaV2CacheLayoutPolicy.finalize_manager(manager)

    def iter_cache_views(self) -> Iterable["torch.Tensor"]:
        # Roles have their own dtypes and page geometries, so each physical
        # pool is scanned once through a page-major view.
        return self._manager._iter_fp4_mla_physical_pool_views()

    @property
    def context_only(self) -> bool:
        # A disaggregated context worker keeps no V scales or packed V.
        return self._manager.fp4_mla_context_only_cache


class Fp4MlaHybridCacheManager(MambaHybridCacheManagerV2):
    """Hybrid recurrent-state manager whose MLA latent cache may use the FP4 MLA layout.

    The FP4 MLA regions are stored when the KV cache dtype selects them (see
    ``uses_fp4_mla_layout``); otherwise the attention layers keep the
    manager's own layout.
    """

    def _create_region_providers(self, context: RegionContext) -> Sequence[RegionProvider]:
        providers = list(super()._create_region_providers(context))
        if uses_fp4_mla_layout(context.dtype, context.kv_cache_type, context.pretrained_config):
            providers.append(Fp4MlaRegionProvider(context))
        return providers

    @classmethod
    def _attention_region_cost(
        cls, model_config, mapping: Mapping, **kwargs
    ) -> Optional[Tuple[int, int]]:
        if not model_uses_fp4_mla_layout(model_config):
            return super()._attention_region_cost(model_config, mapping, **kwargs)
        spec_config = kwargs.get("spec_config")
        _, _, local_attention_layers = _get_local_mamba_cache_layout(
            model_config,
            mapping,
            spec_config=spec_config,
            is_draft=kwargs.get("is_draft", False),
            use_separate_draft_kv_cache=kwargs.get("use_separate_draft_kv_cache", False),
        )
        return Fp4MlaKVCacheManagerV2.get_cache_size_per_token(
            model_config,
            mapping,
            num_layers=local_attention_layers,
            tokens_per_block=kwargs["tokens_per_block"],
            max_batch_size=kwargs["max_batch_size"],
            spec_config=spec_config,
            is_disagg=kwargs.get("is_disagg", False),
        )


__all__ = [
    "FP4_LATENT",
    "FP8_LATENT",
    "HP_RING",
    "Fp4MlaHybridCacheManager",
    "Fp4MlaRegionProvider",
    "fp4_mla_regions",
    "model_uses_fp4_mla_layout",
    "uses_fp4_mla_layout",
]
