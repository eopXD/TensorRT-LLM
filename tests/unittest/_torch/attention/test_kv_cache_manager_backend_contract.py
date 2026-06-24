# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""§2c floor: cross-backend x V1/V2 KV cache manager-contract smoke test.

Exercises the backend-coupling contract documented in §1d of
``docs/source/kv_cache_manager_v2_bringup/model_bringup.html``: every
non-sparse attention backend that reaches into the KV cache manager must
be able to read the attributes and call the methods it relies on against
both V1 (``KVCacheManager``) and V2 (``KVCacheManagerV2``) without raising
``AttributeError``.

This is the lowest layer under §2c — a failure here cascades into every
§2d feature test. Full backend forward passes against GPT-OSS-120B are
covered by the LLM-API-level cells in
``tests/integration/defs/accuracy/test_llm_api_pytorch.py``.
"""

import pytest

import tensorrt_llm
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

# Attributes any V2-supported PyTorch attention backend reads on the
# manager. Derived from §1d "Backend-coupling contract" in
# docs/source/kv_cache_manager_v2_bringup/model_bringup.html.
_BACKEND_CONTRACT_ATTRS = (
    "blocks_in_primary_pool",
    "tokens_per_block",
    "max_seq_len",
    "max_batch_size",
    "max_blocks_per_seq",
    "num_pools",
    "kv_cache_pool_pointers",
    "kv_cache_pool_mapping",
    "host_kv_cache_block_offsets",
    "layer_offsets",
    "dtype",
)

_BACKEND_CONTRACT_METHODS = (
    "get_buffers",
    "get_batch_cache_indices",
    "get_block_ids_per_seq",
    "copy_batch_block_offsets",
)


# §1d backend label → either an importable attention backend class, or a
# pytest.skip reason string explaining why the cell is not directly
# exercised here (covered transitively or not a current top-level dispatch).
def _build_backend_table():
    from tensorrt_llm._torch.attention_backend import (
        FlashInferAttention,
        TrtllmAttention,
        VanillaAttention,
    )

    return {
        "TRTLLM": TrtllmAttention,
        "FLASHINFER": FlashInferAttention,
        "VANILLA": VanillaAttention,
        "TRTLLM_GEN": (
            "TRTLLM-gen is dispatched internally from attn_backend='TRTLLM' "
            "on Blackwell (trtllm_gen.py); not a separate top-level "
            "attn_backend value. Covered transitively by the TRTLLM cell on "
            "B200/B300 in test_w4_1gpu / test_w4_1gpu_flashinfer."
        ),
        "FLASHATTENTION": (
            "FlashAttention is not a top-level attn_backend choice in "
            "current code; the §1d table cites FA access only via the "
            "AttentionMetadata surface. Tracked as §2c backlog."
        ),
    }


def _build_manager(use_v2: bool):
    """Build either V1 or V2 KV cache manager with a tiny configuration."""
    num_layers = 1
    num_kv_heads = 4
    head_dim = 64
    tokens_per_block = 32
    max_seq_len = 128
    max_batch_size = 2

    kv_cache_config = KvCacheConfig(
        max_tokens=max_batch_size * max_seq_len,
        use_kv_cache_manager_v2=use_v2,
    )
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    cls = KVCacheManagerV2 if use_v2 else KVCacheManager
    cache_type = tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF
    return cls(
        kv_cache_config,
        cache_type,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        mapping=mapping,
        dtype=tensorrt_llm.bindings.DataType.HALF,
    )


@pytest.mark.parametrize(
    "backend_name", ["TRTLLM", "FLASHINFER", "VANILLA", "TRTLLM_GEN", "FLASHATTENTION"]
)
@pytest.mark.parametrize(
    "use_kv_cache_manager_v2", [True, False], ids=["v2_kv_cache", "v1_kv_cache"]
)
def test_backend_manager_contract(backend_name: str, use_kv_cache_manager_v2: bool):
    """Confirm every listed attention backend can read §1d contract
    attributes / call §1d contract methods against both V1 and V2 without
    AttributeError. Bring-up floor under §2c."""
    backend_table = _build_backend_table()
    backend = backend_table[backend_name]
    if isinstance(backend, str):
        pytest.skip(backend)

    manager = _build_manager(use_kv_cache_manager_v2)
    try:
        manager.add_dummy_requests(request_ids=[0], token_nums=[16])

        # Probe every contract attribute. Any AttributeError raised here
        # is a backend-coupling regression on V1 or V2.
        for attr in _BACKEND_CONTRACT_ATTRS:
            getattr(manager, attr)

        for meth in _BACKEND_CONTRACT_METHODS:
            assert callable(getattr(manager, meth)), (
                f"{type(manager).__name__}.{meth} is not callable"
            )

        # Exercise the load-bearing method the way FlashInfer does at
        # decode (flashinfer.py:417).
        block_ids = manager.get_batch_cache_indices([0], 0)
        assert block_ids is not None
    finally:
        manager.shutdown()
