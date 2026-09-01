# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The KV connector prefix against a *real* ``KVCacheManagerV2``.

``test_kv_connector_v2_prefix.py`` drives the same code against a stub cache.
That is what makes it fast and exhaustive, and it is the right place for the
arithmetic and the ask-once rules -- but a stub cannot show that a real
``_KVCache`` survives the sequence: that ``resize`` finds real pages for the
offered prefix, that ``history_length`` really moves, that the grow the chunked
path needs succeeds against real pools, and that the page slots handed to the
connector are distinct and real.

The engine-level suite cannot show the scheduling-order claims either, because
whether a request is dropped after being prepared depends on which pass it
reaches the scheduler in -- a race. Preparation and delivery are therefore
driven directly here: ``prepare_context`` plus ``resize_context`` is one
scheduling pass, and ``prepare_resources`` is the batch actually running.

These tests allocate device memory pools.
"""

import gc

import pytest
import torch

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX

DataType = tensorrt_llm.bindings.DataType
CacheType = tensorrt_llm.bindings.internal.batch_manager.CacheType

# These build a real manager, which allocates device pools. The directory is
# listed in the GPU-less l0_cpu stage, so the requirement is declared rather
# than left to fail at `torch.cuda.init()`.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="allocates real KV cache pools"
)

TOKENS_PER_BLOCK = 32
PROMPT_LEN = 96
OFFER_TOKENS = 32


class FakeConnectorManager:
    """Records what the prefix path tells the connector, in order."""

    def __init__(self, num_matched=OFFER_TOKENS, load_async=False):
        self.num_matched = num_matched
        self.load_async = load_async
        self.queries = []
        self.commits = []
        self.allocs = []

    def query_num_new_matched_tokens(self, request, num_computed_tokens):
        self.queries.append((request.py_request_id, num_computed_tokens))
        return self.num_matched, self.load_async

    def commit_new_matched_tokens(self, request, num_tokens, load_kv_async):
        self.commits.append((request.py_request_id, num_tokens, load_kv_async))
        request.py_num_connector_matched_tokens = num_tokens

    def should_add_sequence(self, request):
        return True

    def update_state_after_alloc(self, request, block_ids):
        self.allocs.append((request.py_request_id, list(block_ids)))

    def build_scheduler_output(self, scheduled_batch, kv_cache_manager):
        pass


def make_manager(connector, **overrides):
    kwargs = dict(
        kv_cache_config=KvCacheConfig(max_tokens=2048, enable_block_reuse=True),
        kv_cache_type=CacheType.SELF,
        num_layers=2,
        num_kv_heads=4,
        head_dim=64,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=256,
        max_batch_size=4,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=DataType.HALF,
        vocab_size=32000,
        kv_connector_manager=connector,
    )
    kwargs.update(overrides)
    return KVCacheManagerV2(**kwargs)


def make_request(request_id=1, prompt_len=PROMPT_LEN):
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=4,
        input_tokens=list(range(prompt_len)),
        sampling_config=SamplingConfig(1),
        is_streaming=False,
    )


def schedule(manager, request, num_tokens=None):
    """One scheduling pass: prepare the cache and size it for the chunk."""
    assert manager.prepare_context(request)
    if num_tokens is None:
        num_tokens = request.context_remaining_length
    return manager.resize_context(request, num_tokens)


def run(manager, *requests):
    """One ``prepare_resources``, i.e. the requests reached the final batch."""
    batch = ScheduledRequests()
    for request in requests:
        batch.append_context_request(request)
    manager.prepare_resources(batch)
    return batch


@pytest.fixture
def connector():
    return FakeConnectorManager()


@pytest.fixture
def manager(connector):
    torch.cuda.init()
    gc.collect()
    torch.cuda.empty_cache()
    mgr = make_manager(connector)
    yield mgr
    mgr.shutdown()
    del mgr
    gc.collect()
    torch.cuda.empty_cache()


def test_a_request_dropped_before_the_batch_is_never_asked(manager, connector):
    """The whole point of asking in ``prepare_resources``.

    A request can be prepared and sized and then lose the token budget, fail
    multimodal alignment, or be dropped when the batch cannot be queued. On the
    V1 manager none of that can strand an offer, because V1 asks after all of
    it. Neither can it here.
    """
    request = make_request()

    assert schedule(manager, request)

    assert connector.queries == []
    assert connector.commits == []
    assert request.context_current_position == 0


def test_offer_is_backed_by_real_pages(manager, connector):
    """Read capacity, history and page slots back off the cache the forward
    pass would use -- the difference between "resize was called correctly" and
    "the offered prefix is resident"."""
    request = make_request()
    assert schedule(manager, request)

    run(manager, request)

    kv_cache = manager.kv_cache_map[request.py_request_id]
    assert request.context_current_position == OFFER_TOKENS
    assert kv_cache.history_length == OFFER_TOKENS
    assert kv_cache.capacity >= PROMPT_LEN
    assert kv_cache.is_active

    assert connector.commits == [(request.py_request_id, OFFER_TOKENS, False)]
    assert len(connector.allocs) == 1

    _, page_indices = connector.allocs[0]
    assert len(page_indices) >= PROMPT_LEN // TOKENS_PER_BLOCK
    assert all(index != BAD_PAGE_INDEX for index in page_indices)
    assert len(set(page_indices)) == len(page_indices)


def test_the_unchunked_path_allocates_nothing_for_the_prefix(manager, connector):
    """``resize_context`` already covered the whole prompt, so honouring the
    offer only moves the request's start."""
    request = make_request()
    assert schedule(manager, request)
    kv_cache = manager.kv_cache_map[request.py_request_id]
    before = kv_cache.capacity

    run(manager, request)

    assert kv_cache.capacity == before
    assert request.context_current_position + request.context_chunk_size == PROMPT_LEN


def test_a_chunked_offer_beyond_the_chunk_grows_and_shifts(manager, connector):
    """Chunked prefill keeps its per-chunk allocation, so an offer past the
    chunk has to grow the cache before it can be honoured."""
    connector.num_matched = 64
    request = make_request()
    request.context_chunk_size = TOKENS_PER_BLOCK
    assert schedule(manager, request, num_tokens=TOKENS_PER_BLOCK)
    kv_cache = manager.kv_cache_map[request.py_request_id]
    assert kv_cache.capacity < 64 + TOKENS_PER_BLOCK

    run(manager, request)

    assert request.context_current_position == 64
    assert request.context_chunk_size == TOKENS_PER_BLOCK
    assert kv_cache.capacity >= 64 + TOKENS_PER_BLOCK
    assert kv_cache.history_length == 64
    assert connector.commits == [(request.py_request_id, 64, False)]


def test_a_second_pass_reports_one_allocation(manager, connector):
    """An asynchronously loaded request re-enters on its first context chunk,
    with the same pages and nothing left to load."""
    request = make_request()
    assert schedule(manager, request)

    run(manager, request)
    run(manager, request)

    assert len(connector.queries) == 1
    assert len(connector.commits) == 1
    assert len(connector.allocs) == 1


def test_freeing_the_allocation_makes_the_request_askable_again(manager, connector):
    """V1-faithful: a destructive pause replays ``addSequence`` and with it the
    query, because the pages the first answer described are gone."""
    request = make_request()
    assert schedule(manager, request)
    run(manager, request)
    manager.free_resources(request)

    request.reset_for_recompute()
    assert schedule(manager, request)
    run(manager, request)

    assert len(connector.queries) == 2


def test_no_connector_leaves_prepare_resources_inert(connector):
    torch.cuda.init()
    mgr = make_manager(connector, kv_connector_manager=None)
    try:
        request = make_request()
        assert schedule(mgr, request)
        run(mgr, request)
        assert request.context_current_position == 0
        assert connector.queries == []
    finally:
        mgr.shutdown()
        del mgr
        gc.collect()
        torch.cuda.empty_cache()
