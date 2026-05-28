# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Reproducer for MAX_UTILIZATION scheduler vs KVCacheManager admission-gap divergence.

Background
----------
Under MAX_UTILIZATION + high concurrency + short ISL/OSL (e.g. bielik_11b_v2.2 bench
at maxbs=512, ISL/OSL=128/128, L40S), the C++ KVCacheManager crashes in
WindowBlockManager::allocateBlock with "No free blocks left" at
cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp:2365.

Hypotheses under test
---------------------
B. Admission gap: the scheduler's snapshot of free blocks (taken once per pass in
   start_scheduling) does not see admission-time context block reservations made
   inside the same pass. The cumulative `num_scheduled_blocks` accumulator in
   `MaxUtilizationScheduledBlocksManager` (scheduler.py:1234-1298) should plug this
   gap, but only if `get_needed_blocks_one_step` returns the same number of blocks
   that `addSequenceBatch` + `addToken` actually consume.

C. No per-step revalidation: between scheduler decision and `add_token` allocation,
   no defensive check exists in resource_manager.py:1025-1054. Any off-by-one in the
   predictor is unrecoverable — `TLLM_CHECK_WITH_INFO` at kvCacheManager.cpp:2365
   aborts the process.

Approach
--------
Replace the real C++ KVCacheManager with a Python double (`FaithfulKVCacheManager`)
that mirrors its allocation arithmetic verbatim:
  - get_needed_blocks_one_step: kvCacheManager.cpp:3347-3427
  - start_scheduling snapshot:  kvCacheManager.cpp:3167-3174
  - addSequenceBatch alloc:     kvCacheManager.cpp:1984-2041 (ceil(promptLen / TPB))
  - addToken alloc:             kvCacheManager.cpp:2204-2210 ((numTokens-1) % TPB == 0)

Then drive the real `PyCapacityScheduler` with MAX_UTILIZATION through scenarios
that match the failing benchmark's per-step admission + decode pattern.

Outcomes
--------
- If a scenario asserts "No free blocks left" or "OVER-ADMISSION", the scheduler-side
  accounting is provably wrong and the bug is reproduced at unit-test latency
  (no GPU, no model, no forward pass).
- If all scenarios pass, the Python scheduler + faithful predictor accounting are
  internally consistent. That narrows the bug to either:
    (a) a real divergence between the C++ `getNeededBlocksOneStep` and the
        C++ `addSequenceBatch`/`addToken` allocation (the formulas this mock
        mirrors), or
    (b) a different path: chunked-prefill non-first-chunk (predictor returns 0,
        line 3426), VSWA cross-pool contention, draft tokens added after schedule,
        or the actual `mSchedulingNumFreeBlocks` snapshot semantics.

The test reports the divergence with exact predicted-vs-actual counts so the next
investigation step has concrete evidence to chase.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import pytest

from tensorrt_llm._torch.pyexecutor.llm_request import (LlmRequest,
                                                        LlmRequestState,
                                                        SamplingConfig)
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import PyCapacityScheduler
from tensorrt_llm.llmapi.llm_args import CapacitySchedulerPolicy


# ---------------------------------------------------------------------------
# Request factory (lifted from tests/unittest/_torch/executor/test_py_scheduler.py)
# ---------------------------------------------------------------------------

def _make_request(request_id: int,
                  prompt_len: int = 128,
                  max_new_tokens: int = 128,
                  beam_width: int = 1,
                  state: LlmRequestState = LlmRequestState.CONTEXT_INIT) -> LlmRequest:
    req = LlmRequest(
        request_id=request_id,
        max_new_tokens=max_new_tokens,
        input_tokens=list(range(prompt_len)),
        sampling_config=SamplingConfig(beam_width),
        is_streaming=False,
        draft_tokens=None,
    )
    req.state = state
    return req


# ---------------------------------------------------------------------------
# Faithful KVCacheManager double
# ---------------------------------------------------------------------------

@dataclass
class _PrefixReuseSummary:
    reusable_blocks_allocated: int = 0
    reusable_blocks_all: int = 0
    first_new_block: Optional[object] = None


@dataclass
class _KVCacheStats:
    num_free_blocks_per_window_size: dict = field(default_factory=dict)


class FaithfulKVCacheManager:
    """Python double that mirrors C++ KVCacheManager allocation arithmetic.

    Only models the single-window, non-reuse, non-VSWA, non-draft-token path —
    matching the bielik_11b_v2.2 config (default attention, no SWA, no MTP).
    """

    def __init__(self,
                 total_blocks: int,
                 tokens_per_block: int,
                 window_size: int = 128,
                 chunk_size: int = 8192):
        self._total_blocks = total_blocks
        self._tokens_per_block = tokens_per_block
        self._chunk_size = chunk_size
        self._free_blocks = total_blocks
        self._allocated_per_seq: dict[int, int] = {}
        self._num_tokens_per_seq: dict[int, int] = {}
        self._snapshot_free_blocks = total_blocks  # set by start_scheduling()

        # Scheduler-visible interface
        self.max_attention_window_vec = [window_size]
        self.is_variable_window = False
        self.enable_block_reuse = False

        # Trace points
        self.last_predicted_per_pass: dict[int, int] = {}
        self.last_actual_per_pass: dict[int, int] = {}

    # === Scheduler-side interface (mirrors C++ KVCacheManager) ============

    def get_kv_cache_stats(self) -> _KVCacheStats:
        return _KVCacheStats(num_free_blocks_per_window_size={
            ws: self._free_blocks
            for ws in self.max_attention_window_vec
        })

    def start_scheduling(self) -> None:
        # Mirrors KVCacheManager::startScheduling at kvCacheManager.cpp:3167-3174:
        # mSchedulingNumFreeBlocks = current free blocks (snapshot for this pass).
        self._snapshot_free_blocks = self._free_blocks

    def scheduling_has_free_blocks(self, scheduled_total: int,
                                   window_size: int) -> bool:
        return scheduled_total <= self._snapshot_free_blocks

    def scheduling_remove_sequence(self, req_id: int) -> None:
        # In C++, this restores mSchedulingNumFreeBlocks for a paused req.
        # The Python scheduler does its own bookkeeping via num_scheduled_blocks,
        # so the snapshot needn't move here.
        pass

    def get_needed_blocks_one_step(self,
                                   req: LlmRequest,
                                   two_step_lookahead: bool,
                                   window_size: int) -> int:
        # Verbatim translation of kvCacheManager.cpp:3347-3427.
        if req.is_context_init_state and req.is_first_context_chunk:
            prompt_len = req.prompt_len
            beam = req.sampling_config.beam_width
            prompt_cache_len = min(prompt_len, window_size + self._chunk_size)
            num_shared = prompt_cache_len // self._tokens_per_block
            num_unshared_tokens = prompt_cache_len % self._tokens_per_block
            num_unshared = math.ceil(
                num_unshared_tokens / self._tokens_per_block) * beam
            return num_shared + num_unshared

        if req.is_generation_in_progress_state:
            num_curr = self._num_tokens_per_seq.get(req.py_request_id, 0)
            tokens_per_step = 1  # no draft tokens in this repro
            max_to_add = (2 if two_step_lookahead else 1) * tokens_per_step
            num_next = num_curr + max_to_add
            curr_blocks = math.ceil(num_curr / self._tokens_per_block)
            next_blocks = math.ceil(num_next / self._tokens_per_block)
            return (next_blocks - curr_blocks) * req.sampling_config.beam_width

        # CONTEXT_INIT but not first chunk → C++ returns 0 (line 3426). This is one of
        # the suspected divergence sites: addToken can still allocate during subsequent
        # chunks if a boundary is crossed. For this single-chunk reproducer the
        # entire prompt fits in chunk_size so we never hit this branch.
        return 0

    def get_remaining_blocks_to_completion(self, req: LlmRequest,
                                           window_size: int) -> int:
        return 0  # unused by MAX_UTILIZATION

    def get_max_resource_count(self) -> int:
        return self._total_blocks

    def get_needed_resource_to_completion(self, req: LlmRequest) -> int:
        return 0

    def analyze_prefix_reuse(self, unique_tokens, req) -> _PrefixReuseSummary:
        return _PrefixReuseSummary()

    # === Allocation side (mirrors prepare_resources → addSequenceBatch/addToken) ===

    def add_sequence(self, req: LlmRequest) -> int:
        """Mirror addSequenceBatch for one sequence (no reuse path)."""
        prompt_len = req.prompt_len
        blocks_needed = math.ceil(prompt_len / self._tokens_per_block)
        if blocks_needed > self._free_blocks:
            raise RuntimeError(
                f"add_sequence: No free blocks left. "
                f"req={req.py_request_id} needed={blocks_needed} "
                f"free={self._free_blocks} total={self._total_blocks}")
        self._free_blocks -= blocks_needed
        self._allocated_per_seq[req.py_request_id] = blocks_needed
        # After addSequenceBatch, numTokens == promptLen (prefill consumed by
        # the per-token addToken loop in resource_manager.py:1052).
        self._num_tokens_per_seq[req.py_request_id] = prompt_len
        return blocks_needed

    def add_token(self, req_id: int) -> int:
        """Mirror KVCacheManager::addToken → adjustBlocksIfNeeded.

        Allocates exactly when crossing a block boundary, matching the
        `(numTokens - 1) % tokensPerBlock == 0` predicate in
        kvCacheManager.cpp:2204-2210.

        Returns blocks allocated this call (0 or 1).
        """
        self._num_tokens_per_seq[req_id] += 1
        n = self._num_tokens_per_seq[req_id]
        if (n - 1) % self._tokens_per_block == 0:
            if self._free_blocks == 0:
                raise RuntimeError(
                    f"add_token: No free blocks left. req={req_id} numTokens={n} "
                    f"allocated_total={sum(self._allocated_per_seq.values())} "
                    f"pool_total={self._total_blocks} "
                    f"per_seq={dict(self._allocated_per_seq)}")
            self._free_blocks -= 1
            self._allocated_per_seq[req_id] += 1
            return 1
        return 0

    def remove_sequence(self, req_id: int) -> None:
        if req_id in self._allocated_per_seq:
            self._free_blocks += self._allocated_per_seq.pop(req_id)
            self._num_tokens_per_seq.pop(req_id, None)

    def assert_invariant(self) -> None:
        used = sum(self._allocated_per_seq.values())
        assert used + self._free_blocks == self._total_blocks, (
            f"BOOKKEEPING DRIFT: allocated={used} free={self._free_blocks} "
            f"total={self._total_blocks} per_seq={self._allocated_per_seq}")


# ---------------------------------------------------------------------------
# Scenario A: single-pass admission burst against tight pool
# ---------------------------------------------------------------------------

def test_admission_burst_does_not_overflow_pool():
    """Many CONTEXT_INIT requests into a pool that holds only J of them.

    Hypothesis B: if the scheduler under-counts admission-side reservations,
    it will admit > J requests and the post-schedule add_sequence loop will
    crash. With the faithful mock, this exposes any divergence between
    `get_needed_blocks_one_step(CONTEXT_INIT)` and `addSequenceBatch`.
    """
    TPB = 64
    PROMPT_LEN = 128            # exactly 2 blocks per request
    BLOCKS_PER_REQ = math.ceil(PROMPT_LEN / TPB)
    POOL = 8                    # fits exactly 4 requests
    EXPECTED_FIT = POOL // BLOCKS_PER_REQ
    NUM_REQS = 10

    kv = FaithfulKVCacheManager(total_blocks=POOL,
                                tokens_per_block=TPB,
                                window_size=4096)
    scheduler = PyCapacityScheduler(
        max_num_requests=NUM_REQS,
        kv_cache_manager=kv,
        scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
    )
    requests = [
        _make_request(i, prompt_len=PROMPT_LEN, max_new_tokens=128)
        for i in range(NUM_REQS)
    ]

    fitting, _, paused = scheduler.schedule_request(requests)

    # Apply admissions exactly as PyExecutor.prepare_resources would.
    for req in fitting:
        kv.add_sequence(req)
    kv.assert_invariant()

    assert len(fitting) == EXPECTED_FIT, (
        f"OVER-ADMISSION (hypothesis B confirmed): "
        f"scheduler admitted {len(fitting)} requests, pool fits only {EXPECTED_FIT}. "
        f"POOL={POOL} TPB={TPB} PROMPT_LEN={PROMPT_LEN} "
        f"BLOCKS_PER_REQ={BLOCKS_PER_REQ} kv.free={kv._free_blocks}")


# ---------------------------------------------------------------------------
# Scenario B: simultaneous decode boundary crossings under tight margin
# ---------------------------------------------------------------------------

def test_decode_boundary_burst_does_not_overflow_pool():
    """N active decode requests, all simultaneously at the block boundary.

    Each needs +1 block this step. Pool has K free, K < N. Scheduler must
    pause enough to leave the survivors' add_token allocations satisfiable.
    """
    TPB = 64
    NUM_ACTIVE = 8
    # Each active req has numTokens = TPB (=64), one more decode token crosses to 65,
    # which triggers a 1-block allocation in add_token.
    # Each is already holding 1 block (its initial context block).
    INITIAL_BLOCKS_PER_REQ = 1
    USED = NUM_ACTIVE * INITIAL_BLOCKS_PER_REQ
    FREE_MARGIN = 3            # only 3 of 8 boundary-crossings can succeed
    POOL = USED + FREE_MARGIN

    kv = FaithfulKVCacheManager(total_blocks=POOL,
                                tokens_per_block=TPB,
                                window_size=4096)
    # Seed the manager with NUM_ACTIVE sequences each at numTokens=TPB,
    # holding INITIAL_BLOCKS_PER_REQ blocks. We bypass add_sequence here
    # because we don't need to simulate their prefill — only the state.
    seeded_reqs = []
    for i in range(NUM_ACTIVE):
        req = _make_request(i,
                            prompt_len=TPB,
                            max_new_tokens=128,
                            state=LlmRequestState.GENERATION_IN_PROGRESS)
        kv._allocated_per_seq[req.py_request_id] = INITIAL_BLOCKS_PER_REQ
        kv._num_tokens_per_seq[req.py_request_id] = TPB
        kv._free_blocks -= INITIAL_BLOCKS_PER_REQ
        seeded_reqs.append(req)
    kv.assert_invariant()

    scheduler = PyCapacityScheduler(
        max_num_requests=NUM_ACTIVE,
        kv_cache_manager=kv,
        scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
    )
    fitting, _, paused = scheduler.schedule_request(seeded_reqs)

    # Each fitting request gets one add_token call for the next decode token.
    try:
        for req in fitting:
            kv.add_token(req.py_request_id)
    except RuntimeError as e:
        pytest.fail(
            f"DECODE-BOUNDARY OVERFLOW (hypothesis B/C confirmed): {e}\n"
            f"  scheduler admitted {len(fitting)} req for decode "
            f"but pool had only {FREE_MARGIN} free blocks before step.\n"
            f"  fitting_ids={[r.request_id for r in fitting]} "
            f"paused_ids={[r.request_id for r in paused]}")

    kv.assert_invariant()
    assert len(fitting) <= FREE_MARGIN, (
        f"OVER-ADMISSION at decode boundary: fit={len(fitting)} margin={FREE_MARGIN}")


# ---------------------------------------------------------------------------
# Scenario C: bielik-shaped steady-state with admission churn
# ---------------------------------------------------------------------------

def test_steady_state_admission_churn_bielik_shape():
    """Realistic reproducer: bielik bench shape, scaled down for unit-test latency.

    Geometry mirrors the failing benchmark's pressure points:
      - tokens_per_block = 64
      - ISL = OSL = 128                  (4 blocks lifetime per request)
      - max_batch = 16                   (scaled from 512; ratio preserved against pool)
      - pool = max_batch * 4             (exactly enough for full saturation, no margin)

    With OSL=128 and max_batch=16, ~1 sequence completes per step on average,
    and a fresh CONTEXT_INIT slot opens for a new admission. If the scheduler's
    admission accounting drifts, the add_token loop will crash. If it's sound,
    the test runs to completion at ~milliseconds.
    """
    TPB = 64
    PROMPT_LEN = 128
    MAX_NEW = 128
    MAX_BATCH = 16
    BLOCKS_PER_SEQ_AT_PEAK = math.ceil((PROMPT_LEN + MAX_NEW) / TPB)  # 4
    POOL = MAX_BATCH * BLOCKS_PER_SEQ_AT_PEAK
    NUM_REQS = 4 * MAX_BATCH   # enough turnover to exercise the gap repeatedly
    MAX_STEPS = 500            # generous; should terminate well before this

    kv = FaithfulKVCacheManager(total_blocks=POOL,
                                tokens_per_block=TPB,
                                window_size=4096)
    scheduler = PyCapacityScheduler(
        max_num_requests=MAX_BATCH,
        kv_cache_manager=kv,
        scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
    )

    pending = [
        _make_request(i, prompt_len=PROMPT_LEN, max_new_tokens=MAX_NEW)
        for i in range(NUM_REQS)
    ]
    active: list[LlmRequest] = []
    completed: list[LlmRequest] = []

    for step in range(MAX_STEPS):
        # PyExecutor._fetch_and_activate_new_requests: top up active to max_batch.
        while len(active) < MAX_BATCH and pending:
            active.append(pending.pop(0))
        if not active:
            break

        fitting, _, paused = scheduler.schedule_request(active)

        # PyExecutor.prepare_resources: per-request add_sequence then add_token.
        try:
            for req in fitting:
                if req.is_context_init_state and req.is_first_context_chunk:
                    kv.add_sequence(req)
                    # Simulate one-pass prefill → transition to decode.
                    req.state = LlmRequestState.GENERATION_IN_PROGRESS
                elif req.is_generation_in_progress_state:
                    kv.add_token(req.py_request_id)
        except RuntimeError as e:
            pytest.fail(
                f"REPRODUCED at step={step}: {e}\n"
                f"  fitting_ids={[r.request_id for r in fitting]}\n"
                f"  paused_ids={[r.request_id for r in paused]}\n"
                f"  pool_total={POOL} free_at_fail={kv._free_blocks}\n"
                f"  per_seq_alloc={kv._allocated_per_seq}")

        kv.assert_invariant()

        # Paused requests get put back at the head of the pending queue.
        for req in paused:
            if req in active:
                active.remove(req)
        pending = list(paused) + pending

        # "Forward + sampling done" — drop completed sequences.
        still_active = []
        for req in fitting:
            n = kv._num_tokens_per_seq.get(req.py_request_id, 0)
            if n >= PROMPT_LEN + MAX_NEW:
                kv.remove_sequence(req.py_request_id)
                completed.append(req)
            else:
                still_active.append(req)
        # Non-scheduled active requests stay in active (they'll be retried next step).
        retained = [r for r in active if r not in fitting and r not in paused]
        active = still_active + retained

    assert len(completed) >= NUM_REQS // 2, (
        f"insufficient progress: completed={len(completed)} / {NUM_REQS} "
        f"in {MAX_STEPS} steps")
