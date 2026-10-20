# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Forward-failure survival and multimodal admission validation.

- The executor loops handle a None forward result: the overlap and
  non-overlap loops skip the iteration's sampling and bookkeeping and keep
  running, while the PP loop raises an explicit fatal on the last PP rank
  (inter-PP peers cannot skip a microbatch in lockstep).
- `PyExecutor._validate_request` calls the model-provided
  `validate_multimodal_request_data` hook so a model can reject a bad
  multimodal payload at admission, failing only that request.
- Error cleanup does not race in-flight device work: `_handle_errors`
  waits for the model engine's async input copy before freeing, parks
  failures whose KV blocks an unpinned transfer still touches until the
  transfer completes, and `_terminate_request` defers the free while the
  async transfer manager holds a claim.

All CUDA/MPI machinery is bypassed via object.__new__ + attribute
injection; the loop bodies and the `_validate_request` call site are
real code.
"""

import contextlib
import queue
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from tensorrt_llm._torch.models.modeling_multimodal_mixin import MultimodalModelMixin
from tensorrt_llm._torch.pyexecutor import hang_detector, py_executor
from tensorrt_llm._torch.pyexecutor.error_classification import ErrorBudget
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.py_executor import BatchState, PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import ScheduledRequests

pytestmark = pytest.mark.cpu_only


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _AcceptingModel(MultimodalModelMixin):
    """Model that relies on the mixin's default (accept-everything) hook."""


class _RejectingModel(MultimodalModelMixin):
    def validate_multimodal_request_data(self, mm_data: dict) -> None:
        raise ValueError("multimodal payload exceeds the encoder budget")


def _request(mm_data=None):
    return SimpleNamespace(
        py_disaggregated_params=None,
        sampling_config=None,
        py_multimodal_data=mm_data,
    )


def _validating_executor(model):
    """Executor shell exposing only what _validate_request touches."""
    executor = object.__new__(PyExecutor)
    executor.kv_cache_transceiver = None
    executor.kv_cache_manager = None
    executor.model_engine = SimpleNamespace(model=model)
    executor.sampler = Mock()
    executor._validate_token_id_range = Mock()
    executor._validate_request_budget = Mock()
    return executor


def _scheduled_batch():
    batch = ScheduledRequests()
    batch.generation_requests = [SimpleNamespace(py_batch_idx=0, is_attention_dp_dummy=False)]
    return batch


def _loop_executor(monkeypatch):
    """Executor shell whose one scheduled batch fails its forward.

    The first iteration schedules a batch whose forward returns None; the
    second observes shutdown. Executor-owned helpers are stubbed; the loop
    body under test is real.
    """
    for target in ("torch.cuda.set_device", "cudart.cudaSetDevice", "CUASSERT"):
        monkeypatch.setattr(f"tensorrt_llm._torch.pyexecutor.py_executor.{target}", Mock())

    executor = object.__new__(PyExecutor)
    executor.device_id = 0
    profiler = MagicMock()
    profiler.__enter__.return_value = Mock()
    executor._profiler = Mock(return_value=profiler)
    executor.hang_detector = MagicMock()
    executor._step_scope = MagicMock()
    executor.perf_manager = MagicMock()
    executor.perf_manager.enabled = False
    executor.perf_manager.create_timing_events.return_value = (None, None, None)

    executor.enable_iter_perf_stats = False
    executor.enable_attention_dp = False
    executor.is_benchmark_disagg = False
    executor.iter_counter = 0
    executor._resource_governor_enabled = False
    executor._is_kv_manager_v2 = False
    executor._mm_encoder_item_scheduling_enabled = False
    executor.enable_early_first_token_response = False
    executor.kv_cache_transceiver = None
    executor.kv_connector_manager = None
    executor.guided_decoder = None
    executor.drafter = None
    executor.dwdp_manager = None
    executor.speculation_gate = None
    executor.model_engine = None
    executor.previous_batch = None
    executor.has_previous_draft_tokens = False
    executor.enable_kv_cache_events = False
    executor.active_requests = []
    executor.waiting_queue = []
    executor.inflight_req_ids = set()
    executor._transfer_held_failures = {}
    executor.is_shutdown = False
    executor.dist = Mock(tp_size=1, world_size=1)

    executor._disagg_coordinator = MagicMock()
    executor.resource_manager = MagicMock()

    batches = iter([(_scheduled_batch(), None), (None, None)])
    executor._prepare_and_schedule_batch = lambda: next(batches)
    executor._can_queue = Mock(return_value=(True, True))
    executor._check_benchmark_disagg_gate = Mock(return_value=(True, False))
    executor._forward_step = Mock(return_value=None)
    for name in (
        "_terminate_requests",
        "_pause_requests",
        "_prepare_disagg_gen_transmission_complete",
        "_handle_dynamic_draft_len",
        "_kv_connector_start_batch",
        "_kv_connector_terminate_requests",
        "_finalize_adp_dummy_allocation",
        "_commit_kv_cache_stats",
        "_flush_pending_transfer_responses",
        "_flush_iter_stats_synced",
        "_wait_for_model_engine_input_copy",
        "_enqueue_responses",
        "_update_generation_requests_that_will_complete_next_iteration",
        "_sample_async",
        "_update_request_states",
        "_update_requests",
        "_send_kv_async",
        "_handle_canceled_requests",
        "_handle_responses",
        "_handle_guided_decoder_errors",
        "_maybe_prefetch_next_iter_mm_encoders",
        "_submit_encoder_step",
        "_run_encoder_step",
        "_revert_gen_alloc",
    ):
        setattr(executor, name, Mock())
    return executor


# ---------------------------------------------------------------------------
# The mixin hook contract
# ---------------------------------------------------------------------------


def test_mixin_hook_accepts_everything_by_default():
    assert _AcceptingModel().validate_multimodal_request_data({"image": object()}) is None


def test_mixin_hook_override_can_reject():
    with pytest.raises(ValueError, match="encoder budget"):
        _RejectingModel().validate_multimodal_request_data({"image": object()})


# ---------------------------------------------------------------------------
# The _validate_request call site
# ---------------------------------------------------------------------------


def test_validate_request_rejects_via_model_hook():
    executor = _validating_executor(_RejectingModel())
    with pytest.raises(ValueError, match="encoder budget"):
        PyExecutor._validate_request(executor, _request(mm_data={"image": object()}))


def test_validate_request_accepts_when_model_hook_accepts():
    executor = _validating_executor(_AcceptingModel())
    PyExecutor._validate_request(executor, _request(mm_data={"image": object()}))


def test_validate_request_skips_hook_for_text_only_requests():
    model = Mock(spec=["validate_multimodal_request_data"])
    executor = _validating_executor(model)
    PyExecutor._validate_request(executor, _request(mm_data=None))
    model.validate_multimodal_request_data.assert_not_called()


def test_validate_request_accepts_models_without_the_hook():
    executor = _validating_executor(object())
    PyExecutor._validate_request(executor, _request(mm_data={"image": object()}))


# ---------------------------------------------------------------------------
# Loop-level None handling
# ---------------------------------------------------------------------------


def test_executor_loop_survives_a_failed_forward(monkeypatch):
    executor = _loop_executor(monkeypatch)

    PyExecutor._executor_loop(executor)

    executor._forward_step.assert_called_once()
    executor._sample_async.assert_not_called()
    executor._update_request_states.assert_not_called()
    # The loop reached the scheduling-observed shutdown, i.e. it survived the
    # failed-forward iteration instead of crashing on batch_outputs['logits'].
    assert executor._event_loop_completed


def test_executor_loop_overlap_survives_a_failed_forward(monkeypatch):
    executor = _loop_executor(monkeypatch)
    # A previous batch holding timing events borrowed from the perf pool.
    # Dropping it on the failed iteration skips _process_iter_stats, which
    # normally returns them, so the loop must release them itself.
    prev_start, prev_end = object(), object()
    executor.previous_batch = BatchState(
        scheduled_requests=_scheduled_batch(),
        sample_state=None,
        gpu_forward_start_event=prev_start,
        gpu_forward_end_event=prev_end,
        gpu_forward_events_from_perf_pool=True,
    )

    PyExecutor._executor_loop_overlap(executor)

    executor._forward_step.assert_called_once()
    executor._sample_async.assert_not_called()
    # The failed iteration must not become a previous batch for the next
    # iteration's bookkeeping.
    assert executor.previous_batch is None
    # The dropped previous batch's borrowed events went back to the pool
    # (exactly once: the failed iteration itself borrowed none).
    executor.perf_manager.release_forward_timing_events.assert_called_once_with(
        prev_start, prev_end
    )
    assert executor._event_loop_completed


def test_executor_loop_pp_raises_explicit_fatal_on_last_rank(monkeypatch):
    executor = _loop_executor(monkeypatch)
    executor.dist = Mock(
        rank=0,
        pp_rank=1,
        pp_size=2,
        tp_size=1,
        cp_size=1,
        world_size=2,
        is_first_pp_rank=False,
        is_last_pp_rank=True,
    )
    executor.num_micro_batches = 1
    executor.micro_batches = [None]
    executor._pp_rebalance_drain_iters = None
    executor.pp_async_broadcast_sample_state = True
    executor._fetch_and_activate_new_requests = Mock(return_value=[])
    executor._handle_control_request = Mock()
    executor._pad_attention_dp_dummy_request = Mock()
    executor._pp_schedule_and_propagate = Mock(return_value=(_scheduled_batch(), [], None, None))
    executor._add_inflight_ids = Mock()
    executor.num_scheduled_requests = 0

    with pytest.raises(RuntimeError, match="last PP rank"):
        PyExecutor._executor_loop_pp(executor)

    executor._forward_step.assert_called_once()
    executor._sample_async.assert_not_called()


def _response_gather_entries(executor):
    """Response-gather entries a pass makes besides the synced flush.

    Under attention DP both `_handle_responses` (via `_enqueue_responses`)
    and a direct `_enqueue_responses` call perform exactly one tp_gather,
    and collectives pair by call order across ranks.
    """
    return executor._handle_responses.call_count + executor._enqueue_responses.call_count


def test_executor_loop_failed_forward_keeps_response_gather_parity(monkeypatch):
    """A failed-forward pass must enter as many response gathers as a healthy one.

    Under attention DP a rank-local forward failure leaves batch_outputs None
    on one rank only; if that rank skips the `_handle_responses` gather while
    healthy ranks enter it, the TP group desynchronizes and later pairs an
    int payload against a response list (the fail-fast in _enqueue_responses).
    """
    healthy = _loop_executor(monkeypatch)
    healthy._forward_step = Mock(return_value={"logits": Mock()})
    PyExecutor._executor_loop(healthy)
    healthy._forward_step.assert_called_once()

    failed = _loop_executor(monkeypatch)  # its _forward_step returns None
    PyExecutor._executor_loop(failed)
    failed._forward_step.assert_called_once()
    failed._sample_async.assert_not_called()

    healthy_gathers = _response_gather_entries(healthy)
    failed_gathers = _response_gather_entries(failed)
    assert healthy_gathers == failed_gathers
    assert failed_gathers > 0
    # The synced flush is entered once per completed pass on both sides.
    assert (
        healthy._flush_pending_transfer_responses.call_count
        == failed._flush_pending_transfer_responses.call_count
    )


class _SimulatedTP:
    """Two ranks with blocking, ordered collectives and an abortable model channel."""

    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.rounds = {}
        self.aborted = False
        self.killed = threading.Event()
        self.contributions = []

    def collective(self, rank: int, step: int, kind: str, value: object) -> list[object]:
        deadline = time.monotonic() + 5
        with self.condition:
            entries = self.rounds.setdefault(step, {})
            assert rank not in entries
            entries[rank] = (kind, value)
            self.contributions.append((rank, step, kind, value))
            self.condition.notify_all()
            while len(entries) != 2 and not self.aborted:
                remaining = deadline - time.monotonic()
                assert remaining > 0, f"rank {rank} stranded at recovery collective {step}"
                self.condition.wait(remaining)
            if self.aborted:
                raise RuntimeError("simulated TP world aborted")
            assert {entry[0] for entry in entries.values()} == {kind}, "collective order differs"
            return [entries[0][1], entries[1][1]]

    def wait_for_abort(self) -> None:
        with self.condition:
            assert self.condition.wait_for(lambda: self.aborted, timeout=5)
        raise RuntimeError("simulated TP world aborted")

    def abort(self, *args, **kwargs) -> None:
        with self.condition:
            self.aborted = True
            self.killed.set()
            self.condition.notify_all()

    def rank(self, rank: int) -> SimpleNamespace:
        step = 0

        def collective(kind: str, value: object) -> list[object]:
            nonlocal step
            result = self.collective(rank, step, kind, value)
            step += 1
            return result

        return SimpleNamespace(
            rank=rank,
            tp_rank=rank,
            tp_size=2,
            cp_size=1,
            pp_size=1,
            has_pp=False,
            world_size=2,
            mapping=SimpleNamespace(tp_group=[0, 1]),
            forward_allgather=lambda value: collective("gather", value),
            forward_max=lambda value: max(collective("max", value)),
        )


def _tp_executors(
    monkeypatch: pytest.MonkeyPatch,
    world: _SimulatedTP,
    *,
    phase: str = "forward",
    failing_rank: int = 1,
    blocked: str | None = None,
) -> list[PyExecutor]:
    monkeypatch.setattr(py_executor.torch.cuda, "stream", lambda _: contextlib.nullcontext())
    monkeypatch.setattr(py_executor.torch.cuda, "current_stream", Mock(return_value=Mock()))

    def synchronize(device: int) -> None:
        if blocked == "device" and device == 0:
            world.wait_for_abort()

    monkeypatch.setattr(py_executor.torch.cuda, "synchronize", Mock(side_effect=synchronize))
    monkeypatch.setattr(hang_detector, "propagate_hard_kill", world.abort)
    monkeypatch.setenv(hang_detector.RANK_CRASH_KILL_GRACE_ENV, "0.5" if blocked else "5")
    monkeypatch.setattr(py_executor, "HandleLogits", Mock())
    monkeypatch.setattr(py_executor, "HandleAdditionalOutputs", Mock())
    executors = []
    for rank in range(2):
        executor = _loop_executor(monkeypatch)
        del executor._forward_step
        del executor._sample_async
        del executor._handle_guided_decoder_errors
        del executor._wait_for_model_engine_input_copy
        del executor._enqueue_responses
        executor.device_id = rank
        executor.dist = world.rank(rank)
        executor.gather_all_responses = False
        executor.response_cv = threading.Condition()
        executor.responses = {}
        executor.result_wait_queues = {}
        executor._error_budget = ErrorBudget(recovery_rate=0)
        executor._fatal_error = None
        executor._pending_transfer_responses = []
        executor._pending_response_terminations = []
        executor._iter_adp_dummy_ctx_tokens = 0
        executor._iter_adp_dummy_gen_tokens = 0
        executor._compute_adp_dummy_tokens = Mock(return_value=(0, 0))
        executor._maybe_record_hang_diagnostic_phase = Mock()
        executor._attach_encoder_output_to_execution_stream = Mock()
        executor._mark_cross_kv_projection_consumed = Mock()
        executor._kv_connector_wait_for_save = Mock()
        executor._terminate_request = Mock()
        executor._process_previous_batch = Mock()
        executor.sampler = Mock()
        executor.execution_stream = Mock()
        executor.model_engine = SimpleNamespace(route_capture=None)
        executor._forward_calls = []
        executor._sampled_steps = []
        executor._updated_steps = []
        executor._failure_injected = False

        def fail_once(at: str, ex=executor) -> None:
            if phase == at and ex.dist.rank == failing_rank and not ex._failure_injected:
                ex._failure_injected = True
                raise ValueError(f"rank {failing_rank} {phase} failed")

        def forward(*args, ex=executor, fail=fail_once, **kwargs) -> dict[str, object]:
            ex._forward_calls.append(ex.iter_counter)
            if ex.iter_counter == 0:
                fail("forward")
                if blocked == "model":
                    world.wait_for_abort()
            return {"logits": object()}

        executor.model_engine.forward = forward

        def sample(batch, outputs, prefix_sum, ex=executor, fail=fail_once) -> SimpleNamespace:
            ex._sampled_steps.append(ex.iter_counter)
            fail("sampling")
            return SimpleNamespace(device=None)

        executor.sampler.sample_async.side_effect = sample
        executor._update_request_states.side_effect = (
            lambda batch, ex=executor: ex._updated_steps.append(
                (ex.iter_counter, [r.py_request_id for r in ex.active_requests])
            )
        )

        def wait_for_input_copy(ex=executor, fail=fail_once) -> None:
            if ex.iter_counter == 0:
                if ex._sampled_steps:
                    fail("input_copy")
                else:
                    fail("input_copy_before_schedule")

        executor.model_engine.wait_for_input_copy = wait_for_input_copy
        executor._validate_request = lambda request, fail=fail_once: (
            fail("admission") if request.py_request_id == 1 else None
        )
        if phase in ("guided_decoder", "guided_exception"):
            executor.guided_decoder = MagicMock()

            def guided_execute(logits, ex=executor, fail=fail_once):
                fail("guided_exception")
                if ex.dist.rank == failing_rank and ex.iter_counter == 0:
                    ex._failure_injected = True
                    return [(1, f"rank {failing_rank} {phase} failed")]
                return []

            executor.guided_decoder.execute.side_effect = guided_execute

        def schedule(ex=executor) -> tuple[ScheduledRequests | None, None]:
            if ex.iter_counter >= 2 or ex.is_shutdown:
                return None, None
            ids = [ex.iter_counter + 1]
            if ex.iter_counter == 0 and phase in ("admission", "guided_decoder"):
                ids.append(11)
            requests = [_failed_request(req_id, remaining=0, position=8) for req_id in ids]
            for request in requests:
                request.py_batch_idx = 0
                request.is_attention_dp_dummy = False
            ex.active_requests = []
            ex._fetch_new_requests = Mock(return_value=requests)
            requests = ex._fetch_and_activate_new_requests()
            batch = ScheduledRequests()
            batch.generation_requests = requests
            return batch, None

        executor._prepare_and_schedule_batch = schedule
        if phase == "input_copy_before_schedule":
            executor.active_requests = [_failed_request(1, remaining=0, position=8)]
        executors.append(executor)
    return executors


def _run_tp(executors: list[PyExecutor], world: _SimulatedTP, overlap: bool) -> list[BaseException]:
    failures = queue.Queue()
    loop = PyExecutor._executor_loop_overlap if overlap else PyExecutor._executor_loop

    def run(executor: PyExecutor) -> None:
        try:
            loop(executor)
        except BaseException as error:
            failures.put(error)

    threads = [threading.Thread(target=run, args=(ex,), daemon=True) for ex in executors]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=7)
        assert not any(thread.is_alive() for thread in threads), "simulated ranks hung"
    finally:
        if any(thread.is_alive() for thread in threads):
            world.abort()
            for thread in threads:
                thread.join(timeout=1)
    return list(failures.queue)


@pytest.mark.parametrize(
    "phase,overlap,failing_rank",
    [
        ("forward", False, 1),
        ("forward", True, 1),
        ("admission", False, 1),
        ("guided_decoder", True, 0),
        ("guided_exception", False, 1),
        ("sampling", True, 1),
        ("input_copy", False, 1),
        ("input_copy_before_schedule", True, 1),
    ],
)
def test_tp_forward_failure_recovers_together(
    monkeypatch: pytest.MonkeyPatch, overlap: bool, phase: str, failing_rank: int
) -> None:
    world = _SimulatedTP()
    executors = _tp_executors(monkeypatch, world, phase=phase, failing_rank=failing_rank)
    if overlap and phase in ("forward", "input_copy_before_schedule"):
        for executor in executors:
            executor.previous_batch = BatchState(
                scheduled_requests=_scheduled_batch(),
                sample_state=None,
                gpu_forward_start_event=object(),
                gpu_forward_end_event=object(),
                gpu_forward_events_from_perf_pool=True,
            )
    failures = _run_tp(executors, world, overlap)

    assert failures == []
    assert not world.killed.is_set()
    request_scoped = phase in ("admission", "guided_decoder")
    for executor in executors:
        assert executor._updated_steps == ([(0, [11]), (1, [2])] if request_scoped else [(1, [2])])
        assert executor._forward_calls == ([1] if phase == "input_copy_before_schedule" else [0, 1])
        assert executor._error_budget.budget == pytest.approx(1.0 if request_scoped else 0.9)
        if not request_scoped:
            executor.sampler.invalidate_in_graph_sampling.assert_called_once()
        if overlap and phase in ("forward", "input_copy_before_schedule"):
            executor.perf_manager.release_forward_timing_events.assert_called_once()
            executor._process_previous_batch.assert_not_called()
        executor._terminate_request.assert_called_once()
        assert executor._terminate_request.call_args.args[0].py_request_id == 1
        assert executor._event_loop_completed
    assert executors[failing_rank]._failure_injected
    assert executors[1].responses == {}
    errors = executors[0].responses[1]
    assert len(errors) == 1
    assert errors[0].request_id == 1
    assert f"rank {failing_rank} {phase} failed" in errors[0].error_msg
    assert {rank for rank, _, _, _ in world.contributions} == {0, 1}


@pytest.mark.parametrize("blocked", ["model", "device"])
def test_tp_forward_failure_aborts_a_stranded_peer(
    monkeypatch: pytest.MonkeyPatch, blocked: str
) -> None:
    world = _SimulatedTP()
    executors = _tp_executors(monkeypatch, world, blocked=blocked)
    failures = _run_tp(executors, world, overlap=False)

    assert world.killed.is_set()
    assert len(failures) == 2
    for executor in executors:
        assert executor._forward_calls == [0]
        assert executor._sampled_steps == []
        assert executor._updated_steps == []
        executor._terminate_request.assert_not_called()


# ---------------------------------------------------------------------------
# First-chunk prefill failure must not publish unwritten V1 reuse blocks
# ---------------------------------------------------------------------------


def _error_executor():
    """Executor shell exposing only what the non-ADP _handle_errors path touches."""
    executor = object.__new__(PyExecutor)
    executor._error_budget = Mock()
    executor._fatal_error = None
    executor.enable_attention_dp = False
    executor.dist = Mock(world_size=1)
    executor._pending_transfer_responses = []
    executor._pending_response_terminations = []
    executor._transfer_held_failures = {}
    executor.kv_cache_transceiver = None
    executor._enqueue_responses = Mock()
    executor._terminate_request = Mock()
    executor._wait_for_model_engine_input_copy = Mock()
    return executor


def _failed_request(request_id, *, remaining, position):
    return SimpleNamespace(
        py_request_id=request_id,
        py_client_id=100 + request_id,
        context_remaining_length=remaining,
        context_current_position=position,
        state=None,
    )


def test_handle_errors_tags_first_chunk_prefill_failures_as_reuse_poisoned():
    executor = _error_executor()
    first_chunk = _failed_request(1, remaining=8, position=0)
    mid_context = _failed_request(2, remaining=4, position=4)
    generation = _failed_request(3, remaining=0, position=8)
    # The disagg coordinator's error path feeds TransferRequest objects
    # (no context_* attributes) through _handle_errors; tagging must not
    # raise on them and must leave them untagged.
    transfer_like = SimpleNamespace(py_request_id=4, py_client_id=104, state=None)
    requests = [first_chunk, mid_context, generation, transfer_like]
    executor.active_requests = list(requests)

    PyExecutor._handle_errors(
        executor,
        "forward failed",
        requests=requests,
        charge_budget=False,
    )

    # Only the request that never completed a context chunk wrote no KV.
    assert getattr(first_chunk, "py_kv_reuse_poisoned", False)
    assert not getattr(mid_context, "py_kv_reuse_poisoned", False)
    assert not getattr(generation, "py_kv_reuse_poisoned", False)
    assert not getattr(transfer_like, "py_kv_reuse_poisoned", False)


def _kv_cache_manager_shell():
    manager = object.__new__(KVCacheManager)
    manager.impl = Mock()
    manager._preprepared_dummy_request_ids = set()
    return manager


def test_free_resources_releases_poisoned_request_without_reuse_store():
    """A poisoned release must advance the context cursor off position 0.

    The remove_sequence binding only accepts (request_id, LlmRequest, bool),
    so the no-store release is achieved by moving the cursor off 0 before
    the call: usable-token accounting then yields zero tokens and the legacy
    position-0 fallback -- which would publish unwritten blocks under the
    prompt's keys -- no longer applies.
    """
    manager = _kv_cache_manager_shell()
    positions_at_call = []
    manager.impl.remove_sequence.side_effect = lambda _req_id, req, _pin: positions_at_call.append(
        req.context_current_position
    )
    request = SimpleNamespace(
        py_request_id=7, py_kv_reuse_poisoned=True, context_current_position=0
    )

    KVCacheManager.free_resources(manager, request)

    # The binding's exact (request_id, LlmRequest, bool) signature is kept.
    manager.impl.remove_sequence.assert_called_once_with(7, request, False)
    assert positions_at_call == [1]


def test_free_resources_keeps_reuse_store_for_untagged_requests():
    manager = _kv_cache_manager_shell()
    request = SimpleNamespace(py_request_id=7, context_current_position=0)

    KVCacheManager.free_resources(manager, request)

    manager.impl.remove_sequence.assert_called_once_with(7, request, False)
    # An untagged request keeps its cursor: a position-0 release with a real
    # completed prefill is the legacy fallback's intended territory.
    assert request.context_current_position == 0


# ---------------------------------------------------------------------------
# Error cleanup must not race pending input copies or in-flight KV transfers
# ---------------------------------------------------------------------------


def _transfer_executor():
    """_error_executor plus a disagg transceiver and transfer manager."""
    executor = _error_executor()
    executor.kv_cache_transceiver = Mock()
    executor.kv_cache_transceiver.has_inflight_transfer = Mock(return_value=False)
    executor.async_transfer_manager = Mock()
    executor.async_transfer_manager.requests_in_transfer = Mock(return_value={})
    return executor


def _gen_receive_request(request_id):
    request = _failed_request(request_id, remaining=0, position=8)
    request.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
    return request


def test_handle_errors_waits_for_input_copy_before_terminating():
    """The input-copy wait must precede the frees.

    The engine's async input H2D copy reads host page metadata that the
    frees rewrite in place.
    """
    executor = _error_executor()
    order = []
    executor._wait_for_model_engine_input_copy = Mock(side_effect=lambda: order.append("wait"))
    executor._terminate_request = Mock(side_effect=lambda _req: order.append("terminate"))
    request = _failed_request(1, remaining=0, position=8)
    executor.active_requests = [request]

    PyExecutor._handle_errors(executor, "forward failed", requests=[request], charge_budget=False)

    assert order == ["wait", "terminate"]


def test_handle_errors_survives_a_failing_input_copy_wait():
    """A corrupted CUDA context makes the wait raise; cleanup must proceed."""
    executor = _error_executor()
    executor._wait_for_model_engine_input_copy = Mock(
        side_effect=RuntimeError("CUDA error: unspecified launch failure")
    )
    request = _failed_request(1, remaining=0, position=8)
    executor.active_requests = [request]

    PyExecutor._handle_errors(executor, "forward failed", requests=[request], charge_budget=False)

    executor._terminate_request.assert_called_once_with(request)


def test_handle_errors_parks_requests_with_pending_kv_receives():
    """A request with a pending KV receive is parked, not failed.

    The receive writes into the request's blocks with no pin; failing it
    immediately would free and reallocate them under the fabric.
    """
    executor = _transfer_executor()
    receiving = _gen_receive_request(1)
    computing = _failed_request(2, remaining=0, position=8)
    executor.active_requests = [receiving, computing]

    PyExecutor._handle_errors(executor, "forward failed", charge_budget=False)

    # The receiving request keeps its blocks, state, and active slot until the
    # transceiver lets go; the compute-phase request fails now.
    assert receiving.state == LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS
    assert executor.active_requests == [receiving]
    assert len(executor._transfer_held_failures) == 1
    executor._terminate_request.assert_called_once_with(computing)
    (payload,) = executor._enqueue_responses.call_args.args
    assert [req_id for req_id, _ in payload] == [2]


def test_handle_errors_fails_pinned_sends_immediately():
    """A context send past start_transfer fails immediately.

    Its blocks are pinned, so the failure (and its error response) must not
    wait for the transfer; only the free is deferred, by
    _terminate_request's claim check.
    """
    executor = _transfer_executor()
    sending = _failed_request(3, remaining=0, position=8)
    sending.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
    executor.async_transfer_manager.requests_in_transfer.return_value = {3: sending}
    executor.active_requests = [sending]

    PyExecutor._handle_errors(executor, "forward failed", charge_budget=False)

    assert not executor._transfer_held_failures
    assert executor.active_requests == []
    executor._terminate_request.assert_called_once_with(sending)
    (payload,) = executor._enqueue_responses.call_args.args
    assert [req_id for req_id, _ in payload] == [3]


def test_sweep_fails_parked_request_after_receive_completes():
    executor = _transfer_executor()
    receiving = _gen_receive_request(1)
    executor.active_requests = [receiving]
    PyExecutor._handle_errors(executor, "forward failed", charge_budget=False)
    assert executor._transfer_held_failures

    # Still receiving: the sweep must not touch it.
    PyExecutor._sweep_transfer_held_failures(executor)
    executor._terminate_request.assert_not_called()

    receiving.state = LlmRequestState.GENERATION_IN_PROGRESS
    PyExecutor._sweep_transfer_held_failures(executor)

    assert not executor._transfer_held_failures
    assert executor.active_requests == []
    executor._terminate_request.assert_called_once_with(receiving)
    (payload,) = executor._enqueue_responses.call_args.args
    assert [req_id for req_id, _ in payload] == [1]


def test_sweep_drops_requests_terminated_elsewhere():
    executor = _transfer_executor()
    gone = _gen_receive_request(1)
    executor._transfer_held_failures[id(gone)] = (gone, "forward failed")
    executor.active_requests = []

    PyExecutor._sweep_transfer_held_failures(executor)

    assert not executor._transfer_held_failures
    executor._terminate_request.assert_not_called()
    executor._enqueue_responses.assert_not_called()


def test_fatal_error_flushes_parked_failures():
    """A fatal error fails parked requests with everything else.

    The executor exits instead of reusing blocks, so the hold no longer
    applies and the requests must not be stranded.
    """
    executor = _transfer_executor()
    receiving = _gen_receive_request(1)
    executor.active_requests = [receiving]
    PyExecutor._handle_errors(executor, "forward failed", charge_budget=False)
    assert executor._transfer_held_failures

    executor._error_budget.consume.return_value = True
    executor._error_budget.budget = 0.0
    executor.is_shutdown = False
    executor.waiting_queue = []
    raw_queue = Mock()
    raw_queue.empty.return_value = True
    executor.executor_request_queue = Mock()
    executor.executor_request_queue.get_request_queue.return_value = raw_queue

    PyExecutor._handle_errors(executor, "device lost")

    assert not executor._transfer_held_failures
    assert executor.active_requests == []
    executor._terminate_request.assert_called_once_with(receiving)


def _terminating_executor():
    """Executor shell exposing only what _terminate_request touches."""
    executor = object.__new__(PyExecutor)
    executor.kv_connector_manager = None
    executor._is_kv_manager_v2 = False
    executor.kv_cache_transceiver = Mock()
    executor.async_transfer_manager = Mock()
    executor._disagg_pp_termination_handler = None
    executor._do_terminate_request = Mock()
    return executor


def test_terminate_request_defers_free_while_transfer_holds_a_claim():
    """V1 with a transceiver defers the free while a claim is held.

    A request still in the async transfer manager must not be freed;
    end_transfer pops the claim and the release path re-enters
    _terminate_request, which would otherwise double-free.
    """
    executor = _terminating_executor()
    request = SimpleNamespace(py_request_id=7, is_dummy_request=False)
    executor.async_transfer_manager.requests_in_transfer.return_value = {7: request}

    PyExecutor._terminate_request(executor, request)
    executor._do_terminate_request.assert_not_called()

    # The claim is gone (end_transfer popped it): the re-entry frees.
    executor.async_transfer_manager.requests_in_transfer.return_value = {}
    PyExecutor._terminate_request(executor, request)
    executor._do_terminate_request.assert_called_once_with(request)
