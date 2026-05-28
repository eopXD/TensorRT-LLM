# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PyExecutor-level guards for the v1 pause-release-blocks fix.

Verifies two distinct claims about the fix in py_executor.py:

  (A) Behaviour: the helpers do what the fix promises.
      - `_release_paused_blocks(reqs)` calls `resource_manager.free_resources(req)`
        for every request, in order.
      - `_pause_requests(reqs)` calls `req.pause(self.max_input_len)` for every
        request.
      Tested by borrowing the real methods onto a thin mock executor
      (pattern from tests/unittest/_torch/executor/test_benchmark_disagg.py).

  (B) Wiring: the loop bodies invoke the helpers in the right relative order.
      - `_executor_loop`:        terminate -> release -> pause -> prepare_resources
      - `_executor_loop_overlap`: terminate -> release -> prepare_resources -> ...
                                  -> _update_requests(previous_batch.sample_state)
                                  -> pause
      Tested via `inspect.getsource` on the loop method bodies (pattern from
      tests/unittest/_torch/executor/test_py_executor_creator_flash_mla_tokens_per_block.py).

These are wiring-pin guards. A future refactor that moves the call sites will
trip a specific assertion below — refactor consciously.
"""

from __future__ import annotations

import inspect
import re
from unittest.mock import Mock, call

from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor


# ===========================================================================
# (A) Behaviour tests — borrow real helper methods onto a thin mock executor
# ===========================================================================

class _MockExecutor:
    """Minimal stand-in. Only carries the attributes the borrowed helpers
    touch: `resource_manager` for `_release_paused_blocks`,
    `max_input_len` for `_pause_requests`."""

    def __init__(self, resource_manager=None, max_input_len: int = 4096):
        self.resource_manager = resource_manager
        self.max_input_len = max_input_len

    _release_paused_blocks = PyExecutor._release_paused_blocks
    _pause_requests = PyExecutor._pause_requests


def test_release_paused_blocks_calls_free_resources_for_each_request():
    rm = Mock()
    ex = _MockExecutor(resource_manager=rm)
    reqs = [Mock(name=f"req{i}", py_request_id=i) for i in range(3)]

    ex._release_paused_blocks(reqs)

    assert rm.free_resources.call_count == len(reqs)
    # assert_has_calls with no any_order=True pins both presence and order.
    rm.free_resources.assert_has_calls([call(r) for r in reqs])


def test_pause_requests_calls_req_pause_with_max_input_len():
    ex = _MockExecutor(max_input_len=8192)
    reqs = [Mock() for _ in range(3)]

    ex._pause_requests(reqs)

    for req in reqs:
        req.pause.assert_called_once_with(8192)


# ===========================================================================
# (B) Wiring tests — verify call order in the two loop bodies
# ===========================================================================

_V1_GATE_RE = re.compile(
    r"if not self\._is_kv_manager_v2\s*:\s*\n"
    r"((?:[ \t]+.*\n)+)"
)


def _v1_gate_body(loop_method) -> str:
    """Return the indented body of the first `if not self._is_kv_manager_v2:`
    block inside `loop_method`. Empty string if not found."""
    src = inspect.getsource(loop_method)
    match = _V1_GATE_RE.search(src)
    return match.group(1) if match else ""


def test_executor_loop_release_called_between_terminate_and_pause():
    """Non-overlap path: terminate, release, pause must all sit inside the v1
    gate, in that order. Failure modes:
      - release missing => fix not applied
      - release > pause => token-fold runs before block release; bug remains
      - empty body     => v1 gate moved/removed; refactor caught"""
    body = _v1_gate_body(PyExecutor._executor_loop)

    term = body.find("_terminate_requests(")
    release = body.find("_release_paused_blocks(")
    pause = body.find("_pause_requests(")

    assert term >= 0 and release >= 0 and pause >= 0, (
        f"v1 gate of _executor_loop missing one of "
        f"_terminate_requests/_release_paused_blocks/_pause_requests; "
        f"indices: terminate={term} release={release} pause={pause}")
    assert term < release < pause, (
        f"v1 gate call order broken in _executor_loop: "
        f"terminate@{term}, release@{release}, pause@{pause}. "
        f"Required: terminate < release < pause.")


def test_executor_loop_overlap_release_in_early_v1_gate():
    """Overlap path is split: the EARLY v1 gate (after _terminate_requests)
    must contain `_release_paused_blocks`. If release drifts to the late
    gate, it runs after prepare_resources and the bug returns."""
    src = inspect.getsource(PyExecutor._executor_loop_overlap)
    early = _V1_GATE_RE.search(src)
    assert early, "first v1 gate missing in _executor_loop_overlap"
    body = early.group(1)
    assert "_terminate_requests(" in body, (
        "_terminate_requests not in early v1 gate of _executor_loop_overlap")
    assert "_release_paused_blocks(" in body, (
        "_release_paused_blocks not in early v1 gate of "
        "_executor_loop_overlap — overlap-mode half of the fix missing.")


def test_executor_loop_overlap_pause_in_late_v1_gate():
    """Overlap mode must have at least two v1 gates (early=release,
    late=pause). The token-fold pause MUST live in the late gate so it
    runs after `_update_requests` applied the previous batch's tokens."""
    src = inspect.getsource(PyExecutor._executor_loop_overlap)
    gates = list(_V1_GATE_RE.finditer(src))
    assert len(gates) >= 2, (
        f"expected >=2 v1 gates in _executor_loop_overlap "
        f"(early=release, late=pause), found {len(gates)}")
    assert "_pause_requests(" in gates[-1].group(1), (
        "_pause_requests not in the late v1 gate of _executor_loop_overlap")


def test_executor_loop_overlap_pause_follows_update_requests():
    """Load-bearing constraint for overlap mode: req.pause folds generated
    tokens into the prompt. If pause runs BEFORE _update_requests, the
    previous batch's last sampled token is dropped silently — request
    resumes one token short of where it actually got to."""
    src = inspect.getsource(PyExecutor._executor_loop_overlap)
    update = src.find("self._update_requests(")
    pause = src.find("_pause_requests(")
    assert update >= 0 and pause >= 0, (
        f"_update_requests={update}, _pause_requests={pause}; "
        f"both required.")
    assert update < pause, (
        f"_update_requests @{update} must precede _pause_requests "
        f"@{pause} in _executor_loop_overlap, otherwise req.pause folds "
        f"the prompt up to the second-to-last generated token.")
