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
"""Tests for ``kv_cache_manager_v2._agent_aware``.

The module is stdlib-only by construction, so these run without a GPU and
without the C++ bindings. They are loaded through the same file-path shim the
offline harness uses, because importing the package ``__init__`` would pull in
CUDA.
"""

import importlib.util
import pathlib
import sys
import types
import unittest

_PKG = "_kvcm2_pure_test"


def _load_agent_aware():
    if f"{_PKG}._agent_aware" in sys.modules:
        return sys.modules[f"{_PKG}._agent_aware"]
    here = pathlib.Path(__file__).resolve()
    src = None
    for parent in here.parents:
        candidate = parent / "tensorrt_llm" / "runtime" / "kv_cache_manager_v2"
        if candidate.is_dir():
            src = candidate
            break
    assert src is not None, f"could not locate kv_cache_manager_v2 above {here}"

    pkg = types.ModuleType(_PKG)
    pkg.__path__ = [str(src)]
    sys.modules[_PKG] = pkg
    for name in ("_common", "_agent_aware"):
        spec = importlib.util.spec_from_file_location(f"{_PKG}.{name}", src / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{_PKG}.{name}"] = module
        spec.loader.exec_module(module)
        setattr(pkg, name, module)
    return sys.modules[f"{_PKG}._agent_aware"]


aa = _load_agent_aware()


class TestDeriveAgentId(unittest.TestCase):
    def test_same_prefix_same_identity(self):
        keys = [bytes([i]) * 32 for i in range(12)]
        self.assertEqual(aa.derive_agent_id(keys), aa.derive_agent_id(list(keys)))

    def test_differs_when_sampled_window_differs(self):
        a = [bytes([i]) * 32 for i in range(12)]
        b = list(a)
        b[7] = b"\xff" * 32  # inside skip=4/take=4 window
        self.assertNotEqual(aa.derive_agent_id(a), aa.derive_agent_id(b))

    def test_ignores_content_past_the_window(self):
        """Session-specific tail must not change the agent identity.

        This is the property that makes the identity stable: two runs of one
        agent share the anchor but diverge afterwards, and they must still map
        to the same agent.
        """
        a = [bytes([i]) * 32 for i in range(12)]
        b = a[:8] + [b"\xaa" * 32] * 4
        self.assertEqual(aa.derive_agent_id(a), aa.derive_agent_id(b))

    def test_short_chain_returns_none(self):
        self.assertIsNone(aa.derive_agent_id([bytes([i]) * 32 for i in range(3)]))
        self.assertIsNone(aa.derive_agent_id([]))

    def test_rejects_bad_parameters(self):
        keys = [b"\x00" * 32] * 10
        with self.assertRaises(ValueError):
            aa.derive_agent_id(keys, skip=-1)
        with self.assertRaises(ValueError):
            aa.derive_agent_id(keys, take=0)


class TestTransitionModel(unittest.TestCase):
    def test_maximum_likelihood_estimate(self):
        m = aa.TransitionModel()
        for _ in range(3):
            m.observe("a")
            m.observe("b")
        m.observe("a")
        m.observe("c")
        # From "a": 3 transitions to "b", 1 to "c".
        self.assertAlmostEqual(m.probability("a", "b"), 0.75)
        self.assertAlmostEqual(m.probability("a", "c"), 0.25)
        self.assertEqual(m.predict_next("a"), "b")

    def test_repeated_agent_is_not_a_self_transition(self):
        m = aa.TransitionModel()
        m.observe("a")
        self.assertFalse(m.observe("a"))
        m.observe("b")
        self.assertEqual(m.probability("a", "a"), 0.0)
        self.assertAlmostEqual(m.probability("a", "b"), 1.0)

    def test_window_evicts_old_transitions(self):
        m = aa.TransitionModel(window=2)
        m.observe("a")
        m.observe("b")  # a->b
        m.observe("a")  # b->a
        m.observe("c")  # a->c, evicts a->b
        self.assertEqual(m.probability("a", "b"), 0.0)
        self.assertAlmostEqual(m.probability("a", "c"), 1.0)

    def test_unknown_source_has_no_mass(self):
        m = aa.TransitionModel()
        self.assertEqual(m.probability("nobody", "x"), 0.0)
        self.assertIsNone(m.predict_next("nobody"))

    def test_none_is_ignored(self):
        m = aa.TransitionModel()
        self.assertFalse(m.observe(None))
        self.assertIsNone(m.current)

    def test_rejects_bad_window(self):
        with self.assertRaises(ValueError):
            aa.TransitionModel(window=0)


class TestSurvivalScorer(unittest.TestCase):
    def _chain_model(self):
        """A -> b -> c -> d, deterministic."""
        m = aa.TransitionModel()
        for _ in range(10):
            for agent in ("a", "b", "c", "d"):
                m.observe(agent)
        return m

    def test_current_agent_scores_highest(self):
        m = self._chain_model()
        m.observe("a")
        s = aa.SurvivalScorer(m, max_hops=4)
        self.assertEqual(s.survival("a"), 1.0)
        self.assertGreater(s.survival("b"), s.survival("c"))

    def test_hop_distance_is_monotone(self):
        m = self._chain_model()
        m.observe("a")
        s = aa.SurvivalScorer(m, max_hops=4)
        values = [s.survival(x) for x in ("a", "b", "c", "d")]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_unknown_agent_scores_zero(self):
        m = self._chain_model()
        m.observe("a")
        s = aa.SurvivalScorer(m, max_hops=4)
        self.assertEqual(s.survival("never-seen"), 0.0)
        self.assertEqual(s.survival(None), 0.0)

    def test_refreshes_when_current_agent_changes(self):
        m = self._chain_model()
        m.observe("a")
        s = aa.SurvivalScorer(m, max_hops=4)
        self.assertEqual(s.survival("a"), 1.0)
        m.observe("b")
        self.assertEqual(s.survival("b"), 1.0)
        self.assertLess(s.survival("a"), 1.0)

    def test_degenerates_to_lru_without_structure(self):
        """No observed transitions => score is the recency residual, i.e. LRU.

        This is the safety property: the policy cannot rank worse than LRU on a
        workload with no agent structure.
        """
        s = aa.SurvivalScorer(aa.TransitionModel())
        for residual in (0.0, 0.25, 1.0):
            self.assertEqual(s.score("anything", residual), residual)

    def test_single_agent_scores_are_constant(self):
        m = aa.TransitionModel()
        m.observe("only")
        s = aa.SurvivalScorer(m)
        # One agent means one survival value, so ordering is decided purely by
        # the recency residual.
        self.assertAlmostEqual(
            s.score("only", 0.2) - s.score("only", 0.1),
            0.2 - 0.1,
        )

    def test_threshold_prunes_weak_edges(self):
        m = aa.TransitionModel()
        for _ in range(99):
            m.observe("a")
            m.observe("b")
        m.observe("a")
        m.observe("rare")  # 1/100 = 0.01
        m.observe("a")
        strict = aa.SurvivalScorer(m, threshold=0.5, max_hops=2)
        loose = aa.SurvivalScorer(m, threshold=0.001, max_hops=2)
        self.assertEqual(strict.survival("rare"), 0.0)
        self.assertGreater(loose.survival("rare"), 0.0)

    def test_rejects_bad_max_hops(self):
        with self.assertRaises(ValueError):
            aa.SurvivalScorer(aa.TransitionModel(), max_hops=0)


class TestRecencyClock(unittest.TestCase):
    def test_residual_is_normalized_and_monotone(self):
        c = aa.RecencyClock()
        stamps = [c.tick() for _ in range(10)]
        c.note_oldest(stamps[0])
        residuals = [c.residual(s) for s in stamps]
        self.assertEqual(residuals[0], 0.0)
        self.assertEqual(residuals[-1], 1.0)
        self.assertEqual(residuals, sorted(residuals))

    def test_residual_is_zero_before_any_span(self):
        self.assertEqual(aa.RecencyClock().residual(0), 0.0)

    def test_residual_is_clamped(self):
        c = aa.RecencyClock()
        for _ in range(5):
            c.tick()
        c.note_oldest(2)
        self.assertEqual(c.residual(99), 1.0)
        self.assertEqual(c.residual(-99), 0.0)


class TestSurvivalToPriority(unittest.TestCase):
    def test_endpoints_and_monotonicity(self):
        lo = aa.survival_to_priority(0.0)
        hi = aa.survival_to_priority(1.0)
        self.assertLess(lo, hi)
        self.assertLessEqual(aa.survival_to_priority(0.4), aa.survival_to_priority(0.6))

    def test_clamps_out_of_range_input(self):
        self.assertEqual(aa.survival_to_priority(-1.0), aa.survival_to_priority(0.0))
        self.assertEqual(aa.survival_to_priority(2.0), aa.survival_to_priority(1.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
