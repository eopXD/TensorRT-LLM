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
"""Agent-aware scoring for KV-cache eviction.

In a multi-agent deployment the request stream is a walk over a small set of
agents, each carrying a fixed prompt anchor (system prompt + tool definitions +
few-shot examples). The anchor is byte-identical on every appearance, so under
content-addressed prefix caching all sessions of one agent share a single copy
of those blocks. Whether that copy survives to the agent's next appearance is
decided purely by eviction order, and recency alone does not know that an agent
which just ran is about to run again.

This module supplies the three pieces needed to make eviction order agent-aware,
and nothing else. It is stdlib-only and imports no CUDA, so it can be exercised
offline and unit-tested without a GPU:

* :func:`derive_agent_id` -- a stable identity for a block chain.
* :class:`TransitionModel` -- online first-order Markov estimate over agents.
* :class:`SurvivalScorer` -- reachability-based survival proxy, plus the
  combined ``score`` used to rank eviction candidates.

The queue plumbing that consumes these lives in ``_eviction_controller.py``;
keeping it separate is what lets the offline simulator score blocks with the
same code the engine runs.

Degeneracy is a designed property, not a fallback branch. With one agent, no
observed transitions, or a flat transition matrix, every survival value is equal
and ``score`` reduces to the recency residual -- i.e. exactly LRU. The policy
therefore cannot rank worse than LRU on a workload with no agent structure.
"""

from __future__ import annotations

from collections import deque
from typing import Hashable, Sequence

from ._common import PRIORITY_MAX, PRIORITY_MIN, Priority

# An agent identity. In the engine this is a digest; the offline harness uses
# plain strings. Any hashable works.
AgentId = Hashable

# Defaults follow the reference design. ``SKIP``/``TAKE`` are tuned so the
# sampled window lands inside the system+tools anchor for a Llama-3.1-class
# chat template: the first few blocks are template boilerplate shared by every
# agent, so skipping them is what makes the identity discriminative.
DEFAULT_SKIP_BLOCKS = 4
DEFAULT_TAKE_BLOCKS = 4

# Edges below this probability are dropped before the reachability search. At
# 0.01 this preserves essentially all observed mass while keeping the graph
# sparse enough that the search stays cheap.
DEFAULT_EDGE_THRESHOLD = 0.01

# Hop count at which an agent is treated as "will not be needed soon". Also the
# normalizer for the survival proxy.
DEFAULT_MAX_HOPS = 8

# Sliding window of transitions retained. Bounds both memory and adaptation lag.
DEFAULT_WINDOW = 4096

# Weight on the predictive term relative to the recency residual. Both terms are
# in [0, 1], so 1.0 gives them equal authority.
DEFAULT_PREDICTIVE_WEIGHT = 1.0


def derive_agent_id(
    block_keys: Sequence[bytes],
    skip: int = DEFAULT_SKIP_BLOCKS,
    take: int = DEFAULT_TAKE_BLOCKS,
) -> AgentId:
    """Derive a stable agent identity from a sequence's block-key chain.

    ``block_keys`` is the chain produced by
    :func:`._cache_key.sequence_to_blockchain_keys`, *excluding* the root
    sentinel. Because each key already hashes every preceding block, a single
    key at a fixed depth identifies the whole prefix up to that depth; taking a
    window of them keeps the identity stable when a short prompt has fewer
    blocks than ``skip + take``.

    Returns ``None`` when the chain is too short to sample -- callers treat that
    as "unknown agent", which scores as unpredictable rather than as a distinct
    agent, so short requests do not pollute the transition model.
    """
    if skip < 0 or take <= 0:
        raise ValueError(f"invalid skip/take: {skip}/{take}")
    window = block_keys[skip : skip + take]
    if not window:
        return None
    # The deepest sampled key already commits to every shallower one, so it is a
    # sufficient identity on its own. Returning it directly avoids a second hash
    # on a path that runs once per sequence.
    return window[-1]


class TransitionModel:
    """Online first-order Markov estimate over agent identities.

    Maintains sliding-window pairwise counts ``n(a, b)`` and returns the
    maximum-likelihood estimate ``W(b|a) = n(a, b) / n(a)``. Both ``observe``
    and ``probability`` are O(1); state is O(|A|^2).
    """

    __slots__ = ("_window", "_events", "_pair_counts", "_from_counts", "_previous", "_current")

    def __init__(self, window: int = DEFAULT_WINDOW) -> None:
        if window <= 0:
            raise ValueError(f"window must be positive, got {window}")
        self._window = window
        self._events: deque = deque()
        self._pair_counts: dict = {}
        self._from_counts: dict = {}
        self._previous: AgentId = None
        self._current: AgentId = None

    @property
    def current(self) -> AgentId:
        return self._current

    @property
    def num_agents(self) -> int:
        return len(self._from_counts)

    def observe(self, agent: AgentId) -> bool:
        """Record that ``agent`` is now running. Returns True if it changed.

        A repeated agent is not recorded as a self-transition: consecutive
        requests from the same agent are one visit for prediction purposes, and
        counting them would inflate the self-edge until it dominates every row.
        """
        if agent is None or agent == self._current:
            return False
        self._previous, self._current = self._current, agent
        if self._previous is not None:
            self._add(self._previous, agent, +1)
            self._events.append((self._previous, agent))
            while len(self._events) > self._window:
                old_from, old_to = self._events.popleft()
                self._add(old_from, old_to, -1)
        return True

    def _add(self, src: AgentId, dst: AgentId, delta: int) -> None:
        key = (src, dst)
        count = self._pair_counts.get(key, 0) + delta
        if count <= 0:
            self._pair_counts.pop(key, None)
        else:
            self._pair_counts[key] = count
        total = self._from_counts.get(src, 0) + delta
        if total <= 0:
            self._from_counts.pop(src, None)
        else:
            self._from_counts[src] = total

    def probability(self, src: AgentId, dst: AgentId) -> float:
        total = self._from_counts.get(src, 0)
        if total <= 0:
            return 0.0
        return self._pair_counts.get((src, dst), 0) / total

    def successors(self, src: AgentId, threshold: float) -> list:
        total = self._from_counts.get(src, 0)
        if total <= 0:
            return []
        cutoff = threshold * total
        return [dst for (s, dst), n in self._pair_counts.items() if s == src and n >= cutoff]

    def predict_next(self, src: AgentId = None) -> AgentId:
        """Most likely successor of ``src`` (default: the current agent)."""
        src = self._current if src is None else src
        if src is None or self._from_counts.get(src, 0) <= 0:
            return None
        best, best_n = None, 0
        for (s, dst), n in self._pair_counts.items():
            if s == src and n > best_n:
                best, best_n = dst, n
        return best

    def state_bytes(self) -> int:
        """Rough resident size, for the overhead accounting in a writeup."""
        return 16 * (len(self._pair_counts) + len(self._from_counts)) + 16 * len(self._events)


class SurvivalScorer:
    """Reachability-based survival proxy over the learned transition graph.

    Exact survival probability over a K-step horizon costs O(|A|^3 K). The
    proxy instead thresholds the transition matrix into a digraph, takes the
    hop count ``E[a]`` from the current agent by breadth-first search, and maps
    it to ``p_surv(a) = 1 - min(E[a], max_hops) / max_hops``. This is monotone
    in the exact quantity, which is all that ranking needs, and costs one BFS
    per agent change rather than one per eviction.
    """

    __slots__ = ("_model", "_threshold", "_max_hops", "_weight", "_survival", "_source")

    def __init__(
        self,
        model: TransitionModel,
        threshold: float = DEFAULT_EDGE_THRESHOLD,
        max_hops: int = DEFAULT_MAX_HOPS,
        predictive_weight: float = DEFAULT_PREDICTIVE_WEIGHT,
    ) -> None:
        if max_hops <= 0:
            raise ValueError(f"max_hops must be positive, got {max_hops}")
        self._model = model
        self._threshold = threshold
        self._max_hops = max_hops
        self._weight = predictive_weight
        self._survival: dict = {}
        self._source: AgentId = None

    def invalidate(self) -> None:
        self._survival = {}
        self._source = None

    def _refresh(self) -> None:
        """Recompute hop counts from the current agent. One BFS, O(|A| + |E|)."""
        source = self._model.current
        self._source = source
        self._survival = {}
        if source is None:
            return
        # The current agent is running now, so its blocks are maximally worth
        # keeping: distance 0.
        distance = {source: 0}
        queue = deque([source])
        while queue:
            node = queue.popleft()
            d = distance[node]
            if d >= self._max_hops:
                continue
            for successor in self._model.successors(node, self._threshold):
                if successor not in distance:
                    distance[successor] = d + 1
                    queue.append(successor)
        max_hops = self._max_hops
        for agent, d in distance.items():
            self._survival[agent] = 1.0 - min(d, max_hops) / max_hops

    def survival(self, agent: AgentId) -> float:
        """``p_surv`` in [0, 1]. Unknown or unreachable agents score 0."""
        if self._source is not self._model.current:
            self._refresh()
        if agent is None:
            return 0.0
        return self._survival.get(agent, 0.0)

    def score(self, agent: AgentId, recency_residual: float) -> float:
        """Eviction rank for a block; the *lowest* score is evicted first.

        ``recency_residual`` is the block's position in the global recency
        order, normalized to [0, 1] with 0 the oldest. When no agent structure
        has been learned every survival term is 0 and this returns the recency
        residual unchanged, reproducing LRU order exactly.
        """
        return self._weight * self.survival(agent) + recency_residual


class RecencyClock:
    """Monotone stamp source plus the normalization needed for the residual.

    Blocks carry the stamp they were last pushed with. The residual is
    ``(stamp - oldest) / (now - oldest)``, so it is comparable across the
    per-agent queues without maintaining a single global list.
    """

    __slots__ = ("_now", "_oldest")

    def __init__(self) -> None:
        self._now = 0
        self._oldest = 0

    def tick(self) -> int:
        self._now += 1
        return self._now

    @property
    def now(self) -> int:
        return self._now

    def note_oldest(self, stamp: int) -> None:
        self._oldest = stamp

    def residual(self, stamp: int) -> float:
        span = self._now - self._oldest
        if span <= 0:
            return 0.0
        return min(1.0, max(0.0, (stamp - self._oldest) / span))


def survival_to_priority(survival: float) -> Priority:
    """Map ``p_surv`` onto the engine's retention priority scale.

    Provided for the cheap integration path, where agent awareness is expressed
    through the existing per-block priority instead of a new eviction policy.
    Note the semantics differ: priority is fixed when a page is created, while
    survival changes on every agent transition, so this path cannot re-rank a
    resident block. It is useful for an A/B against the full policy, not as a
    replacement for it.
    """
    span = PRIORITY_MAX - PRIORITY_MIN
    return Priority(PRIORITY_MIN + int(round(min(1.0, max(0.0, survival)) * span)))
