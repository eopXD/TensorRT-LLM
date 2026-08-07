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
"""Eviction policies for the offline sweep.

Four policies, ordered by how much they are allowed to know:

``lru``            recency only. The engine's current behaviour.
``depth``          block position in the chain. No agent knowledge at all, and
                   cheap to implement for real, so it is the baseline the
                   agent-aware policy actually has to beat.
``anchor-oracle``  told exactly which blocks are anchors. Not implementable --
                   it reads a label the engine cannot compute -- but it bounds
                   what any "keep the anchors" heuristic can achieve.
``agent-aware``    the CacheSage design, scoring with the production
                   ``_agent_aware`` module.

Including ``depth`` and ``anchor-oracle`` is the point. The CacheSage paper
reports no ablation, so it never establishes that learning agent transitions
beats simply noticing that shallow blocks are shared and deep ones are not.
"""

from __future__ import annotations

from collections import OrderedDict

from kvcm_shim import _agent_aware
from replay_sim import Block

if _agent_aware is None:  # pragma: no cover - only when the module is absent
    raise ImportError(
        "kv_cache_manager_v2/_agent_aware.py not found; this PoC adds it. "
        "Run against a checkout that includes the agent-aware change."
    )

RecencyClock = _agent_aware.RecencyClock
SurvivalScorer = _agent_aware.SurvivalScorer
TransitionModel = _agent_aware.TransitionModel


class DepthPolicy:
    """Evict the deepest block first; LRU within a depth bucket.

    Rationale that needs no agent model: a block's chain position predicts how
    many sequences can reach it. Shallow blocks sit on the shared prefix that
    every session of an agent walks through; deep blocks hang off one session's
    divergent history and will never be matched again once that session ends.
    """

    name = "depth"

    def __init__(self, bucket: int = 8) -> None:
        self._bucket = bucket
        self._buckets: dict[int, "OrderedDict[bytes, Block]"] = {}

    def _key(self, block: Block) -> int:
        return block.ordinal // self._bucket

    def push(self, block: Block) -> None:
        self._buckets.setdefault(self._key(block), OrderedDict())[block.key] = block

    def pop(self) -> Block:
        deepest = max(k for k, q in self._buckets.items() if q)
        queue = self._buckets[deepest]
        _, block = queue.popitem(last=False)
        return block

    def remove(self, block: Block) -> None:
        queue = self._buckets.get(self._key(block))
        if queue is not None:
            queue.pop(block.key, None)

    def __len__(self) -> int:
        return sum(len(q) for q in self._buckets.values())

    def observe_request(self, agent: str, session_id: int) -> None:
        pass


class AnchorOraclePolicy:
    """Evict non-anchor blocks before anchor blocks; LRU within each class.

    Upper bound for anchor pinning. Uses ``Block.is_anchor``, which the
    generator knows and the engine does not, so this is a reference line, not a
    candidate implementation.
    """

    name = "anchor-oracle"

    def __init__(self) -> None:
        self._other: "OrderedDict[bytes, Block]" = OrderedDict()
        self._anchor: "OrderedDict[bytes, Block]" = OrderedDict()

    def _queue(self, block: Block) -> "OrderedDict[bytes, Block]":
        return self._anchor if block.is_anchor else self._other

    def push(self, block: Block) -> None:
        self._queue(block)[block.key] = block

    def pop(self) -> Block:
        queue = self._other if self._other else self._anchor
        _, block = queue.popitem(last=False)
        return block

    def remove(self, block: Block) -> None:
        self._queue(block).pop(block.key, None)

    def __len__(self) -> int:
        return len(self._other) + len(self._anchor)

    def observe_request(self, agent: str, session_id: int) -> None:
        pass


class AgentAwarePolicy:
    """CacheSage: per-agent LRU queues ranked by learned survival probability.

    The production ``EvictionPolicy`` protocol has ``push``/``pop``/``remove``
    but no way to re-score a resident block, while survival changes on every
    agent transition. Keeping one LRU queue per agent sidesteps that: recency is
    monotone *within* a queue, so the minimum-scoring block overall is always
    the minimum over the queue heads. That is exact, not an approximation, and
    costs O(|A|) per eviction with no re-push.
    """

    name = "agent-aware"

    def __init__(
        self,
        window: int = 4096,
        threshold: float = 0.01,
        max_hops: int = 8,
        predictive_weight: float = 1.0,
    ) -> None:
        self.model = TransitionModel(window=window)
        self.scorer = SurvivalScorer(
            self.model,
            threshold=threshold,
            max_hops=max_hops,
            predictive_weight=predictive_weight,
        )
        self.clock = RecencyClock()
        self._queues: dict[str, "OrderedDict[bytes, Block]"] = {}
        self._stamps: dict[bytes, int] = {}

    def observe_request(self, agent: str, session_id: int) -> None:
        self.model.observe(agent)

    def push(self, block: Block) -> None:
        self._stamps[block.key] = self.clock.tick()
        self._queues.setdefault(block.agent, OrderedDict())[block.key] = block

    def _heads(self) -> list[tuple[str, Block]]:
        heads = []
        for agent, queue in self._queues.items():
            if queue:
                heads.append((agent, next(iter(queue.values()))))
        return heads

    def pop(self) -> Block:
        heads = self._heads()
        if not heads:
            raise IndexError("no evictable blocks")
        # Normalize recency against the oldest resident block, so residuals are
        # comparable across queues.
        self.clock.note_oldest(min(self._stamps[block.key] for _, block in heads))
        best_agent, best_block, best_score = None, None, None
        for agent, block in heads:
            score = self.scorer.score(agent, self.clock.residual(self._stamps[block.key]))
            if best_score is None or score < best_score:
                best_agent, best_block, best_score = agent, block, score
        self._queues[best_agent].pop(best_block.key, None)
        self._stamps.pop(best_block.key, None)
        return best_block

    def remove(self, block: Block) -> None:
        queue = self._queues.get(block.agent)
        if queue is not None:
            queue.pop(block.key, None)

    def __len__(self) -> int:
        return sum(len(q) for q in self._queues.values())


class CompositePolicy:
    """Score blocks on depth and agent survival together, with tunable weights.

    ``score(b) = w_depth * shallowness(b) + w_pred * p_surv(a_b) + rho_b``

    where ``shallowness = 1 - min(ordinal, D) / D`` is high for blocks near the
    root. Setting ``w_pred = 0`` recovers a smooth ``depth`` policy and
    ``w_depth = 0`` recovers ``agent-aware``, so sweeping the two weights
    measures the *marginal* contribution of the agent model on top of the free
    signal. That is the ablation the CacheSage paper omits.

    Queues are keyed by ``(agent, depth bucket)``; recency is monotone within
    each, so the global minimum is still the minimum over queue heads.
    """

    name = "composite"

    def __init__(
        self,
        w_depth: float = 1.0,
        w_pred: float = 1.0,
        depth_norm: int = 64,
        bucket: int = 8,
        window: int = 4096,
        threshold: float = 0.01,
        max_hops: int = 8,
    ) -> None:
        self._w_depth = w_depth
        self._depth_norm = max(1, depth_norm)
        self._bucket = bucket
        self.model = TransitionModel(window=window)
        self.scorer = SurvivalScorer(
            self.model, threshold=threshold, max_hops=max_hops, predictive_weight=w_pred
        )
        self.clock = RecencyClock()
        self._queues: dict[tuple, "OrderedDict[bytes, Block]"] = {}
        self._stamps: dict[bytes, int] = {}

    def _shallowness(self, ordinal: int) -> float:
        return 1.0 - min(ordinal, self._depth_norm) / self._depth_norm

    def _key(self, block: Block) -> tuple:
        return (block.agent, block.ordinal // self._bucket)

    def observe_request(self, agent: str, session_id: int) -> None:
        self.model.observe(agent)

    def push(self, block: Block) -> None:
        self._stamps[block.key] = self.clock.tick()
        self._queues.setdefault(self._key(block), OrderedDict())[block.key] = block

    def pop(self) -> Block:
        heads = [
            (qkey, next(iter(q.values()))) for qkey, q in self._queues.items() if q
        ]
        if not heads:
            raise IndexError("no evictable blocks")
        self.clock.note_oldest(min(self._stamps[b.key] for _, b in heads))
        best_qkey, best_block, best_score = None, None, None
        for qkey, block in heads:
            score = (
                self._w_depth * self._shallowness(block.ordinal)
                + self.scorer.score(block.agent, self.clock.residual(self._stamps[block.key]))
            )
            if best_score is None or score < best_score:
                best_qkey, best_block, best_score = qkey, block, score
        self._queues[best_qkey].pop(best_block.key, None)
        self._stamps.pop(best_block.key, None)
        return best_block

    def remove(self, block: Block) -> None:
        queue = self._queues.get(self._key(block))
        if queue is not None:
            queue.pop(block.key, None)

    def __len__(self) -> int:
        return sum(len(q) for q in self._queues.values())


def registry() -> dict:
    from replay_sim import LRUPolicy

    return {
        "lru": LRUPolicy,
        "depth": DepthPolicy,
        "anchor-oracle": AnchorOraclePolicy,
        "agent-aware": AgentAwarePolicy,
        "composite": CompositePolicy,
    }
