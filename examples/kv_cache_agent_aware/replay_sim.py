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
"""CUDA-free replay simulator for KV-cache prefix reuse under a block budget.

Models exactly the part of ``kv_cache_manager_v2`` that decides hit rate: the
content-addressed block chain, prefix matching that stops at the first miss, and
a pluggable eviction policy over the resident set. Block keys come from the
production :func:`sequence_to_blockchain_keys`, so block boundaries and match
semantics track the engine.

What is deliberately *not* modelled, because none of it moves hit rate:
GPU memory layout, attention kernels, copy engines, host/disk tiers, and time.
Requests are interleaved round-robin across ``concurrency`` sessions and each
one completes before the next is issued, so exactly one request holds pins at a
time. The interleaving is what creates the cache pressure that eviction order
has to cope with; per-request wall-clock overlap does not change which blocks
are resident when the next match runs.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol

from agent_trace import Turn, Workload
from kvcm_shim import sequence_to_blockchain_keys

# The reuse scope namespaces the radix tree (cache salt, LoRA task id, ...).
# A single tenant with no salt is the common serving case.
DEFAULT_REUSE_SCOPE: tuple = (None, None, None)


@dataclass
class Block:
    key: bytes
    # Agent that first touched this block. Stable across sessions for anchor
    # blocks, because identical anchor tokens hash identically.
    agent: str
    session_id: int
    ordinal: int
    is_anchor: bool
    last_used: int = 0


class EvictionPolicy(Protocol):
    """Mirrors ``kv_cache_manager_v2._eviction_controller.EvictionPolicy``."""

    def push(self, block: Block) -> None: ...

    def pop(self) -> Block: ...

    def remove(self, block: Block) -> None: ...

    def __len__(self) -> int: ...


class LRUPolicy:
    """Baseline. Matches ``LRUEvictionPolicy`` semantics: FIFO on push order."""

    name = "lru"

    def __init__(self) -> None:
        self._queue: "OrderedDict[bytes, Block]" = OrderedDict()

    def push(self, block: Block) -> None:
        self._queue[block.key] = block

    def pop(self) -> Block:
        _, block = self._queue.popitem(last=False)
        return block

    def remove(self, block: Block) -> None:
        self._queue.pop(block.key, None)

    def __len__(self) -> int:
        return len(self._queue)

    # Hooks the agent-aware policy uses; no-ops here so the driver is uniform.
    def observe_request(self, agent: str, session_id: int) -> None:
        pass


@dataclass
class SimResult:
    policy: str
    budget_blocks: int
    tokens_per_block: int
    concurrency: int
    prompt_tokens: int = 0
    matched_tokens: int = 0
    anchor_tokens: int = 0
    anchor_matched_tokens: int = 0
    evictions: int = 0
    anchor_evictions: int = 0
    oversized_requests: int = 0
    # Counted directly rather than derived from token counts: prompt_tokens
    # includes the trailing partial block, which is never part of the chain, so
    # dividing it by tokens_per_block would inflate the denominator and
    # understate the hit rate relative to what the engine reports.
    total_blocks: int = 0
    matched_blocks: int = 0
    per_agent: dict = field(default_factory=dict)

    @property
    def hit_rate(self) -> float:
        return self.matched_tokens / max(1, self.prompt_tokens)

    @property
    def block_hit_rate(self) -> float:
        """Matches the engine's reused / (reused + missed) over whole blocks."""
        return self.matched_blocks / max(1, self.total_blocks)

    @property
    def anchor_hit_rate(self) -> float:
        return self.anchor_matched_tokens / max(1, self.anchor_tokens)

    def __str__(self) -> str:
        return (
            f"{self.policy:<16} budget={self.budget_blocks:>5} "
            f"hit={self.hit_rate:6.2%} block_hit={self.block_hit_rate:6.2%} "
            f"anchor_hit={self.anchor_hit_rate:6.2%} "
            f"evict={self.evictions:>7} anchor_evict={self.anchor_evictions:>6}"
        )


def _chain(turn: Turn, tokens_per_block: int) -> list[tuple[bytes, bool]]:
    """Full blocks of a turn's prompt as ``(key, is_anchor)``.

    The root sentinel and the trailing partial block are dropped: the root holds
    no tokens, and a partial block is not reusable by default in the engine.
    """
    pairs = list(sequence_to_blockchain_keys(tokens_per_block, DEFAULT_REUSE_SCOPE, turn.tokens))
    out: list[tuple[bytes, bool]] = []
    for i, (token_block, key) in enumerate(pairs):
        if i == 0 or len(token_block) < tokens_per_block:
            continue
        # Block i covers tokens [(i-1)*tpb, i*tpb). It is an anchor block only
        # if it lies wholly inside the agent's anchor span.
        end_token = i * tokens_per_block
        out.append((key, end_token <= turn.anchor_len))
    return out


def simulate(
    workload: Workload,
    budget_blocks: int,
    policy_factory: Callable[[], EvictionPolicy] = LRUPolicy,
    tokens_per_block: int = 32,
    concurrency: int = 4,
) -> SimResult:
    policy = policy_factory()
    result = SimResult(
        policy=getattr(policy, "name", policy.__class__.__name__),
        budget_blocks=budget_blocks,
        tokens_per_block=tokens_per_block,
        concurrency=concurrency,
    )
    resident: dict[bytes, Block] = {}
    clock = 0

    for turn in _interleave(workload, concurrency):
        clock += 1
        chain = _chain(turn, tokens_per_block)
        result.prompt_tokens += turn.prompt_len
        anchor_blocks = sum(1 for _, is_anchor in chain if is_anchor)
        result.anchor_tokens += anchor_blocks * tokens_per_block

        # Prefix match: stops at the first miss, exactly like the radix tree.
        matched = 0
        for key, _ in chain:
            if key not in resident:
                break
            matched += 1
        result.matched_tokens += matched * tokens_per_block
        result.anchor_matched_tokens += min(matched, anchor_blocks) * tokens_per_block
        result.total_blocks += len(chain)
        result.matched_blocks += matched

        agent_stats = result.per_agent.setdefault(
            turn.agent, {"prompt_tokens": 0, "matched_tokens": 0, "turns": 0}
        )
        agent_stats["prompt_tokens"] += turn.prompt_len
        agent_stats["matched_tokens"] += matched * tokens_per_block
        agent_stats["turns"] += 1

        if hasattr(policy, "observe_request"):
            policy.observe_request(turn.agent, turn.session_id)

        if len(chain) > budget_blocks:
            # A request that cannot fit in the whole cache would be chunked or
            # rejected by the scheduler. Count it rather than silently distort
            # the hit rate.
            result.oversized_requests += 1

        # Pin everything this request touches, so allocation never evicts a
        # block the request itself just matched.
        pinned: list[Block] = []
        for key, _ in chain:
            block = resident.get(key)
            if block is not None:
                block.last_used = clock
                policy.remove(block)
                pinned.append(block)

        for ordinal, (key, is_anchor) in enumerate(chain):
            if key in resident:
                continue
            while len(resident) + 1 > budget_blocks:
                if len(policy) == 0:
                    break  # everything is pinned; cannot make room
                victim = policy.pop()
                del resident[victim.key]
                result.evictions += 1
                if victim.is_anchor:
                    result.anchor_evictions += 1
            if len(resident) + 1 > budget_blocks:
                break
            block = Block(
                key=key,
                agent=turn.agent,
                session_id=turn.session_id,
                ordinal=ordinal,
                is_anchor=is_anchor,
                last_used=clock,
            )
            resident[key] = block
            pinned.append(block)

        # Unpin in chain order, so the LRU queue orders blocks by position and
        # the head of the queue is the oldest block of the least recent request.
        for block in pinned:
            policy.push(block)

    return result


def _interleave(workload: Workload, concurrency: int) -> Iterable[Turn]:
    """Round-robin ``concurrency`` sessions, in session order.

    Sessions are admitted in id order; when one finishes another takes its slot.
    With ``concurrency=1`` this degenerates to running each session to
    completion, which is the paper's synthetic-workload setting.
    """
    from agent_trace import iter_by_session

    pending = [turns for _, turns in iter_by_session(workload)]
    if concurrency <= 1:
        for turns in pending:
            yield from turns
        return

    next_session = 0
    active: list[list[Turn]] = []
    while next_session < len(pending) and len(active) < concurrency:
        active.append(list(pending[next_session]))
        next_session += 1

    while active:
        still_active: list[list[Turn]] = []
        for turns in active:
            yield turns.pop(0)
            if turns:
                still_active.append(turns)
            elif next_session < len(pending):
                still_active.append(list(pending[next_session]))
                next_session += 1
        active = still_active


def reuse_ceiling(
    workload: Workload, tokens_per_block: int = 32, concurrency: int = 4
) -> SimResult:
    """Hit rate with an unbounded cache: the upper bound any policy can reach."""
    total_blocks = sum(len(_chain(t, tokens_per_block)) for t in workload.turns)
    return simulate(
        workload,
        budget_blocks=total_blocks + 1,
        policy_factory=LRUPolicy,
        tokens_per_block=tokens_per_block,
        concurrency=concurrency,
    )


def working_set_blocks(workload: Workload, tokens_per_block: int = 32) -> int:
    """Distinct blocks the workload ever touches."""
    seen: set[bytes] = set()
    for turn in workload.turns:
        for key, _ in _chain(turn, tokens_per_block):
            seen.add(key)
    return len(seen)


def anchor_working_set_blocks(workload: Workload, tokens_per_block: int = 32) -> int:
    """Distinct anchor blocks: the cross-session-reusable core of the workload."""
    seen: set[bytes] = set()
    for turn in workload.turns:
        for key, is_anchor in _chain(turn, tokens_per_block):
            if is_anchor:
                seen.add(key)
    return len(seen)
