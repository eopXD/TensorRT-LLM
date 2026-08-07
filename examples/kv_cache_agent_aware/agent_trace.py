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
"""Synthetic multi-agent workload generator.

Emits token-level request traces with the prompt structure a supervisor-style
multi-agent framework (AutoGen ``SelectorGroupChat``, LangGraph, MetaGPT)
produces, so that KV-cache reuse experiments see realistic prefix sharing.

Prompt layout for agent ``a`` on turn ``t`` of a session::

    anchor(a)            agent-specific: system prompt + tool defs + few-shots
    history(<t)          session-specific: every prior (speaker, message) pair
    task(a, t)           this turn's instruction

Two consequences drive every KV-cache result downstream, and both are real
properties of the frameworks, not artifacts of this generator:

1. ``anchor(a)`` is the only span reusable *across sessions*. It is byte
   identical every time agent ``a`` runs, so under content-addressed prefix
   caching all sessions share one copy of those blocks.
2. Within one session the agents *cannot* share history blocks with each other.
   History is identical text, but it sits behind a different ``anchor(a)``
   prefix per agent, and prefix matching is anchored at token 0. So a session
   with ``|A|`` active agents grows ``|A|`` divergent branches off the root.

Agent order within a session follows a first-order Markov chain. The chain's
predictability is a tunable: :func:`make_transition_matrix` binary-searches the
mix between a structured chain and the uniform chain to hit a target relative
conditional entropy reduction ``R = 1 - H(X_{t+1}|X_t) / H(X_{t+1})``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterator, Sequence

# Token ids are drawn from disjoint ranges so that distinct spans never collide
# in the block hash. Real vocabularies are ~128k; staying under that keeps the
# ids plausible for a Llama-class tokenizer.
VOCAB_SIZE = 128_000


@dataclass(frozen=True)
class AgentSpec:
    """One agent's identity and prompt-shape parameters."""

    name: str
    # System prompt + tool definitions + few-shot examples. Fixed content, so
    # these tokens hash identically on every appearance of this agent.
    anchor_tokens: int
    # Mean tokens this agent emits per turn, and mean instruction length.
    mean_reply_tokens: int
    mean_task_tokens: int


@dataclass
class Turn:
    """One generated request."""

    session_id: int
    turn_index: int
    agent: str
    tokens: list[int]
    # Span boundaries within ``tokens``, for phi accounting.
    anchor_len: int
    history_len: int
    task_len: int
    # Tokens the agent emits; appended to session history for the next turn.
    reply_tokens: int

    @property
    def prompt_len(self) -> int:
        return len(self.tokens)


@dataclass
class Workload:
    agents: list[AgentSpec]
    transition: list[list[float]]
    initial: list[float]
    turns: list[Turn]
    # Diagnostics, so a trace can be rejected before any policy work happens.
    phi_first_turn: float = 0.0
    phi_all_turns: float = 0.0
    entropy_reduction: float = 0.0
    mean_turns_per_session: float = 0.0
    stats: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Transition matrix
# ---------------------------------------------------------------------------


def _stationary(matrix: Sequence[Sequence[float]], iters: int = 2000) -> list[float]:
    """Power-iterate to the stationary distribution."""
    n = len(matrix)
    pi = [1.0 / n] * n
    for _ in range(iters):
        nxt = [0.0] * n
        for i in range(n):
            pi_i = pi[i]
            if pi_i == 0.0:
                continue
            row = matrix[i]
            for j in range(n):
                nxt[j] += pi_i * row[j]
        total = sum(nxt)
        pi = [v / total for v in nxt]
    return pi


def _entropy(p: Sequence[float]) -> float:
    return -sum(v * math.log2(v) for v in p if v > 0.0)


def entropy_reduction(matrix: Sequence[Sequence[float]]) -> float:
    """Relative conditional entropy reduction ``1 - H(X_{t+1}|X_t) / H(X_{t+1})``.

    This is the quantity the CacheSage paper reports as ``R in [0.40, 0.48]``
    across its five workloads. ``R = 0`` means the next agent is independent of
    the current one (nothing to learn, agent-aware eviction must degenerate to
    LRU); ``R = 1`` means the chain is deterministic.
    """
    pi = _stationary(matrix)
    h_marginal = _entropy(pi)
    if h_marginal <= 0.0:
        return 0.0
    h_conditional = sum(pi[i] * _entropy(matrix[i]) for i in range(len(matrix)))
    return 1.0 - h_conditional / h_marginal


def make_transition_matrix(
    num_agents: int,
    target_r: float,
    rng: random.Random,
    out_degree: int = 2,
    tol: float = 1e-3,
) -> list[list[float]]:
    """Build a row-stochastic matrix whose entropy reduction is ``target_r``.

    A structured chain (each agent hands off to a small set of successors) is
    blended with the uniform chain: ``M(w) = w * structured + (1-w) * uniform``.
    ``R`` is monotone in ``w`` (``w=0`` gives ``R=0``), so a bisection on ``w``
    hits any achievable target.
    """
    if not 0.0 <= target_r < 1.0:
        raise ValueError(f"target_r must be in [0, 1), got {target_r}")

    uniform = [[1.0 / num_agents] * num_agents for _ in range(num_agents)]

    # Structured chain: a dominant successor plus a few weaker ones. The
    # dominant edge is the "Planner -> Coder" style handoff; the weak edges keep
    # every state reachable so the chain stays irreducible.
    #
    # Per-workload edge jitter is applied here, before the bisection, not after.
    # The paper measures 27-47 pp swings on individual edges across workloads,
    # which is why the transition model must be learned online rather than
    # configured -- but jittering a solved matrix would move R back off target.
    structured = [[0.0] * num_agents for _ in range(num_agents)]
    for i in range(num_agents):
        successors = [(i + 1 + k) % num_agents for k in range(out_degree)]
        weights = [0.70] + [0.30 / max(1, out_degree - 1)] * (out_degree - 1)
        for j, w in zip(successors, weights):
            structured[i][j] += w
        for j in range(num_agents):
            structured[i][j] *= math.exp(rng.gauss(0.0, 0.35))
        total = sum(structured[i])
        structured[i] = [v / total for v in structured[i]]

    def blend(w: float) -> list[list[float]]:
        return [
            [w * structured[i][j] + (1.0 - w) * uniform[i][j] for j in range(num_agents)]
            for i in range(num_agents)
        ]

    lo, hi = 0.0, 1.0
    if entropy_reduction(blend(hi)) < target_r:
        raise ValueError(
            f"target_r={target_r} unreachable with num_agents={num_agents}, "
            f"out_degree={out_degree}; max is {entropy_reduction(blend(1.0)):.3f}"
        )
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if entropy_reduction(blend(mid)) < target_r:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return blend(0.5 * (lo + hi))


# ---------------------------------------------------------------------------
# Token span synthesis
# ---------------------------------------------------------------------------


class _SpanAllocator:
    """Hands out disjoint, deterministic token-id ranges.

    Distinct spans must not share token ids, or unrelated blocks would hash
    equal and the simulator would report reuse that a real deployment cannot
    get. Each span gets its own base offset and is filled with a simple
    deterministic walk, wrapping within the vocabulary.
    """

    def __init__(self) -> None:
        self._next_base = 1000

    def new_span(self, length: int) -> list[int]:
        base = self._next_base
        self._next_base = (self._next_base + max(length, 1) * 7 + 13) % (VOCAB_SIZE - 4096)
        return [1000 + (base + i * 31) % (VOCAB_SIZE - 2000) for i in range(length)]


def _sample_len(rng: random.Random, mean: int, spread: float = 0.30) -> int:
    """Log-normal-ish length sample, floored at 1."""
    return max(1, int(mean * math.exp(rng.gauss(0.0, spread))))


DEFAULT_ROSTER: list[AgentSpec] = [
    # Anchor sizes reflect real agent system prompts: a tool-heavy agent carries
    # more tool-schema tokens than a pure-reasoning one.
    AgentSpec("planner", anchor_tokens=1400, mean_reply_tokens=220, mean_task_tokens=180),
    AgentSpec("researcher", anchor_tokens=2100, mean_reply_tokens=380, mean_task_tokens=160),
    AgentSpec("coder", anchor_tokens=2600, mean_reply_tokens=520, mean_task_tokens=200),
    AgentSpec("tester", anchor_tokens=1900, mean_reply_tokens=300, mean_task_tokens=170),
    AgentSpec("critic", anchor_tokens=1100, mean_reply_tokens=260, mean_task_tokens=150),
    AgentSpec("summarizer", anchor_tokens=900, mean_reply_tokens=200, mean_task_tokens=140),
]


def generate(
    num_sessions: int = 50,
    turns_per_session: int = 20,
    roster: Sequence[AgentSpec] | None = None,
    target_r: float = 0.44,
    target_phi: float = 0.43,
    seed: int = 0,
) -> Workload:
    """Generate a multi-agent trace.

    ``target_phi`` sets the first-turn anchor fraction by sizing the turn-0 task
    span; the paper measures ``phi in [0.34, 0.52]`` concentrated at session
    start, decaying monotonically as history accumulates. ``target_r`` sets the
    chain predictability. Both are reported back on the returned
    :class:`Workload` so a trace can be validated before it is used.
    """
    rng = random.Random(seed)
    agents = list(roster if roster is not None else DEFAULT_ROSTER)
    num_agents = len(agents)
    spans = _SpanAllocator()

    transition = make_transition_matrix(num_agents, target_r, rng)
    initial = _stationary(transition)

    # Anchors are allocated once and reused verbatim on every appearance. This
    # is the whole point: it is what makes cross-session block sharing possible.
    anchors = {a.name: spans.new_span(a.anchor_tokens) for a in agents}

    turns: list[Turn] = []
    for session_id in range(num_sessions):
        history: list[int] = []
        idx = _sample_categorical(rng, initial)
        for turn_index in range(turns_per_session):
            spec = agents[idx]
            if turn_index == 0:
                # Size turn 0's task so that phi = anchor / (anchor + task)
                # lands on target. Later turns inherit the mean task length and
                # let phi decay naturally as history grows.
                task_len = max(1, int(spec.anchor_tokens * (1.0 - target_phi) / target_phi))
                task_len = _sample_len(rng, task_len, spread=0.15)
            else:
                task_len = _sample_len(rng, spec.mean_task_tokens)

            anchor = anchors[spec.name]
            task = spans.new_span(task_len)
            tokens = anchor + history + task

            reply_tokens = _sample_len(rng, spec.mean_reply_tokens)
            turns.append(
                Turn(
                    session_id=session_id,
                    turn_index=turn_index,
                    agent=spec.name,
                    tokens=tokens,
                    anchor_len=len(anchor),
                    history_len=len(history),
                    task_len=task_len,
                    reply_tokens=reply_tokens,
                )
            )

            # The supervisor appends this turn's instruction and the agent's
            # reply to the shared transcript every agent sees next turn.
            history = history + task + spans.new_span(reply_tokens)
            idx = _sample_categorical(rng, transition[idx])

    workload = Workload(
        agents=agents,
        transition=transition,
        initial=initial,
        turns=turns,
    )
    _annotate(workload)
    return workload


def _sample_categorical(rng: random.Random, probs: Sequence[float]) -> int:
    r = rng.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if r < acc:
            return i
    return len(probs) - 1


def _annotate(workload: Workload) -> None:
    first = [t for t in workload.turns if t.turn_index == 0]
    workload.phi_first_turn = sum(t.anchor_len for t in first) / max(
        1, sum(t.prompt_len for t in first)
    )
    workload.phi_all_turns = sum(t.anchor_len for t in workload.turns) / max(
        1, sum(t.prompt_len for t in workload.turns)
    )
    workload.entropy_reduction = entropy_reduction(workload.transition)
    sessions = {t.session_id for t in workload.turns}
    workload.mean_turns_per_session = len(workload.turns) / max(1, len(sessions))
    workload.stats = {
        "num_sessions": len(sessions),
        "num_turns": len(workload.turns),
        "num_agents": len(workload.agents),
        "total_prompt_tokens": sum(t.prompt_len for t in workload.turns),
        "mean_prompt_tokens": sum(t.prompt_len for t in workload.turns) / max(1, len(workload.turns)),
        "max_prompt_tokens": max((t.prompt_len for t in workload.turns), default=0),
    }


def iter_by_session(workload: Workload) -> Iterator[tuple[int, list[Turn]]]:
    """Group turns by session, preserving turn order within each session."""
    by_session: dict[int, list[Turn]] = {}
    for turn in workload.turns:
        by_session.setdefault(turn.session_id, []).append(turn)
    for session_id in sorted(by_session):
        yield session_id, by_session[session_id]


def describe(workload: Workload) -> str:
    s = workload.stats
    lines = [
        f"sessions={s['num_sessions']} turns={s['num_turns']} agents={s['num_agents']}",
        f"phi(first turn)={workload.phi_first_turn:.3f}  "
        f"phi(all turns)={workload.phi_all_turns:.3f}",
        f"entropy reduction R={workload.entropy_reduction:.3f}",
        f"prompt tokens: mean={s['mean_prompt_tokens']:.0f} max={s['max_prompt_tokens']} "
        f"total={s['total_prompt_tokens']}",
    ]
    return "\n".join(lines)


PAPER_PHI_RANGE = (0.34, 0.52)
PAPER_R_RANGE = (0.40, 0.48)


def validate(workload: Workload) -> list[str]:
    """Return the list of paper-range checks this trace fails (empty is good)."""
    problems = []
    lo, hi = PAPER_PHI_RANGE
    if not lo <= workload.phi_first_turn <= hi:
        problems.append(
            f"phi(first turn)={workload.phi_first_turn:.3f} outside paper range [{lo}, {hi}]"
        )
    lo, hi = PAPER_R_RANGE
    if not lo <= workload.entropy_reduction <= hi:
        problems.append(
            f"R={workload.entropy_reduction:.3f} outside paper range [{lo}, {hi}]"
        )
    return problems
