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
"""Sweep eviction policies against cache budget on a synthetic agentic trace.

python3 run_sweep.py                      # default sweep
python3 run_sweep.py --concurrency 1
python3 run_sweep.py --degenerate         # single-agent LRU-equivalence check
"""

from __future__ import annotations

import argparse
import json
import sys

import agent_trace as at
import replay_sim as rs
from policies import registry


def build_workload(args: argparse.Namespace) -> at.Workload:
    workload = at.generate(
        num_sessions=args.sessions,
        turns_per_session=args.turns,
        target_r=args.target_r,
        target_phi=args.target_phi,
        seed=args.seed,
    )
    problems = at.validate(workload)
    print(at.describe(workload))
    if problems:
        print("TRACE OUT OF PAPER RANGE:")
        for p in problems:
            print(f"  - {p}")
        if not args.allow_out_of_range:
            print("refusing to sweep an unrepresentative trace; pass --allow-out-of-range")
            sys.exit(2)
    return workload


def sweep(args: argparse.Namespace) -> dict:
    workload = build_workload(args)
    tpb = args.tokens_per_block

    total = rs.working_set_blocks(workload, tpb)
    anchor = rs.anchor_working_set_blocks(workload, tpb)
    ceiling = rs.reuse_ceiling(workload, tpb, concurrency=args.concurrency)
    print(
        f"\nworking set = {total} blocks; anchor working set = {anchor} blocks "
        f"({anchor / max(1, total):.2%} of total)"
    )
    print(f"reuse ceiling (unbounded cache) = {ceiling.hit_rate:.2%}\n")

    if args.budgets:
        budgets = args.budgets
    else:
        # Concentrate on the pressured regime: below the point where LRU alone
        # already reaches the ceiling there is nothing for a policy to fix.
        budgets = [int(total * f) for f in (0.01, 0.02, 0.04, 0.06, 0.10, 0.15, 0.25)]
    budgets = sorted({b for b in budgets if b > 0})

    policies = registry()
    names = args.policies or list(policies)
    results: dict[str, list[rs.SimResult]] = {n: [] for n in names}

    header = f"{'budget':>7} {'% of WS':>8} " + " ".join(f"{n:>14}" for n in names)
    print(header)
    print("-" * len(header))
    for budget in budgets:
        row = []
        for name in names:
            res = rs.simulate(
                workload,
                budget_blocks=budget,
                policy_factory=policies[name],
                tokens_per_block=tpb,
                concurrency=args.concurrency,
            )
            results[name].append(res)
            row.append(f"{res.hit_rate:>13.2%}")
        print(f"{budget:>7} {budget / max(1, total):>7.1%} " + " ".join(row))

    baseline = names[0]
    if len(names) > 1:
        print(f"\ndelta vs {baseline} (percentage points):")
        print(header)
        print("-" * len(header))
        for i, budget in enumerate(budgets):
            base = results[baseline][i].hit_rate
            row = [
                f"{(results[n][i].hit_rate - base) * 100:>+13.1f}" for n in names
            ]
            print(f"{budget:>7} {budget / max(1, total):>7.1%} " + " ".join(row))

    print("\nanchor-block hit rate (the span the policies are fighting over):")
    print(header)
    print("-" * len(header))
    for i, budget in enumerate(budgets):
        row = [f"{results[n][i].anchor_hit_rate:>13.2%}" for n in names]
        print(f"{budget:>7} {budget / max(1, total):>7.1%} " + " ".join(row))

    return {
        "trace": {
            "phi_first_turn": workload.phi_first_turn,
            "entropy_reduction": workload.entropy_reduction,
            "working_set_blocks": total,
            "anchor_working_set_blocks": anchor,
            "reuse_ceiling": ceiling.hit_rate,
            **workload.stats,
        },
        "concurrency": args.concurrency,
        "tokens_per_block": tpb,
        "budgets": budgets,
        "results": {
            n: [
                {
                    "budget_blocks": r.budget_blocks,
                    "hit_rate": r.hit_rate,
                    "anchor_hit_rate": r.anchor_hit_rate,
                    "evictions": r.evictions,
                    "anchor_evictions": r.anchor_evictions,
                    "oversized_requests": r.oversized_requests,
                }
                for r in results[n]
            ]
            for n in names
        },
    }


def degenerate_check(args: argparse.Namespace) -> None:
    """A single-agent workload must produce byte-identical LRU and agent-aware runs.

    This is the property that makes the policy safe to enable by default: with
    no agent structure the survival term is constant and the score collapses to
    the recency residual.
    """
    roster = [at.DEFAULT_ROSTER[0]]
    workload = at.generate(
        num_sessions=args.sessions,
        turns_per_session=args.turns,
        roster=roster,
        target_r=0.0,
        seed=args.seed,
    )
    policies = registry()
    ok = True
    for budget in (200, 600, 1500, 4000):
        lru = rs.simulate(workload, budget, policies["lru"], args.tokens_per_block, args.concurrency)
        aa = rs.simulate(
            workload, budget, policies["agent-aware"], args.tokens_per_block, args.concurrency
        )
        same = (lru.matched_tokens == aa.matched_tokens) and (lru.evictions == aa.evictions)
        ok &= same
        print(
            f"budget={budget:>5} lru_hit={lru.hit_rate:.4%} agent_aware_hit={aa.hit_rate:.4%} "
            f"evict {lru.evictions} vs {aa.evictions}  {'MATCH' if same else 'DIVERGED'}"
        )
    print("\nsingle-agent degeneration:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sessions", type=int, default=50)
    p.add_argument("--turns", type=int, default=20)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--target-r", type=float, default=0.44)
    p.add_argument("--target-phi", type=float, default=0.43)
    p.add_argument("--tokens-per-block", type=int, default=32)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--budgets", type=int, nargs="*", default=None)
    p.add_argument("--policies", type=str, nargs="*", default=None)
    p.add_argument("--allow-out-of-range", action="store_true")
    p.add_argument("--degenerate", action="store_true", help="run the LRU-equivalence check")
    p.add_argument("--json", type=str, default=None, help="write results to this path")
    args = p.parse_args()

    if args.degenerate:
        degenerate_check(args)
        return

    payload = sweep(args)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
