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
"""Live end-to-end check of the offline simulator's prediction.

Replays the same synthetic multi-agent trace the simulator uses through a real
TensorRT-LLM engine with KVCacheManagerV2, at a constrained cache size, under
each TLLM_KV_EVICTION_POLICY setting, and reports the engine's own
reused/missed block counters.

The point is falsification: the simulator predicts a hit-rate delta between
policies, and this reports the delta the engine actually produces. If they
disagree materially the simulator is wrong and every offline conclusion has to
be re-derived.

Prompts are raw token ids from the generator, so the text is meaningless -- but
KV-cache behaviour depends only on token identity and block boundaries, which
are exactly reproduced. Output length is pinned to 1 token so the run measures
prefill reuse rather than decode.

    TLLM_KV_EVICTION_POLICY=lru python3 run_live.py --max-tokens 65536 \
        --model /home/scratch.trt_llm_data_ci/llm-models/llama-3.1-model/Llama-3.1-8B-Instruct
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent_trace as at  # noqa: E402
import replay_sim as rs  # noqa: E402


def build_trace(args):
    workload = at.generate(
        num_sessions=args.sessions,
        turns_per_session=args.turns,
        target_r=args.target_r,
        target_phi=args.target_phi,
        seed=args.seed,
    )
    print(at.describe(workload), flush=True)
    problems = at.validate(workload)
    if problems:
        print("WARNING: trace outside paper range: " + "; ".join(problems), flush=True)
    return workload


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--sessions", type=int, default=12)
    p.add_argument("--turns", type=int, default=10)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--target-r", type=float, default=0.44)
    p.add_argument("--target-phi", type=float, default=0.43)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--tokens-per-block", type=int, default=32)
    p.add_argument("--max-tokens", type=int, required=True,
                   help="KV cache size in tokens; the budget knob")
    # Requests are issued one at a time, exactly as the simulator models them:
    # "concurrency" is the session interleaving order, not simultaneous
    # execution. Batching several requests together would add capacity-scheduler
    # effects the simulator does not model, making any gap uninterpretable.
    p.add_argument("--max-num-seqs", type=int, default=1)
    p.add_argument("--json", type=str, default=None)
    args = p.parse_args()

    policy = os.environ.get("TLLM_KV_EVICTION_POLICY", "lru")
    print(f"=== TLLM_KV_EVICTION_POLICY={policy} max_tokens={args.max_tokens} ===", flush=True)

    workload = build_trace(args)
    turns = list(rs._interleave(workload, args.concurrency))
    budget_blocks = args.max_tokens // args.tokens_per_block

    # Offline prediction for exactly this trace, order, and budget.
    predicted = rs.simulate(
        workload,
        budget_blocks=budget_blocks,
        policy_factory=_sim_policy(policy),
        tokens_per_block=args.tokens_per_block,
        concurrency=args.concurrency,
    )
    print(
        f"SIM_PREDICTION policy={policy} budget_blocks={budget_blocks} "
        f"token_hit={predicted.hit_rate:.4f} block_hit={_block_hit(predicted, args):.4f}",
        flush=True,
    )

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import KvCacheConfig

    kv = KvCacheConfig(
        enable_block_reuse=True,
        use_kv_cache_manager_v2=True,
        max_tokens=args.max_tokens,
        tokens_per_block=args.tokens_per_block,
        # The simulator counts only whole blocks, dropping each request's
        # trailing partial block. Leaving partial reuse on would let the engine
        # score hits the simulator never models, so the two numbers would not be
        # comparable and any gap would be uninterpretable.
        enable_partial_reuse=False,
    )
    llm = LLM(
        model=args.model,
        kv_cache_config=kv,
        max_batch_size=args.max_num_seqs,
        max_num_tokens=16384,
        enable_chunked_prefill=True,
        # Without this the iteration-stats queue stays empty and get_stats()
        # returns nothing, which reads as "hit rate 0" rather than as an error.
        enable_iter_perf_stats=True,
    )

    sampling = SamplingParams(max_tokens=1, temperature=0.0)

    t0 = time.time()
    sent = 0
    for turn in turns:
        llm.generate([{"prompt_token_ids": list(turn.tokens)}], sampling, use_tqdm=False)
        sent += 1
        if sent % 25 == 0:
            print(f"  ... {sent}/{len(turns)} requests", flush=True)
    elapsed = time.time() - t0

    # The exact nesting and spelling of the KV-cache counters has moved around
    # between backends and versions, so scan for them instead of hard-coding a
    # path, and dump one raw record so the log carries the ground truth.
    reused = missed = 0
    records = []
    try:
        records = list(llm.get_stats(timeout=30))
    except Exception as exc:  # noqa: BLE001 - stats surface varies by backend
        print(f"WARNING: could not read stats: {exc!r}", flush=True)

    # Whether these counters are cumulative or per-iteration decides how to
    # aggregate, and getting it wrong silently produces a plausible but wrong
    # hit rate. Compute both readings and print them; if the counters are
    # cumulative the max is correct and the sum is inflated, and the raw sample
    # below shows which it is. They agree only when there is a single record.
    reused_sum = missed_sum = 0
    for record in records:
        entry = json.loads(record) if isinstance(record, str) else record
        found = _scan_counters(entry)
        reused = max(reused, found.get("reused", 0))
        missed = max(missed, found.get("missed", 0))
        reused_sum += found.get("reused", 0)
        missed_sum += found.get("missed", 0)
    total_sum = reused_sum + missed_sum
    print(
        f"AGG_CHECK records={len(records)} "
        f"max(reused)={reused} max(missed)={missed} "
        f"sum(reused)={reused_sum} sum(missed)={missed_sum} "
        f"sum_block_hit={(reused_sum / total_sum) if total_sum else float('nan'):.4f}",
        flush=True,
    )

    if records:
        first = records[-1]
        first = json.loads(first) if isinstance(first, str) else first
        print("RAW_STATS_SAMPLE=" + json.dumps(first)[:2000], flush=True)
    else:
        print("WARNING: no stats records returned", flush=True)

    total = reused + missed
    measured = reused / total if total else float("nan")
    print(
        f"MEASURED policy={policy} reused={reused} missed={missed} "
        f"block_hit={measured:.4f} requests={sent} elapsed={elapsed:.1f}s",
        flush=True,
    )

    payload = {
        "policy": policy,
        "max_tokens": args.max_tokens,
        "budget_blocks": budget_blocks,
        "concurrency": args.concurrency,
        "requests": sent,
        "elapsed_s": elapsed,
        "predicted_token_hit": predicted.hit_rate,
        "predicted_block_hit": _block_hit(predicted, args),
        "measured_reused_blocks": reused,
        "measured_missed_blocks": missed,
        "measured_block_hit": measured,
    }
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote {args.json}", flush=True)

    llm.shutdown()


def _scan_counters(node, out=None) -> dict:
    """Recursively find cumulative reused/missed block counters in a stats record.

    Matches ``reusedBlocks``/``reused_blocks`` and ``missedBlocks``/
    ``missed_blocks`` at any depth, and deliberately ignores the ``iter``-
    prefixed per-iteration variants -- those reset every step, so maxing over
    them would understate the total.
    """
    if out is None:
        out = {}
    if isinstance(node, dict):
        for key, value in node.items():
            low = key.lower()
            if isinstance(value, (int, float)) and not low.startswith("iter"):
                if "reused" in low and "block" in low:
                    out["reused"] = max(out.get("reused", 0), int(value))
                elif "missed" in low and "block" in low:
                    out["missed"] = max(out.get("missed", 0), int(value))
            else:
                _scan_counters(value, out)
    elif isinstance(node, list):
        for item in node:
            _scan_counters(item, out)
    return out


def _sim_policy(name: str):
    from policies import registry

    reg = registry()
    if name not in reg:
        raise SystemExit(f"unknown policy {name!r}; have {sorted(reg)}")
    return reg[name]


def _block_hit(result: rs.SimResult, args) -> float:
    """Convert the simulator's token hit rate to the engine's block hit rate.

    The engine reports reused / (reused + missed) over whole blocks; the
    simulator counts matched tokens over prompt tokens. Both numerator and
    denominator are block-quantised in the simulator, so dividing out the block
    size makes them directly comparable.
    """
    matched_blocks = result.matched_tokens / args.tokens_per_block
    total_blocks = result.prompt_tokens / args.tokens_per_block
    return matched_blocks / total_blocks if total_blocks else float("nan")


if __name__ == "__main__":
    main()
