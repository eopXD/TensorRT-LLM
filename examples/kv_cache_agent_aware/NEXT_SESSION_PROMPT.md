# Follow-up prompt: build a credible agentic KV-cache benchmark

Paste the block below into a fresh Claude Code session. It is written to be
self-contained — it does not assume the prior session's context.

---

## Prompt

> **Context.** In `~/Downloads/TensorRT-LLM-2`, branch
> `user/yuehtingc/kv-cache-agent-aware-poc` (also on remote `fork` =
> `git@github.com:eopXD/TensorRT-LLM.git`), there is a proof-of-concept
> evaluating agent-aware KV-cache eviction (CacheSage, arXiv:2605.27744) against
> KV cache manager v2. Read `examples/kv_cache_agent_aware/README.md` first — it
> has the full result and the reasoning.
>
> Short version of what is already established, so you do not redo it:
> - The offline harness (`agent_trace.py` + `replay_sim.py` + `policies.py`) is
>   CUDA-free and hashes blocks with the production
>   `sequence_to_blockchain_keys`. It works and is validated.
> - Agent-transition prediction does **not** beat LRU on the synthetic traces, at
>   any parameter setting or concurrency. Chain depth (`Block.ordinal`) carries
>   the win, and only under high interleaving.
> - Two causes: (a) `p_surv` is per-agent, so it protects an agent's dead session
>   history as much as its reusable anchor, and the anchor is <1% of blocks;
>   (b) at τ=0.01 with 6 agents the transition graph is complete, so every BFS
>   depth is 1 and the survival term is a constant.
>
> **The weakness to fix.** Every number rests on a *synthetic* trace generator.
> Structure is modelled on AutoGen `SelectorGroupChat`, but token content, anchor
> sizes and turn-length distributions are invented. Before anyone acts on the
> negative result, the benchmark needs to rest on real traces.
>
> **Your task: replace the synthetic generator with a real-trace pipeline, and
> extend the harness to measure what actually matters for serving.** Specifically:
>
> **1. Capture real multi-agent traces.** Pick one and justify it:
>    - Run a real framework (AutoGen `SelectorGroupChat`, LangGraph, or
>      CrewAI) over a task set (GAIA, MT-Bench, SWE-bench-lite, GSM8K) against
>      `trtllm-serve`, logging every request's full token ids, arrival time,
>      output length, and the framework's own agent label.
>    - Or convert the AA-AgentPerf trajectories
>      (https://artificialanalysis.ai/methodology/agentperf) — but note these are
>      *single*-agent coding sessions, so first determine empirically whether
>      tool-call phases within a trajectory can be relabelled as distinct agent
>      identities. If they cannot, say so and use a framework capture instead.
>      Do not silently treat AA-AgentPerf as a multi-agent workload; it is not
>      one, and that is exactly the trap this whole investigation started from.
>    - Store traces in a compact replayable format (token ids + timestamps +
>      labels). Keep them out of git; put them under
>      `/lustre/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/yuehtingc/`.
>
>    Report the measured `phi` (anchor fraction), `R` (entropy reduction),
>    prompt-length distribution, and turns-per-session for the real traces, and
>    compare against the synthetic ones. **If they differ materially, rerun the
>    policy sweep on the real traces and update the README's conclusion.**
>
> **2. Make the simulator time-aware.** It currently interleaves round-robin with
>    no clock, which cannot express the closed-loop dependency that defines
>    agentic load: each turn waits for the previous response plus simulated tool
>    time. Add an event-driven clock with per-request prefill/decode cost models
>    and configurable tool-think delay, so the harness can report **TTFT and
>    output speed**, not just hit rate. Hit rate is a proxy; the SLOs are not.
>
> **3. Add the policies the current sweep is missing.**
>    - The in-tree `BlockReusePolicy.PER_CONVERSATION` + `ConversationManager`
>      drop-plan behaviour (`tensorrt_llm/_torch/pyexecutor/kv_cache_manager_v2.py:137`
>      and `:159`) — this is TRT-LLM's existing session-scoped policy and the
>      closest in-tree analogue of the Continuum baseline the paper compares to.
>    - A depth+recency hybrid tuned per concurrency, since `depth` alone is
>      harmful at concurrency 1 (−27 pp) and strong at concurrency 4 (+10 pp).
>      Find out whether one setting works across both, or whether it has to adapt.
>
> **4. Close the loop on real hardware.** The offline harness predicts hit rate;
>    verify the prediction end-to-end.
>    - There is a built checkout at
>      `/lustre/fsw/.../users/yuehtingc/TensorRT-LLM-cachesage` on
>      `nsc-svg-slurm-1` (B200, x86_64, direct-SSH, single-GPU jobs allowed) with
>      a saved container `trtllm-devel-x86.sqsh`. Reuse
>      `cachesage-wheel.sbatch` / `cachesage-inbuild.sh` in that scratch dir.
>    - Run `trtllm-serve` + `tensorrt_llm/serve/scripts/benchmark_serving.py`
>      with `TLLM_KV_EVICTION_POLICY` set to `lru` vs `depth`, on the real traces,
>      and compare measured hit rate against the simulator's prediction for the
>      same block budget. **If they disagree by more than a few pp, the simulator
>      is wrong and must be fixed before any conclusion stands.**
>
> **Ground rules.**
> - Do not delete or weaken the trace-fidelity gate in `run_sweep.py`. If a real
>   trace falls outside the paper's `phi`/`R` ranges, that is a finding to
>   report, not a check to disable.
> - Keep `TLLM_KV_EVICTION_POLICY` defaulting to `lru`. No default behaviour
>   change without end-to-end SLO evidence.
> - Report negative results as prominently as positive ones. The value of this
>   work so far is a well-supported null result; do not quietly bury a second one.
> - Cite `file:line` for every symbol you reference.

---

## Reference: state at handoff

- Branch `user/yuehtingc/kv-cache-agent-aware-poc`, commit `9b6a26efea`,
  based on `main` @ `9564b3b4ff`.
- Wheel build on nsc-svg: job `1621714`, logs at
  `/lustre/fsw/.../users/yuehtingc/cachesage-wheel-1621714.log`,
  status file `cachesage-wheel-1621714.status`.
- Unit tests: `tests/unittest/kv_cache_manager_v2_tests/test_agent_aware.py`
  (24 tests, no GPU needed).

## Open questions worth an explicit answer

1. Does the null result survive real traces, or is it an artifact of synthetic
   anchor-size / turn-length distributions?
2. Is there a prompt scale at which the agent term *does* pay? The failure mode
   predicts it should help when anchor blocks are a large fraction of an agent's
   footprint — i.e. short prompts and small caches, the paper's regime. Sweep
   mean prompt length and find the crossover, if one exists. That would reconcile
   this result with the paper's rather than merely contradicting it.
3. Does depth-aware eviction interact badly with SWA / VSWA layers, where blocks
   outside the window are already dropped by a different mechanism?
4. Under disaggregated serving, does depth-aware eviction change what the CTX
   worker hands to the GEN worker, and does that shift the KV-transfer volume?
