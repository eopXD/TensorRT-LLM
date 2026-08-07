# Agent-aware KV-cache eviction: PoC and measurement

Evaluates the CacheSage design ([arXiv:2605.27744](https://arxiv.org/abs/2605.27744),
"A Policy-Driven Runtime Layer for Agentic LLM Serving") against TensorRT-LLM's
KV cache manager v2.

**Headline result: the agent-transition mechanism does not beat LRU on
realistically-scaled multi-agent traces. Chain depth — a signal already present
in `Block.ordinal` and free to read — carries the entire win, and only under
high request interleaving.**

Everything below is reproducible on a laptop; no GPU is required.

---

## 1. What the paper proposes

Blocks are tagged with an agent identity. An online first-order Markov model
learns `W(b|a)` over agent transitions. Eviction ranks blocks by

```
score(b) = w_pred * p_surv(a_b) + rho_b
```

where `p_surv` comes from BFS hop distance to the block's agent on a
probability-thresholded transition graph (τ = 0.01, `max_hops` = 8) and `rho_b`
is the normalized recency residual. The paper reports +13 to +37 pp hit rate on
five workloads. It reports **no ablation**, so it never establishes which part
of the design is load-bearing.

## 2. Running it

```bash
cd examples/kv_cache_agent_aware

python3 run_sweep.py                                   # concurrency 4 sweep
python3 run_sweep.py --concurrency 1
python3 run_sweep.py --degenerate                      # LRU-equivalence check
python3 run_sweep.py --policies lru depth --json out.json

python3 ../../tests/unittest/kv_cache_manager_v2_tests/test_agent_aware.py
```

| File | Role |
| --- | --- |
| `agent_trace.py` | Multi-agent trace generator. Solves for a target entropy reduction `R` and anchor fraction `phi`. |
| `replay_sim.py` | CUDA-free replay of prefix matching + eviction under a block budget. |
| `policies.py` | `lru`, `depth`, `anchor-oracle`, `agent-aware`, `composite`. |
| `run_sweep.py` | Driver. |
| `kvcm_shim.py` | Loads the CUDA-free parts of `kv_cache_manager_v2` on a GPU-less machine. |

The simulator hashes blocks with the **production** `sequence_to_blockchain_keys`
and the production SHA-256 `Hasher`, and the agent scoring is the **production**
`kv_cache_manager_v2/_agent_aware.py`. Only the queue plumbing is re-implemented.

### Trace fidelity gate

`run_sweep.py` refuses to sweep a trace whose `phi` and `R` fall outside the
paper's reported ranges (`phi ∈ [0.34, 0.52]`, `R ∈ [0.40, 0.48]`), so a null
result cannot be blamed on an unrepresentative workload. The default
configuration lands at `phi(first turn) = 0.42`, `R = 0.44` on every seed tried.

### Prompt structure being modelled

```
anchor(a)      agent-specific: system prompt + tool defs + few-shots
history(<t)    session-specific: every prior (speaker, message) pair
task(a, t)     this turn's instruction
```

Two consequences drive every result, and both are real properties of
supervisor-style frameworks rather than artifacts of the generator:

1. `anchor(a)` is the **only** span reusable across sessions. Under
   content-addressed prefix caching all sessions of one agent already share one
   copy — so this is purely an eviction-ordering problem, not a deduplication
   one.
2. Within a session, agents **cannot** share history blocks with each other.
   The history text is identical but sits behind a different `anchor(a)` prefix,
   and prefix matching is anchored at token 0. A session with `|A|` active agents
   grows `|A|` divergent branches off the root.

## 3. Results

Trace: 30 sessions × 16 turns, 6 agents, `phi₀` = 0.422, `R` = 0.439,
mean prompt 7,691 tokens, `tokens_per_block` = 32.
Working set 45,335 blocks; **anchor working set 310 blocks (0.68%)**;
reuse ceiling (unbounded cache) 60.51%.

### Concurrency 4 — hit rate

| budget | % of WS | lru | depth | anchor-oracle | agent-aware |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 453 | 1% | 4.80% | **14.24%** | 12.92% | 4.80% |
| 906 | 2% | 10.54% | 20.65% | **21.45%** | 10.46% |
| 1,813 | 4% | 21.36% | 26.23% | **26.65%** | 21.06% |
| 2,720 | 6% | 32.17% | 29.95% | **34.18%** | 31.71% |
| 4,533 | 10% | 50.25% | 35.90% | **50.39%** | 49.89% |
| 6,800 | 15% | 60.51% | 41.16% | 60.51% | 60.51% |

`agent-aware` is at or below LRU everywhere (−0.5 to +0.0 pp).

### Concurrency 1 — hit rate

| budget | % of WS | lru | depth | agent-aware |
| ---: | ---: | ---: | ---: | ---: |
| 453 | 1% | 4.92% | **15.56%** | 4.71% |
| 906 | 2% | 20.15% | **24.91%** | 19.79% |
| 1,813 | 4% | **57.23%** | 29.95% | 56.79% |
| 2,720 | 6% | **60.18%** | 33.20% | 60.18% |
| 11,333 | 25% | **60.51%** | 50.18% | 60.51% |

`depth` is **not** a universal win: at concurrency 1 it costs up to −27 pp,
because with one session running the deep blocks are exactly what that session's
next turn extends. Its advantage comes from interleaving, not from depth per se.

### Parameter sensitivity — agent-aware, budget 906, concurrency 4

All 24 combinations of τ ∈ {0.01, 0.10, 0.20, 0.35} × `max_hops` ∈ {2, 4, 8} ×
`w_pred` ∈ {1, 3} land between **−1.5 and −0.1 pp vs LRU**. None beats it. The
best is τ=0.01/hops=8/w=1 at −0.1 pp; `depth` at the same budget is +10.1 pp.

### The decisive ablation

`composite` scores `w_depth * shallowness + w_pred * p_surv + rho`, so
`w_pred = 0` is a pure depth policy and `w_depth = 0` is pure agent-aware.
`w_depth = 0, w_pred = 0` reproduces LRU **exactly**, which validates the
generalization.

Marginal contribution of the agent term *on top of* depth (`w_depth = 3`):

| budget | `w_pred`=1 | `w_pred`=3 |
| ---: | ---: | ---: |
| 906 | −0.10 pp | −0.34 pp |
| 1,813 | −0.15 pp | −0.33 pp |
| 2,720 | +0.28 pp | −0.11 pp |

Noise, trending negative. **The agent model adds nothing that depth has not
already captured.**

## 4. Why it fails

Two independent causes, both structural:

**(a) The predictive term has the wrong granularity.** `p_surv` is per-*agent*
and therefore uniform across every block that agent touches. But the agent's
reusable anchor is 310 of 45,335 blocks (0.68%); the rest is session history
that becomes unreachable the moment the session ends. Raising an agent's
survival protects its dead tail exactly as much as its anchor — mostly
protecting garbage, which is why the policy comes in slightly *below* LRU.

This does not bite at the paper's scale. With a 120–250 block cache and short
prompts, "the agent's blocks" ≈ "the agent's anchor blocks" and the conflation
is invisible. It bites hard at the prompt lengths that agentic serving actually
produces (here 7.7k mean, 13.8k max; AA-AgentPerf's contexts grow further).

**(b) At the paper's τ, the transition graph is complete.** With 6 agents and
`R` = 0.44, the smallest row entry is ≈0.012 — above τ = 0.01. So every agent is
1 hop from every other, every BFS depth is 1, and `p_surv` is the constant
0.875. The score collapses to `const + rho`, i.e. *exactly* LRU. The paper's
justification for τ = 0.01 ("preserves ≥99% of observed transitions") is
precisely what makes the hop-count proxy carry no information. Raising τ to
sparsify the graph does not rescue it — see the sensitivity table — because
cause (a) still applies.

## 5. What was landed

| Change | Status |
| --- | --- |
| `_agent_aware.py` — identity, transition model, survival scorer | Complete, 24 unit tests, no CUDA imports |
| `Page.eviction_ordinal` | Reads existing `Block.ordinal`; no new state |
| `DepthAwareEvictionPolicy` | Behind `TLLM_KV_EVICTION_POLICY=depth`, **default `lru`** |
| Agent-id plumbing into `Page` | **Not done** — the measurement says it would not pay |

`_agent_aware.py` is complete and tested so the agent path is one plumbing step
away if a future workload justifies it. It is currently imported by nothing on
the runtime path.

The single-agent degeneration check (`run_sweep.py --degenerate`) confirms
byte-identical hit rate and eviction counts vs LRU at every budget, which is the
safety property that makes such a policy enable-able by default.

### Verification performed

| Check | Where | Result |
| --- | --- | --- |
| `_agent_aware` unit tests | laptop + container | 24/24 pass |
| Single-agent LRU degeneration | laptop | identical hit rate and eviction counts at every budget |
| x86_64 wheel build (`89-real;100-real`) | nsc-svg job 1621789 | `BUILD_RC=0`, `tensorrt_llm-1.3.0rc24-cp312-cp312-linux_x86_64.whl` |
| Policy selection under real runtime | nsc-svg job 1622249 | `lru`→`PrioritizedLRUEvictionPolicy`, `depth`→`PrioritizedDepthAwareEvictionPolicy` |
| Full `test_kv_cache_manager_v2.py` (B200) | nsc-svg job 1622249 | **122 tests OK (13 skipped) under BOTH policy settings** |

The suite run under `TLLM_KV_EVICTION_POLICY=depth` is what exercises
`Page.eviction_ordinal` and `DepthAwareEvictionPolicy` against real page
lifecycles; it is the evidence that the new policy is not merely importable.

Not verified: end-to-end serving hit rate / TTFT on real traces. That is the
first item in `NEXT_SESSION_PROMPT.md`.

## 6. Threats to validity

- **Synthetic traces.** Structure is modelled on AutoGen `SelectorGroupChat`;
  token *content* is synthetic. Content does not affect block hashing behaviour,
  but real anchor-size and turn-length distributions could shift the numbers.
  Replacing the generator with captured traces is the top follow-up.
- **Agent identity is an oracle here.** The simulator uses the true agent label,
  not the derived `skip/take` hash. That makes the negative result *stronger* —
  a derived identity can only be noisier.
- **Prefetch (§3.4) was not evaluated.** Given anchors are already shared
  cross-session by content addressing, prefetch can only help a cold start.
- **One roster shape.** 6 agents, one anchor-size distribution. `|A|` = 12+ was
  not swept.
