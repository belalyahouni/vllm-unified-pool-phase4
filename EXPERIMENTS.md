# Experiments

Evaluation plan for the dynamic unified expert/KV pool. The system shares one
per-layer GPU buffer between MoE expert weights and KV-cache pages and moves memory
between them at runtime — evicting whichever is currently *less useful* (experts for
KV, or KV for experts, or experts for experts, or KV for KV) depending on the LRU. The
claim: a single unified-pool configuration matches a per-workload optimally-tuned
static offloading split on each workload, and adapts across workload shifts, which no
fixed split can do.

- **System under test:** modified vLLM v0.17.1, dynamic unified pool
  (`--expert-unified-pool`, `--expert-pool-page-tokens 16`).
- **Baseline:** standard expert offloading with a fixed split
  (`--expert-offload`, static `--expert-cache-size`), plus no-offload vanilla.
- **Model:** `allenai/OLMoE-1B-7B-0924-Instruct` (16 MoE layers, 64 experts/layer,
  top-k 8; one expert = 12 MiB BF16 = 1536 KV tokens; `max_position_embeddings 4096`).
- **Hardware:** single NVIDIA L4 (23 GB).

---

## Notation

Units: **1 super-block = 1 expert = 1536 KV tokens = F=96 pages; 1 page = 16 tokens.**
So the pool = `M` super-blocks = `M×F` = 6336 pages (`--num-gpu-blocks-override`).

| Symbol | Meaning |
|---|---|
| **M** | pool size in **super-blocks** (M = 66 = 64 experts + 2 sb) — the per-layer GPU budget shared between experts and KV |
| **F** | **pages per super-block** = 96 (= expert-slot ÷ page-size); one super-block holds one expert *or* 96 KV pages |
| **super-block (sb)** | contiguous region = one expert = 1536 KV tokens = F pages |
| **page (pg)** | smallest pool/KV unit = 16 tokens (`--expert-pool-page-tokens 16`, `--block-size 16`) |
| **C** | static **expert-cache size** — experts kept resident per layer in a fixed split (`--expert-cache-size C`) |
| **C_kv / C_ex / C_comp** | static oracle cache sizes = **32 / 64 / 48** (reasoned from measured footprints, not swept) |
| **C_init** | the unified pool's *initial* (warm-up) expert count |
| **N** | number of distinct prompts in the KV-heavy (`kv_distinct`) workload |
| **top-k** | experts the router activates per token = 8 (the resident-expert floor) |
| **distinct-char run** | KV-heavy prompt = one character repeated (~3073 tokens); a repeated char routes to a fixed ~8 experts at any length; 16 distinct chars union to ~29 experts (≤ 32) |
| **TTFT** | time-to-first-token = prefill latency (cold ≈ 60 s, warm cache-hit ≈ 0.1 s) — the load-bearing metric |
| **TPOT** | time-per-output-token = decode latency |
| **KV** | attention key/value cache |

---

## Common setup

**Pool budget.** **M = 67 super-blocks** (`--num-gpu-blocks-override 6432`, F = 96 pages
of 16 tokens; ≈ 12.6 GB pool, boots on L4 at `--gpu-memory-utilization 0.90`). Sized as
**64 experts + 2 sb prefix-KV + 1 sb "active" headroom**: the expert-oracle `C=64` then
gets 3 sb KV, which is the minimum that can actually *serve* a ~3072-token request — at
only 2 sb (M=66) the request needs all 192 blocks but vLLM's block watermark reserves a
sliver, so it never schedules and hangs. The 3rd sb supplies that headroom. Contention
comes from the workloads and the static split, not from pool tightness.

**Mandatory envelope (every run, all configs):**

```
--enforce-eager --enable-prefix-caching --trust-remote-code
--max-model-len 4096 --max-num-batched-tokens 1 --no-async-scheduling
--attention-backend TRITON_ATTN --block-size 16 --gpu-memory-utilization 0.90
```

TP = PP = 1, concurrency = 1 (sequential, one-token-at-a-time — required for relocation
safety, applied uniformly). This is a **latency + cache-efficiency** study, not
throughput-under-load (stated as a scoped PoC limitation).

**Config definitions.** Static KV blocks = `(M − C) × 96` (budget-matched):

| Config | Flags (beyond envelope) |
|---|---|
| Vanilla | *(no offload; all 64 experts resident, ~16 GB)* |
| Static, cache C | `--expert-offload --expert-cache-size C --num-gpu-blocks-override $(((M-C)*96))` |
| Ours (unified) | `--expert-offload --expert-unified-pool --expert-pool-page-tokens 16 --expert-cache-size C_init --num-gpu-blocks-override 6432` |

Ours uses lean `C_init = 8` (grows into whatever the workload needs; lean init is the
robust choice — a high init leaves residual stale experts that over-subscribe under KV
pressure). For M = 67: unified override `6432`; static override `(67−C)×96`
(C = 32 → 3360; C = 48 → 1824; C = 64 → 288).

**Static oracles** (reasoned from the measured footprints — *no calibration sweep
needed*): **C_kv = 32** (≥ the KV-heavy workload's measured expert union of ~29 — enough
experts to avoid cold-pass thrash, everything else to KV), **C_ex = 64** (all experts;
the random workload activates ~all 64), **C_comp = 48** (midpoint).

**Metrics.**
- `vllm bench serve --save-result` (driven via the OpenAI API): median & p99 **TTFT**
  (ms — cold prefill ≈ 60 s vs warm cache-hit ≈ 0.1 s is the load-bearing signal),
  median **TPOT**, **total_token_throughput**, `completed`/`failed`.
- `VLLM_UNIFIED_POOL_TRACE=1` + `scripts/summarize_trace.py`: per-step pool composition
  (expert sb / prefix pg / alloc-kv pg / free), eviction counts by direction,
  **relocation count**.

**Seeds.** Deterministic (greedy). Prefix workloads seed 1; random workload seeds
{1,2,3}. Idle VRAM per config via `nvidia-smi`.

---

## Workloads

- **KV-heavy** — `kv_distinct`: **N = 16** distinct prompts, each a **3071-token** run of a
  *distinct single character* (`a A C e o c r n u h 6 2 7 f v q`), **`output_len = 1`**
  (3071 + 1 = 3072 tokens = exactly 2 sb per request → 16 requests fill ~32 sb of KV). A
  repeated character routes to a fixed ~8 experts at any length (no positional drift);
  different characters give distinct token ids → distinct cacheable KV (no prefix sharing).
  Measured cumulative input expert **union = 29 (≤ 32)** across all 16, so the pool settles
  at ~33 experts + **~32 sb KV** (fills the KV budget) with no over-subscription. **`output_len`
  must stay tiny**: generated tokens are diverse and each loads extra experts — at
  `output_len = 20` the footprint inflates from ~29 to ~63 and over-subscribes the pool
  (measured). Forward cold pass then reverse measured pass. Validated on L4: ours caches
  **all 16** from **every** init tried — `C_init = 8` (settles ~33 experts) and `C_init =
  32/48/64` (all shed to ~35) — the pool converges to the KV-oracle state regardless of init
  (the `min`-eviction fix sheds the seeded-but-cold experts under KV pressure). Fair
  (trace-off) warm-hit median ≈ **256–298 ms**, i.e. *below* the static KV-oracle C=32
  (319 ms) — no latency penalty for being dynamic. (Earlier 370–430 ms readings were a
  `VLLM_UNIFIED_POOL_TRACE` logging artifact, since the ours cells traced and the statics
  didn't; re-run trace-off for the fair numbers.) Files:
  `prompts/kv_distinct_{fwd,rev}.jsonl` (`scripts/make_kv_distinct.py`).

- **Expert-heavy** — `--dataset-name random --random-input-len 256 --random-output-len 80
  --random-prefix-len 0 --random-range-ratio 0 --num-prompts 12`. High expert diversity
  (→ all 64), tiny KV per request.
- **Phase-shift** — two single-transition runs on one persistent server (warmups set so
  pool state carries across the boundary): **KV → Expert** and **Expert → KV**. Together
  they exercise both swap directions without a round-trip.

---

## Experiment 1 — Oracle matching

**Claim.** One unified config equals the KV-oracle on the KV-heavy workload *and* the
Expert-oracle on the expert-heavy workload; each oracle is bad in the other regime.

**Configs.** Static `C_kv`=32, `C_ex`=64, `C_comp`=48, vanilla, ours (init 48) — five
configs × two workloads, server restarted per cell. KV-heavy cold then warm (report
warm, reverse order); expert-heavy seeds {1,2,3}.

**Expected.** KV-heavy: static `C_ex`=64 leaves ~2 sb KV → holds ~1 of N prefixes →
warm-pass re-prefills at ~62 s; `C_kv`=32 and ours ≈ **0.1 s**; vanilla ≈ optimal.
Expert-heavy: static `C_kv`=32 thrashes experts → high TPOT; `C_ex`=64, vanilla, ours ≈
best. Ours within a small margin of the best static on **both**, one config.

**Cold-start convergence (sub-panel).** Record the unified cell's *per-request* TTFT on
KV-heavy, not just the warm endpoint — the decay from `C_init` to the warm asymptote is
the pool converging (shedding experts, growing KV). Run one unified cell from the
expert-biased `C_init = 64` for a sharp transient; the warm endpoint is unchanged (the
pool converges to the same state regardless of init), so this only strengthens the
oracle match. Report the convergence count (requests to reach the KV-oracle asymptote).

**Figure.** Grouped bars (warm TTFT on KV-heavy, TPOT on expert-heavy) + table with
TTFT/TPOT/tok-s and idle VRAM; plus a per-request convergence curve (inset).

---

## Experiment 2 — Dynamic phase-shift adaptation

**Claim.** On a *persistent* server fed one workload then the other, the pool moves memory
both ways at runtime and stays near the best config in *each* phase — while every fixed
split spikes in its off-phase and no fixed split is good in both. A static split cannot
reallocate; that's the whole point.

**Procedure.** Same E1 workloads back-to-back on **one persistent server** (no restart;
`--num-warmups 0` throughout so pool state carries across the boundary). Two orders:
**KV→Expert** and **Expert→KV** (together they exercise both swap directions — evict-KV-
grow-experts, and shed-experts-grow-KV). The **KV phase = 16 cold + 16 warm-reverse**
(`kv_distinct`, out=1) — repeats are required so C=64's misses actually show (a single
cold pass is ~62 s for *everyone*). The **expert phase = 220 random requests** (256/80,
out=80). Configs: **ours init 48** (run twice — trace **off** for clean latency, trace
**on** for the composition figure), **static C=32 / C=48 / C=64**. 1 seed; all metrics
saved (`--save-detailed`).

**Why 220 expert requests (the balance).** The two failure modes are very different in
size, so the expert phase is tuned to make the two *extreme* statics equally bad (else the
result is biased toward the KV workload). From E1 (M=67): C=64's KV-warm cost ≈ 718 s
(15 misses × ~50 s); C=32's expert cost ≈ 8.4 s/req (thrash) vs C=64's 5.2 s/req. Balancing
`total(C32) = total(C64)`: `5 + 8.4·N = 718 + 5.2·N → N ≈ 223`. So **N = 220**.

**Result (measured on L4; badness = KV-warm + expert, mean of both orders; predictions in
parentheses):**

| config | KV-warm (hits) | expert TPOT | **badness** | role |
|---|---|---|---|---|
| static C=32 | 4 s (**16/16**) | 24.7 ms (thrash) | **1854 s** (1878) | extreme — bad in expert phase |
| static C=64 | 713 s (**1/16**) | 15.5 ms | **1858 s** (1878) | extreme — bad in KV phase; **within 0.2% of C=32** ✓ |
| static C=48 | 319 s (9/16) | 17.8 ms | **1627 s** (1646) | **least-bad static** (partial both) |
| **ours init=48** | **4 s (16/16)** | **14.7 ms** | **1096 s** (1109) | **best — 41% below extremes, 33% below static48** |

The N=220 tuning landed on prediction (extremes equal to 0.2%, static48 the compromise).
The mechanism: **ours matches the KV-oracle's caching (16/16, 4 s) *and* the expert-oracle's
TPOT (14.7 ms — the best of any config) on one persistent server across the shift**, while
every fixed split collapses in its off-phase. Both phase orders agree within ~1%. Relocation
fired **5845× (KV→Expert)** and **1672× (Expert→KV)** — the swap is real memory movement.

Per-request latency vs **request index** (phase boundary marked, **log y-axis** so the
~50 s KV-miss spike and the ~2× expert-thrash spike are both visible): each static spikes
in its off-phase; ours stays low throughout.

**Figure.** Latency-over-index line (ours vs the three statics, boundary shaded) + a
badness-total bar. Uses the trace-**off** ours runs for latency.

### Experiment 2b — VRAM reallocation (the visual)

The trace-**on** ours runs (`VLLM_UNIFIED_POOL_TRACE=1`, shorter expert phase = 60 req to
keep the trace file manageable; raw trace kept in the boot log + a compact per-request
composition CSV): stacked area of pool composition (expert-sb / prefix-KV-pg / free) over
request index, aligned under the E2 latency line, with a **relocations-per-step** overlay.
At the boundary you see the KV band shrink and the expert band grow (KV→Expert) or the
reverse (Expert→KV), with relocation spiking — visual proof that latency stability is
driven by memory physically moving between experts and KV, which a fixed split cannot do.

---

## Tooling notes

- `scripts/summarize_trace.py` (from Phase 3): extend for the `UNIFIED RELOCATE` marker
  and super-block occupancy fields.
- Drivers to add under `scripts/`: `run_oracle_matching.sh` (E1, incl. the per-request
  convergence trace), `run_phase_shift.sh` (E2/E2b — the `KV→X` and `X→KV` runs), reusing
  the envelope.
- **Pod/setup gotchas (see `GPU_VALIDATION.md`):** keep the HF cache on `/workspace`
  (persistent volume) not the container overlay; `pod_setup.sh` needs a follow-up
  `pip install vllm==0.17.1 && pip uninstall -y vllm` to pull runtime deps, then
  re-extract the `dist-info` from the wheel so platform detection sees CUDA; kill GPU
  procs by PID (an orphaned `VLLM::EngineCore` survives `pkill -f api_server` and holds
  the util gate).
