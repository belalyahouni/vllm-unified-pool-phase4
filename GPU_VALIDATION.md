# Phase 4 — GPU validation & experiment notes

Findings worth keeping from validating Phase 4 (fine-grained pages + per-expert
super-blocks + KV relocation) on a real GPU. Focus: how to reproduce the setup,
how to run it, what workloads actually exercise the design, and what we proved.

Hardware used: RunPod **NVIDIA L4 (23 GB)**, model `allenai/OLMoE-1B-7B-0924-Instruct`
(64 experts/layer, top-8, 16 layers; one expert ≈ 12 MiB BF16 = 1536 tokens of KV;
`max_position_embeddings = 4096`).

---

## 1. Status — what is validated on GPU

| Property | Result |
|---|---|
| Builds & runs (fork + torch 2.10 + vLLM 0.17.1 kernels) | ✅ |
| Warm-up round-trip check (F-strided super-block view + DMA vs CPU truth) | ✅ at **F=1 and F=96** |
| Paranoid check (view vs CPU at kernel-call time) | ✅ |
| Coherent generation; **F=1 output byte-identical to F=96** (page-size-independent) | ✅ (this is the equivalence result) |
| Direction 1: KV evicts experts (`kv-evicts-expert`) under memory pressure | ✅ |
| Direction 2: expert reclaims KV super-block, **relocating warm survivors** (`kv-vacate` + `RELOCATE`) | ✅ **104 relocations, zero corruption** |
| Pure-Python fuzzer (`scripts/test_unified_pool_logic.py`) | ✅ 400 seeds / ~450 relocations, all invariants hold |

The fuzzer previously found and we fixed two real bugs: (1) multi-block KV allocation
double-removing a prefix page from the free queue; (2) `ensure_loaded` leaving a
mapping without loaded bytes if a later miss raised. Both fixed in the repo.

---

## 2. Pod setup that works (reproduction recipe)

**Use a RunPod *default PyTorch template*** (e.g. `runpod/pytorch:...`), **not** a bare
image. RunPod's own templates keep the container alive and expose SSH; bare images do not.

Then one command does everything:
```
curl -fsSL https://raw.githubusercontent.com/belalyahouni/vllm-unified-pool-phase4/main/scripts/pod_setup.sh | bash
```
`scripts/pod_setup.sh` (committed): installs `torch==2.10.0` (cu129, else cu128),
`git clone`s the fork, downloads the `vllm==0.17.1` wheel purely for its compiled
`.so` kernels, drops them into the fork tree, installs vLLM's deps + drops the stock
`vllm` package so the fork stays active (via a `.pth`), and installs dist-info +
`_version.py`. Ends with `SETUP_OK`.

Serve with `python3 -m vllm.entrypoints.openai.api_server` (there is **no `vllm`
console script** — the pip package is uninstalled so the fork on the `.pth` wins).

### Setup pitfalls (do NOT repeat these)
- **RunPod direct-TCP SSH endpoints are unreachable from our dev sandbox** — only the
  proxy `<id>@ssh.runpod.io` works. The proxy forces an interactive PTY and ignores a
  normal `ssh host "cmd"` arg; pipe commands via **stdin** (base64-wrap them; strip PTY
  ANSI/prompt noise with `tr -d '\r' | perl -pe 's/\e\[[0-9;?]*[A-Za-z]//g'`).
- **`vllm/vllm-openai:v0.17.1` auto-runs `vllm serve` and eats the whole GPU.** Its
  Docker ENTRYPOINT can't be neutralized by RunPod's "Container Start Command" (that
  sets CMD/args, which get *appended* to the entrypoint). And `vllm serve` is pid-1's
  child, so killing it stops the container. Avoid this image for interactive work.
- **Bare `pytorch/pytorch` images** don't stay up with a `sleep infinity` start command
  on RunPod ("container is not running"). Use a RunPod template instead.
- The fork's base is **slightly ahead of the released vLLM 0.17.1** (it references
  `DCPCommBackend`, `SignalCallback`, `group_and_batch_mm_kwargs` the release lacks), so
  **overlaying only the changed files onto a stock 0.17.1 install fails**. You must use
  the fork's *whole* Python tree.
- **Editable install (`pip install -e .`) does not work**: setuptools_scm can't derive
  the version (repo nests `vllm/` under the git root → set
  `SETUPTOOLS_SCM_PRETEND_VERSION=0.17.1`), and even then PEP 660's editable finder
  mis-maps the nested package as a namespace (`vllm.__file__` is `None`, submodules fail).
- **Non-editable `pip install .`** fails on newer setuptools rejecting vLLM 0.17.1's
  `pyproject.toml` `project.license` (dual file/text definition). Hence the
  wheel-for-kernels + `.pth` approach in `pod_setup.sh`.
- **Port 8001 is taken by RunPod's nginx** — serve on 8100.
- **OLMoE caps at `--max-model-len 4096`** (RoPE); 8192 is rejected (would NaN).

---

## 3. How to run Phase 4

Required envelope (asserted at stage-1; all mandatory): `TP=PP=1`, `--enable-prefix-caching`,
`--enforce-eager`, `--max-num-batched-tokens 1`, `async_scheduling=False`
(`--no-async-scheduling`), `--attention-backend TRITON_ATTN` (needs contiguous per-block
K/V layout).

Phase-4 knobs:
- `--expert-offload --expert-unified-pool`
- `--expert-pool-page-tokens P` — page size in tokens (default 16). `F = 1536 / P`
  pages per super-block for OLMoE. `P=16 → F=96`; `P=1536 → F=1` (== Phase 3 behaviour).
- `--expert-cache-size N` — experts warmed at startup (also the rough resident target).
- `--block-size P` and `--num-gpu-blocks-override (num_super_blocks * F)` — the pool has
  `num_super_blocks` super-blocks; each expert = one super-block = F pages.

Env vars: `VLLM_UNIFIED_POOL_RELOCATE` (default 1; `=0` for evict-only A/B),
`VLLM_UNIFIED_POOL_PARANOID=1` (one-shot view check on L0 first forward),
`VLLM_UNIFIED_POOL_TRACE=1` (per-step occupancy + evict/claim/relocate lines; `=2` verbose).

Trace markers to grep:
- `UNIFIED CACHE L{i} step=.. F=.. expert_sb X/Y ours (expert-ours-sb, expert-other-sb, prefix-pages, alloc-kv-pages, pinned-sb)` — occupancy over time.
- `UNIFIED KV_CLAIM page=.. sb=.. tier=truly-free|kv-evicts-prefix|kv-evicts-expert` — KV allocation.
- `UNIFIED EVICT ... tier=expert-local|kv-vacate|make-hole|kv-broadcast` — evictions.
- `UNIFIED RELOCATE src=.. dst=.. step=..` — a warm KV page moved to a hole.
- Boot: `UnifiedPool warm-up sanity check passed: N pairs` and `Phase-4 paranoid: ... verified`.

---

## 4. Behavioural findings (the important part)

### The pool self-partitions under ordinary workloads
With short prompts, **relocation / `kv-vacate` never fire**, because:
1. Experts grow via misses to fill the whole expert super-block range (`[1, num_super_blocks)`),
   regardless of `--expert-cache-size` (that's only the *starting* count).
2. KV confines itself to the reserved super-block 0 (its ~95 non-null pages are KV-usable)
   and, when it needs more, evicts its *own* coldest prefixes rather than spilling into the
   expert range.
3. With `--expert-cache-size ≥ top_k (8)`, a layer always has a non-needed resident expert to
   drop (`expert-local`), so that branch beats `kv-vacate` in the dual-LRU.

So experts (hot) and KV (reserved zone) coexist in disjoint super-blocks and never contend.
This is arguably a *positive* property — the design avoids the cost of relocation in the
common case — but it means you cannot observe relocation without a crafted workload.

### The workload that exercises the full KV↔expert flex (and relocation)
**Tight pool + long-context prompts.** Concretely what worked:
```
--num-gpu-blocks-override 1728   # 18 super-blocks (heavily restricted)
--expert-cache-size 16
--expert-pool-page-tokens 16     # F = 96
--max-model-len 4096
```
Then feed **long (~3.7k-token) repetitive prompts**. A single long prefill:
- balloons KV to ~234 live pages (~2.5 super-blocks) → **KV evicts cold experts**
  (`kv-evicts-expert` on whole super-blocks; the repetitive text uses a *narrow* expert
  set, so most experts are cold and evictable); then
- its own expert misses need those super-blocks back → **`kv-vacate` relocates the warm KV
  pages** to holes and loads the expert into the vacated contiguous region.

We saw a super-block flip **expert → KV → expert**, with a contiguous run of 12 warm KV
pages relocated intact (e.g. `src=1248..1259 → dst=717..728`) rather than dropped. A single
long prompt in a tight pool thrashes *both* directions continuously — the explicit
"KV-heavy phase then expert-heavy phase" split was not even necessary.

Levers that force the flex, in order of effect:
- **Long context** (large `--max-model-len` + long prompts) → big KV that must displace experts.
- **Small pool** (`num_super_blocks` not much larger than `--expert-cache-size`) → contention.
- `--expert-cache-size < top_k (8)` also forces `kv-vacate` (resident experts are then all
  needed each step, so `expert-local` has nothing to evict) — but risks pool exhaustion; use with care.

### Cost/behaviour observed
- Relocation moves a page's bytes in **every attention layer's** KV buffer (KV is global),
  bounded by ~`num_layers * expert_slot_bytes` per expert miss.
- No corruption across ~97k expert evictions (earlier run) and 104 relocations — output
  stayed coherent throughout.
- `--max-num-batched-tokens 1` makes prefill 1 token/step, so long-context / KV-heavy
  workloads are **slow** (~minutes per multi-thousand-token prompt). Drive serve in the
  background and poll a logfile.
