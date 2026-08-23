# Workstream A — what the unified pool's shared address space costs

Measured on RunPod **NVIDIA L4 (23 GB)**, torch 2.10.0+cu129, fork at
`938fca8`. Raw data in `new_results/prof_a/`.

Two instruments:

* `VLLM_UNIFIED_POOL_PROFILE=1` — in-process profiler
  (`vllm/model_executor/layers/fused_moe/pool_profiler.py`) that splits the
  pool's work into GPU byte movement (CUDA events on the transfer stream)
  and the Python victim-selection logic (`perf_counter`), plus per-tier
  counts. Zero-cost when off: the `timed()` decorator returns the
  undecorated function.
* `scripts/microbench_pool.py` — standalone, no server/model/bench.
  `--part gpu` times the byte-movement primitives; `--part cpu` drives the
  real allocator logic at full scale through the fuzzer's fakes and sweeps
  the fraction of the pool holding evictable KV.

## Headline

**Data movement is not the bottleneck. The Python victim search is, by two
orders of magnitude, and only in the KV-heavy direction.**

At 95% KV occupancy, resolving **one** expert miss costs **197 ms of pure
Python** against **0.95 ms** for the 12 MiB DMA it exists to schedule —
a **208x** ratio. The relocation machinery that the design discussion
treats as the expensive part costs 0.116 ms per page.

This inverts the working assumption (including mine): the contiguity
requirement is *not* where the overhead lives, so removing it would not
recover most of the cost. The victim search would.

## 1. GPU byte movement (`--part gpu`, 200 reps, CUDA events)

| op | mean | p99 | bytes | effective |
|---|---|---|---|---|
| `expert_h2d` — one expert, pinned host → pool | **0.946 ms** | 0.947 ms | 12 MiB | 12.39 GiB/s |
| `reloc_page` — one logical page, 16 × 128 KiB D2D | **0.116 ms** | 0.195 ms | 2 MiB | 16.83 GiB/s |
| `reloc_fused` — same 2 MiB as one contiguous copy | **0.0122 ms** | 0.016 ms | 2 MiB | 160.32 GiB/s |

* One expert load == **8.2 page relocations**.
* Splitting a page across per-layer buffers costs **9.5x** a single
  contiguous copy of the same bytes. Relocation is launch-overhead-bound
  (16 tiny copies), not bandwidth-bound.
* `expert_h2d` at 12.39 GiB/s is PCIe-bound and is the irreducible term.

### Relocation cost vs page size

| F | page | mean | bytes moved | effective |
|---|---|---|---|---|
| 1 | 12 MiB | 1.728 ms | 192 MiB | 108.5 GiB/s |
| 6 | 2 MiB | 0.267 ms | 32 MiB | 117.0 GiB/s |
| 24 | 512 KiB | 0.109 ms | 8 MiB | 71.9 GiB/s |
| 96 | 128 KiB | 0.108 ms | 2 MiB | 18.1 GiB/s |
| 384 | 32 KiB | 0.110 ms | 0.5 MiB | 4.4 GiB/s |

Below ~512 KiB pages the time is flat at ~0.11 ms: it is entirely the
fixed cost of issuing 16 copies. Fine pages are therefore free in
*latency* but waste bandwidth (18 GiB/s at the F=96 we ship vs 117 GiB/s
at F=6).

## 2. CPU selection logic (`--part cpu`, M=67, F=96, 16 layers, 6432 blocks)

Microseconds per call, as the share of the pool holding evictable
cached-prefix KV rises:

| KV frac | KV super-blocks | prefix pages | `kv_cost_sweep` | `coldest_prefix_page` | `oldest_global_expert` | `first_free_page` |
|---|---|---|---|---|---|---|
| 0.00 | 1 | 48 | 613 | 19 | 106 | 8.7 |
| 0.25 | 16 | 768 | 17,020 | 300 | 86 | 8.7 |
| 0.50 | 33 | 1584 | 53,823 | 611 | 56 | 8.7 |
| 0.75 | 50 | 2400 | 110,157 | 920 | 27 | 8.6 |
| 0.95 | 63 | 3024 | **164,557** | 1165 | 5 | 8.8 |

End-to-end paths (rebuilt state per call):

| KV frac | `vacate_kv_super_block` | `select_kv_victim` | `evict_for_expert` |
|---|---|---|---|
| 0.00 | 477 µs | 6.1 µs | 831 µs |
| 0.25 | 5,053 µs | 2.9 µs | 20,605 µs |
| 0.50 | 9,876 µs | 2.9 µs | 62,875 µs |
| 0.75 | 14,519 µs | 4.4 µs | 129,353 µs |
| 0.95 | 36,750 µs | 3.9 µs | **197,211 µs** |

`evict_for_expert` is ~99% `kv_cost_sweep`. The cause is structural:
`_evict_for_expert` calls `_kv_super_block_cost` for every super-block
(`unified_pool.py:924`), and each call rescans the entire block array
(`:994`) — O(M × num_blocks) = 67 × 6432 ≈ 430k Python iterations per
miss. `select_kv_victim` stays flat at a few µs, so the KV-allocation
direction is fine; only the expert-miss direction blows up.

Caveat: nothing is pinned in the synthetic state, whereas a real forward
pins the 8 needed experts' super-blocks, and `_kv_super_block_cost`
early-returns on pinned and expert-held super-blocks. These figures are
therefore an upper bound — but pinning excludes only ~8 of 67
super-blocks, so the bound is tight.

## 3. Why the serving runs did not reveal this

Profiled serving runs (`new_results/prof_a/*_reloc*.json`):

| cell | expert misses | miss tiers | relocations | expert DMA |
|---|---|---|---|---|
| `paper_exp` (M=67) | 877 | 821 co-located, 56 free-pure | **0** | 1005 × 0.953 ms |
| `tight_exp` (M=18) | 143,776 | **100% `expert-local`** | 0 | 144,048 × 0.952 ms = **137 s** |

Expert-heavy workloads only ever *grow* the expert set, so they resolve
misses from free space (M=67) or by same-type eviction (M=18) and never
enter the cross-type path. In `tight_exp` the CPU logic was 5.9 s against
137 s of DMA — 4% — which is why the expert-heavy direction looks cheap
and why the microbenchmark, not a serving run, is what exposes the
KV-heavy cost.

Sanity check: 1005 DMAs = 877 misses + 128 warm-up loads (8 experts × 16
layers), exactly as expected.

`tight_exp_reloc0` was killed mid-benchmark and is partial — do not
compare it against `tight_exp_reloc1`.

## 4. The fix (commits `bf9cd3a`, `3120431`)

`_evict_for_expert` no longer scores every super-block. `_cheapest_kv_super_block`
walks `prefix_lru` once — the pages that exist, not every block in the
pool — and ranks candidates by **fewest warm pages** (equivalently most
cold pages, the rule the paper states), trying them cheapest-first and
applying the validity checks only as each is tried. Warm means above the
cold frontier: the step of the F-th coldest prefix page, mirroring
`_vacate_kv_super_block`, which drops a page only when nothing colder
exists to displace. At most F pages are ever cleared, so the frontier
needs no tunable constant.

Relocation is untouched: `_vacate_kv_super_block` and `_relocate_kv_page`
still move warm pages into scattered holes and kill only the coldest.

Cost per operation (µs, same pod, 20 reps):

| KV frac | `evict_for_expert` before | after | | `vacate_kv_super_block` before | after |
|---|---|---|---|---|---|
| 0.00 | 831 | **23.9** | | 500 | **65.8** |
| 0.25 | 20,605 | **186.6** | | 5,323 | **556.6** |
| 0.50 | 62,875 | **353.0** | | 10,307 | **949.0** |
| 0.75 | 129,353 | **526.6** | | 15,474 | **1,324.7** |
| 0.95 | **197,211** | **655.4** | | **29,902** | **2,635.8** |

At 95% KV occupancy a miss that has to evict KV went from ~227 ms of
Python (197 ms deciding + 30 ms clearing) to **~3.3 ms** — about 69x.
Deciding is now 0.66 ms against the 0.95 ms DMA it schedules.

`vacate_kv_super_block` improved 11.3x, not the "well under 1 ms" first
predicted: the remaining 2.6 ms is one pass over `prefix_lru` plus a sort
to build the coldest-first list, and the per-page relocation bookkeeping
itself. Reducing it further would need incrementally maintained state
rather than a per-vacate rebuild.

Three deliberate consequences:

* Candidate ranking is by *risk to warm KV*, not copy count: a
  mostly-cold super-block can cost an extra relocation, since cold pages
  are preserved too when holes are available. The fuzzer sizes the trade
  at 651 relocations across 400 seeds vs 645 before — six extra page
  copies, ~0.7 ms total — for never disturbing warm KV to save a copy.
* The mixed-LRU score is now the chosen super-block's *oldest* page rather
  than its warmest, so the comparison is oldest-expert vs oldest-KV — what
  the Method section describes. The old warmest-page score made KV look
  warmer than it was and biased the decision toward evicting experts.
* `prefix_lru` insertion order no longer tracks recency. Nothing reads it
  that way; the level-2 verbose trace now sorts explicitly.

Verified: fuzzer 400 seeds pass with relocations unchanged at 651 across
the vacate rewrite, deterministic regressions pass, 25 unit tests pass.

## 5. Consequences for the paper

1. **The overhead claim in future work needs rewriting.** "Making experts
   non-contiguous would remove the relocation overhead, leaving only page
   tracking" is measurably the wrong emphasis: relocation is 0.116 ms/page
   and 1/8th of one expert load, while the victim search was 197 ms/miss
   -- and is now 0.55 ms after a pure bookkeeping change (section 4), with
   contiguity and the kernel untouched.
2. **The result is stronger than it looks.** The 1.48x under workload
   shift is achieved *while paying* up to ~197 ms of avoidable Python per
   expert miss in exactly the phase the shift creates (a KV-full pool
   taking expert misses). That overhead is not fundamental to the design.
3. **The cheapest real optimisation was bookkeeping, not layout** --
   now done, 358x, kernel and contiguity untouched (section 4).
4. **Page size is a real knob with a real cost.** F=96 gives 18 GiB/s on
   relocation vs 117 GiB/s at F=6, for identical latency. Worth a sentence.

## Reproducing

```bash
# CPU half runs anywhere; GPU half needs a CUDA device (no model weights).
python3 scripts/microbench_pool.py --part both --json out.json

# In-process profile of a real serving run (adds overhead; not for latency):
VLLM_UNIFIED_POOL_PROFILE=1 VLLM_UNIFIED_POOL_PROF_JSON=/tmp/p.json \
  python3 -m vllm.entrypoints.openai.api_server ...
```

Pod gotcha not in `GPU_VALIDATION.md`: the RunPod PyTorch template sets
`LD_LIBRARY_PATH=/usr/local/cuda/lib64`, whose CUDA 12.8 `libnvJitLink.so.12`
shadows the pip 12.9 one that torch 2.10.0+cu129 needs, so `import torch`
fails with `undefined symbol: __nvJitLinkGetErrorLogSize_12_9` and
`pod_setup.sh` aborts before cloning. Prepend
`.../dist-packages/nvidia/nvjitlink/lib` to `LD_LIBRARY_PATH`.
