# 150326

This is the code repository for my BSc dissertation at the University of Leeds (COMP3931, 2025/26). The project is about letting expert weights and the KV cache share the same per-layer GPU memory in vLLM, instead of splitting the memory between them at startup and never changing the split.

The code is built on top of vLLM v0.17.1 and was tested with OLMoE-1B-7B-0924-Instruct on a single NVIDIA L40.

## Branches

The project was built in phases, and each phase lives on its own branch so they can be checked out and run independently.

- `base-vllm`: the unmodified vLLM v0.17.1 snapshot that the project was forked from. Used as a clean baseline.
- `phase-1-expert-offload`: a static expert cache. Each MoE layer keeps a fixed number of experts on the GPU and pages the rest in from CPU pinned memory. This is the offloading foundation that everything else builds on.
- `phase-2-unified-pool-mvp`: the first end-to-end unified pool. Expert pages and KV blocks share a single per-layer buffer with a dual LRU. The kernel still reads from a per-layer staging tensor, so the pool runs end to end but the kernel does not depend on it.
- `phase-3-unified-pool-no-staging`: the staging tensors are removed and the kernel reads expert weights directly from the pool buffer through a strided view. This is the version evaluated in the report.
- `phase-4-fine-grained-pages`: the page size is shrunk (tunable via `--expert-pool-page-tokens`, default 16) and pages are grouped into per-expert super-blocks of F contiguous pages, where F = expert size / page size. An expert still spans one contiguous super-block (the kernel reads it through the same strided view, now with a super-block stride), while KV blocks mix into the buffer at the smaller page granularity so a partially-filled KV block no longer wastes a whole expert's worth of memory. When an expert miss needs a contiguous super-block, warm cached-prefix KV pages trapped inside it are relocated into holes (moving the hash identity between two fixed-id blocks, so vLLM's block tables stay append-only) rather than evicted, so only the globally-coldest pages are dropped. Phase 3 is the special case F=1 (`--expert-pool-page-tokens 1536` for OLMoE).

## What is where

Phase 3 has most of the material because the evaluation was done on that branch.

- `vllm/`: the modified vLLM source. The project's code changes are mostly under `vllm/vllm/model_executor/layers/fused_moe/` and `vllm/vllm/v1/`.
- `scripts/`: prompt generators, the trace summariser, and the shell scripts that drive each test.
- `prompts/`: the JSONL prompt files the generators produce.
- `logs/`: bench, server and trace logs from each test (`test1A`, `test1B`, `test2A`, `test2B`, `static_sweep`, `budget_sweep`, `many_prefixes`).
- `results/`: parsed JSON results from those runs.

The earlier branches contain only the code for that phase plus a small amount of supporting material.

## Running it

The exact commands and flags for each test are in the matching `scripts/run_*.sh` file. The relevant flags are:

- `--expert-offload`: turn on expert offloading.
- `--expert-cache-size N`: number of expert slots per layer.
- `--expert-unified-pool`: use the unified pool (phase-2, phase-3 and phase-4 branches).
- `--expert-pool-page-tokens N`: (phase-4 only) unified-pool page size in tokens; must be a multiple of 16 and divide the expert size in tokens. Default 16. F = expert-tokens / N pages per super-block.
- `VLLM_UNIFIED_POOL_TRACE=1`: emit the per-step pool trace.
- `VLLM_UNIFIED_POOL_PARANOID=1`: (phase-3/4) one-shot check that the pool views match CPU weights at the first forward.
- `VLLM_UNIFIED_POOL_RELOCATE=0`: (phase-4 only) disable KV relocation (evict-only) for A/B comparison; default on.

Phase 4 has a dedicated launcher, `scripts/run_unified_g1_phase4.sh`, which sets these flags and scales `--num-gpu-blocks-override` by F so the KV byte budget matches Phase 3 (making token outputs directly comparable). See its header for the bring-up ladder (F=1 → F=4 → F=96) and the GPU verification recipe.
