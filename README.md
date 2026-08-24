# Dynamic Unified Memory Pool for MoE Inference

A single GPU memory region shared between KV cache pages and expert weights, with a mixed LRU that decides which to evict under memory pressure. No manual tuning, no static partitioning, no restarts.

Built on vLLM v0.17.1. Tested with OLMoE-1B-7B on NVIDIA L4.

## The Problem

In standard MoE inference, GPU memory is split at startup: some for KV cache, some for expert weights. The split is fixed. A KV-heavy workload wastes expert memory; an expert-heavy workload starves the KV cache. Changing the split requires a restart.

## The Solution

**Unified pool.** One per-layer GPU buffer shared between KV pages and expert pages. A mixed LRU compares recency across both types and evicts whichever is colder. The pool adapts automatically to workload shifts.

## Quick Start

```bash
# Unified pool
vllm serve olmoe-1b-7b-0125 \
  --expert-offload \
  --expert-unified-pool \
  --enable-prefix-caching
```

## Flags

| Flag | Default | Purpose |
|---|---|---|
| `--expert-offload` | `False` | CPU-pinned expert weights with GPU cache |
| `--expert-cache-size` | `0` | Starting expert slots per layer |
| `--expert-unified-pool` | `False` | Shared GPU memory for KV and experts |
| `--expert-pool-page-tokens` | `16` | Page granularity in tokens |

## Constraints

- Requires `--enable-prefix-caching` and `--expert-offload`
- `tensor_parallel_size=1`, `pipeline_parallel_size=1`
- `--expert-pool-page-tokens` must be a multiple of 16

## Project Structure

- `vllm/` — modified vLLM source (fork of v0.17.1)
- `scripts/` — test launchers, trace summarizer, prompt generators
- `prompts/` — JSONL prompt files
- `logs/` — bench, server, and trace logs
- `results/` — parsed benchmark results

## Traces and Debugging

- `VLLM_UNIFIED_POOL_TRACE=1` — emit per-step pool trace (evictions, allocations, LRU state)
- `VLLM_UNIFIED_POOL_PARANOID=1` — verify pool views match CPU weights on first forward
- `VLLM_UNIFIED_POOL_RELOCATE=0` — disable KV relocation for A/B comparison