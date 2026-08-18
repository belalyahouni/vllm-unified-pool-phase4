# Eviction Bias Tracing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade level-1 unified-pool tracing and its summarizer so A/B runs directly measure per-layer residency, hit rate, diversity, evictions, and mixed-LRU choices.

**Architecture:** Extend existing text trace records rather than adding a telemetry subsystem. Emit cumulative layer state in `UNIFIED CACHE`, one structured record per actual mixed-LRU comparison, and pre-drop expert scores; stream these records through the existing summary script.

**Tech Stack:** Python, existing unified-pool trace gate, `re`, standalone deterministic tests.

## Global Constraints

- Keep latency experiments trace-off.
- Level 1 must contain all mechanism-success metrics.
- Do not add dependencies, per-token full LRU dumps, GPU timers, or a new logging framework.
- Preserve existing occupancy field meanings and allocator behavior.
- Do not stage unrelated workspace changes.

---

## File Structure

- `vllm/model_executor/layers/fused_moe/unified_pool.py`: emit structured cache, decision, and eviction records.
- `scripts/summarize_trace.py`: parse current records and aggregate per-layer outcomes.
- `scripts/test_unified_pool_logic.py`: capture runtime trace regressions using the existing allocator fake.
- `tests/test_summarize_trace.py`: verify report generation from a synthetic trace.

### Task 1: Runtime Trace Records

**Files:**
- Modify: `vllm/model_executor/layers/fused_moe/unified_pool.py`
- Test: `scripts/test_unified_pool_logic.py`

**Interfaces:**
- Produces: `UNIFIED CACHE ... expert-bias=... hits=... misses=... ever-activated=...`
- Produces: `UNIFIED DECISION side=expert-miss|kv-alloc ... expert-score=... kv-score=... chosen=expert|kv`
- Produces: expert `UNIFIED EVICT` fields `score=... step=...`.

- [ ] Write tests that enable `_TRACE_ENABLED`, capture stdout, and assert literal required fields.
- [ ] Run `python3 scripts/test_unified_pool_logic.py --deterministic-only` and confirm failures for missing records.
- [ ] Capture the score before `layer.drop`, append cache counters, and emit decisions only in branches where both candidates exist.
- [ ] Rerun deterministic tests and confirm they pass.

### Task 2: Streaming Trace Summary

**Files:**
- Modify: `scripts/summarize_trace.py`
- Create: `tests/test_summarize_trace.py`

**Interfaces:**
- Consumes the records from Task 1.
- Produces markdown sections `Per-layer bias outcomes` and `Mixed-LRU decisions`.

- [ ] Write a synthetic current-format trace test with two layers, both decision sides, and expert evictions.
- [ ] Run `python3 -m unittest tests.test_summarize_trace -v` and confirm the obsolete parser fails.
- [ ] Replace obsolete cache and eviction regexes, parse decisions, and aggregate counts and mean signed margins.
- [ ] Rerun the summary test and confirm it passes.

### Task 3: Verification and Pod Deployment

**Files:**
- Verify all changed files.

**Interfaces:**
- Consumes Tasks 1 and 2.
- Produces a pushed commit and verified RunPod checkout.

- [ ] Run `python3 scripts/test_unified_pool_logic.py`.
- [ ] Run `python3 -m unittest tests.test_summarize_trace -v`.
- [ ] Compile all changed Python files and run `git diff --check`.
- [ ] Commit only trace-related files and push `main`.
- [ ] Fast-forward `/root/vllm-unified-pool-phase4` on RunPod and clear changed module bytecode.
- [ ] Run deterministic tests and a synthetic summary on RunPod; verify expected fields and sections.
