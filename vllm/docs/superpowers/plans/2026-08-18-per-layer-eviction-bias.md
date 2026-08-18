# Per-Layer Eviction Bias Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal depth-based expert timestamp bias so broad early layers yield memory to valuable prefix KV while hot late-layer experts remain protected.

**Architecture:** Add one nonnegative scale to `OffloadConfig`, compute one fixed bias per sorted MoE-layer rank during unified-pool setup, and stamp the existing expert LRU with `step + bias`. Keep KV timestamps and the measured coldest-holder shared-fate policy unchanged.

**Tech Stack:** Python, Pydantic configuration, PyTorch-backed vLLM unified pool, standalone CPU allocator harness.

## Global Constraints

- Keep the existing mixed LRU, fixed aligned super-blocks, and contiguous expert storage.
- Add no dependency, online estimator, per-layer quota, kernel change, or static-cache behavior change.
- `expert_bias_scale=0.0` must reproduce unbiased expert timestamps.
- The default OLMoE bias must range from `-480` to `0` manager steps over 16 MoE layers.
- Preserve TP=1, PP=1, eager execution, prefix caching, and synchronous scheduling requirements.
- Do not commit unless explicitly requested.

---

## File Structure

- `vllm/config/offload.py`: define the validated bias scale.
- `vllm/engine/arg_utils.py`: expose and propagate `--expert-bias-scale`.
- `vllm/v1/worker/gpu_model_runner.py`: compute rank-based biases and pass them into each pool.
- `vllm/model_executor/layers/fused_moe/unified_pool.py`: calculate/store biases, stamp expert recency, and score shared victims.
- `scripts/test_unified_pool_logic.py`: deterministic policy regressions and randomized invariant coverage.

### Task 1: Specify Bias Calculation and Timestamp Semantics

**Files:**
- Modify: `vllm/model_executor/layers/fused_moe/unified_pool.py:85-256`
- Test: `scripts/test_unified_pool_logic.py:296-361,550-560`

**Interfaces:**
- Produces: `compute_layer_timestamp_bias(layer_rank: int, num_moe_layers: int, scale: float) -> float`
- Produces: `UnifiedPool.timestamp_bias: float`
- Changes: `UnifiedPool.assign(..., step: int)` and `bump_expert(..., step: int)` store a `float` biased timestamp.

- [ ] **Step 1: Repair the test fixture and write failing calculation tests**

Initialize `ever_activated` and `timestamp_bias` in `make_pool`. Add assertions that 16 layers produce `-32.0`, approximately `-14.9333` at rank 8, and `0.0`; scale zero produces zero; and a one-layer model produces zero.

- [ ] **Step 2: Run deterministic tests and confirm the new function is absent**

Run: `python3 scripts/test_unified_pool_logic.py --deterministic-only`
Expected: FAIL because `compute_layer_timestamp_bias` is not defined or importable.

- [ ] **Step 3: Implement the minimal calculation and timestamp stamping**

Use:

```python
def compute_layer_timestamp_bias(
    layer_rank: int, num_moe_layers: int, scale: float
) -> float:
    if num_moe_layers <= 1:
        return 0.0
    depth = layer_rank / (num_moe_layers - 1)
    return -scale * 30 * num_moe_layers * (1 - depth)
```

Accept `timestamp_bias` in `UnifiedPool.__init__`, change `expert_lru` values to `float`, and stamp both assignments and bumps with `step + self.timestamp_bias`.

- [ ] **Step 4: Add assignment and bump regressions**

Set a fixture layer's bias to `-480`, assign at step 500, and assert timestamp 20. Bump at step 600 and assert timestamp 120.

- [ ] **Step 5: Run deterministic tests**

Run: `python3 scripts/test_unified_pool_logic.py --deterministic-only`
Expected: calculation and timestamp tests pass; any remaining failure is confined to global holder scoring.

### Task 2: Favor KV Over Broad Early Experts

**Files:**
- Modify: `vllm/model_executor/layers/fused_moe/unified_pool.py:968-999`
- Test: `scripts/test_unified_pool_logic.py:492-509`

**Interfaces:**
- Consumes: biased `UnifiedPool.expert_lru` timestamps from Task 1.
- Consumes: existing `_oldest_global_expert()` coldest-holder shared-fate scoring.

- [ ] **Step 1: Retain the existing coldest-holder regression**

Use two shared super-blocks with holder timestamps `(1, 100)` and `(20, 30)` and assert that the victim is the first super-block at score 1.

- [ ] **Step 2: Run the regression**

Run: `python3 scripts/test_unified_pool_logic.py --deterministic-only`
Expected: PASS, preserving the measured high-init convergence fix.

- [ ] **Step 3: Add an expert-versus-KV policy regression**

Construct pressure with no truly free page. Assert that an early expert stamped at 68 loses to prefix KV stamped at 80, while an unpenalized late expert stamped at 100 wins against the same prefix timestamp.

- [ ] **Step 4: Run deterministic tests**

Run: `python3 scripts/test_unified_pool_logic.py --deterministic-only`
Expected: PASS.

### Task 3: Add Configuration and Pool Construction Plumbing

**Files:**
- Modify: `vllm/config/offload.py:97-128`
- Modify: `vllm/engine/arg_utils.py:453-459,1033-1048,1926-1933`
- Modify: `vllm/v1/worker/gpu_model_runner.py:6797-6872`
- Modify: `vllm/model_executor/layers/fused_moe/unified_pool.py:97-120,1167-1202,1248-1261`
- Test: `scripts/test_unified_pool_logic.py`

**Interfaces:**
- Produces: `OffloadConfig.expert_bias_scale: float = Field(default=1.0, ge=0.0)`
- Produces: CLI `--expert-bias-scale`.
- Consumes: `compute_layer_timestamp_bias` from Task 1.

- [ ] **Step 1: Write a failing configuration regression**

Extend the lightweight `OffloadConfig` test object with `expert_bias_scale=1.0`. Assert the field default is represented in source plumbing and rely on Pydantic's `ge=0.0` constraint for negative input rejection.

- [ ] **Step 2: Add the field and CLI propagation**

Add the field to `OffloadConfig`, `EngineArgs`, argument registration, and `create_engine_config` construction, following `expert_pool_page_tokens` exactly.

- [ ] **Step 3: Compute and inject biases during Stage 2 setup**

Sort unique MoE layer indices, map each index to rank, calculate its bias from the configured scale, and pass `timestamp_bias` to each `UnifiedPool`. Log each layer's value through existing startup and trace/stat lines.

- [ ] **Step 4: Run deterministic tests**

Run: `python3 scripts/test_unified_pool_logic.py --deterministic-only`
Expected: PASS.

### Task 4: Full Host Verification

**Files:**
- Verify only.

**Interfaces:**
- Consumes all preceding changes.
- Produces evidence that the allocator remains internally consistent.

- [ ] **Step 1: Run all randomized allocator invariants**

Run: `python3 scripts/test_unified_pool_logic.py`
Expected: `PASS: 400 randomized seeds, all invariants held.`

- [ ] **Step 2: Compile modified Python files**

Run: `python3 -m py_compile vllm/vllm/config/offload.py vllm/vllm/engine/arg_utils.py vllm/vllm/model_executor/layers/fused_moe/unified_pool.py vllm/vllm/v1/worker/gpu_model_runner.py scripts/test_unified_pool_logic.py`
Expected: exit 0.

- [ ] **Step 3: Run targeted Ruff checks**

Run from `vllm/`: `uv run ruff check vllm/config/offload.py vllm/engine/arg_utils.py vllm/model_executor/layers/fused_moe/unified_pool.py vllm/v1/worker/gpu_model_runner.py ../scripts/test_unified_pool_logic.py`
Expected: exit 0, or report environment unavailability without claiming success.

- [ ] **Step 4: Review the final diff for scope**

Run: `git diff --check` and inspect `git diff --` for only the five implementation/test files plus these design documents. Do not alter unrelated user changes.

### Task 5: CUDA A/B Validation Matrix

**Files:**
- No required source change.

**Interfaces:**
- Consumes: `--expert-bias-scale {0.0,0.5,1.0,1.5}`.
- Produces: the scale selected from measured KV-heavy and expert-heavy behavior.

- [ ] **Step 1: Run the existing E1 unified configuration at scale zero**

Use the existing L4 envelope and unified pool flags, adding `--expert-bias-scale 0.0`. Record KV-heavy warm TTFT/hits, per-layer resident experts, and expert-heavy TPOT.

- [ ] **Step 2: Repeat at scales 0.5, 1.0, and 1.5**

Keep model, prompts, pool size, cache initialization, ordering, trace state, and seeds unchanged. Trace and latency runs must remain separate because tracing previously inflated TTFT.

- [ ] **Step 3: Select the smallest effective scale**

Prefer the lowest scale that retains all 16 KV-heavy warm prefixes and reduces early-layer residency. Reject a scale with a meaningful expert-heavy TPOT regression. If `1.0` succeeds, retain the estimated `-480..0` default.
