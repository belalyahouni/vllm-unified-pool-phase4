# Phase 4 Correctness Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct Phase 4 KV relocation, LRU policy, super-block selection, pin lifetime, and cross-layer eviction without changing the fused-MoE kernel or the documented operating envelope.

**Architecture:** Give `UnifiedPoolManager` explicit page-size state and every attention-layer KV pool buffer. Keep timestamp recency authoritative, model candidate costs locally, and make forward pin cleanup exception-safe. Extend the existing deterministic/fuzz harness rather than adding dependencies.

**Tech Stack:** Python, PyTorch/CUDA streams, vLLM BlockPool, Pydantic, standalone invariant test script.

## Global Constraints

- Modify only the Phase 4 tree.
- Preserve fixed aligned super-blocks and contiguous expert storage.
- Never rename a KV block ID during relocation.
- Keep `async_scheduling=False`, `max_num_batched_tokens=1`, TP=1, and PP=1 assumptions.
- Add no dependencies and do not change the fused-MoE kernel.
- Do not commit unless the user explicitly requests it.

---

## File Structure

- `vllm/model_executor/layers/fused_moe/unified_pool.py`: manager KV-buffer ownership, relocation, LRU ordering, candidate costs, pin accounting, and expert-victim scoring.
- `vllm/v1/worker/gpu_model_runner.py`: normalize and register every attention-layer KV buffer.
- `vllm/model_executor/layers/fused_moe/layer.py`: exception-safe forward cleanup.
- `vllm/config/offload.py`: conditional page-token validation.
- `scripts/test_unified_pool_logic.py`: deterministic regressions and randomized invariants.

### Task 1: Make relocation own complete KV-buffer state

**Files:**
- Modify: `vllm/model_executor/layers/fused_moe/unified_pool.py:269-319,1079-1089`
- Modify: `vllm/v1/worker/gpu_model_runner.py:6822-6883`
- Test: `scripts/test_unified_pool_logic.py`

**Interfaces:**
- `UnifiedPoolManager(block_pool, device, pages_per_super_block, page_size_bytes, kv_pool_buffers)` consumes `dict[int, torch.Tensor]` containing every attention-layer addressable KV byte buffer.
- `_copy_page_all_layers(src_id: int, dst_id: int) -> None` copies exactly one page in every registered attention layer.

- [ ] **Step 1: Write failing deterministic tests**

Add a manager fixture with two byte buffers, including one layer without a `UnifiedPool`, and assert `_copy_page_all_layers(1, 3)` copies the page in both buffers. Also assert construction rejects a buffer shorter than `num_gpu_blocks * page_size_bytes`.

- [ ] **Step 2: Run the focused tests and confirm failure**

Run: `python3 scripts/test_unified_pool_logic.py --deterministic-only`

Expected: failure from missing constructor state or missing non-MoE copy coverage.

- [ ] **Step 3: Implement explicit manager state**

Change construction conceptually to:

```python
def __init__(
    self,
    block_pool,
    device: torch.device,
    pages_per_super_block: int,
    page_size_bytes: int,
    kv_pool_buffers: dict[int, torch.Tensor],
) -> None:
    assert page_size_bytes > 0
    self.page_size_bytes = page_size_bytes
    required_bytes = block_pool.num_gpu_blocks * page_size_bytes
    self.kv_pool_buffers = {}
    for layer_idx, buffer in kv_pool_buffers.items():
        assert buffer.numel() == required_bytes
        self.kv_pool_buffers[layer_idx] = buffer
```

Use `self.kv_pool_buffers.values()` in `_copy_page_all_layers`. In `setup_unified_pool`, slice every entry in `attn_layers` to `required_bytes` before manager construction, then use the normalized mapping when constructing each MoE pool.

- [ ] **Step 4: Run deterministic tests**

Run: `python3 scripts/test_unified_pool_logic.py --deterministic-only`

Expected: all Task 1 tests pass.

### Task 2: Preserve prefix recency and choose the actual oldest prefix

**Files:**
- Modify: `vllm/model_executor/layers/fused_moe/unified_pool.py:509-556,1000-1023`
- Test: `scripts/test_unified_pool_logic.py`

**Interfaces:**
- `_relocate_kv_page` transfers the source timestamp without making it newer.
- `_oldest_prefix_page() -> tuple[int | None, int | None]` returns the eligible minimum-timestamp page.

- [ ] **Step 1: Write failing recency regressions**

Create prefixes with timestamps `{1: 2, 2: 8, 3: 10}`, relocate page 1 to page 4, and assert page 4 remains the oldest. Deliberately scramble insertion order and assert KV victim selection still chooses the minimum timestamp.

- [ ] **Step 2: Run tests and confirm the wrong victim**

Run: `python3 scripts/test_unified_pool_logic.py --deterministic-only`

Expected: relocated page is placed at the MRU end or victim selection returns a newer page.

- [ ] **Step 3: Implement timestamp-authoritative ordering**

After moving the recency entry, rebuild the ordered dictionary stably by timestamp:

```python
self.prefix_lru[dst_id] = src_step
self.prefix_lru = OrderedDict(
    sorted(self.prefix_lru.items(), key=lambda item: item[1])
)
```

Add a scan that returns the eligible entry with the minimum timestamp and use it from `_pick_one_kv_victim` instead of breaking on the first dictionary entry.

- [ ] **Step 4: Run deterministic tests**

Run: `python3 scripts/test_unified_pool_logic.py --deterministic-only`

Expected: recency and victim tests pass.

### Task 3: Rank KV windows by actual relocation work

**Files:**
- Modify: `vllm/model_executor/layers/fused_moe/unified_pool.py:782-923`
- Test: `scripts/test_unified_pool_logic.py`

**Interfaces:**
- `_kv_super_block_cost(super_block_id: int) -> tuple[int, int, int] | None` returns `(predicted_relocations, occupied_pages, warmest_step)` or `None` for held/live/pinned windows.

- [ ] **Step 1: Write failing candidate-ranking tests**

Construct two vacatable windows: one densely occupied and slightly colder, another sparse with fewer pages requiring preservation. Assert the sparse/lower-relocation window wins. Add a tie where fewer occupied pages wins, then a full tie where the colder warmest timestamp wins.

- [ ] **Step 2: Run tests and confirm current warmest-only ranking fails**

Run: `python3 scripts/test_unified_pool_logic.py --deterministic-only`

Expected: selection returns the window chosen only by warmest timestamp.

- [ ] **Step 3: Implement a local cost simulation**

Count external pure holes and sort external prefix timestamps. Simulate the existing warmest-first vacate loop for each target: a target page counts as relocated when a pure hole exists or an external prefix is strictly colder; otherwise it is evicted in place. Rank candidates by predicted relocations, then occupied target pages, then warmest target timestamp. Keep live, pinned, held, and super-block-zero rejection unchanged.

- [ ] **Step 4: Run deterministic and randomized tests**

Run: `python3 scripts/test_unified_pool_logic.py`

Expected: deterministic tests pass and `PASS: 400 randomized seeds, all invariants held.`

### Task 4: Make forward pin cleanup exception-safe

**Files:**
- Modify: `vllm/model_executor/layers/fused_moe/unified_pool.py:698-778`
- Modify: `vllm/model_executor/layers/fused_moe/layer.py:1634-1712`
- Test: `scripts/test_unified_pool_logic.py`

**Interfaces:**
- `release_pinned(layer, completed: bool = True) -> None` always clears pins and increments `forward_count` only when completed.
- `end_forward_step()` runs only after a successful kernel call.

- [ ] **Step 1: Write failing cleanup tests**

Force `_select_super_block_for_expert` to raise after one hit and one successful miss. Assert pins are empty and `forward_count` is unchanged. Add a forward-path test seam that raises after `ensure_loaded` and checks the same conditions.

- [ ] **Step 2: Run tests and confirm pins leak**

Run: `python3 scripts/test_unified_pool_logic.py --deterministic-only`

Expected: pinned super-blocks remain populated.

- [ ] **Step 3: Implement cleanup without false completion accounting**

Guard the forward body with:

```python
completed = False
try:
    manager.ensure_loaded(pool, needed_expert_ids)
    # remap, checks, and kernel call
    completed = True
    return result
finally:
    manager.release_pinned(pool, completed=completed)
    if completed:
        manager.end_forward_step()
```

Keep weight-data restoration in its existing inner `try/finally`. Also clear pins inside `ensure_loaded` before re-raising selection/DMA failures so direct callers receive the same guarantee.

- [ ] **Step 4: Run deterministic and randomized tests**

Run: `python3 scripts/test_unified_pool_logic.py`

Expected: cleanup regressions and all randomized invariants pass.

### Task 5: Score expert super-blocks by their warmest holder

**Files:**
- Modify: `vllm/model_executor/layers/fused_moe/unified_pool.py:927-948`
- Test: `scripts/test_unified_pool_logic.py`

**Interfaces:**
- `_oldest_global_expert()` returns the evictable super-block whose maximum holder timestamp is globally smallest.

- [ ] **Step 1: Write a failing cross-layer recency test**

Create super-block A with holder timestamps 1 and 100 and super-block B with timestamps 20 and 30. Assert B is selected because evicting A would also discard the timestamp-100 expert.

- [ ] **Step 2: Run test and confirm A is incorrectly selected**

Run: `python3 scripts/test_unified_pool_logic.py --deterministic-only`

Expected: current per-layer scan returns A.

- [ ] **Step 3: Implement holder-aware scoring**

Iterate `super_block_holder`, skip any super-block pinned by any layer, resolve each holder's expert and LRU timestamp, compute `max(holder_steps)`, and choose the smallest such maximum. Do not change broadcast eviction.

- [ ] **Step 4: Run deterministic and randomized tests**

Run: `python3 scripts/test_unified_pool_logic.py`

Expected: holder-recency test and 400-seed invariants pass.

### Task 6: Restrict page-token validation to unified pooling

**Files:**
- Modify: `vllm/config/offload.py:171-180`
- Test: `scripts/test_unified_pool_logic.py` or an import-safe focused config test

**Interfaces:**
- `OffloadConfig(expert_unified_pool=False, expert_pool_page_tokens=17)` is accepted because the value is inactive.
- The same value with `expert_unified_pool=True` is rejected.

- [ ] **Step 1: Write both validation cases**

Assert inactive non-multiple values pass and active non-multiple values raise the existing message.

- [ ] **Step 2: Run and confirm the inactive case fails**

Run the focused test command chosen by the existing environment.

Expected: Pydantic validation error for the inactive case.

- [ ] **Step 3: Make validation conditional**

Use:

```python
if self.expert_unified_pool and self.expert_pool_page_tokens % 16 != 0:
    raise ValueError(...)
```

- [ ] **Step 4: Run focused validation tests**

Expected: inactive case passes and active case fails.

### Task 7: Final verification and review

**Files:**
- Review all files listed above.

- [ ] **Step 1: Run the complete standalone suite**

Run: `python3 scripts/test_unified_pool_logic.py`

Expected: deterministic regressions pass, 400 randomized seeds pass, and relocations are exercised.

- [ ] **Step 2: Compile changed Python files without leaving bytecode artifacts**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  vllm/config/offload.py \
  vllm/model_executor/layers/fused_moe/layer.py \
  vllm/model_executor/layers/fused_moe/unified_pool.py \
  vllm/v1/worker/gpu_model_runner.py \
  scripts/test_unified_pool_logic.py
```

Expected: exit code 0.

- [ ] **Step 3: Run available lint**

Run `ruff check` on the changed files if Ruff is installed; otherwise report it as skipped rather than installing dependencies.

- [ ] **Step 4: Recheck only the Phase 3-to-Phase 4 diff**

Run `diff -qr` excluding `.git`, `__pycache__`, and `*.pyc`, then inspect the changed hunks. Confirm no unrelated files changed.

- [ ] **Step 5: Record unavailable GPU gates**

If no suitable CUDA host is available, explicitly leave warm-up sanity, `VLLM_UNIFIED_POOL_PARANOID=1`, Phase 3 output equivalence, and relocation stress as unverified. Do not claim end-to-end correctness from CPU tests.
