# Phase 4 — Fine-grained pages + per-expert super-blocks (with KV relocation)

## Context

This is the deferred Phase 4 of the vLLM unified-pool dissertation project. Phase 3 shares a
per-layer GPU buffer between MoE expert weights and KV-cache blocks, but its page size is fixed
at **one full expert footprint = 12 MiB** (OLMoE BF16), which forces a KV block of **1,536
tokens**. A page can hold *only* a whole expert or a whole 1,536-token KV block — never a partial
one. When a workload's prefix demand is not a multiple of 1,536 tokens, the unused tail of the last
block sits idle → **internal fragmentation**. This is the biggest remaining limitation in Phase 3.

**Goal:** shrink the page to a **tunable** size (default = a 16-token KV block, the fused-MoE
kernel's minimum granularity) and group `F` contiguous pages into a **per-expert super-block**, so
KV blocks mix into the buffer at fine granularity (no wasted tail) while an expert still occupies
one contiguous region. A **relocation primitive** compacts a super-block on demand so experts stay
contiguous, letting us evict only the globally-coldest pages instead of whatever is physically in
the way. Keep Phase 3's exact operating envelope and make the **absolute minimal** set of changes.

**Forced architectural choice:** Phase 3 feeds the *stock* fused-MoE Triton kernel a single-stride
`torch.as_strided` view of the pool buffer (`unified_pool.py:135-148`). That only works if an
expert's `F` pages are **physically contiguous**. A scattered/gather layout would require rewriting
the kernel — out of scope. So the figure's **fixed super-block grid + relocation** design is not one
option among many; it is the only minimal-kernel-change path.

## Step 0 — Create the branch folder

Copy `150326-phase-3-unified-pool-no-staging/` → **`150326-phase-4-fine-grained-pages/`** (code tree:
`vllm/`, `scripts/`, `prompts/`, `README.md`, `.gitignore`; Phase-3 `logs/`, `results/`, `.git/`
excluded — they are Phase-3 run artifacts / history). Update the README's Branches list to add the
Phase 4 bullet. Phase 3 stays pristine as the evaluated baseline. All edits below are inside this
folder.

## Design

### Two ID spaces (the core idea)

- **Page / `block_id` space (unchanged, vLLM-owned):** `block_id ∈ [0, num_gpu_blocks)`, one page =
  `page_size_bytes = expert_pool_page_tokens * bytes_per_token` = **one KV block**. This is the
  `BlockPool.blocks` index and the `free_block_queue` element. **KV always allocates single pages.**
- **Super-block space (new, expert-owned):** `F = expert_slot_bytes // page_size_bytes` (must divide
  evenly). Super-block `s` = pages `[s*F, s*F+F)`; `num_super_blocks = num_gpu_blocks // F`. An expert
  occupies exactly one super-block — its `w13`+`w2` bytes are contiguous because
  `w13_bytes + w2_bytes == expert_slot_bytes == F * page_size_bytes`.

A page's global state stays exactly one of **KV-prefix** (global — same `block_id` holds that
block's KV in every layer), **expert** (per-layer, tracked in `super_block_holder`), or **free** —
mutually exclusive, enforced by the existing `_evict_prefix_globally` / `_broadcast_drop_all_layers`
calls.

### Strided view change (`unified_pool.py:135-148`)

Only the leading dimension changes; inner shape/strides/offsets are untouched:
- add `super_block_stride_elems = F * page_size_elems` (== `expert_slot_bytes // elem_size`);
- `pool_w13_view`: `size=(num_super_blocks, *w13_shape)`, `stride=(super_block_stride_elems, *w13_inner)`, `offset=0`;
- `pool_w2_view`: same size/stride, `offset = w13_bytes // elem_size`;
- `stride(-1)==1` asserts (`:151-152`) still hold. w13/w2 boundary need **not** be page-aligned.

### Relocation primitive (the crux) — safe because of `ref_cnt`

vLLM treats `block_id` as immutable ("append-only block tables", `block_pool.py:47-51`). We never
rename a block. Instead: **relocate only cached-prefix pages with `ref_cnt == 0`** by transferring
the hash + bytes between two fixed-id blocks. This is provably safe here: `EngineCore.step()` runs
`scheduler.schedule()` fully before the forward, and every block referenced by the current step has
`ref_cnt ≥ 1` (via `block_pool.touch`, `block_pool.py:459`). So a `ref_cnt==0` cached page is *not*
read this step; moving it mid-forward into a free hole corrupts nothing, and **no block-table cell
is ever rewritten** (invariant preserved). This keeps us inside Phase 3's `async_scheduling==False`
+ `max_num_batched_tokens==1` + single-process envelope.

`_relocate_kv_page(A → B)` where A is a `ref_cnt==0` prefix page and B is a hole (`_block_hash is None`,
not inside any pinned super-block nor the target super-block):
1. **Bytes:** for every layer, `pool.pool_buffer.narrow(0, B*psz, psz).copy_(narrow(0, A*psz, psz))`
   on `transfer_stream` (prefix is global → all layers copy). Cost per relocation =
   `num_layers * page_size_bytes`; per miss ≤ F relocations ⇒ bounded by `num_layers * expert_slot_bytes`.
2. **Hash surgery** (add a `relocate_prefix_hash(src, dst)` method to `block_pool.py`, sibling of the
   existing `evict_prefix_hash` at `:511`, to keep this next to `cached_block_hash_to_block`):
   `h = blocks[A].block_hash` → `cached_block_hash_to_block.pop(h, A)` → `blocks[A].reset_hash()` →
   `blocks[B].block_hash = h` (setter asserts B was hash-free) → `cached_block_hash_to_block.insert(h, blocks[B])`.
3. **LRU:** `prefix_lru.pop(A)`; `prefix_lru[B] = <A's step>`. **Do NOT touch `block_holder`** — it
   tracks expert pages only, which are mutually exclusive with relocatable KV pages.
4. Order all relocation copies **before** the expert DMA on `transfer_stream`, then a single
   `wait_stream` on the compute stream (mirror `unified_pool.py:550-555`).

Future prefix hits resolve to B via `get_cached_block`; A is now a free hole.

### Super-block allocator (replaces `ensure_loaded` per-miss body + `_select_victim_block`)

Per expert miss (layer L, expert `eid`):
1. Classify each aligned super-block `s ∈ [1, num_super_blocks)` (skip `s=0`, see invariants). For its
   F pages mark each: **pinned/live** (page `ref_cnt>0`, or s in any layer's `pinned_super_blocks`),
   **expert** (in `super_block_holder`), **kv-prefix** (`ref_cnt==0` + hash), or **free**.
2. **Reject** any super-block containing a pinned/live page (can't vacate this step).
3. **Choose** the candidate minimizing relocation work: rank by count of kv-prefix pages *warmer than
   the global cold frontier* (ascending), tie-break fewest occupied pages. Fully-free/expert-only
   super-blocks cost zero relocations (common case) — prefer them.
4. **Vacate** the target: for each kv-prefix page — relocate it (§ relocation) if it is warm and a hole
   is available, else evict it (`_evict_prefix_globally`, `:332`). For each expert page —
   `_broadcast_drop_all_layers(s)` (cheap; experts are re-DMA-able). Holes = existing free pages +
   pages freed by evicting the globally-coldest kv-prefix pages elsewhere (walk `prefix_lru`
   oldest-first, skipping target + pinned super-blocks). Fallback-to-evict guarantees a miss never
   fails for lack of holes.
5. **Assign + DMA:** `assign(s, eid, step)`; `_add_holder(L, s)`; `pinned_super_blocks.add(s)`; DMA the
   expert into `[s*expert_slot_bytes, +expert_slot_bytes)`. Keep all F pages resident in
   `free_block_queue` (the Phase-3 trick — expert protection lives in the victim selectors, not the queue).
6. No vacatable candidate → raise (analog of `:649-654`).

**KV-side victim** (`_pick_one_kv_victim`, `:710-783`): still returns single pages. A page is
truly-free only if hash-free AND its super-block `p//F` is not held/pinned. If the KV-evicts-expert
branch wins, `_broadcast_drop_all_layers(s)` frees all F pages; return one, the other F−1 become holes.

### Eviction / allocation situations (all handled by the allocator)

Super-blocks are a **fixed grid** defined once at startup (`s → pages [s*F, s*F+F)`) and **never
relocated or re-aligned** — an expert always lands on one of these fixed windows. Two eviction shapes
exist. (a) Evicting a whole **expert** is trivial: an expert *is* a super-block and (by mutual
exclusion) holds no KV, so dropping it yields F contiguous free pages immediately. (b) Freeing space
from **KV** pages requires vacating a chosen window: evict the globally-coldest KV pages (creating
holes) and **relocate the warm survivors trapped inside the target** into those holes — we shuffle the
*survivors*, not the evicted pages. Target = the window minimizing relocations, guarded by recency
(never drop a *hot* expert or evict a *hot* KV page just to save a swap).

| # | Target super-block state | Action | Relocations |
|---|---|---|---|
| S1 | Already fully free | use directly | 0 |
| S2 | Holds only a cold (droppable) expert | drop expert, load new | 0 |
| S3 | All occupied pages are cold KV (coldest happen to align in one window) | evict them all | 0 |
| S4 | Cold KV + warm KV (the figure's case) | evict cold; relocate warm into holes | #warm |
| S5 | Free pages + warm KV | relocate warm into holes | #warm |
| S6 | Warm KV but too few holes | relocate what fits, evict the remainder (fallback — never fails) | ≤ #warm |
| S7 | Contains a live (`ref_cnt>0`) or pinned page | reject, try another window this step | — |
| S8 | No vacatable window anywhere | raise (pool undersized) | — |

Global consequence: KV is global (a block's data lives in every layer's buffer), so one relocation is
`num_layers` page-copies, and vacating a super-block for *one* layer's expert miss frees that window
**globally** — after which every layer may independently place its own expert there (per-layer expert
maps in `super_block_holder`, exactly as Phase 3 shares a `block_id` across layers).

## Minimal edit list (per file, with current Phase-3 line refs)

- **`config/offload.py`** — after `expert_cache_size` (`:102-107`) add
  `expert_pool_page_tokens: int = Field(default=16, ge=16)`; in `validate_offload_config` (`:118`)
  assert multiple of 16 and only meaningful with `expert_unified_pool`.
- **`engine/arg_utils.py`** — dataclass field after `:458`; CLI flag after `:1042` (auto-picked by
  `get_kwargs`, `:1006`); thread into `OffloadConfig(...)` after `:1926`. (Same pattern as `expert_cache_size`.)
- **`v1/worker/gpu_model_runner.py`** — `_unified_pool_stage1` (`:6605`, derivation `:6710-6744`):
  `page_size_bytes = page_tokens * bytes_per_token`; replace `block_size_tokens = expert_slot_bytes //
  bytes_per_token` (`:6724`) with `= page_tokens`; add `assert expert_slot_bytes % page_size_bytes == 0`
  and `F = expert_slot_bytes // page_size_bytes`; repoint the `%16` check (`:6726`) at `page_tokens`;
  `cache_config.block_size = page_tokens` (`:6740`); stash `page_size_bytes` and `F`.
  `setup_unified_pool` (`:6771-6880`): `page_size_bytes = stashed` (was `= expert_slot_bytes`, `:6818`);
  pass `expert_slot_bytes`/`F` into `UnifiedPool` (`:6842`); capacity check (`:6857-6870`) uses
  `num_super_blocks = num_gpu_blocks // F`, `available = num_super_blocks - 1`, warn warm-up costs
  `warm_count * F` pages.
- **`model_executor/layers/fused_moe/unified_pool.py`** (bulk of the work):
  - `UnifiedPool.__init__`: accept `expert_slot_bytes`; relax assert `:101` to `w13+w2 ==
    expert_slot_bytes == F*page_size_bytes`; compute `F`, `num_super_blocks`; build super-block views.
  - Rename block→super-block bookkeeping: `block_id_at → super_block_id_at` (`:157-162`, value = view
    row = super_block_id), `expert_at_block/block_at_expert → expert_at_super_block/super_block_at_expert`,
    `pinned_blocks → pinned_super_blocks`, `block_holder → super_block_holder`. `expert_lru` unchanged.
  - `assign`/`drop` (`:182-208`) key on super_block. `_broadcast_drop_all_layers`/`_on_kv_allocation`
    (`:303-361`) translate page `p → s = p//F`; pinned assert on `pinned_super_blocks`.
  - `_dma_expert_into_block_async` (`:787-799`): `sb_offset = super_block_id * expert_slot_bytes`.
  - **New** `_relocate_kv_page(A, B)` (§ relocation).
  - Replace `ensure_loaded` body + `_select_victim_block` + KV victims with the super-block allocator.
  - `warm_up` (`:391-474`): warm `warm_count` super-blocks; generalize the round-trip check
    (`:452-468`) to index views by `super_block_id`.
  - Trace occupancy math (`:809-887`) → super-block units; add a `UNIFIED RELOCATE A->B` line.
- **`v1/core/block_pool.py`** — add `relocate_prefix_hash(src_id, dst_id)` (sibling of `evict_prefix_hash`,
  `:511`) doing the hash surgery in § relocation step 2. No other core changes.
- **`model_executor/layers/fused_moe/layer.py`** — `_forward_with_unified_pool` (`:1620-1712`):
  `remapped_ids = pool.super_block_id_at[topk_ids]` (`:1649`); `global_num_experts = pool.num_super_blocks`
  (`:1695`); paranoid check (`:1659-1686`) indexes `pool.pool_w13_view[sb]` / `expert_at_super_block`.
  `unified_pool_stage1` (`:765-805`) already returns `expert_slot_bytes` — no change.
- **`v1/engine/core.py` / `v1/worker/gpu_worker.py`** — no changes (`F`/`page_size_bytes` are derived
  worker-side; the RPC just passes `block_pool`).

## Recommended implementation order (each milestone is runnable)

- **M1 — plumbing:** config knob + page-size derivation only. Decouple `page_size_bytes` from
  `expert_slot_bytes`. Validate at `page_tokens` giving **F=1** (== Phase 3 behaviour) → output must
  match Phase 3 exactly.
- **M2 — super-blocks, no relocation:** views/DMA/warm-up/forward-remap + allocator that only uses
  fully-free/expert super-blocks and **evicts** (does not relocate) trapped KV. Validate at a modest
  **F=4** (`page_tokens=384`) then **F=96** (`page_tokens=16`): warm-up round-trip check + paranoid
  check pass; output-equivalence vs Phase 3 at `--seed 1`.
- **M3 — relocation:** add `_relocate_kv_page` + `relocate_prefix_hash`; allocator relocates warm
  trapped pages instead of evicting. Validate output-equivalence still holds and trace shows warm pages
  surviving while evictions concentrate on the coldest.

## Invariants & edge cases

- **Reserve super-block 0:** page 0 is the permanent null block (`block_pool.py:175`), so `[0,F)` can
  never fully assemble. Allocator range `[1, num_super_blocks)`; capacity uses `num_super_blocks-1`.
- `num_gpu_blocks` need not be a multiple of F — tail pages serve as KV/holes, never a full super-block.
- Relocation destination B must be hash-free and outside any pinned/target super-block.
- Trapped **live** (`ref_cnt>0`) page ⇒ super-block not vacatable this step; size `num_gpu_blocks` so an
  F-contiguous vacatable window always exists (live pages ≤ one request's context at batch=1).
- Manual hash move bypasses `enable_kv_cache_events`/`metrics_collector` — fine while off (this config);
  note if ever enabled.

## Verification

1. **M1 gate:** F=1 run byte-identical outputs to Phase 3.
2. **Startup invariants:** generalized warm-up round-trip check (`unified_pool.py:452-468`) passes for all
   warmed super-blocks × all layers — proves the F-strided view + super-block DMA land correctly.
3. **`VLLM_UNIFIED_POOL_PARANOID=1`** (`layer.py:1655`): L0/forward-0 view-vs-CPU check passes — proves no
   corruption at kernel-call time after relocation/DMA.
4. **Output equivalence:** adapt `scripts/run_unified_g1.sh` (Phase 3 uses `--block-size 1536
   --num-gpu-blocks-override 68 --expert-cache-size 64`). Phase 4: add `--expert-pool-page-tokens 16`,
   set `--block-size 16` and a proportionally larger `--num-gpu-blocks-override` (e.g. 68·96 = 6528 for
   the same byte budget). Run Phase 3 vs Phase 4 with identical `--seed 1`; greedy/enforce-eager token
   outputs must match. Any divergence = corrupted KV or expert.
5. **Fragmentation win:** `VLLM_UNIFIED_POOL_TRACE=1` (`run_unified_g1_trace.sh`) — compare the `UNIFIED
   CACHE` occupancy line; Phase 4 retains many more small KV-prefix pages for the same byte budget. The
   new `UNIFIED RELOCATE` counter should stay within the `num_layers * expert_slot_bytes` per-miss bound.
6. **Allocator stress:** many-distinct-prefix workload (`scripts/run_many_prefixes.sh` /
   `run_manyprefix_unified_only.sh`) to force fragmentation and exercise relocation + the "no vacatable
   super-block" raise path.
