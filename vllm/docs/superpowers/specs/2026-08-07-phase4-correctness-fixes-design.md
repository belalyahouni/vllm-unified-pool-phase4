# Phase 4 Correctness Fixes Design

## Goal

Make Phase 4 relocation and super-block eviction correct for the documented OLMoE configuration, while also closing the inherited pin-cleanup and cross-layer expert-recency defects and safely supporting models whose attention and MoE layers do not coincide.

## Scope

- Fix relocation's missing page-size state.
- Copy relocated KV pages in every attention-layer pool buffer, not only registered MoE layers.
- Preserve prefix recency across relocation and select prefix victims by timestamp.
- Rank vacatable KV super-blocks by predicted relocation work, occupied pages, and recency instead of only the warmest resident page.
- Clear pins after every forward-path failure without counting a failed forward as completed.
- Score a cross-layer expert super-block by its warmest holder before broadcast eviction.
- Validate `expert_pool_page_tokens` only when unified pooling is active.
- Add deterministic regressions while retaining the randomized invariant test.

No allocator rewrite, kernel changes, compatibility layer, dependency, or unrelated cleanup is included.

## Architecture

`GpuModelRunner.setup_unified_pool` will normalize every attention-layer KV tensor to the addressable pool range and pass that complete mapping plus the page size to `UnifiedPoolManager`. Per-MoE `UnifiedPool` objects remain responsible only for expert views and expert mappings. Relocation will iterate the manager's complete KV-buffer mapping.

Prefix LRU timestamps remain authoritative. Relocation transfers the source timestamp to the destination and restores timestamp order; victim selection scans timestamps so correctness does not depend on insertion order.

KV-window selection will use a local cost calculation matching the actual vacate operation: predicted relocated target pages first, total occupied target pages second, and the target's warmest timestamp last. This preserves the existing local allocator shape without introducing a second policy object.

Forward pin lifetime will be guarded by `try/finally`. Pin clearing is separated from successful-forward accounting so an exception cannot leak pins or advance recency counters.

## Error Handling and Invariants

- Manager construction validates positive page size and exact buffer capacity.
- Every registered MoE layer must have a corresponding attention KV buffer.
- Relocation sources remain hash-bearing, unreferenced pages; destinations remain hash-free pages outside held and pinned super-blocks.
- Failed forwards clear all pins, restore temporary fused-MoE state, and do not increment completion counters.
- A KV allocation cannot evict a super-block based on one cold holder when another holder is hot.

## Testing

Add deterministic tests for physical relocation copying, timestamp ordering, victim selection after relocation, all-attention-layer copy coverage, relocation-cost candidate ranking, exceptional pin cleanup, warmest-holder expert scoring, and conditional configuration validation. Continue running the 400-seed state-machine test, Python compilation, source diff checks, and available lint. GPU warm-up, paranoid checking, Phase 3 output equivalence, and relocation stress remain required on a CUDA host.
