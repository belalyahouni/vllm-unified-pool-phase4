# Eviction Bias Tracing Design

## Goal

Make level-1 unified-pool traces sufficient to determine whether per-layer eviction bias changes residency and mixed-LRU decisions beneficially, without using verbose per-token LRU dumps or contaminating latency runs.

## Runtime Trace

Keep the existing level-1 `UNIFIED CACHE` snapshot and add cumulative layer metrics: `hits`, `misses`, `ever-activated`, and the configured `expert-bias`. Existing occupancy fields remain unchanged so composition tooling can be updated without changing their meaning.

Add one `UNIFIED DECISION` line only when the mixed LRU compares an expert candidate with a KV candidate:

- `side=expert-miss`: a missing expert needs a super-block and compares the calling layer's coldest resident expert against the selected KV super-block.
- `side=kv-alloc`: KV needs a page and compares the globally coldest biased expert holder against the coldest prefix page.

Each line includes the current manager step, relevant layer, expert and KV scores, and `chosen=expert|kv`. These are policy scores, not wall-clock costs. They provide direct evidence that early biased experts lose decisions that an unbiased expert would win.

Expert eviction lines include the evicted expert's score and current manager step. The score must be captured before removing the LRU mapping.

## Summary

Update `scripts/summarize_trace.py` for the current `UNIFIED CACHE` grammar and the new fields. The report includes:

- Per-layer bias, last resident count, activated diversity, cumulative hit rate, and distinct/count evictions.
- Mixed-LRU decision counts by side, choice, and layer.
- Score-margin aggregates showing how decisively expert or KV won.
- Existing pool composition, pressure direction, relocation, and eviction summaries.

Level 1 is sufficient for mechanism and composition analysis. Level 2 remains optional for exact routed expert IDs. Latency comparisons continue to use trace-off runs.

## Testing

Add deterministic runtime tests that capture emitted `UNIFIED CACHE`, `UNIFIED DECISION`, and expert eviction lines. Add summary parser tests using a compact synthetic current-format trace and assert the generated report contains per-layer metrics and decision totals. Run the 400-seed allocator test and Python compilation after implementation.
