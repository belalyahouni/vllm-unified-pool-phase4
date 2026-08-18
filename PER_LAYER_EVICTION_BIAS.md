# Per-Layer Eviction Bias

## Problem

### Layer-Dependent Expert Diversity

MoE models do not use experts uniformly across layers. Early layers act as generalists, activating nearly all experts (e.g., 58 out of 64). Later layers act as specialists, activating only a small handful (e.g., 6 out of 64) for a given domain like coding. This pattern is consistent across OLMoE and similar MoE architectures.

### Why the Uniform Mixed LRU Fails

The unified pool uses a mixed LRU that compares the recency of expert accesses against KV page accesses and evicts whichever is colder. This treats every expert touch equally regardless of which layer it belongs to. The result is a structural imbalance:

**Early layers** cycle through almost all experts every few forward passes. Their expert LRU entries are constantly refreshed, making them appear recent and protected from eviction. But caching any single early-layer expert has low value — it is one of 60, and a different one will be needed next time. The LRU is protecting items with low reuse probability.

**Later layers** reuse the same few experts repeatedly. These experts are extremely valuable to cache — a miss means stalling the forward pass for a DMA transfer. But because there are so few of them, their LRU entries can age while early-layer experts dominate the recency rankings. High-value items are vulnerable to eviction.

### KV Cache Is Global

This is the critical compounding factor. A super-block occupied by any layer's expert is unavailable for KV in all layers. If early layers load 60 experts, those 60 super-blocks are consumed globally — even though later layers only need 6. The apparent free space in later layers is an illusion; the super-blocks are already taken by early-layer experts. Early-layer expert hoarding starves KV cache everywhere.

---

## Solution

### Per-Layer Timestamp Bias

Each layer is assigned a bias value that shifts how old its experts appear to the mixed LRU. When an expert is accessed, instead of recording the true timestamp, the system records `timestamp + bias`. Every downstream comparison — expert-miss eviction, KV-allocation victim selection — uses these biased timestamps automatically.

- **Early layers** (high diversity, low per-expert value): negative bias. Experts appear artificially older than they really are → evicted more readily → super-blocks freed for KV cache globally.

- **Later layers** (low diversity, high per-expert value): positive bias. Experts appear artificially newer → protected from eviction → the few critical experts stay resident.

The bias creates a gradient across depth: the pool tilts toward KV at early layers and toward experts at later layers, matching how the model actually uses them.

### Deriving the Bias: Cost Ratios

The bias is not arbitrary. It is derived from the relative cost of getting each decision wrong.

**Cost of evicting a KV page:** the page must be recomputed through every layer (prefill). On an L4 GPU, this takes roughly 30–80ms for a 16-token block.

**Cost of evicting an expert:** the expert weights (~24 MB) must be DMA'd from CPU pinned memory over PCIe. This takes roughly 1.5ms regardless of GPU compute speed.

**The ratio is roughly 30× on L4.** KV recomputation is far more expensive than expert reloading because L4 compute is ~10× slower than datacenter GPUs while PCIe bandwidth is the same. This means the bias should favor keeping KV in nearly all circumstances — the crossover point where an expert becomes worth keeping over KV is only reached at extremely low expert diversity (~4%).

### The Bias Formula

The bias for each layer is computed from its position in the model, using the observed pattern that expert diversity decreases with depth:

```
bias = scale × 30 × (0.04 − estimated_diversity)
estimated_diversity = 1 − (layer_index / total_moe_layers)
```

Where `scale` defaults to 1.0 and can be adjusted (or set to 0 to disable the bias entirely).

For a 16-layer model with default scale:

| Layer | Est. Diversity | Bias | Effect |
|---|---|---|---|
| 0 (early) | 1.0 | −29 | Strongly evict experts, protect KV |
| 7 (mid) | 0.53 | −15 | Moderately evict experts |
| 15 (late) | 0.0 | +1 | Slightly protect experts |

The bias is negative for nearly all layers on L4, reflecting the high cost of KV recomputation. The gradient still exists — early layers are more aggressive about evicting experts than late layers — but the zero point has shifted so far toward KV-favoring that even late layers only approach neutrality.

### What Changes in Practice

With the bias enabled, the unified pool naturally settles into a different allocation:

- Early layers hold fewer experts (perhaps 40–50 instead of 60), freeing 10–20 super-blocks for KV cache globally.
- Late layers hold roughly the same number of experts (they only needed a few anyway), but those experts are now protected from eviction by KV pressure.
- KV cache gains meaningful space without sacrificing expert hit rate where it matters.

### Preserving the Core Thesis

The bias mechanism preserves everything the unified pool stands for: a single shared memory region, a mixed LRU, no static partitioning, no manual tuning. The bias is computed automatically from layer position and requires no per-workload configuration. It is a refinement of the eviction policy, not a replacement of it.