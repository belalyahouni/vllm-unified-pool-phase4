# Per-Layer Eviction Bias Design

## Goal

Prevent broad early MoE layers from retaining nearly every expert at the expense of globally useful prefix KV, while preserving the small hot expert sets in later layers and retaining the unified pool's dynamic allocation.

## Policy

Keep the existing mixed LRU and change only expert timestamps. Each MoE layer receives a fixed non-positive timestamp bias based on its rank among the model's MoE layers:

```text
depth = moe_rank / (num_moe_layers - 1)
bias_steps = -expert_bias_scale * 30 * num_moe_layers * (1 - depth)
```

For a single-MoE-layer model, the bias is zero. `expert_bias_scale` defaults to `1.0`, must be nonnegative, and `0.0` restores unbiased timestamps.

OLMoE has 16 MoE layers, so the default bias ranges linearly from `-480` manager steps at the first layer to `0` at the last. The manager clock advances once per MoE-layer forward, making 480 steps approximately 30 complete token rounds. This directly expresses the estimated 30:1 cost ratio between recomputing a missed prefix page and reloading one expert. It is also long enough to demote broadly cycled early-layer experts: with top-k 8 and 64 broadly used experts, an individual expert is expected roughly once per eight token rounds, while a specialist layer using eight experts can touch each expert every round. A cached-prefix miss costs tens of milliseconds, compared with roughly 1.5 ms to reload an expert, so close eviction decisions should favor KV without imposing a static expert quota.

Every assignment and hit records `manager_step + layer_bias`. KV timestamps remain unchanged. The bias therefore affects both expert-miss decisions and KV-allocation decisions through the existing comparisons.

## Shared Super-Blocks

Retain the existing shared-fate policy: KV reclaim scores a super-block by its coldest biased holder and broadcasts eviction to every holder. The project changed from warmest to coldest scoring specifically because warmest co-locations starved KV and prevented high initial expert occupancy from converging. The new depth bias supplies the missing layer value signal without reversing that measured fix: early broad experts become the cold holders first, while later specialists remain individually warm and are cheaply reloaded only when co-location gives them shared fate with a low-value expert.

## Behavior

- Required experts are always loaded and pinned for the active forward; the policy never changes correctness.
- There is no per-layer capacity or static partition.
- Under KV-heavy pressure, unpinned early-layer experts lose close comparisons to prefix KV and are reloaded if needed later.
- Under expert-heavy pressure with little reusable prefix KV, repeatedly touched experts still displace old KV. When late layers also use all experts, their unpenalized timestamps protect shared expert super-blocks, so steady-state expert residency should remain close to the current policy.
- The initial scale is an estimate to validate with existing KV-heavy and expert-heavy experiments. Tuning changes one scalar rather than the policy.

## Scope

- Add `expert_bias_scale` to offload configuration and CLI plumbing.
- Compute biases from sorted MoE-layer rank during unified-pool setup.
- Store the bias in each `UnifiedPool` and apply it in `assign` and `bump_expert`.
- Expose the configured bias in startup and trace output.
- Extend the standalone deterministic and randomized allocator tests.

No online diversity estimator, hard quota, extra policy object, kernel change, dependency, or static-cache behavior change is included.

## Testing

Deterministic tests cover the exact 16-layer endpoint values, disabled bias, single-layer behavior, biased assignment and access, expert-versus-KV decisions, and shared super-block protection. Existing relocation, failure cleanup, configuration, and randomized allocator invariants continue to run.

Host verification consists of deterministic tests, all 400 randomized seeds, Python compilation, and targeted Ruff checks. CUDA validation compares scales `0.0`, `0.5`, `1.0`, and `1.5` on the existing E1 KV-heavy and expert-heavy workloads. The starting default is accepted if it preserves all 16 KV-heavy warm hits and materially reduces broad early-layer residency without a meaningful expert-heavy TPOT regression; otherwise only the scalar default is adjusted.
