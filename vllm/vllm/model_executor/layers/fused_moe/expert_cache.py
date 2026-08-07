# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LRU cache for MoE expert weights on the GPU.

Experts live on CPU pinned memory and the GPU keeps a fixed number of
slots holding the most recently used ones. A miss copies the expert from
CPU to GPU and evicts the least recently used slot that is not needed
this batch.
"""

import os
from collections import OrderedDict

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Resolve the trace gate once at module load. The trace fires per layer
# per forward step, and a per-call os.environ.get was measurable on the
# expert-offload path. Set VLLM_EXPERT_CACHE_TRACE=1 only when needed.
_EXPERT_CACHE_TRACE = (
    os.environ.get("VLLM_EXPERT_CACHE_TRACE", "0") == "1"
)


class ExpertCache:
    """Holds a small set of expert weights on GPU and pages the rest in from CPU."""

    def __init__(
        self,
        cache_size: int,
        cpu_w13: torch.Tensor,
        cpu_w2: torch.Tensor,
        device: torch.device,
    ) -> None:
        self.cache_size = cache_size
        self.num_experts = cpu_w13.shape[0]
        self.cpu_w13 = cpu_w13
        self.cpu_w2 = cpu_w2
        self.device = device

        assert cache_size <= self.num_experts, (
            f"cache_size ({cache_size}) must be <= num_experts "
            f"({self.num_experts})"
        )

        self.cache_w13 = torch.empty(
            (cache_size, *cpu_w13.shape[1:]),
            dtype=cpu_w13.dtype,
            device=device,
        )
        self.cache_w2 = torch.empty(
            (cache_size, *cpu_w2.shape[1:]),
            dtype=cpu_w2.dtype,
            device=device,
        )

        self.expert_to_slot: dict[int, int] = {}
        # OrderedDict iterates oldest-first, which is what we need for LRU.
        self.lru_order: OrderedDict[int, None] = OrderedDict()

        # Use a separate stream so CPU->GPU copies don't block the compute stream.
        self.transfer_stream = torch.cuda.Stream(device=device)

        self.hits = 0
        self.misses = 0

        self._warm_cache()

    def _warm_cache(self) -> None:
        """Load experts 0..cache_size-1 into the cache slots."""
        for slot in range(self.cache_size):
            expert_id = slot
            self.cache_w13[slot].copy_(self.cpu_w13[expert_id])
            self.cache_w2[slot].copy_(self.cpu_w2[expert_id])
            self.expert_to_slot[expert_id] = slot
            self.lru_order[expert_id] = None

        logger.info(
            "ExpertCache: warmed %d/%d experts on %s "
            "(w13: %s, w2: %s)",
            self.cache_size,
            self.num_experts,
            self.device,
            list(self.cache_w13.shape),
            list(self.cache_w2.shape),
        )

    def ensure_experts_loaded(self, needed_expert_ids: list[int]) -> None:
        """Load any missing experts into the GPU cache.

        Eviction skips experts that are themselves in needed_expert_ids so
        we don't evict and then immediately reload the same weight.
        """
        needed_set = set(needed_expert_ids)

        missing_expert_ids: list[int] = []
        hit_ids: list[int] = []
        for expert_id in needed_expert_ids:
            if expert_id in self.expert_to_slot:
                self.hits += 1
                hit_ids.append(expert_id)
            else:
                self.misses += 1
                missing_expert_ids.append(expert_id)

        if _EXPERT_CACHE_TRACE:
            print(
                f"ExpertCache: needed={needed_expert_ids} "
                f"hits={hit_ids} misses={missing_expert_ids}",
                flush=True,
            )

        if not missing_expert_ids:
            return

        # Walk LRU -> MRU, skipping anything still needed this batch.
        eviction_candidates = iter([
            expert_id
            for expert_id in self.lru_order
            if expert_id not in needed_set
        ])
        experts_and_slots_to_copy: list[tuple[int, int]] = []
        for expert_id in missing_expert_ids:
            evicted_expert_id = next(eviction_candidates)
            slot = self.expert_to_slot.pop(evicted_expert_id)
            del self.lru_order[evicted_expert_id]
            if _EXPERT_CACHE_TRACE:
                print(
                    f"ExpertCache: evict expert {evicted_expert_id} "
                    f"from slot {slot}, load expert {expert_id}",
                    flush=True,
                )

            self.expert_to_slot[expert_id] = slot
            self.lru_order[expert_id] = None
            experts_and_slots_to_copy.append((expert_id, slot))

        with torch.cuda.stream(self.transfer_stream):
            for expert_id, slot in experts_and_slots_to_copy:
                self.cache_w13[slot].copy_(self.cpu_w13[expert_id], non_blocking=True)
                self.cache_w2[slot].copy_(self.cpu_w2[expert_id], non_blocking=True)

        # Compute stream waits for the copies to finish before reading the slots.
        torch.cuda.current_stream(self.device).wait_stream(self.transfer_stream)

    def mark_recently_used(self, expert_ids: list[int]) -> None:
        """Mark each expert as most recently used."""
        for expert_id in expert_ids:
            self.lru_order.move_to_end(expert_id, last=True)

    def log_stats(self) -> None:
        total = self.hits + self.misses
        hit_rate = self.hits / total * 100 if total > 0 else 0.0
        logger.info(
            "ExpertCache stats: hits=%d misses=%d total=%d hit_rate=%.1f%%",
            self.hits,
            self.misses,
            total,
            hit_rate,
        )
