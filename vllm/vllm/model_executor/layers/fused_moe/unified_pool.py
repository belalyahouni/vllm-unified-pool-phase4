# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified per-layer page pool for expert weights and KV blocks (Phase 4).

Each layer's KV byte tensor is aliased as a pool buffer shared between
cached-prefix KV blocks and expert weight pages.

Phase 4 shrinks the page: a page (== one KV block) is a tunable number
of tokens, and an expert now spans a *super-block* of F contiguous
pages, where F = expert_slot_bytes // page_size_bytes. There are two id
spaces:

* the page / block_id space (0..num_gpu_blocks), owned by vLLM's
  BlockPool: one page == one KV block. KV always allocates single pages.
* the super-block space (0..num_super_blocks), owned by the pool: an
  expert occupies exactly one super-block s == pages [s*F, s*F+F). The
  unmodified Triton fused MoE kernel reads an expert straight out of the
  pool buffer through an F-strided view whose row s is super-block s.

Phase 3 is the special case F == 1 (page == whole expert).

Mutual exclusion: at any time a super-block either holds expert weights
(in one or more layers; no KV in any of its pages) or its pages are
individually cached-prefix KV / free. Expert claims evict any KV in the
target super-block first; KV allocation broadcasts a drop of any expert
whose super-block the allocated page falls in.

topk_ids is rewritten per layer from global expert ids to super-block
ids, and global_num_experts is set to num_super_blocks for the kernel
call. No staging tensor, no extra GPU memory beyond the pool itself.
"""

from __future__ import annotations

import heapq
import os
from collections import Counter, OrderedDict, deque
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.pool_profiler import PROFILER

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

logger = init_logger(__name__)


# Trace gate is resolved once at module load. Per-call os.environ.get
# was measurable here because the gate fires tens of thousands of
# times per request. Levels: 0/unset = off, 1 = essential lines only
# (composition, evict, kv_claim, prefix add/remove, relocate), 2 = also
# dump step headers and the LRU snapshots (debug only, slow).
_TRACE_LEVEL = os.environ.get("VLLM_UNIFIED_POOL_TRACE", "0")
_TRACE_ENABLED = _TRACE_LEVEL in ("1", "2")
_TRACE_VERBOSE = _TRACE_LEVEL == "2"


def _trace_enabled() -> bool:
    return _TRACE_ENABLED


def _trace_verbose() -> bool:
    return _TRACE_VERBOSE


def move_experts_to_cpu(
    w13_weight: torch.nn.Parameter,
    w2_weight: torch.nn.Parameter,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move expert weight tensors to CPU pinned memory and return them."""
    cpu_w13 = w13_weight.data
    cpu_w2 = w2_weight.data
    if cpu_w13.is_cuda:
        cpu_w13 = cpu_w13.cpu()
    if cpu_w2.is_cuda:
        cpu_w2 = cpu_w2.cpu()
    if not cpu_w13.is_pinned():
        cpu_w13 = cpu_w13.pin_memory()
    if not cpu_w2.is_pinned():
        cpu_w2 = cpu_w2.pin_memory()
    return cpu_w13, cpu_w2


class UnifiedPool:
    """Per-layer pool state.

    Tracks which expert sits at which super-block, the per-layer LRU of
    experts, and the set of super-blocks pinned for the current forward.
    The kernel reads weights directly out of pool_buffer through the
    F-strided views (pool_w13_view, pool_w2_view), so a stale eviction
    here is silent corruption rather than a recoverable miss.
    """

    _UNLOADED = -1  # sentinel for super_block_id_at

    def __init__(
        self,
        layer_idx: int,
        num_experts: int,
        cpu_w13: torch.Tensor,
        cpu_w2: torch.Tensor,
        pool_buffer: torch.Tensor,
        page_size_bytes: int,
        expert_slot_bytes: int,
        w13_bytes: int,
        w2_bytes: int,
        device: torch.device,
        working_set_window: int,
    ) -> None:
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.cpu_w13 = cpu_w13
        self.cpu_w2 = cpu_w2
        self.pool_buffer = pool_buffer
        self.page_size_bytes = page_size_bytes
        self.expert_slot_bytes = expert_slot_bytes
        self.w13_bytes = w13_bytes
        self.w2_bytes = w2_bytes
        self.device = device
        self.working_set_window = working_set_window

        # One expert spans F contiguous pages (a super-block). Phase 3
        # is F == 1.
        assert expert_slot_bytes == w13_bytes + w2_bytes, (
            f"expert_slot_bytes ({expert_slot_bytes}) must equal "
            f"w13_bytes + w2_bytes ({w13_bytes + w2_bytes})"
        )
        assert expert_slot_bytes % page_size_bytes == 0, (
            f"expert_slot_bytes ({expert_slot_bytes}) must be a multiple "
            f"of page_size_bytes ({page_size_bytes})"
        )
        self.pages_per_super_block = expert_slot_bytes // page_size_bytes

        # The element size has to divide page_size, expert_slot_bytes,
        # w13_bytes and w2_bytes cleanly so the strided views land on
        # whole elements.
        elem_size = cpu_w13.element_size()
        assert page_size_bytes % elem_size == 0, (
            f"page_size_bytes ({page_size_bytes}) must be a multiple of "
            f"element size ({elem_size})"
        )
        assert expert_slot_bytes % elem_size == 0
        assert w13_bytes % elem_size == 0
        assert w2_bytes % elem_size == 0

        self._cpu_w13_bytes = cpu_w13.view(torch.int8).reshape(num_experts, -1)
        self._cpu_w2_bytes = cpu_w2.view(torch.int8).reshape(num_experts, -1)

        # Strided views over the pool buffer; the kernel reads from these.
        # Reinterpret the int8 buffer as the layer's weight dtype. Row s
        # of the view is super-block s == the F contiguous pages starting
        # at s * expert_slot_bytes. The leading stride is therefore the
        # whole super-block (F pages), while the inner strides stay the
        # natural row-major per-expert layout so DMAs deposit into the
        # same layout the view reads from.
        pool_typed = pool_buffer.view(cpu_w13.dtype)
        page_size_elems = page_size_bytes // elem_size
        super_block_stride_elems = expert_slot_bytes // elem_size
        w13_offset_elems = 0
        w2_offset_elems = w13_bytes // elem_size

        num_gpu_blocks = pool_typed.numel() // page_size_elems
        self.num_gpu_blocks = num_gpu_blocks
        num_super_blocks = num_gpu_blocks // self.pages_per_super_block
        self.num_super_blocks = num_super_blocks

        w13_per_expert_shape = cpu_w13.shape[1:]
        w13_per_expert_strides = cpu_w13[0].contiguous().stride()
        self.pool_w13_view = torch.as_strided(
            pool_typed,
            size=(num_super_blocks, *w13_per_expert_shape),
            stride=(super_block_stride_elems, *w13_per_expert_strides),
            storage_offset=w13_offset_elems,
        )
        w2_per_expert_shape = cpu_w2.shape[1:]
        w2_per_expert_strides = cpu_w2[0].contiguous().stride()
        self.pool_w2_view = torch.as_strided(
            pool_typed,
            size=(num_super_blocks, *w2_per_expert_shape),
            stride=(super_block_stride_elems, *w2_per_expert_strides),
            storage_offset=w2_offset_elems,
        )
        # The fused MoE kernel asserts stride(-1) == 1 on its weight
        # tensors, so check up front.
        assert self.pool_w13_view.stride(-1) == 1
        assert self.pool_w2_view.stride(-1) == 1

        # GPU-side expert -> super_block_id table for the topk_ids remap.
        # int64 matches the kernel's int64 cast on expert ids and avoids
        # overflow in stride * offset. The value is the view row, i.e.
        # the super-block id.
        self.super_block_id_at = torch.full(
            (num_experts,),
            self._UNLOADED,
            dtype=torch.int64,
            device=device,
        )

        self.expert_at_super_block: dict[int, int] = {}
        self.super_block_at_expert: dict[int, int] = {}
        self.expert_lru: OrderedDict[int, float] = OrderedDict()
        self.pinned_super_blocks: set[int] = set()
        # DIAGNOSTIC: every expert id the workload has ever routed to this
        # layer (the genuine footprint), vs what is merely resident.
        self.ever_activated: set[int] = set()
        self.recent_expert_sets: deque[set[int]] = deque()
        self.recent_expert_counts: Counter[int] = Counter()

        self.hits = 0
        self.misses = 0
        self.forward_count = 0

    def has_expert(self, expert_id: int) -> bool:
        return expert_id in self.super_block_at_expert

    def super_block_of_expert(self, expert_id: int) -> int:
        return self.super_block_at_expert[expert_id]

    def expert_of_super_block(self, super_block_id: int) -> int | None:
        return self.expert_at_super_block.get(super_block_id)

    def assign(self, super_block_id: int, expert_id: int, step: int) -> None:
        assert super_block_id not in self.expert_at_super_block, (
            f"L{self.layer_idx}: super-block {super_block_id} already mapped "
            f"to expert {self.expert_at_super_block[super_block_id]}"
        )
        assert expert_id not in self.super_block_at_expert, (
            f"L{self.layer_idx}: expert {expert_id} already mapped to "
            f"super-block {self.super_block_at_expert[expert_id]}"
        )
        self.expert_at_super_block[super_block_id] = expert_id
        self.super_block_at_expert[expert_id] = super_block_id
        self.expert_lru[expert_id] = step
        # Mirror onto the GPU lookup for the forward-path remap.
        self.super_block_id_at[expert_id] = super_block_id

    def drop(self, super_block_id: int) -> int | None:
        expert_id = self.expert_at_super_block.pop(super_block_id, None)
        if expert_id is None:
            return None
        del self.super_block_at_expert[expert_id]
        self.expert_lru.pop(expert_id, None)
        # Invalidate the GPU lookup so super_block_id_at[topk_ids] can't
        # return a stale value. ensure_loaded is responsible for making
        # sure no expert id in the next forward resolves to _UNLOADED.
        self.super_block_id_at[expert_id] = self._UNLOADED
        return expert_id

    def bump_expert(self, expert_id: int, step: int) -> None:
        """Mark expert as MRU and stamp it with the current step.

        Called for every expert touched this forward, hit or miss, so
        the eviction step can compare expert recency against prefix recency.
        """
        if expert_id in self.expert_lru:
            self.expert_lru[expert_id] = step
            self.expert_lru.move_to_end(expert_id, last=True)

    def record_expert_accesses(self, expert_ids: list[int]) -> None:
        if self.working_set_window <= 0:
            return
        expert_set = set(expert_ids)
        self.recent_expert_sets.append(expert_set)
        self.recent_expert_counts.update(expert_set)
        if len(self.recent_expert_sets) > self.working_set_window:
            self.recent_expert_counts.subtract(self.recent_expert_sets.popleft())
            self.recent_expert_counts += Counter()

    @property
    def working_set_ready(self) -> bool:
        return len(self.recent_expert_sets) == self.working_set_window

    @property
    def working_set_size(self) -> int:
        return len(self.recent_expert_counts)


class UnifiedPoolManager:
    """Owns the per-layer pools and the cross-layer bookkeeping.

    super_block_holder maps super_block_id -> the set of layers that
    currently hold an expert there. It's used both for membership
    lookups and to broadcast invalidations when KV reclaims a page in
    the super-block. Expert misses only touch the calling layer; KV
    allocations broadcast to every holder.

    prefix_lru is shared because attention touches the same page id at
    every layer in lockstep, so a single global prefix recency list
    (keyed by page/block_id) is enough.
    """

    def __init__(
        self,
        block_pool,
        device: torch.device,
        pages_per_super_block: int,
        page_size_bytes: int,
        kv_pool_buffers: dict[int, torch.Tensor],
    ) -> None:
        from vllm.v1.core.block_pool import BlockPool

        assert isinstance(block_pool, BlockPool)
        assert pages_per_super_block >= 1
        self.block_pool: BlockPool = block_pool
        self.device = device
        self.pages_per_super_block = pages_per_super_block
        assert page_size_bytes > 0
        self.page_size_bytes = page_size_bytes
        required_bytes = block_pool.num_gpu_blocks * page_size_bytes
        self.kv_pool_buffers: dict[int, torch.Tensor] = {}
        for layer_idx, buffer in kv_pool_buffers.items():
            assert buffer.numel() == required_bytes, (
                f"L{layer_idx}: KV pool buffer has {buffer.numel()} bytes, "
                f"expected {required_bytes}"
            )
            self.kv_pool_buffers[layer_idx] = buffer
        self.num_super_blocks = block_pool.num_gpu_blocks // pages_per_super_block
        self.layers: dict[int, UnifiedPool] = {}
        # super_block_id -> set of layers holding an expert there.
        self.super_block_holder: dict[int, set[int]] = {}
        self.transfer_stream = torch.cuda.Stream(device=device)
        self.step = 0  # incremented per forward; base recency timestamp

        # Phase 4 relocation. When on (default), vacating a KV super-block
        # for an expert preserves its warm cached-prefix pages by moving
        # them into holes instead of evicting them, so only the globally
        # coldest pages are killed. Set VLLM_UNIFIED_POOL_RELOCATE=0 to
        # fall back to evict-only (the M2 behaviour) for A/B comparison.
        self._relocation_enabled = (
            os.environ.get("VLLM_UNIFIED_POOL_RELOCATE", "1") != "0"
        )
        self.relocations = 0  # total KV pages relocated (stats)

        # page/block_id -> last-used step, oldest first. Updated by the
        # BlockPool prefix callbacks below.
        self.prefix_lru: OrderedDict[int, int] = OrderedDict()

        self.block_pool.register_on_allocation_callback(self._on_kv_allocation)
        self.block_pool.register_on_prefix_added_callback(self._on_prefix_added)
        self.block_pool.register_on_prefix_removed_callback(self._on_prefix_removed)
        # Override BlockPool's default popleft_n so KV allocations also
        # consult the unified LRUs, the same way expert misses do.
        self.block_pool.register_kv_victim_selector(self._select_kv_victim_blocks)

    def register_layer(self, layer: UnifiedPool) -> None:
        assert layer.layer_idx not in self.layers, (
            f"Layer {layer.layer_idx} already registered with the unified pool"
        )
        assert layer.pages_per_super_block == self.pages_per_super_block
        assert layer.num_super_blocks == self.num_super_blocks
        self.layers[layer.layer_idx] = layer

    def _adaptive_expert_target(self) -> int | None:
        if not self.layers or not all(
            layer.working_set_ready for layer in self.layers.values()
        ):
            return None
        total = sum(layer.working_set_size for layer in self.layers.values())
        return (total + len(self.layers) - 1) // len(self.layers)

    def _expert_footprint(self) -> int:
        return len(self.super_block_holder)

    # Super-block <-> page helpers.

    def _pages_of(self, super_block_id: int) -> range:
        base = super_block_id * self.pages_per_super_block
        return range(base, base + self.pages_per_super_block)

    def _super_block_of_page(self, block_id: int) -> int:
        return block_id // self.pages_per_super_block

    def _any_layer_pins_super_block(self, super_block_id: int) -> bool:
        for layer in self.layers.values():
            if super_block_id in layer.pinned_super_blocks:
                return True
        return False

    def _super_block_has_live_page(self, super_block_id: int) -> bool:
        """True if any page of the super-block is a live KV block (ref_cnt>0).

        A live page is one a running request currently references this
        step, so its super-block cannot be vacated for an expert.
        """
        for p in self._pages_of(super_block_id):
            if self.block_pool.blocks[p].ref_cnt > 0:
                return True
        return False

    # Mapping helpers.

    def _add_holder(self, layer_idx: int, super_block_id: int) -> None:
        self.super_block_holder.setdefault(super_block_id, set()).add(layer_idx)

    def _remove_holder(self, layer_idx: int, super_block_id: int) -> None:
        holders = self.super_block_holder.get(super_block_id)
        if holders is None:
            return
        holders.discard(layer_idx)
        if not holders:
            del self.super_block_holder[super_block_id]

    def _drop_layer_mapping(
        self,
        layer: UnifiedPool,
        super_block_id: int,
        cause: str,
        tier: str | None = None,
    ) -> None:
        """Drop only this layer's mapping for the super-block.

        Used on an expert miss where the layer reuses its own cold
        expert's super-block. Other layers keep their mappings.
        """
        expert_id = layer.expert_of_super_block(super_block_id)
        score = layer.expert_lru.get(expert_id) if expert_id is not None else None
        evicted = layer.drop(super_block_id)
        if evicted is None:
            return
        self._remove_holder(layer.layer_idx, super_block_id)
        if _trace_enabled():
            tier_str = f" tier={tier}" if tier else ""
            print(
                f"UNIFIED EVICT sb={super_block_id} L{layer.layer_idx} "
                f"kind=expert E{evicted} cause={cause}{tier_str} "
                f"score={score:.3f} step={self.step}",
                flush=True,
            )

    def _broadcast_drop_all_layers(self, super_block_id: int, cause: str) -> None:
        """Drop every layer's expert mapping at the super-block.

        Called when KV is reclaiming a page inside the super-block. The
        kernel reads pool_buffer directly, so dropping a mapping on a
        pinned super-block would silently corrupt a live read. Assert on
        it. async_scheduling=False makes this trivial in practice but
        it's worth catching if that ever changes.
        """
        holders = self.super_block_holder.pop(super_block_id, None)
        if not holders:
            return
        for layer_idx in list(holders):
            layer = self.layers.get(layer_idx)
            if layer is None:
                continue
            assert super_block_id not in layer.pinned_super_blocks, (
                f"KV-allocation broadcast tried to drop a pinned super-block: "
                f"sb={super_block_id} L{layer_idx} cause={cause}. The MoE "
                f"kernel may be reading those bytes — refusing to corrupt. "
                f"Check async_scheduling is disabled."
            )
            expert_id = layer.expert_of_super_block(super_block_id)
            score = layer.expert_lru.get(expert_id) if expert_id is not None else None
            evicted = layer.drop(super_block_id)
            if evicted is not None and _trace_enabled():
                print(
                    f"UNIFIED EVICT sb={super_block_id} L{layer_idx} "
                    f"kind=expert E{evicted} cause={cause} tier=kv-broadcast "
                    f"score={score:.3f} step={self.step}",
                    flush=True,
                )

    def _evict_prefix_globally(
        self, block_id: int, cause: str, tier: str | None = None
    ) -> None:
        """Clear the page's prefix hash everywhere.

        Once any layer overwrites the page's bytes the prefix is broken
        in every layer. Clearing the hash fires on_prefix_removed which
        drops it from prefix_lru. The page stays in the free queue as a
        plain free page (evict_prefix_hash does not dequeue it).
        """
        block = self.block_pool.blocks[block_id]
        if block.block_hash is None:
            return
        self.block_pool.evict_prefix_hash(block_id)
        if _trace_enabled():
            tier_str = f" tier={tier}" if tier else ""
            print(
                f"UNIFIED EVICT page={block_id} L=all "
                f"kind=kv-prefix cause={cause}{tier_str}",
                flush=True,
            )

    @PROFILER.timed("vacate_kv_super_block")
    def _vacate_kv_super_block(self, super_block_id: int, cause: str) -> None:
        """Clear all cached-prefix KV pages out of the super-block.

        The caller guarantees the super-block has no live (ref_cnt>0)
        page and is not expert-held. After this every page of the
        super-block is a plain free page, so an expert can be written
        across it.

        With relocation enabled, cached-prefix pages are preserved by
        moving them (warmest first) into holes — a free page, or a hole
        created by evicting a *colder* page elsewhere — so the pages that
        actually get killed are the globally coldest ones, not whatever
        happened to sit in this super-block. With relocation disabled the
        pages are simply evicted in place.
        """
        prefix_pages = [
            p
            for p in self._pages_of(super_block_id)
            if self.block_pool.blocks[p].block_hash is not None
        ]
        if not prefix_pages:
            return

        if not self._relocation_enabled:
            for p in prefix_pages:
                self._evict_prefix_globally(p, cause=cause, tier="kv-vacate")
            return

        # Relocate warmest-first: warm pages grab holes first, so if holes
        # run out the pages evicted in place are the coldest.
        prefix_pages.sort(key=lambda p: self.prefix_lru.get(p, -1), reverse=True)
        for p in prefix_pages:
            warmth = self.prefix_lru.get(p, -1)
            hole = self._first_free_page(exclude_super_block=super_block_id)
            if hole is None:
                # No free page: make one by evicting a page colder than p
                # from a different super-block.
                victim = self._coldest_prefix_page(
                    exclude_super_block=super_block_id, colder_than=warmth
                )
                if victim is not None:
                    self._evict_prefix_globally(victim, cause=cause, tier="make-hole")
                    hole = victim
            if hole is not None:
                self._relocate_kv_page(p, hole)
            else:
                # p is among the globally coldest — nothing colder to
                # displace — so evict it in place.
                self._evict_prefix_globally(p, cause=cause, tier="kv-vacate-evict")

    @PROFILER.timed("first_free_page")
    def _first_free_page(self, exclude_super_block: int) -> int | None:
        """A pure-free page (no hash, super-block not expert-held, not
        pinned) outside ``exclude_super_block``, usable as a relocation
        destination. Not removed from the free queue — after relocation it
        becomes a cached-prefix page that stays in the queue."""
        queue = self.block_pool.free_block_queue
        cursor = queue.fake_free_list_head.next_free_block
        while cursor is not None and cursor is not queue.fake_free_list_tail:
            nxt = cursor.next_free_block
            if cursor.block_hash is None and not cursor.is_null:
                s = self._super_block_of_page(cursor.block_id)
                if (
                    s != exclude_super_block
                    and not self.super_block_holder.get(s)
                    and not self._any_layer_pins_super_block(s)
                ):
                    return cursor.block_id
            cursor = nxt
        return None

    @PROFILER.timed("coldest_prefix_page")
    def _coldest_prefix_page(
        self, exclude_super_block: int, colder_than: int
    ) -> int | None:
        """The globally coldest cached-prefix page strictly colder than
        ``colder_than``, outside ``exclude_super_block`` and not pinned.
        Evicting it frees a hole for a warmer page being preserved.

        Scans every prefix_lru entry (rather than trusting queue/insertion
        order) because relocation can leave order and step value out of
        sync.
        """
        best_p: int | None = None
        best_step: int | None = None
        for p, step in self.prefix_lru.items():
            if step >= colder_than:
                continue
            s = self._super_block_of_page(p)
            if s == exclude_super_block:
                continue
            if self._any_layer_pins_super_block(s):
                continue
            if best_step is None or step < best_step:
                best_p = p
                best_step = step
        return best_p

    @PROFILER.timed("relocate_page")
    def _relocate_kv_page(self, src_id: int, dst_id: int) -> None:
        """Physically move a cached-prefix KV page from src to dst.

        Copies the page bytes in every layer's pool buffer (a KV block is
        global — the same page id holds that block's KV in every layer),
        then transfers the hash identity from src to dst without changing
        any block id, and moves the prefix recency entry. src must be a
        ref_cnt==0 cached prefix; dst must be a hash-free hole outside the
        target/pinned super-blocks. The byte copies run on transfer_stream
        and are ordered before the expert DMA on that same stream, so the
        single wait_stream barrier in ensure_loaded covers them.
        """
        self._copy_page_all_layers(src_id, dst_id)
        src_step = self.prefix_lru.get(src_id, self.step)
        moved = self.block_pool.relocate_prefix_hash(src_id, dst_id)
        if not moved:
            return
        # dst inherits src's recency (it holds the same, still-warm KV).
        self.prefix_lru.pop(src_id, None)
        self.prefix_lru.pop(dst_id, None)
        self.prefix_lru[dst_id] = src_step
        # Full re-sort of every prefix entry on every relocated page.
        with PROFILER.cpu("prefix_lru_resort"):
            self.prefix_lru = OrderedDict(
                sorted(self.prefix_lru.items(), key=lambda item: item[1])
            )
        self.relocations += 1
        if _trace_enabled():
            print(
                f"UNIFIED RELOCATE src={src_id} dst={dst_id} step={src_step}",
                flush=True,
            )

    # BlockPool callbacks.

    def _on_kv_allocation(self, block_ids: list[int]) -> None:
        """KV is about to overwrite these pages; drop every layer's
        expert mapping for the super-block each page falls in. The KV
        victim selector already drops experts it reclaims, so this is a
        defensive no-op for those; it still covers any other path that
        allocates a page inside a held super-block.
        """
        seen: set[int] = set()
        for block_id in block_ids:
            s = self._super_block_of_page(block_id)
            if s in seen:
                continue
            seen.add(s)
            self._broadcast_drop_all_layers(s, cause="kv-alloc")

    def _on_prefix_added(self, block_id: int) -> None:
        """A cached-prefix page has been freed: bump it to MRU.

        Returning to the free queue with a hash counts as a use, so
        prefix recency stamps it with the current step.
        """
        self.prefix_lru[block_id] = self.step
        self.prefix_lru.move_to_end(block_id, last=True)
        if _trace_enabled():
            print(
                f"UNIFIED PREFIX_ADDED p{block_id} step={self.step} "
                f"size={len(self.prefix_lru)}",
                flush=True,
            )

    def _on_prefix_removed(self, block_id: int) -> None:
        """Page is no longer an evictable cached prefix; drop from prefix_lru."""
        removed = self.prefix_lru.pop(block_id, None)
        if _trace_enabled():
            was_present = "yes" if removed is not None else "no"
            print(
                f"UNIFIED PREFIX_REMOVED p{block_id} "
                f"was_present={was_present} size={len(self.prefix_lru)}",
                flush=True,
            )

    # Stage 2 warm-up.

    def warm_up(self, warm_count: int) -> None:
        """Pre-load warm_count experts per layer at startup.

        Each warmed expert e is placed at super-block s = e + 1 (super-
        block 0 is reserved because it contains BlockPool's null page 0),
        shared across every layer (different physical bytes per layer
        because the pool buffer is per-layer). Warming warm_count experts
        therefore consumes warm_count super-blocks == warm_count * F
        pages in total.

        The shared-id approach means a KV reclaim of a warmed super-block
        invalidates that expert in every layer at once.
        """
        if warm_count <= 0:
            return
        for layer in self.layers.values():
            assert warm_count <= layer.num_experts, (
                f"warm_count ({warm_count}) > num_experts "
                f"({layer.num_experts}) for L{layer.layer_idx}"
            )
        assert warm_count <= self.num_super_blocks - 1, (
            f"warm_count ({warm_count}) > usable super-blocks "
            f"({self.num_super_blocks - 1}; super-block 0 reserved)"
        )
        layers_list = list(self.layers.values())
        if not layers_list:
            return
        for expert_id in range(warm_count):
            super_block_id = expert_id + 1  # reserve super-block 0
            # At startup every candidate page is free (no hash). Sanity-
            # check that assumption so a misconfiguration fails loudly.
            for p in self._pages_of(super_block_id):
                assert self.block_pool.blocks[p].block_hash is None, (
                    f"warm-up: page {p} of super-block {super_block_id} "
                    f"already has a prefix hash"
                )
            for layer in layers_list:
                layer.assign(super_block_id, expert_id, step=self.step)
                self._add_holder(layer.layer_idx, super_block_id)
                self._dma_expert_into_super_block_sync(layer, expert_id, super_block_id)
        # Warm-up DMAs run on transfer_stream. Wait for them and then
        # device-sync. If the first forward is all hits, ensure_loaded
        # won't call wait_stream itself, so any unflushed warm-up would
        # surface as stale reads.
        torch.cuda.current_stream(self.device).wait_stream(self.transfer_stream)
        torch.cuda.synchronize(self.device)
        for layer in layers_list:
            logger.info(
                "UnifiedPool L%d: warmed %d/%d experts (F=%d pages each)",
                layer.layer_idx,
                warm_count,
                layer.num_experts,
                self.pages_per_super_block,
            )

        # Post-warm-up sanity check. Each warmed (expert, super-block)
        # pair should round-trip: pool_w13_view[super_block_id] equals
        # cpu_w13[expert_id], same for w2. A failure here means the
        # F-strided super-block view is misaligned or the DMA didn't land.
        for layer in layers_list:
            for expert_id, s in layer.super_block_at_expert.items():
                w13_view_row = layer.pool_w13_view[s]
                w2_view_row = layer.pool_w2_view[s]
                w13_truth = layer.cpu_w13[expert_id].to(layer.device)
                w2_truth = layer.cpu_w2[expert_id].to(layer.device)
                if not torch.equal(w13_view_row, w13_truth):
                    raise RuntimeError(
                        f"UnifiedPool L{layer.layer_idx}: pool_w13_view"
                        f"[{s}] != cpu_w13[{expert_id}] after warm-up. "
                        f"Super-block view layout is wrong, or DMA didn't land."
                    )
                if not torch.equal(w2_view_row, w2_truth):
                    raise RuntimeError(
                        f"UnifiedPool L{layer.layer_idx}: pool_w2_view"
                        f"[{s}] != cpu_w2[{expert_id}] after warm-up."
                    )
        logger.info(
            "UnifiedPool warm-up sanity check passed: %d (expert, super-block) "
            "pairs verified across %d layers.",
            warm_count * len(layers_list),
            len(layers_list),
        )

    # Forward-path API.

    @PROFILER.timed("ensure_loaded")
    def ensure_loaded(self, layer: UnifiedPool, needed_expert_ids: list[int]) -> None:
        """Make sure every needed expert is loaded at layer.

        Hits and miss-claimed super-blocks are pinned for the rest of
        this forward (released by release_pinned). DMAs end with a
        wait_stream barrier on the compute stream. Every needed expert
        is bumped to MRU regardless of hit/miss, using the layer's fixed
        timestamp bias. Trace snapshots are captured before any mutation.
        """
        hit_results: list[tuple[int, int]] = []  # (eid, super_block_id)
        miss_eids: list[int] = []
        needed_set = set(needed_expert_ids)
        layer.ever_activated.update(needed_set)  # DIAGNOSTIC: genuine footprint
        layer.record_expert_accesses(needed_expert_ids)
        if _trace_enabled():
            print(
                f"UNIFIED NEEDED L{layer.layer_idx} experts={sorted(needed_set)}",
                flush=True,
            )
        for eid in needed_expert_ids:
            if layer.has_expert(eid):
                hit_results.append((eid, layer.super_block_of_expert(eid)))
            else:
                miss_eids.append(eid)

        # Trace before any state changes.
        if _trace_enabled():
            self._trace_pre_mutation(layer, needed_expert_ids)

        # Counters, pinning hit super-blocks, and MRU bumps for hits.
        layer.hits += len(hit_results)
        layer.misses += len(miss_eids)
        for eid, super_block_id in hit_results:
            layer.pinned_super_blocks.add(super_block_id)
            layer.bump_expert(eid, self.step)

        # Claim a super-block per miss and DMA the expert into it right
        # away, so the pool never carries a mapping whose GPU bytes are not
        # yet loaded (matters if a later miss in this loop raises pool
        # exhaustion — earlier misses stay fully consistent). Vacating a
        # super-block may enqueue relocation copies on transfer_stream;
        # they precede this miss's expert DMA on the same stream, so a
        # single wait_stream barrier at the end covers everything.
        miss_assignments: list[tuple[int, int, str]] = []  # (eid, sb, tier)
        try:
            for eid in miss_eids:
                super_block_id, tier = self._select_super_block_for_expert(
                    layer, eid, needed_set
                )
                # assign() stamps the expert with the current step as MRU.
                layer.assign(super_block_id, eid, step=self.step)
                self._add_holder(layer.layer_idx, super_block_id)
                layer.pinned_super_blocks.add(super_block_id)
                miss_assignments.append((eid, super_block_id, tier))
                PROFILER.count(f"miss_tier_{tier}")
                with torch.cuda.stream(self.transfer_stream):
                    self._dma_expert_into_super_block_async(layer, eid, super_block_id)

            # Barrier the compute stream on the transfer stream once all miss
            # relocations + DMAs have been enqueued.
            if miss_eids:
                torch.cuda.current_stream(self.device).wait_stream(self.transfer_stream)
        except Exception:
            torch.cuda.current_stream(self.device).wait_stream(self.transfer_stream)
            for _eid, super_block_id, _tier in miss_assignments:
                self._drop_layer_mapping(
                    layer, super_block_id, cause="expert-load-failed"
                )
            layer.pinned_super_blocks.clear()
            raise

        if _TRACE_VERBOSE:
            hit_parts = [f"E{eid}@sb{sb}" for eid, sb in hit_results]
            miss_parts = [
                f"E{eid}->sb{sb}({tier})" for eid, sb, tier in miss_assignments
            ]
            print(
                f"UNIFIED RESULT L{layer.layer_idx} "
                f"hits=[{','.join(hit_parts)}] "
                f"misses=[{','.join(miss_parts)}]",
                flush=True,
            )
            print(f"--- end L{layer.layer_idx} ---", flush=True)

    def release_pinned(self, layer: UnifiedPool, completed: bool = True) -> None:
        layer.pinned_super_blocks.clear()
        if completed:
            layer.forward_count += 1

    def end_forward_step(self) -> None:
        self.step += 1
        # Opportunistic: resolves only event pairs the GPU has already
        # finished, so it never inserts a stall.
        PROFILER.drain()
        PROFILER.maybe_periodic_dump(self.step)

    # Per-layer expert-miss super-block selection.

    @PROFILER.timed("select_super_block")
    def _select_super_block_for_expert(
        self, layer: UnifiedPool, eid: int, needed_set: set[int]
    ) -> tuple[int, str]:
        """Choose a super-block to hold expert ``eid`` for ``layer``.

        Returns (super_block_id, tier). On return the super-block is
        guaranteed to hold no KV and to have this layer's slot free, so
        the caller can assign + DMA straight away.

        Tier 1 (no eviction): co-locate onto a super-block already held
        by other layers (mutual exclusion => no KV in it), else a fully
        free super-block. Tier 2 compares this layer's coldest evictable
        expert against the coldest vacatable KV super-block and takes the
        colder (mirrors the Phase-3 dual LRU).
        """
        ns = self.num_super_blocks
        blocks = self.block_pool.blocks
        F = self.pages_per_super_block

        free_pure_s: int | None = None
        cross_layer_s: int | None = None
        for s in range(1, ns):  # reserve super-block 0
            if s in layer.pinned_super_blocks:
                continue
            if s in layer.expert_at_super_block:
                continue
            holders = self.super_block_holder.get(s)
            if holders:
                # Expert-held by other layer(s); mutual exclusion means
                # no KV lives here, so this layer can co-locate its
                # expert with zero eviction.
                if cross_layer_s is None:
                    cross_layer_s = s
                continue
            # Holder empty: pages are free / cached-prefix / live.
            base = s * F
            has_prefix = False
            has_live = False
            for p in range(base, base + F):
                b = blocks[p]
                if b.ref_cnt > 0:
                    has_live = True
                    break
                if b.block_hash is not None:
                    has_prefix = True
            if has_live:
                continue
            if not has_prefix and free_pure_s is None:
                free_pure_s = s

        # Prefer packing experts onto an already-expert super-block; it
        # keeps whole super-blocks free for KV (less fragmentation).
        if cross_layer_s is not None:
            return cross_layer_s, "free-cross-layer-expert"
        if free_pure_s is not None:
            return free_pure_s, "free-pure"

        return self._evict_for_expert(layer, eid, needed_set)

    @PROFILER.timed("evict_for_expert")
    def _evict_for_expert(
        self, layer: UnifiedPool, eid: int, needed_set: set[int]
    ) -> tuple[int, str]:
        """Tier 2: evict to make room for an expert miss."""
        # Option A: this layer's coldest evictable expert (not needed
        # this forward, not pinned). Reusing its super-block only drops
        # this layer's mapping; other layers keep theirs.
        own_expert_eid: int | None = None
        own_expert_step: float | None = None
        for e2, st in layer.expert_lru.items():
            if e2 in needed_set:
                continue
            s2 = layer.super_block_at_expert.get(e2)
            if s2 is None or s2 in layer.pinned_super_blocks:
                continue
            own_expert_eid = e2
            own_expert_step = st
            break

        # Option B: the cheapest vacatable KV super-block, scored by its
        # oldest page so the comparison below is oldest-expert vs
        # oldest-KV.
        best_kv_s, best_kv_step = self._cheapest_kv_super_block()

        cause = f"expert-L{layer.layer_idx}"
        target = self._adaptive_expert_target()
        footprint_at_target = target is not None and self._expert_footprint() >= target
        if own_expert_eid is not None and footprint_at_target:
            s2 = layer.super_block_at_expert[own_expert_eid]
            self._drop_layer_mapping(layer, s2, cause=cause, tier="expert-local")
            return s2, "expert-local"
        # Choose the colder option. Prefix wins ties (matches Phase 3).
        if own_expert_eid is not None and best_kv_s is not None:
            assert own_expert_step is not None
            assert best_kv_step is not None
            choose_kv = best_kv_step <= own_expert_step
            if _trace_enabled():
                print(
                    f"UNIFIED DECISION side=expert-miss step={self.step} "
                    f"layer={layer.layer_idx} expert-score={own_expert_step:.3f} "
                    f"kv-score={best_kv_step:.3f} "
                    f"chosen={'kv' if choose_kv else 'expert'}",
                    flush=True,
                )
            if choose_kv:
                self._vacate_kv_super_block(best_kv_s, cause=cause)
                return best_kv_s, "kv-vacate"
            s2 = layer.super_block_at_expert[own_expert_eid]
            self._drop_layer_mapping(layer, s2, cause=cause, tier="expert-local")
            return s2, "expert-local"
        if own_expert_eid is not None:
            s2 = layer.super_block_at_expert[own_expert_eid]
            self._drop_layer_mapping(layer, s2, cause=cause, tier="expert-local")
            return s2, "expert-local"
        if best_kv_s is not None:
            self._vacate_kv_super_block(best_kv_s, cause=cause)
            return best_kv_s, "kv-vacate"

        raise RuntimeError(
            f"UnifiedPool L{layer.layer_idx}: pool exhausted while resolving "
            "expert miss. No free super-block, no evictable expert, and no "
            "vacatable KV super-block (every candidate has a live or pinned "
            "page). Reduce --max-num-batched-tokens, lower "
            "--expert-cache-size, or increase --num-gpu-blocks-override."
        )

    @PROFILER.timed("cheapest_kv_super_block")
    def _cheapest_kv_super_block(self) -> tuple[int | None, int | None]:
        """The cheapest super-block to vacate for an expert, plus the KV
        recency score to weigh against expert recency.

        Candidates are ranked by how many *warm* pages they hold, fewest
        first -- equivalently, by most cold pages, which is the rule the
        paper states. Vacating a super-block puts its pages at risk: they
        are relocated if a hole or a colder page exists to displace, and
        dropped otherwise. Choosing the coldest-filled super-block
        therefore minimises the warm KV exposed to that risk, and stays
        meaningful whether or not the pool is full:

        * not full: few pages at all, so few warm ones;
        * full but mostly cold: 96 pages of which 80 are cold, 16 warm;
        * full and warm: 90 warm, correctly avoided.

        Note this ranks by *risk to warm KV*, not by number of copies: a
        mostly-cold super-block can still cost more relocations than a
        near-empty one, because cold pages are preserved too when holes are
        available. At 0.116 ms per page copy that is the right trade.

        Cold means "among the globally coldest ``pages_per_super_block``
        prefix pages", mirroring ``_vacate_kv_super_block``, which drops a
        page only when nothing colder exists elsewhere -- so the pages it
        drops are the globally coldest. At most F pages are ever cleared,
        hence that frontier and no tunable constant.

        Everything comes out of one pass over ``prefix_lru`` -- the pages
        that exist -- rather than the previous sweep, which computed the
        exact relocation plan for every super-block and rescanned the whole
        block array each time: O(num_super_blocks * num_blocks), measured
        at 165 ms per miss at 95% KV occupancy. See docs/MECHANISM_COST.md.

        The returned score is the chosen super-block's *oldest* page, so
        the caller compares oldest-expert against oldest-KV, matching the
        mixed-LRU policy as described. The validity checks (not
        expert-held, unpinned, no live page) are applied only to
        candidates as they are tried, not to all of them up front.
        """
        if not self.prefix_lru:
            return None, None

        # Cold frontier: the step of the F-th coldest prefix page. Pages at
        # or below it are droppable, so they cost nothing to clear.
        F = self.pages_per_super_block
        steps = self.prefix_lru.values()
        if len(self.prefix_lru) <= F:
            cold_frontier = max(steps)
        else:
            cold_frontier = heapq.nsmallest(F, steps)[-1]

        warm: Counter[int] = Counter()
        oldest: dict[int, int] = {}
        for p, step in self.prefix_lru.items():
            s = self._super_block_of_page(p)
            if step > cold_frontier:
                warm[s] += 1
            elif s not in warm:
                warm[s] = 0  # candidate with only cold pages: free to clear
            prev = oldest.get(s)
            if prev is None or step < prev:
                oldest[s] = step
        if not warm:
            return None, None

        for s, _n in sorted(warm.items(), key=lambda kv: (kv[1], oldest[kv[0]])):
            if s == 0:  # reserved: holds BlockPool's null page
                continue
            if self.super_block_holder.get(s):
                continue
            if self._any_layer_pins_super_block(s):
                continue
            if self._super_block_has_live_page(s):
                continue
            return s, oldest[s]
        return None, None

    # KV-side victim selection (no layer of origin).

    @PROFILER.timed("oldest_global_expert")
    def _oldest_global_expert(self) -> tuple[int | None, float | None]:
        """Super-block containing the globally coldest biased expert.

        Returns (super_block_id, step) or (None, None) if every expert
        super-block is pinned by some layer. The existing shared-fate policy
        scores a super-block by its coldest holder so one stale co-location
        cannot starve global KV. Per-layer timestamp bias determines which
        layers become cold first without changing that allocation behavior.
        """
        best_step: float | None = None
        best_s: int | None = None
        for s, holders in self.super_block_holder.items():
            if self._any_layer_pins_super_block(s):
                continue
            holder_steps: list[float] = []
            for layer_idx in holders:
                layer = self.layers[layer_idx]
                eid = layer.expert_at_super_block.get(s)
                if eid is not None:
                    holder_steps.append(layer.expert_lru[eid])
            if not holder_steps:
                continue
            step = min(holder_steps)
            if best_step is None or step < best_step:
                best_step = step
                best_s = s
        return best_s, best_step

    @PROFILER.timed("select_kv_victim")
    def _select_kv_victim_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """Pick num_blocks victim pages for KV allocation.

        Mirrors the expert-miss selector but for the KV side and at page
        granularity. Tier 1 takes pages that are truly free (no hash,
        super-block not expert-held, not pinned). Tier 2 compares the
        oldest expert super-block across every layer with the oldest
        prefix page and takes the colder; evicting an expert super-block
        frees all F of its pages (one is returned, the rest stay in the
        free queue). Returned pages are already off the free queue;
        BlockPool.get_new_blocks does the rest of the bookkeeping.
        """
        if num_blocks == 0:
            return []
        queue = self.block_pool.free_block_queue
        ret: list[KVCacheBlock] = []
        for _ in range(num_blocks):
            ret.append(self._pick_one_kv_victim())
        assert queue.num_free_blocks >= 0
        return ret

    @PROFILER.timed("pick_kv_victim")
    def _pick_one_kv_victim(self) -> KVCacheBlock:
        """Pick one KV victim page and remove it from the free queue."""
        queue = self.block_pool.free_block_queue
        F = self.pages_per_super_block

        # Tier 1: truly free — no hash, super-block not expert-held, not
        # pinned.
        cursor = queue.fake_free_list_head.next_free_block
        while cursor is not None and cursor is not queue.fake_free_list_tail:
            nxt = cursor.next_free_block
            block_id = cursor.block_id
            if cursor.block_hash is not None:
                cursor = nxt
                continue
            s = self._super_block_of_page(block_id)
            if self.super_block_holder.get(s):
                cursor = nxt
                continue
            if self._any_layer_pins_super_block(s):
                cursor = nxt
                continue
            queue.remove(cursor)
            PROFILER.count("kv_claim_truly-free")
            if _trace_enabled():
                print(
                    f"UNIFIED KV_CLAIM page={block_id} tier=truly-free",
                    flush=True,
                )
            return cursor

        # Tier 2: pick whichever LRU has the colder head.
        oldest_expert_s, oldest_expert_step = self._oldest_global_expert()

        oldest_prefix_bid: int | None = None
        oldest_prefix_step: int | None = None
        for block_id, step in self.prefix_lru.items():
            s = self._super_block_of_page(block_id)
            if self._any_layer_pins_super_block(s):
                continue
            if oldest_prefix_step is None or step < oldest_prefix_step:
                oldest_prefix_bid = block_id
                oldest_prefix_step = step

        target = self._adaptive_expert_target()
        if (
            oldest_expert_s is not None
            and target is not None
            and self._expert_footprint() > target
        ):
            return self._kv_take_page_evicting_expert(oldest_expert_s)

        # Expert wins ties on the KV side (matches Phase 3's bias).
        if oldest_expert_s is not None and oldest_prefix_bid is not None:
            assert oldest_expert_step is not None
            assert oldest_prefix_step is not None
            choose_expert = oldest_expert_step <= oldest_prefix_step
            if _trace_enabled():
                print(
                    f"UNIFIED DECISION side=kv-alloc step={self.step} layer=all "
                    f"expert-score={oldest_expert_step:.3f} "
                    f"kv-score={oldest_prefix_step:.3f} "
                    f"chosen={'expert' if choose_expert else 'kv'}",
                    flush=True,
                )
            if choose_expert:
                return self._kv_take_page_evicting_expert(oldest_expert_s)
            return self._kv_take_prefix_page(oldest_prefix_bid)
        if oldest_expert_s is not None:
            return self._kv_take_page_evicting_expert(oldest_expert_s)
        if oldest_prefix_bid is not None:
            return self._kv_take_prefix_page(oldest_prefix_bid)

        raise RuntimeError(
            "UnifiedPool: pool exhausted resolving KV allocation. No truly-"
            "free page, no evictable expert super-block, no evictable prefix "
            "page. Reduce --max-num-batched-tokens or increase "
            "--num-gpu-blocks-override."
        )

    def _kv_take_prefix_page(self, block_id: int) -> KVCacheBlock:
        """Take a cached-prefix page for KV allocation: remove it from the
        free queue and drop it from prefix_lru immediately.

        The prefix_lru pop matters when get_new_blocks asks for more than
        one block at once: without it a second _pick_one_kv_victim in the
        same call could re-select this page (its hash is only cleared
        later, inside get_new_blocks) and double-remove it from the queue.
        The hash itself is cleared by get_new_blocks -> _maybe_evict, whose
        on_prefix_removed callback then pops prefix_lru again (a no-op)."""
        block = self.block_pool.blocks[block_id]
        self.block_pool.free_block_queue.remove(block)
        self.prefix_lru.pop(block_id, None)
        PROFILER.count("kv_claim_kv-evicts-prefix")
        if _trace_enabled():
            print(
                f"UNIFIED KV_CLAIM page={block_id} tier=kv-evicts-prefix",
                flush=True,
            )
        return block

    def _kv_take_page_evicting_expert(self, super_block_id: int) -> KVCacheBlock:
        """Evict the expert at super-block s (all layers) and return one of
        its pages, removed from the free queue. The other F-1 pages stay in
        the queue as free pages."""
        PROFILER.count("kv_claim_kv-evicts-expert")
        self._broadcast_drop_all_layers(super_block_id, cause="kv-alloc-evict-expert")
        queue = self.block_pool.free_block_queue
        for p in self._pages_of(super_block_id):
            block = self.block_pool.blocks[p]
            if block.is_null:
                continue
            queue.remove(block)
            if _trace_enabled():
                print(
                    f"UNIFIED KV_CLAIM page={p} sb={super_block_id} "
                    f"tier=kv-evicts-expert",
                    flush=True,
                )
            return block
        raise RuntimeError(
            f"UnifiedPool: super-block {super_block_id} had no usable page "
            f"after evicting its expert."
        )

    # DMA helpers.

    def _copy_page_all_layers(self, src_id: int, dst_id: int) -> None:
        """Device-to-device copy of one page's bytes from src to dst in
        every layer's pool buffer (a KV block is global). Runs on
        transfer_stream, ordered before the expert DMA on the same stream
        so ensure_loaded's single wait_stream barrier covers it."""
        page = self.page_size_bytes
        self.transfer_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.transfer_stream):
            ev = PROFILER.gpu_start()
            for buffer in self.kv_pool_buffers.values():
                dst_bytes = buffer.narrow(0, dst_id * page, page)
                src_bytes = buffer.narrow(0, src_id * page, page)
                dst_bytes.copy_(src_bytes, non_blocking=True)
            # One page copied in every layer: this is the per-relocation
            # cost the contiguity requirement forces us to pay.
            PROFILER.gpu_end(ev, "reloc_d2d_page", page * len(self.kv_pool_buffers))
            PROFILER.count("reloc_d2d_copies", len(self.kv_pool_buffers))

    def _dma_expert_into_super_block_async(
        self, layer: UnifiedPool, expert_id: int, super_block_id: int
    ) -> None:
        """Async CPU->GPU copy of expert weights into the super-block.

        The super-block is F contiguous pages == expert_slot_bytes of
        contiguous pool memory starting at super_block_id *
        expert_slot_bytes. w13 goes first, then w2, matching the strided
        view's inner layout.
        """
        sb_offset = super_block_id * layer.expert_slot_bytes
        w13_dst = layer.pool_buffer.narrow(0, sb_offset, layer.w13_bytes)
        w2_dst = layer.pool_buffer.narrow(
            0, sb_offset + layer.w13_bytes, layer.w2_bytes
        )
        ev = PROFILER.gpu_start()
        w13_dst.copy_(layer._cpu_w13_bytes[expert_id], non_blocking=True)
        w2_dst.copy_(layer._cpu_w2_bytes[expert_id], non_blocking=True)
        PROFILER.gpu_end(ev, "expert_dma_h2d", layer.expert_slot_bytes)

    def _dma_expert_into_super_block_sync(
        self, layer: UnifiedPool, expert_id: int, super_block_id: int
    ) -> None:
        with torch.cuda.stream(self.transfer_stream):
            self._dma_expert_into_super_block_async(layer, expert_id, super_block_id)

    # Trace helpers (active when VLLM_UNIFIED_POOL_TRACE>=1).

    def _trace_pre_mutation(
        self, layer: UnifiedPool, needed_expert_ids: list[int]
    ) -> None:
        """Dump pool composition (level >=1) and, at verbose level, the
        step header, both LRUs and the router's request. Runs before any
        mutation in ensure_loaded.

        UNIFIED CACHE includes step=self.step so the line is
        self-contained for downstream parsers. Expert occupancy is in
        super-block units; KV occupancy is in page units.
        """
        capacity_sb = self.num_super_blocks
        n_expert_ours = len(layer.expert_at_super_block)
        n_expert_other = sum(
            1
            for s, holders in self.super_block_holder.items()
            if layer.layer_idx not in holders
        )
        n_prefix_pages = len(self.prefix_lru)
        n_alloc_kv_pages = sum(
            1
            for b in self.block_pool.blocks
            if b.block_hash is not None and b.block_id not in self.prefix_lru
        )
        n_pinned_sb = len(layer.pinned_super_blocks)
        expert_target = self._adaptive_expert_target()
        target_str = "pending" if expert_target is None else str(expert_target)

        # Required at level 1 for the dissertation overlay figure.
        print(
            f"UNIFIED CACHE L{layer.layer_idx} step={self.step} "
            f"F={self.pages_per_super_block} "
            f"expert_sb {n_expert_ours}/{capacity_sb} ours "
            f"(expert-ours-sb={n_expert_ours}, expert-other-sb={n_expert_other}, "
            f"prefix-pages={n_prefix_pages}, alloc-kv-pages={n_alloc_kv_pages}, "
            f"pinned-sb={n_pinned_sb}, working-set={layer.working_set_size}, "
            f"expert-target={target_str}, "
            f"hits={layer.hits}, misses={layer.misses}, "
            f"ever-activated={len(layer.ever_activated)})",
            flush=True,
        )

        if not _TRACE_VERBOSE:
            return

        # Verbose-only per-step diagnostics.
        print(
            f"=== STEP {self.step} L{layer.layer_idx} need={needed_expert_ids} ===",
            flush=True,
        )

        # OrderedDict has MRU at the end (move_to_end last=True); flip so
        # the printed order reads MRU first.
        expert_lru_str = ", ".join(
            f"E{eid}@sb{layer.super_block_at_expert.get(eid, '?')}#step{step}"
            for eid, step in reversed(layer.expert_lru.items())
        )
        print(
            f"UNIFIED EXPERT_LRU L{layer.layer_idx} "
            f"MRU→LRU [{len(layer.expert_lru)}]: {expert_lru_str}",
            flush=True,
        )

        prefix_items = list(reversed(self.prefix_lru.items()))[:8]
        prefix_lru_str = ", ".join(f"p{bid}#step{step}" for bid, step in prefix_items)
        print(
            f"UNIFIED PREFIX_LRU MRU→LRU "
            f"[top 8 of {len(self.prefix_lru)}]: {prefix_lru_str}",
            flush=True,
        )

        print(
            f"UNIFIED REQUEST L{layer.layer_idx}: "
            + ",".join(f"E{e}" for e in needed_expert_ids),
            flush=True,
        )

    # Stats / introspection.

    def log_stats(self) -> None:
        num_kv_prefix = len(self.prefix_lru)
        logger.info(
            "UnifiedPool: relocation=%s, total KV pages relocated=%d",
            "on" if self._relocation_enabled else "off",
            self.relocations,
        )
        for layer in self.layers.values():
            total = layer.hits + layer.misses
            hit_rate = layer.hits / total * 100 if total > 0 else 0.0
            num_expert_super_blocks = len(layer.expert_at_super_block)
            logger.info(
                "UnifiedPool L%d: hits=%d misses=%d hit_rate=%.1f%% "
                "expert_super_blocks=%d kv_prefix_pages=%d working_set=%d",
                layer.layer_idx,
                layer.hits,
                layer.misses,
                hit_rate,
                num_expert_super_blocks,
                num_kv_prefix,
                layer.working_set_size,
            )

    def shutdown_log(self) -> None:
        logger.info("UnifiedPool shutdown stats:")
        self.log_stats()
        PROFILER.count("total_steps", self.step)
        PROFILER.count("total_relocations", self.relocations)
        for layer in self.layers.values():
            PROFILER.count("total_expert_hits", layer.hits)
            PROFILER.count("total_expert_misses", layer.misses)
        PROFILER.dump()
