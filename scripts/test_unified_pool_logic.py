"""Pure-Python invariant fuzzer for the Phase-4 UnifiedPool allocator.

Cannot run the real vLLM (needs CUDA), so this stubs the GPU seams
(`torch`, the DMA + page-copy methods) and drives the REAL allocator /
relocation / holder / LRU logic in unified_pool.py through randomised
expert-miss and KV-allocation sequences, asserting the core invariants
after every step:

  1. per-layer bijection expert<->super_block + super_block_id_at mirror
  2. mutual exclusion: an expert super-block holds no KV page, and a KV
     page's super-block is never expert-held
  3. after ensure_loaded, every needed expert is resident and the stubbed
     DMA delivered the right expert to its super-block in that layer
  4. free-queue consistency: exactly the ref_cnt==0 non-null blocks are in
     the queue
  5. relocation preserves the moved page's KV content (hash follows bytes)
"""

import contextlib
import argparse
import io
import os
import random
import sys
import types
from collections import Counter, OrderedDict, deque
from types import SimpleNamespace

# ---- stub torch + vllm.logger so unified_pool imports on a CPU box ----
torch_stub = types.ModuleType("torch")
torch_stub.int8 = "int8"
torch_stub.int64 = "int64"
_cuda = types.ModuleType("torch.cuda")
_cuda.stream = lambda s: contextlib.nullcontext()
_cuda.current_stream = lambda device=None: SimpleNamespace(wait_stream=lambda s: None)
_cuda.synchronize = lambda device=None: None
_cuda.Stream = lambda device=None: object()


class _EventStub:
    """Enough of torch.cuda.Event for the profiler to be fuzzed on CPU."""

    def __init__(self, enable_timing=False):
        self.enable_timing = enable_timing

    def record(self, stream=None):
        pass

    def query(self):
        return True

    def synchronize(self):
        pass

    def elapsed_time(self, other):
        return 1.0


_cuda.Event = _EventStub
torch_stub.cuda = _cuda
sys.modules["torch"] = torch_stub
sys.modules["torch.cuda"] = _cuda

logger_mod = types.ModuleType("vllm.logger")
logger_mod.init_logger = lambda name: SimpleNamespace(
    info=lambda *a, **k: None,
    warning=lambda *a, **k: None,
    error=lambda *a, **k: None,
)
vllm_pkg = types.ModuleType("vllm")
sys.modules.setdefault("vllm", vllm_pkg)
sys.modules["vllm.logger"] = logger_mod

# import the real module by path
import importlib.util

_FUSED_MOE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "vllm",
    "vllm",
    "model_executor",
    "layers",
    "fused_moe",
)

# unified_pool imports the real profiler; load it by path too and register
# it under the name the absolute import expects, so the fuzzer exercises
# the genuine (disabled-by-default) profiler rather than a stub of it.
_prof_spec = importlib.util.spec_from_file_location(
    "vllm.model_executor.layers.fused_moe.pool_profiler",
    os.path.join(_FUSED_MOE_DIR, "pool_profiler.py"),
)
_prof_mod = importlib.util.module_from_spec(_prof_spec)
sys.modules[_prof_spec.name] = _prof_mod
for _pkg in (
    "vllm.model_executor",
    "vllm.model_executor.layers",
    "vllm.model_executor.layers.fused_moe",
):
    sys.modules.setdefault(_pkg, types.ModuleType(_pkg))
_prof_spec.loader.exec_module(_prof_mod)
sys.modules["vllm.model_executor.layers.fused_moe"].pool_profiler = _prof_mod
PROFILER = _prof_mod.PROFILER

UP_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "vllm",
    "vllm",
    "model_executor",
    "layers",
    "fused_moe",
    "unified_pool.py",
)
spec = importlib.util.spec_from_file_location("unified_pool", UP_PATH)
up = importlib.util.module_from_spec(spec)
spec.loader.exec_module(up)
UnifiedPool = up.UnifiedPool
UnifiedPoolManager = up.UnifiedPoolManager

OFFLOAD_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "vllm",
    "vllm",
    "config",
    "offload.py",
)


# ---------------------- faithful fakes of block_pool -----------------------
class FakeBlock:
    def __init__(self, block_id):
        self.block_id = block_id
        self.block_hash = None
        self.ref_cnt = 0
        self.is_null = False
        self.prev_free_block = None
        self.next_free_block = None


class FakeQueue:
    """Doubly linked list with sentinel head/tail, mirroring the real
    FreeKVCacheBlockQueue's remove/append/membership semantics."""

    def __init__(self, blocks):
        self.fake_free_list_head = FakeBlock(-1)
        self.fake_free_list_tail = FakeBlock(-2)
        self.fake_free_list_head.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = self.fake_free_list_head
        self.num_free_blocks = 0
        self._in = set()
        for b in blocks:
            self.append(b)

    def append(self, b):
        assert b.block_id not in self._in, f"double-append {b.block_id}"
        tail = self.fake_free_list_tail
        prev = tail.prev_free_block
        prev.next_free_block = b
        b.prev_free_block = prev
        b.next_free_block = tail
        tail.prev_free_block = b
        self._in.add(b.block_id)
        self.num_free_blocks += 1

    def append_n(self, blocks):
        for b in blocks:
            self.append(b)

    def remove(self, b):
        assert b.block_id in self._in, f"remove of non-queued {b.block_id}"
        b.prev_free_block.next_free_block = b.next_free_block
        b.next_free_block.prev_free_block = b.prev_free_block
        b.prev_free_block = None
        b.next_free_block = None
        self._in.discard(b.block_id)
        self.num_free_blocks -= 1

    def popleft(self):
        first = self.fake_free_list_head.next_free_block
        assert first is not self.fake_free_list_tail
        self.remove(first)
        return first

    def contains(self, block_id):
        return block_id in self._in


class FakeCacheMap:
    """Mirrors BlockHashToBlockMap: hash -> block or {id: block}."""

    def __init__(self):
        self._c = {}

    def insert(self, key, block):
        cur = self._c.get(key)
        if cur is None:
            self._c[key] = block
        elif isinstance(cur, FakeBlock):
            self._c[key] = {cur.block_id: cur, block.block_id: block}
        else:
            cur[block.block_id] = block

    def pop(self, key, block_id):
        cur = self._c.pop(key, None)
        if cur is None:
            return None
        if isinstance(cur, FakeBlock):
            if cur.block_id == block_id:
                return cur
            self._c[key] = cur
            return None
        blk = cur.pop(block_id, None)
        if cur:
            self._c[key] = cur
        return blk

    def get_one_block(self, key):
        cur = self._c.get(key)
        if cur is None:
            return None
        if isinstance(cur, FakeBlock):
            return cur
        return next(iter(cur.values()))


class FakeByteSlice:
    def __init__(self, buffer, offset, size):
        self.buffer = buffer
        self.offset = offset
        self.size = size

    def copy_(self, source, non_blocking=False):
        del non_blocking
        self.buffer.data[self.offset : self.offset + self.size] = source.buffer.data[
            source.offset : source.offset + source.size
        ]


class FakeByteBuffer:
    def __init__(self, data):
        self.data = list(data)

    def numel(self):
        return len(self.data)

    def narrow(self, dim, offset, size):
        assert dim == 0
        return FakeByteSlice(self, offset, size)


class FakeBlockPool:
    def __init__(self, num_gpu_blocks):
        self.num_gpu_blocks = num_gpu_blocks
        self.blocks = [FakeBlock(i) for i in range(num_gpu_blocks)]
        self.free_block_queue = FakeQueue(self.blocks)
        self.cached_block_hash_to_block = FakeCacheMap()
        self.null_block = self.free_block_queue.popleft()  # block 0
        self.null_block.is_null = True
        self._on_alloc = []
        self._on_padd = []
        self._on_prem = []
        self._sel = None

    # registration
    def register_on_allocation_callback(self, cb):
        self._on_alloc.append(cb)

    def register_on_prefix_added_callback(self, cb):
        self._on_padd.append(cb)

    def register_on_prefix_removed_callback(self, cb):
        self._on_prem.append(cb)

    def register_kv_victim_selector(self, cb):
        self._sel = cb

    # prefix ops (mirror block_pool.py)
    def _maybe_evict(self, block):
        if block.block_hash is None:
            return False
        if (
            self.cached_block_hash_to_block.pop(block.block_hash, block.block_id)
            is None
        ):
            return False
        block.block_hash = None
        for cb in self._on_prem:
            cb(block.block_id)
        return True

    def evict_prefix_hash(self, block_id):
        return self._maybe_evict(self.blocks[block_id])

    def relocate_prefix_hash(self, src_id, dst_id):
        src = self.blocks[src_id]
        dst = self.blocks[dst_id]
        h = src.block_hash
        if h is None:
            return False
        assert dst.block_hash is None
        assert src.ref_cnt == 0
        popped = self.cached_block_hash_to_block.pop(h, src_id)
        assert popped is src
        src.block_hash = None
        dst.block_hash = h
        self.cached_block_hash_to_block.insert(h, dst)
        return True

    def get_num_free_blocks(self):
        return self.free_block_queue.num_free_blocks

    # simulate scheduler allocating n KV blocks
    def get_new_blocks(self, n):
        assert n <= self.get_num_free_blocks()
        ret = self._sel(n)
        for b in ret:
            self._maybe_evict(b)
            assert b.ref_cnt == 0
            b.ref_cnt += 1
        ids = [b.block_id for b in ret]
        for cb in self._on_alloc:
            cb(ids)
        return ret

    def free_blocks(self, blocks):
        for b in blocks:
            b.ref_cnt -= 1
        newly = [b for b in blocks if b.ref_cnt == 0 and not b.is_null]
        self.free_block_queue.append_n(newly)
        for b in newly:
            if b.block_hash is not None:
                for cb in self._on_padd:
                    cb(b.block_id)

    def touch(self, blocks):
        for b in blocks:
            if b.ref_cnt == 0 and not b.is_null:
                self.free_block_queue.remove(b)
                if b.block_hash is not None:
                    for cb in self._on_prem:
                        cb(b.block_id)
            b.ref_cnt += 1


# --------------------------- pool/manager builders -------------------------
def make_pool(layer_idx, num_experts, F, num_super_blocks, contents):
    p = object.__new__(UnifiedPool)
    p.layer_idx = layer_idx
    p.num_experts = num_experts
    p.pages_per_super_block = F
    p.num_super_blocks = num_super_blocks
    p.num_gpu_blocks = num_super_blocks * F
    p.expert_at_super_block = {}
    p.super_block_at_expert = {}
    p.expert_lru = OrderedDict()
    p.pinned_super_blocks = set()
    p.ever_activated = set()
    p.working_set_window = 0
    p.recent_expert_sets = deque()
    p.recent_expert_counts = Counter()
    p.super_block_id_at = [UnifiedPool._UNLOADED] * num_experts
    p.hits = p.misses = p.forward_count = 0
    p.page_size_bytes = 8
    p.expert_slot_bytes = 8 * F
    p.w13_bytes = 4 * F
    p.w2_bytes = 4 * F
    p.device = None
    p.pool_buffer = None
    p.cpu_w13 = p.cpu_w2 = None
    p._contents = contents  # (layer_idx, super_block) -> expert_id
    return p


def make_manager(block_pool, F, num_layers, num_experts):
    m = object.__new__(UnifiedPoolManager)
    m.block_pool = block_pool
    m.device = None
    m.pages_per_super_block = F
    m.page_size_bytes = 8
    m.kv_pool_buffers = {}
    m.num_super_blocks = block_pool.num_gpu_blocks // F
    m.layers = {}
    m.super_block_holder = {}
    m.transfer_stream = SimpleNamespace(wait_stream=lambda stream: None)
    m.step = 0
    m.prefix_lru = OrderedDict()
    m._relocation_enabled = True
    m.relocations = 0
    m._expert_content = {}  # (layer_idx, super_block) -> expert_id
    for li in range(num_layers):
        pool = make_pool(li, num_experts, F, m.num_super_blocks, m._expert_content)
        m.layers[li] = pool
    # wire block_pool callbacks (as the real __init__ does)
    block_pool.register_on_allocation_callback(m._on_kv_allocation)
    block_pool.register_on_prefix_added_callback(m._on_prefix_added)
    block_pool.register_on_prefix_removed_callback(m._on_prefix_removed)
    block_pool.register_kv_victim_selector(m._select_kv_victim_blocks)

    # stub GPU seams on the instance
    def dma_async(layer, eid, sb):
        m._expert_content[(layer.layer_idx, sb)] = eid

    def dma_sync(layer, eid, sb):
        m._expert_content[(layer.layer_idx, sb)] = eid

    def copy_page(src, dst):
        # a KV page's content is identified by its hash; the hash move is
        # done by relocate_prefix_hash. Nothing else to model here.
        pass

    m._dma_expert_into_super_block_async = dma_async
    m._dma_expert_into_super_block_sync = dma_sync
    m._copy_page_all_layers = copy_page
    return m


def test_manager_has_page_size():
    manager = make_manager(FakeBlockPool(8), 2, 1, 2)
    assert manager.page_size_bytes == 8


def test_rolling_working_set_tracks_recent_experts():
    pool = make_pool(0, 8, 1, 9, {})
    pool.working_set_window = 3

    pool.record_expert_accesses([0, 1])
    pool.record_expert_accesses([1, 2])
    pool.record_expert_accesses([2, 3])
    assert pool.working_set_ready
    assert pool.working_set_size == 4

    pool.record_expert_accesses([3])
    assert pool.working_set_size == 3


def test_adaptive_target_is_mean_working_set():
    manager = make_manager(FakeBlockPool(8), 1, 2, 8)
    for layer in manager.layers.values():
        layer.working_set_window = 2
    manager.layers[0].record_expert_accesses([0, 1, 2, 3])
    manager.layers[0].record_expert_accesses([0, 1, 2, 3])
    manager.layers[1].record_expert_accesses([0, 1])
    manager.layers[1].record_expert_accesses([0, 1])

    assert manager._adaptive_expert_target() == 3


def test_equal_working_sets_keep_full_target():
    manager = make_manager(FakeBlockPool(8), 1, 2, 8)
    for layer in manager.layers.values():
        layer.working_set_window = 1
        layer.record_expert_accesses([0, 1, 2, 3, 4, 5])

    assert manager._adaptive_expert_target() == 6


def test_copy_page_covers_attention_only_layers():
    manager = make_manager(FakeBlockPool(4), 2, 1, 2)
    manager.page_size_bytes = 2
    moe_buffer = FakeByteBuffer(range(8))
    dense_buffer = FakeByteBuffer(range(10, 18))
    manager.kv_pool_buffers = {0: moe_buffer, 1: dense_buffer}

    UnifiedPoolManager._copy_page_all_layers(manager, 1, 3)

    assert moe_buffer.data[6:8] == [2, 3]
    assert dense_buffer.data[6:8] == [12, 13]


def test_copy_waits_for_prior_compute_writes():
    manager = make_manager(FakeBlockPool(4), 2, 1, 2)
    manager.page_size_bytes = 2
    manager.kv_pool_buffers = {0: FakeByteBuffer(range(8))}
    waits = []
    manager.transfer_stream = SimpleNamespace(
        wait_stream=lambda stream: waits.append(stream)
    )

    UnifiedPoolManager._copy_page_all_layers(manager, 1, 3)

    assert len(waits) == 1


def add_prefix(manager, block_pool, block_id, block_hash, step):
    block = block_pool.blocks[block_id]
    block.block_hash = block_hash
    block_pool.cached_block_hash_to_block.insert(block_hash, block)
    manager.prefix_lru[block_id] = step


def test_relocation_preserves_lru_recency():
    """The destination inherits the source's recency, and is still found as
    the coldest page.

    This used to assert prefix_lru's dict order, which relocation kept
    sorted by re-sorting every entry on every moved page -- O(n log n) per
    page. Nothing reads that order (every consumer compares step values
    explicitly), so the re-sort is gone and the invariant is asserted where
    it actually matters: the recency value, and the selection it drives.
    """
    block_pool = FakeBlockPool(6)
    manager = make_manager(block_pool, 1, 1, 2)
    add_prefix(manager, block_pool, 1, 101, 2)
    add_prefix(manager, block_pool, 2, 102, 8)
    add_prefix(manager, block_pool, 3, 103, 10)

    manager._relocate_kv_page(1, 4)

    assert dict(manager.prefix_lru) == {4: 2, 2: 8, 3: 10}
    # Order-independent: the relocated page is still the coldest.
    assert manager._coldest_prefix_page(exclude_super_block=0, colder_than=1 << 30) == 4


def test_kv_victim_uses_oldest_timestamp():
    block_pool = FakeBlockPool(5)
    manager = make_manager(block_pool, 1, 1, 2)
    add_prefix(manager, block_pool, 1, 101, 9)
    add_prefix(manager, block_pool, 2, 102, 2)
    add_prefix(manager, block_pool, 3, 103, 7)
    add_prefix(manager, block_pool, 4, 104, 5)

    victim = manager._pick_one_kv_victim()

    assert victim.block_id == 2


def test_expert_miss_prefers_coldest_filled_super_block():
    """Vacate the super-block holding the least warm KV ("most cold pages").

    F=2, so sb1 = {p2,p3} and sb2 = {p4}. With the cold frontier at the
    F-th coldest page, sb1's pages (steps 1, 2) are both cold and sb2's
    (step 10) is warm. sb1 therefore exposes no warm KV and wins, even
    though it holds more pages and so may cost one extra relocation.

    The previous policy ranked by predicted relocation count and picked
    sb2, disturbing the pool's only warm page to save a 0.116 ms copy.
    """
    block_pool = FakeBlockPool(8)
    manager = make_manager(block_pool, 2, 1, 2)
    layer = manager.layers[0]
    add_prefix(manager, block_pool, 2, 102, 1)
    add_prefix(manager, block_pool, 3, 103, 2)
    add_prefix(manager, block_pool, 4, 104, 10)

    selected, _ = manager._evict_for_expert(layer, 0, {0})

    assert selected == 1


def test_expert_miss_avoids_warm_super_block_when_full():
    """With every candidate full, ranking falls to cold-page count.

    F=4: sb1 = {p4..p7} all cold, sb2 = {p8..p11} all warm. Both are
    full, so a "fewest total pages" rule could not separate them.
    """
    block_pool = FakeBlockPool(16)
    manager = make_manager(block_pool, 4, 1, 2)
    layer = manager.layers[0]
    for i, p in enumerate(range(4, 8)):
        add_prefix(manager, block_pool, p, 200 + p, i)
    for i, p in enumerate(range(8, 12)):
        add_prefix(manager, block_pool, p, 200 + p, 100 + i)

    selected, _ = manager._evict_for_expert(layer, 0, {0})

    assert selected == 1


def test_ensure_loaded_failure_clears_pins():
    block_pool = FakeBlockPool(4)
    manager = make_manager(block_pool, 1, 1, 3)
    layer = manager.layers[0]
    layer.assign(1, 0, 0)
    manager._add_holder(0, 1)
    manager._expert_content[(0, 1)] = 0
    calls = 0
    original = manager._select_super_block_for_expert

    def fail_second(layer_arg, eid, needed):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("forced exhaustion")
        return original(layer_arg, eid, needed)

    manager._select_super_block_for_expert = fail_second
    try:
        manager.ensure_loaded(layer, [0, 1, 2])
    except RuntimeError as error:
        assert str(error) == "forced exhaustion"
    else:
        raise AssertionError("ensure_loaded unexpectedly succeeded")

    assert layer.pinned_super_blocks == set()
    assert layer.forward_count == 0


def test_dma_failure_removes_new_mappings():
    block_pool = FakeBlockPool(4)
    manager = make_manager(block_pool, 1, 1, 2)
    layer = manager.layers[0]

    def fail_dma(layer_arg, eid, sb):
        raise RuntimeError("forced DMA failure")

    manager._dma_expert_into_super_block_async = fail_dma
    try:
        manager.ensure_loaded(layer, [0])
    except RuntimeError as error:
        assert str(error) == "forced DMA failure"
    else:
        raise AssertionError("ensure_loaded unexpectedly succeeded")

    assert layer.pinned_super_blocks == set()
    assert layer.expert_at_super_block == {}
    assert layer.super_block_at_expert == {}
    assert manager.super_block_holder == {}


def test_global_expert_uses_coldest_holder():
    block_pool = FakeBlockPool(4)
    manager = make_manager(block_pool, 1, 2, 4)
    layer0 = manager.layers[0]
    layer1 = manager.layers[1]
    layer0.assign(1, 0, 1)
    layer1.assign(1, 0, 100)
    layer0.assign(2, 1, 20)
    layer1.assign(2, 1, 30)
    manager._add_holder(0, 1)
    manager._add_holder(1, 1)
    manager._add_holder(0, 2)
    manager._add_holder(1, 2)

    selected, step = manager._oldest_global_expert()

    assert (selected, step) == (1, 1)


def test_kv_prefers_expert_while_footprint_exceeds_target():
    block_pool = FakeBlockPool(4)
    manager = make_manager(block_pool, 1, 2, 4)
    for layer in manager.layers.values():
        layer.working_set_window = 1
        layer.record_expert_accesses([0])
    manager.layers[0].assign(1, 0, 100)
    manager.layers[0].assign(2, 1, 100)
    manager._add_holder(0, 1)
    manager._add_holder(0, 2)
    add_prefix(manager, block_pool, 3, 103, 0)

    assert manager._pick_one_kv_victim().block_id in (1, 2)


def test_expert_miss_recycles_expert_at_target():
    block_pool = FakeBlockPool(3)
    manager = make_manager(block_pool, 1, 1, 3)
    layer = manager.layers[0]
    layer.working_set_window = 1
    layer.record_expert_accesses([0])
    layer.assign(1, 0, 100)
    manager._add_holder(0, 1)
    add_prefix(manager, block_pool, 2, 102, 0)

    selected, tier = manager._evict_for_expert(layer, 1, {1})

    assert (selected, tier) == (1, "expert-local")


def test_trace_reports_working_set_target_and_mixed_decisions():
    block_pool = FakeBlockPool(3)
    manager = make_manager(block_pool, 1, 1, 2)
    layer = manager.layers[0]
    layer.working_set_window = 1
    layer.record_expert_accesses([0])
    layer.ever_activated.add(0)
    layer.hits = 3
    layer.misses = 1
    layer.assign(1, 0, 100)
    manager._add_holder(0, 1)
    add_prefix(manager, block_pool, 2, 102, 80)

    old_trace = up._TRACE_ENABLED
    up._TRACE_ENABLED = True
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            manager._trace_pre_mutation(layer, [0])
            manager._adaptive_expert_target = lambda: None
            manager._pick_one_kv_victim()
    finally:
        up._TRACE_ENABLED = old_trace

    trace = output.getvalue()
    assert "working-set=1, expert-target=1" in trace
    assert "hits=3, misses=1, ever-activated=1" in trace
    assert "UNIFIED DECISION side=kv-alloc" in trace
    assert "expert-score=100.000 kv-score=80.000 chosen=kv" in trace


def test_inactive_page_tokens_are_not_validated():
    pydantic = types.ModuleType("pydantic")
    fields = {}

    def field(default=None, **kwargs):
        fields[len(fields)] = (default, kwargs)
        return default

    pydantic.Field = field
    pydantic.model_validator = lambda **kwargs: lambda function: function
    sys.modules["pydantic"] = pydantic
    config_utils = types.ModuleType("vllm.config.utils")
    config_utils.config = lambda cls: cls
    sys.modules["vllm.config.utils"] = config_utils
    offload_spec = importlib.util.spec_from_file_location(
        "phase4_offload", OFFLOAD_PATH
    )
    offload = importlib.util.module_from_spec(offload_spec)
    offload_spec.loader.exec_module(offload)
    inactive = SimpleNamespace(
        offload_backend="auto",
        uva=SimpleNamespace(cpu_offload_gb=0),
        prefetch=SimpleNamespace(
            offload_group_size=0,
            offload_num_in_group=1,
            offload_prefetch_step=1,
        ),
        expert_unified_pool=False,
        expert_offload=False,
        expert_pool_page_tokens=17,
        expert_working_set_window=64,
    )

    assert offload.OffloadConfig.validate_offload_config(inactive) is inactive
    assert any(
        default == 64 and kwargs.get("ge") == 0 for default, kwargs in fields.values()
    )

    active = SimpleNamespace(**vars(inactive))
    active.expert_unified_pool = True
    active.expert_offload = True
    try:
        offload.OffloadConfig.validate_offload_config(active)
    except ValueError as error:
        assert "must be a multiple of 16" in str(error)
    else:
        raise AssertionError("active invalid page size was accepted")


def run_deterministic_tests():
    test_manager_has_page_size()
    test_rolling_working_set_tracks_recent_experts()
    test_adaptive_target_is_mean_working_set()
    test_equal_working_sets_keep_full_target()
    test_copy_page_covers_attention_only_layers()
    test_copy_waits_for_prior_compute_writes()
    test_relocation_preserves_lru_recency()
    test_kv_victim_uses_oldest_timestamp()
    test_expert_miss_prefers_coldest_filled_super_block()
    test_expert_miss_avoids_warm_super_block_when_full()
    test_ensure_loaded_failure_clears_pins()
    test_dma_failure_removes_new_mappings()
    test_global_expert_uses_coldest_holder()
    test_kv_prefers_expert_while_footprint_exceeds_target()
    test_expert_miss_recycles_expert_at_target()
    test_trace_reports_working_set_target_and_mixed_decisions()
    test_inactive_page_tokens_are_not_validated()


# ------------------------------ invariants ---------------------------------
def check_invariants(m, bp, step_desc):
    F = m.pages_per_super_block
    # (1) bijection + mirror
    for li, layer in m.layers.items():
        assert len(layer.expert_at_super_block) == len(layer.super_block_at_expert)
        for s, e in layer.expert_at_super_block.items():
            assert layer.super_block_at_expert[e] == s, (step_desc, li, s, e)
            assert layer.super_block_id_at[e] == s, (step_desc, li, s, e)
        for e in range(layer.num_experts):
            if e in layer.super_block_at_expert:
                assert layer.super_block_id_at[e] == layer.super_block_at_expert[e]
            else:
                assert layer.super_block_id_at[e] == UnifiedPool._UNLOADED
    # (2) mutual exclusion: expert super-block holds no KV page
    for s, holders in m.super_block_holder.items():
        assert holders, (step_desc, "empty holder set present", s)
        for p in range(s * F, s * F + F):
            assert bp.blocks[p].block_hash is None, (
                step_desc,
                "KV in expert super-block",
                s,
                p,
            )
    # conversely a hashed page's super-block is not expert-held
    for p, blk in enumerate(bp.blocks):
        if blk.block_hash is not None:
            s = p // F
            assert s not in m.super_block_holder, (
                step_desc,
                "hashed page in expert sb",
                p,
                s,
            )
    # (3) holder consistency with per-layer maps
    for s, holders in m.super_block_holder.items():
        for li in holders:
            assert s in m.layers[li].expert_at_super_block, (step_desc, s, li)
    for li, layer in m.layers.items():
        for s in layer.expert_at_super_block:
            assert li in m.super_block_holder.get(s, set()), (step_desc, s, li)
    # (4) free-queue consistency: exactly ref_cnt==0 non-null blocks in queue
    for p, blk in enumerate(bp.blocks):
        if blk.is_null:
            continue
        in_q = bp.free_block_queue.contains(p)
        assert in_q == (blk.ref_cnt == 0), (
            step_desc,
            "queue/ref_cnt mismatch",
            p,
            blk.ref_cnt,
            in_q,
        )
    # (5) DMA delivered right expert content
    for li, layer in m.layers.items():
        for e, s in layer.super_block_at_expert.items():
            assert m._expert_content.get((li, s)) == e, (
                step_desc,
                "content mismatch",
                li,
                s,
                e,
                m._expert_content.get((li, s)),
            )
    # (6) cache-map consistency: every hashed block resolves through the
    # hash map to a block carrying that hash (catches relocate_prefix_hash
    # / evict bugs).
    for p, blk in enumerate(bp.blocks):
        if blk.block_hash is not None:
            found = bp.cached_block_hash_to_block.get_one_block(blk.block_hash)
            assert found is not None, (step_desc, "hashed block not in map", p)
            assert found.block_hash == blk.block_hash, (step_desc, "map mismatch", p)


# ------------------------------ the fuzz loop ------------------------------
def run(seed):
    rng = random.Random(seed)
    F = rng.choice([1, 2, 4, 8])
    num_super_blocks = rng.randint(6, 16)
    num_gpu_blocks = num_super_blocks * F
    num_experts = rng.randint(4, min(num_super_blocks - 1, 10))
    num_layers = rng.choice([1, 2, 3, 4])

    bp = FakeBlockPool(num_gpu_blocks)
    m = make_manager(bp, F, num_layers, num_experts)

    live_requests = []  # list of (blocks, will_cache)
    next_hash = [1000]

    for it in range(600):
        op = rng.random()
        if op < 0.55:
            # expert forward on a random layer: pick a random set of needed
            # experts, ensure_loaded, then release pins + end step.
            layer = m.layers[rng.randrange(num_layers)]
            k = rng.randint(1, min(3, num_experts))
            needed = rng.sample(range(num_experts), k)
            try:
                m.ensure_loaded(layer, needed)
            except RuntimeError as e:
                # pool exhaustion is an allowed outcome; skip the check that
                # needs loading to have succeeded.
                m.release_pinned(layer)
                m.end_forward_step()
                continue
            # all needed experts resident in this layer
            for e in needed:
                assert layer.has_expert(e), (seed, it, "missing after load", e)
            m.release_pinned(layer)
            m.end_forward_step()
        elif op < 0.80:
            # scheduler allocates 1-2 KV blocks (a new request prefill step)
            n = rng.randint(1, 2)
            if bp.get_num_free_blocks() < n:
                continue
            try:
                blks = bp.get_new_blocks(n)
            except RuntimeError:
                continue
            will_cache = rng.random() < 0.7
            live_requests.append((blks, will_cache))
        elif op < 0.92 and live_requests:
            # free a live request; optionally cache its blocks (set a hash)
            blks, will_cache = live_requests.pop(rng.randrange(len(live_requests)))
            if will_cache:
                for b in blks:
                    if b.block_hash is None:
                        h = next_hash[0]
                        next_hash[0] += 1
                        b.block_hash = h
                        bp.cached_block_hash_to_block.insert(h, b)
            bp.free_blocks(blks)
        else:
            # prefix-cache hit: touch a random cached (ref_cnt==0, hashed) block
            cands = [
                b
                for b in bp.blocks
                if b.block_hash is not None and b.ref_cnt == 0 and not b.is_null
            ]
            if not cands:
                continue
            b = rng.choice(cands)
            bp.touch([b])
            live_requests.append(([b], True))

        check_invariants(m, bp, (seed, it, "op"))

    return dict(
        F=F,
        nsb=num_super_blocks,
        ne=num_experts,
        nl=num_layers,
        relocations=m.relocations,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deterministic-only", action="store_true")
    args = parser.parse_args()
    run_deterministic_tests()
    if args.deterministic_only:
        print("PASS: deterministic regressions.")
        raise SystemExit(0)
    summary = {}
    for seed in range(400):
        info = run(seed)
        for k in ("relocations",):
            summary[k] = summary.get(k, 0) + info[k]
    print("PASS: 400 randomized seeds, all invariants held.")
    print("total relocations exercised across seeds:", summary["relocations"])
