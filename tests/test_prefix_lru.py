"""Unit tests for PrefixLRU, the unified pool's recency-ordered page list.

The structure replaced an OrderedDict whose *ordering* was unreliable: a
relocated page had to be re-keyed, which can only append, so an old page
landed in the newest slot and every reader had to scan to find the true
oldest. The properties worth pinning down are therefore:

  * traversal order really is recency order, head (coldest) to tail;
  * rename (relocation) changes neither recency nor position;
  * the ordered queries answer from the cold/warm end without scanning;
  * validate() actually catches a corrupted list rather than passing.
"""

import importlib.util
import os
import sys
import types
from types import SimpleNamespace

import pytest

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


def _load_unified_pool():
    """Load unified_pool.py with torch and vllm.logger stubbed out."""
    torch_stub = types.ModuleType("torch")
    torch_stub.int8 = "int8"
    torch_stub.int64 = "int64"
    cuda = types.ModuleType("torch.cuda")
    cuda.Event = object
    cuda.Stream = lambda device=None: object()
    cuda.current_stream = lambda device=None: SimpleNamespace(
        wait_stream=lambda s: None
    )
    cuda.stream = lambda s: None
    cuda.synchronize = lambda device=None: None
    torch_stub.cuda = cuda
    logger_mod = types.ModuleType("vllm.logger")
    logger_mod.init_logger = lambda name: SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    prof_mod = types.ModuleType("vllm.model_executor.layers.fused_moe.pool_profiler")

    class _Prof:
        enabled = False

        def timed(self, name):
            return lambda fn: fn

        def cpu(self, name):
            import contextlib

            return contextlib.nullcontext()

        def gpu_start(self):
            return None

        def gpu_end(self, *a):
            pass

        def count(self, *a):
            pass

        def drain(self, *a, **k):
            pass

        def maybe_periodic_dump(self, *a):
            pass

        def dump(self):
            pass

    prof_mod.PROFILER = _Prof()
    sys.modules["torch"] = torch_stub
    sys.modules["torch.cuda"] = cuda
    sys.modules.setdefault("vllm", types.ModuleType("vllm"))
    sys.modules["vllm.logger"] = logger_mod
    for pkg in (
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.fused_moe",
    ):
        sys.modules.setdefault(pkg, types.ModuleType(pkg))
    sys.modules[
        "vllm.model_executor.layers.fused_moe.pool_profiler"
    ] = prof_mod
    spec = importlib.util.spec_from_file_location("unified_pool_for_test", UP_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def up():
    return _load_unified_pool()


@pytest.fixture
def lru(up):
    return up.PrefixLRU()


def pages(lru):
    return [p for p, _ in lru.items()]


# -- basic mapping behaviour --


def test_empty(lru):
    assert len(lru) == 0
    assert not lru
    assert lru.oldest() is None
    assert list(lru.items()) == []
    assert lru.get(7) is None
    assert lru.get(7, -1) == -1
    assert 7 not in lru
    lru.validate()


def test_touch_appends_in_order(lru):
    for i, page in enumerate([5, 6, 7]):
        lru.touch(page, i)
    assert pages(lru) == [5, 6, 7]
    assert lru.oldest() == (5, 0)
    assert lru[7] == 2
    assert len(lru) == 3
    lru.validate()


def test_touch_existing_restamps_and_moves_to_newest(lru):
    lru.touch(1, 0)
    lru.touch(2, 1)
    lru.touch(3, 2)
    lru.touch(1, 3)  # page 1 used again
    assert pages(lru) == [2, 3, 1]
    assert lru[1] == 3
    assert lru.oldest() == (2, 1)
    assert len(lru) == 3
    lru.validate()


def test_pop(lru):
    lru.touch(1, 0)
    lru.touch(2, 1)
    assert lru.pop(1) == 0
    assert 1 not in lru
    assert pages(lru) == [2]
    assert lru.pop(99) is None
    assert lru.pop(99, "x") == "x"
    lru.validate()


def test_pop_head_then_tail_leaves_empty(lru):
    lru.touch(1, 0)
    lru.touch(2, 1)
    lru.pop(1)
    lru.pop(2)
    assert len(lru) == 0
    assert lru.oldest() is None
    assert list(lru.items()) == []
    lru.validate()


def test_values_follow_items(lru):
    for i, p in enumerate([4, 5, 6]):
        lru.touch(p, i * 10)
    assert list(lru.values()) == [0, 10, 20]


# -- rename: the operation the whole structure exists for --


def test_rename_keeps_recency_and_position(lru):
    lru.touch(1, 0)
    lru.touch(2, 1)
    lru.touch(3, 2)

    lru.rename(2, 99, fallback_step=1000)

    # 99 sits exactly where 2 sat, with 2's step -- not at the tail.
    assert pages(lru) == [1, 99, 3]
    assert lru[99] == 1
    assert 2 not in lru
    assert len(lru) == 3
    lru.validate()


def test_rename_head_stays_head(lru):
    lru.touch(1, 0)
    lru.touch(2, 1)
    lru.rename(1, 50, fallback_step=999)
    assert pages(lru) == [50, 2]
    assert lru.oldest() == (50, 0)
    lru.validate()


def test_rename_tail_stays_tail(lru):
    lru.touch(1, 0)
    lru.touch(2, 1)
    lru.rename(2, 50, fallback_step=999)
    assert pages(lru) == [1, 50]
    assert lru[50] == 1
    lru.touch(3, 2)  # tail links still intact
    assert pages(lru) == [1, 50, 3]
    lru.validate()


def test_rename_returns_inherited_step(lru):
    lru.touch(1, 42)
    assert lru.rename(1, 2, fallback_step=999) == 42


def test_rename_missing_source_inserts_as_newest(lru):
    lru.touch(1, 5)
    step = lru.rename(404, 7, fallback_step=9)
    assert step == 9
    assert pages(lru) == [1, 7]
    assert lru[7] == 9
    lru.validate()


def test_rename_over_existing_destination_does_not_duplicate(lru):
    lru.touch(1, 0)
    lru.touch(2, 1)
    lru.touch(3, 2)
    lru.rename(1, 3, fallback_step=999)  # dst 3 already present
    assert pages(lru) == [3, 2]
    assert lru[3] == 0  # inherited from 1, and took 1's slot
    assert len(lru) == 2
    lru.validate()


def test_repeated_rename_never_disturbs_order(lru):
    """A vacate renames many pages; order must survive all of them."""
    for i in range(10):
        lru.touch(i, i)
    for i in range(10):
        lru.rename(i, 100 + i, fallback_step=999)
    assert pages(lru) == [100 + i for i in range(10)]
    assert list(lru.values()) == list(range(10))
    lru.validate()


# -- ordered queries --


def test_coldest_first_respects_limit(lru):
    for i in range(10):
        lru.touch(i, i)
    assert [p for p, _ in lru.coldest_first(limit=3)] == [0, 1, 2]
    assert len([1 for _ in lru.coldest_first()]) == 10
    assert [p for p, _ in lru.coldest_first(limit=0)] == []


def test_newest_first_respects_limit(lru):
    for i in range(10):
        lru.touch(i, i)
    assert [p for p, _ in lru.newest_first(limit=3)] == [9, 8, 7]
    assert [p for p, _ in lru.newest_first()][0] == 9


def test_oldest_tracks_evictions(lru):
    for i in range(3):
        lru.touch(i, i)
    assert lru.oldest() == (0, 0)
    lru.pop(0)
    assert lru.oldest() == (1, 1)


# -- insert_ordered (test/benchmark setup only) --


def test_insert_ordered_places_by_step(lru):
    lru.insert_ordered(10, 5)
    lru.insert_ordered(20, 1)  # oldest -> head
    lru.insert_ordered(30, 9)  # newest -> tail
    lru.insert_ordered(40, 7)  # middle
    assert pages(lru) == [20, 10, 40, 30]
    assert list(lru.values()) == [1, 5, 7, 9]
    lru.validate()


def test_insert_ordered_into_empty(lru):
    lru.insert_ordered(1, 3)
    assert pages(lru) == [1]
    assert lru.oldest() == (1, 3)
    lru.validate()


def test_insert_ordered_replaces_existing(lru):
    lru.insert_ordered(1, 1)
    lru.insert_ordered(2, 2)
    lru.insert_ordered(1, 5)  # same page, later step -> moves after 2
    assert pages(lru) == [2, 1]
    assert len(lru) == 2
    lru.validate()


def test_insert_ordered_ties_keep_list_valid(lru):
    for p in (1, 2, 3):
        lru.insert_ordered(p, 7)
    assert len(lru) == 3
    lru.validate()


# -- validate() must actually catch corruption --


def test_validate_catches_step_inversion(lru, up):
    lru.touch(1, 5)
    lru.touch(2, 9)
    lru._nodes[2].step = 0  # now warmer entry sits before a colder one
    with pytest.raises(AssertionError):
        lru.validate()


def test_validate_catches_orphaned_map_entry(lru, up):
    lru.touch(1, 0)
    lru._nodes[999] = up._PrefixNode(999, 0)  # mapped but not linked
    with pytest.raises(AssertionError):
        lru.validate()


def test_validate_catches_broken_link(lru):
    lru.touch(1, 0)
    lru.touch(2, 1)
    lru._nodes[2].prev = None  # link no longer points back
    with pytest.raises(AssertionError):
        lru.validate()


def test_validate_passes_after_heavy_mixed_use(lru):
    import random

    rng = random.Random(1234)
    step = 0
    live = set()
    for _ in range(2000):
        op = rng.random()
        if op < 0.5 or not live:
            page = rng.randrange(200)
            step += 1
            lru.touch(page, step)
            live.add(page)
        elif op < 0.75:
            page = rng.choice(sorted(live))
            lru.pop(page)
            live.discard(page)
        else:
            src = rng.choice(sorted(live))
            dst = rng.randrange(200, 400)
            lru.rename(src, dst, fallback_step=step)
            live.discard(src)
            live.add(dst)
        lru.validate()
    assert len(lru) == len(live)
