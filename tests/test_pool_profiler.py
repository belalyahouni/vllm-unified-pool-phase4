"""Unit tests for the unified-pool overhead profiler.

The profiler is measurement code, so the properties that matter are that
it is exactly zero-cost when disabled, that it never loses or
double-counts an event, and that GPU timings are only resolved once the
events have actually completed.
"""

import importlib.util
import os
import sys
import types
from types import SimpleNamespace

import pytest

PROF_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "vllm",
    "vllm",
    "model_executor",
    "layers",
    "fused_moe",
    "pool_profiler.py",
)


class _EventStub:
    """torch.cuda.Event stand-in with controllable completion."""

    def __init__(self, enable_timing=False):
        self.enable_timing = enable_timing
        self.done = True
        self.synchronized = False

    def record(self, stream=None):
        pass

    def query(self):
        return self.done

    def synchronize(self):
        self.synchronized = True
        self.done = True

    def elapsed_time(self, other):
        return 2.0


def _load_profiler_module(enabled: bool):
    """Import a fresh copy of pool_profiler with the env gate set."""
    torch_stub = types.ModuleType("torch")
    cuda = types.ModuleType("torch.cuda")
    cuda.Event = _EventStub
    torch_stub.cuda = cuda
    logger_mod = types.ModuleType("vllm.logger")
    logger_mod.init_logger = lambda name: SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    saved = {k: sys.modules.get(k) for k in ("torch", "torch.cuda", "vllm", "vllm.logger")}
    sys.modules["torch"] = torch_stub
    sys.modules["torch.cuda"] = cuda
    sys.modules.setdefault("vllm", types.ModuleType("vllm"))
    sys.modules["vllm.logger"] = logger_mod
    prev = os.environ.get("VLLM_UNIFIED_POOL_PROFILE")
    os.environ["VLLM_UNIFIED_POOL_PROFILE"] = "1" if enabled else "0"
    try:
        spec = importlib.util.spec_from_file_location(
            f"pool_profiler_{enabled}", PROF_PATH
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        if prev is None:
            os.environ.pop("VLLM_UNIFIED_POOL_PROFILE", None)
        else:
            os.environ["VLLM_UNIFIED_POOL_PROFILE"] = prev
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


@pytest.fixture(scope="module")
def prof_on():
    return _load_profiler_module(True)


@pytest.fixture(scope="module")
def prof_off():
    return _load_profiler_module(False)


def test_disabled_decorator_returns_original_function(prof_off):
    """Zero-cost when off: no wrapper object at all."""
    p = prof_off.PoolProfiler(enabled=False)

    def fn(x):
        return x + 1

    assert p.timed("t")(fn) is fn
    assert p.report()["cpu"] == {}


def test_disabled_records_nothing(prof_off):
    p = prof_off.PoolProfiler(enabled=False)
    with p.cpu("a"):
        pass
    p.count("c", 5)
    ev = p.gpu_start()
    p.gpu_end(ev, "g", 123)
    r = p.report()
    assert ev is None
    assert r["cpu"] == {} and r["gpu"] == {} and r["counts"] == {}


def test_cpu_timer_counts_and_accumulates(prof_on):
    p = prof_on.PoolProfiler(enabled=True)
    for _ in range(3):
        with p.cpu("a"):
            pass
    r = p.report()["cpu"]["a"]
    assert r["count"] == 3
    assert r["total_ms"] >= 0.0
    assert r["depth"] == -1  # unknown name -> flagged, not silently 0


def test_decorator_times_and_propagates_exceptions(prof_on):
    p = prof_on.PoolProfiler(enabled=True)

    @p.timed("ensure_loaded")
    def boom():
        raise ValueError("x")

    with pytest.raises(ValueError):
        boom()
    # Timed even though it raised, so error paths are not invisible.
    assert p.report()["cpu"]["ensure_loaded"]["count"] == 1


def test_decorator_preserves_metadata_and_return(prof_on):
    p = prof_on.PoolProfiler(enabled=True)

    @p.timed("select_super_block")
    def fn(a, b=2):
        """doc"""
        return a + b

    assert fn(1, b=3) == 4
    assert fn.__name__ == "fn"
    assert fn.__doc__ == "doc"


def test_known_timers_get_nesting_depth(prof_on):
    p = prof_on.PoolProfiler(enabled=True)
    with p.cpu("ensure_loaded"):
        with p.cpu("cheapest_kv_super_block"):
            pass
    cpu = p.report()["cpu"]
    assert cpu["ensure_loaded"]["depth"] == 0
    assert cpu["cheapest_kv_super_block"]["depth"] == 2


def test_gpu_events_accumulate_time_and_bytes(prof_on):
    p = prof_on.PoolProfiler(enabled=True)
    for _ in range(2):
        ev = p.gpu_start()
        p.gpu_end(ev, "expert_dma_h2d", 1024)
    g = p.report()["gpu"]["expert_dma_h2d"]
    assert g["count"] == 2
    assert g["total_ms"] == pytest.approx(4.0)  # 2 events x 2.0ms stub
    assert g["mean_ms"] == pytest.approx(2.0)
    assert g["total_mib"] == pytest.approx(2048 / 2**20)
    assert g["gib_per_s"] > 0


def test_drain_defers_incomplete_events_without_forcing(prof_on):
    """An unfinished event must not be resolved, and must not be lost."""
    p = prof_on.PoolProfiler(enabled=True)
    ev = p.gpu_start()
    p.gpu_end(ev, "reloc_d2d_page", 64)
    # Mark the end event as still running.
    name, start, end, nbytes = p._pending[0]
    end.done = False

    p.drain()  # opportunistic: should skip it
    assert p._gpu == {}
    assert len(p._pending) == 1
    assert not end.synchronized

    p.drain(force=True)  # report time: waits and resolves
    assert end.synchronized
    assert p._pending == []
    assert p.report()["gpu"]["reloc_d2d_page"]["count"] == 1


def test_report_forces_pending_events(prof_on):
    p = prof_on.PoolProfiler(enabled=True)
    ev = p.gpu_start()
    p.gpu_end(ev, "g", 8)
    p._pending[0][2].done = False
    assert p.report()["gpu"]["g"]["count"] == 1
    assert p._pending == []


def test_counters_sum(prof_on):
    p = prof_on.PoolProfiler(enabled=True)
    p.count("miss_tier_kv-vacate")
    p.count("miss_tier_kv-vacate", 4)
    assert p.report()["counts"]["miss_tier_kv-vacate"] == 5


def test_format_report_mentions_all_sections(prof_on):
    p = prof_on.PoolProfiler(enabled=True)
    with p.cpu("ensure_loaded"):
        pass
    ev = p.gpu_start()
    p.gpu_end(ev, "expert_dma_h2d", 16)
    p.count("total_relocations", 3)
    text = p.format_report()
    assert "GPU byte movement" in text
    assert "CPU selection logic" in text
    assert "expert_dma_h2d" in text
    assert "ensure_loaded" in text
    assert "total_relocations: 3" in text


def test_max_pending_triggers_forced_drain(prof_on):
    """The pending list must not grow without bound on a long run."""
    p = prof_on.PoolProfiler(enabled=True)
    for _ in range(prof_on._MAX_PENDING_EVENTS + 10):
        ev = p.gpu_start()
        p.gpu_end(ev, "expert_dma_h2d", 1)
    assert len(p._pending) < prof_on._MAX_PENDING_EVENTS
    assert p.report()["gpu"]["expert_dma_h2d"]["count"] == (
        prof_on._MAX_PENDING_EVENTS + 10
    )


def test_periodic_dump_writes_on_interval_only(prof_on, tmp_path, monkeypatch):
    """A signal-killed server must still leave a report behind."""
    p = prof_on.PoolProfiler(enabled=True)
    out = tmp_path / "periodic.json"
    monkeypatch.setenv("VLLM_UNIFIED_POOL_PROF_JSON", str(out))
    monkeypatch.setattr(prof_on, "_DUMP_EVERY", 10)

    p.maybe_periodic_dump(7)
    assert not out.exists()

    p.count("total_steps", 10)
    p.maybe_periodic_dump(10)
    assert out.exists()

    import json

    assert json.loads(out.read_text())["counts"]["total_steps"] == 10


def test_periodic_dump_disabled_by_zero(prof_on, tmp_path, monkeypatch):
    p = prof_on.PoolProfiler(enabled=True)
    out = tmp_path / "never.json"
    monkeypatch.setenv("VLLM_UNIFIED_POOL_PROF_JSON", str(out))
    monkeypatch.setattr(prof_on, "_DUMP_EVERY", 0)
    p.maybe_periodic_dump(0)
    assert not out.exists()


def test_write_json_noop_without_env(prof_on, monkeypatch):
    monkeypatch.delenv("VLLM_UNIFIED_POOL_PROF_JSON", raising=False)
    prof_on.PoolProfiler(enabled=True).write_json()  # must not raise


def test_write_json_records_pid(prof_on, tmp_path, monkeypatch):
    p = prof_on.PoolProfiler(enabled=True)
    out = tmp_path / "pid.json"
    monkeypatch.setenv("VLLM_UNIFIED_POOL_PROF_JSON", str(out))
    p.write_json()
    import json

    assert json.loads(out.read_text())["pid"] == os.getpid()


def test_json_dump_written(prof_on, tmp_path):
    p = prof_on.PoolProfiler(enabled=True)
    p.count("x", 1)
    out = tmp_path / "prof.json"
    os.environ["VLLM_UNIFIED_POOL_PROF_JSON"] = str(out)
    try:
        p.dump()
    finally:
        os.environ.pop("VLLM_UNIFIED_POOL_PROF_JSON", None)
    import json

    assert json.loads(out.read_text())["counts"]["x"] == 1
