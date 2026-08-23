# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Overhead profiler for the unified pool (workstream A).

Answers "what does the shared address space actually cost?" by splitting
the pool's per-miss work into three buckets that are conflated in
end-to-end latency:

* ``gpu`` — the byte movement: expert HtoD DMA out of pinned host memory,
  and the device-to-device page copies that relocation performs. Measured
  with CUDA events on the transfer stream.
* ``cpu`` — the Python victim-selection logic (the super-block cost
  sweep, the prefix scans, the prefix_lru re-sort). Measured with
  ``perf_counter``. This is the bucket the design discussion assumes is
  free, and the one that scales as O(num_super_blocks * num_blocks) per
  expert miss.
* ``counts`` — how often each path fires, so a per-event mean can be
  derived and the buckets can be normalised per miss.

Gated on ``VLLM_UNIFIED_POOL_PROFILE=1``; when off every entry point is a
single attribute check, so instrumented call sites cost nothing in the
latency runs. Because CUDA events serialise nothing by themselves, the
GPU timings are collected without forcing a synchronize: completed event
pairs are drained opportunistically and the remainder resolved once at
report time.

CPU timers nest (``ensure_loaded`` contains ``select_super_block``
contains ``evict_for_expert`` contains ``kv_cost_sweep``), so times are
*inclusive* and must not be summed across levels. ``report()`` tags each
timer with its nesting depth so a reader cannot accidentally double count.
"""

from __future__ import annotations

import functools
import json
import os
import time
from collections import defaultdict
from contextlib import contextmanager

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Resolved once at import: the gate is checked on paths that fire tens of
# thousands of times per request.
_PROFILE_ENABLED = os.environ.get("VLLM_UNIFIED_POOL_PROFILE", "0") == "1"

# Nesting depth of each CPU timer, for reporting only. Timers at depth d
# are contained in their nearest enclosing timer of depth < d, so summing
# across depths double counts.
_TIMER_DEPTH = {
    "ensure_loaded": 0,
    "select_super_block": 1,
    "evict_for_expert": 2,
    "kv_cost_sweep": 3,
    "vacate_kv_super_block": 3,
    "first_free_page": 4,
    "coldest_prefix_page": 4,
    "relocate_page": 4,
    "prefix_lru_resort": 5,
    "select_kv_victim": 0,
    "pick_kv_victim": 1,
    "oldest_global_expert": 2,
}

# Above this many unresolved CUDA event pairs, force a resolve so the
# event pool cannot grow without bound on a long run.
_MAX_PENDING_EVENTS = 4096


class PoolProfiler:
    """Accumulates CPU wall time, GPU event time and event counts."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        # name -> [count, total_seconds]
        self._cpu: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
        # name -> [count, total_ms, total_bytes]
        self._gpu: dict[str, list[float]] = defaultdict(lambda: [0, 0.0, 0])
        self._counts: dict[str, int] = defaultdict(int)
        # (name, start_event, end_event, nbytes) awaiting elapsed_time.
        self._pending: list[tuple[str, torch.cuda.Event, torch.cuda.Event, int]] = []

    # CPU timing.

    @contextmanager
    def cpu(self, name: str):
        """Time a block of Python. Inclusive of any nested timers."""
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            slot = self._cpu[name]
            slot[0] += 1
            slot[1] += dt

    def timed(self, name: str):
        """Decorator form of ``cpu``, for whole methods.

        When profiling is off the undecorated function is returned, so
        instrumented methods keep their original call cost exactly.
        """

        def deco(fn):
            if not self.enabled:
                return fn

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                t0 = time.perf_counter()
                try:
                    return fn(*args, **kwargs)
                finally:
                    dt = time.perf_counter() - t0
                    slot = self._cpu[name]
                    slot[0] += 1
                    slot[1] += dt

            return wrapper

        return deco

    # GPU timing. Events are recorded on whichever stream is current at
    # the call site, which for the pool is always transfer_stream.

    def gpu_start(self) -> torch.cuda.Event | None:
        if not self.enabled:
            return None
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        return ev

    def gpu_end(self, start: torch.cuda.Event | None, name: str, nbytes: int) -> None:
        if not self.enabled or start is None:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self._pending.append((name, start, end, nbytes))
        if len(self._pending) >= _MAX_PENDING_EVENTS:
            self.drain(force=True)

    def drain(self, force: bool = False) -> None:
        """Resolve finished event pairs into totals.

        Without ``force`` only pairs whose end event has already completed
        are resolved, so profiling never inserts a stall. ``force`` waits
        on the stragglers and is used once at report time.
        """
        if not self.enabled or not self._pending:
            return
        still: list[tuple[str, torch.cuda.Event, torch.cuda.Event, int]] = []
        for name, start, end, nbytes in self._pending:
            if not force and not end.query():
                still.append((name, start, end, nbytes))
                continue
            if force:
                end.synchronize()
            slot = self._gpu[name]
            slot[0] += 1
            slot[1] += start.elapsed_time(end)
            slot[2] += nbytes
        self._pending = still

    # Plain counters.

    def count(self, name: str, n: int = 1) -> None:
        if not self.enabled:
            return
        self._counts[name] += n

    # Reporting.

    def report(self) -> dict:
        self.drain(force=True)
        cpu = {
            name: {
                "count": int(c),
                "total_ms": t * 1e3,
                "mean_us": (t / c * 1e6) if c else 0.0,
                "depth": _TIMER_DEPTH.get(name, -1),
            }
            for name, (c, t) in sorted(self._cpu.items())
        }
        gpu = {
            name: {
                "count": int(c),
                "total_ms": t,
                "mean_ms": (t / c) if c else 0.0,
                "total_mib": b / 2**20,
                "gib_per_s": ((b / 2**30) / (t / 1e3)) if t > 0 else 0.0,
            }
            for name, (c, t, b) in sorted(self._gpu.items())
        }
        return {
            "cpu": cpu,
            "gpu": gpu,
            "counts": dict(sorted(self._counts.items())),
        }

    def format_report(self) -> str:
        r = self.report()
        lines = ["UNIFIED PROF ==== overhead breakdown ===="]
        lines.append("UNIFIED PROF -- GPU byte movement (CUDA events)")
        for name, d in r["gpu"].items():
            lines.append(
                f"UNIFIED PROF   {name}: n={d['count']} "
                f"total={d['total_ms']:.2f}ms mean={d['mean_ms']:.3f}ms "
                f"moved={d['total_mib']:.1f}MiB "
                f"eff={d['gib_per_s']:.2f}GiB/s"
            )
        lines.append("UNIFIED PROF -- CPU selection logic (inclusive; do not sum)")
        for name, d in sorted(r["cpu"].items(), key=lambda kv: kv[1]["depth"]):
            lines.append(
                f"UNIFIED PROF   {'  ' * max(d['depth'], 0)}{name}: "
                f"n={d['count']} total={d['total_ms']:.2f}ms "
                f"mean={d['mean_us']:.1f}us"
            )
        lines.append("UNIFIED PROF -- counts")
        for name, v in r["counts"].items():
            lines.append(f"UNIFIED PROF   {name}: {v}")
        lines.append("UNIFIED PROF ==== end ====")
        return "\n".join(lines)

    def dump(self) -> None:
        """Log the human-readable report and, if requested, write JSON."""
        if not self.enabled:
            return
        for line in self.format_report().splitlines():
            print(line, flush=True)
        path = os.environ.get("VLLM_UNIFIED_POOL_PROF_JSON")
        if path:
            try:
                with open(path, "w") as f:
                    json.dump(self.report(), f, indent=2)
                logger.info("UnifiedPool profile written to %s", path)
            except OSError as e:
                logger.warning("Could not write pool profile to %s: %s", path, e)


# Module-level singleton: the pool is a per-process object and the
# profiler has to be reachable from both the manager and the layer.
PROFILER = PoolProfiler(_PROFILE_ENABLED)


def profiler_enabled() -> bool:
    return _PROFILE_ENABLED
