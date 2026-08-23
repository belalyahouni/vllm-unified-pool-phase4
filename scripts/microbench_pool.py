"""Standalone microbenchmark for unified-pool mechanism cost (workstream A).

No server, no model, no bench harness: measures the pool's primitives
directly, so the per-operation costs for the paper's appendix can be
produced in seconds and swept over configurations.

Two independent parts:

``--part gpu`` (needs a CUDA device, no model weights)
    Times the actual byte movement with CUDA events:
      * ``expert_h2d``   — one expert's weights, pinned host -> pool.
      * ``reloc_page``   — one logical KV page relocation, which is
                           ``num_layers`` small device-to-device copies
                           because a KV block is global.
      * ``reloc_fused``  — the same bytes as a single contiguous copy,
                           i.e. what a relocation would cost if a page
                           were not split across per-layer buffers. The
                           gap between these two is the cost attributable
                           to the layout, not to the data volume.
    Also sweeps page size (F) to show how relocation cost scales as pages
    get finer.

``--part cpu`` (pure Python, runs anywhere)
    Drives the *real* allocator logic from unified_pool.py at full scale
    (M super-blocks, F pages each, L layers) via the fuzzer's fakes, and
    times the victim-selection functions as a function of how much of the
    pool holds evictable KV. This is the axis the serving runs cannot
    isolate: ``_kv_super_block_cost`` returns early whenever a
    super-block is expert-held or pinned, so its true O(M * num_blocks)
    behaviour only appears once a large share of the pool is unpinned KV
    (the KV-heavy / workload-shift regime).

Defaults match the measured OLMoE-1B-7B configuration: expert_slot_bytes
= 12 MiB, F = 96 pages, page = 128 KiB, 16 layers, M = 67.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

MIB = 2**20

# Measured from a real run: 12060 MiB / 1005 expert loads == 12 MiB.
DEFAULT_EXPERT_BYTES = 12 * MIB
DEFAULT_F = 96
DEFAULT_LAYERS = 16
DEFAULT_M = 67


def _pct(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def _stats(name, samples_ms, nbytes):
    mean = statistics.fmean(samples_ms) if samples_ms else 0.0
    return {
        "op": name,
        "n": len(samples_ms),
        "mean_ms": mean,
        "median_ms": statistics.median(samples_ms) if samples_ms else 0.0,
        "p99_ms": _pct(samples_ms, 99),
        "bytes": nbytes,
        "mib": nbytes / MIB,
        "gib_per_s": ((nbytes / 2**30) / (mean / 1e3)) if mean > 0 else 0.0,
    }


# --------------------------------- GPU part ---------------------------------


def bench_gpu(args):
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("--part gpu needs a CUDA device")
    dev = torch.device("cuda")
    stream = torch.cuda.Stream(device=dev)
    expert_bytes = args.expert_bytes
    page_bytes = expert_bytes // args.pages_per_super_block
    results = []

    def timed(fn, reps, warmup=10):
        """Time fn() with one CUDA event pair per rep, on `stream`."""
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                fn()
        torch.cuda.synchronize()
        samples = []
        with torch.cuda.stream(stream):
            pairs = []
            for _ in range(reps):
                a = torch.cuda.Event(enable_timing=True)
                b = torch.cuda.Event(enable_timing=True)
                a.record()
                fn()
                b.record()
                pairs.append((a, b))
        torch.cuda.synchronize()
        for a, b in pairs:
            samples.append(a.elapsed_time(b))
        return samples

    # 1. Expert HtoD: pinned host -> a super-block sized slice of the pool.
    host = torch.empty(expert_bytes, dtype=torch.int8).pin_memory()
    pool = torch.empty(expert_bytes * 4, dtype=torch.int8, device=dev)
    slot = [0]

    def expert_h2d():
        # Rotate the destination so we are not always writing one address.
        off = (slot[0] % 4) * expert_bytes
        slot[0] += 1
        pool.narrow(0, off, expert_bytes).copy_(host, non_blocking=True)

    results.append(
        _stats("expert_h2d", timed(expert_h2d, args.reps), expert_bytes)
    )

    # 2. Relocation of one logical page == one small copy per layer, in each
    #    layer's own buffer. This is what _copy_page_all_layers does.
    layer_bufs = [
        torch.empty(page_bytes * 64, dtype=torch.int8, device=dev)
        for _ in range(args.layers)
    ]

    def reloc_page():
        for buf in layer_bufs:
            dst = buf.narrow(0, 0, page_bytes)
            src = buf.narrow(0, page_bytes * 8, page_bytes)
            dst.copy_(src, non_blocking=True)

    reloc_bytes = page_bytes * args.layers
    results.append(_stats("reloc_page", timed(reloc_page, args.reps), reloc_bytes))

    # 3. The same byte volume as ONE contiguous copy: the cost floor if a
    #    page were not fragmented across per-layer buffers.
    fused = torch.empty(reloc_bytes * 2, dtype=torch.int8, device=dev)

    def reloc_fused():
        fused.narrow(0, 0, reloc_bytes).copy_(
            fused.narrow(0, reloc_bytes, reloc_bytes), non_blocking=True
        )

    results.append(_stats("reloc_fused", timed(reloc_fused, args.reps), reloc_bytes))

    # 4. Page-size sweep: how relocation cost scales with F.
    sweep = []
    for f in args.f_sweep:
        pb = expert_bytes // f
        bufs = [
            torch.empty(pb * 4, dtype=torch.int8, device=dev)
            for _ in range(args.layers)
        ]

        def one():
            for buf in bufs:
                buf.narrow(0, 0, pb).copy_(buf.narrow(0, pb * 2, pb), non_blocking=True)

        s = _stats(f"reloc_page_F{f}", timed(one, max(20, args.reps // 4)), pb * args.layers)
        s["F"] = f
        s["page_kib"] = pb / 1024
        sweep.append(s)
        del bufs

    return {"primitives": results, "page_size_sweep": sweep}


# --------------------------------- CPU part ---------------------------------


def _load_harness():
    """Import the fuzzer module, which stubs torch and builds the fakes."""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    os.environ.setdefault("VLLM_UNIFIED_POOL_PROFILE", "0")
    import test_unified_pool_logic as h  # noqa: E402

    return h


def _build_state(h, M, F, layers, num_experts, kv_super_blocks, warm_pages_per_sb):
    """Pool with `kv_super_blocks` super-blocks of evictable cached-prefix
    KV; every other super-block holds an expert in every layer."""
    bp = h.FakeBlockPool(M * F)
    m = h.make_manager(bp, F, layers, num_experts)
    # Super-block 0 is reserved (holds the null page).
    kv_range = range(1, 1 + kv_super_blocks)
    expert_range = range(1 + kv_super_blocks, M)

    step = 0
    for s in kv_range:
        for i, p in enumerate(m._pages_of(s)):
            if p == 0 or i >= warm_pages_per_sb:
                continue
            h.add_prefix(m, bp, p, ("h", p), step)
            step += 1

    eid = 0
    for s in expert_range:
        for li in range(layers):
            layer = m.layers[li]
            e = eid % num_experts
            if e in layer.super_block_at_expert:
                continue
            layer.assign(s, e, step=step)
            m._add_holder(li, s)
        eid += 1
        step += 1
    m.step = step + 1
    return m, bp


def bench_cpu(args):
    h = _load_harness()
    M, F, layers = args.super_blocks, args.pages_per_super_block, args.layers
    num_experts = args.num_experts
    rows = []

    for frac in args.kv_fracs:
        kv_sb = max(1, int(round(frac * (M - 1))))
        warm = max(1, int(F * args.warm_frac))

        # --- read-only functions: many reps against one fixed state ---
        m, bp = _build_state(h, M, F, layers, num_experts, kv_sb, warm)
        ro = {}
        for name, fn in (
            ("cheapest_kv_super_block", m._cheapest_kv_super_block),
            ("oldest_global_expert", m._oldest_global_expert),
            (
                "coldest_prefix_page",
                lambda: m._coldest_prefix_page(
                    exclude_super_block=0, colder_than=1 << 30
                ),
            ),
            ("first_free_page", lambda: m._first_free_page(exclude_super_block=0)),
        ):
            reps = args.cpu_reps
            for _ in range(3):
                fn()
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            ro[name] = (time.perf_counter() - t0) / reps * 1e6  # us

        # --- mutating functions: rebuild state for every call ---
        mut = {}
        for name in ("vacate_kv_super_block", "select_kv_victim", "evict_for_expert"):
            samples = []
            for _ in range(args.mut_reps):
                mm, _bp = _build_state(h, M, F, layers, num_experts, kv_sb, warm)
                if name == "vacate_kv_super_block":
                    call = lambda: mm._vacate_kv_super_block(1, cause="bench")
                elif name == "select_kv_victim":
                    call = lambda: mm._select_kv_victim_blocks(1)
                else:
                    layer0 = mm.layers[0]
                    call = lambda: mm._evict_for_expert(layer0, 999, set())
                t0 = time.perf_counter()
                try:
                    call()
                except Exception:
                    samples = []
                    break
                samples.append((time.perf_counter() - t0) * 1e6)
            mut[name] = statistics.fmean(samples) if samples else None

        rows.append(
            {
                "kv_frac": frac,
                "kv_super_blocks": kv_sb,
                "warm_pages_per_sb": warm,
                "prefix_pages": len(m.prefix_lru),
                "expert_super_blocks": len(m.super_block_holder),
                "read_only_us": ro,
                "mutating_us": mut,
                "relocations_per_vacate": None,
            }
        )
    return {
        "config": {
            "M": M,
            "F": F,
            "layers": layers,
            "num_experts": num_experts,
            "num_blocks": M * F,
        },
        "rows": rows,
    }


# ---------------------------------- output ----------------------------------


def print_gpu(out):
    print("\n== GPU byte movement (CUDA events) ==")
    print(f"{'op':<18}{'n':>6}{'mean_ms':>10}{'p99_ms':>10}{'MiB':>9}{'GiB/s':>9}")
    for r in out["primitives"]:
        print(
            f"{r['op']:<18}{r['n']:>6}{r['mean_ms']:>10.4f}{r['p99_ms']:>10.4f}"
            f"{r['mib']:>9.3f}{r['gib_per_s']:>9.2f}"
        )
    prim = {r["op"]: r for r in out["primitives"]}
    if "expert_h2d" in prim and "reloc_page" in prim:
        e, p = prim["expert_h2d"]["mean_ms"], prim["reloc_page"]["mean_ms"]
        if p > 0:
            print(f"\n  one expert load == {e / p:.1f} page relocations")
    if "reloc_page" in prim and "reloc_fused" in prim:
        a, b = prim["reloc_page"]["mean_ms"], prim["reloc_fused"]["mean_ms"]
        if b > 0:
            print(
                f"  per-layer split costs {a / b:.1f}x a single contiguous copy "
                f"of the same {prim['reloc_page']['mib']:.3f} MiB"
            )
    print("\n== relocation cost vs page size ==")
    print(f"{'F':>6}{'page_KiB':>10}{'mean_ms':>10}{'MiB':>9}{'GiB/s':>9}")
    for r in out["page_size_sweep"]:
        print(
            f"{r['F']:>6}{r['page_kib']:>10.1f}{r['mean_ms']:>10.4f}"
            f"{r['mib']:>9.3f}{r['gib_per_s']:>9.2f}"
        )


def print_cpu(out):
    c = out["config"]
    print(
        f"\n== CPU selection logic (M={c['M']}, F={c['F']}, "
        f"{c['layers']} layers, {c['num_blocks']} blocks) =="
    )
    names = ["cheapest_kv_super_block", "oldest_global_expert",
             "coldest_prefix_page", "first_free_page"]
    hdr = f"{'kv_frac':>8}{'kv_sb':>7}{'pfx_pg':>8}"
    for n in names:
        hdr += f"{n[:14]:>16}"
    print(hdr + "   (microseconds per call)")
    for r in out["rows"]:
        line = f"{r['kv_frac']:>8.2f}{r['kv_super_blocks']:>7}{r['prefix_pages']:>8}"
        for n in names:
            line += f"{r['read_only_us'].get(n, 0.0):>16.1f}"
        print(line)
    print(f"\n{'kv_frac':>8}{'vacate_kv_sb':>16}{'select_kv_victim':>18}{'evict_for_expert':>18}")
    for r in out["rows"]:
        mu = r["mutating_us"]
        def f(k):
            v = mu.get(k)
            return f"{v:>.1f}" if v is not None else "n/a"
        print(f"{r['kv_frac']:>8.2f}{f('vacate_kv_super_block'):>16}"
              f"{f('select_kv_victim'):>18}{f('evict_for_expert'):>18}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", choices=("gpu", "cpu", "both"), default="both")
    ap.add_argument("--expert-bytes", type=int, default=DEFAULT_EXPERT_BYTES)
    ap.add_argument("--pages-per-super-block", type=int, default=DEFAULT_F)
    ap.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    ap.add_argument("--super-blocks", type=int, default=DEFAULT_M)
    ap.add_argument("--num-experts", type=int, default=64)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--cpu-reps", type=int, default=20)
    ap.add_argument("--mut-reps", type=int, default=5)
    ap.add_argument("--warm-frac", type=float, default=0.5,
                    help="fraction of a KV super-block's pages that are warm")
    ap.add_argument("--kv-fracs", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 0.95])
    ap.add_argument("--f-sweep", type=int, nargs="+", default=[1, 6, 24, 96, 384])
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    out = {"args": vars(args)}
    if args.part in ("gpu", "both"):
        try:
            out["gpu"] = bench_gpu(args)
            print_gpu(out["gpu"])
        except SystemExit as e:
            print(f"[skip gpu] {e}")
    if args.part in ("cpu", "both"):
        out["cpu"] = bench_cpu(args)
        print_cpu(out["cpu"])
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
