#!/usr/bin/env python3
"""Summarize a unified-pool trace log into a tiny markdown report.

Usage: summarize_trace.py <trace.log> [<trace.log> ...]
Writes <basename>.summary.md next to each input.

Streams line-by-line so it handles multi-GB lvl2 traces without loading
into memory.
"""

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


RE_EVICT = re.compile(
    r"UNIFIED EVICT (?:page|sb)=(?P<location>\S+) (?P<layer>L\S+) "
    r"kind=(?P<kind>\S+) (?:E(?P<eid>\d+) )?cause=(?P<cause>\S+)"
    r"(?: tier=(?P<tier>\S+))?(?: score=(?P<score>-?[\d.]+) step=(?P<step>\d+))?"
)
RE_KV_CLAIM = re.compile(
    r"UNIFIED KV_CLAIM page=(?P<page>\S+)(?: sb=\S+)? tier=(?P<tier>\S+)"
)
RE_PREFIX_ADD = re.compile(r"UNIFIED PREFIX_ADDED")
RE_PREFIX_REM = re.compile(r"UNIFIED PREFIX_REMOVED")
RE_CACHE = re.compile(
    r"UNIFIED CACHE L(?P<layer>\d+) step=(?P<step>\d+) "
    r"F=(?P<F>\d+) "
    r"expert_sb (?P<eo>\d+)/(?P<cap>\d+) ours "
    r"\(expert-ours-sb=(?P<eo2>\d+), expert-other-sb=(?P<ex>\d+), "
    r"prefix-pages=(?P<pf>\d+), alloc-kv-pages=(?P<kv>\d+), "
    r"pinned-sb=(?P<pn>\d+), "
    r"hits=(?P<hits>\d+), misses=(?P<misses>\d+), "
    r"ever-activated=(?P<activated>\d+)\)"
)
RE_DECISION = re.compile(
    r"UNIFIED DECISION side=(?P<side>expert-miss|kv-alloc) step=(?P<step>\d+) "
    r"layer=(?P<layer>all|\d+) expert-score=(?P<expert>-?[\d.]+) "
    r"kv-score=(?P<kv>-?[\d.]+) chosen=(?P<chosen>expert|kv)"
)
RE_REQUEST = re.compile(r"UNIFIED REQUEST L(?P<layer>\d+): (?P<experts>[\d,E ]+)")
RE_NUM_BLOCKS = re.compile(r"num_gpu_blocks_override=(\d+)|num_gpu_blocks=(\d+)")
RE_NON_DEFAULT = re.compile(r"non-default args: (\{[^\}]+\})")


def fmt_pct(n, total):
    if total == 0:
        return "0%"
    return f"{100 * n / total:.1f}%"


def summarize(path: Path) -> str:
    evict_kind = Counter()
    evict_cause = Counter()
    evict_tier = Counter()
    evict_kind_cause_tier = Counter()
    kv_tier = Counter()
    prefix_added = 0
    prefix_removed = 0
    expert_evicted_per_layer = defaultdict(set)
    expert_requested_per_layer = defaultdict(set)
    expert_evict_count_per_layer = Counter()
    decisions = Counter()
    decisions_per_layer = Counter()
    decision_margins = defaultdict(list)
    request_lines_seen = 0
    last_cache_per_layer = {}
    cache_lines_seen = 0
    config_line = None
    cache_cap = None
    # Track expert-vs-kv swing: per layer, the most expert-heavy and most
    # kv-heavy snapshot (where total = expert-ours + alloc-kv > 0).
    swing_per_layer = {}  # layer -> {"max": (ratio, step, eo, kv), "min": (...)}
    # Across the whole run (any layer at any step):
    global_max = None  # (ratio, layer, step, eo, kv)
    global_min = None

    size = path.stat().st_size
    with path.open("r", errors="replace") as f:
        for line in f:
            if config_line is None:
                m = RE_NON_DEFAULT.search(line)
                if m:
                    config_line = m.group(1)
            if "UNIFIED " not in line:
                continue
            m = RE_EVICT.search(line)
            if m:
                kind = m["kind"]
                cause = m["cause"]
                tier = m["tier"]
                evict_kind[kind] += 1
                evict_cause[cause] += 1
                if tier:
                    evict_tier[tier] += 1
                evict_kind_cause_tier[(kind, cause, tier)] += 1
                if kind == "expert" and m["eid"] and m["layer"].startswith("L"):
                    layer = m["layer"][1:]
                    if layer.isdigit():
                        expert_evicted_per_layer[int(layer)].add(int(m["eid"]))
                        expert_evict_count_per_layer[int(layer)] += 1
                continue
            m = RE_KV_CLAIM.search(line)
            if m:
                kv_tier[m["tier"]] += 1
                continue
            if RE_PREFIX_ADD.search(line):
                prefix_added += 1
                continue
            if RE_PREFIX_REM.search(line):
                prefix_removed += 1
                continue
            m = RE_CACHE.search(line)
            if m:
                cache_lines_seen += 1
                layer = int(m["layer"])
                last_cache_per_layer[layer] = {
                    "step": int(m["step"]),
                    "cap": int(m["cap"]),
                    "expert_ours": int(m["eo"]),
                    "expert_other": int(m["ex"]),
                    "prefix": int(m["pf"]),
                    "alloc_kv": int(m["kv"]),
                    "pinned": int(m["pn"]),
                    "hits": int(m["hits"]),
                    "misses": int(m["misses"]),
                    "activated": int(m["activated"]),
                }
                cache_cap = int(m["cap"])
                eo = int(m["eo"])
                kv = int(m["kv"])
                step = int(m["step"])
                total_ek = eo + kv
                if total_ek > 0:
                    ratio = eo / total_ek
                    entry = swing_per_layer.setdefault(
                        layer, {"max": None, "min": None}
                    )
                    if entry["max"] is None or ratio > entry["max"][0]:
                        entry["max"] = (ratio, step, eo, kv)
                    if entry["min"] is None or ratio < entry["min"][0]:
                        entry["min"] = (ratio, step, eo, kv)
                    sample = (ratio, layer, step, eo, kv)
                    if global_max is None or ratio > global_max[0]:
                        global_max = sample
                    if global_min is None or ratio < global_min[0]:
                        global_min = sample
                continue
            m = RE_DECISION.search(line)
            if m:
                side = m["side"]
                chosen = m["chosen"]
                layer = m["layer"]
                decisions[(side, chosen)] += 1
                decisions_per_layer[(layer, chosen)] += 1
                decision_margins[(side, chosen)].append(
                    float(m["kv"]) - float(m["expert"])
                )
                continue
            m = RE_REQUEST.search(line)
            if m:
                request_lines_seen += 1
                layer = int(m["layer"])
                for tok in m["experts"].split(","):
                    tok = tok.strip().lstrip("E")
                    if tok.isdigit():
                        expert_requested_per_layer[layer].add(int(tok))

    total_evict = sum(evict_kind.values())
    total_kv_claim = sum(kv_tier.values())

    has_lvl2 = request_lines_seen > 0

    lines = []
    lines.append(f"# Trace summary: `{path.name}`")
    lines.append("")
    lines.append(f"- File size: {size / (1024 * 1024):.1f} MB")
    lines.append(f"- Trace level: {'2 (verbose)' if has_lvl2 else '1 (essential)'}")
    if cache_cap is not None:
        lines.append(f"- Pool capacity (pages): {cache_cap}")
    if config_line:
        lines.append("")
        lines.append("## Run config")
        lines.append("")
        for tok in config_line.strip("{}").split(", "):
            lines.append(f"- {tok}")

    lines.append("")
    lines.append("## Per-layer outcomes")
    lines.append("")
    lines.append(
        "| Layer | resident experts | activated | hits | misses | hit rate | evictions |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for layer in sorted(last_cache_per_layer):
        data = last_cache_per_layer[layer]
        total = data["hits"] + data["misses"]
        lines.append(
            f"| L{layer} | {data['expert_ours']} | "
            f"{data['activated']} | {data['hits']} | {data['misses']} | "
            f"{fmt_pct(data['hits'], total)} | {expert_evict_count_per_layer[layer]} |"
        )

    lines.append("")
    lines.append("## Mixed-LRU decisions")
    lines.append("")
    lines.append("| Side | Chosen victim | Count | Mean `kv-score - expert-score` |")
    lines.append("|---|---|---:|---:|")
    for side, chosen in sorted(decisions):
        margins = decision_margins[(side, chosen)]
        mean_margin = sum(margins) / len(margins)
        lines.append(
            f"| `{side}` | `{chosen}` | {decisions[(side, chosen)]} | {mean_margin:.3f} |"
        )
    lines.append("")
    lines.append("| Layer | Comparisons | Expert chosen | KV chosen |")
    lines.append("|---|---:|---:|---:|")
    decision_layers = sorted({layer for layer, _ in decisions_per_layer})
    for layer in decision_layers:
        expert = decisions_per_layer[(layer, "expert")]
        kv = decisions_per_layer[(layer, "kv")]
        label = layer if layer == "all" else f"L{layer}"
        lines.append(f"| {label} | {expert + kv} | {expert} | {kv} |")

    lines.append("")
    lines.append("## Eviction & KV-claim totals")
    lines.append("")
    lines.append(f"- `UNIFIED EVICT` total: **{total_evict}**")
    lines.append(f"- `UNIFIED KV_CLAIM` total: **{total_kv_claim}**")
    lines.append(f"- `UNIFIED PREFIX_ADDED` total: {prefix_added}")
    lines.append(f"- `UNIFIED PREFIX_REMOVED` total: {prefix_removed}")
    lines.append(f"- `UNIFIED CACHE` snapshots: {cache_lines_seen}")
    if has_lvl2:
        lines.append(f"- `UNIFIED REQUEST` lines (lvl2): {request_lines_seen}")

    lines.append("")
    lines.append("## Direction of pressure (the headline)")
    lines.append("")
    expert_for_kv = evict_kind_cause_tier[("expert", "kv-alloc", "kv-broadcast")] + sum(
        c
        for (k, ca, t), c in evict_kind_cause_tier.items()
        if k == "expert" and ca == "kv-alloc" and t != "kv-broadcast"
    )
    kv_evicts_expert = kv_tier.get("kv-evicts-expert", 0)
    kv_evicts_prefix = kv_tier.get("kv-evicts-prefix", 0)
    kv_truly_free = kv_tier.get("truly-free", 0)
    expert_evicts_kv = sum(
        c
        for (k, ca, _), c in evict_kind_cause_tier.items()
        if k == "kv" and ca != "kv-alloc"
    )
    lines.append(
        f"- **Expert evicted to make room for KV** (`EVICT kind=expert cause=kv-alloc`): {expert_for_kv}"
    )
    lines.append(
        f"- **KV claim that displaced an expert** (`KV_CLAIM tier=kv-evicts-expert`): {kv_evicts_expert}"
    )
    lines.append(
        f"- **KV claim that displaced a prefix** (`KV_CLAIM tier=kv-evicts-prefix`): {kv_evicts_prefix}"
    )
    lines.append(f"- **KV claim of truly-free page** (no eviction): {kv_truly_free}")
    if expert_evicts_kv:
        lines.append(f"- KV evicted for expert (other-kind cases): {expert_evicts_kv}")

    if total_evict:
        lines.append("")
        lines.append("## EVICT breakdown")
        lines.append("")
        lines.append("By kind:")
        for k, c in evict_kind.most_common():
            lines.append(f"- `{k}`: {c} ({fmt_pct(c, total_evict)})")
        lines.append("")
        lines.append("By cause:")
        for c_, n in evict_cause.most_common():
            lines.append(f"- `{c_}`: {n} ({fmt_pct(n, total_evict)})")
        lines.append("")
        lines.append("By tier:")
        for t, n in evict_tier.most_common():
            lines.append(f"- `{t}`: {n} ({fmt_pct(n, total_evict)})")

    if total_kv_claim:
        lines.append("")
        lines.append("## KV_CLAIM breakdown by tier")
        lines.append("")
        for t, n in kv_tier.most_common():
            lines.append(f"- `{t}`: {n} ({fmt_pct(n, total_kv_claim)})")

    lines.append("")
    lines.append("## Expert diversity per layer")
    lines.append("")
    if has_lvl2:
        lines.append(
            "Distinct experts observed in `UNIFIED REQUEST` (full router demand). "
            "Each layer has 64 experts total."
        )
        lines.append("")
        lines.append("| Layer | distinct requested |")
        lines.append("|---|---|")
        for layer in sorted(expert_requested_per_layer):
            lines.append(f"| L{layer} | {len(expert_requested_per_layer[layer])} |")
        if expert_evicted_per_layer:
            lines.append("")
            lines.append("Experts ever evicted per layer (lower bound on churn):")
            lines.append("")
            lines.append("| Layer | distinct evicted |")
            lines.append("|---|---|")
            for layer in sorted(expert_evicted_per_layer):
                lines.append(f"| L{layer} | {len(expert_evicted_per_layer[layer])} |")
    else:
        lines.append(
            "Lvl 1 traces do not log `UNIFIED REQUEST`, so true diversity isn't recoverable. "
            "Reporting **distinct experts ever evicted** per layer (lower bound on the experts "
            "that were live at some point and got displaced)."
        )
        lines.append("")
        lines.append("| Layer | distinct evicted |")
        lines.append("|---|---|")
        for layer in sorted(expert_evicted_per_layer):
            lines.append(f"| L{layer} | {len(expert_evicted_per_layer[layer])} |")

    if global_max is not None and global_min is not None:
        lines.append("")
        lines.append("## Expert-vs-KV swing")
        lines.append("")
        lines.append(
            "Ratio = `expert-ours / (expert-ours + alloc-kv)`. 1.0 = experts dominate, "
            "0.0 = KV dominates. Computed per layer at every `UNIFIED CACHE` snapshot."
        )
        lines.append("")
        r, l, s, eo, kv = global_max
        lines.append(
            f"- **Most expert-heavy moment**: ratio={r:.3f} at L{l} step={s} "
            f"(expert-ours={eo}, alloc-kv={kv})"
        )
        r, l, s, eo, kv = global_min
        lines.append(
            f"- **Most KV-heavy moment**: ratio={r:.3f} at L{l} step={s} "
            f"(expert-ours={eo}, alloc-kv={kv})"
        )
        lines.append("")
        lines.append("Per-layer swing:")
        lines.append("")
        lines.append(
            "| Layer | min ratio (most KV) | max ratio (most expert) | swing |"
        )
        lines.append("|---|---|---|---|")
        for layer in sorted(swing_per_layer):
            entry = swing_per_layer[layer]
            mn = entry["min"]
            mx = entry["max"]
            if mn is None or mx is None:
                continue
            lines.append(
                f"| L{layer} | {mn[0]:.3f} (eo={mn[2]}, kv={mn[3]}) | "
                f"{mx[0]:.3f} (eo={mx[2]}, kv={mx[3]}) | "
                f"{mx[0] - mn[0]:.3f} |"
            )

    if last_cache_per_layer:
        lines.append("")
        lines.append("## Final pool composition (last `UNIFIED CACHE` per layer)")
        lines.append("")
        lines.append("| Layer | step | expert-ours | prefix | alloc-kv |")
        lines.append("|---|---|---|---|---|")
        for layer in sorted(last_cache_per_layer):
            d = last_cache_per_layer[layer]
            lines.append(
                f"| L{layer} | {d['step']} | {d['expert_ours']} | "
                f"{d['prefix']} | {d['alloc_kv']} |"
            )
        avg = lambda key: (
            sum(d[key] for d in last_cache_per_layer.values())
            / len(last_cache_per_layer)
        )
        lines.append("")
        lines.append(
            f"Mean across layers: expert-ours={avg('expert_ours'):.1f}, "
            f"prefix={avg('prefix'):.1f}, alloc-kv={avg('alloc_kv'):.1f}."
        )

    lines.append("")
    return "\n".join(lines)


def main(argv):
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    for arg in argv[1:]:
        path = Path(arg)
        if not path.exists():
            print(f"skip (missing): {path}", file=sys.stderr)
            continue
        out = path.with_suffix(path.suffix + ".summary.md")
        try:
            summary = summarize(path)
        except Exception as exc:
            print(f"error on {path}: {exc}", file=sys.stderr)
            continue
        out.write_text(summary)
        size_in = path.stat().st_size / (1024 * 1024)
        size_out = out.stat().st_size / 1024
        print(f"{path.name}: {size_in:.1f} MB -> {out.name} ({size_out:.1f} KB)")


if __name__ == "__main__":
    main(sys.argv)
