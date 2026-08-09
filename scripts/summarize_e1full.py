#!/usr/bin/env python3
import json, os, statistics, re, glob
R = "/workspace/e1full_results"
L = "/workspace/e1full_logs"

def load(p):
    try: return json.load(open(p))
    except Exception: return None

def final_experts(tag):
    p = f"{L}/{tag}_kv_boot.log"
    if not os.path.exists(p): return "--"
    last = None
    with open(p) as f:
        for line in f:
            if "UNIFIED CACHE L0 " in line:
                last = line
    if not last: return "--"
    m = re.search(r"expert-ours-sb=(\d+).*ever-activated=(\d+)", last)
    return f"{m.group(1)}e/{m.group(2)}act" if m else "--"

order = ["vanilla", "static32", "static48", "static64", "ours8", "ours32", "ours48", "ours64"]
label = {"vanilla":"vanilla (no-offload)","static32":"static C=32 (KV-oracle)",
         "static48":"static C=48 (mid)","static64":"static C=64 (expert-oracle)",
         "ours8":"ours init=8","ours32":"ours init=32","ours48":"ours init=48","ours64":"ours init=64"}

print("="*84)
print("EXPERIMENT 1 (M=67) — KV-heavy (warm reverse pass); TTFT is the signal")
print("="*84)
print(f"{'config':<28}{'med TTFT':>11}{'mean TTFT':>11}{'p99 TTFT':>11}{'hits/16':>9}{'pool(L0)':>12}")
for c in order:
    d = load(f"{R}/{c}_kv_warm.json")
    if not d: print(f"{label[c]:<28}{'(missing)':>11}"); continue
    tt = d.get("ttfts") or []
    h = sum(1 for x in tt if x < 1)
    pe = final_experts(c) if c.startswith("ours") else "--"
    def ms(k): v=d.get(k); return f"{v/1000:.2f}s" if v and v>=1000 else (f"{v:.0f}ms" if v is not None else "--")
    print(f"{label[c]:<28}{ms('median_ttft_ms'):>11}{ms('mean_ttft_ms'):>11}{ms('p99_ttft_ms'):>11}{str(h)+'/16':>9}{pe:>12}")

print()
print("="*84)
print("EXPERIMENT 1 (M=67) — expert-heavy (random, seeds 1-3); TPOT is the signal (thrash=high)")
print("="*84)
print(f"{'config':<28}{'med TPOT':>11}{'med TTFT':>11}{'tok/s':>10}")
for c in order:
    tp, tf, th = [], [], []
    for s in (1,2,3):
        d = load(f"{R}/{c}_exp_seed{s}.json")
        if d:
            if d.get("median_tpot_ms") is not None: tp.append(d["median_tpot_ms"])
            if d.get("median_ttft_ms") is not None: tf.append(d["median_ttft_ms"])
            if d.get("total_token_throughput") is not None: th.append(d["total_token_throughput"])
    if not tp: print(f"{label[c]:<28}{'(missing)':>11}"); continue
    print(f"{label[c]:<28}{statistics.mean(tp):>9.1f}ms{statistics.mean(tf):>9.0f}ms{statistics.mean(th):>10.1f}")

print()
print("Cold-start convergence (ours, KV-heavy): warm median vs init (same endpoint, rising transient)")
for c in ["ours8","ours32","ours48","ours64"]:
    d = load(f"{R}/{c}_kv_warm.json");
    if d: print(f"  init={c[4:]:>2}: warm median {d.get('median_ttft_ms',0):.0f}ms, final pool {final_experts(c)}")
