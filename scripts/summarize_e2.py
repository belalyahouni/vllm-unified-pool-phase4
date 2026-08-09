#!/usr/bin/env python3
import json, os
R = "/workspace/e2_results"

def d(p):
    try: return json.load(open(p))
    except Exception: return None

print("=== E2 badness (warm+expert); cold shown separately ===")
hdr = ("config", "order", "cold", "kvwarm", "hits", "exp", "eTPOTms", "BADNESS", "total")
print("%-11s%-9s%8s%9s%7s%9s%9s%10s%9s" % hdr)
rows = {}
for c in ["static32", "static48", "static64", "ours48nt"]:
    for o in ["kv2exp", "exp2kv"]:
        kc = d(f"{R}/{c}_{o}_kvcold.json"); kw = d(f"{R}/{c}_{o}_kvwarm.json"); ex = d(f"{R}/{c}_{o}_exp.json")
        if not (kw and ex):
            print("%-11s%-9s (missing)" % (c, o)); continue
        cold = kc.get("duration", 0) if kc else 0
        warm = kw.get("duration", 0); expd = ex.get("duration", 0)
        wh = sum(1 for x in (kw.get("ttfts") or []) if x < 1)
        etp = ex.get("median_tpot_ms", -1)
        bad = warm + expd
        rows.setdefault(c, []).append(bad)
        print("%-11s%-9s%7.0fs%8.0fs%7s%8.0fs%9.1f%9.0fs%8.0fs"
              % (c, o, cold, warm, f"{wh}/16", expd, etp, bad, cold + warm + expd))

print("\n=== badness summary (mean of both orders) ===")
lab = {"static32": "static C=32 (extreme)", "static48": "static C=48 (mid)",
       "static64": "static C=64 (extreme)", "ours48nt": "ours init=48"}
for c in ["static32", "static64", "static48", "ours48nt"]:
    if c in rows:
        m = sum(rows[c]) / len(rows[c])
        print(f"  {lab[c]:<24} {m:.0f} s")
