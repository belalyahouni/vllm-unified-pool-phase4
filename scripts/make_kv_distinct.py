"""KV-heavy, expert-light workload for Phase 4 E1.

16 prompts, each a long run of a DISTINCT single character. Each character run
tokenizes to that character's run-length tokens (distinct token ids per char), so
the 12 prompts occupy separate cacheable KV (no prefix sharing), while a repeated
single character routes to a fixed ~8 experts at any length (no positional drift).
The 12 chosen characters have a measured cumulative expert union of ~29 (<=32),
verified by running them back-to-back on the L4 (see GPU_VALIDATION). This keeps
the pool at ~29 experts + ~24 sb KV = ~53/66 sb -> no over-subscription -> the
unified pool holds all 12 stably (contrast: the old sliding-`f` design unioned to
46-63 experts and over-subscribed the pool).

Forward file = order below (cold pass). Reverse file = reversed (measured warm
pass; reverse access avoids LRU thrash -> clean hit/miss cliff).
"""
import os, json
os.environ.setdefault("HF_HOME", "/workspace/hf")
from transformers import AutoTokenizer

MODEL = "allenai/OLMoE-1B-7B-0924-Instruct"
CHARS = ["a", "A", "C", "e", "o", "c", "r", "n", "u", "h",
         "6", "2", "7", "f", "v", "q"]  # 16 chars, measured input union ~29 (<=32)
TARGET_TOKENS = 3071  # prompt + 1 output = 3072 tokens = exactly 192 pages = 2 super-blocks
                      # (so static C=64's 2 sb KV admits exactly one; drops the inherited
                      #  Phase-3 "+1 sealer" token, which the output token now handles)
OUT_FWD = os.environ.get("OUT_FWD", "/workspace/kv_distinct_fwd.jsonl")
OUT_REV = os.environ.get("OUT_REV", "/workspace/kv_distinct_rev.jsonl")

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

prompts = []
for ch in CHARS:
    ids = tok.encode(ch * 40000, add_special_tokens=False)[:TARGET_TOKENS]
    prompts.append((ch, tok.decode(ids)))

seen = set()
for ch, text in prompts:
    re_ids = tok.encode(text, add_special_tokens=False)
    key = tuple(re_ids)
    assert key not in seen, f"prompt {ch!r} not distinct from a prior prompt"
    seen.add(key)
    print(f"  {ch!r}: {len(re_ids)} tokens")

with open(OUT_FWD, "w") as f:
    for ch, text in prompts:
        f.write(json.dumps({"prompt": text, "char": ch, "output_tokens": 1}) + "\n")
with open(OUT_REV, "w") as f:
    for ch, text in reversed(prompts):
        f.write(json.dumps({"prompt": text, "char": ch, "output_tokens": 1}) + "\n")

print(f"wrote {len(prompts)} distinct-char prompts -> {OUT_FWD} + {OUT_REV}")
