"""Generate repeat-pattern prompts for the static-sweep gradient test.

N distinct prompts, each 3073 tokens: all "a"-tokens except for a single "f"
token at sliding position i (i = 0..N-1). Each prompt occupies 2 unique
cacheable prefix blocks at block_size=1536:
    block 1 = tokens 0..1535 (contains the unique "f")
    block 2 = tokens 1536..3071 (all "a"; same content across prompts but each
              has a unique hash because it chains off a unique block 1)
    block 3 = token 3072 (partial terminal, not cacheable)

Expert variety stays minimal: only the "a" and "f" tokens are activated, so the
KV pressure variable is isolated from any expert-cache effects.

Pass-1 access: forward order 0..N-1 (warm-up).
Pass-2 access: reverse order N-1..0 (measured). With reverse access under LRU,
the K most-recently-cached prompts (those still resident at end of pass 1) are
hit first → no thrashing → hit rate degrades smoothly with cache budget.

Outputs two JSONL files:
    repeat_prompts.jsonl         — forward order
    repeat_prompts_reverse.jsonl — reverse order
"""

import json

from transformers import AutoTokenizer

MODEL = "allenai/OLMoE-1B-7B-0924-Instruct"
PREFIX_TOKENS = 3073
NUM_PROMPTS = 30
OUTPUT_LEN = 20
OUT_FWD = "/home/belal/150326/prompts/repeat_prompts.jsonl"
OUT_REV = "/home/belal/150326/prompts/repeat_prompts_reverse.jsonl"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)


def get_a_token_sequence(n_tokens):
    raw = "a" * 50000
    ids = tok.encode(raw, add_special_tokens=False)
    assert len(ids) >= n_tokens, f"50000×'a' → only {len(ids)} tokens"
    return ids[:n_tokens]


def get_f_token():
    ids = tok.encode("f", add_special_tokens=False)
    assert len(ids) == 1, f"'f' tokenized to {len(ids)} tokens: {ids}"
    return ids[0]


a_ids = get_a_token_sequence(PREFIX_TOKENS)
f_id = get_f_token()
print(f"a-token sequence: {len(a_ids)} tokens, first 5 = {a_ids[:5]}")
print(f"f-token: {f_id} (decodes to {tok.decode([f_id])!r})")


def build_prompt(i):
    """Prompt i: a-tokens with position i replaced by f-token."""
    ids = list(a_ids)
    ids[i] = f_id
    text = tok.decode(ids, skip_special_tokens=True)
    re_ids = tok.encode(text, add_special_tokens=False)
    return text, ids, re_ids


BLOCK_SIZE = 1536
MIN_LEN = 2 * BLOCK_SIZE + 1  # need block 2 full → cacheable
MAX_LEN = 3 * BLOCK_SIZE      # must still fit in 3 blocks

prompts = []
for i in range(NUM_PROMPTS):
    text, intended_ids, re_ids = build_prompt(i)
    prompts.append((i, text, intended_ids, re_ids))

seen = set()
for i, _, _, re_ids in prompts:
    key = tuple(re_ids)
    assert key not in seen, f"prompt[{i}] not distinct from a prior prompt"
    seen.add(key)
    assert MIN_LEN <= len(re_ids) <= MAX_LEN, (
        f"prompt[{i}] re-encoded to {len(re_ids)} tokens, "
        f"need {MIN_LEN}..{MAX_LEN} for block-1+block-2 cacheable in 3 blocks"
    )

# Sanity: the f-token (or its boundary triplet [23342, 39639, 2320]) must land
# in block 1 of every prompt so each prompt's block 1 hashes uniquely.
for i, _, _, re_ids in prompts:
    block1 = re_ids[:BLOCK_SIZE]
    assert any(t != re_ids[0] for t in block1), (
        f"prompt[{i}] block 1 is uniform — f position outside block 1?"
    )

lengths = sorted({len(p[3]) for p in prompts})
print(f"All {NUM_PROMPTS} prompts distinct. Token lengths after round-trip: "
      f"{lengths}")

with open(OUT_FWD, "w") as f:
    for _, text, _, _ in prompts:
        json.dump({"prompt": text, "output_tokens": OUTPUT_LEN}, f)
        f.write("\n")

with open(OUT_REV, "w") as f:
    for _, text, _, _ in reversed(prompts):
        json.dump({"prompt": text, "output_tokens": OUTPUT_LEN}, f)
        f.write("\n")

print(f"Wrote {NUM_PROMPTS} prompts → {OUT_FWD}")
print(f"Wrote {NUM_PROMPTS} prompts (reversed) → {OUT_REV}")
