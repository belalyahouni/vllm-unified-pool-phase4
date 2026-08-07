"""Generate 10 single-character prefix prompts for the prefix-capacity test.

Each prompt repeats a different character to exactly 3,073 tokens:
two full blocks at block_size=1536 plus one token so block 2 is
not the last block (same reason as in make_alternating_prompts.py).

Run once before the benchmark:
    python make_many_prefixes.py
"""

import json

from transformers import AutoTokenizer

MODEL = "allenai/OLMoE-1B-7B-0924-Instruct"
PREFIX_TOKENS = 3073
NUM_PROMPTS = 10
OUTPUT_LEN = 5
OUT = "/home/belal/150326/prompts/many_prefixes.jsonl"

# 10 distinct characters — each tokenises to a different repeated-token
# pattern, giving us 10 distinct prefixes with low per-prefix expert variety.
CHARS = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
assert len(CHARS) == NUM_PROMPTS, f"need {NUM_PROMPTS} chars, got {len(CHARS)}"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)


def make_char_prefix(ch, n_tokens):
    """Build a string of repeated `ch` that tokenises to exactly n_tokens."""
    raw = ch * 50000
    ids = tok.encode(raw, add_special_tokens=False)
    assert len(ids) >= n_tokens, (
        f"50000×'{ch}' tokenized to only {len(ids)} tokens; bump the multiplier"
    )
    return tok.decode(ids[:n_tokens], skip_special_tokens=True)


with open(OUT, "w") as f:
    for ch in CHARS:
        prompt = make_char_prefix(ch, PREFIX_TOKENS)
        json.dump({"prompt": prompt, "output_tokens": OUTPUT_LEN}, f)
        f.write("\n")

print(f"Wrote {NUM_PROMPTS} distinct-prefix prompts to {OUT}")
# Sanity: confirm each tokenises to the right length and they're distinct
seen_first10 = []
for line in open(OUT):
    p = json.loads(line)["prompt"]
    n = len(tok.encode(p, add_special_tokens=False))
    seen_first10.append(p[:10])
    assert n == PREFIX_TOKENS, f"prompt re-tokenises to {n}, expected {PREFIX_TOKENS}"
assert len(set(seen_first10)) == NUM_PROMPTS, "prompts not distinct"
print(f"  All {NUM_PROMPTS} prompts at exactly {PREFIX_TOKENS} tokens, distinct first-10-chars")
