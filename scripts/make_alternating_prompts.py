"""Generate alternating-prefix prompts (JSONL) for scenario A.

Two repeated-character prefixes of PREFIX_TOKENS tokens, alternated
across NUM_PROMPTS requests. No random suffix; each prompt is just
the prefix. A pure single-character prompt only routes to ~11-20 of
64 experts per layer in our probe, so the workload exercises the
prefix-vs-expert tradeoff cleanly.

OLMoE's BPE makes the per-character token count uneven (e.g. "a"*N
tokenises differently to "b"*N), so we over-generate at the
character level and then truncate at the token level to land both
prefixes at exactly PREFIX_TOKENS.

PREFIX_TOKENS is set to one token past 2 * block_size (= 3072) on
purpose. At exactly 3072 tokens the last block of the prompt is full
block 2, and vLLM does not promote a prompt's terminal block to the
prefix cache, so the hit rate caps around 50%. Adding a single token
makes block 3 the terminal block, so block 2 is no longer last and
both prefix blocks end up cached.

Run once before the benchmark:
    python make_alternating_prompts.py
"""

import json

from transformers import AutoTokenizer

MODEL = "allenai/OLMoE-1B-7B-0924-Instruct"
PREFIX_TOKENS = 3073
NUM_PROMPTS = 20
OUTPUT_LEN = 20
OUT = "/home/belal/150326/prompts/alternating_prompts.jsonl"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)


def make_char_prefix(ch, n_tokens):
    """Build a string of repeated `ch` that tokenizes to exactly n_tokens.

    Oversize first (50k chars is enough for any single-char repeat to exceed
    n_tokens with the OLMoE tokenizer), then truncate the token list.
    """
    raw = ch * 50000
    ids = tok.encode(raw, add_special_tokens=False)
    assert len(ids) >= n_tokens, (
        f"50000×'{ch}' tokenized to only {len(ids)} tokens; bump the multiplier"
    )
    return tok.decode(ids[:n_tokens], skip_special_tokens=True)


prefixes = [
    make_char_prefix("a", PREFIX_TOKENS),
    make_char_prefix("b", PREFIX_TOKENS),
]

with open(OUT, "w") as f:
    for i in range(NUM_PROMPTS):
        prompt = prefixes[i % 2]
        json.dump({"prompt": prompt, "output_tokens": OUTPUT_LEN}, f)
        f.write("\n")

print(f"Wrote {NUM_PROMPTS} alternating prompts to {OUT}")
