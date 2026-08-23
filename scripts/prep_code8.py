"""Build 8 real-code prompts of ~3072 tokens for the KV-heavy experiment.

Corpus is the OLMoE section 5.3 code text; the sha256 is printed so a run
can be tied to the exact input (expected
1a8e4b8eb4d4c67b994e1b74ddbd0b93d84fe3572eb32ed86b9c56185c7f6872).

Prompts are decoded from disjoint token chunks, so the 8 prompts share no
prefix and each one costs full KV -- which is what creates the KV pressure
the unified pool has to resolve against expert residency.
"""

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source",
        default="/workspace/code_bias_ab/github_oss_with_stack_texts.txt",
    )
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--num-prompts", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=3072)
    ap.add_argument("--out-dir", default="/root")
    args = ap.parse_args()

    src = Path(args.source)
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok(src.read_text(), truncation=False)["input_ids"]
    need = args.num_prompts * args.tokens
    if len(ids) < need:
        raise SystemExit(f"corpus has {len(ids)} tokens, need {need}")
    chunks = [
        ids[i * args.tokens : (i + 1) * args.tokens] for i in range(args.num_prompts)
    ]

    def write(path: Path, ordered) -> None:
        lengths = []
        with path.open("w") as out:
            for chunk in ordered:
                prompt = tok.decode(chunk, skip_special_tokens=False)
                # Decode/re-encode is not identity; assert we stay inside the
                # 4096 RoPE limit OLMoE caps at.
                n = len(tok(prompt, truncation=False)["input_ids"])
                if n > 4095:
                    raise SystemExit(f"prompt retokenized to {n} tokens")
                lengths.append(n)
                out.write(json.dumps({"prompt": prompt}) + "\n")
        print(path.name, lengths)

    out_dir = Path(args.out_dir)
    write(out_dir / "code8_fwd.jsonl", chunks)
    write(out_dir / "code8_rev.jsonl", list(reversed(chunks)))
    print("sha256", hashlib.sha256(src.read_bytes()).hexdigest())
    print("corpus_tokens", len(ids))


if __name__ == "__main__":
    main()
