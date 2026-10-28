"""Build a reproducible DeepSeek full-category quality regression suite.

The suite intentionally samples a fixed budget from standard datasets rather
than pretending a 4 GB GPU can run the entire MMLU/GSM8K test split.  MMLU
covers every subject, GSM8K uses a fixed random sample, long-chain problems are
generated deterministically, and perplexity uses a fixed WikiText-2 slice.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_META = Path(r"E:\DeepInfer-MoE\meta")
DEFAULT_MMLU = ROOT / ".tmp_quality" / "mmlu_test"
DEFAULT_GSM8K = ROOT / ".tmp_quality" / "gsm8k_test.jsonl"
DEFAULT_WIKITEXT = ROOT / ".tmp_quality" / "wikitext2_valid.txt"
DEFAULT_OUTPUT = ROOT / "docs" / "quality" / "deepseek_v2_lite_full_quality_suite.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmlu-dir", type=Path, default=DEFAULT_MMLU)
    parser.add_argument("--gsm8k-jsonl", type=Path, default=DEFAULT_GSM8K)
    parser.add_argument("--wikitext", type=Path, default=DEFAULT_WIKITEXT)
    parser.add_argument("--meta-dir", type=Path, default=DEFAULT_META)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mmlu-per-subject", type=int, default=1)
    parser.add_argument("--gsm8k-limit", type=int, default=32)
    parser.add_argument("--long-chain-count", type=int, default=32)
    parser.add_argument("--ppl-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261028)
    return parser.parse_args()


def load_mmlu(directory: Path, per_subject: int, seed: int) -> tuple[list[dict], dict]:
    items: list[dict] = []
    source_counts: dict[str, int] = {}
    files = sorted(directory.glob("*_test.csv"))
    if not files:
        raise FileNotFoundError(f"no MMLU CSV files under {directory}")
    for file_idx, path in enumerate(files):
        subject = path.name.removesuffix("_test.csv")
        rows: list[dict] = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row_idx, row in enumerate(csv.reader(handle)):
                if len(row) < 6:
                    continue
                question, a, b, c, d, answer = row[:6]
                answer = answer.strip().upper()
                if answer not in {"A", "B", "C", "D"}:
                    continue
                rows.append(
                    {
                        "id": f"mmlu_{subject}_{row_idx:04d}",
                        "subject": subject,
                        "question": question,
                        "choices": [a, b, c, d],
                        "answer": answer,
                    }
                )
        if not rows:
            continue
        source_counts[subject] = len(rows)
        rng = random.Random(seed + file_idx * 1009)
        take = min(max(0, per_subject), len(rows))
        items.extend(rng.sample(rows, take))
    return items, source_counts


def load_gsm8k(path: Path, limit: int, seed: int) -> tuple[list[dict], int]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            answer_text = str(item.get("answer", ""))
            if "####" not in answer_text:
                continue
            answer = answer_text.rsplit("####", 1)[1].strip().replace(",", "")
            rows.append(
                {
                    "id": f"gsm8k_{len(rows):04d}",
                    "question": item["question"],
                    "answer": answer,
                }
            )
    rng = random.Random(seed)
    take = min(max(0, limit), len(rows))
    return rng.sample(rows, take), len(rows)


def make_long_chain(count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    items: list[dict] = []
    for index in range(count):
        value = rng.randint(3, 30)
        parts = [f"Start with {value}."]
        for _ in range(5):
            operation = rng.choice(("add", "subtract", "multiply"))
            operand = rng.randint(2, 9)
            if operation == "add":
                value += operand
                parts.append(f"Add {operand}.")
            elif operation == "subtract":
                operand = min(operand, max(1, value - 1))
                value -= operand
                parts.append(f"Subtract {operand}.")
            else:
                value *= operand
                parts.append(f"Multiply by {operand}.")
        divisors = [divisor for divisor in range(2, 7) if value % divisor == 0]
        if not divisors:
            value *= 2
            parts.append("Multiply by 2.")
            divisors = [2]
        divisor = rng.choice(divisors)
        value //= divisor
        parts.append(f"Divide by {divisor}.")
        items.append(
            {
                "id": f"long_chain_{index:04d}",
                "question": " ".join(parts) + " What is the final integer result?",
                "answer": str(value),
                "steps": len(parts),
            }
        )
    return items


def token_slice(tokenizer, path: Path, max_tokens: int) -> tuple[str, int]:
    text = path.read_text(encoding="utf-8")
    encoded = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    ids = encoded["input_ids"][0][:max_tokens]
    sliced = tokenizer.decode(ids, skip_special_tokens=True)
    return sliced, int(ids.numel())


def main() -> int:
    args = parse_args()
    mmlu, mmlu_counts = load_mmlu(
        args.mmlu_dir, args.mmlu_per_subject, args.seed
    )
    gsm8k, gsm8k_total = load_gsm8k(
        args.gsm8k_jsonl, args.gsm8k_limit, args.seed
    )
    long_chain = make_long_chain(args.long_chain_count, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.meta_dir, trust_remote_code=True)
    ppl_text, ppl_tokens = token_slice(
        tokenizer, args.wikitext, args.ppl_tokens
    )
    suite = {
        "name": "DeepSeek-V2-Lite Q4_K_M full-category regression",
        "description": (
            "One fixed sample per MMLU subject, a fixed GSM8K sample, "
            "deterministic arithmetic long-chain problems, and a fixed "
            "WikiText-2 perplexity slice.  This is a practical regression "
            "budget, not the full 14k MMLU / 1.3k GSM8K test splits."
        ),
        "seed": args.seed,
        "sources": {
            "mmlu": "https://github.com/hengzzzhou/STS/tree/main/data/mmlu/test",
            "gsm8k": "https://github.com/openai/grade-school-math/blob/master/grade_school_math/data/test.jsonl",
            "wikitext": "https://github.com/pytorch/examples/blob/main/word_language_model/data/wikitext-2/valid.txt",
        },
        "sampling": {
            "mmlu": "random sample per subject with seed",
            "gsm8k": "random sample from the official test split with seed",
            "long_chain": "deterministic synthetic 5-7 operation integer chains",
            "perplexity": "first N tokens of WikiText-2 validation",
        },
        "counts": {
            "mmlu_sampled": len(mmlu),
            "mmlu_subjects": len(mmlu_counts),
            "mmlu_source_per_subject": mmlu_counts,
            "gsm8k_sampled": len(gsm8k),
            "gsm8k_source_total": gsm8k_total,
            "long_chain_sampled": len(long_chain),
            "perplexity_tokens": ppl_tokens,
        },
        "mmlu": mmlu,
        "gsm8k": gsm8k,
        "long_chain": long_chain,
        "perplexity": {
            "id": "wikitext2_valid_slice",
            "source": str(args.wikitext),
            "tokens": ppl_tokens,
            "text": ppl_text,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(suite, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"wrote {args.output} | mmlu={len(mmlu)}/{len(mmlu_counts)} subjects "
        f"gsm8k={len(gsm8k)}/{gsm8k_total} long_chain={len(long_chain)} "
        f"ppl_tokens={ppl_tokens}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
