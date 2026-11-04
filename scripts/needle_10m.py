"""A 10M-token needle-in-a-haystack with exact retrieval verification.

Builds a 10M-token Python-stdlib haystack with N unique 7-digit facts at
different depths, indexes it with the production FTS5 retriever, and asks one
question per needle.  An answer is correct only when the needle line was in the
retrieved chunks and the exact value was extracted from those chunks.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer

from deepinfer.longctx_suite import build_document
from deepinfer.retrieval.hybrid import (
    build_index,
    facts_from_lines,
    retrieve,
    structured_answer,
)


def make_needles(count: int, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    values = rng.sample(range(1_000_000, 10_000_000), count)
    needles: list[dict] = []
    questions: list[dict] = []
    for index, value in enumerate(values):
        key = f"needle {index:03d}"
        text = f"The special magic number for {key} is: {value}."
        depth = (index + 0.5) / count + rng.uniform(-0.35 / count, 0.35 / count)
        needles.append({"text": text, "depth": min(0.99, max(0.01, depth)), "key": key, "value": str(value)})
        questions.append({"question": f"What is the special magic number for {key}?", "needle": index})
    return needles, questions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokens", type=int, default=10_000_000)
    parser.add_argument("--needles", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--meta", default="/root/meta_coder")
    parser.add_argument("--out-dir", default="/root/di_bench/needle10m")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.meta, trust_remote_code=True)
    needles, questions = make_needles(args.needles, args.seed)
    began = time.perf_counter()
    document = build_document(tok, args.tokens, needles, os.path.dirname(os.__file__))
    text = tok.decode(document, skip_special_tokens=True)
    text_path = out_dir / "haystack.txt"
    text_path.write_text(text, encoding="utf-8")
    build_seconds = time.perf_counter() - began
    db_path = out_dir / "index.sqlite"
    index_began = time.perf_counter()
    chunks = build_index(text, db_path)
    index_seconds = time.perf_counter() - index_began

    import sqlite3

    con = sqlite3.connect(db_path)
    results = []
    retrieve_total = 0.0
    answer_total = 0.0
    for item, question in zip(needles, questions, strict=True):
        t = time.perf_counter()
        rows = retrieve(con, question["question"], args.top_k)
        retrieve_total += time.perf_counter() - t
        bodies = [body for _, body in rows]
        t = time.perf_counter()
        answer = structured_answer(question["question"], facts_from_lines(bodies), text)
        answer_total += time.perf_counter() - t
        hit = any(item["text"] in body for body in bodies)
        results.append({
            **item,
            "question": question["question"],
            "retrieved": hit,
            "answer": answer,
            "correct": hit and answer == item["value"],
            "chunks": len(rows),
        })
        if not results[-1]["correct"]:
            print(f"MISS {item['key']} expected={item['value']} answer={answer!r} retrieved={hit}", flush=True)
    con.close()
    correct = sum(result["correct"] for result in results)
    summary = {
        "tokens": len(document),
        "needles": len(results),
        "correct": correct,
        "retrieval_hits": sum(result["retrieved"] for result in results),
        "chunks": chunks,
        "top_k": args.top_k,
        "build_seconds": round(build_seconds, 2),
        "index_seconds": round(index_seconds, 3),
        "retrieve_total_seconds": round(retrieve_total, 3),
        "answer_total_seconds": round(answer_total, 3),
    }
    payload = {"summary": summary, "results": results}
    out = Path(args.out) if args.out else out_dir / "needle10m_results.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(f"wrote {out}", flush=True)
    if correct != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
