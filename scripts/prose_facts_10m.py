"""10M unlabeled-prose map-reduce fact extraction benchmark.

The rule map stage reads every passage, the reduce stage deduplicates triples,
and the shared FactGraph answers prose questions.  A model map backend is
available separately for the same interface.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer

from deepinfer.longctx_suite import build_document
from deepinfer.retrieval.fact_extraction import (
    RulesMapReduceExtractor,
    merge_into_graph,
)
from deepinfer.retrieval.global_queries import global_answer
from deepinfer.retrieval.global_reasoning import FactGraph


def make_needles(groups: int) -> tuple[list[dict], dict]:
    needles: list[dict] = []
    for index in range(1, groups + 1):
        acme = f"Acme{index:04d}"
        beta = f"Beta{index:04d}"
        alice = f"Alice{index:04d}"
        city = f"Berlin{index:04d}"
        employees = 1200 + index
        needles.extend([
            {"text": f"{acme} owns {beta}.", "group": "owns"},
            {"text": f"{beta} is located in {city}.", "group": "located"},
            {"text": f"{alice} works for {acme}.", "group": "works"},
            {"text": f"{acme} has {employees:,} employees.", "group": "employees"},
        ])
    for index, needle in enumerate(needles):
        needle["depth"] = (index + 0.5) / len(needles)
    return needles, {"acme": "Acme0001", "beta": "Beta0001", "alice": "Alice0001",
                     "city": "Berlin0001", "employees": 1201}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokens", type=int, default=10_000_000)
    parser.add_argument("--groups", type=int, default=1000)
    parser.add_argument("--meta", default="/root/meta_coder")
    parser.add_argument("--out-dir", default="/root/di_bench/prose10m")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.meta, trust_remote_code=True)
    needles, expected = make_needles(args.groups)
    began = time.perf_counter()
    document = build_document(tok, args.tokens, needles, os.path.dirname(os.__file__))
    text = tok.decode(document, skip_special_tokens=True)
    text_path = out_dir / "haystack.txt"
    text_path.write_text(text, encoding="utf-8")
    build_seconds = time.perf_counter() - began

    t = time.perf_counter()
    extractor = RulesMapReduceExtractor(max_chars=4000)
    triples = extractor.run(text)
    extract_seconds = time.perf_counter() - t
    t = time.perf_counter()
    graph = FactGraph(text)
    merged = merge_into_graph(graph, triples)
    graph_seconds = time.perf_counter() - t

    questions = [
        ("owns", "Who owns Beta0001?", f"Answer: {expected['acme']}"),
        ("located", "Where is Beta0001 located?", f"Answer: {expected['city']}"),
        ("works", "Who works for Acme0001?", f"Answer: {expected['alice']}"),
        ("employees", "How many employees does Acme0001 have?", f"Answer: {expected['employees']}"),
    ]
    checks = {}
    answers = {}
    for name, question, expected_answer in questions:
        t = time.perf_counter()
        result = global_answer(question, text, graph=graph)
        elapsed = time.perf_counter() - t
        actual = None if result is None else result.answer
        checks[name] = actual == expected_answer
        answers[name] = {"question": question, "expected": expected_answer, "actual": actual,
                         "kind": None if result is None else result.kind, "seconds": round(elapsed, 3)}
    summary = {
        "tokens": len(document),
        "text_bytes": text_path.stat().st_size,
        "prose_fact_lines": len(needles),
        "map_reduce_triples": len(triples),
        "merged_triples": merged,
        "graph_relations": graph.relations,
        "graph_prose_triples": graph.prose_triples,
        "build_seconds": round(build_seconds, 2),
        "extract_seconds": round(extract_seconds, 2),
        "graph_seconds": round(graph_seconds, 2),
        "checks": checks,
    }
    out = Path(args.out) if args.out else out_dir / "prose10m_results.json"
    out.write_text(json.dumps({"summary": summary, "answers": answers}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(f"wrote {out}", flush=True)
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
