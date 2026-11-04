"""Global fact-graph reasoning on a 10M-token haystack.

The haystack contains thousands of metric records, entities, and directed
relations.  Questions exercise aggregation, filtering, distinct counts,
entity filtering, and a two-hop graph query over the complete document.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer

from deepinfer.longctx_suite import build_document
from deepinfer.retrieval.global_queries import global_answer
from deepinfer.retrieval.global_reasoning import FactGraph


def make_facts(seed: int, metrics: int, entities: int) -> tuple[list[dict], list[dict], dict]:
    rng = random.Random(seed)
    categories = ("alpha", "beta", "gamma", "delta", "epsilon")
    lines: list[dict] = []
    metric_rows: list[tuple[str, int]] = []
    for index in range(metrics):
        category = categories[index % len(categories)]
        value = rng.randrange(1, 101)
        metric_rows.append((category, value))
        lines.append({"text": f"GLOBAL METRIC category={category} value={value}", "group": "metric"})
    for index in range(entities):
        entity_type = "person" if index % 2 == 0 else "org"
        group = ("alpha", "beta", "gamma")[index % 3]
        lines.append({"text": f"ENTITY id=e{index:04d} type={entity_type} group={group}", "group": "entity"})
    for index in range(max(0, entities - 1)):
        lines.append({"text": f"RELATION subject=e{index:04d} relation=owns object=e{index + 1:04d}", "group": "relation"})
        lines.append({"text": f"RELATION subject=e{index:04d} relation=located_in object=e{index + 1:04d}", "group": "relation"})
    for index, line in enumerate(lines):
        line["depth"] = (index + 0.5) / len(lines)
    totals = Counter()
    for category, value in metric_rows:
        totals[category] += value
    expected = {
        "largest": max(totals, key=lambda name: (totals[name], name)),
        "top3": sorted(totals.items(), key=lambda item: (-item[1], item[0]))[:3],
        "alpha_total": totals["alpha"],
        "alpha_total_ge50": sum(value for category, value in metric_rows if category == "alpha" and value >= 50),
        "owns_subjects": entities - 1,
        "person_alpha": sum(1 for index in range(entities) if index % 2 == 0 and index % 3 == 0),
        "two_hop": "e0003",
        "three_hop": "e0004",
    }
    return lines, [{"category": category, "value": value} for category, value in metric_rows], expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokens", type=int, default=10_000_000)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--metrics", type=int, default=2000)
    parser.add_argument("--entities", type=int, default=2000)
    parser.add_argument("--meta", default="/root/meta_coder")
    parser.add_argument("--out-dir", default="/root/di_bench/global_reasoning10m")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.meta, trust_remote_code=True)
    needles, _metric_rows, expected = make_facts(args.seed, args.metrics, args.entities)
    began = time.perf_counter()
    document = build_document(tok, args.tokens, needles, os.path.dirname(os.__file__))
    text = tok.decode(document, skip_special_tokens=True)
    text_path = out_dir / "haystack.txt"
    text_path.write_text(text, encoding="utf-8")
    build_seconds = time.perf_counter() - began

    t = time.perf_counter()
    graph = FactGraph(text)
    graph_seconds = time.perf_counter() - t
    questions = [
        ("largest", "Which category has the largest total value in the GLOBAL METRIC records?",
         f"Answer: {expected['largest']}"),
        ("top3", "List the top 3 categories by total value.",
         "Answer: " + ", ".join(f"{name}={total}" for name, total in expected["top3"])),
        ("alpha_total", "What is the total value in category=alpha?", f"Answer: {expected['alpha_total']}"),
        ("alpha_ge50", "What is the total value in category=alpha, excluding values below 50?",
         f"Answer: {expected['alpha_total_ge50']}"),
        ("owns_subjects", "How many distinct subjects have relation=owns?",
         f"Answer: {expected['owns_subjects']}"),
        ("person_alpha", "How many entities have type=person and group=alpha?",
         f"Answer: {expected['person_alpha']}"),
        ("two_hop", "Which object is reached from subject=e0001 by relation=owns followed by relation=located_in?",
         f"Answer: {expected['two_hop']}"),
        ("three_hop", ("Which object is reached from subject=e0001 by relation=owns followed by relation=located_in "
                       "followed by relation=owns?"), f"Answer: {expected['three_hop']}"),
    ]
    checks = {}
    answers = {}
    for name, question, expected_answer in questions:
        t = time.perf_counter()
        result = global_answer(question, text)
        elapsed = time.perf_counter() - t
        actual = None if result is None else result.answer
        checks[name] = actual == expected_answer
        answers[name] = {
            "question": question,
            "expected": expected_answer,
            "actual": actual,
            "kind": None if result is None else result.kind,
            "metrics": None if result is None else result.metrics,
            "seconds": round(elapsed, 3),
        }
    summary = {
        "tokens": len(document),
        "text_bytes": text_path.stat().st_size,
        "metrics_records": graph.metrics,
        "entities": graph.entities,
        "relations": graph.relations,
        "build_seconds": round(build_seconds, 2),
        "graph_seconds": round(graph_seconds, 2),
        "checks": checks,
        "expected": expected,
    }
    payload = {"summary": summary, "answers": answers}
    out = Path(args.out) if args.out else out_dir / "global_reasoning10m_results.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(f"wrote {out}", flush=True)
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
