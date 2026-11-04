"""Validate full-document global queries on a 10M-token haystack.

Three query families that top-k retrieval cannot answer reliably:

1. exact literal count over every token;
2. extractive summary with per-depth coverage;
3. aggregation over every structured metric record.
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
from deepinfer.retrieval.global_queries import (
    aggregate_metrics,
    count_occurrences,
    extractive_summary,
    parse_metric_records,
)


def make_needles(seed: int, markers: int = 137, metrics: int = 100) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    needles: list[dict] = []
    categories = ("alpha", "beta", "gamma", "delta", "epsilon")
    totals = {category: 0 for category in categories}
    for index in range(markers):
        needles.append({"text": f"GLOBAL_MARKER_X occurrence {index:03d}.", "group": "marker"})
    for index in range(metrics):
        category = categories[index % len(categories)]
        value = 10 + (index * 7) % 41
        totals[category] += value
        needles.append({"text": f"GLOBAL METRIC category={category} value={value}", "group": "metric"})
    themes = (
        "THEME_ALPHA: the system prefers low-latency retrieval and incremental indexing.",
        "THEME_BETA: the serving layer keeps a persistent evidence index for warm starts.",
        "THEME_GAMMA: global counts scan the complete text instead of sampling top-k chunks.",
        "THEME_DELTA: summaries explicitly cover depth buckets across the whole document.",
        "THEME_EPSILON: structured facts are aggregated deterministically over every record.",
    )
    for index, theme in enumerate(themes):
        needles.append({"text": theme, "group": "theme", "depth_hint": (index + 0.5) / len(themes)})
    # Spread all needles across the full depth range, then sort by depth for insertion.
    for index, needle in enumerate(needles):
        if "depth_hint" in needle:
            needle["depth"] = needle["depth_hint"]
            continue
        base = (index + 0.5) / len(needles)
        jitter = rng.uniform(-0.35 / len(needles), 0.35 / len(needles))
        needle["depth"] = min(0.99, max(0.01, base + jitter))
    needles.sort(key=lambda item: item["depth"])
    return needles, {"categories": list(categories), "totals": totals}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokens", type=int, default=10_000_000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--markers", type=int, default=137)
    parser.add_argument("--metrics", type=int, default=100)
    parser.add_argument("--meta", default="/root/meta_coder")
    parser.add_argument("--out-dir", default="/root/di_bench/global10m")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.meta, trust_remote_code=True)
    needles, expected = make_needles(args.seed, args.markers, args.metrics)
    began = time.perf_counter()
    document = build_document(tok, args.tokens, needles, os.path.dirname(os.__file__))
    text = tok.decode(document, skip_special_tokens=True)
    text_path = out_dir / "haystack.txt"
    text_path.write_text(text, encoding="utf-8")
    build_seconds = time.perf_counter() - began

    count_question = "How many times does GLOBAL_MARKER_X occur in the entire document?"
    summary_question = "Summarize the main themes of this 10M-token document."
    compare_question = ("Compare the total value by category in the GLOBAL METRIC records. "
                        "Report the totals and which category has the largest total.")

    t = time.perf_counter()
    counted = count_occurrences(text, "GLOBAL_MARKER_X")
    count_seconds = time.perf_counter() - t

    t = time.perf_counter()
    summary_result = extractive_summary(text, max_chars=12_000, buckets=16)
    summary_seconds = time.perf_counter() - t

    t = time.perf_counter()
    metrics = aggregate_metrics(text)
    compare_seconds = time.perf_counter() - t

    records = parse_metric_records(text)
    theme_names = [f"THEME_{name}" for name in ("ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON")]
    checks = {
        "count": counted["count"] == args.markers and counted["count"] == text.count("GLOBAL_MARKER_X"),
        "summary": all(theme in summary_result.answer for theme in theme_names),
        "compare": metrics is not None
        and metrics.metrics["totals"] == expected["totals"]
        and metrics.metrics["largest"] == max(expected["totals"], key=lambda name: expected["totals"][name]),
    }
    summary_stats = {
        "tokens": len(document),
        "text_bytes": text_path.stat().st_size,
        "markers_expected": args.markers,
        "markers_counted": counted["count"],
        "metrics_records": len(records),
        "categories": expected["totals"],
        "checks": checks,
        "build_seconds": round(build_seconds, 2),
        "count_seconds": round(count_seconds, 3),
        "summary_seconds": round(summary_seconds, 3),
        "compare_seconds": round(compare_seconds, 3),
        "summary_selected": summary_result.metrics.get("selected"),
        "summary_buckets": summary_result.metrics.get("buckets"),
    }
    payload = {
        "summary": summary_stats,
        "questions": {
            "count": {"question": count_question, "answer": counted},
            "summary": {"question": summary_question, "answer": summary_result.answer, "metrics": summary_result.metrics},
            "compare": {"question": compare_question, "answer": None if metrics is None else metrics.answer,
                        "metrics": None if metrics is None else metrics.metrics},
        },
    }
    out = Path(args.out) if args.out else out_dir / "global10m_results.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary_stats, ensure_ascii=False), flush=True)
    print(f"wrote {out}", flush=True)
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
