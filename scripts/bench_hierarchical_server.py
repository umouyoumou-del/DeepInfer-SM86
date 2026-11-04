"""Benchmark the hierarchical server's /ask endpoint with concurrent clients.

The server has one GPU worker (or a pure structured-fact fast path), so this
measures the production request queue rather than pretending to be GPU batch
decode.  All suite questions are sent with their evaluation instructions.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deepinfer.longctx_suite import build_suite, score


def ask(base: str, key: str, question: dict) -> tuple[dict, float, dict | None]:
    body = json.dumps({
        "question": question["question"], "instruction": question["instruction"], "deep": True,
    }).encode()
    request = urllib.request.Request(
        f"{base.rstrip('/')}/ask?key={urllib.parse.quote(key)}",
        data=body, method="POST", headers={"Content-Type": "application/json"},
    )
    began = time.perf_counter()
    done = None
    with urllib.request.urlopen(request, timeout=1800) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            if event.get("event") == "done":
                done = event
                break
    return question, time.perf_counter() - began, done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:6007")
    parser.add_argument("--key", default="")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    _, questions = build_suite(args.seed)
    began = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda question: ask(args.url, args.key, question), questions))
    wall = time.perf_counter() - began
    results = []
    full = items = expected = 0
    latencies = []
    for question, seconds, done in rows:
        answer = "" if done is None else done.get("answer", "")
        scored = score(question, answer)
        full += int(scored["full"])
        items += len(scored["hits"])
        expected += len(question["expected"])
        latencies.append(seconds)
        results.append({
            "kind": question["kind"], "question": question["question"], "seconds": round(seconds, 3),
            "final": None if done is None else done.get("final"),
            "retrieval_ms": None if done is None else done.get("retrieval_ms"),
            "first_token": None if done is None else done.get("first_token"),
            "full": scored["full"], "hits": scored["hits"], "expected": question["expected"],
        })
        print(f"{question['kind']:<2} {'FULL' if scored['full'] else 'MISS'} "
              f"{len(scored['hits'])}/{len(question['expected'])} {seconds:.3f}s", flush=True)
    latencies.sort()
    summary = {
        "full": full, "questions": len(rows), "items": items, "expected": expected,
        "wall_seconds": round(wall, 3), "workers": args.workers,
        "p50_seconds": round(statistics.median(latencies), 3),
        "p95_seconds": round(latencies[int(0.95 * (len(latencies) - 1))], 3),
        "max_seconds": round(max(latencies), 3),
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
