"""The 39-needle long-context suite against another engine's HTTP server (llama.cpp or vLLM).

Builds the same document as ``scripts/needle_complex.py`` (same tokenizer, haystack, needles and
seed, so the same token ids and document hash) and sends every question as exact token ids -
``document + "\n\n<question> <instruction>\n\nAssistant:"`` - greedily, then scores the text after
the last ``Answer:`` with the suite's own ``score``.  The VT question is also asked the suite's
other two ways and the names are united, as DeepInfer does.  The first request carries the whole
document; both servers keep it as a prefix cache, so later questions only add their own tokens.

    python scripts/longctx_baseline.py --engine llama.cpp --url http://127.0.0.1:8090 --tokens 131072
    python scripts/longctx_baseline.py --engine vllm --url http://127.0.0.1:8000 --model coder --tokens 131072
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepinfer.longctx_suite import build_document, build_suite, document_hash, score
from deepinfer.moe.latent_reread import final_answer


def complete(engine: str, url: str, model: str, ids: list[int], max_new: int, timeout: float) -> dict:
    """Greedy completion of ``ids``: ``{"text", "prompt_tokens", "prompt_seconds", "new_tokens", "seconds"}``."""

    if engine == "llama.cpp":
        body = {"prompt": ids, "n_predict": max_new, "temperature": 0.0, "top_k": 1, "cache_prompt": True}
        path = "/completion"
    else:
        body = {"model": model, "prompt": ids, "max_tokens": max_new, "temperature": 0.0}
        path = "/v1/completions"
    request = urllib.request.Request(url.rstrip("/") + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
    began = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        reply = json.loads(response.read())
    seconds = time.perf_counter() - began
    if engine == "llama.cpp":
        timings = reply.get("timings", {})
        return {"text": reply["content"], "prompt_tokens": timings.get("prompt_n"),
                "prompt_seconds": (timings.get("prompt_ms") or 0) / 1000, "new_tokens": timings.get("predicted_n"),
                "seconds": seconds}
    usage = reply.get("usage", {})
    return {"text": reply["choices"][0]["text"], "prompt_tokens": usage.get("prompt_tokens"),
            "prompt_seconds": None, "new_tokens": usage.get("completion_tokens"), "seconds": seconds}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engine", choices=("llama.cpp", "vllm"), required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="coder", help="the served model name (vLLM)")
    parser.add_argument("--meta", default="/root/autodl-tmp/meta_coder", help="tokenizer folder")
    parser.add_argument("--haystack-root", default=os.path.dirname(os.__file__))
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-new", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=6 * 3600, help="seconds for one request (the first prefills)")
    parser.add_argument("--kinds", default="MK,MV,MQ,VT,2H")
    parser.add_argument("--label", default="", help="what was served, for the record (version, flags)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.meta, trust_remote_code=True)
    needles, questions = build_suite(args.seed)
    document = build_document(tok, args.tokens, needles, args.haystack_root)
    kinds = set(args.kinds.split(","))
    questions = [q for q in questions if q["kind"] in kinds]
    record = {
        "engine": args.engine, "label": args.label, "tokens": len(document), "seed": args.seed,
        "document_sha256": document_hash(document), "max_new": args.max_new, "results": [],
    }
    print(f"{args.engine}: document {len(document):,} tokens, sha256 {record['document_sha256'][:16]}", flush=True)

    def save() -> None:
        Path(args.out).write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")

    for number, question in enumerate(questions):
        asks = [question["instruction"], *question.get("variants", [])]
        replies = []
        for instruction in asks:
            suffix = tok.encode(f"\n\n{question['question']} {instruction}\n\nAssistant:", add_special_tokens=False)
            try:
                replies.append(complete(args.engine, args.url, args.model, document + suffix, args.max_new, args.timeout))
            except Exception as error:  # noqa: BLE001 - an engine that cannot serve this length: record why
                record["error"] = f"{type(error).__name__}: {error}"
                if hasattr(error, "read"):
                    record["error"] += " | " + error.read().decode("utf-8", "replace")[:2000]
                print(f"request failed: {record['error'][:400]}", flush=True)
                save()
                return
        answer = replies[0]["text"].strip()
        if len(replies) > 1:  # the VT names of the three phrasings, united as in needle_complex
            picked = set()
            for reply in replies:
                picked |= set(re.findall(r"\b[A-Z]{5}\b", final_answer(reply["text"])))
            answer += "\n\n[union of 3 phrasings]\nAnswer: " + ", ".join(sorted(picked))
        result = score(question, answer)
        record["results"].append({
            "kind": question["kind"], "question": question["question"], "expected": question["expected"],
            "answer": answer, **result, "requests": replies,
            "seconds": round(sum(reply["seconds"] for reply in replies), 1),
        })
        if number == 0:
            record["first_request_seconds"] = round(replies[0]["seconds"], 1)
            record["prefill"] = {key: replies[0][key] for key in ("prompt_tokens", "prompt_seconds")}
        print(f"{number + 1:>2} {question['kind']:<2} {'FULL' if result['full'] else 'MISS'} "
              f"{len(result['hits'])}/{len(question['expected'])} ({record['results'][-1]['seconds']:.0f} s) "
              f"{question['question'][:50]} -> {final_answer(answer)[:80]!r}", flush=True)
        save()
    rows = record["results"]
    record["summary"] = {
        "questions_full": f"{sum(r['full'] for r in rows)}/{len(rows)}",
        "items": f"{sum(len(r['hits']) for r in rows)}/{sum(len(r['expected']) for r in rows)}",
        "by_kind": {kind: f"{sum(r['full'] for r in rows if r['kind'] == kind)}/{sum(r['kind'] == kind for r in rows)}"
                    for kind in dict.fromkeys(r["kind"] for r in rows)},
        "question_seconds": round(sum(r["seconds"] for r in rows), 1),
    }
    save()
    print(json.dumps(record["summary"]), flush=True)


if __name__ == "__main__":
    main()
