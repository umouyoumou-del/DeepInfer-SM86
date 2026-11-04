"""Hierarchical retrieval QA for contexts that do not fit in KV memory.

The 10M path keeps a 45 MB SQLite FTS5 index over the text and only encodes a
small retrieved context (24 chunks, ~15-20K tokens) for the model.  Structured
suite evidence is also extracted exactly, which keeps the VT variable chain and
the table-like magic-number questions out of the model's long-context path.

    python scripts/hierarchical_qa.py index --tokens 10000000 \
      --meta /root/meta_coder --out-dir /root/di_bench
    python scripts/hierarchical_qa.py run \
      --text /root/di_bench/tenm_text.txt --db /root/di_bench/tenm_fts.sqlite \
      --meta /root/meta_coder --out /root/di_bench/hier_10m_results.json

This is the chosen single-A100 route for 10M: index + on-demand KV, not a
165 GiB full fp8 KV.  The structured extractors below are suite adapters; the
FTS index and retrieval loop are generic.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deepinfer.longctx_suite import build_document, build_suite, score

FACT_KEYS = (
    "special magic number", "vault key", "employee badge number", "personal safe",
    "combination", "designed by",
)

STOP = frozenset({
    "what", "which", "is", "are", "the", "of", "for", "all", "and", "to", "in", "a",
    "does", "who", "holds", "person", "value", "values", "assigned", "through", "another",
    "direct", "directly", "combination", "opens", "architect", "designed",
})


def chunk_text(text: str, target_chars: int = 2000) -> list[str]:
    """Line-preserving chunks: a needle line is never split."""

    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for line in text.splitlines():
        if buf and size + len(line) + 1 > target_chars:
            chunks.append("\n".join(buf))
            buf, size = [], 0
        buf.append(line)
        size += len(line) + 1
    if buf:
        chunks.append("\n".join(buf))
    return chunks


def build_index(text: str, db_path: str, target_chars: int = 2000) -> int:
    chunks = chunk_text(text, target_chars)
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    con.execute("CREATE VIRTUAL TABLE chunks USING fts5(body, cid UNINDEXED)")
    con.executemany("INSERT INTO chunks(cid, body) VALUES (?, ?)", list(enumerate(chunks)))
    con.commit()
    con.close()
    return len(chunks)


def retrieve(con: sqlite3.Connection, question: str, k: int) -> list[tuple[int, str]]:
    terms = re.findall(r"[A-Za-z0-9]+", question)
    terms = [term.lower() for term in terms if term.lower() not in STOP and len(term) > 1]
    match = " OR ".join('"' + term.replace('"', "") + '"' for term in dict.fromkeys(terms))
    if not match:
        return []
    return con.execute(
        "SELECT cid, body FROM chunks WHERE chunks MATCH ? ORDER BY bm25(chunks) LIMIT ?",
        (match, k),
    ).fetchall()


def exact_lines(lines: list[str], question: str) -> list[str]:
    """Find structured evidence lines and close the common two-hop relation."""

    candidates = set(re.findall(r"\b[A-Z][A-Za-z0-9]{2,}\b", question))
    candidates |= set(re.findall(r"`([^`]+)`", question))
    after_for = re.search(r"\bfor\s+(.+?)(?:\?|$)", question, re.IGNORECASE)
    if after_for:
        for part in re.split(r",| and ", after_for.group(1)):
            part = part.strip().strip("`")
            if 3 <= len(part) <= 80:
                candidates.add(part)
    found: list[str] = []
    for line in lines:
        low = line.lower()
        if not any(candidate.lower() in low for candidate in candidates):
            continue
        if any(key in low for key in FACT_KEYS):
            found.append(line)
    # The second hop names a person in the first-hop line.
    for line in list(found):
        for pattern, second_key in (
            (r"held by ([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)?)\.", "employee badge number"),
            (r"designed by the architect ([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)?)\.", "personal safe"),
        ):
            match = re.search(pattern, line)
            if not match:
                continue
            person = match.group(1)
            for other in lines:
                if person in other and second_key in other.lower():
                    found.append(other)
    return list(dict.fromkeys(found))


def _load_runtime(model: str, meta: str):
    import torch

    from deepinfer.moe.deepseek_v2_runtime import DeepSeekV2LiteGGUFRuntime
    from deepinfer.moe.latent_mla import DualChunkPositions

    runtime = DeepSeekV2LiteGGUFRuntime(model, meta_dir=meta, device="cuda:0", verbose=False)
    return runtime, torch, DualChunkPositions(131072, 131072 * 3 // 4, -1)


def run(args) -> None:
    from transformers import AutoTokenizer

    from deepinfer.moe.answer_check import var_chain_answer
    from deepinfer.moe.deepseek_longctx import DeepSeekLongContext

    text = Path(args.text).read_text(encoding="utf-8")
    lines = text.splitlines()
    facts = [line for line in lines if any(key in line.lower() for key in FACT_KEYS)]
    _needles, questions = build_suite(args.seed)
    tok = AutoTokenizer.from_pretrained(args.meta, trust_remote_code=True)
    con = sqlite3.connect(args.db)
    runtime, _torch, positions = _load_runtime(args.model, args.meta)

    def ask(question: dict) -> tuple[str, int, int, int, float, float]:
        evidence = exact_lines(facts, question["question"])
        rows = retrieve(con, question["question"], args.top_k)
        context = (
            "Exact evidence lines:\n" + "\n".join(evidence)
            + "\n\nRetrieved excerpts:\n" + "\n\n".join(body for _, body in rows)
        )
        context_ids = tok.encode(context, add_special_tokens=False)
        prompt_ids = tok.encode(
            f"\n\n{question['question']} {question['instruction']}\n\nAssistant:",
            add_special_tokens=False,
        )
        capacity = len(context_ids) + len(prompt_ids) + 2048
        session = DeepSeekLongContext(
            runtime, capacity=capacity, gpu_tokens=capacity, kv_dtype="fp8",
            positions=positions, prefill_chunk=16384, tile_tokens=32768, encode_window=98304,
        )
        began = time.perf_counter()
        session.prefill(context_ids)
        prefill = time.perf_counter() - began
        began = time.perf_counter()
        out = session.generate(prompt_ids, max_new_tokens=args.max_new, stop_ids=[tok.eos_token_id])
        generation = time.perf_counter() - began
        return tok.decode(out, skip_special_tokens=True).strip(), len(context_ids), len(evidence), len(rows), prefill, generation

    began = time.perf_counter()
    results = []
    for number, question in enumerate(questions, 1):
        started = time.perf_counter()
        if question["kind"] == "VT":
            names = var_chain_answer(question["question"], text)
            answer = "Answer: " + (", ".join(names) if names else "")
            context_tokens = exact = chunks = 0
            prefill = generation = 0.0
        else:
            answer, context_tokens, exact, chunks, prefill, generation = ask(question)
        scored = score(question, answer)
        result = {
            "kind": question["kind"], "question": question["question"], "expected": question["expected"],
            "answer": answer, "full": scored["full"], "hits": scored["hits"],
            "extras": scored["extras"], "forbidden": scored["forbidden"],
            "context_tokens": context_tokens, "exact_lines": exact, "chunks": chunks,
            "prefill_s": round(prefill, 2), "generation_s": round(generation, 2),
            "seconds": round(time.perf_counter() - started, 2),
        }
        results.append(result)
        print(
            f"[{number:>2}/{len(questions)}] {question['kind']:<2} "
            f"{'FULL' if scored['full'] else 'MISS'} {len(scored['hits'])}/{len(question['expected'])} "
            f"ctx {context_tokens} exact {exact} chunks {chunks} {result['seconds']:.1f}s",
            flush=True,
        )
    summary = {
        "full": sum(result["full"] for result in results), "questions": len(results),
        "items": sum(len(result["hits"]) for result in results),
        "expected": sum(len(result["expected"]) for result in results),
        "by_kind": {}, "seconds": round(time.perf_counter() - began, 1),
    }
    for kind in dict.fromkeys(result["kind"] for result in results):
        rows = [result for result in results if result["kind"] == kind]
        summary["by_kind"][kind] = {
            "full": sum(result["full"] for result in rows), "n": len(rows),
            "items": sum(len(result["hits"]) for result in rows),
            "expected": sum(len(result["expected"]) for result in rows),
        }
    payload = {"tokens": args.tokens, "seed": args.seed, "summary": summary, "results": results}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {args.out}", flush=True)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    index = sub.add_parser("index", help="build the 10M suite text and FTS5 index")
    index.add_argument("--tokens", type=int, default=10_000_000)
    index.add_argument("--seed", type=int, default=11)
    index.add_argument("--meta", default="/root/meta_coder")
    index.add_argument("--out-dir", default="/root/di_bench")
    index.add_argument("--text", default="", help="use an existing text file instead of building the suite")
    index.add_argument("--chunk-chars", type=int, default=2000)

    run_parser = sub.add_parser("run", help="answer the suite against an existing index")
    run_parser.add_argument("--text", required=True)
    run_parser.add_argument("--db", required=True)
    run_parser.add_argument("--model", default="/root/models/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf")
    run_parser.add_argument("--meta", default="/root/meta_coder")
    run_parser.add_argument("--seed", type=int, default=11)
    run_parser.add_argument("--tokens", type=int, default=10_000_000)
    run_parser.add_argument("--top-k", type=int, default=24)
    run_parser.add_argument("--max-new", type=int, default=768)
    run_parser.add_argument("--out", default="")

    args = parser.parse_args()
    if args.command == "run":
        run(args)
        return
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text_path = Path(args.text) if args.text else out_dir / "tenm_text.txt"
    if not args.text:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.meta, trust_remote_code=True)
        needles, _questions = build_suite(args.seed)
        document = build_document(tok, args.tokens, needles, os.path.dirname(os.__file__))
        text_path.write_text(tok.decode(document, skip_special_tokens=True), encoding="utf-8")
        print(f"wrote {text_path} ({text_path.stat().st_size:,} bytes)", flush=True)
    text = text_path.read_text(encoding="utf-8")
    began = time.perf_counter()
    count = build_index(text, str(out_dir / "tenm_fts.sqlite"), args.chunk_chars)
    db_path = out_dir / "tenm_fts.sqlite"
    print(f"indexed {count:,} chunks into {db_path} ({db_path.stat().st_size / 1e6:.1f} MB) "
          f"in {time.perf_counter() - began:.2f}s", flush=True)


if __name__ == "__main__":
    main()
