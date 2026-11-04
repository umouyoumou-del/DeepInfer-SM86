"""Build and persist a FactGraph from structured text + map-reduce prose triples.

Examples:

    python scripts/build_fact_graph.py --text haystack.txt --out facts.sqlite --backend rules
    python scripts/build_fact_graph.py --text prose.txt --out facts.sqlite --backend model \
      --model /root/models/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deepinfer.retrieval.fact_extraction import (
    ModelMapReduceExtractor,
    RulesMapReduceExtractor,
    merge_into_graph,
)
from deepinfer.retrieval.global_reasoning import FactGraph


def load_triples(path: str) -> list[dict]:
    if not path:
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("triples", [])
    return [item for item in data if isinstance(item, dict)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--text", required=True)
    parser.add_argument("--out", required=True, help="SQLite fact graph to write")
    parser.add_argument("--backend", choices=("rules", "model", "none"), default="rules")
    parser.add_argument("--triples", default="", help="optional JSON list of already-extracted triples")
    parser.add_argument("--model", default="/root/models/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf")
    parser.add_argument("--meta", default="/root/meta_coder")
    parser.add_argument("--max-chars", type=int, default=4000, help="rules map chunk size")
    parser.add_argument("--chunk-tokens", type=int, default=2048, help="model map chunk size")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-chunks", type=int, default=0, help="0 means all chunks (model backend)")
    args = parser.parse_args()

    text = Path(args.text).read_text(encoding="utf-8")
    began = time.perf_counter()
    triples = load_triples(args.triples)
    if args.backend == "rules":
        triples.extend(RulesMapReduceExtractor(max_chars=args.max_chars).run(text))
    elif args.backend == "model":
        import torch
        from transformers import AutoTokenizer

        from deepinfer.moe.deepseek_v2_runtime import DeepSeekV2LiteGGUFRuntime

        tok = AutoTokenizer.from_pretrained(args.meta, trust_remote_code=True)
        rt = DeepSeekV2LiteGGUFRuntime(args.model, meta_dir=args.meta, device="cuda:0", verbose=False)
        extractor = ModelMapReduceExtractor(
            rt, tok, max_new_tokens=args.max_new_tokens, chunk_tokens=args.chunk_tokens
        )
        with torch.inference_mode():
            triples.extend(extractor.run(text, max_chunks=args.max_chunks or None))
    triples = list({(item.get("subject"), item.get("relation"), item.get("object")): item
                    for item in triples}.values())
    graph = FactGraph(text)
    merged = merge_into_graph(graph, triples)
    graph.save(args.out)
    summary = {
        "text_bytes": Path(args.text).stat().st_size,
        "triples": len(triples),
        "merged": merged,
        "metrics": graph.metrics,
        "entities": graph.entities,
        "relations": graph.relations,
        "prose_triples": graph.prose_triples,
        "seconds": round(time.perf_counter() - began, 2),
        "out": str(Path(args.out).resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
