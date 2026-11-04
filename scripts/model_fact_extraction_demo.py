"""Small proof of the model-backed map-reduce prose extractor.

This deliberately uses a short passage so the expensive model map phase can be
validated independently from the 10M rule-based full-corpus benchmark.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TEXT = """Acme Corporation owns Beta Labs.
Beta Labs is located in Berlin.
Alice Chen works for Acme Corporation.
Acme Corporation has 1,200 employees.
Beta Labs was founded by Dana Lee.
Dana Lee reports to Alice Chen.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="/root/models/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf")
    parser.add_argument("--meta", default="/root/meta_coder")
    parser.add_argument("--chunk-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-chunks", type=int, default=1)
    parser.add_argument("--out", default="/root/di_bench/model_fact_extraction.json")
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer

    from deepinfer.moe.deepseek_v2_runtime import DeepSeekV2LiteGGUFRuntime
    from deepinfer.retrieval.fact_extraction import (
        ModelMapReduceExtractor,
        merge_into_graph,
    )
    from deepinfer.retrieval.global_reasoning import FactGraph

    tok = AutoTokenizer.from_pretrained(args.meta, trust_remote_code=True)
    rt = DeepSeekV2LiteGGUFRuntime(args.model, meta_dir=args.meta, device="cuda:0", verbose=False)
    extractor = ModelMapReduceExtractor(
        rt, tok, max_new_tokens=args.max_new_tokens, chunk_tokens=args.chunk_tokens
    )
    began = time.perf_counter()
    with torch.inference_mode():
        triples = extractor.run(TEXT, max_chunks=args.max_chunks)
    seconds = time.perf_counter() - began
    graph = FactGraph(TEXT)
    merged = merge_into_graph(graph, triples)
    checks = {
        "owns": graph.inverse_query("Beta Labs", "owns") == ["Acme Corporation"],
        "located": graph.query("Beta Labs", "located_in") == ["Berlin"],
        "works_for": graph.inverse_query("Acme Corporation", "works_for") == ["Alice Chen"],
        "employees": graph.query("Acme Corporation", "has_employees") == ["1200"],
    }
    payload = {
        "seconds": round(seconds, 2),
        "triples": triples,
        "merged": merged,
        "graph_relations": graph.relations,
        "checks": checks,
    }
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
