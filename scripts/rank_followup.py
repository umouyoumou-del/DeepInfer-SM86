"""How well does a follow-up locate query find the next hop of the VT chain?  Ranks under several queries.

Builds the complex suite's document (``needle_complex.py``), prefills it or loads a cache saved with
``--save-cache``, locates the VT question as round 1 does, then for each hop k - an answer naming the
first k variables of the chain - ranks every VAR line's block, among the blocks not read yet, under:

* ``question+final``  question + "Answer: <names>"   (what later rounds add with follow_final)
* ``final``           "Answer: <names>" alone
* ``answer``          question + the last 256 tokens of a recorded round-1 answer (``--answers``, hop 1)

    python scripts/rank_followup.py --tokens 3145728 --answers answers.json --out ranks_3m.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from needle_complex import NEUTRAL_PROMPT, build_document, build_suite
from needle_longctx import MODELS, document_hash

from deepinfer.moe.deepseek_longctx import DeepSeekLongContext
from deepinfer.moe.deepseek_v2_runtime import DeepSeekV2LiteGGUFRuntime
from deepinfer.moe.latent_mla import DualChunkPositions, LatentKVCache
from deepinfer.moe.latent_reread import (
    BLOCK,
    neutral_baseline,
    pick_blocks,
    score_blocks,
)

_GIB = 2**30
CHAIN = ["GHTAY", "OKOSG", "QHUJP", "AVCOU", "INRCW"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--load-cache", default="")
    parser.add_argument("--answers", default="", help="JSON list of recorded round-1 answers")
    parser.add_argument("--tokens", type=int, default=2097152)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--encode-window", type=int, default=98304)
    parser.add_argument("--models-dir", default="/root/autodl-tmp/models")
    parser.add_argument("--meta-root", default="/root/autodl-tmp")
    parser.add_argument("--haystack-root", default=os.path.dirname(os.__file__))
    parser.add_argument("--near-window", type=int, default=98304)
    parser.add_argument("--workspace-gib", type=float, default=4.0)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    gguf, meta, trained = MODELS["coder"]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(os.path.join(args.meta_root, meta), trust_remote_code=True)
    needles, questions = build_suite(args.seed)
    document = build_document(tok, args.tokens, needles, args.haystack_root)
    rt = DeepSeekV2LiteGGUFRuntime(
        os.path.join(args.models_dir, gguf), meta_dir=os.path.join(args.meta_root, meta), device="cuda:0", verbose=False,
    )
    per_token = LatentKVCache.bytes_per_token(27, "fp8")
    capacity = len(document) + 2048
    gpu_tokens = min(capacity, int(max(0, torch.cuda.mem_get_info()[0] - args.workspace_gib * _GIB) // per_token))
    started = time.perf_counter()
    if args.load_cache:
        session, saved = DeepSeekLongContext.load(rt, args.load_cache, capacity=capacity, gpu_tokens=gpu_tokens)
        if saved.get("document_sha256") != document_hash(document):
            raise SystemExit("the cache was built from a different document")
    else:
        session = DeepSeekLongContext(
            rt, capacity=capacity, gpu_tokens=gpu_tokens, kv_dtype="fp8",
            positions=DualChunkPositions(trained, trained * 3 // 4, -1), encode_window=args.encode_window,
        )
        session.prefill(document)
    torch.cuda.synchronize()
    print(f"{len(document):,} tokens ready in {(time.perf_counter() - started) / 60:.1f} min", flush=True)
    baseline = neutral_baseline(session, tok.encode(NEUTRAL_PROMPT, add_special_tokens=False), reduce="max")
    question = next(q for q in questions if q["kind"] == "VT")
    var_lines = sorted((n for n in needles if n["text"].startswith("VAR")), key=lambda n: n["position"])
    near_start = max(0, session.length - args.near_window) // BLOCK

    def blocks_of(needle: dict) -> list[int]:
        return sorted({needle["position"] // BLOCK, (needle["position"] + needle["length"] - 1) // BLOCK})

    def ranks(scores: torch.Tensor, exclude: list[int]) -> dict[str, int]:
        masked = scores.clone()
        if exclude:
            masked[torch.as_tensor(sorted(set(exclude)), device=masked.device)] = float("-inf")
        out = {}
        for needle in var_lines:
            best = max(float(masked[b]) for b in blocks_of(needle))
            out[needle["text"]] = int((masked > best).sum()) if best != float("-inf") else -1
        return out

    report: dict = {"tokens": len(document), "encode_window": args.encode_window, "hops": []}
    bare = tok.encode(f"\n\n{question['question']}\n\nAssistant:", add_special_tokens=False)
    scores = score_blocks(session, bare, reduce="max", baseline=baseline, sink=64)
    chosen = pick_blocks(scores, block=BLOCK, top=32, near_start=near_start)
    report["round1"] = ranks(scores, [])
    print("round 1 ranks:", report["round1"], flush=True)
    head = tok.encode(f"\n\n{question['question']}\n", add_special_tokens=False)
    tail = tok.encode("\n\nAssistant:", add_special_tokens=False)
    by_name = {needle["text"].split()[1]: needle for needle in var_lines}
    recorded = []
    if args.answers:
        with open(args.answers, encoding="utf-8") as handle:
            recorded = json.load(handle)
    for hop in range(1, len(CHAIN)):
        names = ", ".join(CHAIN[:hop])
        found = [b for name in CHAIN[:hop] for b in blocks_of(by_name[name])]
        exclude = chosen + found
        variants = {
            "question+final": head + tok.encode(f"Answer: {names}", add_special_tokens=False) + tail,
            "final": tok.encode(f"\n\nAnswer: {names}", add_special_tokens=False) + tail,
        }
        if hop == 1:
            for number, answer in enumerate(recorded):
                variants[f"answer{number}"] = head + tok.encode(answer, add_special_tokens=False)[-256:] + tail
        row = {"names": names, "next": CHAIN[hop]}
        for name, query in variants.items():
            follow = score_blocks(session, query, reduce="max", baseline=baseline, sink=64)
            rank = ranks(follow, exclude)
            row[name] = rank
            next_line = by_name[CHAIN[hop]]["text"]
            print(f"hop {hop} [{names}] {name:15s} ({len(query)} tokens): next {CHAIN[hop]} rank {rank[next_line]}; "
                  f"all {rank}", flush=True)
        report["hops"].append(row)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
