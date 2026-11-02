"""Keep one long session alive and run experiment snippets against it.

Prefills (or loads) the complex suite's document once, then watches a directory: each ``*.py``
dropped there runs with the session in scope and its output lands next to it as ``.out``.  A 3M
prefill takes 25 minutes and its cache does not fit on the test host's disks; an experiment on the
live session takes seconds, the whole suite about ten minutes.  A file named ``stop`` ends it.

In scope: ``rt tok session document needles questions baseline args`` (the harness's answering
options as a SimpleNamespace), ``nc`` (scripts/needle_complex.py) and ``lr``
(deepinfer.moe.latent_reread); snippets may ``importlib.reload`` them after code changes.

    python scripts/longctx_workbench.py --tokens 3145728 --watch /root/autodl-tmp/bench_jobs
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import importlib
import io
import os
import sys
import time
import traceback
from types import SimpleNamespace

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import needle_complex as nc
from needle_longctx import MODELS, document_hash

from deepinfer.moe import latent_reread as lr
from deepinfer.moe.deepseek_longctx import DeepSeekLongContext
from deepinfer.moe.deepseek_v2_runtime import DeepSeekV2LiteGGUFRuntime
from deepinfer.moe.latent_mla import DualChunkPositions, LatentKVCache

_GIB = 2**30


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokens", type=int, default=3145728)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--encode-window", type=int, default=98304)
    parser.add_argument("--load-cache", default="")
    parser.add_argument("--watch", required=True)
    parser.add_argument("--workspace-gib", type=float, default=4.0)
    parser.add_argument("--models-dir", default="/root/autodl-tmp/models")
    parser.add_argument("--meta-root", default="/root/autodl-tmp")
    parser.add_argument("--haystack-root", default=os.path.dirname(os.__file__))
    cli = parser.parse_args()

    gguf, meta, trained = MODELS["coder"]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(os.path.join(cli.meta_root, meta), trust_remote_code=True)
    needles, questions = nc.build_suite(cli.seed)
    document = nc.build_document(tok, cli.tokens, needles, cli.haystack_root)
    rt = DeepSeekV2LiteGGUFRuntime(
        os.path.join(cli.models_dir, gguf), meta_dir=os.path.join(cli.meta_root, meta), device="cuda:0", verbose=False,
    )
    per_token = LatentKVCache.bytes_per_token(27, "fp8")
    capacity = len(document) + 2048
    gpu_tokens = min(capacity, int(max(0, torch.cuda.mem_get_info()[0] - cli.workspace_gib * _GIB) // per_token))
    started = time.perf_counter()
    if cli.load_cache:
        session, saved = DeepSeekLongContext.load(rt, cli.load_cache, capacity=capacity, gpu_tokens=gpu_tokens)
        if saved.get("document_sha256") != document_hash(document):
            raise SystemExit("the cache was built from a different document")
    else:
        session = DeepSeekLongContext(
            rt, capacity=capacity, gpu_tokens=gpu_tokens, kv_dtype="fp8",
            positions=DualChunkPositions(trained, trained * 3 // 4, -1), encode_window=cli.encode_window,
        )
        session.prefill(document)
    torch.cuda.synchronize()
    print(f"{len(document):,} tokens ready in {(time.perf_counter() - started) / 60:.1f} min", flush=True)
    baseline = lr.neutral_baseline(session, tok.encode(nc.NEUTRAL_PROMPT, add_special_tokens=False), reduce="max")
    args = SimpleNamespace(
        score="max-contrast", mode="feedback", refine=12, order="document", top=32, neighbors=1, max_new=1024,
        follow_top=8, rounds=8, near_window=98304, follow_final=32, final_with_question=False, final_entity_rows=False,
    )
    scope = {
        "rt": rt, "tok": tok, "session": session, "document": document, "needles": needles, "questions": questions,
        "baseline": baseline, "args": args, "nc": nc, "lr": lr, "torch": torch, "importlib": importlib,
    }
    os.makedirs(cli.watch, exist_ok=True)
    print(f"watching {cli.watch}", flush=True)
    while not os.path.exists(os.path.join(cli.watch, "stop")):
        for path in sorted(glob.glob(os.path.join(cli.watch, "*.py"))):
            with open(path, encoding="utf-8") as handle:
                code = handle.read()
            os.remove(path)
            output = io.StringIO()
            began = time.perf_counter()
            with contextlib.redirect_stdout(output):
                try:
                    exec(compile(code, path, "exec"), scope)  # noqa: S102 - our own experiment snippets
                except Exception:  # noqa: BLE001
                    traceback.print_exc(file=output)
            output.write(f"\n[done in {time.perf_counter() - began:.1f} s]\n")
            with open(path[:-3] + ".out", "w", encoding="utf-8") as handle:
                handle.write(output.getvalue())
            print(f"ran {os.path.basename(path)} in {time.perf_counter() - began:.1f} s", flush=True)
        time.sleep(2)


if __name__ == "__main__":
    main()
