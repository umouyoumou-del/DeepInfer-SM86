"""Where one locate + re-read round spends its time over a long latent cache.

The cache is filled with random latents - the timing of locate does not depend
on what the keys hold, so no prefill is needed - with the same GPU / pinned-host
split a real run gets.  Timed: ``locate`` for a bare question and for a
question plus a fed-back answer (the question's forward over the whole cache
and the block-mass passes separately), the line-level refine of a re-read
document, and prefill + greedy decode of the answer context.

    python scripts/bench_locate.py --tokens 2097152 --out docs/results/longctx/bench_locate_2m.json
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

from needle_longctx import MODELS, haystack_tokens, header_ids

from deepinfer.moe import deepseek_longctx, latent_reread
from deepinfer.moe.deepseek_longctx import DeepSeekLongContext
from deepinfer.moe.deepseek_v2_runtime import DeepSeekV2LiteGGUFRuntime
from deepinfer.moe.latent_mla import DualChunkPositions, LatentKVCache

_GIB = 2**30
QUESTION = "\n\nWhat is the special magic number for crimson otter?\n\nAssistant:"
NEUTRAL = "\n\nSummarize the text above in one sentence.\n\nAssistant:"


class Clock:
    """Synchronized wall time of wrapped functions, by label."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}

    def wrap(self, owner, name: str, label: str) -> None:
        original = getattr(owner, name)

        def timed(*args, **kwargs):
            torch.cuda.synchronize()
            began = time.perf_counter()
            result = original(*args, **kwargs)
            torch.cuda.synchronize()
            self.seconds[label] = self.seconds.get(label, 0.0) + time.perf_counter() - began
            return result

        setattr(owner, name, timed)

    def take(self) -> dict[str, float]:
        taken, self.seconds = self.seconds, {}
        return {label: round(value, 3) for label, value in taken.items()}


def timed(call):
    torch.cuda.synchronize()
    began = time.perf_counter()
    result = call()
    torch.cuda.synchronize()
    return result, time.perf_counter() - began


def fill(session: DeepSeekLongContext, tokens: int, text: list[int]) -> None:
    """Random latents in every layer; slot tokens from real text, so re-read documents have lines."""

    generator = torch.Generator(device=session.device).manual_seed(0)
    step = 1 << 17
    for layer in range(len(session.layers)):
        for start in range(0, tokens, step):
            count = min(step, tokens - start)
            latent = torch.randn(
                count, session.latent_dim, device=session.device, dtype=torch.float16, generator=generator
            )
            pe = torch.randn(count, session.rope_dim, device=session.device, dtype=torch.float16, generator=generator)
            session.cache.write(layer, start, latent, pe)
    torch.cuda.synchronize()
    session.extents = [[0, tokens, 0]]
    session.next_position = tokens
    session.slot_tokens = (text * (tokens // len(text) + 1))[:tokens]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=sorted(MODELS), default="coder")
    parser.add_argument("--models-dir", default="/root/autodl-tmp/models")
    parser.add_argument("--meta-root", default="/root/autodl-tmp")
    parser.add_argument("--haystack-root", default=os.path.dirname(os.__file__))
    parser.add_argument("--tokens", type=int, default=2097152)
    parser.add_argument("--kv", choices=("fp16", "fp8"), default="fp8")
    parser.add_argument("--workspace-gib", type=float, default=4.0)
    parser.add_argument("--answer-tokens", type=int, default=256, help="fed-back answer length in the long query")
    parser.add_argument("--document", type=int, default=7680, help="re-read document length (32 + 8 blocks, 1 neighbour)")
    parser.add_argument("--decode", type=int, default=200, help="greedy tokens timed in the answer context")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--profile-decode", action="store_true", help="torch.profiler table of three decode steps")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    gguf, meta, trained = MODELS[args.model]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(os.path.join(args.meta_root, meta), trust_remote_code=True)
    text = haystack_tokens(tok, args.haystack_root, 65536)
    rt = DeepSeekV2LiteGGUFRuntime(
        os.path.join(args.models_dir, gguf), meta_dir=os.path.join(args.meta_root, meta),
        device="cuda:0", verbose=False,
    )
    per_token = LatentKVCache.bytes_per_token(27, args.kv)
    capacity = args.tokens + 2048
    free = torch.cuda.mem_get_info()[0]
    gpu_tokens = min(capacity, int(max(0, free - args.workspace_gib * _GIB) // per_token))
    session = DeepSeekLongContext(
        rt, capacity=capacity, gpu_tokens=gpu_tokens, kv_dtype=args.kv,
        positions=DualChunkPositions(trained, trained * 3 // 4, -1), encode_window=98304,
    )
    _, seconds = timed(lambda: fill(session, args.tokens, text))
    print(f"filled {args.tokens:,} tokens ({gpu_tokens:,} on the GPU) in {seconds:.0f} s", flush=True)

    clock = Clock()
    clock.wrap(latent_reread, "block_mass", "block_mass")
    clock.wrap(deepseek_longctx, "prefill_attention", "attention")
    clock.wrap(deepseek_longctx, "moe_prefill", "moe")
    question = tok.encode(QUESTION, add_special_tokens=False)
    neutral = tok.encode(NEUTRAL, add_special_tokens=False)
    head = tok.encode("\n\nWhat is the special magic number for crimson otter?\n", add_special_tokens=False)
    tail = tok.encode("\n\nAssistant:", add_special_tokens=False)
    long_query = head + text[1000 : 1000 + args.answer_tokens] + tail
    report: dict = {
        "tokens": args.tokens, "gpu_tokens": gpu_tokens, "kv": args.kv,
        "gpu": torch.cuda.get_device_name(0), "locate": {},
    }
    for name, ids in (("neutral", neutral), ("question", question), ("question+answer", long_query)):
        runs = []
        for _ in range(args.repeats):
            clock.take()
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            _, seconds = timed(lambda ids=ids: latent_reread.locate(session, ids, reduce="max"))
            parts = clock.take()
            runs.append({
                "seconds": round(seconds, 2), **parts,
                "peak_extra_gib": round((torch.cuda.max_memory_allocated() - before) / _GIB, 2),
            })
        report["locate"][name] = {"query_tokens": len(ids), "runs": runs}
        print(f"locate {name} ({len(ids)} tokens): {runs}", flush=True)

    # refine: a short context over the re-read text, then two per-token locates in it
    header = header_ids(tok)
    document = header + text[5000 : 5000 + args.document]
    refine = latent_reread.Refine(tok, question, neutral, keep=12)
    clock.wrap(DeepSeekLongContext, "prefill", "short_prefill")
    runs = []
    for _ in range(args.repeats):
        clock.take()
        (compact, _kept), seconds = timed(lambda: refine(rt, document, len(header), long_query))
        runs.append({"seconds": round(seconds, 2), **clock.take()})
    report["refine"] = {"document_tokens": len(document), "compact_tokens": len(compact), "runs": runs}
    print(f"refine {len(document)} -> {len(compact)} tokens: {runs}", flush=True)

    # answer: prefill of the compact document, then greedy decode through generate()
    # (graph decode unless DEEPINFER_GRAPH_DECODE=0), no stop id, so every run makes args.decode tokens
    prompt = tok.encode(QUESTION, add_special_tokens=False)
    report["answer"] = {"compact_tokens": len(compact), "decode_tokens": args.decode}
    for mode in ("graph", "steps"):
        os.environ["DEEPINFER_GRAPH_DECODE"] = "1" if mode == "graph" else "0"
        prefill_runs, decode_runs = [], []
        for _ in range(args.repeats):
            short = DeepSeekLongContext(rt, capacity=len(compact) + len(prompt) + args.decode + 8, kv_dtype="fp16")
            _, prefill_seconds = timed(lambda short=short: short.prefill(compact))
            produced, seconds = timed(lambda short=short: short.generate(prompt, max_new_tokens=args.decode))
            prefill_runs.append(round(prefill_seconds, 3))
            decode_runs.append(round(1000 * seconds / len(produced), 1))
        report["answer"][mode] = {"prefill_seconds": prefill_runs, "generate_ms_per_token": decode_runs}
    os.environ.pop("DEEPINFER_GRAPH_DECODE", None)
    print(f"answer: {report['answer']}", flush=True)
    if args.profile_decode:
        short = DeepSeekLongContext(rt, capacity=len(compact) + len(prompt) + 8, kv_dtype="fp16")
        short.prefill(compact)
        logits = short.prefill(prompt, bounded=False)
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                logits = short.decode(int(torch.argmax(logits)))
            torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20, max_name_column_width=60))
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=12, max_name_column_width=60))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
