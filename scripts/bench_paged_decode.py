"""Decode cost at long context on the paged serve path: whole model vs one layer's kernel.

docs/HANDOFF.md reported "128K context, int8 KV, window 4096 + sink 4: 9.2 ms, 108.78 tok/s".
That figure came from timing one paged-attention call for layer 0 (``make_views(...)[0]``),
once, without warm-up.  A decode step runs that kernel in all 27 layers and the MoE on top.
This script measures both on the same pool, after warm-up:

* ``model``: ``forward_logits`` for one new token with ``past`` tokens already in the pool -
  every layer, router, experts, lm_head - which is what tok/s means;
* ``kernel``: one layer's paged MLA attention alone (``DeepSeekPagedCache.attend``), the
  quantity the 9.2 ms measured.

Pool contents do not affect cost (the kernels read every addressed slot either way), so the
pool is left as allocated instead of prefilling 128K tokens.

    python scripts/bench_paged_decode.py --past 131072 --kv int8 --configs w4096s4,w8192s4,w16384s4,full
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepinfer.engine.kv_cache import PagedKVCacheManager
from deepinfer.moe.deepseek_v2_runtime import DeepSeekV2LiteGGUFRuntime
from deepinfer.moe.paged_adapter import DeepSeekPagedCache

BLOCK = 16


def parse(spec: str) -> tuple[int | None, int | None]:
    if spec == "full":
        return None, None
    window, _, sink = spec[1:].partition("s")
    return int(window), int(sink)


def timed(fn, warmup: int, steps: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(steps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model", default="/root/autodl-tmp/models/DeepSeek-V2-Lite-Chat-Q4_K_M.gguf"
    )
    parser.add_argument("--meta", default="/root/autodl-tmp/meta")
    parser.add_argument("--past", type=int, default=131072)
    parser.add_argument(
        "--short",
        type=int,
        default=256,
        help="short-context baseline for the whole model",
    )
    parser.add_argument("--kv", choices=("fp16", "int8"), default="int8")
    parser.add_argument("--configs", default="w4096s4,w8192s4,w16384s4,full")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--kernel-steps", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-cache", type=int, default=12288)
    parser.add_argument(
        "--pin",
        choices=("auto", "yes", "no"),
        default="auto",
        help="experts resident on the GPU (auto: if they fit) or streamed from host memory",
    )
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    rt = DeepSeekV2LiteGGUFRuntime(
        args.model,
        meta_dir=args.meta,
        device=args.device,
        max_cache=args.max_cache,
        pin_all_experts={"auto": None, "yes": True, "no": False}[args.pin],
        verbose=False,
    )
    weights_gib = torch.cuda.memory_allocated(args.device) / 2**30
    cfg = rt.config
    key_dim = int(cfg.qk_nope_head_dim) + int(cfg.qk_rope_head_dim)
    value_dim = int(cfg.v_head_dim)
    layers, heads = int(cfg.num_hidden_layers), int(cfg.num_attention_heads)
    capacity = args.past + args.warmup + args.steps + 2 * BLOCK
    mgr = PagedKVCacheManager(
        num_layers=layers,
        num_kv_heads=heads,
        head_dim=max(key_dim, value_dim),
        block_size=BLOCK,
        num_blocks=(capacity + BLOCK - 1) // BLOCK + 2,
        dtype=torch.float16,
        kv_cache_dtype=args.kv,
        device=args.device,
    )
    blocks = mgr.allocate_tokens(capacity)
    cap = len(blocks) * BLOCK
    pool_gib = mgr.memory_bytes / 2**30
    free, total = torch.cuda.mem_get_info(args.device)
    print(
        f"model {weights_gib:.2f} GiB (experts resident: {bool(rt.pin_active)}), pool {args.kv} {pool_gib:.2f} GiB "
        f"for {cap} tokens; GPU {total / 2**30:.2f} GiB, {free / 2**30:.2f} free",
        flush=True,
    )
    token = torch.tensor([[int(rt.tokenizer.eos_token_id)]], device=args.device)

    def model_step(start: int):
        state = {"pos": start}

        def step():
            views = mgr.make_views(
                [blocks], [state["pos"]], max_length=cap, new_tokens=1
            )
            pos = torch.tensor([[state["pos"]]], device=args.device)
            rt.forward_logits(token, pos, kv_cache=views)
            state["pos"] += 1

        return step

    result = {
        "script": "scripts/bench_paged_decode.py",
        "gpu": torch.cuda.get_device_name(args.device),
        "torch": torch.__version__,
        "kv": args.kv,
        "past": args.past,
        "pool_gib": round(pool_gib, 3),
        "model_gib": round(weights_gib, 3),
        "experts_resident": bool(rt.pin_active),
        "free_gib_after_pool": round(free / 2**30, 3),
        "layers": layers,
        "runs": [],
    }
    rt.paged_sliding_window, rt.paged_attention_sink = None, None
    short = timed(model_step(args.short), args.warmup, args.steps)
    result["short_context"] = {
        "past": args.short,
        "model_ms_median": round(statistics.median(short), 2),
    }
    print(
        f"short context past={args.short}: model {statistics.median(short):.1f} ms/token "
        f"({1000 / statistics.median(short):.1f} tok/s)",
        flush=True,
    )

    for spec in args.configs.split(","):
        window, sink = parse(spec)
        rt.paged_sliding_window, rt.paged_attention_sink = window, sink
        model = timed(model_step(args.past), args.warmup, args.steps)
        views = mgr.make_views([blocks], [args.past], max_length=cap, new_tokens=1)
        pos = torch.tensor([[args.past]], device=args.device)
        cache = DeepSeekPagedCache(
            views,
            pos,
            past_length=args.past,
            max_length=args.past + 1,
            key_dim=key_dim,
            value_dim=value_dim,
            head_dim=max(key_dim, value_dim),
            native_attention=True,
            sliding_window=window,
            attention_sink=sink,
        )
        q = torch.randn(1, heads, 1, key_dim, dtype=torch.float16, device=args.device)
        attend = functools.partial(cache.attend, q, 0, pos, scale=key_dim**-0.5)
        kernel = timed(attend, args.warmup, args.kernel_steps)
        run = {
            "config": spec,
            "window": window,
            "sink": sink,
            "model_ms_median": round(statistics.median(model), 2),
            "model_tok_s": round(1000 / statistics.median(model), 2),
            "model_ms_samples": [round(x, 2) for x in model],
            "kernel_one_layer_ms_median": round(statistics.median(kernel), 3),
            "kernel_x_layers_ms": round(statistics.median(kernel) * layers, 2),
        }
        result["runs"].append(run)
        print(
            f"{spec:<10} model {run['model_ms_median']:8.1f} ms/token ({run['model_tok_s']:6.2f} tok/s) | "
            f"one layer's attention kernel {run['kernel_one_layer_ms_median']:7.3f} ms, "
            f"x{layers} = {run['kernel_x_layers_ms']:.1f} ms",
            flush=True,
        )
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=1)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
