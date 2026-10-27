"""Measure paged continuous-batching throughput on a resident SM86 model.

This script intentionally compares the same prompts under:
1. serial submission (one request at a time), and
2. concurrent submission through the paged continuous scheduler.

It reports wall time and aggregate generated tokens/s.  Prefix caching is left
enabled (the default) so repeated prompts can exercise the shared-prefix path;
run with ``--no-prefix-cache`` to measure cold-prefix behavior.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.paged_scheduler import PagedBatchScheduler
from apps.runtime import QwenSm86Runtime

DEFAULT_PROMPTS = [
    "用一句话解释 CUDA Graph。",
    "写三句话说明分页 KV cache 的作用。",
    "把这句话翻译成英文：小显存也需要连续批处理。",
    "列出三个降低推理显存占用的方法。",
]


def _run_serial(scheduler: PagedBatchScheduler, prompts: list[str], max_new: int) -> dict:
    started = time.perf_counter()
    completion_tokens = 0
    for prompt in prompts:
        request = scheduler.submit(
            [{"role": "user", "content": prompt}],
            max_new_tokens=max_new,
            temperature=0.0,
        )
        result = request.future.result(timeout=600)
        completion_tokens += result.completion_tokens
    wall = time.perf_counter() - started
    return {
        "wall_seconds": round(wall, 6),
        "completion_tokens": completion_tokens,
        "aggregate_tokens_per_second": round(completion_tokens / wall, 3),
    }


def _run_concurrent(
    scheduler: PagedBatchScheduler, prompts: list[str], max_new: int
) -> dict:
    started = time.perf_counter()
    requests = [
        scheduler.submit(
            [{"role": "user", "content": prompt}],
            max_new_tokens=max_new,
            temperature=0.0,
        )
        for prompt in prompts
    ]
    results = [request.future.result(timeout=600) for request in requests]
    wall = time.perf_counter() - started
    completion_tokens = sum(result.completion_tokens for result in results)
    return {
        "wall_seconds": round(wall, 6),
        "completion_tokens": completion_tokens,
        "aggregate_tokens_per_second": round(completion_tokens / wall, 3),
        "per_request": [
            {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "prefix_cached_tokens": result.prefix_cached_tokens,
            }
            for result in results
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backend", choices=("w4a16", "mixed"), default="w4a16")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-cache", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--kv-blocks", type=int, default=None)
    parser.add_argument("--no-prefix-cache", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the SM86 benchmark")

    runtime = QwenSm86Runtime(
        args.checkpoint,
        max_cache=args.max_cache,
        use_graph=False,
        single_request_cache=False,
        backend=args.backend,
        allow_extended_context=True,
    )
    scheduler = PagedBatchScheduler(
        runtime,
        max_num_seqs=args.max_num_seqs,
        block_size=args.block_size,
        num_gpu_blocks=args.kv_blocks,
        prefix_cache_blocks=0 if args.no_prefix_cache else None,
    )
    try:
        # Capture CUDA kernels / allocator warmup outside the measured region.
        warmup = scheduler.submit(
            [{"role": "user", "content": "你好"}],
            max_new_tokens=2,
            temperature=0.0,
        )
        warmup.future.result(timeout=600)
        serial = _run_serial(scheduler, DEFAULT_PROMPTS, args.max_new_tokens)
        concurrent = _run_concurrent(scheduler, DEFAULT_PROMPTS, args.max_new_tokens)
        report = {
            "checkpoint": str(args.checkpoint),
            "backend": args.backend,
            "max_new_tokens": args.max_new_tokens,
            "max_cache": args.max_cache,
            "max_num_seqs": args.max_num_seqs,
            "block_size": args.block_size,
            "prefix_cache": not args.no_prefix_cache,
            "serial": serial,
            "concurrent": concurrent,
            "speedup": round(
                concurrent["aggregate_tokens_per_second"]
                / serial["aggregate_tokens_per_second"],
                3,
            ),
            "scheduler": scheduler.status(),
        }
        text = json.dumps(report, ensure_ascii=False, indent=2)
        print(text)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")
        return 0
    finally:
        scheduler.stop()


if __name__ == "__main__":
    raise SystemExit(main())
