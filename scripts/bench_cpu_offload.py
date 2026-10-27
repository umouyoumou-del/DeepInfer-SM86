"""Force paged-KV CPU swap and verify both requests still complete.

The scheduler is deliberately sized for exactly one request.  A streaming
request is started first; after its first token, a second request arrives and
must evict the first request's KV blocks to CPU.  The first request is then
reloaded and resumes when the second finishes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.paged_scheduler import PagedBatchScheduler
from apps.runtime import QwenSm86Runtime


def _prompt_tokens(tokenizer, content: str) -> int:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return int(tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backend", choices=("w4a16", "mixed"), default="w4a16")
    parser.add_argument("--max-cache", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--kv-cache-dtype",
        choices=("fp16", "int8", "fp8", "fp8_e5m2"),
        default="fp16",
    )
    parser.add_argument("--max-cpu-offload-mb", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.iterations < 1:
        raise SystemExit("--iterations must be >= 1")

    runtime = QwenSm86Runtime(
        args.checkpoint,
        max_cache=args.max_cache,
        use_graph=False,
        single_request_cache=False,
        backend=args.backend,
        allow_extended_context=True,
    )
    prompts = ["请简短回答：问题一", "请简短回答：问题二"]
    prompt_lengths = [_prompt_tokens(runtime.tokenizer, text) for text in prompts]
    blocks_per_request = max(
        math.ceil((length + args.max_new_tokens) / args.block_size)
        for length in prompt_lengths
    )
    scheduler = PagedBatchScheduler(
        runtime,
        max_num_seqs=2,
        block_size=args.block_size,
        prefill_chunk_tokens=32,
        kv_cache_dtype=args.kv_cache_dtype,
        enable_prefill_decode_overlap=True,
        enable_cpu_offload=True,
        max_cpu_offload_mb=args.max_cpu_offload_mb,
        num_gpu_blocks=blocks_per_request,
        prefix_cache_blocks=0,
        gpu_memory_utilization=0.80,
        reserve_memory_mb=384,
    )
    results = []
    try:
        for iteration in range(args.iterations):
            scheduler.manager.prefix_cache.clear()
            before = {
                "swapped": scheduler.metrics.get("requests_swapped_total"),
                "resumed": scheduler.metrics.get("requests_resumed_total"),
                "offload_blocks": scheduler.metrics.get(
                    "kv_cpu_offload_blocks_total"
                ),
                "reload_blocks": scheduler.metrics.get(
                    "kv_cpu_reload_blocks_total"
                ),
            }
            started = time.perf_counter()
            first = scheduler.submit(
                [{"role": "user", "content": prompts[0]}],
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                stream=True,
                tenant=f"offload-{iteration}-first",
            )
            assert first.stream_queue is not None
            first_delta = first.stream_queue.get(timeout=600)
            if not first_delta:
                raise RuntimeError("first request produced an empty first delta")
            second = scheduler.submit(
                [{"role": "user", "content": prompts[1]}],
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                tenant=f"offload-{iteration}-second",
            )
            while first.stream_queue.get(timeout=600) is not None:
                pass
            first_result = first.future.result(timeout=600)
            second_result = second.future.result(timeout=600)
            after = {
                "swapped": scheduler.metrics.get("requests_swapped_total"),
                "resumed": scheduler.metrics.get("requests_resumed_total"),
                "offload_blocks": scheduler.metrics.get(
                    "kv_cpu_offload_blocks_total"
                ),
                "reload_blocks": scheduler.metrics.get(
                    "kv_cpu_reload_blocks_total"
                ),
            }
            deltas = {key: int(after[key] - before[key]) for key in before}
            if deltas["swapped"] < 1 or deltas["resumed"] < 1:
                raise RuntimeError(
                    "CPU offload was not exercised; reduce --kv-blocks or "
                    "increase prompt/output length"
                )
            results.append(
                {
                    "iteration": iteration,
                    "wall_seconds": round(time.perf_counter() - started, 6),
                    "first_completion_tokens": first_result.completion_tokens,
                    "second_completion_tokens": second_result.completion_tokens,
                    "first_tpot_seconds": first_result.tpot_seconds,
                    "second_tpot_seconds": second_result.tpot_seconds,
                    "swapped_delta": deltas["swapped"],
                    "resumed_delta": deltas["resumed"],
                    "offload_blocks_delta": deltas["offload_blocks"],
                    "reload_blocks_delta": deltas["reload_blocks"],
                }
            )
            print(json.dumps(results[-1], ensure_ascii=False), flush=True)
        status = scheduler.status()
        report = {
            "checkpoint": str(args.checkpoint),
            "backend": args.backend,
            "kv_cache_dtype": args.kv_cache_dtype,
            "block_size": args.block_size,
            "max_new_tokens": args.max_new_tokens,
            "gpu_blocks": scheduler.manager.num_blocks,
            "blocks_per_request": blocks_per_request,
            "max_cpu_offload_mb": args.max_cpu_offload_mb,
            "prompt_tokens": prompt_lengths,
            "iterations": results,
            "cpu_offload": status["cpu_offload"],
            "memory_budget": status["memory_budget"],
            "metrics": scheduler.metrics.snapshot(),
        }
        text = json.dumps(report, ensure_ascii=False, indent=2)
        print(text)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8", newline="\n")
        return 0
    finally:
        scheduler.stop()


if __name__ == "__main__":
    raise SystemExit(main())
