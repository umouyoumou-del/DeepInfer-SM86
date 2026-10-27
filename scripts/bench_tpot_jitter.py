"""Measure TPOT jitter with and without a long prefill insertion.

A short streaming request is started first.  In the insertion run, a long
prompt arrives after the first streamed token.  The scheduler must keep
decoding while advancing the long request in prefill chunks.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.paged_scheduler import PagedBatchScheduler
from apps.runtime import QwenSm86Runtime


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    p50 = _percentile(values, 0.50)
    p95 = _percentile(values, 0.95)
    return {
        "mean": round(statistics.fmean(values), 6),
        "p50": round(p50, 6),
        "p95": round(p95, 6),
        "max": round(max(values), 6),
        "p95_minus_p50": round(p95 - p50, 6),
    }


def _drain_stream(queue, on_first: Callable[[], None] | None = None) -> None:
    first = True
    while True:
        delta = queue.get(timeout=300)
        if delta is None:
            break
        if first and on_first is not None:
            on_first()
        first = False


def _make_long_prompt(repeat: int) -> str:
    sentence = "这是一段用于测试分块预填充和流式解码抖动的长文本。"
    return sentence * repeat


def _run_pair(scheduler: PagedBatchScheduler, long_prompt: str, max_new: int) -> dict:
    scheduler.manager.prefix_cache.clear()
    short_prompt = f"请简短回答：交错调度有什么好处？运行编号 {time.time_ns()}"
    short = scheduler.submit(
        [{"role": "user", "content": short_prompt}],
        max_new_tokens=max_new,
        temperature=0.0,
        stream=True,
        tenant="tpot-baseline",
    )
    assert short.stream_queue is not None
    _drain_stream(short.stream_queue)
    baseline = short.future.result(timeout=600)

    scheduler.manager.prefix_cache.clear()
    short2 = scheduler.submit(
        [{"role": "user", "content": short_prompt + " 插入测试"}],
        max_new_tokens=max_new,
        temperature=0.0,
        stream=True,
        tenant="tpot-insertion",
    )
    assert short2.stream_queue is not None
    long_request = None

    def insert_long() -> None:
        nonlocal long_request
        long_request = scheduler.submit(
            [{"role": "user", "content": long_prompt}],
            max_new_tokens=2,
            temperature=0.0,
            tenant="long-prefill",
        )

    _drain_stream(short2.stream_queue, on_first=insert_long)
    inserted = short2.future.result(timeout=600)
    assert long_request is not None
    long_result = long_request.future.result(timeout=600)
    baseline_intervals = list(baseline.inter_token_seconds)
    insertion_intervals = list(inserted.inter_token_seconds)
    return {
        "baseline": {
            "ttft_seconds": baseline.ttft_seconds,
            "tpot_seconds": baseline.tpot_seconds,
            "inter_token": _summary(baseline_intervals),
        },
        "with_insertion": {
            "ttft_seconds": inserted.ttft_seconds,
            "tpot_seconds": inserted.tpot_seconds,
            "inter_token": _summary(insertion_intervals),
            "long_prompt_tokens": long_result.prompt_tokens,
            "long_ttft_seconds": long_result.ttft_seconds,
        },
        "insertion_jitter_delta_seconds": round(
            _summary(insertion_intervals)["p95_minus_p50"]
            - _summary(baseline_intervals)["p95_minus_p50"],
            6,
        ),
        "insertion_max_itl_seconds": round(
            max(insertion_intervals, default=0.0), 6
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backend", choices=("w4a16", "mixed"), default="w4a16")
    parser.add_argument("--max-cache", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--chunk-sizes", default="8,16,32,64")
    parser.add_argument("--long-repeat", type=int, default=8)
    parser.add_argument("--kv-cache-dtype", choices=("fp16", "int8"), default="fp16")
    parser.add_argument(
        "--no-prefill-decode-overlap",
        action="store_true",
        help="disable dual-CUDA-stream prefill/decode overlap",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the jitter benchmark")

    runtime = QwenSm86Runtime(
        args.checkpoint,
        max_cache=args.max_cache,
        use_graph=False,
        single_request_cache=False,
        backend=args.backend,
        allow_extended_context=True,
    )
    long_prompt = _make_long_prompt(args.long_repeat)
    prompt_text = runtime.tokenizer.apply_chat_template(
        [{"role": "user", "content": long_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    long_tokens = int(
        runtime.tokenizer(prompt_text, return_tensors="pt")["input_ids"].shape[1]
    )
    if long_tokens >= args.max_cache - args.max_new_tokens:
        raise SystemExit(
            f"long prompt is {long_tokens} tokens; increase --max-cache or "
            "reduce --long-repeat"
        )

    reports = []
    for chunk_size in [int(value) for value in args.chunk_sizes.split(",") if value]:
        scheduler = PagedBatchScheduler(
            runtime,
            max_num_seqs=2,
            block_size=16,
            prefill_chunk_tokens=chunk_size,
            prefix_cache_blocks=0,
            gpu_memory_utilization=0.80,
            reserve_memory_mb=384,
            kv_cache_dtype=args.kv_cache_dtype,
            enable_prefill_decode_overlap=not args.no_prefill_decode_overlap,
        )
        try:
            warmup = scheduler.submit(
                [{"role": "user", "content": "预热"}],
                max_new_tokens=2,
                temperature=0.0,
            )
            warmup.future.result(timeout=600)
            result = _run_pair(scheduler, long_prompt, args.max_new_tokens)
            result.update(
                {
                    "prefill_chunk_tokens": chunk_size,
                    "kv_cache_dtype": args.kv_cache_dtype,
                    "long_prompt_tokens": long_tokens,
                    "prefill_steps_total": int(
                        scheduler.metrics.get("prefill_steps_total")
                    ),
                    "prefill_decode_overlap_steps": int(
                        scheduler.metrics.get("prefill_decode_overlap_steps")
                    ),
                    "prefill_decode_stream_overlap_steps": int(
                        scheduler.metrics.get(
                            "prefill_decode_stream_overlap_steps"
                        )
                    ),
                    "prefill_decode_overlap_enabled": bool(
                        args.no_prefill_decode_overlap is False
                    ),
                    "max_decode_batch_size": int(
                        scheduler.metrics.get("max_decode_batch_size")
                    ),
                }
            )
            reports.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        finally:
            scheduler.stop()
            del scheduler
            gc.collect()
            torch.cuda.empty_cache()

    report = {
        "checkpoint": str(args.checkpoint),
        "backend": args.backend,
        "max_cache": args.max_cache,
        "max_new_tokens": args.max_new_tokens,
        "long_prompt_tokens": long_tokens,
        "reports": reports,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
