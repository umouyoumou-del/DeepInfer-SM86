"""Concurrency, TTFT/TPOT, and OOM benchmark for the paged scheduler.

Runs the same workload at several active-request widths through one resident
scheduler.  The first run is the serial baseline; later runs show whether
batched decode actually improves aggregate throughput.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.paged_scheduler import PagedBatchScheduler
from apps.runtime import QwenSm86Runtime

DEFAULT_PROMPT = "请用三句话解释分页 KV cache 对连续批处理的意义。"


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "mean": round(statistics.fmean(values), 6),
        "p50": round(_percentile(values, 0.50), 6),
        "p95": round(_percentile(values, 0.95), 6),
    }


def _run_concurrency(
    scheduler: PagedBatchScheduler,
    *,
    concurrency: int,
    run_index: int,
    max_new_tokens: int,
) -> dict:
    # Unique suffixes keep concurrent prompts from sharing a prefix by accident.
    prompts = [
        f"{DEFAULT_PROMPT}（基准运行 {run_index}，请求 {index}）"
        for index in range(concurrency)
    ]
    scheduler.manager.prefix_cache.clear()
    oom_before = scheduler.metrics.get("kv_cache_oom_total")
    overlap_before = scheduler.metrics.get("prefill_decode_overlap_steps")
    stream_overlap_before = scheduler.metrics.get(
        "prefill_decode_stream_overlap_steps"
    )
    cache_before = scheduler.manager.prefix_cache.stats()
    started = time.perf_counter()
    requests = [
        scheduler.submit(
            [{"role": "user", "content": prompt}],
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )
        for prompt in prompts
    ]
    results = []
    failures = 0
    for request in requests:
        try:
            results.append(request.future.result(timeout=600))
        except Exception as exc:  # noqa: BLE001 - benchmark must record failures
            failures += 1
            print(f"[bench] request failed: {type(exc).__name__}: {exc}", flush=True)
    wall = time.perf_counter() - started
    completion_tokens = sum(result.completion_tokens for result in results)
    ttft = [result.ttft_seconds for result in results]
    tpot = [result.tpot_seconds for result in results if result.completion_tokens > 1]
    inter_token = [
        value
        for result in results
        for value in result.inter_token_seconds
    ]
    inter_summary = _latency_summary(inter_token)
    oom_after = scheduler.metrics.get("kv_cache_oom_total")
    overlap_after = scheduler.metrics.get("prefill_decode_overlap_steps")
    stream_overlap_after = scheduler.metrics.get(
        "prefill_decode_stream_overlap_steps"
    )
    cache_after = scheduler.manager.prefix_cache.stats()
    cache_hits = int(cache_after["hits"] - cache_before["hits"])
    cache_misses = int(cache_after["misses"] - cache_before["misses"])
    cache_lookups = cache_hits + cache_misses
    return {
        "concurrency": concurrency,
        "wall_seconds": round(wall, 6),
        "requests": len(requests),
        "failed_requests": failures,
        "oom_rate": round(failures / len(requests), 6) if requests else 0.0,
        "completion_tokens": completion_tokens,
        "aggregate_tokens_per_second": round(completion_tokens / wall, 3)
        if wall > 0
        else 0.0,
        "ttft_seconds": _latency_summary(ttft),
        "tpot_seconds": _latency_summary(tpot),
        "inter_token_seconds": inter_summary,
        "tpot_jitter_p95_minus_p50": round(
            inter_summary["p95"] - inter_summary["p50"], 6
        ),
        "kv_cache_oom_delta": int(oom_after - oom_before),
        "prefill_decode_overlap_steps": int(overlap_after - overlap_before),
        "prefill_decode_stream_overlap_steps": int(
            stream_overlap_after - stream_overlap_before
        ),
        "max_decode_batch_size": int(
            scheduler.metrics.get("max_decode_batch_size", 0)
        ),
        "prefix_cache_hits": cache_hits,
        "prefix_cache_misses": cache_misses,
        "prefix_cache_hit_rate": (
            round(cache_hits / cache_lookups, 6) if cache_lookups else 0.0
        ),
        "prefix_reused_tokens": sum(
            result.prefix_cached_tokens for result in results
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backend", choices=("w4a16", "mixed"), default="w4a16")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-cache", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--prefill-chunk-tokens", type=int, default=64)
    parser.add_argument(
        "--kv-cache-dtype",
        choices=("fp16", "int8", "fp8", "fp8_e5m2"),
        default="fp16",
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="enable CPU block swap when GPU KV blocks are exhausted",
    )
    parser.add_argument(
        "--evict-policy",
        choices=("offload", "drop"),
        default="offload",
        help="drop frees cold KV instead of staging it to the host",
    )
    parser.add_argument("--max-cpu-offload-mb", type=int, default=2048)
    parser.add_argument(
        "--concurrency",
        default="1,2,4,8",
        help="comma-separated active request widths",
    )
    parser.add_argument("--kv-blocks", type=int, default=None)
    parser.add_argument("--kv-budget-mb", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--reserve-memory-mb", type=int, default=512)
    parser.add_argument("--no-prefix-cache", action="store_true")
    parser.add_argument(
        "--no-prefill-decode-overlap",
        action="store_true",
        help="disable dual-CUDA-stream prefill/decode overlap",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the SM86 benchmark")

    widths = sorted({int(value) for value in args.concurrency.split(",") if value})
    if not widths or widths[0] < 1:
        raise SystemExit("--concurrency must contain positive integers")
    max_width = widths[-1]
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
        max_num_seqs=max_width,
        block_size=args.block_size,
        prefill_chunk_tokens=args.prefill_chunk_tokens,
        kv_cache_dtype=args.kv_cache_dtype,
        enable_prefill_decode_overlap=not args.no_prefill_decode_overlap,
        enable_cpu_offload=args.cpu_offload,
        evict_policy=args.evict_policy,
        max_cpu_offload_mb=args.max_cpu_offload_mb,
        num_gpu_blocks=args.kv_blocks,
        prefix_cache_blocks=0 if args.no_prefix_cache else None,
        gpu_memory_utilization=args.gpu_memory_utilization,
        reserve_memory_mb=args.reserve_memory_mb,
        kv_budget_mb=args.kv_budget_mb,
    )
    try:
        warmup = scheduler.submit(
            [{"role": "user", "content": "你好"}],
            max_new_tokens=2,
            temperature=0.0,
        )
        warmup.future.result(timeout=600)
        runs = [
            _run_concurrency(
                scheduler,
                concurrency=width,
                run_index=index,
                max_new_tokens=args.max_new_tokens,
            )
            for index, width in enumerate(widths)
        ]
        baseline = runs[0]["aggregate_tokens_per_second"]
        for run in runs:
            run["speedup_vs_first"] = (
                round(run["aggregate_tokens_per_second"] / baseline, 3)
                if baseline > 0
                else 0.0
            )
        report = {
            "checkpoint": str(args.checkpoint),
            "backend": args.backend,
            "max_new_tokens": args.max_new_tokens,
            "max_cache": args.max_cache,
            "block_size": args.block_size,
            "prefill_chunk_tokens": args.prefill_chunk_tokens,
            "prefill_decode_overlap": not args.no_prefill_decode_overlap,
            "kv_cache_dtype": args.kv_cache_dtype,
            "cpu_offload": args.cpu_offload,
            "max_cpu_offload_mb": args.max_cpu_offload_mb,
            "prefix_cache": not args.no_prefix_cache,
            "concurrency_runs": runs,
            "memory_budget": scheduler.memory_budget.as_dict(),
            "scheduler": scheduler.status(),
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
