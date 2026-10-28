"""Smoke-test the DeepSeek MoE runtime behind the existing paged scheduler."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.paged_scheduler import PagedBatchScheduler
from deepinfer.moe.deepseek_v2_runtime import DeepSeekV2LiteGGUFRuntime


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            r"E:\DeepInfer-MoE\models\DeepSeek-V2-Lite-Chat-GGUF"
            r"\DeepSeek-V2-Lite-Chat-Q4_K_M.gguf"
        ),
    )
    parser.add_argument("--meta-dir", type=Path, default=Path(r"E:\DeepInfer-MoE\meta"))
    parser.add_argument("--max-cache", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--prefill-chunk-tokens", type=int, default=8)
    parser.add_argument("--num-gpu-blocks", type=int, default=32)
    parser.add_argument(
        "--enable-prefill-decode-overlap",
        action="store_true",
        help="request scheduler prefill/decode CUDA stream overlap",
    )
    parser.add_argument(
        "--prefix-reuse",
        action="store_true",
        help="submit the first prompt a second time after the first batch finishes",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    runtime = DeepSeekV2LiteGGUFRuntime(
        args.model,
        meta_dir=args.meta_dir,
        max_cache=args.max_cache,
        expert_chunk_size=8,
        verbose=False,
    )
    prompts = [
        [{"role": "user", "content": "只回答数字：5-2=?"}],
        [{"role": "user", "content": "中国的首都是哪里？只回答城市名。"}],
    ]
    baseline = []
    scheduler = None
    try:
        for messages in prompts:
            result = runtime.generate(
                messages, max_new_tokens=args.max_new_tokens, temperature=0.0
            )
            baseline.append(
                {
                    "text": result.text,
                    "token_ids": result.token_ids,
                }
            )
        scheduler = PagedBatchScheduler(
            runtime,
            max_num_seqs=args.max_num_seqs,
            block_size=args.block_size,
            prefill_chunk_tokens=args.prefill_chunk_tokens,
            enable_prefill_decode_overlap=args.enable_prefill_decode_overlap,
            num_gpu_blocks=args.num_gpu_blocks,
            prefix_cache_blocks=4,
            reserve_memory_mb=256,
            gpu_memory_utilization=0.80,
        )
        started = time.perf_counter()
        requests = [
            scheduler.submit(
                messages,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
            )
            for messages in prompts
        ]
        results = [request.future.result(timeout=180) for request in requests]
        elapsed = time.perf_counter() - started
        records = []
        for index, (request, result) in enumerate(zip(requests, results, strict=True)):
            paged_ids = list(request.generated_ids)
            records.append(
                {
                    "text": result.text,
                    "token_ids": paged_ids,
                    "baseline_text": baseline[index]["text"],
                    "baseline_token_ids": baseline[index]["token_ids"],
                    "text_equal": result.text == baseline[index]["text"],
                    "token_ids_equal": paged_ids == baseline[index]["token_ids"],
                    "finish_reason": result.finish_reason,
                }
            )
        prefix_reuse = None
        if args.prefix_reuse:
            request = scheduler.submit(
                prompts[0],
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
            )
            result = request.future.result(timeout=180)
            prefix_reuse = {
                "text": result.text,
                "token_ids": list(request.generated_ids),
                "baseline_token_ids": baseline[0]["token_ids"],
                "token_ids_equal": list(request.generated_ids)
                == baseline[0]["token_ids"],
                "prefix_cached_tokens": result.prefix_cached_tokens,
            }
        metrics = scheduler.metrics.snapshot()
        status = scheduler.status()
        report = {
            "elapsed_seconds": elapsed,
            "baseline": baseline,
            "paged": records,
            "prefix_reuse": prefix_reuse,
            "metrics": metrics,
            "status": status,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return 0
    finally:
        if scheduler is not None:
            scheduler.stop()
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
