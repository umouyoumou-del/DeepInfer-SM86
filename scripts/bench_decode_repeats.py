"""Decode throughput with repeats and a median, because single runs vary 65%.

The same host gave 215 / 318 / 353 ms/token for identical work, so any change
smaller than about 30% cannot be judged from one measurement.  This harness runs
the same workload N times and reports median/min/max.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--meta-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt", default="只回答数字：5-2=?")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-cache", type=int, default=256)
    parser.add_argument("--gpu-expert-cache-mb", type=int, default=0)
    parser.add_argument("--pin-all-experts", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import torch

    from deepinfer.moe.deepseek_v2_runtime import DeepSeekV2LiteGGUFRuntime

    runtime = DeepSeekV2LiteGGUFRuntime(
        args.model,
        meta_dir=args.meta_dir,
        device=args.device,
        max_cache=args.max_cache,
        gpu_expert_cache_mb=args.gpu_expert_cache_mb,
        pin_all_experts=bool(args.pin_all_experts),
        verbose=False,
    )
    if args.pin_all_experts:
        runtime.expert_store.preload_gpu_experts()

    ids = runtime.tokenizer(args.prompt, return_tensors="pt")["input_ids"].long()

    for _ in range(max(0, args.warmup)):
        runtime.generate(ids, max_new_tokens=args.max_new_tokens)

    samples: list[float] = []
    for index in range(args.repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        out = runtime.generate(ids, max_new_tokens=args.max_new_tokens)
        elapsed = time.perf_counter() - started
        produced = max(1, len(out.token_ids))
        per_token = elapsed / produced * 1000.0
        samples.append(per_token)
        print(
            f"  run {index + 1}/{args.repeats}: {elapsed:.3f}s "
            f"{produced} tok  {per_token:.1f} ms/token",
            flush=True,
        )

    median = statistics.median(samples)
    result = {
        "repeats": args.repeats,
        "samples_ms_per_token": samples,
        "median_ms_per_token": median,
        "min_ms_per_token": min(samples),
        "max_ms_per_token": max(samples),
        "spread_pct": 100.0 * (max(samples) - min(samples)) / median,
        "median_tokens_per_second": 1000.0 / median,
        "pin_all_experts": bool(args.pin_all_experts),
        "gpu_expert_cache_mb": args.gpu_expert_cache_mb,
    }
    print(
        f"median {median:.1f} ms/token ({result['median_tokens_per_second']:.2f} tok/s) "
        f"range {min(samples):.1f}-{max(samples):.1f} spread {result['spread_pct']:.1f}%"
    )
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
