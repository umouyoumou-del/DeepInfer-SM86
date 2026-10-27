"""Microbenchmark the SM86 batch-tile W4A16 GEMV kernel."""

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

from deepinfer.model_executor.models.qwen2_w4a16 import W4A16Linear


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-features", type=int, default=2048)
    parser.add_argument("--out-features", type=int, default=1024)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--batches", default="1,2,4,8")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(20261101)
    dense = torch.nn.Linear(
        args.in_features, args.out_features, bias=False
    ).cuda().half()
    linear = W4A16Linear(
        args.in_features,
        args.out_features,
        False,
        group_size=args.group_size,
        dtype=torch.float16,
    ).cuda()
    linear.load_from_linear(dense)
    reference_weight = linear._dequantize_rows(0, args.out_features).to(torch.float16)
    rows = []
    for batch in [int(value) for value in args.batches.split(",") if value]:
        inputs = torch.randn(
            batch, args.in_features, device="cuda", dtype=torch.float16
        )
        output = linear(inputs)
        reference = torch.nn.functional.linear(inputs, reference_weight, None)
        cosine = float(
            torch.nn.functional.cosine_similarity(
                output.flatten().float(), reference.flatten().float(), dim=0
            )
        )
        for _ in range(10):
            linear(inputs)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.iterations):
            linear(inputs)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / args.iterations
        rows.append(
            {
                "batch": batch,
                "seconds": round(elapsed, 8),
                "tokens_per_second": round(batch / elapsed, 3),
                "cosine": cosine,
                "max_abs_error": float(
                    (output.float() - reference.float()).abs().max()
                ),
            }
        )
    report = {
        "device": torch.cuda.get_device_name(0),
        "in_features": args.in_features,
        "out_features": args.out_features,
        "group_size": args.group_size,
        "iterations": args.iterations,
        "rows": rows,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
