"""Microbenchmark SM86 tiled W4A16/W3A16 prefill GEMMs.

The benchmark uses synthetic packed weights so it does not need a multi-GB
checkpoint.  Shapes match Qwen2.5-3B/7B linear layers at a representative
prefill tile (M=30) on the RTX 3050 target.  It compares:

* the native tiled GEMM (dequantize one K-slice of weights into shared memory);
* the previous portable fallback (dequantize per-output-row chunks + cuBLAS).

Example:
    python scripts/bench_sm86_tiled_prefill.py ^
      --iterations 20 --output docs/results/qwen_sm86_tiled_prefill.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_qwen2_w4a16_gpu_harness import (
    _install_deepinfer_import_stubs,
    _load_module,
)

DEFAULT_W4_SHAPES = (
    (30, 5632, 2048),   # fused gate/up
    (30, 2048, 5632),   # down projection
    (30, 3584, 18944),  # 7B down projection
)
DEFAULT_W3_SHAPES = (
    (30, 37888, 3584),  # 7B fused gate/up
    (30, 3584, 18944),  # 7B down projection
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "results" / "qwen_sm86_tiled_prefill.json",
    )
    return parser.parse_args()


def _timed_ms(fn: Any, *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - started) / iterations


def _random_packed_w4(module: Any, out_features: int, in_features: int) -> None:
    packed = torch.randint(
        0,
        256,
        (out_features, in_features // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    scales = torch.rand(
        out_features,
        in_features // module.group_size,
        dtype=torch.float16,
        device="cuda",
    ) * 0.05 + 0.01
    module.load_from_packed(packed, scales, None)


def _random_packed_w3(module: Any, out_features: int, in_features: int) -> None:
    row_bytes = in_features * 3 // 8
    packed = torch.randint(
        0,
        256,
        (out_features * row_bytes + 1,),
        dtype=torch.uint8,
        device="cuda",
    )
    scales = torch.rand(
        out_features,
        in_features // module.group_size,
        dtype=torch.float16,
        device="cuda",
    ) * 0.05 + 0.01
    module.load_from_packed(packed, scales, None)


def _bench_one(
    module_cls: Any,
    *,
    bits: int,
    m: int,
    n: int,
    k: int,
    iterations: int,
    warmup: int,
    initialiser: Any,
) -> dict[str, Any]:
    module = module_cls(
        k,
        n,
        False,
        group_size=128,
        dtype=torch.float16,
        device="cuda",
        prefill_chunk_rows=2048,
    )
    initialiser(module, n, k)
    inputs = torch.randn(m, k, dtype=torch.float16, device="cuda")
    tiled_ms = _timed_ms(
        partial(module, inputs),
        warmup=warmup,
        iterations=iterations,
    )
    fallback_ms = _timed_ms(
        partial(module._forward_dequant_cublas, inputs),
        warmup=warmup,
        iterations=iterations,
    )
    assert module.last_prefill_backend == "dequant_cublas"
    row: dict[str, Any] = {
        "bits": bits,
        "m": m,
        "n": n,
        "k": k,
        "tiled_ms": round(tiled_ms, 4),
        "fallback_ms": round(fallback_ms, 4),
        "speedup": round(fallback_ms / tiled_ms, 3),
    }
    del module, inputs
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return row


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires a CUDA GPU")
    _install_deepinfer_import_stubs()
    _load_module("deepinfer.hugging_face.qwen2_config", "hugging_face/qwen2_config.py")
    _load_module("deepinfer.model_executor.models.qwen2", "model_executor/models/qwen2.py")
    w4 = _load_module(
        "deepinfer.model_executor.models.qwen2_w4a16",
        "model_executor/models/qwen2_w4a16.py",
    )
    mixed = _load_module(
        "deepinfer.model_executor.models.qwen2_mixed_sm86",
        "model_executor/models/qwen2_mixed_sm86.py",
    )

    report: dict[str, Any] = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "group_size": 128,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "w4a16": [],
        "w3a16": [],
    }
    for m, n, k in DEFAULT_W4_SHAPES:
        print(f"W4 M={m} N={n} K={k} ...", flush=True)
        row = _bench_one(
            w4.W4A16Linear,
            bits=4,
            m=m,
            n=n,
            k=k,
            iterations=args.iterations,
            warmup=args.warmup,
            initialiser=_random_packed_w4,
        )
        report["w4a16"].append(row)
        print(f"  tiled={row['tiled_ms']:.3f} ms fallback={row['fallback_ms']:.3f} ms ({row['speedup']:.2f}x)", flush=True)
    for m, n, k in DEFAULT_W3_SHAPES:
        print(f"W3 M={m} N={n} K={k} ...", flush=True)
        row = _bench_one(
            mixed.W3A16Linear,
            bits=3,
            m=m,
            n=n,
            k=k,
            iterations=args.iterations,
            warmup=args.warmup,
            initialiser=_random_packed_w3,
        )
        report["w3a16"].append(row)
        print(f"  tiled={row['tiled_ms']:.3f} ms fallback={row['fallback_ms']:.3f} ms ({row['speedup']:.2f}x)", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
