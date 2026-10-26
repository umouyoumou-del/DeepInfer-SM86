"""Estimate CPU-side W4A16 layer cost for a Qwen2.5-7B hybrid offload."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

HIDDEN = 3584
INTERMEDIATE = 18944
GROUP = 128
SHAPES = {
    "q_proj": (HIDDEN, HIDDEN),
    "k_proj": (512, HIDDEN),
    "v_proj": (512, HIDDEN),
    "o_proj": (HIDDEN, HIDDEN),
    "gate_proj": (INTERMEDIATE, HIDDEN),
    "up_proj": (INTERMEDIATE, HIDDEN),
    "down_proj": (HIDDEN, INTERMEDIATE),
}


def dequantize(
    packed: torch.Tensor, scales: torch.Tensor, out_features: int, in_features: int
) -> torch.Tensor:
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    quantized = torch.empty((out_features, in_features), dtype=torch.float32)
    quantized[:, 0::2] = low.float()
    quantized[:, 1::2] = high.float()
    return (
        quantized.view(out_features, in_features // GROUP, GROUP)
        * scales.float()[:, :, None]
    ).view(out_features, in_features)


def main() -> int:
    torch.set_num_threads(6)
    torch.manual_seed(0)
    results: dict[str, Any] = {
        "hidden": HIDDEN,
        "intermediate": INTERMEDIATE,
        "group_size": GROUP,
        "threads": torch.get_num_threads(),
        "shapes": {},
    }
    total_ms = 0.0
    for name, (out_features, in_features) in SHAPES.items():
        packed = torch.randint(
            0, 256, (out_features, in_features // 2), dtype=torch.uint8
        )
        scales = (
            torch.rand(out_features, in_features // GROUP, dtype=torch.float32) * 0.02
            + 0.001
        ).to(torch.float16)
        x = torch.randn(in_features, dtype=torch.float32)
        samples: list[float] = []
        for _ in range(5):
            started = time.perf_counter()
            weight = dequantize(packed, scales, out_features, in_features)
            y = weight @ x
            samples.append(time.perf_counter() - started)
            del weight, y
        median = statistics.median(samples)
        total_ms += median * 1000
        results["shapes"][name] = {
            "out_features": out_features,
            "in_features": in_features,
            "median_ms": round(median * 1000, 3),
            "min_ms": round(min(samples) * 1000, 3),
        }
        print(name, results["shapes"][name], flush=True)
        del packed, scales, x
    results["layer_total_ms"] = round(total_ms, 3)
    results["layer_tokens_per_second_if_only_cpu"] = round(1000.0 / total_ms, 3)
    for layers in (4, 6, 8, 10, 12):
        results[f"{layers}_cpu_layers_tps_ceiling"] = round(
            1000.0 / (total_ms * layers), 3
        )
    output = (
        Path(__file__).resolve().parents[1]
        / "doc"
        / "results"
        / "w4a16_cpu_layer_7b.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
