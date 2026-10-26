"""Measure CPU FP16 matvec cost for one Qwen2.5-7B decoder layer."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch

HIDDEN = 3584
INTERMEDIATE = 18944
SHAPES = {
    "q_proj": (HIDDEN, HIDDEN),
    "k_proj": (512, HIDDEN),
    "v_proj": (512, HIDDEN),
    "o_proj": (HIDDEN, HIDDEN),
    "gate_proj": (INTERMEDIATE, HIDDEN),
    "up_proj": (INTERMEDIATE, HIDDEN),
    "down_proj": (HIDDEN, INTERMEDIATE),
}


def main() -> int:
    torch.set_num_threads(6)
    torch.manual_seed(0)
    result = {
        "hidden": HIDDEN,
        "intermediate": INTERMEDIATE,
        "threads": torch.get_num_threads(),
        "dtype": "float16",
        "shapes": {},
    }
    total_ms = 0.0
    for name, (out_features, in_features) in SHAPES.items():
        weight = torch.randn(out_features, in_features, dtype=torch.float16)
        inputs = torch.randn(in_features, dtype=torch.float16)
        samples = []
        for _ in range(6):
            started = time.perf_counter()
            torch.nn.functional.linear(inputs, weight)
            samples.append(time.perf_counter() - started)
        median_ms = statistics.median(samples) * 1000
        total_ms += median_ms
        result["shapes"][name] = {
            "out_features": out_features,
            "in_features": in_features,
            "median_ms": round(median_ms, 3),
        }
        print(name, result["shapes"][name], flush=True)
        del weight, inputs
    result["layer_total_ms"] = round(total_ms, 3)
    result["layer_tokens_per_second_ceiling"] = round(1000.0 / total_ms, 3)
    result["offload_8_layers_tokens_per_second_ceiling"] = round(
        1000.0 / (total_ms * 8), 3
    )
    output = (
        Path(__file__).resolve().parents[1]
        / "doc"
        / "results"
        / "qwen7b_cpu_fp16_layer.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
