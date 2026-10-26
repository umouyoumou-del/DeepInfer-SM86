"""Quantize Qwen2.5-7B lm_head to 3-bit for the mixed SM86 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

DEFAULT_SOURCE = Path(r"F:\QwenQLoRA\models\Qwen2.5-7B-Instruct")
DEFAULT_CHECKPOINT = Path(r"D:\Qwen7B-Mixed-SM86")
GROUP = 128
CHUNK_ROWS = 8192


def quantize_3bit_chunked(
    weight: torch.Tensor, chunk_rows: int = CHUNK_ROWS
) -> tuple[torch.Tensor, torch.Tensor]:
    out_features, in_features = weight.shape
    row_bytes = in_features * 3 // 8
    packed_chunks: list[torch.Tensor] = []
    scale_chunks: list[torch.Tensor] = []
    for start in range(0, out_features, chunk_rows):
        chunk = weight[start : start + chunk_rows].to(torch.float32)
        rows = chunk.shape[0]
        grouped = chunk.view(rows, in_features // GROUP, GROUP)
        scale = (grouped.abs().amax(dim=-1).clamp_min(1e-8) / 3.0).to(torch.float16)
        q = torch.clamp(torch.round(grouped / scale.float()[:, :, None]), -4, 3).to(
            torch.int8
        )
        q = q.view(rows, in_features)
        q3 = (q & 0x07).to(torch.uint8).view(rows, in_features // 8, 8)
        packed = torch.empty(rows * row_bytes, dtype=torch.uint8)
        packed_rows = packed.view(rows, row_bytes)
        packed_rows[:, 0::3] = (
            q3[:, :, 0] | (q3[:, :, 1] << 3) | ((q3[:, :, 2] & 0x03) << 6)
        )
        packed_rows[:, 1::3] = (
            (q3[:, :, 2] >> 2)
            | (q3[:, :, 3] << 1)
            | (q3[:, :, 4] << 4)
            | ((q3[:, :, 5] & 0x01) << 7)
        )
        packed_rows[:, 2::3] = (
            (q3[:, :, 5] >> 1) | (q3[:, :, 6] << 2) | (q3[:, :, 7] << 5)
        )
        packed_chunks.append(packed)
        scale_chunks.append(scale)
        del chunk, grouped, scale, q, q3, packed, packed_rows
    packed_all = torch.cat(packed_chunks + [torch.zeros(1, dtype=torch.uint8)])
    return packed_all, torch.cat(scale_chunks)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    index = json.loads(
        (args.source / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    shard = args.source / index["weight_map"]["lm_head.weight"]
    with safe_open(shard, framework="pt", device="cpu") as tensors:
        weight = tensors.get_tensor("lm_head.weight")
        packed, scales = quantize_3bit_chunked(weight)
        del weight
    save_file(
        {"packed": packed, "scales": scales}, args.checkpoint / "lm_head3.safetensors"
    )
    manifest_path = args.checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["lm_head_bits"] = 3
    manifest["lm_head_file"] = "lm_head3.safetensors"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "packed_shape": tuple(packed.shape),
                "scales_shape": tuple(scales.shape),
                "packed_bytes": packed.numel(),
                "lm_head_bits": 3,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
