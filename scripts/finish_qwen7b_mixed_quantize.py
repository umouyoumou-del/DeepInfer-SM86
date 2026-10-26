"""Finish the Qwen2.5-7B mixed checkpoint after layer quantization.

The 28 decoder layers are already on disk.  Embedding and lm_head are 545M
parameters each; quantize them in row chunks so a full float32 copy does not
need to fit in host RAM.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

DEFAULT_SOURCE = Path(r"F:\QwenQLoRA\models\Qwen2.5-7B-Instruct")
DEFAULT_OUTPUT = Path(r"D:\Qwen7B-Mixed-SM86")
GROUP = 128
CHUNK_ROWS = 8192


def quantize_4bit_chunked(
    weight: torch.Tensor, chunk_rows: int = CHUNK_ROWS
) -> tuple[torch.Tensor, torch.Tensor]:
    out_features, in_features = weight.shape
    packed_chunks: list[torch.Tensor] = []
    scale_chunks: list[torch.Tensor] = []
    for start in range(0, out_features, chunk_rows):
        chunk = weight[start : start + chunk_rows].to(torch.float32)
        grouped = chunk.view(chunk.shape[0], in_features // GROUP, GROUP)
        scale = (grouped.abs().amax(dim=-1).clamp_min(1e-8) / 7.0).to(torch.float16)
        q = torch.clamp(torch.round(grouped / scale.float()[:, :, None]), -8, 7).to(
            torch.int8
        )
        q = q.view(chunk.shape[0], in_features)
        low = (q[:, 0::2] & 0x0F).to(torch.uint8)
        high = ((q[:, 1::2] & 0x0F) << 4).to(torch.uint8)
        packed_chunks.append(low | high)
        scale_chunks.append(scale)
        del chunk, grouped, scale, q, low, high
    return torch.cat(packed_chunks, dim=0), torch.cat(scale_chunks, dim=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source, output = args.source, args.output
    index = json.loads(
        (source / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    weight_map = index["weight_map"]

    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    ):
        src = source / name
        if src.exists() and not (output / name).exists():
            shutil.copy2(src, output / name)

    def quantize_key(key: str, filename: str) -> None:
        shard = source / weight_map[key]
        with safe_open(shard, framework="pt", device="cpu") as tensors:
            weight = tensors.get_tensor(key)
            packed, scales = quantize_4bit_chunked(weight)
            del weight
        save_file({"packed": packed, "scales": scales}, output / filename)
        print(
            f"{filename} packed={tuple(packed.shape)} scales={tuple(scales.shape)}",
            flush=True,
        )
        del packed, scales

    quantize_key("model.embed_tokens.weight", "embed.safetensors")
    quantize_key("lm_head.weight", "lm_head.safetensors")

    shard = source / weight_map["model.norm.weight"]
    with safe_open(shard, framework="pt", device="cpu") as tensors:
        norm = tensors.get_tensor("model.norm.weight").to(torch.float16)
    save_file({"weight": norm}, output / "norm.safetensors")

    layer_files = [f"layer_{layer:02d}.safetensors" for layer in range(28)]
    missing = [name for name in layer_files if not (output / name).exists()]
    if missing:
        raise SystemExit(f"missing layer files: {missing}")
    manifest = {
        "format": "deepinfer_qwen2_mixed_w4w3_v1",
        "model": "Qwen2.5-7B-Instruct",
        "group_size": GROUP,
        "num_layers": 28,
        "attention_bits": 4,
        "mlp_bits": 3,
        "embed_bits": 4,
        "lm_head_bits": 4,
        "layer_files": layer_files,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    total = sum(path.stat().st_size for path in output.glob("*.safetensors"))
    print(f"done total_safetensors_gib={total / 1024**3:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
