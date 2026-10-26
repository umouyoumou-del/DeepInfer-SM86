"""Build a pure 4-bit Q4 checkpoint for Qwen2.5-7B.

Attention/embedding are copied from the existing mixed checkpoint; MLP and
lm_head are re-quantized from the original BF16 shards to 4-bit.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.finish_qwen7b_mixed_quantize import quantize_4bit_chunked
from scripts.quantize_qwen7b_mixed import quantize_4bit

DEFAULT_SOURCE = Path(r"F:\QwenQLoRA\models\Qwen2.5-7B-Instruct")
DEFAULT_MIXED = Path(r"D:\Qwen7B-Mixed-SM86")
DEFAULT_OUTPUT = Path(r"D:\Qwen7B-Q4-SM86")
GROUP = 128


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--mixed", type=Path, default=DEFAULT_MIXED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    index = json.loads(
        (args.source / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    weight_map = index["weight_map"]
    shard_paths = sorted({args.source / shard for shard in set(weight_map.values())})
    device = torch.device("cuda")

    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ):
        src = args.source / name
        if src.exists():
            shutil.copy2(src, output / name)

    with ExitStack() as stack:
        shards = {
            path.name: stack.enter_context(
                safe_open(path, framework="pt", device="cpu")
            )
            for path in shard_paths
        }

        def get(key: str) -> torch.Tensor:
            return shards[weight_map[key]].get_tensor(key)

        layer_files: list[str] = []
        for layer in range(28):
            prefix = f"model.layers.{layer}"
            mixed_file = args.mixed / f"layer_{layer:02d}.safetensors"
            with safe_open(mixed_file, framework="pt", device="cpu") as old:
                payload = {
                    key: old.get_tensor(key)
                    for key in (
                        "self_attn.q_proj.packed",
                        "self_attn.q_proj.scales",
                        "self_attn.q_proj.bias",
                        "self_attn.k_proj.packed",
                        "self_attn.k_proj.scales",
                        "self_attn.k_proj.bias",
                        "self_attn.v_proj.packed",
                        "self_attn.v_proj.scales",
                        "self_attn.v_proj.bias",
                        "self_attn.o_proj.packed",
                        "self_attn.o_proj.scales",
                        "input_layernorm.weight",
                        "post_attention_layernorm.weight",
                    )
                }
            for module in ("gate_proj", "up_proj", "down_proj"):
                weight = get(f"{prefix}.mlp.{module}.weight")
                packed, scales = quantize_4bit(weight, device)
                payload[f"mlp.{module}.packed"] = packed
                payload[f"mlp.{module}.scales"] = scales
                del weight, packed, scales
            layer_file = f"layer_{layer:02d}.safetensors"
            save_file(payload, output / layer_file)
            layer_files.append(layer_file)
            del payload
            torch.cuda.empty_cache()
            print(f"layer {layer + 1}/28", flush=True)

        for name in ("embed.safetensors", "norm.safetensors"):
            shutil.copy2(args.mixed / name, output / name)

        head_weight = get("lm_head.weight")
        head_packed, head_scales = quantize_4bit_chunked(head_weight)
        save_file(
            {"packed": head_packed, "scales": head_scales},
            output / "lm_head.safetensors",
        )
        del head_weight, head_packed, head_scales

    manifest = {
        "format": "deepinfer_qwen2_q4_sm86_v1",
        "model": "Qwen2.5-7B-Instruct",
        "group_size": GROUP,
        "num_layers": 28,
        "attention_bits": 4,
        "mlp_bits": 4,
        "embed_bits": 4,
        "lm_head_bits": 4,
        "lm_head_file": "lm_head.safetensors",
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
