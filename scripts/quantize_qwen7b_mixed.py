"""Stream Qwen2.5-7B HF shards into a mixed W4/W3 DeepInfer checkpoint.

The BF16 model is never materialized as a whole: one tensor is read and
quantized at a time.  Output uses:
  * 4-bit group-128 attention, embedding and lm_head;
  * 3-bit group-128 MLP (8 signed values packed in 3 bytes).
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

DEFAULT_SOURCE = Path(r"F:\QwenQLoRA\models\Qwen2.5-7B-Instruct")
DEFAULT_OUTPUT = Path(r"D:\Qwen7B-Mixed-SM86")
GROUP = 128


def quantize_4bit(
    weight: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    out_features, in_features = weight.shape
    w = weight.to(device=device, dtype=torch.float32)
    grouped = w.view(out_features, in_features // GROUP, GROUP)
    scale = (grouped.abs().amax(dim=-1).clamp_min(1e-8) / 7.0).to(torch.float16)
    q = torch.clamp(torch.round(grouped / scale.float()[:, :, None]), -8, 7).to(
        torch.int8
    )
    q = q.view(out_features, in_features)
    low = (q[:, 0::2] & 0x0F).to(torch.uint8)
    high = ((q[:, 1::2] & 0x0F) << 4).to(torch.uint8)
    packed = (low | high).to("cpu")
    return packed, scale.to("cpu")


def quantize_3bit(
    weight: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    out_features, in_features = weight.shape
    w = weight.to(device=device, dtype=torch.float32)
    grouped = w.view(out_features, in_features // GROUP, GROUP)
    scale = (grouped.abs().amax(dim=-1).clamp_min(1e-8) / 3.0).to(torch.float16)
    q = torch.clamp(torch.round(grouped / scale.float()[:, :, None]), -4, 3).to(
        torch.int8
    )
    q = q.view(out_features, in_features)
    q3 = (q & 0x07).to(torch.uint8).view(out_features, in_features // 8, 8)
    row_bytes = in_features * 3 // 8
    packed = torch.empty(out_features * row_bytes + 1, dtype=torch.uint8, device=device)
    rows = packed[:-1].view(out_features, row_bytes)
    rows[:, 0::3] = q3[:, :, 0] | (q3[:, :, 1] << 3) | ((q3[:, :, 2] & 0x03) << 6)
    rows[:, 1::3] = (
        (q3[:, :, 2] >> 2)
        | (q3[:, :, 3] << 1)
        | (q3[:, :, 4] << 4)
        | ((q3[:, :, 5] & 0x01) << 7)
    )
    rows[:, 2::3] = (q3[:, :, 5] >> 1) | (q3[:, :, 6] << 2) | (q3[:, :, 7] << 5)
    packed[-1] = 0
    return packed.to("cpu"), scale.to("cpu")


def layer_tensor(tensors: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    if key not in tensors:
        raise KeyError(key)
    return tensors[key]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quantize Qwen2.5-7B into DeepInfer mixed SM86 checkpoint"
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    source = args.source
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    index = json.loads(
        (source / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    weight_map = index["weight_map"]
    shard_paths = sorted({source / shard for shard in set(weight_map.values())})
    print(f"shards={len(shard_paths)} output={output}", flush=True)

    with ExitStack() as stack:
        shards = {
            path.name: stack.enter_context(
                safe_open(path, framework="pt", device="cpu")
            )
            for path in shard_paths
        }

        def get(key: str) -> torch.Tensor:
            return shards[weight_map[key]].get_tensor(key)

        # Copy the small tokenizer/config files.
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
            if src.exists():
                shutil.copy2(src, output / name)

        layer_files: list[str] = []
        for layer in range(28):
            started = time.perf_counter()
            prefix = f"model.layers.{layer}"
            payload: dict[str, torch.Tensor] = {}
            for module, bits in (
                ("self_attn.q_proj", 4),
                ("self_attn.k_proj", 4),
                ("self_attn.v_proj", 4),
                ("self_attn.o_proj", 4),
                ("mlp.gate_proj", 3),
                ("mlp.up_proj", 3),
                ("mlp.down_proj", 3),
            ):
                weight = get(f"{prefix}.{module}.weight")
                packed, scales = (quantize_4bit if bits == 4 else quantize_3bit)(
                    weight, torch.device(args.device)
                )
                payload[f"{module}.packed"] = packed
                payload[f"{module}.scales"] = scales
                bias_key = f"{prefix}.{module}.bias"
                if bias_key in weight_map:
                    payload[f"{module}.bias"] = get(bias_key).to(torch.float16)
                del weight
            for norm in ("input_layernorm", "post_attention_layernorm"):
                payload[f"{norm}.weight"] = get(f"{prefix}.{norm}.weight").to(
                    torch.float16
                )
            layer_file = f"layer_{layer:02d}.safetensors"
            save_file(payload, output / layer_file)
            layer_files.append(layer_file)
            del payload
            torch.cuda.empty_cache()
            print(
                f"layer {layer + 1}/28 {time.perf_counter() - started:.2f}s", flush=True
            )

        for filename, key, bits in (
            ("embed.safetensors", "model.embed_tokens.weight", 4),
            ("lm_head.safetensors", "lm_head.weight", 4),
        ):
            started = time.perf_counter()
            weight = get(key)
            packed, scales = (quantize_4bit if bits == 4 else quantize_3bit)(
                weight, torch.device("cpu")
            )
            save_file({"packed": packed, "scales": scales}, output / filename)
            del weight, packed, scales
            print(f"{filename} {time.perf_counter() - started:.2f}s", flush=True)

        norm = get("model.norm.weight").to(torch.float16)
        save_file({"weight": norm}, output / "norm.safetensors")
        del norm

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
    total_bytes = sum(path.stat().st_size for path in output.glob("*.safetensors"))
    print(f"done total_safetensors_gib={total_bytes / 1024**3:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
