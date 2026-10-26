"""Export the 3B catgirl W4A16 model into a fast-loading layered checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_qwen2_w4a16_gpu_harness import (
    _install_deepinfer_import_stubs,
    _load_module,
)

DEFAULT_MODEL = Path(r"F:\QwenQLoRA\models\Qwen2.5-3B-Instruct-smallshards")
DEFAULT_ADAPTER = Path(r"F:\QwenQLoRA\train\maibot-catgirl-v3\checkpoint-120")
DEFAULT_OUTPUT = Path(r"D:\Qwen3B-W4A16-SM86")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--group-size", type=int, default=128)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    _install_deepinfer_import_stubs()
    config_module = _load_module(
        "deepinfer.hugging_face.qwen2_config", "hugging_face/qwen2_config.py"
    )
    qwen2_module = _load_module(
        "deepinfer.model_executor.models.qwen2", "model_executor/models/qwen2.py"
    )
    w4_module = _load_module(
        "deepinfer.model_executor.models.qwen2_w4a16",
        "model_executor/models/qwen2_w4a16.py",
    )
    hf = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
    model_cfg = config_module.build_qwen2_model_config(hf, args.model / "config.json")
    dense = qwen2_module.Qwen2ForCausalLM.load_model(
        str(args.model),
        model_cfg,
        adapter_path=args.adapter,
        device="cpu",
    )
    print("dense loaded; converting to W4A16 on GPU", flush=True)
    model = w4_module.Qwen2W4A16ForCausalLM.from_dense_model(
        dense,
        device="cuda",
        group_size=args.group_size,
    )
    del dense
    torch.cuda.empty_cache()

    layer_files = []
    for index, layer in enumerate(model.model.layers):
        payload = {}
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            module = getattr(layer.self_attn, name)
            payload[f"self_attn.{name}.packed"] = module.packed.detach().cpu()
            payload[f"self_attn.{name}.scales"] = module.scales.detach().cpu()
            if module.bias is not None:
                payload[f"self_attn.{name}.bias"] = module.bias.detach().cpu()
        for name in ("gate_proj", "up_proj", "down_proj"):
            module = getattr(layer.mlp, name)
            payload[f"mlp.{name}.packed"] = module.packed.detach().cpu()
            payload[f"mlp.{name}.scales"] = module.scales.detach().cpu()
        payload["input_layernorm.weight"] = layer.input_layernorm.weight.detach().cpu()
        payload["post_attention_layernorm.weight"] = (
            layer.post_attention_layernorm.weight.detach().cpu()
        )
        layer_file = f"layer_{index:02d}.safetensors"
        save_file(payload, output / layer_file)
        layer_files.append(layer_file)
        del payload
        print(f"saved layer {index + 1}/{len(model.model.layers)}", flush=True)

    save_file(
        {"weight": model.model.embed_tokens.weight.detach().to(torch.float16).cpu()},
        output / "embed.safetensors",
    )
    save_file(
        {"weight": model.model.norm.weight.detach().cpu()},
        output / "norm.safetensors",
    )
    if not bool(model.cfg.get("tie_word_embeddings", False)):
        save_file(
            {"weight": model.lm_head.weight.detach().to(torch.float16).cpu()},
            output / "lm_head.safetensors",
        )
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ):
        source = args.model / name
        if source.exists():
            shutil.copy2(source, output / name)
    manifest = {
        "format": "deepinfer_qwen2_w4a16_sm86_v1",
        "backend": "w4a16",
        "model": "Qwen2.5-3B-Instruct + maibot-catgirl-v3",
        "group_size": args.group_size,
        "num_layers": len(layer_files),
        "tie_word_embeddings": bool(model.cfg.get("tie_word_embeddings", False)),
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
