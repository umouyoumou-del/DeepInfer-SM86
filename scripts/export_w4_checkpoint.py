"""Export any supported dense GQA model to a layered SM86 W4A16 checkpoint."""

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=128)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    _install_deepinfer_import_stubs()
    spec_module = _load_module("deepinfer.model_spec", "model_spec.py")
    registry_module = _load_module(
        "deepinfer.model_executor.models.registry",
        "model_executor/models/__init__.py",
    )
    hf = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
    model_cfg = spec_module.build_model_config(hf, args.model / "config.json")
    architecture = str(model_cfg.architecture)
    source_cls = registry_module.ModelRegistry.load_model_cls(architecture)
    if source_cls is None:
        raise SystemExit(f"unsupported architecture: {architecture}")
    dense = source_cls.load_model(
        str(args.model),
        model_cfg,
        adapter_path=args.adapter,
        device="cpu",
    )
    if architecture == "LlamaForCausalLM":
        from deepinfer.model_executor.models.llama_w4a16 import (
            LlamaW4A16ForCausalLM as QuantClass,
        )
    elif architecture == "Qwen2ForCausalLM":
        from deepinfer.model_executor.models.qwen2_w4a16 import (
            Qwen2W4A16ForCausalLM as QuantClass,
        )
    else:
        raise SystemExit(f"no W4A16 class registered for {architecture}")

    model = QuantClass.from_dense_model(
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
        save_file(payload, args.output / layer_file)
        layer_files.append(layer_file)
        del payload
        print(f"saved layer {index + 1}/{len(model.model.layers)}", flush=True)

    save_file(
        {"weight": model.model.embed_tokens.weight.detach().to(torch.float16).cpu()},
        args.output / "embed.safetensors",
    )
    save_file(
        {"weight": model.model.norm.weight.detach().cpu()},
        args.output / "norm.safetensors",
    )
    if not bool(model.cfg.get("tie_word_embeddings", False)):
        save_file(
            {"weight": model.lm_head.weight.detach().to(torch.float16).cpu()},
            args.output / "lm_head.safetensors",
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
            shutil.copy2(source, args.output / name)
    manifest = {
        "format": "deepinfer_w4a16_sm86_v1",
        "backend": "w4a16",
        "architecture": architecture,
        "model_type": model_cfg.model_type,
        "group_size": args.group_size,
        "num_layers": len(layer_files),
        "tie_word_embeddings": bool(model.cfg.get("tie_word_embeddings", False)),
        "layer_files": layer_files,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    total = sum(path.stat().st_size for path in args.output.glob("*.safetensors"))
    print(
        f"done architecture={architecture} total_safetensors_gib={total / 1024**3:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
