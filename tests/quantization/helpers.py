from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from transformers import Qwen2Config, Qwen2ForCausalLM

AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]


def tiny_qwen(
    *,
    hidden_size: int = 64,
    intermediate_size: int = 128,
    vocab_size: int = 128,
    num_heads: int = 4,
    num_kv_heads: int = 2,
    head_dim: int = 16,
) -> tuple[Qwen2Config, Qwen2ForCausalLM]:
    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=1,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        attention_dropout=0.0,
        torch_dtype="float32",
    )
    torch.manual_seed(20261027)
    return config, Qwen2ForCausalLM(config).eval()


def _group_scales(
    weight: torch.Tensor,
    group_size: int,
    g_idx: torch.Tensor,
) -> torch.Tensor:
    out_features = int(weight.shape[0])
    groups = int(g_idx.max()) + 1
    scales = torch.zeros((groups, out_features), dtype=torch.float32)
    transposed = weight.T.contiguous()
    for group in range(groups):
        rows = transposed[g_idx == group]
        scales[group] = rows.abs().amax(dim=0).clamp_min(1e-8) / 7.0
    return scales


def pack_gptq_weight(
    weight: torch.Tensor,
    *,
    bits: int = 4,
    group_size: int = 32,
    g_idx: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    out_features, in_features = (int(value) for value in weight.shape)
    if g_idx is None:
        g_idx = torch.arange(in_features, dtype=torch.long) // group_size
    else:
        g_idx = g_idx.to(torch.long)
    groups = int(g_idx.max()) + 1
    scales = _group_scales(weight, group_size, g_idx)
    zeros = torch.full((groups, out_features), float(1 << (bits - 1)))
    quantized = torch.round(
        weight.T / scales[g_idx] + zeros[g_idx]
    ).clamp(0, (1 << bits) - 1).to(torch.int32)
    pack_num = 32 // bits
    qweight = torch.zeros(
        (in_features // pack_num, out_features), dtype=torch.int32
    )
    for row in range(in_features // pack_num):
        for column in range(pack_num):
            qweight[row] |= quantized[row * pack_num + column] << (
                bits * column
            )
    qzeros = torch.zeros((groups, out_features // pack_num), dtype=torch.int32)
    stored_zeros = (zeros - 1).to(torch.int32)
    for row in range(groups):
        for column in range(out_features // pack_num):
            for item in range(pack_num):
                qzeros[row, column] |= stored_zeros[
                    row, column * pack_num + item
                ] << (bits * item)
    return qweight, scales.to(torch.float16), qzeros, g_idx.to(torch.int32)


def pack_awq_weight(
    weight: torch.Tensor,
    *,
    group_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out_features, in_features = (int(value) for value in weight.shape)
    groups = in_features // group_size
    g_idx = torch.arange(in_features, dtype=torch.long) // group_size
    scales = _group_scales(weight, group_size, g_idx)
    zeros = torch.full((groups, out_features), 8.0)
    quantized = torch.round(
        weight.T / scales[g_idx] + zeros[g_idx]
    ).clamp(0, 15).to(torch.int32)
    pack_num = 8
    qweight = torch.zeros(
        (in_features, out_features // pack_num), dtype=torch.int32
    )
    qzeros = torch.zeros((groups, out_features // pack_num), dtype=torch.int32)
    for column in range(out_features // pack_num):
        for item in range(pack_num):
            source_column = column * pack_num + AWQ_ORDER[item]
            qweight[:, column] |= quantized[:, source_column] << (4 * item)
            qzeros[:, column] |= zeros[:, source_column].to(torch.int32) << (
                4 * item
            )
    return qweight, scales.to(torch.float16), qzeros

def make_hf_quant_checkpoint(
    root: Path,
    *,
    quant_format: str,
) -> tuple[Path, Qwen2ForCausalLM]:
    root.mkdir(parents=True, exist_ok=True)
    _config, model = tiny_qwen()
    model.save_pretrained(root, safe_serialization=True)
    config_payload = json.loads((root / "config.json").read_text(encoding="utf-8"))
    config_payload["torch_dtype"] = "float32"
    config_payload["tie_word_embeddings"] = False
    if quant_format == "gptq":
        config_payload["quantization_config"] = {
            "quant_method": "gptq",
            "bits": 4,
            "group_size": 32,
            "sym": True,
            "desc_act": False,
        }
    elif quant_format == "awq":
        config_payload["quantization_config"] = {
            "quant_method": "awq",
            "bits": 4,
            "group_size": 32,
            "version": "gemm",
            "zero_point": True,
        }
    else:
        raise ValueError(quant_format)
    (root / "config.json").write_text(
        json.dumps(config_payload, indent=2), encoding="utf-8"
    )

    state: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": model.model.embed_tokens.weight.detach().cpu(),
        "model.norm.weight": model.model.norm.weight.detach().cpu(),
        "lm_head.weight": model.lm_head.weight.detach().cpu(),
    }
    for layer_index, layer in enumerate(model.model.layers):
        base = f"model.layers.{layer_index}"
        state[f"{base}.input_layernorm.weight"] = (
            layer.input_layernorm.weight.detach().cpu()
        )
        state[f"{base}.post_attention_layernorm.weight"] = (
            layer.post_attention_layernorm.weight.detach().cpu()
        )
        modules = {
            "self_attn.q_proj": layer.self_attn.q_proj,
            "self_attn.k_proj": layer.self_attn.k_proj,
            "self_attn.v_proj": layer.self_attn.v_proj,
            "self_attn.o_proj": layer.self_attn.o_proj,
            "mlp.gate_proj": layer.mlp.gate_proj,
            "mlp.up_proj": layer.mlp.up_proj,
            "mlp.down_proj": layer.mlp.down_proj,
        }
        for short_name, module in modules.items():
            prefix = f"{base}.{short_name}"
            weight = module.weight.detach().cpu()
            if quant_format == "gptq":
                qweight, scales, qzeros, g_idx = pack_gptq_weight(weight)
                state[f"{prefix}.qweight"] = qweight
                state[f"{prefix}.scales"] = scales
                state[f"{prefix}.qzeros"] = qzeros
                state[f"{prefix}.g_idx"] = g_idx
            else:
                qweight, scales, qzeros = pack_awq_weight(weight)
                state[f"{prefix}.qweight"] = qweight
                state[f"{prefix}.scales"] = scales
                state[f"{prefix}.qzeros"] = qzeros
            if module.bias is not None:
                state[f"{prefix}.bias"] = module.bias.detach().cpu()
    save_file(state, str(root / "model.safetensors"))
    return root, model


def quant_tensor(
    writer: Any,
    name: str,
    tensor: torch.Tensor,
    quant_type: Any,
) -> None:
    from gguf.quants import quantize

    array = tensor.detach().cpu().numpy().astype("float32")
    packed = quantize(array, quant_type)
    writer.add_tensor(name, packed, raw_dtype=quant_type)


def f32_tensor(writer: Any, name: str, tensor: torch.Tensor) -> None:
    writer.add_tensor(name, tensor.detach().cpu().numpy().astype("float32"))


def make_tiny_gguf(
    root: Path,
    *,
    quant_type_name: str = "Q4_0",
    hidden_size: int = 64,
    intermediate_size: int = 128,
    vocab_size: int = 128,
    num_heads: int = 4,
    num_kv_heads: int = 2,
    head_dim: int = 16,
) -> tuple[Path, Qwen2ForCausalLM]:
    from gguf import GGMLQuantizationType, GGUFWriter

    quant_type = getattr(GGMLQuantizationType, quant_type_name)
    root.mkdir(parents=True, exist_ok=True)
    _, model = tiny_qwen(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    path = root / "tiny-qwen2.gguf"
    writer = GGUFWriter(str(path), "qwen2")
    writer.add_name("tiny-qwen2")
    writer.add_block_count(1)
    writer.add_context_length(128)
    writer.add_embedding_length(hidden_size)
    writer.add_feed_forward_length(intermediate_size)
    writer.add_head_count(num_heads)
    writer.add_head_count_kv(num_kv_heads)
    writer.add_key_length(head_dim)
    writer.add_value_length(head_dim)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_rope_freq_base(10000.0)
    writer.add_vocab_size(vocab_size)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_pad_token_id(0)

    quant_tensor(
        writer, "token_embd.weight", model.model.embed_tokens.weight, quant_type
    )
    quant_tensor(writer, "output.weight", model.lm_head.weight, quant_type)
    f32_tensor(writer, "output_norm.weight", model.model.norm.weight)
    layer = model.model.layers[0]
    f32_tensor(writer, "blk.0.attn_norm.weight", layer.input_layernorm.weight)
    f32_tensor(
        writer, "blk.0.ffn_norm.weight", layer.post_attention_layernorm.weight
    )
    for gguf_name, module in (
        ("blk.0.attn_q.weight", layer.self_attn.q_proj),
        ("blk.0.attn_k.weight", layer.self_attn.k_proj),
        ("blk.0.attn_v.weight", layer.self_attn.v_proj),
        ("blk.0.attn_output.weight", layer.self_attn.o_proj),
        ("blk.0.ffn_gate.weight", layer.mlp.gate_proj),
        ("blk.0.ffn_up.weight", layer.mlp.up_proj),
        ("blk.0.ffn_down.weight", layer.mlp.down_proj),
    ):
        quant_tensor(writer, gguf_name, module.weight, quant_type)
        if module.bias is not None:
            bias_name = gguf_name.removesuffix(".weight") + ".bias"
            f32_tensor(writer, bias_name, module.bias)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path, model
