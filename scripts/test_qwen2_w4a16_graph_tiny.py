"""Validate native GQA decode + static KV cache + CUDA Graph on a tiny Qwen2."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_qwen2_w4a16_gpu_harness import (
    _install_deepinfer_import_stubs,
    _load_module,
)


def main() -> int:
    _install_deepinfer_import_stubs()
    config_module = _load_module(
        "deepinfer.hugging_face.qwen2_config", "hugging_face/qwen2_config.py"
    )
    qwen2_module = _load_module(
        "deepinfer.model_executor.models.qwen2", "model_executor/models/qwen2.py"
    )
    w4a16_module = _load_module(
        "deepinfer.model_executor.models.qwen2_w4a16",
        "model_executor/models/qwen2_w4a16.py",
    )

    hf = {
        "model_type": "qwen2",
        "architectures": ["Qwen2ForCausalLM"],
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "num_hidden_layers": 2,
        "intermediate_size": 256,
        "vocab_size": 256,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
        "torch_dtype": "float16",
    }
    cfg = config_module.build_qwen2_model_config(hf, ROOT / "tiny-config.json")
    dense = qwen2_module.Qwen2ForCausalLM.load_model(
        "unused", cfg, skip_load=True, device="cpu"
    )
    model = w4a16_module.Qwen2W4A16ForCausalLM.from_dense_model(
        dense, device="cuda", group_size=128
    )
    del dense

    max_seq = 128
    prompt_len = 4
    torch.manual_seed(7)
    input_ids = torch.randint(0, 256, (1, prompt_len), device="cuda", dtype=torch.long)
    prompt_positions = torch.arange(
        prompt_len, device="cuda", dtype=torch.long
    ).unsqueeze(0)
    dynamic_cache = model.new_kv_cache()
    with torch.inference_mode():
        prompt_logits = model.forward_logits(
            input_ids, prompt_positions, kv_cache=dynamic_cache
        )
        first_token = int(prompt_logits[:, -1].argmax(dim=-1).item())
    torch.cuda.synchronize()

    def build_static_cache() -> list:
        result = []
        for layer_cache in dynamic_cache:
            assert (
                layer_cache is not None
                and layer_cache.key is not None
                and layer_cache.value is not None
            )
            static = qwen2_module.Qwen2StaticLayerKVCache(
                key=torch.zeros(
                    (1, layer_cache.key.shape[1], max_seq, layer_cache.key.shape[3]),
                    dtype=layer_cache.key.dtype,
                    device="cuda",
                ),
                value=torch.zeros(
                    (
                        1,
                        layer_cache.value.shape[1],
                        max_seq,
                        layer_cache.value.shape[3],
                    ),
                    dtype=layer_cache.value.dtype,
                    device="cuda",
                ),
            )
            static.key[:, :, :prompt_len, :].copy_(layer_cache.key)
            static.value[:, :, :prompt_len, :].copy_(layer_cache.value)
            result.append(static)
        return result

    def clone_static_cache(source: list) -> list:
        return [
            qwen2_module.Qwen2StaticLayerKVCache(
                key=cache.key.clone(), value=cache.value.clone()
            )
            for cache in source
        ]

    token = torch.tensor([[first_token]], dtype=torch.long, device="cuda")
    position = torch.tensor([[prompt_len]], dtype=torch.long, device="cuda")
    mask = torch.zeros((1, max_seq), dtype=torch.bool, device="cuda")
    mask[:, : prompt_len + 1] = True

    native_cache = build_static_cache()
    with torch.no_grad():
        native_logits = model.forward_logits(
            token,
            position,
            kv_cache=native_cache,
            attention_mask=mask,
        )
        torch.cuda.synchronize()
    native_token = int(native_logits[:, -1].argmax(dim=-1).item())

    sdpa_cache = build_static_cache()
    for layer in model.model.layers:
        layer.self_attn.native_gqa_decode = None
    with torch.no_grad():
        sdpa_logits = model.forward_logits(
            token,
            position,
            kv_cache=sdpa_cache,
            attention_mask=mask,
        )
        torch.cuda.synchronize()
    for layer in model.model.layers:
        layer.self_attn.native_gqa_decode = w4a16_module._native_gqa_decode
    cosine = torch.nn.functional.cosine_similarity(
        native_logits.flatten().float(),
        sdpa_logits.flatten().float(),
        dim=0,
    ).item()
    max_abs_diff = (native_logits.float() - sdpa_logits.float()).abs().max().item()

    # native_cache now contains the first decode token.  Run a reference step
    # for the next token, then capture the same step in a CUDA Graph.
    token.fill_(native_token)
    position.fill_(prompt_len + 1)
    mask.zero_()
    mask[:, : prompt_len + 2] = True
    reference_cache = clone_static_cache(native_cache)
    graph_cache = clone_static_cache(native_cache)
    with torch.no_grad():
        reference_logits = model.forward_logits(
            token,
            position,
            kv_cache=reference_cache,
            attention_mask=mask,
        )
        torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(graph):
        graph_logits = model.forward_logits(
            token,
            position,
            kv_cache=graph_cache,
            attention_mask=mask,
        )
    torch.cuda.synchronize()
    # Capture records but does not execute CuPy kernels; the first replay
    # produces the real output and writes the current-token KV slot.
    graph.replay()
    torch.cuda.synchronize()
    graph_max_diff = (
        (graph_logits.float() - reference_logits.float()).abs().max().item()
    )
    cache_key_diff = max(
        (reference_cache[i].key.float() - graph_cache[i].key.float()).abs().max().item()
        for i in range(len(reference_cache))
    )
    cache_value_diff = max(
        (reference_cache[i].value.float() - graph_cache[i].value.float())
        .abs()
        .max()
        .item()
        for i in range(len(reference_cache))
    )
    graph_cosine = torch.nn.functional.cosine_similarity(
        graph_logits.flatten().float(),
        reference_logits.flatten().float(),
        dim=0,
    ).item()
    slot_diffs = [
        (reference_cache[i].key.float() - graph_cache[i].key.float())
        .abs()
        .amax(dim=(0, 1, 3))
        for i in range(len(reference_cache))
    ]
    argmax_slots = [int(row.argmax().item()) for row in slot_diffs]
    print(
        "diagnostics",
        {
            "cache_key_diff": cache_key_diff,
            "cache_value_diff": cache_value_diff,
            "graph_cosine": graph_cosine,
            "argmax_slots": argmax_slots,
            "argmax_slot_values": [
                round(float(row[slot]), 6)
                for row, slot in zip(slot_diffs, argmax_slots, strict=False)
            ],
        },
    )

    next_token = int(graph_logits[:, -1].argmax(dim=-1).item())
    token.fill_(next_token)
    position.fill_(prompt_len + 2)
    mask.zero_()
    mask[:, : prompt_len + 3] = True
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(graph_logits).all()

    print(
        {
            "status": "passed",
            "prompt_tokens": prompt_len,
            "native_vs_sdpa_cosine": round(cosine, 6),
            "native_vs_sdpa_max_abs_diff": round(max_abs_diff, 6),
            "graph_vs_native_max_abs_diff": round(graph_max_diff, 6),
            "next_token": next_token,
            "max_seq": max_seq,
        }
    )
    if cosine < 0.99:
        raise SystemExit("native GQA cosine below 0.99")
    if graph_max_diff > 0.02:
        raise SystemExit("CUDA Graph native GQA logits drift above 0.02")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
