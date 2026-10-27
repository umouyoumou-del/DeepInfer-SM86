from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import load_file

pytest.importorskip("gguf")

from deepinfer.model_executor.models.qwen2_w4a16 import (
    Qwen2W4A16ForCausalLM,
)
from deepinfer.quantization import convert_checkpoint, detect_source_format
from tests.quantization.helpers import make_tiny_gguf


def test_gguf_q4_0_to_sm86_w4a16(tmp_path) -> None:
    source, reference = make_tiny_gguf(tmp_path / "gguf")
    info = detect_source_format(source)
    assert info.format == "gguf"
    target = tmp_path / "target"
    summary = convert_checkpoint(source, target, group_size=32)
    assert summary["source_format"] == "gguf"
    assert summary["architecture"] == "Qwen2ForCausalLM"
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["lm_head_format"] == "dense_fp16"
    config = json.loads((target / "config.json").read_text(encoding="utf-8"))
    assert config["attention_bias"] is True
    assert config["model_type"] == "qwen2"
    assert config["use_sliding_window"] is False
    layer = load_file(target / manifest["layer_files"][0])
    assert layer["self_attn.q_proj.bias"].shape == (64,)
    assert layer["self_attn.k_proj.bias"].shape == (32,)

    model = Qwen2W4A16ForCausalLM.from_checkpoint(
        target, device="cpu", group_size=32
    )
    input_ids = torch.tensor([[1, 5, 9, 13, 17]])
    with torch.no_grad():
        expected = reference(input_ids).logits
        actual = model.forward_logits(input_ids)
    assert tuple(actual.shape) == tuple(expected.shape)
    assert torch.isfinite(actual).all()
    cosine = torch.nn.functional.cosine_similarity(
        actual.flatten().float(), expected.flatten().float(), dim=0
    )
    assert cosine.item() > 0.95


def test_gguf_q4_k_block_dequantization(tmp_path) -> None:
    import numpy as np
    from gguf import GGMLQuantizationType, GGUFWriter

    # One 144-byte Q4_K block: d=1, dmin=0, all eight sub-scales=1, qs=0x11.
    # This dequantizes to 256 values of 1.0 and exercises the K-quant path.
    block = np.zeros(144, dtype=np.uint8)
    block[0:2] = np.array([1.0], dtype=np.float16).view(np.uint8)
    block[4:8] = 1
    block[12:16] = 1
    block[16:] = 0x11
    path = tmp_path / "q4k.gguf"
    writer = GGUFWriter(str(path), "qwen2")
    writer.add_block_count(1)
    writer.add_embedding_length(256)
    writer.add_feed_forward_length(256)
    writer.add_head_count(4)
    writer.add_head_count_kv(2)
    writer.add_key_length(64)
    writer.add_context_length(128)
    writer.add_tensor(
        "token_embd.weight",
        block.reshape(1, 144),
        raw_dtype=GGMLQuantizationType.Q4_K,
    )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    from deepinfer.quantization.gguf_source import GGUFCheckpointSource

    source = GGUFCheckpointSource(detect_source_format(path))
    try:
        tensor = source._tensor("token_embd.weight")
        assert tuple(tensor.shape) == (1, 256)
        assert torch.allclose(tensor, torch.ones_like(tensor))
    finally:
        source.close()
