from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import load_file

from deepinfer.model_executor.models.qwen2_w4a16 import (
    Qwen2W4A16ForCausalLM,
)
from deepinfer.quantization import convert_checkpoint, detect_source_format
from tests.quantization.helpers import make_hf_quant_checkpoint


@pytest.mark.parametrize("quant_format", ["gptq", "awq"])
def test_gptq_awq_to_sm86_w4a16(tmp_path, quant_format: str) -> None:
    source, reference = make_hf_quant_checkpoint(
        tmp_path / "source", quant_format=quant_format
    )
    info = detect_source_format(source)
    assert info.format == quant_format
    with pytest.raises(ValueError, match="ancestors"):
        convert_checkpoint(source, source, group_size=32, overwrite=True)
    assert info.bits == 4
    assert info.group_size == 32

    target = tmp_path / "target"
    summary = convert_checkpoint(
        source,
        target,
        group_size=32,
    )
    assert summary["source_format"] == quant_format
    assert summary["architecture"] == "Qwen2ForCausalLM"
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "deepinfer_w4a16_sm86_v1"
    assert manifest["source_format"] == quant_format
    assert manifest["num_layers"] == 1
    config = json.loads((target / "config.json").read_text(encoding="utf-8"))
    assert "quantization_config" not in config
    assert config["torch_dtype"] == "float16"

    layer = load_file(target / manifest["layer_files"][0])
    assert "self_attn.q_proj.packed" in layer
    assert "self_attn.q_proj.scales" in layer
    assert "self_attn.q_proj.bias" in layer
    assert layer["self_attn.q_proj.scales"].dtype == torch.float16

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
    assert cosine.item() > 0.98
