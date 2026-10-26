from __future__ import annotations

import torch
from transformers import LlamaConfig
from transformers import LlamaForCausalLM as HFLlamaForCausalLM

from deepinfer.hugging_face import load_hf_model_config
from deepinfer.model_executor.models import ModelRegistry
from deepinfer.model_executor.models.llama import LlamaForCausalLM


def test_llama_spec_and_forward_match_hf(tmp_path):
    config = LlamaConfig(
        vocab_size=512,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        attention_bias=False,
        hidden_act="silu",
    )
    torch.manual_seed(123)
    hf = HFLlamaForCausalLM(config).eval()
    hf.save_pretrained(tmp_path, safe_serialization=True)

    model_cfg = load_hf_model_config(tmp_path / "config.json")
    assert model_cfg.architecture == "LlamaForCausalLM"
    assert model_cfg.gpt.model_type == "llama"
    assert model_cfg.gpt.attention_bias is False
    assert model_cfg.gpt.num_key_value_heads == 2

    assert "LlamaForCausalLM" in ModelRegistry.get_supported_archs()
    assert ModelRegistry.load_model_cls("LlamaForCausalLM") is LlamaForCausalLM

    model = LlamaForCausalLM.load_model(str(tmp_path), model_cfg, device="cpu")
    input_ids = torch.tensor([[1, 5, 9, 11, 23, 42]])
    with torch.no_grad():
        expected = hf(input_ids).logits
        actual = model.forward_logits(input_ids)
    assert tuple(actual.shape) == tuple(expected.shape)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
