from __future__ import annotations

import torch
from transformers import MistralConfig
from transformers import MistralForCausalLM as HFMistralForCausalLM

from deepinfer.hugging_face import load_hf_model_config
from deepinfer.model_executor.models import ModelRegistry
from deepinfer.model_executor.models.mistral import MistralForCausalLM


def test_mistral_sliding_window_spec_and_forward_match_hf(tmp_path) -> None:
    config = MistralConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        sliding_window=8,
        tie_word_embeddings=False,
        attention_dropout=0.0,
        torch_dtype="float32",
    )
    torch.manual_seed(20261027)
    reference = HFMistralForCausalLM(config).eval()
    reference.save_pretrained(tmp_path, safe_serialization=True)

    model_cfg = load_hf_model_config(tmp_path / "config.json")
    assert model_cfg.architecture == "MistralForCausalLM"
    assert model_cfg.gpt.use_sliding_window is True
    assert model_cfg.gpt.sliding_window == 8
    assert "MistralForCausalLM" in ModelRegistry.get_supported_archs()
    assert ModelRegistry.load_model_cls("MistralForCausalLM") is MistralForCausalLM

    model = MistralForCausalLM.load_model(str(tmp_path), model_cfg, device="cpu")
    # Longer than the sliding window so the causal mask is exercised, not just
    # the short-sequence full-attention path.
    input_ids = torch.tensor([[1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45]])
    with torch.no_grad():
        expected = reference(input_ids).logits
        actual = model.forward_logits(input_ids)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
