from __future__ import annotations

import torch
from transformers import Gemma2Config, GemmaConfig
from transformers import Gemma2ForCausalLM as HFGemma2ForCausalLM
from transformers import GemmaForCausalLM as HFGemmaForCausalLM

from deepinfer.hugging_face import load_hf_model_config
from deepinfer.model_executor.models import ModelRegistry
from deepinfer.model_executor.models.gemma import GemmaForCausalLM
from deepinfer.model_executor.models.gemma2 import Gemma2ForCausalLM


def test_gemma_geglu_offset_norm_and_embedding_scale_match_hf(tmp_path) -> None:
    config = GemmaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        torch_dtype="float32",
    )
    torch.manual_seed(20261027)
    reference = HFGemmaForCausalLM(config).eval()
    reference.save_pretrained(tmp_path, safe_serialization=True)

    model_cfg = load_hf_model_config(tmp_path / "config.json")
    assert model_cfg.gpt.rms_norm_weight_offset == 1.0
    assert model_cfg.gpt.embedding_multiplier == 32**0.5
    assert model_cfg.gpt.activation_type == "geglu"
    assert ModelRegistry.load_model_cls("GemmaForCausalLM") is GemmaForCausalLM

    model = GemmaForCausalLM.load_model(str(tmp_path), model_cfg, device="cpu")
    input_ids = torch.tensor([[1, 5, 9, 13, 17, 21, 25]])
    with torch.no_grad():
        expected = reference(input_ids).logits
        actual = model.forward_logits(input_ids)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)


def test_gemma2_sandwich_norm_softcap_and_alternating_swa_match_hf(tmp_path) -> None:
    config = Gemma2Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        sliding_window=4,
        query_pre_attn_scalar=8,
        attn_logit_softcapping=5.0,
        final_logit_softcapping=10.0,
        tie_word_embeddings=True,
        torch_dtype="float32",
    )
    torch.manual_seed(20261027)
    reference = HFGemma2ForCausalLM(config).eval()
    reference.save_pretrained(tmp_path, safe_serialization=True)

    model_cfg = load_hf_model_config(tmp_path / "config.json")
    assert model_cfg.gpt.sandwich_norm is True
    assert model_cfg.gpt.sliding_window_pattern == (True, False, True, False)
    assert model_cfg.gpt.attn_logit_softcapping == 5.0
    assert model_cfg.gpt.final_logit_softcapping == 10.0
    assert model_cfg.gpt.query_pre_attn_scalar == 8.0
    assert ModelRegistry.load_model_cls("Gemma2ForCausalLM") is Gemma2ForCausalLM

    model = Gemma2ForCausalLM.load_model(str(tmp_path), model_cfg, device="cpu")
    # Longer than sliding_window so odd/even layer mask behavior is exercised.
    input_ids = torch.tensor([[1, 5, 9, 13, 17, 21, 25, 29, 33, 37]])
    with torch.no_grad():
        expected = reference(input_ids).logits
        actual = model.forward_logits(input_ids)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
