from __future__ import annotations

import json

from transformers import Qwen2Config, Qwen2ForCausalLM

from deepinfer.model_probe import probe_model


def _write_tiny_qwen(root) -> None:
    config = Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        tie_word_embeddings=True,
        torch_dtype="float32",
    )
    Qwen2ForCausalLM(config).save_pretrained(root)


def test_probe_supported_hf_configuration(tmp_path) -> None:
    _write_tiny_qwen(tmp_path)
    report = probe_model(tmp_path)
    assert report["supported"] is True
    assert report["runtime_ready"] is False
    assert report["architecture"] == "Qwen2ForCausalLM"
    assert "W4A16" in report["reason"]


def test_probe_native_w4_checkpoint(tmp_path) -> None:
    _write_tiny_qwen(tmp_path)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "format": "deepinfer_w4a16_sm86_v1",
                "architecture": "Qwen2ForCausalLM",
                "group_size": 128,
                "layer_files": ["layer_00.safetensors"],
            }
        ),
        encoding="utf-8",
    )
    report = probe_model(tmp_path)
    assert report["supported"] is True
    assert report["runtime_ready"] is True
    assert report["format"] == "deepinfer_w4a16"


def test_probe_explains_unsupported_architecture(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["MixtralForCausalLM"],
                "model_type": "mixtral",
                "hidden_size": 64,
            }
        ),
        encoding="utf-8",
    )
    report = probe_model(tmp_path)
    assert report["supported"] is False
    assert "unsupported" in report["reason"].lower()
