"""Source-level tests for Qwen2 integration in DeepInfer's model_executor.

These tests run entirely on CPU with the reference SDPA attention path.  They
validate config dispatch, HF weight loading, KV-cache continuation and PEFT
LoRA merging.  They do not claim that the SM100 native kernels or the full
DeepInfer cache engine are ready for Qwen2 yet.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import Qwen2Config
from transformers import Qwen2ForCausalLM as HFQwen2ForCausalLM

from deepinfer.hugging_face import load_hf_model_config
from deepinfer.model_executor.model_output import CausalLMForwardOutput
from deepinfer.model_executor.models import ModelRegistry
from deepinfer.model_executor.models.qwen2 import (
    Qwen2ForCausalLM,
    Qwen2InputMetadata,
)

REAL_QWEN2_PATH = Path(r"F:\QwenQLoRA\models\Qwen2.5-3B-Instruct-smallshards")


def _make_tiny_checkpoint(tmp_path: Path) -> tuple[Path, HFQwen2ForCausalLM]:
    torch.manual_seed(20261025)
    config = Qwen2Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=True,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_dropout=0.0,
        use_sliding_window=False,
        torch_dtype="float32",
    )
    model = HFQwen2ForCausalLM(config).eval()
    model.save_pretrained(tmp_path)
    return tmp_path, model


@pytest.fixture()
def tiny_qwen_checkpoint(tmp_path: Path) -> tuple[Path, HFQwen2ForCausalLM]:
    return _make_tiny_checkpoint(tmp_path)


def test_qwen2_config_dispatch_and_registry(tmp_path: Path) -> None:
    _make_tiny_checkpoint(tmp_path)

    model_cfg = load_hf_model_config(tmp_path / "config.json")

    assert model_cfg.architecture == "Qwen2ForCausalLM"
    assert model_cfg.model_type == "qwen2"
    assert model_cfg.gpt.num_key_value_heads == 2
    assert model_cfg.gpt.num_key_value_groups == 2
    assert model_cfg.gpt.full_attention is True
    assert ModelRegistry.load_model_cls("Qwen2ForCausalLM") is Qwen2ForCausalLM


def test_qwen2_model_config_runtime_attrs(tmp_path: Path) -> None:
    from deepinfer.config import ModelConfig

    _make_tiny_checkpoint(tmp_path)
    config = ModelConfig(
        model=str(tmp_path),
        tokenizer=str(tmp_path),
        trust_remote_code=False,
        load_format="auto",
        seed=1,
        max_num_seqs=2,
        max_model_len=64,
    )

    assert config.architecture == "Qwen2ForCausalLM"
    assert config.get_vocab_size() == 128
    assert config.model_cfg.gpt.max_num_seqs == 2
    assert config.model_cfg.gpt.runtime_max_model_len >= 64


def test_qwen2_forward_matches_hf_and_uses_cache(
    tiny_qwen_checkpoint: tuple[Path, HFQwen2ForCausalLM],
) -> None:
    checkpoint, reference = tiny_qwen_checkpoint
    model_cfg = load_hf_model_config(checkpoint / "config.json")
    model = Qwen2ForCausalLM.load_model(str(checkpoint), model_cfg, device="cpu")
    assert model.source_backend == "qwen2_reference_sdpa"
    assert model.native_runner_ready is False

    input_ids = torch.tensor([[1, 5, 9, 13, 17, 21, 25]], dtype=torch.long)
    positions = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        reference_logits = reference(
            input_ids=input_ids,
            position_ids=positions,
            use_cache=False,
            return_dict=True,
        ).logits
        model_logits = model.forward_logits(input_ids, positions)

    torch.testing.assert_close(model_logits, reference_logits, atol=2e-5, rtol=2e-4)

    cache = model.new_kv_cache()
    with torch.no_grad():
        prefix_logits = model.forward_logits(
            input_ids[:, :5],
            positions[:, :5],
            kv_cache=cache,
        )
        continuation_logits = model.forward_logits(
            input_ids[:, 5:],
            positions[:, 5:],
            kv_cache=cache,
        )
    cached_logits = torch.cat((prefix_logits, continuation_logits), dim=1)
    torch.testing.assert_close(cached_logits, model_logits, atol=2e-5, rtol=2e-4)

    metadata = Qwen2InputMetadata(input_ids=input_ids, positions=positions)
    with torch.no_grad():
        output = model.forward(None, metadata)
    assert isinstance(output, CausalLMForwardOutput)
    assert output.hidden_states.shape == (
        1,
        input_ids.shape[1],
        model_cfg.gpt.hidden_size,
    )


def test_qwen2_peft_lora_merge_matches_peft(
    tiny_qwen_checkpoint: tuple[Path, HFQwen2ForCausalLM],
    tmp_path: Path,
) -> None:
    from peft import LoraConfig, get_peft_model

    checkpoint, _ = tiny_qwen_checkpoint
    base = HFQwen2ForCausalLM.from_pretrained(checkpoint).eval()
    peft_model = get_peft_model(
        base,
        LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj", "gate_proj"],
            task_type="CAUSAL_LM",
        ),
    ).eval()
    with torch.no_grad():
        for name, parameter in peft_model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(mean=0.0, std=0.02)

    adapter_dir = tmp_path / "adapter"
    peft_model.save_pretrained(adapter_dir)

    input_ids = torch.tensor([[3, 7, 11, 19, 23]], dtype=torch.long)
    positions = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        peft_logits = peft_model(
            input_ids=input_ids,
            position_ids=positions,
            use_cache=False,
            return_dict=True,
        ).logits

    model_cfg = load_hf_model_config(checkpoint / "config.json")
    merged = Qwen2ForCausalLM.load_model(
        str(checkpoint),
        model_cfg,
        adapter_path=adapter_dir,
        device="cpu",
    )
    with torch.no_grad():
        merged_logits = merged.forward_logits(input_ids, positions)

    torch.testing.assert_close(merged_logits, peft_logits, atol=2e-5, rtol=2e-4)


def test_w4a16_linear_dequant_prefill_path_cpu() -> None:
    from deepinfer.model_executor.models.qwen2_w4a16 import W4A16Linear

    torch.manual_seed(20261025)
    dense = torch.nn.Linear(256, 128, bias=True)
    quantized = W4A16Linear(
        256,
        128,
        True,
        group_size=128,
        dtype=torch.float32,
        device="cpu",
    )
    quantized.load_from_linear(dense)

    inputs = torch.randn(1, 5, 256)
    with torch.no_grad():
        reference = dense(inputs)
        actual = quantized(inputs)

    cosine = torch.nn.functional.cosine_similarity(
        reference.flatten(),
        actual.flatten().float(),
        dim=0,
    )
    assert cosine.item() > 0.985
    assert torch.mean(torch.abs(reference - actual.float())).item() < 0.05


@pytest.mark.skipif(
    not REAL_QWEN2_PATH.is_dir(), reason="real Qwen2.5-3B checkpoint is not present"
)
def test_real_qwen2_checkpoint_key_coverage() -> None:
    """Validate the real 3B config and HF shard names without allocating weights."""

    import json

    model_cfg = load_hf_model_config(REAL_QWEN2_PATH / "config.json")
    assert model_cfg.gpt.num_hidden_layers == 36
    assert model_cfg.gpt.num_attention_heads == 16
    assert model_cfg.gpt.num_key_value_heads == 2
    assert model_cfg.gpt.head_dim == 128

    with torch.device("meta"):
        model = Qwen2ForCausalLM(model_cfg.gpt)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert 3.0e9 < parameter_count < 3.2e9

    index = json.loads(
        (REAL_QWEN2_PATH / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    checkpoint_names = set(index["weight_map"])
    model_names = set(dict(model.named_parameters(remove_duplicate=False)))
    missing_from_checkpoint = model_names - checkpoint_names
    assert missing_from_checkpoint == {"lm_head.weight"}  # tied to embed_tokens
    assert not checkpoint_names - model_names
