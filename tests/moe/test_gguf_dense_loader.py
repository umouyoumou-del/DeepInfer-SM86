"""Name mapping and state-dict assembly for the dense GGUF loader."""

from __future__ import annotations

import re

import numpy as np
import pytest

from deepinfer.moe.gguf_dense_loader import (
    BLOCK_TENSORS,
    UnsupportedGGUFLayout,
    build_dense_state_dict,
    infer_layer_count,
    map_tensor_name,
)


class _FakeIndex:
    def __init__(self, names: list[str]) -> None:
        self._names = list(names)

    def tensor_names(self) -> list[str]:
        return list(self._names)

    def read_tensor(self, name: str) -> np.ndarray:
        return np.zeros((2, 2), dtype=np.float16)


def test_maps_static_and_blocked_tensors() -> None:
    assert map_tensor_name("token_embd.weight") == "model.embed_tokens.weight"
    assert map_tensor_name("output.weight") == "lm_head.weight"
    assert map_tensor_name("blk.3.attn_q.weight") == (
        "model.layers.3.self_attn.q_proj.weight"
    )
    assert map_tensor_name("blk.11.ffn_down.weight") == (
        "model.layers.11.mlp.down_proj.weight"
    )


def test_unknown_tensors_are_skipped_not_fatal() -> None:
    # rope frequency tables are computed at runtime, not loaded
    assert map_tensor_name("rope_freqs.weight") is None


def test_moe_tensors_are_rejected_loudly() -> None:
    with pytest.raises(UnsupportedGGUFLayout):
        map_tensor_name("blk.0.ffn_gate_exps.weight")


def test_state_dict_covers_every_layer() -> None:
    names = ["token_embd.weight", "output_norm.weight"]
    for layer in range(3):
        names += [
            f"blk.{layer}.attn_norm.weight",
            f"blk.{layer}.attn_q.weight",
            f"blk.{layer}.ffn_down.weight",
        ]
    state = build_dense_state_dict(_FakeIndex(names), num_layers=3)
    assert "model.embed_tokens.weight" in state
    for layer in range(3):
        assert f"model.layers.{layer}.self_attn.q_proj.weight" in state
    assert infer_layer_count(_FakeIndex(names)) == 3


def test_missing_layer_is_reported() -> None:
    names = ["token_embd.weight"]
    for layer in (0, 2):
        names.append(f"blk.{layer}.attn_q.weight")
    with pytest.raises(UnsupportedGGUFLayout) as excinfo:
        build_dense_state_dict(_FakeIndex(names), num_layers=3)
    assert "1" in str(excinfo.value)


def test_missing_embeddings_is_reported() -> None:
    with pytest.raises(UnsupportedGGUFLayout):
        build_dense_state_dict(_FakeIndex(["blk.0.attn_q.weight"]))


# --------------------------------------------------------------------------
# End-to-end: build a tiny dense model, export it under llama.cpp names,
# reload it through from_gguf() and check the parameters and a forward pass.
# --------------------------------------------------------------------------

_REVERSE = {
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
}


def _reverse_name(name: str) -> str:
    if name in _REVERSE:
        return _REVERSE[name]
    match = re.match(r"^model\.layers\.(\d+)\.(.+)$", name)
    if match is None:
        return name
    layer, suffix = match.group(1), match.group(2)
    for gguf_suffix, repo_suffix in BLOCK_TENSORS.items():
        if repo_suffix == suffix:
            return f"blk.{layer}.{gguf_suffix}"
    return name


class _DictIndex:
    def __init__(self, tensors: dict[str, np.ndarray]) -> None:
        self._tensors = tensors

    def tensor_names(self) -> list[str]:
        return list(self._tensors)

    def read_tensor(self, name: str) -> np.ndarray:
        return self._tensors[name]


def _tiny_model():
    import torch

    from deepinfer.attrdict import AttrDict
    from deepinfer.model_executor.models.qwen2 import Qwen2ForCausalLM

    cfg = AttrDict(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        hidden_act="silu",
        attention_bias=True,
        head_dim=8,
        sliding_window=None,
    )
    torch.manual_seed(0)
    return Qwen2ForCausalLM(cfg), cfg


def test_from_gguf_roundtrip_preserves_parameters() -> None:
    import torch

    from deepinfer.moe.gguf_dense_loader import from_gguf

    model, cfg = _tiny_model()
    exported = {
        _reverse_name(name): tensor.detach().numpy()
        for name, tensor in model.state_dict().items()
        if name != "lm_head.weight"  # tied embeddings
    }

    reloaded, _ = _tiny_model()
    for parameter in reloaded.parameters():
        torch.nn.init.zeros_(parameter)
    from_gguf(_DictIndex(exported), reloaded)

    for name, tensor in model.state_dict().items():
        if name == "lm_head.weight":
            continue
        assert torch.allclose(
            tensor, reloaded.state_dict()[name], atol=1e-6
        ), f"parameter mismatch: {name}"

    # Tied-embedding checkpoints carry no lm_head.weight; from_gguf() has to
    # rebuild it from the embedding table.
    assert torch.allclose(
        reloaded.lm_head.weight, reloaded.model.embed_tokens.weight
    )
