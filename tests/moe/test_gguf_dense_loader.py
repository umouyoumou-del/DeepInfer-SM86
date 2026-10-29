"""Name mapping and state-dict assembly for the dense GGUF loader."""

from __future__ import annotations

import numpy as np
import pytest

from deepinfer.moe.gguf_dense_loader import (
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
