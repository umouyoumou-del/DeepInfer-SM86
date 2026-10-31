"""OLMoE tensor names and shapes, checked against the spec and the real file."""

from __future__ import annotations

import os

import pytest

from deepinfer.moe.olmoe_spec import OLMoESpec
from deepinfer.moe.olmoe_weights import layer_tensors, validate_layer

OLMOE_GGUF = os.environ.get(
    "OLMOE_GGUF",
    r"E:\DeepInfer-MoE\models\OLMoE-1B-7B-GGUF\OLMoE-1B-7B-0924.Q4_K_M.gguf",
)

SPEC = OLMoESpec(
    block_count=16,
    hidden_size=2048,
    head_count=16,
    head_count_kv=16,
    head_dim=128,
    expert_count=64,
    expert_used_count=8,
    context_length=4096,
    rope_base=10000.0,
    rms_eps=1e-5,
    vocab_size=50304,
    moe_layer_indices=tuple(range(16)),
)


class _FakeIndex:
    def __init__(self, shapes: dict[str, tuple]):
        self._shapes = shapes

    def tensor_names(self):
        return list(self._shapes)

    def tensor(self, name):
        if name not in self._shapes:
            raise KeyError(name)

        class _Info:
            numpy_shape = self._shapes[name]

        return _Info()


def _shapes(layer: int = 0, *, inter: int = 1024, experts: int = 64,
            hidden: int = 2048, skip: str | None = None) -> dict[str, tuple]:
    p = f"blk.{layer}."
    out = {
        p + "attn_q.weight": (hidden, hidden),
        p + "attn_k.weight": (hidden, hidden),
        p + "attn_v.weight": (hidden, hidden),
        p + "attn_output.weight": (hidden, hidden),
        p + "attn_norm.weight": (hidden,),
        p + "ffn_norm.weight": (hidden,),
        p + "attn_q_norm.weight": (hidden,),
        p + "attn_k_norm.weight": (hidden,),
        p + "ffn_gate_inp.weight": (experts, hidden),
        p + "ffn_gate_exps.weight": (experts, inter, hidden),
        p + "ffn_up_exps.weight": (experts, inter, hidden),
        p + "ffn_down_exps.weight": (experts, hidden, inter),
    }
    if skip:
        out.pop(p + skip)
    return out


def test_names_follow_the_checkpoint_layout() -> None:
    tensors = layer_tensors(SPEC, 3)
    assert tensors.attention["q_proj"] == "blk.3.attn_q.weight"
    assert tensors.norms["attn_k_norm"] == "blk.3.attn_k_norm.weight"
    assert tensors.router == "blk.3.ffn_gate_inp.weight"
    assert tensors.expert_down == "blk.3.ffn_down_exps.weight"


def test_validation_accepts_a_consistent_layer() -> None:
    report = validate_layer(_FakeIndex(_shapes()), SPEC, 0)
    assert report["expert_gate"] == (64, 1024, 2048)
    assert report["expert_down"] == (64, 2048, 1024)


def test_validation_flags_missing_tensor() -> None:
    with pytest.raises(ValueError, match="missing"):
        validate_layer(_FakeIndex(_shapes(skip="attn_k_norm.weight")), SPEC, 0)


def test_validation_flags_wrong_attention_shape() -> None:
    shapes = _shapes()
    shapes["blk.0.attn_q.weight"] = (2048, 1024)
    with pytest.raises(ValueError, match="attn_q"):
        validate_layer(_FakeIndex(shapes), SPEC, 0)


def test_validation_flags_down_intermediate_mismatch() -> None:
    shapes = _shapes(inter=1024)
    shapes["blk.0.ffn_down_exps.weight"] = (64, 2048, 999)
    with pytest.raises(ValueError, match="intermediate"):
        validate_layer(_FakeIndex(shapes), SPEC, 0)


@pytest.mark.skipif(not os.path.exists(OLMOE_GGUF), reason="OLMoE GGUF not present")
def test_real_checkpoint_layer_validates() -> None:
    from deepinfer.moe.gguf_index import GGUFIndex
    from deepinfer.moe.olmoe_spec import spec_from_gguf

    index = GGUFIndex(OLMOE_GGUF)
    spec = spec_from_gguf(index)
    for layer in (0, 15):
        report = validate_layer(index, spec, layer)
        assert report["expert_gate"][0] == 64
        assert report["expert_gate"][2] == 2048

@pytest.mark.skipif(not os.path.exists(OLMOE_GGUF), reason="OLMoE GGUF not present")
def test_real_layer_attention_weights_load_and_run() -> None:
    """GGUF -> dequantise -> one real attention branch, end to end."""

    import torch as _torch

    from deepinfer.moe.gguf_index import GGUFIndex
    from deepinfer.moe.olmoe_layer import build_rope, rms_norm
    from deepinfer.moe.olmoe_spec import spec_from_gguf
    from deepinfer.moe.olmoe_weights import load_layer_tensor

    index = GGUFIndex(OLMOE_GGUF)
    spec = spec_from_gguf(index)
    weights = {
        role: load_layer_tensor(index, spec, 0, role)
        for role in ("q_proj", "k_proj", "v_proj", "o_proj", "attn_norm", "attn_q_norm")
    }
    for role, tensor in weights.items():
        assert tensor.dtype == _torch.float32
        assert _torch.isfinite(tensor).all(), f"{role} has non-finite values"

    hidden = _torch.randn(1, 3, spec.hidden_size) * 0.1
    normed = rms_norm(hidden, weights["attn_norm"], spec.rms_eps)
    query = normed @ weights["q_proj"].t()
    query = rms_norm(query, weights["attn_q_norm"], spec.rms_eps)
    assert query.shape == (1, 3, spec.hidden_size)
    assert _torch.isfinite(query).all()

    heads, dim = spec.head_count, spec.head_dim
    reshaped = query.view(1, 3, heads, dim).transpose(1, 2)
    cos, sin = build_rope(
        _torch.arange(3), dim, spec.rope_base, device=query.device, dtype=query.dtype
    )
    from deepinfer.moe.olmoe_layer import apply_rope

    rotated = apply_rope(reshaped, cos, sin)
    assert rotated.shape == reshaped.shape
    assert _torch.isfinite(rotated).all()
    # RoPE must preserve the norm of each head
    assert _torch.allclose(
        rotated.norm(dim=-1), reshaped.norm(dim=-1), atol=1e-3
    )
