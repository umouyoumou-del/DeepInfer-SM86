"""The OLMoE spec must reflect GGUF metadata, not assumptions."""

from __future__ import annotations

import os

import pytest

from deepinfer.moe.olmoe_spec import OLMoESpec, spec_from_gguf

OLMOE_GGUF = os.environ.get(
    "OLMOE_GGUF",
    r"E:\DeepInfer-MoE\models\OLMoE-1B-7B-GGUF\OLMoE-1B-7B-0924.Q4_K_M.gguf",
)


def _fake_index(metadata: dict, names: list[str]):
    class _Fake:
        pass

    fake = _Fake()
    fake.metadata = metadata
    fake.tensor_names = lambda: list(names)
    return fake


BASE_METADATA = {
    "general.architecture": "olmoe",
    "olmoe.block_count": 16,
    "olmoe.embedding_length": 2048,
    "olmoe.attention.head_count": 16,
    "olmoe.attention.head_count_kv": 16,
    "olmoe.expert_count": 64,
    "olmoe.expert_used_count": 8,
    "olmoe.context_length": 4096,
    "olmoe.rope.freq_base": 10000.0,
    "olmoe.attention.layer_norm_rms_epsilon": 1e-5,
    "olmoe.vocab_size": 50304,
}


def _names(blocks: int = 16, shared: bool = False) -> list[str]:
    out: list[str] = []
    for layer in range(blocks):
        out.append(f"blk.{layer}.ffn_gate_exps.weight")
        if shared:
            out.append(f"blk.{layer}.ffn_gate_shexp.weight")
    return out


def test_reads_declared_fields() -> None:
    spec = spec_from_gguf(_fake_index(BASE_METADATA, _names()))
    assert isinstance(spec, OLMoESpec)
    assert spec.block_count == 16
    assert spec.head_count == spec.head_count_kv == 16
    assert spec.head_dim == 128
    assert spec.expert_used_count == 8
    assert spec.is_multi_head is True
    assert spec.kv_group_size == 1
    assert spec.has_shared_experts is False
    assert spec.moe_layer_indices == tuple(range(16))


def test_detects_shared_experts_if_present() -> None:
    spec = spec_from_gguf(_fake_index(BASE_METADATA, _names(shared=True)))
    assert spec.has_shared_experts is True


def test_rejects_non_olmoe() -> None:
    metadata = dict(BASE_METADATA)
    metadata["general.architecture"] = "deepseek2"
    with pytest.raises(ValueError):
        spec_from_gguf(_fake_index(metadata, _names()))


def test_rejects_missing_metadata() -> None:
    metadata = dict(BASE_METADATA)
    del metadata["olmoe.expert_used_count"]
    with pytest.raises(ValueError):
        spec_from_gguf(_fake_index(metadata, _names()))


def test_head_dim_must_divide_hidden() -> None:
    metadata = dict(BASE_METADATA)
    metadata["olmoe.attention.head_count"] = 7
    with pytest.raises(ValueError):
        spec_from_gguf(_fake_index(metadata, _names()))


@pytest.mark.skipif(not os.path.exists(OLMOE_GGUF), reason="OLMoE GGUF not present")
def test_real_checkpoint_matches_expectations() -> None:
    from deepinfer.moe.gguf_index import GGUFIndex

    spec = spec_from_gguf(GGUFIndex(OLMOE_GGUF))
    assert spec.block_count == 16
    assert spec.is_multi_head, "OLMoE is MHA, not GQA"
    assert spec.expert_count == 64
    assert spec.expert_used_count == 8
    assert spec.head_dim == 128
    assert spec.has_shared_experts is False
