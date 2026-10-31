"""Structural description of an OLMoE checkpoint, read from GGUF metadata.

Kept separate from any runtime so the wiring can be checked against declared
facts rather than against assumptions about tensor names.
"""

from __future__ import annotations

from dataclasses import dataclass

from deepinfer.moe.gguf_index import GGUFIndex


@dataclass(frozen=True)
class OLMoESpec:
    block_count: int
    hidden_size: int
    head_count: int
    head_count_kv: int
    head_dim: int
    expert_count: int
    expert_used_count: int
    context_length: int
    rope_base: float
    rms_eps: float
    vocab_size: int
    moe_layer_indices: tuple[int, ...]
    has_shared_experts: bool = False
    architecture: str = "olmoe"

    @property
    def is_multi_head(self) -> bool:
        return self.head_count_kv == self.head_count

    @property
    def kv_group_size(self) -> int:
        if self.head_count_kv <= 0 or self.head_count % self.head_count_kv:
            raise ValueError("head_count must be a multiple of head_count_kv")
        return self.head_count // self.head_count_kv


def _require(metadata: dict, key: str):
    if key not in metadata:
        raise ValueError(f"GGUF metadata is missing {key!r}")
    return metadata[key]


def spec_from_gguf(index: GGUFIndex) -> OLMoESpec:
    """Build a spec, validating the assumptions the port depends on."""

    metadata = index.metadata
    architecture = str(_require(metadata, "general.architecture"))
    if architecture != "olmoe":
        raise ValueError(f"expected an olmoe checkpoint, got {architecture!r}")

    hidden = int(_require(metadata, "olmoe.embedding_length"))
    heads = int(_require(metadata, "olmoe.attention.head_count"))
    heads_kv = int(_require(metadata, "olmoe.attention.head_count_kv"))
    blocks = int(_require(metadata, "olmoe.block_count"))
    if heads <= 0 or hidden % heads:
        raise ValueError(f"hidden size {hidden} is not divisible by {heads} heads")

    names = set(index.tensor_names())
    # OLMoE has no shared expert; a tensor with _shexp would contradict the plan
    has_shared = any("_shexp" in name for name in names)
    moe_layers = tuple(
        layer
        for layer in range(blocks)
        if f"blk.{layer}.ffn_gate_exps.weight" in names
    )

    return OLMoESpec(
        block_count=blocks,
        hidden_size=hidden,
        head_count=heads,
        head_count_kv=heads_kv,
        head_dim=hidden // heads,
        expert_count=int(_require(metadata, "olmoe.expert_count")),
        expert_used_count=int(_require(metadata, "olmoe.expert_used_count")),
        context_length=int(_require(metadata, "olmoe.context_length")),
        rope_base=float(_require(metadata, "olmoe.rope.freq_base")),
        rms_eps=float(_require(metadata, "olmoe.attention.layer_norm_rms_epsilon")),
        vocab_size=int(
            metadata.get("olmoe.vocab_size")
            or metadata.get("tokenizer.ggml.vocab_size", 0)
        ),
        moe_layer_indices=moe_layers,
        has_shared_experts=has_shared,
    )
