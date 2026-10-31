"""One OLMoE layer, compared against transformers' OlmoeDecoderLayer.

This is the assembly check: norm placement, per-head QK-norm before RoPE,
RoPE convention, residual order and the absence of a shared expert all have to
be right simultaneously for the outputs to agree.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers.models.olmoe.configuration_olmoe import OlmoeConfig  # noqa: E402
from transformers.models.olmoe.modeling_olmoe import (  # noqa: E402
    OlmoeDecoderLayer,
    OlmoeRotaryEmbedding,
)

from deepinfer.moe.olmoe_layer import decoder_layer_forward  # noqa: E402

HIDDEN = 64
HEADS = 4
INTER = 32
EXPERTS = 8
TOP_K = 2
TOKENS = 5


def _config() -> OlmoeConfig:
    return OlmoeConfig(
        vocab_size=128,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        num_hidden_layers=1,
        num_attention_heads=HEADS,
        num_key_value_heads=HEADS,
        num_experts=EXPERTS,
        num_experts_per_tok=TOP_K,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        hidden_act="silu",
        attention_bias=False,
        norm_topk_prob=False,
    )


def _weights(layer: OlmoeDecoderLayer) -> dict[str, torch.Tensor]:
    """Stack transformers' per-expert ModuleList into packed tensors."""

    experts = layer.mlp.experts
    gate = torch.stack([e.gate_proj.weight.detach() for e in experts])
    up = torch.stack([e.up_proj.weight.detach() for e in experts])
    down = torch.stack([e.down_proj.weight.detach() for e in experts])
    return {
        "attn_norm": layer.input_layernorm.weight.detach().clone(),
        "ffn_norm": layer.post_attention_layernorm.weight.detach().clone(),
        "q_proj": layer.self_attn.q_proj.weight.detach().clone(),
        "k_proj": layer.self_attn.k_proj.weight.detach().clone(),
        "v_proj": layer.self_attn.v_proj.weight.detach().clone(),
        "o_proj": layer.self_attn.o_proj.weight.detach().clone(),
        "attn_q_norm": layer.self_attn.q_norm.weight.detach().clone(),
        "attn_k_norm": layer.self_attn.k_norm.weight.detach().clone(),
        "router": layer.mlp.gate.weight.detach().clone(),
        "expert_gate": gate,
        "expert_up": up,
        "expert_down": down,
    }


def _run_reference(layer: OlmoeDecoderLayer, hidden: torch.Tensor) -> torch.Tensor:
    """transformers' layer needs position embeddings supplied explicitly."""

    position_ids = torch.arange(hidden.shape[1]).unsqueeze(0)
    cache_position = torch.arange(hidden.shape[1])
    rotary = OlmoeRotaryEmbedding(config=layer.self_attn.config)
    position_embeddings = rotary(hidden, position_ids)
    # transformers receives a causal mask from the model body; without one the
    # layer attends in both directions while our reference path is causal
    tokens = hidden.shape[1]
    causal = torch.full((1, 1, tokens, tokens), float("-inf"))
    causal = torch.triu(causal, diagonal=1)
    with torch.no_grad():
        return layer(
            hidden,
            attention_mask=causal,
            position_embeddings=position_embeddings,
            cache_position=cache_position,
            output_attentions=False,
        )[0]


def _zero_experts(layer: OlmoeDecoderLayer) -> None:
    with torch.no_grad():
        for expert in layer.mlp.experts:
            expert.gate_proj.weight.zero_()
            expert.up_proj.weight.zero_()
            expert.down_proj.weight.zero_()


@pytest.fixture(scope="module")
def reference_layer():
    torch.manual_seed(0)
    config = _config()
    layer = OlmoeDecoderLayer(config, layer_idx=0).to(torch.float32)
    layer.eval()
    return layer, config


def test_matches_transformers_layer(reference_layer) -> None:
    layer, config = reference_layer
    hidden = torch.randn(1, TOKENS, HIDDEN)

    expected = _run_reference(layer, hidden)

    got = decoder_layer_forward(
        hidden,
        _weights(layer),
        num_heads=HEADS,
        eps=config.rms_norm_eps,
        rope_base=config.rope_theta,
        top_k=TOP_K,
    )
    assert torch.allclose(got, expected, atol=1e-4), (got - expected).abs().max()


def test_no_shared_expert_path() -> None:
    """Zeroing the experts must agree with transformers doing the same.

    If this implementation carried any hidden shared-expert or identity term,
    zeroing the routed experts would leave a contribution that transformers
    does not produce.
    """

    torch.manual_seed(4)
    config = _config()
    layer = OlmoeDecoderLayer(config, layer_idx=0).to(torch.float32)
    layer.eval()
    _zero_experts(layer)
    hidden = torch.randn(1, TOKENS, HIDDEN)

    expected = _run_reference(layer, hidden)

    got = decoder_layer_forward(
        hidden,
        _weights(layer),
        num_heads=HEADS,
        eps=config.rms_norm_eps,
        rope_base=config.rope_theta,
        top_k=TOP_K,
    )
    assert torch.allclose(got, expected, atol=1e-4), (got - expected).abs().max()
