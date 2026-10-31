"""Full OLMoE assembly, compared against transformers' OlmoeForCausalLM."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers.models.olmoe.configuration_olmoe import OlmoeConfig  # noqa: E402
from transformers.models.olmoe.modeling_olmoe import OlmoeForCausalLM  # noqa: E402

from deepinfer.moe.olmoe_model import forward  # noqa: E402

VOCAB = 96
HIDDEN = 32
HEADS = 4
INTER = 24
EXPERTS = 8
TOP_K = 2
LAYERS = 2
TOKENS = 4


def _config(tie: bool = False) -> OlmoeConfig:
    return OlmoeConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=HEADS,
        num_experts=EXPERTS,
        num_experts_per_tok=TOP_K,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        hidden_act="silu",
        attention_bias=False,
        norm_topk_prob=False,
        tie_word_embeddings=tie,
    )


def _weights(model: OlmoeForCausalLM, tie: bool) -> dict:
    layers = []
    for layer in model.model.layers:
        experts = layer.mlp.experts
        layers.append(
            {
                "attn_norm": layer.input_layernorm.weight.detach().clone(),
                "ffn_norm": layer.post_attention_layernorm.weight.detach().clone(),
                "q_proj": layer.self_attn.q_proj.weight.detach().clone(),
                "k_proj": layer.self_attn.k_proj.weight.detach().clone(),
                "v_proj": layer.self_attn.v_proj.weight.detach().clone(),
                "o_proj": layer.self_attn.o_proj.weight.detach().clone(),
                "attn_q_norm": layer.self_attn.q_norm.weight.detach().clone(),
                "attn_k_norm": layer.self_attn.k_norm.weight.detach().clone(),
                "router": layer.mlp.gate.weight.detach().clone(),
                "expert_gate": torch.stack([e.gate_proj.weight.detach() for e in experts]),
                "expert_up": torch.stack([e.up_proj.weight.detach() for e in experts]),
                "expert_down": torch.stack([e.down_proj.weight.detach() for e in experts]),
            }
        )
    out = {
        "embed_tokens": model.model.embed_tokens.weight.detach().clone(),
        "final_norm": model.model.norm.weight.detach().clone(),
    }
    if not tie:
        out["lm_head"] = model.lm_head.weight.detach().clone()
    return out, layers


@pytest.mark.parametrize("tie", [False, True])
def test_matches_transformers_model(tie: bool) -> None:
    torch.manual_seed(7)
    config = _config(tie)
    model = OlmoeForCausalLM(config).to(torch.float32)
    model.eval()

    weights, layers = _weights(model, tie)
    tokens = torch.randint(0, VOCAB, (1, TOKENS))
    with torch.no_grad():
        expected = model(tokens, output_router_logits=False).logits

    got = forward(
        tokens,
        weights,
        layers,
        num_heads=HEADS,
        eps=config.rms_norm_eps,
        rope_base=config.rope_theta,
        top_k=TOP_K,
        tie_word_embeddings=tie,
    )
    assert torch.allclose(got, expected, atol=1e-4), (got - expected).abs().max()


def test_requires_at_least_one_layer() -> None:
    with pytest.raises(ValueError):
        forward(
            torch.zeros(1, TOKENS, dtype=torch.long),
            {"embed_tokens": torch.zeros(VOCAB, HIDDEN), "lm_head": torch.zeros(VOCAB, HIDDEN),
             "final_norm": torch.ones(HIDDEN)},
            [],
            num_heads=HEADS,
            eps=1e-5,
            rope_base=10000.0,
            top_k=TOP_K,
        )
