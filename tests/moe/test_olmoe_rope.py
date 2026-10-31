"""RoPE tables for the OLMoE port: values at offsets, and no host syncs.

The KV cache made RoPE depend on a position offset, and the first version read
that offset back from a CUDA tensor - one synchronisation per layer, 16 per
token.  These tests pin both halves of the fix: the sliced table gives the same
angles as the direct formula at any offset, and a cached decode step on the GPU
contains no synchronising operation at all.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from deepinfer.moe.olmoe_layer import (  # noqa: E402
    OLMoEKVKache,
    build_rope,
    decoder_layer_forward,
)

HEAD_DIM = 16
BASE = 10000.0


def _direct(start: int, tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent formulation in float64: angle = position * base^(-2j/d)."""

    positions = torch.arange(start, start + tokens, dtype=torch.float64)
    frequencies = BASE ** (-torch.arange(0, HEAD_DIM, 2, dtype=torch.float64) / HEAD_DIM)
    angles = positions[:, None] * frequencies[None, :]
    return torch.cos(angles), torch.sin(angles)


@pytest.mark.parametrize("start,tokens", [(0, 5), (7, 1), (255, 3), (300, 1), (1000, 4)])
def test_rope_at_offset_matches_direct_formula(start: int, tokens: int) -> None:
    # (255, 3) straddles the initial table and (300, 1) / (1000, 4) force growth
    cos, sin = build_rope(
        torch.arange(start, start + tokens), HEAD_DIM, BASE,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    want_cos, want_sin = _direct(start, tokens)
    assert cos.shape == (tokens, HEAD_DIM // 2)
    assert torch.allclose(cos.double(), want_cos, atol=1e-5)
    assert torch.allclose(sin.double(), want_sin, atol=1e-5)


def _cuda_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    return torch.device("cuda")


@pytest.mark.gpu
def test_rope_rejects_cuda_positions() -> None:
    device = _cuda_or_skip()
    with pytest.raises(ValueError, match="CPU"):
        build_rope(torch.arange(3, device=device), HEAD_DIM, BASE,
                   device=device, dtype=torch.float16)


@pytest.mark.gpu
def test_cached_decode_step_has_no_host_sync() -> None:
    device = _cuda_or_skip()
    hidden, heads, inter, experts, top_k = 64, 4, 32, 8, 2
    generator = torch.Generator().manual_seed(3)

    def param(*shape):
        return (torch.randn(*shape, generator=generator) * 0.1).to(device, torch.float16)

    weights = {
        "attn_norm": param(hidden), "ffn_norm": param(hidden),
        "attn_q_norm": param(hidden), "attn_k_norm": param(hidden),
        "q_proj": param(hidden, hidden), "k_proj": param(hidden, hidden),
        "v_proj": param(hidden, hidden), "o_proj": param(hidden, hidden),
        "router": param(experts, hidden),
        "expert_gate": param(experts, inter, hidden),
        "expert_up": param(experts, inter, hidden),
        "expert_down": param(experts, hidden, inter),
    }
    kwargs = {"num_heads": heads, "eps": 1e-5, "rope_base": BASE, "top_k": top_k}
    cache = OLMoEKVKache()
    prompt = param(1, 4, hidden)
    decoder_layer_forward(prompt, weights, cache=cache, **kwargs)
    step = param(1, 1, hidden)
    decoder_layer_forward(step, weights, cache=cache, **kwargs)  # warm the allocator
    torch.cuda.synchronize()

    torch.cuda.set_sync_debug_mode("error")
    try:
        # a new position each time, so a cache keyed by position would miss
        for _ in range(3):
            out = decoder_layer_forward(step, weights, cache=cache, **kwargs)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert cache.length() == 4 + 1 + 3
    assert torch.isfinite(out).all()
