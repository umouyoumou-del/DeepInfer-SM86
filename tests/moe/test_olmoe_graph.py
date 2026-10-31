"""The CUDA-graph decode must reproduce the eager decode it replaces.

A small synthetic OLMoE with real Q4_K-packed experts runs through both paths:
the eager one (OLMoEKVKache, growing tensors, exact-length attention) and the
graph (static cache, masked attention, device-side position).  Teacher-forced
logits are compared step by step, so a near-tie cannot cascade into a token
mismatch; free-running greedy tokens are compared separately.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from deepinfer.moe.fused_moe import fused_moe_supported  # noqa: E402

pytestmark = pytest.mark.gpu

HIDDEN, HEADS, INTER, EXPERTS, TOP_K, LAYERS, VOCAB = 256, 2, 256, 8, 2, 2, 64
Q4_K_ROW = HIDDEN // 256 * 144  # both k sizes (hidden, inter) are 256


def _require_cuda() -> None:
    if not fused_moe_supported():
        pytest.skip("packed experts need CUDA and CuPy")


def _q4_k(rows: int, rng) -> torch.Tensor:
    """Random Q4_K blocks with small super-block scales, as the kernel tests use."""

    values = rng.integers(0, 256, size=(EXPERTS * rows, 144), dtype=np.uint8)
    values[:, 0:2] = np.frombuffer(struct.pack("<e", 1e-4), dtype=np.uint8)
    values[:, 2:4] = np.frombuffer(struct.pack("<e", 5e-5), dtype=np.uint8)
    return torch.from_numpy(values.reshape(EXPERTS, rows, Q4_K_ROW)).cuda()


def _packed(storage: torch.Tensor):
    stride = storage[0].numel()
    pointers = torch.tensor([storage.data_ptr() + e * stride for e in range(EXPERTS)],
                            dtype=torch.int64, device="cuda")
    return storage, pointers, "Q4_K"


def _model(seed: int = 5):
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)

    def dense(*shape, scale=0.05):
        return (torch.randn(*shape, generator=generator) * scale).to("cuda", torch.float16)

    def norm():
        return (1 + 0.1 * torch.randn(HIDDEN, generator=generator)).cuda()

    weights = {"embed_tokens": dense(VOCAB, HIDDEN, scale=1.0), "lm_head": dense(VOCAB, HIDDEN),
               "final_norm": norm()}
    layers = []
    for _ in range(LAYERS):
        layers.append({
            "attn_norm": norm(), "ffn_norm": norm(), "attn_q_norm": norm(), "attn_k_norm": norm(),
            "q_proj": dense(HIDDEN, HIDDEN), "k_proj": dense(HIDDEN, HIDDEN),
            "v_proj": dense(HIDDEN, HIDDEN), "o_proj": dense(HIDDEN, HIDDEN),
            "router": (torch.randn(EXPERTS, HIDDEN, generator=generator)).cuda(),
            "packed": {"gate": _packed(_q4_k(INTER, rng)), "up": _packed(_q4_k(INTER, rng)),
                       "down": _packed(_q4_k(HIDDEN, rng))},
            "packed_gate_out": INTER,
        })
    return weights, layers


KWARGS = {"num_heads": HEADS, "eps": 1e-5, "rope_base": 10000.0, "top_k": TOP_K}


def _eager_logits(weights, layers, tokens: list[int]) -> list[torch.Tensor]:
    from deepinfer.moe.olmoe_layer import OLMoEKVKache
    from deepinfer.moe.olmoe_model import forward

    cache, out = OLMoEKVKache(), []
    for token in tokens:
        ids = torch.tensor([[token]], device="cuda")
        out.append(forward(ids, weights, layers, cache=cache, **KWARGS)[0, -1].float())
    return out


def _decoder(weights, layers, max_len: int = 32):
    from deepinfer.moe.olmoe_graph import OLMoEGraphDecoder

    decoder = OLMoEGraphDecoder(weights, layers, max_len=max_len, **KWARGS)
    decoder.capture()
    return decoder


def test_teacher_forced_logits_match_eager() -> None:
    _require_cuda()
    weights, layers = _model()
    tokens = [3, 17, 42, 8, 8, 60, 1, 33, 21, 5]
    expected = _eager_logits(weights, layers, tokens)
    decoder = _decoder(weights, layers)
    for step, token in enumerate(tokens):
        decoder.step(token)
        got = decoder.logits[0, -1].float()
        scale = expected[step].abs().max()
        assert torch.allclose(got, expected[step], atol=2e-2 * float(scale), rtol=0), (
            step, float((got - expected[step]).abs().max()), float(scale))


def test_free_running_tokens_match_eager_and_reset_repeats() -> None:
    _require_cuda()
    weights, layers = _model(seed=9)
    prompt, new = [11, 4, 50], 12
    sequence = list(prompt)
    for _ in range(new):
        sequence.append(int(_eager_logits(weights, layers, sequence)[-1].argmax()))
    decoder = _decoder(weights, layers)
    first = decoder.generate(prompt, new)
    assert first == sequence[len(prompt):]
    assert decoder.generate(prompt, new) == first  # reset() starts a clean sequence


def test_running_past_max_len_raises() -> None:
    _require_cuda()
    weights, layers = _model()
    decoder = _decoder(weights, layers, max_len=4)
    with pytest.raises(RuntimeError, match="max_len"):
        decoder.generate([1, 2], 4)  # 2 + 3 replays > 4 slots
