"""A small OLMoE with every matrix packed, plus ggml block decodes, for the fused tests.

The block decodes follow ggml's dequantize_row_q4_K / dequantize_row_q6_K and
share nothing with the CUDA lane mappings.  The random weights are scaled so
attention and the experts move the logits as much as the embedding does; with
tiny weights a wiring error in either would hide under the residual stream.
"""

from __future__ import annotations

import struct

import numpy as np
import torch

from deepinfer.moe.olmoe_fused import FusedLayer, FusedModel, PackedMatrix

BLOCK_BYTES = {"Q4_K": 144, "Q6_K": 210}
HIDDEN, HEADS, INTER, EXPERTS, TOP_K, LAYERS, VOCAB = 256, 2, 256, 8, 2, 2, 64
KWARGS = {"num_heads": HEADS, "eps": 1e-5, "rope_base": 10000.0, "top_k": TOP_K}


def q4_k(blocks: np.ndarray) -> np.ndarray:
    """ggml dequantize_row_q4_K over ``[n, 144]`` blocks -> ``[n, 256]``."""

    d = blocks[:, 0:2].copy().view(np.float16).astype(np.float32)
    dmin = blocks[:, 2:4].copy().view(np.float16).astype(np.float32)
    sc = blocks[:, 4:16].astype(np.int32)
    qs = blocks[:, 16:144].astype(np.int32)
    scale = np.concatenate([sc[:, 0:4] & 63, (sc[:, 8:12] & 0xF) | ((sc[:, 0:4] >> 6) << 4)], axis=1)
    minimum = np.concatenate([sc[:, 4:8] & 63, (sc[:, 8:12] >> 4) | ((sc[:, 4:8] >> 6) << 4)], axis=1)
    out = np.empty((len(blocks), 256), np.float32)
    for pair in range(4):
        q = qs[:, 32 * pair:32 * pair + 32]
        lo, hi = 2 * pair, 2 * pair + 1
        out[:, 64 * pair:64 * pair + 32] = d * scale[:, lo:lo + 1] * (q & 0xF) - dmin * minimum[:, lo:lo + 1]
        out[:, 64 * pair + 32:64 * pair + 64] = d * scale[:, hi:hi + 1] * (q >> 4) - dmin * minimum[:, hi:hi + 1]
    return out


def q6_k(blocks: np.ndarray) -> np.ndarray:
    """ggml dequantize_row_q6_K over ``[n, 210]`` blocks -> ``[n, 256]``."""

    ql = blocks[:, 0:128].astype(np.int32)
    qh = blocks[:, 128:192].astype(np.int32)
    scales = blocks[:, 192:208].copy().view(np.int8).astype(np.float32)
    d = blocks[:, 208:210].copy().view(np.float16).astype(np.float32)
    out = np.empty((len(blocks), 256), np.float32)
    l = np.arange(32)
    is_ = l // 16
    for half in (0, 1):
        qlb, qhb = ql[:, half * 64:half * 64 + 64], qh[:, half * 32:half * 32 + 32]
        scb, base = scales[:, half * 8:half * 8 + 8], half * 128
        out[:, base + l] = d * scb[:, is_] * (((qlb[:, l] & 0xF) | ((qhb[:, l] & 3) << 4)) - 32)
        out[:, base + 32 + l] = d * scb[:, is_ + 2] * (((qlb[:, l + 32] & 0xF) | (((qhb[:, l] >> 2) & 3) << 4)) - 32)
        out[:, base + 64 + l] = d * scb[:, is_ + 4] * (((qlb[:, l] >> 4) | (((qhb[:, l] >> 4) & 3) << 4)) - 32)
        out[:, base + 96 + l] = d * scb[:, is_ + 6] * (((qlb[:, l + 32] >> 4) | (((qhb[:, l] >> 6) & 3) << 4)) - 32)
    return out


def _fp16_bytes(value: float) -> np.ndarray:
    return np.frombuffer(struct.pack("<e", value), dtype=np.uint8)


def blocks(qtype: str, count: int, rng, scale: float) -> np.ndarray:
    """Random blocks decoding to zero-mean values of roughly +-scale."""

    out = rng.integers(0, 256, size=(count, BLOCK_BYTES[qtype]), dtype=np.uint8)
    if qtype == "Q4_K":
        # E[d * sc * q - dmin * m] = 31.5 * (7.5 d - dmin): zero when dmin = 7.5 d
        d = scale / 240.0
        out[:, 0:2] = _fp16_bytes(d)
        out[:, 2:4] = _fp16_bytes(7.5 * d)
    else:
        out[:, 208:210] = _fp16_bytes(scale / 1300.0)
    return out


def decode(qtype: str, raw: np.ndarray) -> np.ndarray:
    return q4_k(raw) if qtype == "Q4_K" else q6_k(raw)


def matrix(qtype: str, rows: int, cols: int, rng, scale: float = 0.06):
    """A random packed matrix and its float32 decode."""

    raw = blocks(qtype, rows * cols // 256, rng, scale)
    packed = PackedMatrix(torch.from_numpy(raw.reshape(-1).copy()).cuda(), qtype, rows, cols)
    return packed, decode(qtype, raw).reshape(rows, cols)


def experts(qtype: str, rows: int, cols: int, rng):
    raw = blocks(qtype, EXPERTS * rows * cols // 256, rng, 0.06)
    storage = torch.from_numpy(raw.reshape(EXPERTS, -1).copy()).cuda()
    stride = storage[0].numel()
    table = torch.tensor([storage.data_ptr() + e * stride for e in range(EXPERTS)],
                         dtype=torch.int64, device="cuda")
    return storage, table, qtype


def synthetic(seed: int = 5):
    """The same weights as a FusedModel and as eager fp16 tensors."""

    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)

    def norm() -> torch.Tensor:
        return (1 + 0.1 * torch.randn(HIDDEN, generator=generator)).cuda()

    def half(reference: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(reference).to("cuda", torch.float16)

    embed, embed_ref = matrix("Q4_K", VOCAB, HIDDEN, rng, scale=1.0)
    head, head_ref = matrix("Q6_K", VOCAB, HIDDEN, rng)
    final_norm = norm()
    fused_layers, eager_layers = [], []
    for index in range(LAYERS):
        # mixed formats, as bartowski's file mixes them per layer
        roles = (("q", "Q4_K"), ("k", "Q4_K"), ("v", "Q6_K" if index == 0 else "Q4_K"), ("o", "Q4_K"))
        matrices = {role: matrix(qtype, HIDDEN, HIDDEN, rng) for role, qtype in roles}
        norms = {name: norm() for name in ("attn_norm", "ffn_norm", "q_norm", "k_norm")}
        router = torch.randn(EXPERTS, HIDDEN, generator=generator).cuda()
        packed = {"gate": experts("Q4_K", INTER, HIDDEN, rng),
                  "up": experts("Q4_K", INTER, HIDDEN, rng),
                  "down": experts("Q4_K" if index == 0 else "Q6_K", HIDDEN, INTER, rng)}
        fused_layers.append(FusedLayer(
            q=matrices["q"][0], k=matrices["k"][0], v=matrices["v"][0], o=matrices["o"][0],
            attn_norm=norms["attn_norm"], ffn_norm=norms["ffn_norm"], q_norm=norms["q_norm"],
            k_norm=norms["k_norm"], router=router, experts=packed, inter=INTER,
        ))
        eager_layers.append({
            "attn_norm": norms["attn_norm"], "ffn_norm": norms["ffn_norm"],
            "attn_q_norm": norms["q_norm"], "attn_k_norm": norms["k_norm"],
            "q_proj": half(matrices["q"][1]), "k_proj": half(matrices["k"][1]),
            "v_proj": half(matrices["v"][1]), "o_proj": half(matrices["o"][1]),
            "router": router, "packed": packed, "packed_gate_out": INTER,
        })
    fused = FusedModel(embed=embed, lm_head=head, final_norm=final_norm, layers=fused_layers)
    eager = {"embed_tokens": half(embed_ref), "lm_head": half(head_ref), "final_norm": final_norm}
    return fused, eager, eager_layers


def eager_for_checkpoint(index, spec, fused):
    """Eager weights for a real GGUF: fp16 dense matrices, the fused model's packed experts."""

    from deepinfer.moe.olmoe_weights import layer_tensors

    def half(name: str) -> torch.Tensor:
        return torch.from_numpy(np.asarray(index.read_tensor(name)).copy()).to("cuda", torch.float16)

    layers = []
    for layer_index, layer in enumerate(fused.layers):
        names = layer_tensors(spec, layer_index)
        layers.append({
            "attn_norm": layer.attn_norm, "ffn_norm": layer.ffn_norm,
            "attn_q_norm": layer.q_norm, "attn_k_norm": layer.k_norm,
            **{role: half(name) for role, name in names.attention.items()},
            "router": layer.router, "packed": layer.experts, "packed_gate_out": layer.inter,
        })
    weights = {"embed_tokens": half("token_embd.weight"), "lm_head": half("output.weight"),
               "final_norm": fused.final_norm}
    kwargs = {"num_heads": spec.head_count, "eps": spec.rms_eps, "rope_base": spec.rope_base,
              "top_k": spec.expert_used_count}
    return weights, layers, kwargs


def eager_logits(weights, layers, tokens: list[int], kwargs=None) -> list[torch.Tensor]:
    """Per-position logits of the eager PyTorch forward, one token at a time."""

    from deepinfer.moe.olmoe_layer import OLMoEKVKache
    from deepinfer.moe.olmoe_model import forward

    cache, out = OLMoEKVKache(), []
    for token in tokens:
        ids = torch.tensor([[token]], device="cuda")
        out.append(forward(ids, weights, layers, cache=cache, **(kwargs or KWARGS))[0, -1].float())
    return out
