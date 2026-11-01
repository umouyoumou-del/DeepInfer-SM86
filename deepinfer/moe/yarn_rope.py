"""YaRN frequency scaling, so a model trained at N tokens can attend beyond it.

OLMoE declares 4,096 tokens and no scaling, so anything past that is
extrapolation.  DeepSeek-V2's config already declares YaRN (factor 40 over an
original 4,096) and ``modeling_deepseek`` applies it itself -
``DeepseekV2YarnRotaryEmbedding`` plus an mscale^2 softmax correction - so the
runtime leaves those modules alone.  The formula here matches that module.

YaRN interpolates only the low frequencies, which carry long-range information,
and leaves the high frequencies alone, which is what keeps local order intact:

    inv_freq_interpolated = inv_freq / factor

applied through a ramp between beta_fast and beta_slow, so index 0 (highest
frequency) is untouched and the last index is divided by the full factor.
"""

from __future__ import annotations

import math

import torch


def _correction_index(
    num_rotations: float, dim: int, base: float, max_position: int
) -> float:
    return (dim * math.log(max_position / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def correction_range(
    dim: int, base: float, max_position: int, *, beta_fast: float, beta_slow: float
) -> tuple[int, int]:
    """Inverse-frequency indices between which the ramp runs."""

    low = math.floor(_correction_index(beta_fast, dim, base, max_position))
    high = math.ceil(_correction_index(beta_slow, dim, base, max_position))
    return max(low, 0), min(high, dim - 1)


def ramp(dim: int, *, beta_fast: float, beta_slow: float, base: float,
         max_position: int) -> torch.Tensor:
    """1 where the frequency is fully interpolated (divided by factor), 0 where
    it is left untouched.

    The ramp rises with the inverse-frequency index: index 0 is the highest
    frequency and keeps its original value, while the tail is the low-frequency
    part that carries long-range information and gets divided.
    """

    low, high = correction_range(
        dim, base, max_position, beta_fast=beta_fast, beta_slow=beta_slow
    )
    if low == high:
        high += 0.001  # the reference implementation's singularity guard
    # indexed by rotary pair (0 .. dim/2-1), as the correction range is; using
    # the element index 0, 2, 4, ... here would ramp twice as fast
    pairs = torch.arange(dim // 2, dtype=torch.float32)
    return torch.clamp((pairs - low) / (high - low), 0.0, 1.0)


def scaled_inv_freq(
    dim: int,
    *,
    base: float = 10000.0,
    factor: float,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    max_position: int,
) -> torch.Tensor:
    """Inverse frequencies with YaRN applied; one entry per rotary pair."""

    if factor < 1.0:
        raise ValueError(f"factor must be >= 1, got {factor}")
    if dim % 2:
        raise ValueError(f"rotary dim must be even, got {dim}")
    inverse = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if factor == 1.0:
        return inverse
    weights = ramp(
        dim, beta_fast=beta_fast, beta_slow=beta_slow, base=base,
        max_position=max_position,
    )
    return inverse * (1.0 - weights) + (inverse / factor) * weights


def build_cos_sin(
    dim: int, seq_len: int, inv_freq: torch.Tensor, *, device, dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """The usual cos/sin tables, doubled over the head dimension."""

    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    frequencies = torch.outer(positions, inv_freq.to(device=device))
    frequencies = torch.cat((frequencies, frequencies), dim=-1)
    return torch.cos(frequencies).to(dtype), torch.sin(frequencies).to(dtype)


def patch_rotary(
    rotary,
    *,
    factor: float,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    max_position: int,
    seq_len: int | None = None,
) -> float:
    """Rewrite a loaded rotary module in place; returns the attention scale fix.

    YaRN also raises the softmax temperature as ``0.1 * ln(factor) + 1``; the
    caller multiplies its attention scale by the returned value.  Returns 1.0
    when ``factor`` is 1.
    """

    if factor == 1.0:
        return 1.0
    dim = int(rotary.dim)
    device = rotary.inv_freq.device
    dtype = rotary.inv_freq.dtype
    rotary.inv_freq = scaled_inv_freq(
        dim, base=float(rotary.base), factor=factor,
        beta_fast=beta_fast, beta_slow=beta_slow, max_position=max_position,
    ).to(device=device, dtype=dtype)
    length = int(seq_len or rotary.max_position_embeddings)
    cos, sin = build_cos_sin(dim, length, rotary.inv_freq, device=device, dtype=dtype)
    rotary.register_buffer("cos_cached", cos, persistent=False)
    rotary.register_buffer("sin_cached", sin, persistent=False)
    rotary.max_seq_len_cached = length
    return 0.1 * math.log(factor) + 1.0


def yarn_from_config(config) -> tuple[float, float, float, int] | None:
    """Read ``(factor, beta_fast, beta_slow, original_max)`` off an HF config."""

    scaling = getattr(config, "rope_scaling", None)
    if not scaling:
        return None
    kind = str(scaling.get("type") or scaling.get("rope_type") or "").lower()
    if kind not in ("yarn", "dynamic", "longrope"):
        return None
    return (
        float(scaling.get("factor", 1.0)),
        float(scaling.get("beta_fast", 32.0)),
        float(scaling.get("beta_slow", 1.0)),
        int(scaling.get("original_max_position_embeddings",
                        getattr(config, "max_position_embeddings", 4096))),
    )
