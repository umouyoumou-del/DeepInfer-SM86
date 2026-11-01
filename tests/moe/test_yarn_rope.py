"""YaRN must interpolate low frequencies only, and change nothing at factor 1."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from deepinfer.moe.yarn_rope import (  # noqa: E402
    build_cos_sin,
    correction_range,
    patch_rotary,
    ramp,
    scaled_inv_freq,
    yarn_from_config,
)

DIM = 64
BASE = 10000.0
TRAINED = 4096


def test_factor_one_is_identity() -> None:
    plain = 1.0 / (BASE ** (torch.arange(0, DIM, 2, dtype=torch.float32) / DIM))
    scaled = scaled_inv_freq(
        DIM, base=BASE, factor=1.0, max_position=TRAINED
    )
    assert torch.allclose(plain, scaled, atol=1e-7)


def test_high_frequencies_untouched_low_frequencies_divided() -> None:
    factor = 4.0
    plain = 1.0 / (BASE ** (torch.arange(0, DIM, 2, dtype=torch.float32) / DIM))
    scaled = scaled_inv_freq(
        DIM, base=BASE, factor=factor, max_position=TRAINED
    )
    # index 0 carries the highest frequency and must be untouched
    assert torch.allclose(scaled[0], plain[0], rtol=1e-5)
    # the last index is the lowest frequency and must be divided by the factor
    assert math.isclose(
        float(scaled[-1]), float(plain[-1]) / factor, rel_tol=1e-4
    )


def test_scaling_is_monotone_and_bounded() -> None:
    factor = 4.0
    plain = 1.0 / (BASE ** (torch.arange(0, DIM, 2, dtype=torch.float32) / DIM))
    scaled = scaled_inv_freq(DIM, base=BASE, factor=factor, max_position=TRAINED)
    ratio = scaled / plain
    # never larger than unscaled, never smaller than one factor step
    assert bool((ratio <= 1.0 + 1e-6).all())
    assert bool((ratio >= 1.0 / factor - 1e-6).all())
    # and it decreases monotonically across indices
    assert bool((ratio[1:] <= ratio[:-1] + 1e-6).all())


def test_ramp_spans_the_correction_range() -> None:
    weights = ramp(
        DIM, beta_fast=32.0, beta_slow=1.0, base=BASE, max_position=TRAINED
    )
    # ramp = 1 means "interpolated": the tail is divided, index 0 is untouched
    assert float(weights[0]) == pytest.approx(0.0)
    assert float(weights[-1]) == pytest.approx(1.0)
    low, high = correction_range(
        DIM, BASE, TRAINED, beta_fast=32.0, beta_slow=1.0
    )
    assert 0 <= low < high


def test_cos_sin_tables_have_expected_shape() -> None:
    inv = scaled_inv_freq(DIM, base=BASE, factor=4.0, max_position=TRAINED)
    cos, sin = build_cos_sin(DIM, 32, inv, device="cpu", dtype=torch.float32)
    assert cos.shape == (32, DIM)
    assert sin.shape == (32, DIM)
    # position 0 is angle zero
    assert torch.allclose(cos[0], torch.ones(DIM))
    assert torch.allclose(sin[0], torch.zeros(DIM), atol=1e-6)


def test_patch_rotary_returns_mscale() -> None:
    class _Rotary(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dim = DIM
            self.base = BASE
            self.max_position_embeddings = TRAINED
            self.register_buffer(
                "inv_freq",
                1.0 / (BASE ** (torch.arange(0, DIM, 2, dtype=torch.float32) / DIM)),
                persistent=False,
            )

    rotary = _Rotary()
    before = rotary.inv_freq.clone()
    mscale = patch_rotary(rotary, factor=4.0, max_position=TRAINED, seq_len=64)
    assert not torch.allclose(before, rotary.inv_freq)
    assert rotary.cos_cached.shape == (64, DIM)
    assert math.isclose(mscale, 0.1 * math.log(4.0) + 1.0, rel_tol=1e-6)

    untouched = _Rotary()
    original = untouched.inv_freq.clone()
    assert patch_rotary(untouched, factor=1.0, max_position=TRAINED) == 1.0
    assert torch.allclose(original, untouched.inv_freq)


def test_reads_scaling_from_config() -> None:
    class _Config:
        max_position_embeddings = 163840
        rope_scaling = {
            "type": "yarn",
            "factor": 4.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": 4096,
        }

    parsed = yarn_from_config(_Config())
    assert parsed == (4.0, 32.0, 1.0, 4096)

    class _Plain:
        rope_scaling = None

    assert yarn_from_config(_Plain()) is None

def _official_deepseek_inv_freq(dim, base, factor, original, beta_fast, beta_slow):
    """DeepseekV2YarnRotaryEmbedding._set_cos_sin_cache, transcribed."""

    def correction_dim(rotations):
        return (dim * math.log(original / (rotations * 2 * math.pi))) / (2 * math.log(base))

    low = max(math.floor(correction_dim(beta_fast)), 0)
    high = min(math.ceil(correction_dim(beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    linear = (torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)
    mask = 1.0 - torch.clamp(linear, 0, 1)
    exponents = torch.arange(0, dim, 2, dtype=torch.float32) / dim
    extra = 1.0 / (base**exponents)
    inter = 1.0 / (factor * base**exponents)
    return inter * (1 - mask) + extra * mask


def test_matches_deepseek_v2_official_yarn() -> None:
    # DeepSeek-V2-Lite / Coder-V2-Lite: rope dim 64, factor 40 over 4096
    ours = scaled_inv_freq(
        64, base=10000.0, factor=40.0, beta_fast=32.0, beta_slow=1.0, max_position=4096
    )
    official = _official_deepseek_inv_freq(64, 10000.0, 40.0, 4096, 32.0, 1.0)
    assert torch.allclose(ours, official, rtol=1e-6, atol=0)
    # the ramp runs over pairs 10..23 there: pair 10 keeps its frequency, 23 is divided
    plain = 1.0 / (10000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    assert float(ours[10] / plain[10]) == pytest.approx(1.0)
    assert float(ours[23] / plain[23]) == pytest.approx(1.0 / 40.0)
