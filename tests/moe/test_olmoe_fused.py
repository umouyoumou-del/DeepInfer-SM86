"""The fused OLMoE decoder against independent references.

Kernel level: the packed GEMV against NumPy decodes of the ggml block formats,
which share nothing with the CUDA lane mapping.  End to end: a small synthetic
OLMoE with every matrix packed (tests.moe.olmoe_synthetic) runs through the
fused decoder and through the eager PyTorch forward on the same weights
dequantised to fp16.  Teacher-forced logits are compared step by step, so a
near-tie cannot cascade into a token mismatch.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from deepinfer.moe.olmoe_fused import OLMoEFusedDecoder, fused_supported, gemv
from tests.moe.olmoe_synthetic import (
    KWARGS,
    eager_for_checkpoint,
    eager_logits,
    matrix,
    synthetic,
)

pytestmark = pytest.mark.gpu

OLMOE_GGUF = os.environ.get(
    "OLMOE_GGUF",
    r"E:\DeepInfer-MoE\models\OLMoE-1B-7B-GGUF\OLMoE-1B-7B-0924.Q4_K_M.gguf",
)


def _require() -> None:
    if not fused_supported():
        pytest.skip("needs CUDA and CuPy")


@pytest.mark.parametrize("cols", [256, 512, 1024, 2048])
def test_gemv_matches_numpy(cols: int) -> None:
    _require()
    rng = np.random.default_rng(cols)
    rows = 40  # not a multiple of the eight rows one block covers
    matrices = [matrix(qtype, rows, cols, rng) for qtype in ("Q4_K", "Q6_K", "Q4_K")]
    x = torch.from_numpy(rng.standard_normal(cols).astype(np.float32)).cuda()
    got = gemv(x, [packed for packed, _ in matrices]).double().cpu().numpy()
    xr = x.double().cpu().numpy()
    for index, (_packed, reference) in enumerate(matrices):
        want = reference.astype(np.float64) @ xr
        np.testing.assert_allclose(got[index], want, rtol=1e-4, atol=1e-4 * np.abs(want).max())


def _decoder(model, max_len: int = 32) -> OLMoEFusedDecoder:
    decoder = OLMoEFusedDecoder(model, max_len=max_len, **KWARGS)
    decoder.capture()
    return decoder


def test_teacher_forced_logits_match_eager() -> None:
    _require()
    fused, eager, layers = synthetic()
    tokens = [3, 17, 42, 8, 8, 60, 1, 33, 21, 5]
    expected = eager_logits(eager, layers, tokens)
    decoder = _decoder(fused)
    for step, token in enumerate(tokens):
        decoder.step(token)
        got = decoder.logits[0, -1]
        scale = float(expected[step].abs().max())
        assert torch.allclose(got, expected[step], atol=2e-2 * scale, rtol=0), (
            step, float((got - expected[step]).abs().max()), scale)
        # the argmax folded into the LM head GEMV is the argmax of its logits
        assert int(decoder.token) == int(got.argmax())


def test_greedy_matches_eager_until_a_near_tie_and_reset_repeats() -> None:
    _require()
    fused, eager, layers = synthetic(seed=9)
    prompt, new = [11, 4, 50], 12
    sequence = list(prompt)
    for _ in range(new):
        sequence.append(int(eager_logits(eager, layers, sequence)[-1].argmax()))
    expected = eager_logits(eager, layers, sequence)

    decoder = _decoder(fused)
    first = decoder.generate(prompt, new)
    assert decoder.generate(prompt, new) == first  # reset() starts a clean sequence
    for offset, token in enumerate(first):
        logits = expected[len(prompt) - 1 + offset]
        top2 = torch.topk(logits, 2).values
        if float(top2[0] - top2[1]) < 2e-2 * float(logits.abs().max()):
            break  # past a near-tie the two paths may legitimately part
        assert token == sequence[len(prompt) + offset], offset


def test_running_past_max_len_raises() -> None:
    _require()
    fused, _eager, _layers = synthetic()
    decoder = _decoder(fused, max_len=4)
    with pytest.raises(RuntimeError, match="max_len"):
        decoder.generate([1, 2], 4)  # 2 + 3 replays > 4 slots


@pytest.mark.skipif(not os.path.exists(OLMOE_GGUF), reason="OLMoE GGUF not present")
def test_real_checkpoint_matches_eager_forward() -> None:
    """Two real layers: fused (packed dense) against eager (fp16 dense)."""

    _require()
    from deepinfer.moe.gguf_index import GGUFIndex
    from deepinfer.moe.olmoe_fused import load_fused_model
    from deepinfer.moe.olmoe_spec import spec_from_gguf

    index = GGUFIndex(OLMOE_GGUF)
    spec = spec_from_gguf(index)
    fused = load_fused_model(index, spec, layers=2)
    eager, eager_layers, kwargs = eager_for_checkpoint(index, spec, fused)
    tokens = [510, 5347, 273, 6181, 310]  # "The capital of France is"
    expected = eager_logits(eager, eager_layers, tokens, kwargs)
    decoder = OLMoEFusedDecoder(fused, max_len=16, **kwargs)
    decoder.capture()
    for step, token in enumerate(tokens):
        decoder.step(token)
        got = decoder.logits[0, -1]
        scale = float(expected[step].abs().max())
        assert torch.allclose(got, expected[step], atol=2e-2 * scale, rtol=0), (
            step, float((got - expected[step]).abs().max()), scale)
