"""The batched prefill against the eager forward and against the fused decode it feeds.

Kernel level: the dequantising tensor-core GEMM against NumPy decodes of the
ggml block formats (with the weights rounded to fp16, as the kernel rounds them
in shared memory).  End to end, on the synthetic OLMoE of tests.moe.olmoe_synthetic:
the last prompt token's logits against the eager PyTorch forward, the key/value
cache against stepping the same prompt through the fused decoder, and decoding
after a prefill against decoding after stepping.  A 300-token prompt gives every
expert more rows than one 64-row tile, so the dispatch's multi-tile runs are hit.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from deepinfer.moe.olmoe_fused import OLMoEFusedDecoder, fused_supported
from deepinfer.moe.olmoe_prefill import OLMoEFusedPrefill, gemm
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


@pytest.mark.parametrize("rows,cols", [(1, 256), (70, 512), (130, 2048)])
def test_gemm_matches_numpy(rows: int, cols: int) -> None:
    _require()
    rng = np.random.default_rng(rows + cols)
    matrices = [matrix(qtype, 192, cols, rng) for qtype in ("Q4_K", "Q6_K", "Q4_K")]
    a = torch.from_numpy(rng.standard_normal((rows, cols)).astype(np.float16)).cuda()
    got = gemm(a, [packed for packed, _ in matrices]).double().cpu().numpy()
    ar = a.double().cpu().numpy()
    for index, (_packed, reference) in enumerate(matrices):
        want = ar @ reference.astype(np.float16).astype(np.float64).T
        np.testing.assert_allclose(got[index], want, rtol=1e-4, atol=1e-4 * np.abs(want).max())


def _pair(model, max_len: int):
    """A captured decoder with a prefill engine, and a second decoder to step through."""

    prefilled = OLMoEFusedDecoder(model, max_len=max_len, **KWARGS)
    prefilled.capture()
    stepped = OLMoEFusedDecoder(model, max_len=max_len, **KWARGS)
    stepped.capture()
    return prefilled, OLMoEFusedPrefill(prefilled, max_tokens=max_len - 8), stepped


@pytest.mark.parametrize("length", [20, 300])
def test_prefill_matches_eager_and_fills_the_cache_like_stepping(length: int) -> None:
    _require()
    fused, eager, layers = synthetic()
    prompt = [int(t) for t in np.random.default_rng(length).integers(0, 64, size=length)]
    expected = eager_logits(eager, layers, prompt)[-1]
    prefilled, engine, stepped = _pair(fused, 512)

    token = engine.run(prompt)
    got = prefilled.logits[0, -1]
    scale = float(expected.abs().max())
    assert torch.allclose(got, expected, atol=2e-2 * scale, rtol=0), (
        float((got - expected).abs().max()), scale)
    assert token == int(got.argmax())
    assert int(prefilled.position) == length and prefilled.used == length
    assert int(prefilled.history[length - 1]) == token

    for t in prompt:
        stepped.step(t)
    for layer in range(len(fused.layers)):
        for ours, theirs in ((prefilled.keys, stepped.keys), (prefilled.values, stepped.values)):
            a, b = ours[layer][:, :length].float(), theirs[layer][:, :length].float()
            assert torch.allclose(a, b, atol=1e-2 * float(b.abs().max()), rtol=0), (
                layer, float((a - b).abs().max()))


def test_decoding_continues_from_the_prefilled_cache() -> None:
    _require()
    fused, _eager, _layers = synthetic(seed=9)
    prompt = [11, 4, 50, 7, 7, 31, 2, 60, 18]
    prefilled, engine, stepped = _pair(fused, 64)
    engine.run(prompt)
    for t in prompt:
        stepped.step(t)
    for _ in range(6):
        ours, theirs = prefilled.step(), stepped.step()
        a, b = prefilled.logits[0, -1], stepped.logits[0, -1]
        assert torch.allclose(a, b, atol=1e-2 * float(b.abs().max()), rtol=0)
        if int(ours) != int(theirs):
            top2 = torch.topk(b, 2).values
            assert float(top2[0] - top2[1]) < 1e-2 * float(b.abs().max())  # only at a near-tie
            break


def test_graph_replay_matches_eager_for_new_tokens() -> None:
    """A replay must read the new prompt, not the one it was captured with."""

    _require()
    fused, _eager, _layers = synthetic()
    rng = np.random.default_rng(3)
    first, second = ([int(t) for t in rng.integers(0, 64, size=40)] for _ in range(2))
    graphed = OLMoEFusedDecoder(fused, max_len=64, **KWARGS)
    graphed.capture()
    engine = OLMoEFusedPrefill(graphed, max_tokens=48)
    eager = OLMoEFusedDecoder(fused, max_len=64, **KWARGS)
    eager.capture()
    reference = OLMoEFusedPrefill(eager, max_tokens=48, use_graphs=False)
    for prompt in (first, second):  # the first captures, the second replays
        assert engine.run(prompt) == reference.run(prompt)
        torch.testing.assert_close(graphed.logits, eager.logits, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(graphed.keys, eager.keys, rtol=0, atol=0)


def test_prefill_checks_its_inputs() -> None:
    _require()
    fused, _eager, _layers = synthetic()
    decoder = OLMoEFusedDecoder(fused, max_len=32, **KWARGS)
    engine = OLMoEFusedPrefill(decoder, max_tokens=16)
    with pytest.raises(RuntimeError, match="capture"):
        engine.run([1, 2, 3])  # capture() would reset the state a prefill leaves
    decoder.capture()
    with pytest.raises(ValueError, match="prompt length"):
        engine.run(list(range(17)))
    with pytest.raises(ValueError, match="max_tokens"):
        OLMoEFusedPrefill(decoder, max_tokens=33)


@pytest.mark.skipif(not os.path.exists(OLMOE_GGUF), reason="OLMoE GGUF not present")
def test_real_checkpoint_prefill_matches_eager_forward() -> None:
    """Two real layers: the last prompt token through the prefill and through eager."""

    _require()
    from deepinfer.moe.gguf_index import GGUFIndex
    from deepinfer.moe.olmoe_fused import load_fused_model
    from deepinfer.moe.olmoe_spec import spec_from_gguf

    index = GGUFIndex(OLMOE_GGUF)
    spec = spec_from_gguf(index)
    fused = load_fused_model(index, spec, layers=2)
    eager, eager_layers, kwargs = eager_for_checkpoint(index, spec, fused)
    # "The capital of France is", then ids spread over the vocabulary
    prompt = [510, 5347, 273, 6181, 310, *range(1000, 50000, 1700)]
    expected = eager_logits(eager, eager_layers, prompt, kwargs)[-1]
    decoder = OLMoEFusedDecoder(fused, max_len=64, **kwargs)
    decoder.capture()
    OLMoEFusedPrefill(decoder, max_tokens=64).run(prompt)
    got = decoder.logits[0, -1]
    scale = float(expected.abs().max())
    assert torch.allclose(got, expected, atol=2e-2 * scale, rtol=0), (
        float((got - expected).abs().max()), scale)
