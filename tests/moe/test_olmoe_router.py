"""OLMoE routing: softmax over all experts, top-k, no renormalisation."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from deepinfer.moe.olmoe_router import reference_route_topk, route_topk  # noqa: E402

HIDDEN = 32
EXPERTS = 64
TOP_K = 8


def _inputs(seed: int = 0, rows: int = 3):
    torch.manual_seed(seed)
    return (
        torch.randn(rows, HIDDEN, dtype=torch.float32),
        torch.randn(HIDDEN, EXPERTS, dtype=torch.float32),
    )


def test_matches_reference() -> None:
    hidden, gate = _inputs()
    ids, weights = route_topk(hidden, gate, top_k=TOP_K)
    ref_ids, ref_weights = reference_route_topk(hidden, gate, top_k=TOP_K)
    assert torch.equal(ids, ref_ids)
    assert torch.allclose(weights, ref_weights, atol=1e-5)


def test_weights_are_not_renormalised() -> None:
    # Uniform logits make every expert 1/64, so an un-renormalised top-8 sums
    # to 8/64 = 0.125 while a re-normalising implementation would report 1.0.
    hidden = torch.zeros(2, HIDDEN)
    gate = torch.zeros(HIDDEN, EXPERTS)
    _, weights = route_topk(hidden, gate, top_k=TOP_K)
    assert torch.allclose(
        weights.sum(dim=-1), torch.full((2,), TOP_K / EXPERTS), atol=1e-6
    )


def test_selection_is_sorted_and_unique() -> None:
    hidden, gate = _inputs(seed=7)
    ids, weights = route_topk(hidden, gate, top_k=TOP_K)
    assert bool((weights.diff(dim=-1) <= 0).all()), "weights must be descending"
    for row in ids:
        assert len(set(row.tolist())) == TOP_K


def test_top_k_equals_expert_count_gives_full_softmax() -> None:
    hidden, gate = _inputs()
    _, weights = route_topk(hidden, gate, top_k=EXPERTS)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(hidden.shape[0]), atol=1e-5)


def test_rejects_mismatched_shapes() -> None:
    hidden, gate = _inputs()
    with pytest.raises(ValueError):
        route_topk(hidden, gate.t().contiguous(), top_k=TOP_K)
    with pytest.raises(ValueError):
        route_topk(hidden.unsqueeze(0), gate, top_k=TOP_K)
