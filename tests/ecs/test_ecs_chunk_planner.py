"""Invariants for the ECS chunk planner.

``plan_chunks`` decides how many experts go into one staging chunk, which
directly bounds both PCIe traffic and launch count.  The existing ECS tests
exercise heat/tiering; these pin the planner's partitioning contract.
"""

from __future__ import annotations

import pytest

from deepinfer.ecs.core import ECSCore, available as cpp_available


@pytest.fixture(params=[False, True])
def core(request) -> ECSCore:
    if request.param and not cpp_available():
        pytest.skip("cpp ECS core not available")
    return ECSCore()


def test_empty_input_plans_nothing(core: ECSCore) -> None:
    assert core.plan_chunks([], [], max_experts=8, max_tokens=64) == []


def test_mismatched_lengths_are_rejected(core: ECSCore) -> None:
    with pytest.raises(ValueError):
        core.plan_chunks([1, 2], [1], max_experts=8, max_tokens=64)


def test_every_expert_appears_exactly_once(core: ECSCore) -> None:
    expert_ids = [0, 3, 5, 7, 9, 11, 13]
    counts = [4, 2, 9, 1, 6, 3, 8]
    chunks = core.plan_chunks(
        expert_ids, counts, max_experts=3, max_tokens=12
    )
    covered: list[int] = []
    for start, length in chunks:
        assert length > 0
        covered.extend(expert_ids[start : start + length])
    assert sorted(covered) == sorted(expert_ids)


def test_chunk_respects_max_experts_and_max_tokens(core: ECSCore) -> None:
    expert_ids = list(range(12))
    counts = [3] * 12
    max_experts, max_tokens = 4, 9
    chunks = core.plan_chunks(
        expert_ids, counts, max_experts=max_experts, max_tokens=max_tokens
    )
    assert chunks, "a non-empty plan is expected"
    for start, length in chunks:
        assert 1 <= length <= max_experts
        assert sum(counts[start : start + length]) <= max_tokens


def test_single_expert_larger_than_budget_still_progresses(core: ECSCore) -> None:
    """One expert over the token budget must not produce an empty plan."""

    chunks = core.plan_chunks([7], [1000], max_experts=8, max_tokens=16)
    assert chunks
    covered = []
    for start, length in chunks:
        covered.extend([7] * 0 + [start + i for i in range(length)])
    assert covered == [0]
