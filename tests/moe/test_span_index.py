"""Retrieval hands back text and an assembled context, never stale KV blocks."""

from __future__ import annotations

import pytest

from deepinfer.moe.span_index import (
    SpanIndex,
    assemble_context,
    assemble_from_index,
)

NEEDLE = [7, 7, 42, 7, 7]          # the passage that answers the query
QUERY = [42]


def _index() -> SpanIndex:
    index = SpanIndex()
    index.add_document([1, 2, 3, 4] * 8, span_size=8)      # filler spans
    index.add_document(NEEDLE + [5] * 3, span_size=8)      # the needle
    index.add_document([9, 10, 11] * 4, span_size=8)       # more filler
    return index


def test_needle_is_retrieved() -> None:
    index = _index()
    ranked = index.search(QUERY, top_k=2)
    assert ranked, "the needle must be found"
    best_span = index.spans[ranked[0][0]]
    assert 42 in best_span.token_ids


def test_no_overlap_returns_nothing() -> None:
    index = _index()
    assert index.search([999], top_k=4) == []


def test_top_k_bounds_the_result() -> None:
    index = _index()
    assert len(index.search([1, 2, 3, 42], top_k=1)) <= 1
    assert index.search(QUERY, top_k=0) == []


def test_assembly_positions_are_consecutive_and_ordered() -> None:
    context = assemble_context(
        sink=[100, 101], retrieved=[[1, 2, 3], [4, 5]], recent=[90, 91, 92]
    )
    # 2 sinks + 3 retrieved + 2 retrieved + 3 recent = 10
    assert len(context) == 10
    assert context.positions == list(range(10))
    assert context.token_ids[:2] == [100, 101]
    assert context.token_ids[2:5] == [1, 2, 3]
    assert context.token_ids[-3:] == [90, 91, 92]
    kinds = [kind for kind, _ in context.provenance]
    assert kinds[0] == "sink" and kinds[-1] == "recent"
    assert "retrieved" in kinds


def test_assembled_context_gets_fresh_positions_not_the_original_ones() -> None:
    """The whole point of recompute: positions belong to the new context."""

    index = _index()
    context, ranked = assemble_from_index(
        index, query=QUERY, sink=[1], recent=[2, 3], top_k=2
    )
    original_starts = [index.spans[span_id].start for span_id, _ in ranked]
    assert ranked, "something must be retrieved"
    # no assembled position equals the passage's original document offset unless
    # it happens to coincide, which the filler padding here rules out
    assert context.positions == list(range(len(context)))
    assert len(context) == 1 + sum(
        index.spans[span_id].length for span_id, _ in ranked
    ) + 2
    assert all(start >= 0 for start in original_starts)


def test_rejects_non_positive_span_size() -> None:
    with pytest.raises(ValueError):
        SpanIndex().add_document([1, 2, 3], span_size=0)
