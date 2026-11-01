"""Session bookkeeping that needs no model: extents, clipping, restore."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from deepinfer.moe.deepseek_longctx import DeepSeekLongContext


def _bare(window: int | None = 100, sink: int = 10) -> DeepSeekLongContext:
    session = DeepSeekLongContext.__new__(DeepSeekLongContext)
    session.encode_window = window
    session.encode_sink = sink
    session.extents = []
    session.next_position = 0
    session.slot_tokens = []
    session.bounds = None
    return session


def test_encode_window_keeps_the_sink_and_the_recent_window() -> None:
    session = _bare()
    assert session._clip_logical([[0, 1000, 0]], 500) == [[0, 10, 0], [400, 500, 400]]


def test_window_and_sink_merge_near_the_start() -> None:
    assert _bare()._clip_logical([[0, 1000, 0]], 50) == [[0, 50, 0]]


def test_inserted_text_is_clipped_by_its_logical_position() -> None:
    # slots 1000..1020 were inserted at logical 300: outside a window ending at 1000
    extents = [[0, 1000, 0], [1000, 1020, 300]]
    assert _bare()._clip_logical(extents, 1000) == [[0, 10, 0], [900, 1000, 900]]
    # but inside one ending at 400
    assert _bare()._clip_logical(extents, 400) == [[0, 10, 0], [300, 400, 300], [1000, 1020, 300]]


def test_extents_merge_only_when_slot_and_position_both_continue() -> None:
    session = _bare()
    session._add_extent(0, 100, 0)
    session._add_extent(100, 50, 100)
    session._add_extent(150, 20, 40)  # an insertion
    session._add_extent(170, 5, 150)
    assert session.extents == [[0, 150, 0], [150, 170, 40], [170, 175, 150]]
