"""Full-document count, summary and metric aggregation paths."""

from __future__ import annotations

from deepinfer.retrieval.global_queries import (
    aggregate_metrics,
    count_occurrences,
    extractive_summary,
    global_answer,
    parse_metric_records,
)

COUNT_TEXT = "GLOBAL_MARKER_X appears here.\n" + "code\n" * 20 + "GLOBAL_MARKER_X again.\nGLOBAL_MARKER_X end.\n"

SUMMARY_TEXT = "\n\n".join([
    "THEME_ALPHA explains that the system prefers low latency retrieval and incremental indexing.",
    "THEME_BETA explains that the serving layer keeps a persistent evidence index for warm starts.",
    "THEME_GAMMA explains that global counts scan every token instead of taking a top-k shortcut.",
    "THEME_DELTA explains that summaries use depth coverage across the whole document.",
    "THEME_EPSILON explains that structured facts are aggregated deterministically.",
] * 3)

METRIC_TEXT = "\n".join([  # noqa: FLY002
    "GLOBAL METRIC category=alpha value=10",
    "GLOBAL METRIC category=beta value=20",
    "GLOBAL METRIC category=alpha value=5",
    "GLOBAL METRIC category=gamma value=3",
])


def test_exact_count_covers_whole_text() -> None:
    result = count_occurrences(COUNT_TEXT, "GLOBAL_MARKER_X")
    assert result["count"] == 3
    answer = global_answer("How many times does GLOBAL_MARKER_X occur in the document?", COUNT_TEXT)
    assert answer is not None and answer.kind == "count" and answer.answer == "Answer: 3"


def test_extractive_summary_covers_depth_buckets() -> None:
    result = extractive_summary(SUMMARY_TEXT, max_chars=4000, buckets=5)
    assert result.kind == "summary"
    for theme in ("THEME_ALPHA", "THEME_BETA", "THEME_GAMMA", "THEME_DELTA", "THEME_EPSILON"):
        assert theme in result.answer


def test_metric_aggregation() -> None:
    records = parse_metric_records(METRIC_TEXT)
    assert len(records) == 4
    result = aggregate_metrics(METRIC_TEXT)
    assert result is not None
    assert result.metrics["totals"] == {"alpha": 15, "beta": 20, "gamma": 3}
    assert result.metrics["largest"] == "beta"
    assert "alpha=15" in result.answer and "Largest total: beta=20" in result.answer
