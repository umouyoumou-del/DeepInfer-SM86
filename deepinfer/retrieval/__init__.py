"""Retrieval backends for contexts that do not fit in KV memory."""

from deepinfer.retrieval.fact_extraction import (
    ModelMapReduceExtractor,
    RulesMapReduceExtractor,
    merge_into_graph,
    reduce_triples,
)
from deepinfer.retrieval.global_queries import (
    aggregate_metrics,
    classify,
    count_occurrences,
    extractive_summary,
    global_answer,
    parse_metric_records,
)
from deepinfer.retrieval.global_reasoning import FactGraph, answer_reasoning
from deepinfer.retrieval.hybrid import (
    FACT_KEYS,
    build_index,
    chunk_text,
    exact_lines,
    facts_from_lines,
    retrieve,
    structured_answer,
)

__all__ = [
    "FACT_KEYS",
    "FactGraph",
    "ModelMapReduceExtractor",
    "RulesMapReduceExtractor",
    "aggregate_metrics",
    "answer_reasoning",
    "build_index",
    "chunk_text",
    "classify",
    "count_occurrences",
    "exact_lines",
    "extractive_summary",
    "facts_from_lines",
    "global_answer",
    "merge_into_graph",
    "parse_metric_records",
    "reduce_triples",
    "retrieve",
    "structured_answer",
]
