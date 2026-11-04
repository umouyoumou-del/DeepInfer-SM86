"""Map-reduce prose extraction feeding the shared FactGraph."""

from __future__ import annotations

from deepinfer.retrieval.fact_extraction import (
    RulesMapReduceExtractor,
    merge_into_graph,
    parse_jsonl,
    reduce_triples,
)
from deepinfer.retrieval.global_queries import global_answer
from deepinfer.retrieval.global_reasoning import FactGraph

TEXT = (
    "Acme Corporation owns Beta Labs.\n"
    "Beta Labs is located in Berlin.\n"
    "Alice Chen works for Acme Corporation.\n"
    "Acme Corporation has 1,200 employees.\n"
)


def test_rules_map_reduce_extracts_prose_triples() -> None:
    triples = RulesMapReduceExtractor(max_chars=100).run(TEXT)
    keys = {(item["subject"], item["relation"], item["object"]) for item in triples}
    assert ("Acme Corporation", "owns", "Beta Labs") in keys
    assert ("Beta Labs", "located_in", "Berlin") in keys
    assert ("Alice Chen", "works_for", "Acme Corporation") in keys
    assert ("Acme Corporation", "has_employees", "1200") in keys


def test_reduce_deduplicates_case_and_whitespace() -> None:
    triples = reduce_triples([
        {"subject": "Acme  Corporation", "relation": "works for", "object": "Beta Labs"},
        {"subject": "acme corporation", "relation": "works_for", "object": "beta labs"},
    ])
    assert len(triples) == 1


def test_parse_jsonl_tolerates_code_fences() -> None:
    triples = parse_jsonl('```json\n{"subject":"A","relation":"owns","object":"B"}\n```')
    assert triples == [{"subject": "A", "relation": "owns", "object": "B",
                        "confidence": 0.8, "source": "prose-model"}]


def test_prose_triples_are_queryable_in_shared_graph() -> None:
    triples = RulesMapReduceExtractor(max_chars=100).run(TEXT)
    graph = FactGraph(TEXT)
    assert merge_into_graph(graph, triples) == 4
    assert graph.prose_triples == 4
    assert graph.inverse_query("Beta Labs", "owns") == ["Acme Corporation"]
    assert graph.query("Beta Labs", "located_in") == ["Berlin"]
    assert global_answer("Who owns Beta Labs?", TEXT).answer == "Answer: Acme Corporation"
    assert global_answer("Where is Beta Labs located?", TEXT).answer == "Answer: Berlin"
    assert global_answer("Who works for Acme Corporation?", TEXT).answer == "Answer: Alice Chen"
    assert global_answer("How many employees does Acme Corporation have?", TEXT).answer == "Answer: 1200"


def test_fact_graph_round_trip(tmp_path) -> None:
    triples = RulesMapReduceExtractor(max_chars=100).run(TEXT)
    graph = FactGraph(TEXT)
    merge_into_graph(graph, triples)
    path = tmp_path / "facts.sqlite"
    graph.save(path)
    loaded = FactGraph.from_sqlite(path)
    assert loaded.prose_triples == 4
    assert loaded.inverse_query("Beta Labs", "owns") == ["Acme Corporation"]
