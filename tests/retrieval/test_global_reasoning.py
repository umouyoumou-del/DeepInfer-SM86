"""Full-document fact-graph aggregation and multi-hop reasoning."""

from __future__ import annotations

from deepinfer.retrieval.global_queries import global_answer
from deepinfer.retrieval.global_reasoning import FactGraph, answer_reasoning

TEXT = "\n".join([  # noqa: FLY002
    "GLOBAL METRIC category=alpha value=10",
    "GLOBAL METRIC category=beta value=20",
    "GLOBAL METRIC category=alpha value=5",
    "GLOBAL METRIC category=gamma value=3",
    "ENTITY id=e1 type=person group=alpha",
    "ENTITY id=e2 type=org group=alpha",
    "ENTITY id=e3 type=person group=beta",
    "RELATION subject=e1 relation=owns object=e2",
    "RELATION subject=e2 relation=located_in object=e3",
    "RELATION subject=e3 relation=owns object=e4",
])


def test_fact_graph_aggregates_and_counts() -> None:
    graph = FactGraph(TEXT)
    assert graph.metric_totals() == {"alpha": 15, "beta": 20, "gamma": 3}
    assert graph.top_categories(2) == [("beta", 20), ("alpha", 15)]
    assert graph.count_distinct_subjects("owns") == 2
    assert graph.count_entities("person", "alpha") == 1
    assert graph.category_total("alpha") == 15


def test_fact_graph_two_hop() -> None:
    graph = FactGraph(TEXT)
    assert graph.two_hop("e1", "owns", "located_in") == ["e3"]
    assert graph.walk("e1", ["owns", "located_in", "owns"]) == ["e4"]


def test_natural_language_reasoning_answers() -> None:
    assert answer_reasoning("Which category has the largest total value?", TEXT).answer == "Answer: beta"
    assert answer_reasoning("List the top 2 categories by total value.", TEXT).answer == "Answer: beta=20, alpha=15"
    assert answer_reasoning("How many distinct subjects have relation=owns?", TEXT).answer == "Answer: 2"
    assert answer_reasoning(
        "Which object is reached from subject=e1 by relation=owns followed by relation=located_in?", TEXT
    ).answer == "Answer: e3"
    assert global_answer("How many entities have type=person and group=alpha?", TEXT).answer == "Answer: 1"
