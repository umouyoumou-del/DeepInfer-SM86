"""The hierarchical service indexes streamed text without a full KV cache."""

from __future__ import annotations

from deepinfer.moe.hierarchical_service import HierarchicalChat


class ToyTokenizer:
    eos_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return bytes(ids).decode("utf-8", errors="replace")


def test_add_text_status_and_reset(tmp_path) -> None:
    chat = HierarchicalChat(None, ToyTokenizer(), index_path=tmp_path / "index.sqlite", capacity=1_000_000)
    source = chat.add_text(
        "The special magic number for onyx badger is: 7646461.\n",
        "sample.txt",
    )
    assert source.tokens == len(b"The special magic number for onyx badger is: 7646461.\n")
    status = chat.status()
    assert status["mode"] == "hierarchical"
    assert status["context"] == source.tokens
    assert status["index_chunks"] >= 1
    chat.reset()
    assert chat.status()["context"] == 0
    assert chat.status()["index_chunks"] == 0


def test_structured_fast_path_event(tmp_path) -> None:
    chat = HierarchicalChat(None, ToyTokenizer(), index_path=tmp_path / "index.sqlite", capacity=1_000_000)
    chat.add_text("The special magic number for onyx badger is: 7646461.\n", "facts.txt")
    events = chat.ask("What is the special magic number for onyx badger?")
    seen = []
    while True:
        event = events.get(timeout=10)
        seen.append(event)
        if event.get("event") in ("done", "error"):
            break
    done = next(event for event in seen if event.get("event") == "done")
    assert done["final"] == "7646461"
    assert done["rounds"] == 1


def test_global_count_event(tmp_path) -> None:
    chat = HierarchicalChat(None, ToyTokenizer(), index_path=tmp_path / "index.sqlite", capacity=1_000_000)
    chat.add_text("GLOBAL_MARKER_X one.\nGLOBAL_MARKER_X two.\nGLOBAL_MARKER_X three.\n", "counts.txt")
    events = chat.ask("How many times does GLOBAL_MARKER_X occur in the document?")
    seen = []
    while True:
        event = events.get(timeout=10)
        seen.append(event)
        if event.get("event") in ("done", "error"):
            break
    done = next(event for event in seen if event.get("event") == "done")
    assert done["final"] == "3"
    assert done["global_kind"] == "count"


def test_global_summary_event(tmp_path) -> None:
    chat = HierarchicalChat(None, ToyTokenizer(), index_path=tmp_path / "index.sqlite", capacity=1_000_000)
    themes = "\n\n".join([
        "THEME_ALPHA explains low latency retrieval and incremental indexing.",
        "THEME_BETA explains persistent evidence indexes and warm starts.",
        "THEME_GAMMA explains full text counting instead of top-k sampling.",
        "THEME_DELTA explains depth coverage across the whole document.",
        "THEME_EPSILON explains deterministic aggregation of every record.",
    ] * 3)
    chat.add_text(themes, "themes.txt")
    events = chat.ask("Summarize the main themes of the document.")
    seen = []
    while True:
        event = events.get(timeout=10)
        seen.append(event)
        if event.get("event") in ("done", "error"):
            break
    done = next(event for event in seen if event.get("event") == "done")
    for theme in ("THEME_ALPHA", "THEME_BETA", "THEME_GAMMA", "THEME_DELTA", "THEME_EPSILON"):
        assert theme in done["answer"]


def test_global_compare_event(tmp_path) -> None:
    chat = HierarchicalChat(None, ToyTokenizer(), index_path=tmp_path / "index.sqlite", capacity=1_000_000)
    chat.add_text(
        "GLOBAL METRIC category=alpha value=10\n"
        "GLOBAL METRIC category=beta value=20\n"
        "GLOBAL METRIC category=alpha value=5\n",
        "metrics.txt",
    )
    events = chat.ask("Compare the total value by category in the GLOBAL METRIC records.")
    seen = []
    while True:
        event = events.get(timeout=10)
        seen.append(event)
        if event.get("event") in ("done", "error"):
            break
    done = next(event for event in seen if event.get("event") == "done")
    assert "alpha=15" in done["answer"]
    assert "Largest total: beta=20" in done["answer"]


def test_persisted_fact_graph_answers_prose_without_model(tmp_path) -> None:
    from deepinfer.retrieval.fact_extraction import (
        RulesMapReduceExtractor,
        merge_into_graph,
    )
    from deepinfer.retrieval.global_reasoning import FactGraph

    text = (
        "Acme Corporation owns Beta Labs.\n"
        "Beta Labs is located in Berlin.\n"
        "Alice Chen works for Acme Corporation.\n"
        "Acme Corporation has 1,200 employees.\n"
    )
    text_path = tmp_path / "prose.txt"
    text_path.write_text(text, encoding="utf-8")
    graph = FactGraph(text)
    merge_into_graph(graph, RulesMapReduceExtractor(max_chars=100).run(text))
    graph_path = tmp_path / "facts.sqlite"
    graph.save(graph_path)
    chat = HierarchicalChat(
        None, ToyTokenizer(), text_path=text_path, index_path=tmp_path / "index.sqlite",
        capacity=1_000_000, fact_graph_path=graph_path,
    )
    events = chat.ask("Who owns Beta Labs?")
    seen = []
    while True:
        event = events.get(timeout=10)
        seen.append(event)
        if event.get("event") in ("done", "error"):
            break
    done = next(event for event in seen if event.get("event") == "done")
    assert done["final"] == "Acme Corporation"
    assert done["global_kind"] == "graph"
