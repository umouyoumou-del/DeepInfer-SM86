"""Global fact-graph reasoning over the complete 10M document.

The graph is built by scanning every line once.  Aggregation and multi-hop
questions are then executed as SQL joins, so they cover all records instead of
a top-k evidence window.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path

from deepinfer.retrieval.global_queries import GlobalResult

_GRAPH_CACHE: dict[int, tuple[str, FactGraph]] = {}

_METRIC = re.compile(r"GLOBAL\s+METRIC\s+category=([A-Za-z0-9_.-]+)\s+value=(-?\d+)")
_RELATION = re.compile(r"RELATION\s+subject=([A-Za-z0-9_.-]+)\s+relation=([A-Za-z0-9_.-]+)\s+object=([A-Za-z0-9_.-]+)")
_ENTITY = re.compile(r"ENTITY\s+id=([A-Za-z0-9_.-]+)\s+type=([A-Za-z0-9_.-]+)\s+group=([A-Za-z0-9_.-]+)")


class FactGraph:
    """SQLite graph of metrics, entities and directed relations."""

    def __init__(self, text: str = "", *, path: str | Path | None = None) -> None:
        self._lock = threading.RLock()
        if path is not None:
            self.con = sqlite3.connect(str(path), check_same_thread=False)
            return
        self.con = sqlite3.connect(":memory:", check_same_thread=False)
        self.con.execute("CREATE TABLE metrics(category TEXT, value INTEGER, position INTEGER)")
        self.con.execute("CREATE TABLE entities(id TEXT, type TEXT, grp TEXT, position INTEGER)")
        self.con.execute("CREATE TABLE relations(subject TEXT, relation TEXT, object TEXT, position INTEGER)")
        self.con.execute("CREATE TABLE triples(subject TEXT, relation TEXT, object TEXT, confidence REAL, source TEXT)")
        self.con.execute("CREATE INDEX metrics_category ON metrics(category)")
        self.con.execute("CREATE INDEX entities_type ON entities(type, grp)")
        self.con.execute("CREATE INDEX relations_sp ON relations(subject, relation)")
        for line in text.splitlines():
            offset = 0
            match = _METRIC.search(line)
            if match:
                self.con.execute("INSERT INTO metrics VALUES (?,?,?)",
                                 (match.group(1), int(match.group(2)), offset))
                continue
            match = _ENTITY.search(line)
            if match:
                self.con.execute("INSERT INTO entities VALUES (?,?,?,?)",
                                 (match.group(1), match.group(2), match.group(3), offset))
                continue
            match = _RELATION.search(line)
            if match:
                self.con.execute("INSERT INTO relations VALUES (?,?,?,?)",
                                 (match.group(1), match.group(2), match.group(3), offset))
        self.con.commit()
        self._offset = 0

    @property
    def metrics(self) -> int:
        return int(self.con.execute("SELECT count(*) FROM metrics").fetchone()[0])

    @property
    def entities(self) -> int:
        return int(self.con.execute("SELECT count(*) FROM entities").fetchone()[0])

    @property
    def relations(self) -> int:
        return int(self.con.execute("SELECT count(*) FROM relations").fetchone()[0])

    @property
    def prose_triples(self) -> int:
        return int(self.con.execute("SELECT count(*) FROM triples").fetchone()[0])

    @classmethod
    def from_sqlite(cls, path: str | Path) -> FactGraph:
        return cls(path=path)

    def save(self, path: str | Path) -> None:
        """Persist the graph so a server can load the reduce output once."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        target = sqlite3.connect(str(destination))
        try:
            with target:
                self.con.backup(target)
        finally:
            target.close()

    def add_triples(self, triples: list[dict]) -> int:
        """Merge map-reduce extracted prose triples into the same relation graph."""

        added = 0
        for item in triples:
            subject = str(item.get("subject", "")).strip()
            relation = str(item.get("relation", "")).strip().lower().replace(" ", "_")
            obj = str(item.get("object", "")).strip()
            if not subject or not relation or not obj:
                continue
            confidence = float(item.get("confidence", 1.0))
            source = str(item.get("source", "prose"))
            self.con.execute("INSERT INTO relations VALUES (?,?,?,?)", (subject, relation, obj, 0))
            self.con.execute("INSERT INTO triples VALUES (?,?,?,?,?)", (subject, relation, obj, confidence, source))
            added += 1
        self.con.commit()
        return added

    def query(self, subject: str, relation: str) -> list[str]:
        relation = relation.strip().lower().replace(" ", "_")
        rows = self.con.execute(
            "SELECT object FROM relations WHERE lower(subject)=lower(?) AND relation=?", (subject, relation)
        )
        return sorted({str(row[0]) for row in rows})

    def inverse_query(self, obj: str, relation: str) -> list[str]:
        relation = relation.strip().lower().replace(" ", "_")
        rows = self.con.execute(
            "SELECT subject FROM relations WHERE lower(object)=lower(?) AND relation=?", (obj, relation)
        )
        return sorted({str(row[0]) for row in rows})

    def metric_totals(self, *, min_value: int | None = None) -> dict[str, int]:
        sql = "SELECT category, sum(value) FROM metrics"
        args: list[int] = []
        if min_value is not None:
            sql += " WHERE value >= ?"
            args.append(min_value)
        sql += " GROUP BY category"
        return {category: int(total) for category, total in self.con.execute(sql, args)}

    def metric_counts(self, *, min_value: int | None = None) -> dict[str, int]:
        sql = "SELECT category, count(*) FROM metrics"
        args: list[int] = []
        if min_value is not None:
            sql += " WHERE value >= ?"
            args.append(min_value)
        sql += " GROUP BY category"
        return {category: int(count) for category, count in self.con.execute(sql, args)}

    def top_categories(self, k: int, *, min_value: int | None = None) -> list[tuple[str, int]]:
        totals = self.metric_totals(min_value=min_value)
        return sorted(totals.items(), key=lambda item: (-item[1], item[0]))[:k]

    def category_total(self, category: str, *, min_value: int | None = None) -> int:
        sql = "SELECT coalesce(sum(value),0) FROM metrics WHERE category=?"
        args: list[object] = [category]
        if min_value is not None:
            sql += " AND value >= ?"
            args.append(min_value)
        return int(self.con.execute(sql, args).fetchone()[0])

    def count_distinct_subjects(self, relation: str) -> int:
        return int(self.con.execute(
            "SELECT count(DISTINCT subject) FROM relations WHERE relation=?", (relation,)
        ).fetchone()[0])

    def count_entities(self, entity_type: str | None = None, group: str | None = None) -> int:
        sql = "SELECT count(*) FROM entities WHERE 1=1"
        args: list[str] = []
        if entity_type is not None:
            sql += " AND type=?"
            args.append(entity_type)
        if group is not None:
            sql += " AND grp=?"
            args.append(group)
        return int(self.con.execute(sql, args).fetchone()[0])

    def walk(self, subject: str, relations: Sequence[str]) -> list[str]:
        """Follow an arbitrary directed relation chain from one subject."""

        current = {subject}
        for relation in relations:
            following: set[str] = set()
            for node in current:
                following.update(row[0] for row in self.con.execute(
                    "SELECT object FROM relations WHERE subject=? AND relation=?", (node, relation)
                ))
            current = following
            if not current:
                break
        return sorted(current)

    def two_hop(self, subject: str, relation1: str, relation2: str) -> list[str]:
        return self.walk(subject, [relation1, relation2])


def _tokens(question: str, field: str) -> list[str]:
    return re.findall(rf"\b{field}=([A-Za-z0-9_.-]+)", question)


def cached_graph(text: str) -> FactGraph:
    """Reuse the full-text graph across questions; extraction is expensive."""

    key = id(text)
    entry = _GRAPH_CACHE.get(key)
    if entry is not None and entry[0] is text:
        return entry[1]
    graph = FactGraph(text)
    if len(_GRAPH_CACHE) >= 2:
        _GRAPH_CACHE.pop(next(iter(_GRAPH_CACHE)))
    _GRAPH_CACHE[key] = (text, graph)
    return graph


def answer_reasoning(question: str, text: str, *, graph: FactGraph | None = None) -> GlobalResult | None:
    """Answer a full-document aggregate / multi-hop question from the fact graph."""

    low = question.lower()
    if not any(key in low for key in (
        "relation", "relations", "entity", "entities", "metric", "category", "distinct subject", "followed by",
        "largest total", "total value", "owns", "own ", "founded", "works for", "located", "based", "employees",
    )):
        return None
    if graph is None:
        graph = cached_graph(text)
    # Prose questions need the map-reduce prose layer merged into the same
    # graph.  Structured-only questions keep the fast structured scan.
    prose_question = re.search(
        r"who owns|what does|who founded|where is|who works for|employees", question, re.IGNORECASE
    )
    if (graph.relations == 0 or prose_question) and graph.prose_triples == 0:
        from deepinfer.retrieval.fact_extraction import (
            RulesMapReduceExtractor,
            merge_into_graph,
        )

        merge_into_graph(graph, RulesMapReduceExtractor().run(text))
    evidence: list[dict] = []
    threshold_match = re.search(r"(?:below|under|below value|excluding values below)\s+(-?\d+)", question)
    min_value = int(threshold_match.group(1)) if threshold_match else None
    # Multi-hop first: "from subject=X by relation=R followed by relation=S".
    if "followed by" in low and ("relation" in low or "subject" in low):
        subjects = _tokens(question, "subject")
        relations = _tokens(question, "relation")
        if len(subjects) == 1 and len(relations) >= 2:
            objects = graph.walk(subjects[0], relations)
            answer = ", ".join(objects) if objects else "no matching object"
            chain = f" --{relations[0]}--> ... --{relations[-1]}--> "
            evidence = [{"line": f"{subjects[0]}{chain}{obj}", "position": 0, "depth": 0.0,
                         "source": "fact graph", "offset": 0} for obj in objects[:8]]
            return GlobalResult("graph", f"Answer: {answer}", evidence,
                                {"subjects": subjects, "relations": relations, "objects": objects})
    # Generic prose triples extracted by map-reduce.
    prose_patterns = (
        (r"who owns (.+?)\?", "inverse", "owns"),
        (r"what does (.+?) own\?", "query", "owns"),
        (r"who founded (.+?)\?", "inverse", "founded"),
        (r"where is (.+?) (?:located|based)\?", "query", "located_in"),
        (r"who works for (.+?)\?", "inverse", "works_for"),
        (r"how many employees does (.+?) have\?", "query", "has_employees"),
    )
    for pattern, operation, relation in prose_patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if not match:
            continue
        entity = match.group(1).strip().rstrip(".")
        values = graph.inverse_query(entity, relation) if operation == "inverse" else graph.query(entity, relation)
        answer = ", ".join(values) if values else "unknown"
        return GlobalResult("graph", f"Answer: {answer}", [],
                            {"entity": entity, "relation": relation, "values": values,
                             "prose_triples": graph.prose_triples})
    # Entity counts.
    if ("entity" in low or "entities" in low) and ("how many" in low or "count" in low):
        types = _tokens(question, "type")
        groups = _tokens(question, "group")
        count = graph.count_entities(types[0] if types else None, groups[0] if groups else None)
        return GlobalResult("graph", f"Answer: {count}", [],
                            {"count": count, "type": types[0] if types else None,
                             "group": groups[0] if groups else None})
    # Distinct subjects by relation.
    if "distinct" in low and "subject" in low:
        relations = _tokens(question, "relation")
        if relations:
            count = graph.count_distinct_subjects(relations[0])
            return GlobalResult("graph", f"Answer: {count}", [],
                                {"relation": relations[0], "distinct_subjects": count})
    # Metric aggregates / rankings.
    if "metric" in low or "category" in low or "total value" in low:
        totals = graph.metric_totals(min_value=min_value)
        if not totals:
            return None
        if "largest total" in low or "which category" in low:
            largest = max(totals.items(), key=lambda item: (item[1], item[0]))
            return GlobalResult("graph", f"Answer: {largest[0]}", [],
                                {"totals": totals, "largest": largest[0], "largest_total": largest[1]})
        top_match = re.search(r"\btop\s+(\d+)\b", low)
        if top_match:
            top = graph.top_categories(int(top_match.group(1)), min_value=min_value)
            answer = ", ".join(f"{name}={total}" for name, total in top)
            return GlobalResult("graph", f"Answer: {answer}", [],
                                {"top": top, "min_value": min_value})
        categories = re.findall(r"category=([A-Za-z0-9_.-]+)", question)
        if len(categories) == 1:
            total = graph.category_total(categories[0], min_value=min_value)
            return GlobalResult("graph", f"Answer: {total}", [],
                                {"category": categories[0], "total": total, "min_value": min_value})
        if "compare" in low or "totals" in low:
            ordered = sorted(totals)
            largest = max(totals, key=lambda name: (totals[name], name))
            answer = ", ".join(f"{name}={totals[name]}" for name in ordered)
            answer += f". Largest total: {largest}={totals[largest]}."
            return GlobalResult("graph", f"Answer: {answer}", [],
                                {"totals": totals, "min_value": min_value, "largest": largest})
    return None


__all__ = ["FactGraph", "answer_reasoning", "cached_graph"]
