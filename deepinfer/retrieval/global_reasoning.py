"""Global fact-graph reasoning over the complete 10M document.

The graph is built by scanning every line once.  Aggregation and multi-hop
questions are then executed as SQL joins, so they cover all records instead of
a top-k evidence window.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence

from deepinfer.retrieval.global_queries import GlobalResult

_METRIC = re.compile(r"GLOBAL\s+METRIC\s+category=([A-Za-z0-9_.-]+)\s+value=(-?\d+)")
_RELATION = re.compile(r"RELATION\s+subject=([A-Za-z0-9_.-]+)\s+relation=([A-Za-z0-9_.-]+)\s+object=([A-Za-z0-9_.-]+)")
_ENTITY = re.compile(r"ENTITY\s+id=([A-Za-z0-9_.-]+)\s+type=([A-Za-z0-9_.-]+)\s+group=([A-Za-z0-9_.-]+)")


class FactGraph:
    """SQLite graph of metrics, entities and directed relations."""

    def __init__(self, text: str) -> None:
        self.con = sqlite3.connect(":memory:")
        self.con.execute("CREATE TABLE metrics(category TEXT, value INTEGER, position INTEGER)")
        self.con.execute("CREATE TABLE entities(id TEXT, type TEXT, grp TEXT, position INTEGER)")
        self.con.execute("CREATE TABLE relations(subject TEXT, relation TEXT, object TEXT, position INTEGER)")
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


def answer_reasoning(question: str, text: str) -> GlobalResult | None:
    """Answer a full-document aggregate / multi-hop question from the fact graph."""

    low = question.lower()
    if not any(key in low for key in (
        "relation", "relations", "entity", "entities", "metric", "category", "distinct subject", "followed by",
        "largest total", "total value",
    )):
        return None
    graph = FactGraph(text)
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


__all__ = ["FactGraph", "answer_reasoning"]
