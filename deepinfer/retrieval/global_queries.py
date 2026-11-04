"""Full-document global queries: exact counts, extractive summaries, aggregations.

Top-k retrieval is the right default for fact lookup, but it is not a global
reader.  This module covers the three query families that need every token:

* literal counts are counted over the complete text, not sampled;
* summaries are selected with per-depth coverage across the whole document;
* machine-readable records are aggregated over every line.

The extractive summary is deterministic and keeps a depth citation for each
selected passage.  It is intentionally not a model-generated abstract: the
caller can synthesize from the returned full-coverage passages.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

_TOKEN = re.compile(r"[A-Za-z0-9_]{3,}")
_STOP = frozenset({
    "the", "and", "for", "from", "that", "this", "with", "are", "was", "were", "have", "has",
    "not", "you", "your", "can", "will", "into", "when", "then", "than", "but", "all", "any",
    "use", "used", "using", "its", "they", "them", "our", "out", "one", "two", "also", "such",
    "may", "more", "most", "some", "only", "self", "none", "true", "false", "def", "class",
    "return", "import", "pass", "while", "else", "elif", "try", "except",
    "raise", "lambda", "async", "await", "args", "kwargs",
})


@dataclass
class GlobalResult:
    kind: str
    answer: str
    evidence: list[dict]
    metrics: dict


def classify(question: str) -> str | None:
    low = question.lower()
    if any(key in low for key in ("summarize", "summary", "overview", "tl;dr", "总结", "概括", "摘要")):
        return "summary"
    if any(key in low for key in ("how many", "count", "occurrence", "occurrences", "多少次", "几个", "计数")):
        return "count"
    if any(key in low for key in ("compare", "distribution", "group by", "aggregate", "largest", "smallest",
                                  "比较", "分布", "统计", "最多", "最少", "汇总")):
        return "compare"
    return None


def count_occurrences(text: str, term: str, *, ignore_case: bool = False) -> dict:
    """Count every literal occurrence in the complete text."""

    flags = re.IGNORECASE if ignore_case else 0
    matches = list(re.finditer(re.escape(term), text, flags))
    return {
        "term": term,
        "count": len(matches),
        "positions": [match.start() for match in matches[:20]],
        "total_chars": len(text),
    }


def _count_term(question: str) -> str | None:
    for pattern in (r"`([^`]+)`", r'"([^"]+)"', r"\b([A-Z][A-Z0-9_]{2,})\b"):
        match = re.search(pattern, question)
        if match:
            return match.group(1).strip()
    match = re.search(r"(?:occurrences? of|count of|occur in|does)\s+([A-Za-z0-9_.-]{2,})", question, re.IGNORECASE)
    return None if match is None else match.group(1)


def _units(text: str) -> list[tuple[int, str, set[str]]]:
    """Line-sized units with their character offsets and content tokens.

    Lines keep short high-information facts (themes, constants, records) from
    being swallowed by a long code paragraph.  Long lines are filtered out; the
    caller still covers every depth bucket.
    """

    units: list[tuple[int, str, set[str]]] = []
    offset = 0
    for raw in text.splitlines(keepends=True):
        chunk = raw.strip()
        if 40 <= len(chunk) <= 1200:
            tokens = {token.lower() for token in _TOKEN.findall(chunk) if token.lower() not in _STOP}
            if len(tokens) >= 3:
                units.append((offset, chunk, tokens))
        offset += len(raw)
    return units

def _prose_score(chunk: str) -> float:
    """Prefer prose sentences over dense code when two units tie on rarity."""

    words = len(re.findall(r"[A-Za-z]{3,}", chunk))
    if not words:
        return 0.0
    punctuation = len(re.findall(r"[={}\[\]();]", chunk))
    stripped = chunk.lstrip()
    code_start = stripped.startswith(("#", "def ", "class ", "async def ", "import ", "from ",
                                     "return ", "elif ", "if ", "for ", "while ", "try:", "except "))
    return words / (1.0 + punctuation + (2.0 if code_start else 0.0))


def extractive_summary(text: str, *, max_chars: int = 12_000, buckets: int = 16) -> GlobalResult:
    """Select high-information paragraphs while covering every depth bucket."""

    began = time.perf_counter()
    units = _units(text)
    if not units:
        return GlobalResult("summary", text[:max_chars], [], {"units": 0, "selected": 0})
    df = Counter()
    for _, _, tokens in units:
        df.update(tokens)
    total = len(units)
    idf = {token: math.log((total + 1) / (count + 1)) + 1.0 for token, count in df.items()}
    scored = []
    for start, chunk, tokens in units:
        total_idf = sum(idf[token] for token in tokens)
        score = total_idf / math.sqrt(len(tokens))
        scored.append((score, max(idf[token] for token in tokens), _prose_score(chunk), start, chunk))
    per_bucket: dict[int, list[tuple[float, float, float, int, str]]] = defaultdict(list)
    length = max(1, len(text))
    for item in scored:
        bucket = min(buckets - 1, item[3] * buckets // length)
        per_bucket[bucket].append(item)
    # First reserve the rarest-term line in every depth bucket, then fill in
    # high-density lines.  The first pass is what keeps themes and unique
    # constants in the summary when the character budget is tight.
    selected: list[tuple[float, float, float, int, str]] = []
    rare_starts: set[int] = set()
    seen: set[int] = set()
    # Pass 0: keep one uppercase-heading line per depth bucket.  In mixed
    # code/docs these are the document's own section/theme markers.
    heading_starts: set[int] = set()
    for bucket in range(buckets):
        headings = [item for item in per_bucket.get(bucket, [])
                    if re.match(r"^[A-Z][A-Z0-9_]{2,}:", item[4].lstrip())]
        if headings:
            heading = max(headings, key=lambda item: (item[2], item[1], item[0]))
            heading_starts.add(heading[3])
            rare_starts.add(heading[3])
            if heading[3] not in seen:
                seen.add(heading[3])
                selected.append(heading)
    # Pass 1: reserve one rare-term line in every depth bucket.  These are
    # emitted first and shortened to 400 chars so the coverage slots fit.
    for bucket in range(buckets):
        items = per_bucket.get(bucket, [])
        if not items:
            continue
        rare = max(items, key=lambda item: (item[1], item[2], -len(item[4])))
        rare_starts.add(rare[3])
        if rare[3] not in seen:
            seen.add(rare[3])
            selected.append(rare)
    # Pass 2: fill the remaining budget with information-dense lines.
    for bucket in range(buckets):
        for item in sorted(per_bucket.get(bucket, []), reverse=True)[:3]:
            if item[3] not in seen:
                seen.add(item[3])
                selected.append(item)
    chosen: list[tuple[int, str]] = []
    lines: list[str] = []
    used = 0
    for _score, _max_idf, _prose, start, chunk in selected:
        # Keep the rare-term line short so every depth bucket gets its slot
        # even when the corpus contains very long code lines.
        limit = 400 if start in rare_starts else 700
        line = f"[{100 * start / length:5.1f}%] {chunk[:limit]}"
        if used + len(line) > max_chars:
            continue
        chosen.append((start, chunk))
        lines.append(line)
        used += len(line) + 1
    paired = sorted(zip(chosen, lines, strict=True), key=lambda pair: pair[0][0])
    chosen = [pair[0] for pair in paired]
    lines = [pair[1] for pair in paired]
    answer = "Full-document extractive summary:" + chr(10) + chr(10).join(lines)
    evidence = [{"line": line, "position": start, "depth": start / length, "source": "document", "offset": 0}
                for (start, _chunk), line in zip(chosen, lines, strict=True)]
    return GlobalResult(
        "summary", answer, evidence,
        {"units": total, "selected": len(lines), "buckets": buckets, "chars": used,
         "seconds": round(time.perf_counter() - began, 3)},
    )


def parse_metric_records(text: str) -> list[dict]:
    """Read every machine-readable GLOBAL METRIC record in the document."""

    records = []
    pattern = re.compile(r"GLOBAL\s+METRIC\s+category=([A-Za-z0-9_.-]+)\s+value=(-?\d+)")
    for match in pattern.finditer(text):
        records.append({"category": match.group(1), "value": int(match.group(2)), "position": match.start()})
    return records


def aggregate_metrics(text: str) -> GlobalResult | None:
    records = parse_metric_records(text)
    if not records:
        return None
    totals: dict[str, int] = defaultdict(int)
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        totals[record["category"]] += record["value"]
        counts[record["category"]] += 1
    ordered = sorted(totals, key=lambda category: (-totals[category], category))
    largest = ordered[0]
    answer = "Totals by category: " + ", ".join(f"{name}={totals[name]}" for name in ordered)
    answer += f". Largest total: {largest}={totals[largest]}."
    evidence = [{"line": f"{name}: count={counts[name]}, total={totals[name]}", "position": 0,
                 "depth": 0.0, "source": "GLOBAL METRIC", "offset": 0} for name in ordered]
    return GlobalResult("compare", answer, evidence, {
        "records": len(records), "categories": len(totals), "totals": dict(totals), "largest": largest,
    })


def global_answer(question: str, text: str, facts: Sequence[str] = (), *, graph=None) -> GlobalResult | None:
    """Route a question to the full-document count/summary/aggregation/graph path."""

    # Fact-graph reasoning (aggregation, distinct counts, multi-hop) is tried
    # before the simpler count/compare classifiers.
    from deepinfer.retrieval.global_reasoning import answer_reasoning

    reasoned = answer_reasoning(question, text, graph=graph)
    if reasoned is not None:
        return reasoned
    kind = classify(question)
    if kind == "summary":
        return extractive_summary(text)
    if kind == "count":
        term = _count_term(question)
        if not term:
            return None
        counted = count_occurrences(text, term)
        answer = f"Answer: {counted['count']}"
        evidence = [{"line": f"{term}: {counted['count']} occurrences", "position": pos, "depth": pos / max(1, len(text)),
                     "source": "full-text count", "offset": 0} for pos in counted["positions"][:8]]
        return GlobalResult("count", answer, evidence, {"term": term, "count": counted["count"],
                                                        "positions": counted["positions"][:8]})
    if kind == "compare":
        return aggregate_metrics(text)
    return None


__all__ = [
    "GlobalResult",
    "aggregate_metrics",
    "classify",
    "count_occurrences",
    "extractive_summary",
    "global_answer",
    "parse_metric_records",
]
