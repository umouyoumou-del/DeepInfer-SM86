"""Map-reduce fact extraction for unlabeled prose.

The map phase runs over bounded passages.  The reduce phase normalizes and
deduplicates triples before they are merged into the same FactGraph used by the
structured 10M reasoning path.  Two map backends are provided:

* ``rules``  - deterministic, fast, and useful for explicit prose patterns;
* ``model``  - the local DeepSeek runtime for open-ended prose extraction.

The rule backend is the production-safe default; the model backend is intended
for offline enrichment where every token can be read by the model.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from typing import Any

from deepinfer.retrieval.global_reasoning import FactGraph

_ENTITY = r"[A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*){0,5}"
_OBJECT = r"[A-Z0-9][A-Za-z0-9&.'-]*(?:\s+[A-Z0-9][A-Za-z0-9&.'-]*){0,5}"
_PROSE = (
    re.compile(rf"(?P<subject>{_ENTITY})\s+(?P<relation>owns|acquired|founded|leads|works for|is based in|"
               rf"is located in|reports to)\s+(?P<object>{_OBJECT})"),
    re.compile(rf"(?P<subject>{_ENTITY})\s+has\s+(?P<count>\d[\d,]*)\s+employees"),
    re.compile(rf"(?P<subject>{_ENTITY})\s+is\s+(?P<relation>owned by)\s+(?P<object>{_OBJECT})"),
)
_RELATION_NORMAL = {
    "owns": "owns",
    "owned by": "owned_by",
    "acquired": "acquired",
    "founded": "founded",
    "leads": "leads",
    "works for": "works_for",
    "is based in": "located_in",
    "is located in": "located_in",
    "reports to": "reports_to",
}


def normalize_relation(relation: str) -> str:
    low = relation.strip().lower()
    return _RELATION_NORMAL.get(low, low.replace(" ", "_"))


def _clean(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip())
    return value.strip(" \t\n,.;:()[]{}\"'")


def chunk_text(text: str, *, max_chars: int = 4000, overlap: int = 0) -> list[str]:
    """Map passages without cutting sentences when a line boundary is available."""

    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in text.splitlines(keepends=True):
        if size + len(line) > max_chars and current:
            chunks.append("".join(current))
            if overlap and len(current) > 1:
                current = current[-overlap:]
            else:
                current = []
            size = sum(len(item) for item in current)
        current.append(line)
        size += len(line)
    if current:
        chunks.append("".join(current))
    return chunks


def extract_prose_rules(text: str) -> list[dict[str, Any]]:
    """Extract explicit prose triples from one map chunk."""

    found: list[dict[str, Any]] = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        if not sentence.strip():
            continue
        for pattern in _PROSE:
            for match in pattern.finditer(sentence):
                subject = _clean(match.group("subject"))
                obj = _clean(match.group("object")) if "object" in match.groupdict() else ""
                if "count" in match.groupdict():
                    relation = "has_employees"
                    obj = match.group("count").replace(",", "")
                else:
                    relation = normalize_relation(match.group("relation"))
                if subject and relation and obj:
                    found.append({
                        "subject": subject, "relation": relation, "object": obj,
                        "confidence": 1.0, "source": "prose-rules",
                    })
    return found


def reduce_triples(triples: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize and deduplicate map output in the reduce phase."""

    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in triples:
        subject = _clean(str(item.get("subject", "")))
        relation = normalize_relation(str(item.get("relation", "")))
        obj = _clean(str(item.get("object", "")))
        if not subject or not relation or not obj:
            continue
        employee_match = re.fullmatch(r"([\d,]+)\s+employees?", obj, re.IGNORECASE)
        if relation in {"has", "has_employees"} and employee_match:
            relation = "has_employees"
            obj = employee_match.group(1).replace(",", "")
        key = (subject.casefold(), relation, obj.casefold())
        if key not in merged:
            merged[key] = {
                "subject": subject, "relation": relation, "object": obj,
                "confidence": float(item.get("confidence", 1.0)),
                "source": str(item.get("source", "prose")),
            }
        else:
            merged[key]["confidence"] = max(merged[key]["confidence"], float(item.get("confidence", 1.0)))
    return sorted(merged.values(), key=lambda item: (item["subject"], item["relation"], item["object"]))


class RulesMapReduceExtractor:
    """Fast deterministic map-reduce backend for explicit prose sentences."""

    def __init__(self, *, max_chars: int = 4000) -> None:
        self.max_chars = max_chars

    def map(self, text: str) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for chunk in chunk_text(text, max_chars=self.max_chars):
            output.extend(extract_prose_rules(chunk))
        return output

    def run(self, text: str) -> list[dict[str, Any]]:
        return reduce_triples(self.map(text))


class ModelMapReduceExtractor:
    """Map-reduce backend that asks the local model to extract JSONL triples."""

    PROMPT = (
        "Extract explicit factual triples from the passage. Return JSONL only, one object per line:\n"
        '{"subject":"...","relation":"...","object":"..."}\n'
        "Use short lowercase snake_case relations. Do not invent facts.\n\nPassage:\n"
    )
    SUFFIX = "\n\nJSONL:\n"

    def __init__(self, runtime, tok, *, max_new_tokens: int = 256, chunk_tokens: int = 2048) -> None:
        from deepinfer.moe.latent_mla import DualChunkPositions

        self.runtime = runtime
        self.tok = tok
        self.max_new_tokens = max_new_tokens
        self.chunk_tokens = chunk_tokens
        self.positions = DualChunkPositions(131072, 131072 * 3 // 4, -1)

    def extract_chunk(self, passage: str) -> list[dict[str, Any]]:
        from deepinfer.moe.deepseek_longctx import DeepSeekLongContext

        passage_ids = self.tok.encode(passage, add_special_tokens=False)[: self.chunk_tokens]
        prefix = self.tok.encode(self.PROMPT, add_special_tokens=False)
        suffix = self.tok.encode(self.SUFFIX, add_special_tokens=False)
        capacity = len(prefix) + len(passage_ids) + len(suffix) + self.max_new_tokens + 8
        session = DeepSeekLongContext(
            self.runtime, capacity=capacity, gpu_tokens=capacity, kv_dtype="fp8",
            positions=self.positions, prefill_chunk=16384, tile_tokens=32768, encode_window=98304,
        )
        session.prefill(prefix + passage_ids)
        out = session.generate(suffix, max_new_tokens=self.max_new_tokens,
                               stop_ids=[self.tok.eos_token_id])
        return parse_jsonl(self.tok.decode(out, skip_special_tokens=True))

    def map(self, text: str, *, max_chunks: int | None = None) -> list[dict[str, Any]]:
        chunks = chunk_text(text, max_chars=max(1000, self.chunk_tokens * 4))
        if max_chunks is not None:
            chunks = chunks[:max_chunks]
        output: list[dict[str, Any]] = []
        for chunk in chunks:
            output.extend(self.extract_chunk(chunk))
        return output

    def run(self, text: str, *, max_chunks: int | None = None) -> list[dict[str, Any]]:
        return reduce_triples(self.map(text, max_chunks=max_chunks))


def parse_jsonl(text: str) -> list[dict[str, Any]]:
    """Parse model JSONL, tolerating code fences and leading prose."""

    triples: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip().strip("`")
        if not line:
            continue
        match = re.search(r"\{.*\}", line)
        if not match:
            continue
        try:
            item = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            triples.append({
                "subject": item.get("subject", ""),
                "relation": item.get("relation", ""),
                "object": item.get("object", ""),
                "confidence": item.get("confidence", 0.8),
                "source": "prose-model",
            })
    return triples


def merge_into_graph(graph: FactGraph, triples: Sequence[dict[str, Any]]) -> int:
    """Merge reduced triples into the same FactGraph used for structured reasoning."""

    return graph.add_triples(list(triples))


__all__ = [
    "ModelMapReduceExtractor",
    "RulesMapReduceExtractor",
    "chunk_text",
    "extract_prose_rules",
    "merge_into_graph",
    "parse_jsonl",
    "reduce_triples",
]
