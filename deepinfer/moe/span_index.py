"""Retrieve text spans, then recompute their KV where they are needed.

Item six established why retrieval cannot hand back KV blocks: a cached block
carries the absolute positions it was computed at, so copying a passage from
position 900,000 into a context sitting at position 4,000 is numerically wrong
under RoPE.  What can be moved is the *text*; its KV has to be rebuilt at the
positions the assembled context gives it.

So the shape of the thing is: index spans of the document, score them against
the current query, and assemble a fresh context out of sinks, the retrieved
spans and the recent window, laid out consecutively.  Recomputing that context
is one ordinary prefill.

Scoring here is token overlap rather than embeddings, so the module has no model
dependency and can be tested exactly.  Swapping in an embedding index only
changes ``_score``, not the assembly contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class Span:
    span_id: int
    token_ids: list[int]
    start: int          # offset in the original document, for provenance
    length: int


@dataclass
class SpanIndex:
    """Fixed-size spans of one or more documents, searchable by token overlap."""

    spans: list[Span] = field(default_factory=list)
    _postings: dict[int, set[int]] = field(default_factory=dict)
    _lengths: dict[int, int] = field(default_factory=dict)

    def add_document(self, token_ids: list[int], *, span_size: int = 128) -> list[int]:
        """Split a document into spans and index it; returns the new span ids."""

        if span_size <= 0:
            raise ValueError("span_size must be positive")
        added: list[int] = []
        for start in range(0, len(token_ids), span_size):
            piece = list(token_ids[start : start + span_size])
            if not piece:
                continue
            span_id = len(self.spans)
            self.spans.append(
                Span(span_id=span_id, token_ids=piece, start=start, length=len(piece))
            )
            for token in set(piece):
                self._postings.setdefault(int(token), set()).add(span_id)
            self._lengths[span_id] = len(piece)
            added.append(span_id)
        return added

    def _score(self, query: list[int], span_id: int) -> float:
        """Overlap normalised by the span length, so long spans do not win by size."""

        span = self.spans[span_id]
        wanted = set(int(token) for token in query)
        hits = sum(1 for token in span.token_ids if int(token) in wanted)
        return hits / max(1, span.length)

    def search(self, query: list[int], *, top_k: int = 4) -> list[tuple[int, float]]:
        """Best ``top_k`` spans for a query, highest score first."""

        if top_k <= 0:
            return []
        candidates: set[int] = set()
        for token in set(int(t) for t in query):
            candidates |= self._postings.get(token, set())
        scored = [(self._score(query, span_id), span_id) for span_id in candidates]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [(span_id, score) for score, span_id in scored[:top_k] if score > 0]


@dataclass
class AssembledContext:
    token_ids: list[int]
    positions: list[int]
    provenance: list[tuple[str, int]]   # ("sink"|"retrieved"|"recent", span_id)

    def __len__(self) -> int:
        return len(self.token_ids)


def assemble_context(
    *,
    sink: list[int],
    retrieved: list[list[int]],
    recent: list[int],
) -> AssembledContext:
    """Lay sinks, retrieved spans and the recent window out at fresh positions.

    Order matters for what the model can condition on cheaply: sinks first
    because attention dumps excess mass there, retrieved spans next so they are
    adjacent to the recent window rather than buried, and the window last so the
    most recent tokens keep the position the model expects.
    """

    tokens: list[int] = []
    provenance: list[tuple[str, int]] = []
    for token in sink:
        tokens.append(int(token))
        provenance.append(("sink", -1))
    for index, span in enumerate(retrieved):
        for token in span:
            tokens.append(int(token))
            provenance.append(("retrieved", index))
    for token in recent:
        tokens.append(int(token))
        provenance.append(("recent", -1))
    return AssembledContext(
        token_ids=tokens,
        positions=list(range(len(tokens))),
        provenance=provenance,
    )


def assemble_from_index(
    index: SpanIndex,
    *,
    query: list[int],
    sink: list[int],
    recent: list[int],
    top_k: int = 4,
) -> tuple[AssembledContext, list[tuple[int, float]]]:
    """Search, then assemble; returns the context and the ranking that produced it."""

    ranked = index.search(query, top_k=top_k)
    spans = [index.spans[span_id].token_ids for span_id, _score in ranked]
    return assemble_context(sink=sink, retrieved=spans, recent=recent), ranked
