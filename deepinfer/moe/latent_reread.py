"""Locate in the KV library, then re-read the located text at short range.

At a million tokens this model can still *find* a fact - block retrieval over
the latent cache lands on the right passage at most depths - but copying seven
digits out of keys sitting far beyond its trained length drifts.  At 8K tokens
it copies perfectly.  So the two jobs are split:

1. locate: the question is run against the whole cached sequence and every
   layer's queries are scored against every cached slot - exact attention,
   normalized over the cache - and the mass is summed per block over layers,
   heads and question tokens.  This is the model's own notion of where the
   answer is; nothing lexical (the question repeats the fact's words, so token
   overlap would be cheating).
2. re-read: the token ids of the best blocks and their neighbours, in their
   original order, become a short document; a fresh short context answers.

Every token of the long sequence stays encoded and searchable; only the answer
is produced where the model is reliable.

With many similar facts in the library ("the special magic number for crimson
falcon / crimson otter / silver otter ..."), summing the mass over every head
ranks the asked fact badly: most heads spread their mass by position (recency,
sinks, the dual-chunk pattern), drowning the few that follow the distinguishing
words.  Two changes fix the ranking (``reduce="max"`` and :func:`contrast`):
take the sharpest (query, head) row per block instead of the sum, and subtract
what a neutral prompt attends to anyway.

Questions whose answer needs facts the question does not name - a chain of
variable assignments, "the badge number of the person who holds the key" - go
through :func:`feedback_locate_and_reread`: each round's answer names what has
been found, and the next locate runs with question + answer.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Sequence

import torch

from deepinfer.moe import latent_locate
from deepinfer.moe.latent_mla import (
    DualChunkPositions,
    LatentKVCache,
    LatentRope,
    MLAWeights,
    extent_regions,
)

BLOCK = 64


@torch.inference_mode()
def block_mass(
    weights: MLAWeights,
    cache: LatentKVCache,
    layer: int,
    q_nope: torch.Tensor,
    q_pe_raw: torch.Tensor,
    query_start: int,
    regions: list[tuple[int, int, int]],
    *,
    positions: DualChunkPositions,
    rope: LatentRope,
    block: int,
    blocks: int,
    tile: int = 65536,
    reduce: str = "sum",
    keep_tokens: Sequence[int] | None = None,
) -> torch.Tensor:
    """``[blocks]`` attention mass of ``n`` queries over the slots in ``regions``.

    Queries are at logical ``query_start ..``; ``q_nope [n,H,128]``,
    ``q_pe_raw [n,H,64]``.  Each (query, head) distribution is normalized over
    the region slots and summed per block of ``block`` slots; ``reduce`` then
    combines the (query, head) rows: ``"sum"`` (total mass) or ``"max"`` (the
    sharpest row - a head that looks at one block counts even if most heads
    spread their mass by position).  ``keep_tokens`` (indices into the ``n``
    queries) limits the rows that count.
    """

    n, heads, _ = q_nope.shape
    device = q_nope.device
    # a long query (a question plus a fed-back answer) means many rows: keep a score
    # tile near 2**25 entries (128 MB in fp32) so memory does not grow with the query
    tile = min(tile, max(1024, (1 << 25) // max(1, n * heads)))
    q_abs = torch.bmm(q_nope.transpose(0, 1).to(weights.w_uk.dtype), weights.w_uk)
    q_abs = q_abs.transpose(0, 1).reshape(n * heads, -1)  # [n*H, 512]
    absolute = torch.arange(query_start, query_start + n, device=device)
    rotated = {
        relation: rope.apply(q_pe_raw, positions.query_positions(absolute, relation)).reshape(
            n * heads, -1
        )
        for relation in {relation for _, _, relation in regions}
    }
    scale = weights.softmax_scale
    if keep_tokens is not None:
        keep = torch.tensor(
            [token * heads + head for token in keep_tokens for head in range(heads)], dtype=torch.long, device=device
        )
        q_abs = q_abs.index_select(0, keep)
        rotated = {relation: rows.index_select(0, keep) for relation, rows in rotated.items()}
        n = len(keep_tokens) if keep_tokens else 0
    if n == 0:
        return torch.zeros(blocks, dtype=torch.float32, device=device)

    def tiles():
        for start, end, relation in regions:
            for lo in range(start, end, tile):
                hi = min(lo + tile, end)
                latent, pe = cache.tile(layer, lo, hi)
                scores = q_abs.to(latent.dtype) @ latent.t()
                scores = scores + rotated[relation].to(pe.dtype) @ pe.t()
                yield lo, hi, scores.float() * scale

    lse = None
    for _, _, scores in tiles():
        part = torch.logsumexp(scores, dim=-1)
        lse = part if lse is None else torch.logaddexp(lse, part)
    if reduce == "max" and block == 1:
        # one slot per block: the sharpest row per slot, without a rows x slots buffer
        sharpest = torch.zeros(blocks, dtype=torch.float32, device=device)
        for lo, hi, scores in tiles():
            peak = torch.exp(scores - lse.unsqueeze(-1)).amax(dim=0)
            sharpest[lo:hi] = torch.maximum(sharpest[lo:hi], peak)
        return sharpest
    if reduce == "max":
        rows = torch.zeros(n * heads, blocks, dtype=torch.float32, device=device)
        for lo, hi, scores in tiles():
            ids = torch.arange(lo, hi, device=device) // block
            rows.index_add_(1, ids, torch.exp(scores - lse.unsqueeze(-1)))
        return rows.max(dim=0).values
    mass = torch.zeros(blocks, dtype=torch.float32, device=device)
    for lo, hi, scores in tiles():
        per_slot = torch.exp(scores - lse.unsqueeze(-1)).sum(dim=0)
        ids = torch.arange(lo, hi, device=device) // block
        mass.index_add_(0, ids, per_slot)
    return mass


def pick_blocks(
    mass: torch.Tensor,
    *,
    block: int,
    top: int,
    skip_before: int = 0,
    exclude: Sequence[int] = (),
    near_start: int | None = None,
) -> list[int]:
    """Indices of the ``top`` heaviest blocks, sorted; sink blocks and ``exclude`` are not candidates.

    With ``near_start`` (a block index) the blocks from there on - the most recent
    stretch, where attention runs high because of position - get a quota in
    proportion to their share of the sequence (at least an eighth of ``top``)
    instead of competing for every slot.
    """

    scores = mass.clone()
    scores[: skip_before // block] = float("-inf")
    if exclude:
        scores[torch.as_tensor(list(exclude), dtype=torch.long, device=scores.device)] = float("-inf")
    if near_start is not None and 0 < near_start < scores.numel():
        share = (scores.numel() - near_start) / scores.numel()
        # at least an eighth of the slots, so a fact that really is recent still fits
        near_top = min(top, max(max(1, top // 8), round(top * share)))
        near = pick_blocks(scores[near_start:], block=block, top=near_top)
        far = pick_blocks(scores[:near_start], block=block, top=top - len(near))
        return sorted(far + [near_start + index for index in near])
    count = min(top, int(torch.isfinite(scores).sum()))
    if count <= 0:
        return []
    return sorted(torch.topk(scores, count).indices.tolist())


def spans_from_blocks(
    blocks: Sequence[int], *, block: int, neighbors: int, length: int
) -> list[tuple[int, int]]:
    """Slot ranges covering ``blocks`` widened by ``neighbors``, merged where they touch."""

    spans: list[list[int]] = []
    for index in sorted(set(blocks)):
        lo = max(0, (index - neighbors) * block)
        hi = min(length, (index + 1 + neighbors) * block)
        if spans and lo <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], hi)
        else:
            spans.append([lo, hi])
    return [(lo, hi) for lo, hi in spans]


def pick_spans(
    mass: torch.Tensor,
    *,
    block: int,
    top: int,
    neighbors: int,
    length: int,
    skip_before: int = 0,
) -> list[tuple[int, int]]:
    """Slot ranges of the ``top`` heaviest blocks, widened by ``neighbors``.

    Blocks wholly below ``skip_before`` (the attention sink, which soaks up mass
    in many heads whatever the question) are not candidates.
    """

    chosen = pick_blocks(mass, block=block, top=top, skip_before=skip_before)
    return spans_from_blocks(chosen, block=block, neighbors=neighbors, length=length)


def _fused_locate(session, block: int) -> bool:
    """Whether :func:`locate` can take the one-pass Triton path (``DEEPINFER_FUSED_LOCATE=0`` turns it off)."""

    if os.environ.get("DEEPINFER_FUSED_LOCATE", "1") == "0" or block != latent_locate.BLOCK:
        return False
    if not latent_locate.available() or torch.device(session.device).type != "cuda":
        return False
    regions = extent_regions(session.extents, session.next_position, session.positions)
    return latent_locate.aligned(regions, session.length)


def _logical(extents, slot: int) -> int:
    for p0, p1, l0 in extents:
        if p0 <= slot < p1:
            return l0 + slot - p0
    return slot


@torch.inference_mode()
def locate(
    session,
    query_ids: Sequence[int],
    *,
    block: int = BLOCK,
    reduce: str = "sum",
    score_tokens: Sequence[int] | None = None,
) -> torch.Tensor:
    """Per-block attention mass of ``query_ids`` over the whole cached sequence.

    The query is run at the end of the sequence with full attention (every
    layer, every cached slot) and removed again; the session is unchanged.
    Layers are summed; within a layer ``reduce`` combines (query, head) rows.
    ``score_tokens`` (indices into ``query_ids``) limits which tokens' rows count;
    every token still runs, so the scored ones keep their context.
    """

    state = session.snapshot()
    length = session.length
    blocks = -(-length // block)
    total = torch.zeros(blocks, dtype=torch.float32, device=session.device)
    query_start = session.next_position

    def local(block_start: int, count: int) -> list[int] | None:
        if score_tokens is None:
            return None
        offset = block_start - query_start
        return [token - offset for token in score_tokens if 0 <= token - offset < count]
    if _fused_locate(session, block):
        # one pass over the cache per layer: the query's attention and its block mass together
        def attend(index, q_nope, q_pe_raw, latent, pe, block_start, regions):
            nonlocal total
            attended, mass = latent_locate.locate_attention(
                session.weights[index], session.cache, index, q_nope, q_pe_raw, latent, pe,
                block_start, regions, positions=session.positions, rope=session.rope,
                length=length, blocks=blocks, reduce=reduce, keep_tokens=local(block_start, q_nope.shape[0]),
            )
            total += mass
            return attended

        try:
            session._run(
                query_ids, session.next_position, context=None, progress=None,
                attend=attend, bounded=False,
            )
        finally:
            session.restore(state)
        return total

    def observe(index, q_nope, q_pe_raw, block_start, regions):
        nonlocal total
        cached = [(a, min(b, length), r) for a, b, r in regions if a < length]
        total += block_mass(
            session.weights[index],
            session.cache,
            index,
            q_nope,
            q_pe_raw,
            block_start,
            cached,
            positions=session.positions,
            rope=session.rope,
            block=block,
            blocks=blocks,
            reduce=reduce,
            keep_tokens=local(block_start, q_nope.shape[0]),
        )

    session._run(
        query_ids, session.next_position, context=None, progress=None,
        observe=observe, bounded=False,
    )
    session.restore(state)
    return total


def contrast(mass: torch.Tensor, baseline: torch.Tensor, mode: str = "difference") -> torch.Tensor:
    """What a query attends to beyond a neutral prompt.

    ``"difference"``: ``mass - baseline * (sum mass / sum baseline)``.  Mass that
    heads put on a block because of where it is (recency, sinks, the dual-chunk
    position pattern) shows up in both and cancels; mass that follows the
    question's words does not.  ``"ratio"``: ``log(mass + c) - log(baseline' + c)``
    with the baseline scaled to the same total and ``c`` its mean - blocks the
    neutral prompt already looks at a lot are discounted in proportion, not by a
    fixed amount (a longer query's sharpest rows rise more in the recent region
    than a short prompt's, which a difference does not cancel).
    """

    scaled = baseline * (mass.sum() / baseline.sum().clamp_min(1e-12))
    if mode == "ratio":
        smoothing = scaled.mean().clamp_min(1e-12)
        return torch.log(mass.clamp_min(0) + smoothing) - torch.log(scaled.clamp_min(0) + smoothing)
    return mass - scaled


def span_document(
    session,
    spans: Sequence[tuple[int, int]],
    *,
    header_ids: Sequence[int],
    separator_ids: Sequence[int],
    slots: list[int] | None = None,
) -> list[int]:
    """Header, then the tokens of ``spans`` in document order, separated.

    ``slots``, if given, receives the cache slot of every document token (-1 for
    the header and separators).
    """

    ordered = sorted(spans, key=lambda span: _logical(session.extents, span[0]))
    document: list[int] = list(header_ids)
    if slots is not None:
        slots.extend([-1] * len(header_ids))
    for number, (lo, hi) in enumerate(ordered):
        if number:
            document.extend(separator_ids)
            if slots is not None:
                slots.extend([-1] * len(separator_ids))
        document.extend(session.slot_tokens[lo:hi])
        if slots is not None:
            slots.extend(range(lo, hi))
    return document


class Refine:
    """Second, line-level locate inside the short re-read context.

    The re-read text holds every block that looked relevant - with many
    similar facts that is most of them, plus the code around them - and the
    model then picks the wrong one of several near-identical lines.  Inside
    the short context the model's attention is back in its trained range, so
    the query is located once more, per token (sharpest row minus a neutral
    prompt's), and only the ``keep`` best *lines* are kept for the answer, in
    their original order (``order="score"``: by score, best last).  Document
    order is what a chain of facts needs; with step-by-step answers it also
    kept 16 near-identical keys apart (score order had fixed one confusion when
    the model answered without reasoning).
    """

    def __init__(
        self,
        tokenizer,
        query_ids: Sequence[int],
        neutral_ids: Sequence[int],
        keep: int = 12,
        order: str = "document",
    ) -> None:
        self.tokenizer = tokenizer
        self.query_ids = list(query_ids)
        self.neutral_ids = list(neutral_ids)
        self.keep = int(keep)
        self.order = order

    def lines(self, document: Sequence[int], start: int) -> list[tuple[int, int]]:
        """Token ranges of the non-blank lines of ``document[start:]`` (byte-level BPE: ``Ċ`` is a newline)."""

        pieces = self.tokenizer.convert_ids_to_tokens(list(document))
        ranges = []
        begin = start
        for index in range(start, len(pieces)):
            if "Ċ" in pieces[index] or index == len(pieces) - 1:
                text = "".join(pieces[begin : index + 1]).replace("Ċ", "").replace("Ġ", "").strip()
                if text and text != "...":
                    ranges.append((begin, index + 1))
                begin = index + 1
        return ranges

    @torch.inference_mode()
    def __call__(
        self,
        rt,
        document: Sequence[int],
        header_length: int,
        query_ids: Sequence[int] | None = None,
        starts: list[int] | None = None,
    ) -> tuple[list[int], list[str]]:
        """``(compact document, kept lines)``; ``query_ids`` overrides the query for this call.

        ``starts``, if given, receives the document index where each kept line begins.
        """

        from deepinfer.moe.deepseek_longctx import DeepSeekLongContext

        query = self.query_ids if query_ids is None else list(query_ids)
        short = DeepSeekLongContext(
            rt, capacity=len(document) + max(len(query), len(self.neutral_ids)) + 8, kv_dtype="fp16"
        )
        short.prefill(document)
        scores = contrast(
            locate(short, query, block=1, reduce="max"),
            locate(short, self.neutral_ids, block=1, reduce="max"),
        )
        del short
        ranges = self.lines(document, header_length)
        if not ranges:
            return list(document), []
        best = [float(scores[lo:hi].max()) for lo, hi in ranges]
        chosen = sorted(range(len(ranges)), key=lambda i: -best[i])[: self.keep]
        chosen = sorted(chosen, key=lambda i: best[i]) if self.order == "score" else sorted(chosen)
        newline = self.tokenizer.encode("\n", add_special_tokens=False)
        compact = list(document[:header_length])
        kept = []
        for index in chosen:
            lo, hi = ranges[index]
            line = list(document[lo:hi])
            compact.extend(line)
            if "Ċ" not in self.tokenizer.convert_ids_to_tokens(line[-1:])[0]:
                compact.extend(newline)
            kept.append(self.tokenizer.decode(line).strip())
            if starts is not None:
                starts.append(lo)
        return compact, kept


@torch.inference_mode()
def reread(
    session,
    spans: Sequence[tuple[int, int]],
    prompt_ids: Sequence[int],
    *,
    header_ids: Sequence[int],
    separator_ids: Sequence[int],
    max_new_tokens: int = 32,
    stop_ids: Sequence[int] = (),
    refine: Refine | None = None,
    kept: list | None = None,
    refine_query: Sequence[int] | None = None,
    on_tokens=None,
    kept_positions: list | None = None,
) -> list[int]:
    """Greedy answer to ``prompt_ids`` in a fresh short context holding only ``spans``.

    With ``refine`` the context is cut down to its best lines first (their text
    is appended to ``kept`` if given, their logical positions in the long
    sequence to ``kept_positions``).  ``on_tokens`` streams the answer's ids.
    """

    from deepinfer.moe.deepseek_longctx import DeepSeekLongContext

    slots: list[int] = []
    document = span_document(session, spans, header_ids=header_ids, separator_ids=separator_ids, slots=slots)
    if refine is not None:
        starts: list[int] = []
        document, lines = refine(session.rt, document, len(header_ids), refine_query, starts=starts)
        if kept is not None:
            kept.extend(lines)
        if kept_positions is not None:
            kept_positions.extend(
                _logical(session.extents, slots[start]) if slots[start] >= 0 else -1 for start in starts
            )
    short = DeepSeekLongContext(
        session.rt,
        capacity=len(document) + len(prompt_ids) + max_new_tokens + 8,
        kv_dtype="fp16",
    )
    short.prefill(document)
    produced = short.generate(prompt_ids, max_new_tokens=max_new_tokens, stop_ids=stop_ids, on_tokens=on_tokens)
    del short
    return produced


@torch.inference_mode()
def neutral_baseline(session, neutral_ids: Sequence[int], *, reduce: str = "max") -> torch.Tensor:
    """Block mass of a prompt that asks nothing in particular, for :func:`contrast`.

    It depends on the cached sequence, not on the question, so compute it once
    per session.
    """

    return locate(session, neutral_ids, block=BLOCK, reduce=reduce)


@torch.inference_mode()
def score_blocks(
    session,
    query_ids: Sequence[int],
    *,
    reduce: str = "sum",
    baseline: torch.Tensor | None = None,
    sink: int = 64,
    contrast_mode: str = "difference",
    score_tokens: Sequence[int] | None = None,
) -> torch.Tensor:
    """Per-block relevance of ``query_ids``: located mass, contrasted with the baseline if given; sink at -inf."""

    scores = locate(session, query_ids, block=BLOCK, reduce=reduce, score_tokens=score_tokens)
    if baseline is not None:
        scores = contrast(scores, baseline, contrast_mode)
    scores[: sink // BLOCK] = float("-inf")
    return scores


@torch.inference_mode()
def locate_and_reread(
    session,
    question_ids: Sequence[int],
    *,
    header_ids: Sequence[int],
    separator_ids: Sequence[int],
    top: int = 8,
    neighbors: int = 1,
    sink: int = 64,
    max_new_tokens: int = 32,
    stop_ids: Sequence[int] = (),
    reduce: str = "sum",
    baseline: torch.Tensor | None = None,
    locate_ids: Sequence[int] | None = None,
    refine: Refine | None = None,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Answer ``question_ids`` about the session's sequence by locate + re-read.

    ``header_ids`` start the short context (BOS and the user-turn prefix) and
    ``separator_ids`` go between non-adjacent spans.  ``reduce`` / ``baseline``
    choose the block scoring (see :func:`block_mass`, :func:`contrast`);
    ``locate_ids`` (default: the question) is what is located.  The session is
    left as it was.  Returns the generated ids and the re-read spans.
    """

    scores = score_blocks(
        session, question_ids if locate_ids is None else locate_ids,
        reduce=reduce, baseline=baseline, sink=sink,
    )
    chosen = pick_blocks(scores, block=BLOCK, top=top)
    spans = spans_from_blocks(chosen, block=BLOCK, neighbors=neighbors, length=session.length)
    produced = reread(
        session, spans, question_ids,
        header_ids=header_ids, separator_ids=separator_ids,
        max_new_tokens=max_new_tokens, stop_ids=stop_ids, refine=refine,
    )
    return produced, spans


def _normalized(text: str) -> str:
    return " ".join(text.split())


_ITEMS = re.compile(r"\b(?:\d{3,}|[A-Z][A-Z0-9]{3,})\b")


def answer_items(text: str) -> list[str]:
    """Numbers and all-caps identifiers (codes, variable names) in the final answer."""

    return sorted(set(_ITEMS.findall(final_answer(text))))


def answer_key(text: str) -> str:
    """What an answer commits to, for telling whether two rounds agree: its numbers and
    all-caps identifiers (codes, variable names), or its words if it has none."""

    items = answer_items(text)
    return " ".join(items) if items else " ".join(re.findall(r"\w+", final_answer(text).lower()))


def final_answer(text: str) -> str:
    """The part after the last ``Answer:`` (the whole text if there is none)."""

    index = text.lower().rfind("answer:")
    return text[index + len("answer:") :].strip() if index >= 0 else text.strip()


@torch.inference_mode()
def feedback_locate_and_reread(
    session,
    tokenizer,
    question: str,
    *,
    answer_instruction: str = "",
    header_ids: Sequence[int],
    separator_ids: Sequence[int],
    baseline: torch.Tensor | None = None,
    reduce: str = "max",
    top: int = 32,
    follow_top: int = 8,
    neighbors: int = 1,
    sink: int = 64,
    rounds: int = 5,
    max_new_tokens: int = 96,
    stop_ids: Sequence[int] = (),
    refine: Refine | None = None,
    carry: bool = False,
    answer_tokens: int = 256,
    contrast_mode: str = "difference",
    near_window: int = 0,
    follow_final: int = 32,
    final_tokens: int = 64,
    final_with_question: bool = False,
    final_entity_rows: bool = False,
    on_event=None,
) -> dict:
    """Locate, re-read and answer; then locate again with the question *and the answer*.

    A question can depend on facts it does not name - the variable a chain
    passes through, the person who holds a key.  The model's answer names what
    it has found so far (with a quote-then-answer instruction it also copies the
    sentences it used), so the next locate, run with question + answer, can
    reach the facts that refer to those names.  Round 1 locates the question
    alone and reads the ``top`` blocks; each later round adds the ``follow_top``
    best blocks not read yet and answers again over everything read (with
    ``carry`` it is also shown its previous answer - off by default: the model
    tends to confirm it rather than extend it).  Stops
    when the final answer (after the last ``Answer:``) no longer changes, or
    after ``rounds``.  The session is unchanged.

    Each later round also adds the ``follow_final`` best blocks for the final
    answer alone ("Answer: GHTAY").  The whole answer is what reaches a fact named
    only in its reasoning (the key holder of a two-hop question), but its many
    rows also light up code that shares its wording: at 3M slots the next link of
    a variable chain ranked 154-241 under the whole answer, 76 with the question
    in front of the final answer and 5-26 under the final answer alone, over four
    hops (scripts/rank_followup.py) - hence 32 such blocks a round, left to the
    line-level refine to sort out.  Scoring only the answer's own tokens
    (``final_entity_rows``) did not rank better.

    ``on_event(dict)``, if given, follows the search as it happens: ``located``
    (a round's new blocks and the time it took), ``tokens`` (the round's answer
    ids as they are generated) and ``answer`` (the round's answer, the lines it
    read with their positions, and whether the search stops).
    """

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    length = session.length
    prompt = encode(f"\n\n{question} {answer_instruction}".rstrip() + "\n\nAssistant:")
    query = encode(f"\n\n{question}\n\nAssistant:")
    final_query: list[int] = []
    final_rows: list[int] | None = None
    chosen: list[int] = []
    answers: list[str] = []
    trace: list[dict] = []
    produced: list[int] = []
    spans: list[tuple[int, int]] = []
    for round_index in range(int(rounds)):
        began = time.perf_counter()
        scores = score_blocks(
            session, query, reduce=reduce, baseline=baseline, sink=sink, contrast_mode=contrast_mode
        )
        near_start = max(0, length - near_window) // BLOCK if near_window else None
        new = pick_blocks(
            scores, block=BLOCK, top=top if round_index == 0 else follow_top, exclude=chosen,
            near_start=near_start,
        )
        final_new: list[int] = []
        if round_index > 0 and follow_final and final_query:
            final_scores = score_blocks(
                session, final_query, reduce=reduce, baseline=baseline, sink=sink, contrast_mode=contrast_mode,
                score_tokens=final_rows,
            )
            final_new = pick_blocks(
                final_scores, block=BLOCK, top=follow_final, exclude=chosen + new, near_start=near_start
            )
            new = new + final_new
        located = time.perf_counter()
        if on_event is not None:
            on_event({
                "event": "located", "round": round_index + 1, "blocks": sorted(new), "block": BLOCK,
                "seconds": round(located - began, 2),
            })
        chosen.extend(new)
        spans = spans_from_blocks(chosen, block=BLOCK, neighbors=neighbors, length=length)
        kept: list[str] = []
        asked = prompt
        if answers and carry:
            # build on the last answer instead of re-deriving everything: a chain then
            # needs one new hop per round, not the whole closure at once
            asked = encode(
                f"\n\n{question}\nAn earlier answer, from less of the text, was: {answers[-1]}\n"
                f"Check it against the text above and give the complete answer. {answer_instruction}".rstrip()
                + "\n\nAssistant:"
            )
        kept_positions: list[int] = []
        produced = reread(
            session, spans, asked,
            header_ids=header_ids, separator_ids=separator_ids,
            max_new_tokens=max_new_tokens, stop_ids=stop_ids, refine=refine, kept=kept,
            refine_query=None if round_index == 0 else query,
            on_tokens=None if on_event is None else (
                lambda ids, number=round_index + 1: on_event({"event": "tokens", "round": number, "ids": ids})
            ),
            kept_positions=kept_positions,
        )
        answer = tokenizer.decode(produced, skip_special_tokens=True).strip()
        trace.append({
            "round": round_index + 1, "new_blocks": new, "final_blocks": final_new, "lines": kept,
            "line_positions": kept_positions, "answer": answer,
            "query_tokens": len(query), "answer_tokens": len(produced),
            "seconds": {"locate": round(located - began, 2), "reread": round(time.perf_counter() - located, 2)},
        })
        final = answer_key(answer)
        # a round that neither wrote "Answer: ..." nor named a code / identifier (it ran
        # out, or talked about something else) has not committed and cannot end the search
        committed = bool(answer_items(answer)) or (
            "answer:" in answer.lower() and bool(final_answer(answer).strip())
        )
        stable = committed and bool(answers) and final == answers[-1]
        answers.append(final)
        if on_event is not None:
            on_event({
                "event": "answer", "round": round_index + 1, "text": answer, "lines": kept,
                "line_positions": kept_positions, "stop": stable or round_index + 1 == int(rounds),
            })
        if stable:
            break
        # the query is written into the cache while it is located: keep it within the room left
        head, tail = encode(f"\n\n{question}\n"), encode("\n\nAssistant:")
        room = session.cache.capacity - session.length - len(head) - len(tail) - 8
        # the end of the answer holds its conclusion; cap what is located with
        query = head + encode(answer)[-min(answer_tokens, max(0, room)) :] + tail
        conclusion = final_answer(answer) if "answer:" in answer.lower() else ", ".join(answer_items(answer))
        named = encode(f" {conclusion}")[: min(final_tokens, max(0, room))] if conclusion else []
        lead = (head if final_with_question else encode("\n\n")) + encode("Answer:")
        final_query = lead + named + tail if named else []
        # optionally only the named things' rows score (did not rank the next link better at 3M)
        final_rows = list(range(len(lead), len(lead) + len(named))) if final_entity_rows else None
    return {
        "answer_ids": produced,
        "spans": spans,
        "blocks": chosen,
        "rounds": trace,
        "refine_query": query,
        "reread_tokens": sum(hi - lo for lo, hi in spans),
    }
