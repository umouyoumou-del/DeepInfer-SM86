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
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from deepinfer.moe.latent_mla import (
    DualChunkPositions,
    LatentKVCache,
    LatentRope,
    MLAWeights,
)


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
) -> torch.Tensor:
    """``[blocks]`` attention mass of ``n`` queries over the slots in ``regions``.

    Queries are at logical ``query_start ..``; ``q_nope [n,H,128]``,
    ``q_pe_raw [n,H,64]``.  Each (query, head) distribution is normalized over
    the region slots, then everything is summed per block of ``block`` slots.
    """

    n, heads, _ = q_nope.shape
    device = q_nope.device
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
    mass = torch.zeros(blocks, dtype=torch.float32, device=device)
    for lo, hi, scores in tiles():
        per_slot = torch.exp(scores - lse.unsqueeze(-1)).sum(dim=0)
        ids = torch.arange(lo, hi, device=device) // block
        mass.index_add_(0, ids, per_slot)
    return mass


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

    scores = mass.clone()
    scores[: skip_before // block] = float("-inf")
    count = min(top, int(torch.isfinite(scores).sum()))
    chosen = sorted(torch.topk(scores, count).indices.tolist())
    spans: list[list[int]] = []
    for index in chosen:
        lo = max(0, (index - neighbors) * block)
        hi = min(length, (index + 1 + neighbors) * block)
        if spans and lo <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], hi)
        else:
            spans.append([lo, hi])
    return [(lo, hi) for lo, hi in spans]


def _logical(extents, slot: int) -> int:
    for p0, p1, l0 in extents:
        if p0 <= slot < p1:
            return l0 + slot - p0
    return slot


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
) -> tuple[list[int], list[tuple[int, int]]]:
    """Answer ``question_ids`` about the session's sequence by locate + re-read.

    ``header_ids`` start the short context (BOS and the user-turn prefix) and
    ``separator_ids`` go between non-adjacent spans.  The session is left as
    it was.  Returns the generated ids and the slot spans that were re-read.
    """

    from deepinfer.moe.deepseek_longctx import DeepSeekLongContext

    state = session.snapshot()
    length = session.length
    block = 64
    blocks = -(-length // block)
    total = torch.zeros(blocks, dtype=torch.float32, device=session.device)

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
        )

    session._run(
        question_ids, session.next_position, context=None, progress=None,
        observe=observe, bounded=False,
    )
    session.restore(state)
    spans = pick_spans(total, block=block, top=top, neighbors=neighbors, length=length, skip_before=sink)
    spans.sort(key=lambda span: _logical(session.extents, span[0]))  # document order
    document: list[int] = list(header_ids)
    for number, (lo, hi) in enumerate(spans):
        if number:
            document.extend(separator_ids)
        document.extend(session.slot_tokens[lo:hi])
    short = DeepSeekLongContext(
        session.rt,
        capacity=len(document) + len(question_ids) + max_new_tokens + 8,
        kv_dtype="fp16",
    )
    short.prefill(document)
    produced = short.generate(question_ids, max_new_tokens=max_new_tokens, stop_ids=stop_ids)
    del short
    return produced, spans
