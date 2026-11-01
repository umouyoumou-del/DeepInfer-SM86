"""Block retrieval over the latent cache: attend to the blocks that can matter.

Full attention reads every cached token for every generated token.  Here each
block of ``block`` cached slots keeps, per layer, the per-dimension minimum and
maximum of its 576-wide (latent, rotated k_pe) rows.  For a query those bounds
give an upper bound on any score inside the block (Quest, Tang et al. 2024):

    ub(block) = sum_d max(q_d * lo_d, q_d * hi_d) = q+ . hi + q- . lo

computed for all blocks with two small GEMMs on the absorbed query, so finding
the candidates costs a read of the bounds (6% of the cache) rather than of the
cache.  The query then attends exactly - same fused kernel - to the first
``sink`` slots, the ``top_blocks`` best blocks and the most recent ``window``
slots.  Mean-pooled block vectors would rank a block by its average token and
lose a one-sentence fact; an upper bound cannot under-rate it.

With ``reposition`` the retrieved blocks are also *moved*: in their original
order, right in front of the window, so the query sees sink + retrieved
passages + recent text at short distances, the way it would read a short
document.  Only the 64 rotary dims carry position in MLA, so this is a
different query rotation per retrieved run - nothing cached changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from deepinfer.moe.latent_mla import (
    LATENT_DIM,
    ROPE_DIM,
    DualChunkPositions,
    LatentKVCache,
    LatentRope,
    MLAWeights,
    merge_partials,
)


@dataclass(frozen=True)
class RetrievalConfig:
    block: int = 64
    top_blocks: int = 64
    window: int = 4096
    sink: int = 64
    reposition: bool = False


class BlockBounds:
    """Per-layer ``[blocks, 576]`` minimum and maximum of the cached rows."""

    def __init__(
        self, *, num_layers: int, capacity: int, block: int, device: torch.device | str
    ) -> None:
        self.block = int(block)
        count = -(-int(capacity) // self.block)
        width = LATENT_DIM + ROPE_DIM
        self.lo = torch.empty((num_layers, count, width), dtype=torch.float16, device=device)
        self.hi = torch.empty_like(self.lo)
        self.ready = 0  # blocks below this index hold bounds of full blocks

    def update(self, cache: LatentKVCache, layer: int, stop: int) -> None:
        """Recompute the bounds of every block that overlaps ``[ready*block, stop)``."""

        first = self.ready * self.block
        if stop <= first:
            return
        for start in range(first, stop, 65536):
            end = min(start + 65536, stop)
            latent, pe = cache.tile(layer, start, end)
            rows = torch.cat((latent, pe), dim=-1)
            full = (end - start) // self.block * self.block
            b0 = start // self.block
            if full:
                view = rows[:full].view(-1, self.block, rows.shape[-1])
                self.lo[layer, b0 : b0 + view.shape[0]] = view.amin(dim=1)
                self.hi[layer, b0 : b0 + view.shape[0]] = view.amax(dim=1)
            if end - start > full:  # a partial last block
                tail = rows[full:]
                self.lo[layer, b0 + full // self.block] = tail.amin(dim=0)
                self.hi[layer, b0 + full // self.block] = tail.amax(dim=0)

    def commit(self, stop: int) -> None:
        """Mark the blocks wholly below ``stop`` as final."""

        self.ready = max(self.ready, int(stop) // self.block)


def _always_kept(
    extents: list[tuple[int, int, int]],
    count: int,
    block: int,
    query_position: int,
    config: RetrievalConfig,
    device,
) -> torch.Tensor:
    """Blocks holding a sink or window slot, judged by *logical* position.

    Slots and positions differ once text has been inserted: an inserted fact
    sits in the newest slots but deep in the sequence, and must compete like
    any other block rather than ride along in the window.
    """

    keep = torch.zeros(count, dtype=torch.bool, device=device)
    window_start = query_position - config.window + 1
    for p0, p1, l0 in extents:
        b0, b1 = p0 // block, -(-p1 // block)
        blocks = torch.arange(b0, b1, device=device)
        first = torch.clamp(blocks * block, min=p0) - p0 + l0
        last = torch.clamp((blocks + 1) * block, max=p1) - 1 - p0 + l0
        keep[b0:b1] |= (first < config.sink) | (last >= window_start)
    return keep


def select_segments(
    bounds: BlockBounds,
    layer: int,
    q_abs: torch.Tensor,
    q_pe_by_relation: dict[int, torch.Tensor],
    regions: list[tuple[int, int, int]],
    config: RetrievalConfig,
    *,
    extents: list[tuple[int, int, int]] | None = None,
    query_position: int | None = None,
    with_kind: bool = False,
) -> list[tuple]:
    """``(slot start, slot end, relation)`` for sink + best blocks + window.

    ``regions`` are the query's full key regions (physical, relation); the
    result is their intersection with the chosen slots.  ``extents`` map slots
    to logical positions for the sink/window test (default: slot == position).
    ``with_kind`` adds a fourth field: True for blocks chosen by score.
    """

    total = max(end for _, end, _ in regions)
    block = bounds.block
    count = -(-total // block)
    if extents is None:
        extents = [(0, total, 0)]
    if query_position is None:
        query_position = total - 1
    keep = _always_kept(extents, count, block, query_position, config, q_abs.device)
    # upper bounds per head and block, rotary part with each block's relation
    lo = bounds.lo[layer, :count].float()
    hi = bounds.hi[layer, :count].float()
    q = q_abs.float()
    score = q.clamp_min(0) @ hi[:, :LATENT_DIM].t() + q.clamp_max(0) @ lo[:, :LATENT_DIM].t()
    rope_score = torch.full_like(score, float("-inf"))
    for start, end, relation in regions:
        b0, b1 = start // block, -(-end // block)
        qp = q_pe_by_relation[relation].float()
        part = qp.clamp_min(0) @ hi[b0:b1, LATENT_DIM:].t() + qp.clamp_max(0) @ lo[b0:b1, LATENT_DIM:].t()
        rope_score[:, b0:b1] = torch.maximum(rope_score[:, b0:b1], part)
    best = (score + rope_score).amax(dim=0)  # a block matters if any head wants it
    best[keep] = float("-inf")
    picks = min(config.top_blocks, int((~keep).sum()))
    picked = torch.zeros_like(keep)
    if picks:
        picked[torch.topk(best, picks).indices] = True
    chosen = torch.nonzero(keep | picked).flatten().tolist()
    picked_set = set(torch.nonzero(picked).flatten().tolist())
    # merge runs of chosen blocks of the same kind, then clip to each region
    runs = []
    for index in chosen:
        kind = index in picked_set
        if runs and runs[-1][1] == index and runs[-1][2] == kind:
            runs[-1][1] = index + 1
        else:
            runs.append([index, index + 1, kind])
    segments = []
    for b0, b1, kind in runs:
        lo_slot, hi_slot = b0 * block, min(b1 * block, total)
        for start, end, relation in regions:
            a, b = max(lo_slot, start), min(hi_slot, end)
            if a < b:
                segments.append((a, b, relation, kind) if with_kind else (a, b, relation))
    return segments


def _logical(extents: list[tuple[int, int, int]], slot: int) -> int:
    for p0, p1, l0 in extents:
        if p0 <= slot < p1:
            return l0 + slot - p0
    raise ValueError(f"slot {slot} is in no extent")


def sparse_decode_attention(
    weights: MLAWeights,
    cache: LatentKVCache,
    bounds: BlockBounds,
    layer: int,
    q_nope: torch.Tensor,
    q_pe_raw: torch.Tensor,
    position: int,
    regions: list[tuple[int, int, int]],
    *,
    positions: DualChunkPositions,
    rope: LatentRope,
    config: RetrievalConfig,
    extents: list[tuple[int, int, int]] | None = None,
) -> torch.Tensor:
    """Absorbed attention of one query over the selected slots only."""

    from deepinfer.moe import latent_mla_triton as fused

    device = q_nope.device
    q_abs = weights.absorb_query(q_nope)
    here = torch.tensor([position], device=device)
    relations = sorted({relation for _, _, relation in regions})
    q_pe_by_relation = {
        relation: rope.apply(q_pe_raw.unsqueeze(0), positions.query_positions(here, relation))[0]
        for relation in relations
    }
    if extents is None:
        extents = [(0, max(end for _, end, _ in regions), 0)]
    chosen = select_segments(
        bounds, layer, q_abs, q_pe_by_relation, regions, config,
        extents=extents, query_position=position, with_kind=True,
    )
    index_of = {relation: i for i, relation in enumerate(relations)}
    rotations = [q_pe_by_relation[r] for r in relations]
    segments = []  # (slot start, slot end, index into rotations)
    retrieved = []
    for start, end, relation, picked in chosen:
        if config.reposition and picked:
            retrieved.append((start, end))
        else:
            segments.append((start, end, index_of[relation]))
    if retrieved:
        # lay the retrieved runs out in order, ending where the window starts
        retrieved.sort(key=lambda run: _logical(extents, run[0]))
        cursor = position - config.window + 1 - sum(end - start for start, end in retrieved)
        for start, end in retrieved:
            first = _logical(extents, start)
            key_first = int(positions.key_positions(torch.tensor([first]))[0])
            # distance of the run's first token becomes position - cursor
            query = torch.tensor([key_first + position - cursor], device=device)
            rotations.append(rope.apply(q_pe_raw.unsqueeze(0), query)[0])
            segments.append((start, end, len(rotations) - 1))
            cursor += end - start
    q_pe = torch.stack(rotations).to(torch.float16)
    gpu, host = [], []
    for start, end, rotation in segments:
        for where, lo, hi, offset in cache._segments(start, end):
            (gpu if where == "gpu" else host).append((offset, offset + hi - lo, rotation))
    acc = lse = None
    if gpu:
        out, out_lse = fused.decode_partial(
            q_abs, q_pe, cache.gpu_latent[layer],
            None if cache.gpu_scale is None else cache.gpu_scale[layer],
            cache.gpu_pe[layer], gpu, sm_scale=weights.softmax_scale,
        )
        acc, lse = merge_partials(acc, lse, out, out_lse)
    for s0, s1, rotation in host:  # selected host rows only - a few blocks
        latent = cache.host_latent[layer, s0:s1].to(device, non_blocking=True)
        scale = None if cache.host_scale is None else cache.host_scale[layer, s0:s1].to(device)
        pe = cache.host_pe[layer, s0:s1].to(device, non_blocking=True)
        out, out_lse = fused.decode_partial(
            q_abs, q_pe, latent, scale, pe, [(0, s1 - s0, rotation)],
            sm_scale=weights.softmax_scale,
        )
        acc, lse = merge_partials(acc, lse, out, out_lse)
    return weights.project_output(acc)
