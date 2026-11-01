"""How much attention each KV block actually receives.

Both long-context features that need to know which history matters read this:

* H2O-style eviction ranks blocks by the mass they carry, so the blocks that get
  swapped out are the ones the model is not looking at;
* retrieval ranks blocks the same way, so the ones fetched back are the ones a
  query would have attended to anyway.

Per-token scores are the obvious thing to export, but at a 1M context that is a
million floats per query head.  Summing into blocks gives ``max_blocks``
numbers - 65,536 at 1M with block_size 16 - and that is all either consumer
needs.

This is the reference implementation in PyTorch.  It recomputes the scores
rather than harvesting them from the fused kernel, so it costs roughly one
attention pass.  That is acceptable for the prefill-time calls these consumers
make; harvesting them from the kernel is the optimisation.
"""

from __future__ import annotations

import torch


def block_attention_mass(
    query: torch.Tensor,
    key_pool: torch.Tensor,
    block_table: torch.Tensor,
    lengths: torch.Tensor,
    *,
    block_size: int,
    scale: float,
) -> torch.Tensor:
    """Return ``[batch, max_blocks]`` attention mass per physical block.

    ``query`` is ``[batch, heads, q_len, head_dim]``, ``key_pool`` is
    ``[num_blocks, kv_heads, block_size, head_dim]`` and ``lengths`` is the
    number of valid positions per row.  Mass sums to 1 per (batch, query
    position, head) over the positions that were visible.
    """

    if query.dim() != 4:
        raise ValueError(f"query must be [B,H,Q,D], got {tuple(query.shape)}")
    batch, heads, q_len, head_dim = query.shape
    if key_pool.dim() != 4:
        raise ValueError(f"key_pool must be [blocks, kv, block, D], got {tuple(key_pool.shape)}")
    if block_table.shape[0] != batch or lengths.numel() != batch:
        raise ValueError("block_table and lengths must agree with batch")
    if key_pool.shape[-1] != head_dim:
        raise ValueError("query and key head_dim disagree")

    kv_heads = key_pool.shape[1]
    if heads % kv_heads:
        raise ValueError("heads must be a multiple of kv_heads")
    group = heads // kv_heads
    max_blocks = int(block_table.shape[1])
    mass = torch.zeros(
        batch, max_blocks, dtype=torch.float32, device=query.device
    )

    for row in range(batch):
        length = int(lengths[row])
        if length <= 0:
            continue
        positions = torch.arange(length, device=query.device)
        block_index = positions // block_size
        offset = positions % block_size
        physical = block_table[row, block_index]
        valid = physical >= 0
        if not bool(valid.any()):
            continue
        keys = key_pool[physical.clamp(min=0), 0, offset, :].float()  # [len, D]
        scores = torch.einsum("hqd,ld->hql", query[row].float(), keys) * scale
        if q_len > 1:
            # causal: position i sees 0..i
            causal = torch.ones(q_len, length, device=query.device, dtype=torch.bool)
            causal = causal.tril(diagonal=length - q_len)
            scores = scores.masked_fill(~causal.unsqueeze(0), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        # sum over heads and query positions, then scatter into physical blocks
        per_position = weights.sum(dim=(0, 1))  # [len]
        mass[row].index_add_(0, physical[valid], per_position[valid])
    return mass


def hottest_blocks(mass: torch.Tensor, count: int) -> torch.Tensor:
    """Indices of the ``count`` blocks carrying the most mass, largest first."""

    if count <= 0:
        return torch.empty(0, dtype=torch.long, device=mass.device)
    flat = mass.reshape(-1)
    order = torch.argsort(flat, descending=True)
    return order[: min(count, flat.numel())]


def coverage(mass: torch.Tensor, keep: int) -> float:
    """Fraction of total mass retained by keeping the ``keep`` hottest blocks."""

    total = float(mass.sum())
    if total <= 0:
        return 0.0
    kept = mass.reshape(-1)[hottest_blocks(mass, keep)]
    return float(kept.sum()) / total


class BlockMassHeat:
    """``heat_fn`` adapter: physical block id -> attention mass.

    ``PrefixCache`` asks ``heat_fn(block_id)`` for each candidate and evicts the
    coldest, which is exactly H2O's rule once the number is attention mass
    instead of recency: the block the model is not looking at is the one that
    can leave.  Mass is naturally zero for blocks the last measurement did not
    touch, and unknown ids also score zero, so a stale or partial map degrades
    to "evict the ones we know nothing about" rather than raising.
    """

    def __init__(self, mass, *, default: float = 0.0) -> None:
        self._mass = mass
        self._default = float(default)
        self.calls = 0

    def update(self, mass) -> None:
        self._mass = mass

    def __call__(self, block_id: int) -> float:
        self.calls += 1
        try:
            index = int(block_id)
        except Exception:  # noqa: BLE001
            return self._default
        flat = self._mass.reshape(-1) if hasattr(self._mass, "reshape") else self._mass
        if index < 0 or index >= len(flat):
            return self._default
        return float(flat[index])


def from_layer_mass(mass_row) -> BlockMassHeat:
    """Adapter for one row of ``block_attention_mass`` output."""

    return BlockMassHeat(mass_row)
