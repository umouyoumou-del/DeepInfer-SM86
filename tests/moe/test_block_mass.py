"""Block-level attention mass must match a dense reference and be usable for ranking."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from deepinfer.moe.block_mass import (  # noqa: E402
    block_attention_mass,
    coverage,
    hottest_blocks,
)

BLOCK = 4
BLOCKS = 8
HEADS = 2
DIM = 8
SCALE = 1.0 / (DIM**0.5)


def _fixture(length: int = 12, seed: int = 0):
    torch.manual_seed(seed)
    key_pool = torch.randn(BLOCKS, 1, BLOCK, DIM)
    table = torch.arange(BLOCKS, dtype=torch.long).unsqueeze(0)
    lengths = torch.tensor([length])
    query = torch.randn(1, HEADS, 1, DIM)
    return query, key_pool, table, lengths


def _dense_mass(query, key_pool, table, length):
    """Independent formulation: build the key matrix explicitly, one block at a time."""

    positions = torch.arange(length)
    keys = []
    owner = []
    for position in positions:
        block = int(table[0, int(position) // BLOCK])
        offset = int(position) % BLOCK
        keys.append(key_pool[block, 0, offset])
        owner.append(block)
    keys = torch.stack(keys)
    scores = torch.einsum("hqd,ld->hql", query[0], keys) * SCALE
    weights = torch.softmax(scores, dim=-1)
    mass = torch.zeros(BLOCKS)
    for index, block in enumerate(owner):
        # weights is [heads, q_len, positions]; one query position here
        mass[block] += float(weights[:, 0, index].sum())
    return mass


def test_matches_dense_reference() -> None:
    query, key_pool, table, lengths = _fixture()
    got = block_attention_mass(
        query, key_pool, table, lengths, block_size=BLOCK, scale=SCALE
    )
    expected = _dense_mass(query, key_pool, table, 12)
    assert torch.allclose(got[0], expected, atol=1e-5), (got[0] - expected).abs().max()


def test_mass_sums_to_one_per_position_and_head() -> None:
    query, key_pool, table, lengths = _fixture()
    mass = block_attention_mass(
        query, key_pool, table, lengths, block_size=BLOCK, scale=SCALE
    )
    # one query position, two heads -> total 2
    assert float(mass.sum()) == pytest.approx(HEADS, rel=1e-4)


def test_multi_token_is_causal() -> None:
    torch.manual_seed(3)
    key_pool = torch.randn(BLOCKS, 1, BLOCK, DIM)
    table = torch.arange(BLOCKS, dtype=torch.long).unsqueeze(0)
    lengths = torch.tensor([8])
    query = torch.randn(1, HEADS, 4, DIM)
    mass = block_attention_mass(
        query, key_pool, table, lengths, block_size=BLOCK, scale=SCALE
    )
    assert torch.isfinite(mass).all()
    # 4 query positions x 2 heads = 8 units of mass
    assert float(mass.sum()) == pytest.approx(HEADS * 4, rel=1e-4)


def test_ranking_and_coverage() -> None:
    mass = torch.tensor([[0.0, 0.5, 0.3, 0.0, 0.2, 0.0, 0.0, 0.0]])
    top = hottest_blocks(mass, 2)
    assert top.tolist() == [1, 2]
    assert coverage(mass, 2) == pytest.approx(0.8)
    assert coverage(mass, 0) == 0.0
    assert hottest_blocks(mass, 0).numel() == 0


def test_zero_length_is_all_zero() -> None:
    query, key_pool, table, _ = _fixture()
    mass = block_attention_mass(
        query, key_pool, table, torch.tensor([0]), block_size=BLOCK, scale=SCALE
    )
    assert float(mass.abs().sum()) == 0.0
