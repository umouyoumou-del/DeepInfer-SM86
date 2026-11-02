"""Locate + re-read: block mass is exact attention mass, span picking is what it says."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from deepinfer.moe.latent_mla import (
    DualChunkPositions,
    LatentKVCache,
    LatentRope,
    MLAWeights,
)
from deepinfer.moe.latent_reread import (
    Refine,
    answer_key,
    block_mass,
    contrast,
    final_answer,
    pick_blocks,
    pick_spans,
    spans_from_blocks,
)


class _Attention:
    num_heads = 4
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 128
    softmax_scale = 192**-0.5

    def __init__(self) -> None:
        self.kv_b_proj = torch.nn.Linear(512, 4 * 256, bias=False)
        torch.nn.init.normal_(self.kv_b_proj.weight, std=0.05)


@pytest.mark.parametrize(
    "positions",
    [DualChunkPositions.plain(), DualChunkPositions(trained=128, chunk=96)],
    ids=["plain", "dca"],
)
def test_block_mass_is_the_softmax_mass_summed_per_block(positions) -> None:
    torch.manual_seed(0)
    attention = _Attention()
    weights = MLAWeights.from_attention(attention)
    rope = LatentRope(1.0 / (10000 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64)))
    n, block = 300, 16
    latent = torch.randn(n, 512)
    pe_raw = torch.randn(n, 64)
    cache = LatentKVCache(num_layers=1, capacity=n, kv_dtype="fp16", device="cpu")
    cache.write(0, 0, latent.half(), rope.apply(pe_raw, positions.key_positions(torch.arange(n))))
    q_nope = torch.randn(3, 4, 128)
    q_pe_raw = torch.randn(3, 4, 64)
    start = n  # three questions tokens right after the cache
    regions = positions.history_regions(start)
    mass = block_mass(
        weights, cache, 0, q_nope, q_pe_raw, start, regions,
        positions=positions, rope=rope, block=block, blocks=-(-n // block), tile=64,
    )
    # naive: every (query, head) softmax over all cached slots, summed per block
    stored_latent, stored_pe = cache.tile(0, 0, n)
    expected = torch.zeros(-(-n // block))
    for row in range(3):
        position = torch.tensor([start + row])
        for head in range(4):
            q_abs = q_nope[row, head].float() @ weights.w_uk[head].float()
            scores = stored_latent.float() @ q_abs
            for k0, k1, relation in regions:
                q_pe = rope.apply(
                    q_pe_raw[row, head].view(1, 64), positions.query_positions(position, relation)
                )[0]
                scores[k0:k1] += stored_pe[k0:k1].float() @ q_pe
            p = torch.softmax(scores * weights.softmax_scale, dim=0)
            expected.index_add_(0, torch.arange(n) // block, p)
    assert torch.allclose(mass, expected, atol=2e-2, rtol=2e-2)
    assert float(mass.sum()) == pytest.approx(12.0, rel=1e-3)  # 3 queries x 4 heads

    # reduce="max": per block, the largest single (query, head) row instead of the sum
    sharpest = block_mass(
        weights, cache, 0, q_nope, q_pe_raw, start, regions,
        positions=positions, rope=rope, block=block, blocks=-(-n // block), tile=64, reduce="max",
    )
    rows = []
    for row in range(3):
        position = torch.tensor([start + row])
        for head in range(4):
            q_abs = q_nope[row, head].float() @ weights.w_uk[head].float()
            scores = stored_latent.float() @ q_abs
            for k0, k1, relation in regions:
                q_pe = rope.apply(
                    q_pe_raw[row, head].view(1, 64), positions.query_positions(position, relation)
                )[0]
                scores[k0:k1] += stored_pe[k0:k1].float() @ q_pe
            p = torch.softmax(scores * weights.softmax_scale, dim=0)
            rows.append(torch.zeros(-(-n // block)).index_add_(0, torch.arange(n) // block, p))
    assert torch.allclose(sharpest, torch.stack(rows).max(dim=0).values, atol=2e-2, rtol=2e-2)

    # block = 1 (the line-level refine): one slot per block, taken without a rows x slots buffer
    per_slot = block_mass(
        weights, cache, 0, q_nope, q_pe_raw, start, regions,
        positions=positions, rope=rope, block=1, blocks=n, tile=64, reduce="max",
    )
    slot_rows = []
    for row in range(3):
        position = torch.tensor([start + row])
        for head in range(4):
            q_abs = q_nope[row, head].float() @ weights.w_uk[head].float()
            scores = stored_latent.float() @ q_abs
            for k0, k1, relation in regions:
                q_pe = rope.apply(
                    q_pe_raw[row, head].view(1, 64), positions.query_positions(position, relation)
                )[0]
                scores[k0:k1] += stored_pe[k0:k1].float() @ q_pe
            slot_rows.append(torch.softmax(scores * weights.softmax_scale, dim=0))
    assert torch.allclose(per_slot, torch.stack(slot_rows).max(dim=0).values, atol=2e-3, rtol=2e-2)


def test_pick_spans_skips_the_sink_and_merges_neighbours() -> None:
    mass = torch.zeros(20)
    mass[0] = 100.0  # the sink always looks heavy
    mass[5] = 3.0
    mass[6] = 2.0
    mass[15] = 1.0
    spans = pick_spans(mass, block=10, top=3, neighbors=1, length=195, skip_before=10)
    # blocks 5 and 6 widen to 4..7 and merge; block 15 widens to 14..16; length clips
    assert spans == [(40, 80), (140, 170)]
    assert pick_spans(mass, block=10, top=1, neighbors=0, length=200, skip_before=10) == [(50, 60)]


def test_pick_blocks_excludes_what_was_read_and_spans_merge_across_rounds() -> None:
    mass = torch.tensor([9.0, 1.0, 8.0, 7.0, 0.5, 6.0, 0.1, 5.0])
    assert pick_blocks(mass, block=10, top=2, skip_before=10) == [2, 3]
    # a later round: the blocks already read are not candidates again
    assert pick_blocks(mass, block=10, top=2, skip_before=10, exclude=[2, 3]) == [5, 7]
    assert pick_blocks(mass, block=10, top=10, skip_before=80) == []
    # blocks chosen in different rounds, widened by one neighbour, merge where they touch
    assert spans_from_blocks([7, 2, 3, 5], block=10, neighbors=1, length=75) == [(10, 75)]
    assert spans_from_blocks([1, 6], block=10, neighbors=0, length=80) == [(10, 20), (60, 70)]


def test_contrast_removes_what_a_neutral_prompt_attends_to_anyway() -> None:
    # blocks 0 and 3 are heavy because of where they are; block 2 because of the question
    baseline = torch.tensor([5.0, 1.0, 1.0, 5.0])
    question = torch.tensor([10.0, 2.0, 6.0, 10.0])
    scores = contrast(question, baseline)
    assert int(scores.argmax()) == 2
    assert float(scores.sum()) == pytest.approx(0.0, abs=1e-5)


def test_final_answer_and_answer_key() -> None:
    reasoning = "The special magic number for crimson otter is: 7646461.\nSo...\nAnswer: 8812311"
    assert final_answer(reasoning) == "8812311"
    assert final_answer("just 8812311") == "just 8812311"
    # rounds agree when they commit to the same codes / identifiers, however worded
    assert answer_key("Answer:\nVAR GHTAY, VAR OKOSG") == answer_key("answer: GHTAY, OKOSG") == "GHTAY OKOSG"
    assert answer_key("The special magic number for amber lynx is: 4186027.") == answer_key("Answer: 4186027")
    assert answer_key("Answer: Kaila Brenn") == "kaila brenn"


class _Pieces:
    """Byte-level BPE pieces for a few fixed token ids (Ċ is a newline, Ġ a space)."""

    def convert_ids_to_tokens(self, ids):
        pieces = {1: "Hello", 2: "Ġworld", 3: "Ċ", 4: "...", 5: "ĊĊ", 6: "VAR", 7: "ĠX", 8: "Ġ=", 9: "Ġ5"}
        return [pieces[i] for i in ids]


def test_refine_splits_the_reread_text_into_lines() -> None:
    refine = Refine(_Pieces(), [1], [2])
    # header (one token), "Hello world\n", the span separator "...\n\n", "VAR X = 5"
    document = [1, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert refine.lines(document, start=1) == [(1, 4), (6, 10)]
