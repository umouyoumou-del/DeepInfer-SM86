"""Split-K paged MLA decode against a direct definition.

For every query: take the keys it may see (positions < min(valid, position + 1);
with a window, only the first ``sink`` and the last ``window`` of those), read
them through the block table, dequantize, softmax(q . k * scale) and average V.
Pools are fp16, INT8 with per-token scales, and FP8 E4M3.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cupy")

from deepinfer.moe.paged_mla_split import (
    paged_mla_decode_split,
    paged_mla_split_supported,
    split_plan,
)

CUDA = torch.cuda.is_available()
BLOCK = 16
HEADS = 4


def _pools(kind: str, blocks: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    shape = (blocks, HEADS, BLOCK, 192)
    if kind == "fp16":
        k = torch.randn(shape, generator=g, device="cuda").half()
        v = torch.randn(shape, generator=g, device="cuda").half()
        return k, v, None, None, k.float(), v.float()
    if kind == "int8":
        k = torch.randint(
            -127, 128, shape, generator=g, device="cuda", dtype=torch.int8
        )
        v = torch.randint(
            -127, 128, shape, generator=g, device="cuda", dtype=torch.int8
        )
        ks = (torch.rand(shape[:-1], generator=g, device="cuda") * 0.02 + 0.002).half()
        vs = (torch.rand(shape[:-1], generator=g, device="cuda") * 0.02 + 0.002).half()
        return (
            k,
            v,
            ks,
            vs,
            k.float() * ks.float()[..., None],
            v.float() * vs.float()[..., None],
        )
    k = torch.randn(shape, generator=g, device="cuda").to(torch.float8_e4m3fn)
    v = torch.randn(shape, generator=g, device="cuda").to(torch.float8_e4m3fn)
    return k, v, None, None, k.float(), v.float()


def _table(lengths: list[int], blocks: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(blocks, generator=g).tolist()
    need = [(n + BLOCK - 1) // BLOCK for n in lengths]
    width = max(need) + 2
    table = torch.full((len(lengths), width), -1, dtype=torch.long)
    start = 0
    for row, count in enumerate(need):
        table[row, :count] = torch.tensor(perm[start : start + count])
        start += count
    return table.cuda()


def _reference(q, kf, vf, table, lengths, positions, window, sink, scale):
    batch, heads, q_len, _ = q.shape
    out = torch.zeros(batch, heads, q_len, 128, device="cuda")
    for b in range(batch):
        for i in range(q_len):
            limit = min(int(lengths[b]), int(positions[b, i]) + 1)
            keep = [
                t for t in range(limit) if not window or t < sink or t >= limit - window
            ]
            if not keep:
                continue
            idx = torch.tensor(keep, device="cuda")
            blocks = table[b, idx // BLOCK]
            offsets = idx % BLOCK
            k = kf[blocks, :, offsets, :]  # [n, heads, 192]
            v = vf[blocks, :, offsets, :128]
            weights = torch.softmax(
                torch.einsum("hd,nhd->hn", q[b, :, i].float(), k) * scale, dim=-1
            )
            out[b, :, i] = torch.einsum("hn,nhd->hd", weights, v)
    return out


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("kind", ["fp16", "int8", "fp8"])
@pytest.mark.parametrize("attention", ["full", "window"])
@pytest.mark.parametrize("q_len", [1, 2])
@pytest.mark.parametrize("lengths", [[300, 77], [5000]], ids=["short", "long"])
def test_split_decode_matches_the_definition(
    kind: str, attention: str, q_len: int, lengths: list[int]
) -> None:
    blocks = sum((n + BLOCK - 1) // BLOCK for n in lengths) + 8
    k, v, ks, vs, kf, vf = _pools(kind, blocks, seed=len(lengths) * 7 + q_len)
    table = _table(lengths, blocks, seed=q_len)
    valid = torch.tensor(lengths, device="cuda")
    positions = torch.stack(
        [torch.arange(n - q_len, n, device="cuda") for n in lengths]
    )
    q = torch.randn(len(lengths), HEADS, q_len, 192, device="cuda").half() * 0.3
    window, sink = (64, 4) if attention == "window" else (None, None)
    scale = 192**-0.5
    assert paged_mla_split_supported(q, k, v)
    ours = paged_mla_decode_split(
        q,
        k,
        v,
        table,
        valid,
        positions,
        block_size=BLOCK,
        max_blocks=int(table.shape[1]),
        scale=scale,
        key_scale_pool=ks,
        value_scale_pool=vs,
        sliding_window=window,
        attention_sink=sink,
    )
    expected = _reference(
        q, kf, vf, table, valid, positions, window or 0, sink or 0, scale
    )
    torch.testing.assert_close(ours.float(), expected, atol=3e-3, rtol=3e-3)


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_a_query_with_no_visible_keys_gets_zeros() -> None:
    k, v, _, _, _, _ = _pools("fp16", 4, seed=0)
    table = _table([16], 4, seed=0)
    q = torch.randn(1, HEADS, 1, 192, device="cuda").half()
    out = paged_mla_decode_split(
        q,
        k,
        v,
        table,
        torch.tensor([0], device="cuda"),
        torch.tensor([[0]], device="cuda"),
        block_size=BLOCK,
        max_blocks=int(table.shape[1]),
        scale=0.1,
    )
    assert torch.count_nonzero(out) == 0


def test_split_plan_fills_the_gpu_and_bounds_the_slices() -> None:
    assert split_plan(4100, 16, 80) == (40, 103)  # 4K window, one decode step, 80 SMs
    assert split_plan(131072, 16, 80) == (
        256,
        512,
    )  # 128K full attention: 512-key slices
    assert split_plan(1, 16, 80) == (1, 1)
    assert split_plan(300, 16, 80) == (10, 30)  # never more slices than ~32 keys each
