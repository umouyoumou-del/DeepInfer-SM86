"""Split-shard discovery and index merging."""

from __future__ import annotations

import numpy as np
import pytest

from deepinfer.moe.gguf_index import GGUFError
from deepinfer.moe.gguf_split import (
    SplitGGUFError,
    SplitGGUFIndex,
    discover_split_shards,
    open_gguf,
)


def _write_shard(directory, stem, number, total, *, metadata_extra=None):
    """Create a minimal but structurally valid GGUF shard on disk."""

    path = directory / f"{stem}-{number:05d}-of-{total:05d}.gguf"
    meta = {"general.architecture": "qwen2"}
    if metadata_extra:
        meta.update(metadata_extra)
    meta.setdefault("split.count", total)
    meta.setdefault("split.no", number - 1)
    # Use the reference writer so the parser is exercised for real.
    from gguf import GGUFWriter

    writer = GGUFWriter(str(path), "qwen2", use_temp_file=False)
    for key, value in meta.items():
        if isinstance(value, int):
            writer.add_uint32(key, value)
        else:
            writer.add_string(key, str(value))
    writer.add_tensor(
        f"blk.{number - 1}.attn_q.weight",
        np.zeros((2, 2), dtype=np.float32),
    )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


@pytest.fixture()
def shard_dir(tmp_path):
    _write_shard(tmp_path, "model", 1, 2)
    _write_shard(tmp_path, "model", 2, 2)
    return tmp_path


def test_discovers_and_orders_shards(shard_dir) -> None:
    shards = discover_split_shards(shard_dir)
    assert [path.name for path in shards] == [
        "model-00001-of-00002.gguf",
        "model-00002-of-00002.gguf",
    ]


def test_missing_shard_is_reported(shard_dir) -> None:
    (shard_dir / "model-00002-of-00002.gguf").unlink()
    with pytest.raises(SplitGGUFError) as excinfo:
        discover_split_shards(shard_dir)
    assert "missing" in str(excinfo.value)


def test_single_shard_path_resolves_the_whole_set(shard_dir) -> None:
    shards = discover_split_shards(shard_dir / "model-00002-of-00002.gguf")
    assert len(shards) == 2


def test_index_merges_tensors_and_routes_reads(shard_dir) -> None:
    index = SplitGGUFIndex(discover_split_shards(shard_dir))
    assert index.shard_count == 2
    assert index.tensor_names() == [
        "blk.0.attn_q.weight",
        "blk.1.attn_q.weight",
    ]
    assert index.tensor_owner("blk.0.attn_q.weight") == 0
    assert index.tensor_owner("blk.1.attn_q.weight") == 1
    assert index.read_tensor("blk.1.attn_q.weight").shape == (2, 2)
    assert index.metadata["general.architecture"] == "qwen2"
    assert index.stats()["shards"] == 2
    index.close()


def test_unknown_tensor_names_the_shards(shard_dir) -> None:
    index = SplitGGUFIndex(discover_split_shards(shard_dir))
    with pytest.raises(KeyError) as excinfo:
        index.read_tensor("blk.9.attn_q.weight")
    assert "not present in any shard" in str(excinfo.value)


def test_split_count_mismatch_is_rejected(shard_dir) -> None:
    extra = _write_shard(shard_dir, "other", 1, 3)
    shards = discover_split_shards(shard_dir / "model-00001-of-00002.gguf")
    with pytest.raises(SplitGGUFError):
        # Two shards opened, but their split.count agrees on 3.
        SplitGGUFIndex(shards + [extra])


def test_open_gguf_transparently_handles_both_shapes(shard_dir) -> None:
    split = open_gguf(shard_dir)
    assert isinstance(split, SplitGGUFIndex)
    from deepinfer.moe.gguf_index import GGUFIndex

    single = open_gguf(shard_dir / "model-00001-of-00002.gguf")
    assert isinstance(single, SplitGGUFIndex)
    assert len(split) == 2


def test_non_gguf_file_still_raises() -> None:
    with pytest.raises(GGUFError):
        SplitGGUFIndex([])
