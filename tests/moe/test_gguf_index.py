from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
from gguf import GGMLQuantizationType, dequantize

from deepinfer.moe.gguf_index import GGUFIndex, GGUFTruncatedError


def _write_string(buffer: bytearray, value: str) -> None:
    encoded = value.encode("utf-8")
    buffer.extend(struct.pack("<Q", len(encoded)))
    buffer.extend(encoded)


def _write_gguf(
    path: Path,
    tensors: list[tuple[str, tuple[int, ...], GGMLQuantizationType, bytes]],
    *,
    alignment: int = 32,
) -> None:
    header = bytearray(b"GGUF")
    header.extend(struct.pack("<IQQ", 3, len(tensors), 1))
    _write_string(header, "general.alignment")
    header.extend(struct.pack("<I", 4))
    header.extend(struct.pack("<I", alignment))

    tensor_data = bytearray()
    offsets: list[int] = []
    for _name, _dims, _qtype, raw in tensors:
        offset = (len(tensor_data) + alignment - 1) // alignment * alignment
        tensor_data.extend(b"\x00" * (offset - len(tensor_data)))
        offsets.append(offset)
        tensor_data.extend(raw)
    for (name, dims, qtype, _raw), offset in zip(tensors, offsets, strict=True):
        _write_string(header, name)
        header.extend(struct.pack("<I", len(dims)))
        for dim in dims:
            header.extend(struct.pack("<Q", dim))
        header.extend(struct.pack("<I", int(qtype)))
        header.extend(struct.pack("<Q", offset))
    data_start = (len(header) + alignment - 1) // alignment * alignment
    header.extend(b"\x00" * (data_start - len(header)))
    path.write_bytes(bytes(header) + bytes(tensor_data))


def _q4_k_experts(expert_count: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(17)
    raw = rng.integers(0, 256, size=(expert_count, 2, 144), dtype=np.uint8)
    raw[:, :, 0:2] = np.frombuffer(struct.pack("<e", 0.01), dtype=np.uint8)
    raw[:, :, 2:4] = np.frombuffer(struct.pack("<e", 0.001), dtype=np.uint8)
    return raw.reshape(-1), raw


def test_index_parses_f32_tensor_and_metadata(tmp_path: Path) -> None:
    values = np.arange(8, dtype=np.float32).reshape(2, 4)
    path = tmp_path / "tiny.gguf"
    _write_gguf(
        path,
        [("test.weight", (4, 2), GGMLQuantizationType.F32, values.tobytes())],
    )

    with GGUFIndex(path) as index:
        assert index.metadata["general.alignment"] == 32
        tensor = index.tensor("test.weight")
        assert tensor.dims == (4, 2)
        assert tensor.numpy_shape == (2, 4)
        assert tensor.nbytes == values.nbytes
        parsed = index.read_tensor("test.weight")
        np.testing.assert_array_equal(parsed, values)


def test_expert_tensor_dequantizes_only_requested_expert(tmp_path: Path) -> None:
    raw, blocks = _q4_k_experts()
    path = tmp_path / "experts.gguf"
    _write_gguf(
        path,
        [
            (
                "blk.1.ffn_gate_exps.weight",
                (256, 2, 3),
                GGMLQuantizationType.Q4_K,
                raw.tobytes(),
            )
        ],
    )

    with GGUFIndex(path) as index:
        expert = index.expert_tensor("blk.1.ffn_gate_exps.weight", 1)
        reference = dequantize(blocks[1], GGMLQuantizationType.Q4_K)
        assert expert.shape == (2, 256)
        np.testing.assert_allclose(expert, reference, rtol=0, atol=0)
        with pytest.raises(IndexError):
            index.expert_tensor("blk.1.ffn_gate_exps.weight", 3)


def test_truncated_gguf_is_rejected_before_loading(tmp_path: Path) -> None:
    raw, _ = _q4_k_experts(1)
    path = tmp_path / "truncated.gguf"
    _write_gguf(
        path,
        [
            (
                "blk.1.ffn_gate_exps.weight",
                (256, 2, 1),
                GGMLQuantizationType.Q4_K,
                raw.tobytes(),
            )
        ],
    )
    path.write_bytes(path.read_bytes()[:-16])
    with pytest.raises(GGUFTruncatedError):
        GGUFIndex(path)
