"""A tiny, dependency-light GGUF index for streamed MoE inference.

The engine only needs a few things from a GGUF checkpoint:

* tensor names, shapes, quantization types and byte offsets;
* the absolute tensor-data offset;
* the ability to read a small packed expert slice without loading the
  multi-gigabyte checkpoint into anonymous memory.

This module intentionally does not use ``GGUFReader``.  ``GGUFReader`` maps
every tensor eagerly and is convenient for conversion tools, but a malformed
or partially downloaded file can fail only after it has allocated large
arrays.  The parser below validates the complete file first, then keeps a
read-only file mapping for on-demand slice reads.
"""

from __future__ import annotations

import math
import mmap
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType, dequantize


class GGUFError(RuntimeError):
    """Base error for malformed GGUF checkpoints."""


class GGUFTruncatedError(GGUFError):
    """Raised when tensor metadata points past the end of the file."""


@dataclass(frozen=True, slots=True)
class GGUFTensorInfo:
    """One GGUF tensor descriptor.

    ``dims`` follows the GGML convention: the innermost input dimension is
    first.  For a linear weight this is ``(in_features, out_features)``.  The
    dequantized NumPy view uses the reverse order, i.e. ``(out, in)``.
    """

    name: str
    dims: tuple[int, ...]
    qtype: GGMLQuantizationType
    offset: int
    nbytes: int

    @property
    def element_count(self) -> int:
        return math.prod(self.dims)

    @property
    def block_size(self) -> int:
        return GGML_QUANT_SIZES[self.qtype][0]

    @property
    def type_size(self) -> int:
        return GGML_QUANT_SIZES[self.qtype][1]

    @property
    def is_quantized(self) -> bool:
        return not (
            self.qtype
            in (
                GGMLQuantizationType.F32,
                GGMLQuantizationType.F16,
                GGMLQuantizationType.BF16,
            )
        )

    @property
    def byte_shape(self) -> tuple[int, ...]:
        """Shape of the raw tensor bytes as expected by :func:`gguf.dequantize`."""

        if not self.is_quantized:
            raise ValueError(f"{self.name} is not quantized")
        reversed_dims = tuple(reversed(self.dims))
        last = reversed_dims[-1] // self.block_size * self.type_size
        return (*reversed_dims[:-1], last)

    @property
    def numpy_shape(self) -> tuple[int, ...]:
        """Dequantized shape in ``(out, in, ...)`` order."""

        return tuple(reversed(self.dims))


# GGUF metadata value type ids.
_GGUF_UINT8 = 0
_GGUF_INT8 = 1
_GGUF_UINT16 = 2
_GGUF_INT16 = 3
_GGUF_UINT32 = 4
_GGUF_INT32 = 5
_GGUF_FLOAT32 = 6
_GGUF_BOOL = 7
_GGUF_STRING = 8
_GGUF_ARRAY = 9
_GGUF_UINT64 = 10
_GGUF_INT64 = 11
_GGUF_FLOAT64 = 12

_SCALAR_STRUCTS: dict[int, struct.Struct] = {
    _GGUF_UINT8: struct.Struct("<B"),
    _GGUF_INT8: struct.Struct("<b"),
    _GGUF_UINT16: struct.Struct("<H"),
    _GGUF_INT16: struct.Struct("<h"),
    _GGUF_UINT32: struct.Struct("<I"),
    _GGUF_INT32: struct.Struct("<i"),
    _GGUF_FLOAT32: struct.Struct("<f"),
    _GGUF_BOOL: struct.Struct("<?"),
    _GGUF_UINT64: struct.Struct("<Q"),
    _GGUF_INT64: struct.Struct("<q"),
    _GGUF_FLOAT64: struct.Struct("<d"),
}


class _HeaderReader:
    def __init__(self, file_obj, path: Path) -> None:
        self.file = file_obj
        self.path = path

    def read_exact(self, size: int) -> bytes:
        data = self.file.read(size)
        if len(data) != size:
            raise GGUFTruncatedError(
                f"{self.path} ended while reading GGUF header at offset "
                f"{self.file.tell()} (wanted {size} bytes, got {len(data)})"
            )
        return data

    def u32(self) -> int:
        return struct.unpack("<I", self.read_exact(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read_exact(8))[0]

    def string(self) -> str:
        size = self.u64()
        return self.read_exact(size).decode("utf-8", "replace")

    def scalar(self, value_type: int):
        if value_type == _GGUF_STRING:
            return self.string()
        if value_type == _GGUF_ARRAY:
            raise ValueError("arrays are not scalars")
        try:
            return _SCALAR_STRUCTS[value_type].unpack(self.read_exact(
                _SCALAR_STRUCTS[value_type].size
            ))[0]
        except KeyError as exc:
            raise GGUFError(f"unknown GGUF metadata value type {value_type}") from exc

    def skip_array_elements(self, element_type: int, count: int) -> None:
        if element_type == _GGUF_STRING:
            for _ in range(count):
                size = self.u64()
                self.file.seek(size, 1)
            return
        if element_type == _GGUF_ARRAY:
            for _ in range(count):
                nested_type = self.u32()
                nested_count = self.u64()
                self.skip_array_elements(nested_type, nested_count)
            return
        try:
            element_size = _SCALAR_STRUCTS[element_type].size
        except KeyError as exc:
            raise GGUFError(
                f"unknown GGUF array element type {element_type}"
            ) from exc
        self.file.seek(element_size * count, 1)

    def metadata_value(self, value_type: int, *, max_array: int = 32):
        if value_type != _GGUF_ARRAY:
            return self.scalar(value_type)
        element_type = self.u32()
        count = self.u64()
        if count > max_array:
            self.skip_array_elements(element_type, count)
            return {
                "array_element_type": element_type,
                "array_length": count,
            }
        return [self.metadata_value(element_type, max_array=max_array) for _ in range(count)]


class GGUFIndex:
    """Parsed GGUF header plus a read-only file mapping."""

    def __init__(self, path: str | Path, *, use_mmap: bool = True) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.file_size = self.path.stat().st_size
        self.tensors: dict[str, GGUFTensorInfo] = {}
        self.metadata: dict[str, object] = {}
        self.version = 0
        self.alignment = 32
        self.data_start = 0
        self._mmap: mmap.mmap | None = None
        self._file_handle = None

        with self.path.open("rb") as file_obj:
            reader = _HeaderReader(file_obj, self.path)
            magic = reader.read_exact(4)
            if magic != b"GGUF":
                raise GGUFError(f"{self.path} is not a GGUF file (magic={magic!r})")
            self.version = reader.u32()
            if self.version not in (2, 3):
                raise GGUFError(f"unsupported GGUF version {self.version}")
            tensor_count = reader.u64()
            metadata_count = reader.u64()

            for _ in range(metadata_count):
                key = reader.string()
                value_type = reader.u32()
                self.metadata[key] = reader.metadata_value(value_type)

            self.alignment = int(self.metadata.get("general.alignment", 32))
            if self.alignment < 1 or self.alignment & (self.alignment - 1):
                raise GGUFError(f"invalid GGUF alignment {self.alignment}")

            for _ in range(tensor_count):
                name = reader.string()
                n_dims = reader.u32()
                if n_dims < 1 or n_dims > 8:
                    raise GGUFError(f"tensor {name!r} has invalid rank {n_dims}")
                dims = tuple(reader.u64() for _ in range(n_dims))
                qtype_raw = reader.u32()
                try:
                    qtype = GGMLQuantizationType(qtype_raw)
                    block_size, type_size = GGML_QUANT_SIZES[qtype]
                except (ValueError, KeyError) as exc:
                    raise GGUFError(
                        f"tensor {name!r} uses unsupported GGML type {qtype_raw}"
                    ) from exc
                offset = reader.u64()
                elements = math.prod(dims)
                if elements % block_size:
                    raise GGUFError(
                        f"tensor {name!r} element count {elements} is not a "
                        f"multiple of block size {block_size}"
                    )
                self.tensors[name] = GGUFTensorInfo(
                    name=name,
                    dims=dims,
                    qtype=qtype,
                    offset=offset,
                    nbytes=(elements // block_size) * type_size,
                )

            header_end = file_obj.tell()
            self.data_start = (
                (header_end + self.alignment - 1) // self.alignment * self.alignment
            )

        self.tensor_data_bytes = max(
            (tensor.offset + tensor.nbytes for tensor in self.tensors.values()),
            default=0,
        )
        required_size = self.data_start + self.tensor_data_bytes
        if required_size > self.file_size:
            raise GGUFTruncatedError(
                f"{self.path} is incomplete: tensor data needs {required_size} "
                f"bytes but file has only {self.file_size}; resume the download "
                "before attempting to load it"
            )

        if use_mmap:
            self._file_handle = self.path.open("rb")
            self._mmap = mmap.mmap(
                self._file_handle.fileno(),
                length=0,
                access=mmap.ACCESS_READ,
            )

    def close(self) -> None:
        if self._mmap is not None:
            try:
                self._mmap.close()
            except BufferError:
                # A caller may still hold a NumPy view over the mapping.  The
                # process will release it on exit; dropping the mapping here
                # is safer than hiding a real truncation bug.
                pass
            self._mmap = None
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None

    def __enter__(self) -> GGUFIndex:
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def tensor(self, name: str) -> GGUFTensorInfo:
        try:
            return self.tensors[name]
        except KeyError as exc:
            raise KeyError(f"GGUF tensor {name!r} was not found in {self.path}") from exc

    def tensor_names(self, prefix: str = "") -> list[str]:
        return sorted(name for name in self.tensors if name.startswith(prefix))

    def read_raw(
        self,
        tensor: GGUFTensorInfo | str,
        *,
        count: int | None = None,
        relative_offset: int = 0,
        copy: bool = False,
    ) -> np.ndarray:
        """Return a uint8 view/copy of a tensor byte range."""

        info = self.tensor(tensor) if isinstance(tensor, str) else tensor
        if count is None:
            count = info.nbytes - relative_offset
        if relative_offset < 0 or count < 0 or relative_offset + count > info.nbytes:
            raise ValueError(
                f"byte range [{relative_offset}, {relative_offset + count}) "
                f"is outside tensor {info.name!r} ({info.nbytes} bytes)"
            )
        absolute_offset = self.data_start + info.offset + relative_offset
        if self._mmap is not None:
            array = np.frombuffer(
                self._mmap,
                dtype=np.uint8,
                count=count,
                offset=absolute_offset,
            )
            return np.array(array, copy=True) if copy else array
        with self.path.open("rb") as file_obj:
            file_obj.seek(absolute_offset)
            data = file_obj.read(count)
        if len(data) != count:
            raise GGUFTruncatedError(
                f"short read from {self.path} at {absolute_offset}: "
                f"wanted {count}, got {len(data)}"
            )
        return np.frombuffer(data, dtype=np.uint8)

    def read_tensor(self, name: str) -> np.ndarray:
        """Dequantize a full tensor once.  This is for non-expert weights only."""

        info = self.tensor(name)
        raw = self.read_raw(info)
        if not info.is_quantized:
            dtype = {
                GGMLQuantizationType.F32: np.float32,
                GGMLQuantizationType.F16: np.float16,
            }.get(info.qtype)
            if dtype is None:
                raise NotImplementedError(
                    f"unquantized tensor {name!r} type {info.qtype.name} is not supported"
                )
            return np.frombuffer(raw, dtype=dtype).reshape(info.numpy_shape)
        return dequantize(raw.reshape(info.byte_shape), info.qtype)

    def expert_tensor(
        self,
        name: str,
        expert_index: int,
        *,
        expert_count: int | None = None,
    ) -> np.ndarray:
        """Read and dequantize exactly one expert from a 3-D expert tensor.

        GGUF stores expert tensors as ``(in, out, expert)`` with the input
        dimension contiguous.  Returning an ``(out, in)`` matrix matches
        ``torch.nn.functional.linear`` and the HuggingFace parameter shape.
        """

        info = self.tensor(name)
        if len(info.dims) != 3:
            raise ValueError(f"expected a 3-D expert tensor for {name!r}, got {info.dims}")
        in_features, out_features, stored_experts = info.dims
        if expert_count is not None and stored_experts != expert_count:
            raise ValueError(
                f"{name!r} stores {stored_experts} experts, expected {expert_count}"
            )
        if expert_index < 0 or expert_index >= stored_experts:
            raise IndexError(
                f"expert {expert_index} is outside {name!r} ({stored_experts} experts)"
            )
        elements_per_expert = in_features * out_features
        bytes_per_expert = elements_per_expert // info.block_size * info.type_size
        raw = self.read_raw(
            info,
            count=bytes_per_expert,
            relative_offset=expert_index * bytes_per_expert,
        )
        if info.is_quantized:
            byte_shape = (
                out_features,
                in_features // info.block_size * info.type_size,
            )
            return dequantize(raw.reshape(byte_shape), info.qtype)
        dtype = {
            GGMLQuantizationType.F32: np.float32,
            GGMLQuantizationType.F16: np.float16,
        }.get(info.qtype)
        if dtype is None:
            raise NotImplementedError(
                f"expert tensor type {info.qtype.name} is not supported"
            )
        return np.frombuffer(raw, dtype=dtype).reshape(out_features, in_features)


def inspect_gguf(path: str | Path) -> dict[str, object]:
    """Small diagnostic helper used by the CLI and tests."""

    with GGUFIndex(path) as index:
        qtypes: dict[str, int] = {}
        total_bytes = 0
        for tensor in index.tensors.values():
            qtypes[tensor.qtype.name] = qtypes.get(tensor.qtype.name, 0) + 1
            total_bytes += tensor.nbytes
        return {
            "path": str(index.path),
            "file_size": index.file_size,
            "data_start": index.data_start,
            "tensor_count": len(index.tensors),
            "tensor_data_bytes": total_bytes,
            "quantization_types": qtypes,
            "metadata": {
                key: value
                for key, value in index.metadata.items()
                if not key.startswith("tokenizer.")
            },
        }
