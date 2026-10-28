"""Runtime-built AVX2 dequantizer for CPU-offloaded packed experts.

The shared library is compiled once from ``cpu_avx2_packed.cpp`` using the
MinGW g++ available on this machine.  Q8_0 uses explicit AVX2 int8->float
conversion; Q4_K/Q5_0 use the same block layouts as the GPU/PyTorch reference
and are compiled with ``-mavx2 -mfma`` so the scalar decode loops auto-vectorize.
If no compiler is available the caller can fall back to PyTorch CPU dequant.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import torch

_SOURCE = Path(__file__).with_name("cpu_avx2_packed.cpp")
_LIB: Any = None
_LIB_PATH: Path | None = None
_FUNCTIONS: dict[str, Any] = {}

_BLOCK_BYTES = {"Q4_K": (256, 144), "Q5_0": (32, 22), "Q8_0": (32, 34)}


def _library_path() -> Path:
    digest = hashlib.sha1(_SOURCE.read_bytes()).hexdigest()[:10]
    suffix = ".dll" if os.name == "nt" else ".so"
    return Path(tempfile.gettempdir()) / f"deepinfer_cpu_avx2_{digest}{suffix}"


def _compile() -> Path:
    compiler = os.environ.get("DEEPINFER_CPU_AVX2_COMPILER") or shutil.which("g++")
    if not compiler:
        raise RuntimeError("g++ is unavailable; cannot build CPU AVX2 dequantizer")
    if not _SOURCE.exists():
        raise RuntimeError(f"missing AVX2 source: {_SOURCE}")
    target = _library_path()
    if target.exists() and target.stat().st_mtime >= _SOURCE.stat().st_mtime:
        return target
    command = [
        compiler,
        "-O3",
        "-mavx2",
        "-mfma",
        "-std=c++17",
        "-shared",
        "-static-libgcc",
        "-static-libstdc++",
        "-o",
        str(target),
        str(_SOURCE),
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "CPU AVX2 compile failed: "
            + completed.stderr.decode("utf-8", "replace")[-2000:]
        )
    return target


def _load() -> Any:
    global _LIB, _LIB_PATH
    if _LIB is not None:
        return _LIB
    _LIB_PATH = _compile()
    _LIB = ctypes.CDLL(str(_LIB_PATH))
    for qtype, name in (
        ("Q4_K", "dp_dequant_q4_k"),
        ("Q5_0", "dp_dequant_q5_0"),
        ("Q8_0", "dp_dequant_q8_0"),
    ):
        function = getattr(_LIB, name)
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        function.restype = None
        _FUNCTIONS[qtype] = function
    return _LIB


def available() -> bool:
    try:
        _load()
    except Exception:
        return False
    return True


def dequantize(
    packed: torch.Tensor,
    quant_type: str,
    *,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """Dequantize packed CPU rows to FP32 using the AVX2 shared library."""

    qtype = quant_type.upper()
    if qtype not in _BLOCK_BYTES:
        raise NotImplementedError(f"AVX2 CPU dequant does not support {qtype}")
    _load()
    block_size, block_bytes = _BLOCK_BYTES[qtype]
    if in_features % block_size:
        raise ValueError(f"{qtype} in_features must be divisible by {block_size}")
    blocks = in_features // block_size
    expected = out_features * blocks * block_bytes
    if packed.numel() != expected:
        raise ValueError(
            f"{qtype} packed byte mismatch: got {packed.numel()}, expected {expected}"
        )
    packed_cpu = packed.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    output = torch.empty(
        (out_features, in_features), dtype=torch.float32, device="cpu"
    )
    _FUNCTIONS[qtype](
        ctypes.c_void_p(packed_cpu.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        int(out_features),
        int(blocks),
    )
    return output


__all__ = ["available", "dequantize"]
