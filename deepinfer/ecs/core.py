"""Runtime-built C++ control-plane backend for ECS.

The shared object contains only CPU bookkeeping.  It is deliberately separate
from the CUDA expert kernels and from the Python tensor ownership layer.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

_SOURCE = Path(__file__).with_name("ecs_core.cpp")
_LIBRARY: ctypes.CDLL | None = None
_LIBRARY_PATH: Path | None = None
_LOCK = threading.Lock()


def _compiler() -> str | None:
    configured = os.environ.get("DEEPINFER_ECS_CXX")
    if configured:
        return configured
    for candidate in ("g++", "clang++"):
        found = shutil.which(candidate)
        if found:
            return found
    return None


def _build_library() -> Path:
    compiler = _compiler()
    if compiler is None:
        raise RuntimeError("no C++ compiler available for ECS core")
    if not _SOURCE.exists():
        raise RuntimeError(f"missing ECS core source: {_SOURCE}")
    digest = hashlib.sha1(_SOURCE.read_bytes()).hexdigest()[:12]
    target = Path(tempfile.gettempdir()) / f"deepinfer_ecs_core_{digest}.dll"
    if target.exists() and target.stat().st_mtime >= _SOURCE.stat().st_mtime:
        return target
    command = [
        compiler,
        "-O3",
        "-std=c++17",
        "-shared",
        "-o",
        str(target),
        str(_SOURCE),
    ]
    if os.name == "nt":
        command.insert(-4, "-static")
    else:
        command[-4:-4] = ["-static-libgcc", "-static-libstdc++"]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "ECS C++ build failed:\n"
            + (completed.stdout or "")
            + (completed.stderr or "")
        )
    return target


def available() -> bool:
    try:
        load()
        return True
    except Exception:
        return False


def load() -> ctypes.CDLL:
    """Build and load the control-plane shared object once per process."""

    global _LIBRARY, _LIBRARY_PATH
    with _LOCK:
        if _LIBRARY is not None:
            return _LIBRARY
        _LIBRARY_PATH = _build_library()
        _LIBRARY = ctypes.CDLL(str(_LIBRARY_PATH))
        _LIBRARY.ecs_reset.argtypes = []
        _LIBRARY.ecs_reset.restype = None
        _LIBRARY.ecs_touch.argtypes = [
            ctypes.c_uint64,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_double,
        ]
        _LIBRARY.ecs_touch.restype = ctypes.c_int
        _LIBRARY.ecs_set_tier.argtypes = [
            ctypes.c_uint64,
            ctypes.c_int32,
            ctypes.c_int32,
        ]
        _LIBRARY.ecs_set_tier.restype = ctypes.c_int
        _LIBRARY.ecs_remove.argtypes = [ctypes.c_uint64]
        _LIBRARY.ecs_remove.restype = ctypes.c_int
        _LIBRARY.ecs_choose_victim.argtypes = [
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        _LIBRARY.ecs_choose_victim.restype = ctypes.c_int
        _LIBRARY.ecs_lookup.argtypes = [
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        _LIBRARY.ecs_lookup.restype = ctypes.c_int
        _LIBRARY.ecs_entry_count.argtypes = []
        _LIBRARY.ecs_entry_count.restype = ctypes.c_int
        _LIBRARY.ecs_plan_chunks.argtypes = [
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
        ]
        _LIBRARY.ecs_plan_chunks.restype = ctypes.c_int
        return _LIBRARY


class ECSCore:
    """Typed Python facade over the C++ policy core."""

    def __init__(self) -> None:
        self.lib = load()

    def reset(self) -> None:
        self.lib.ecs_reset()

    def touch(
        self,
        key: int,
        tier: int,
        device: int,
        bytes: int,
        tick: int,
        heat_delta: float,
    ) -> int:
        return int(
            self.lib.ecs_touch(
                int(key),
                int(tier),
                int(device),
                int(bytes),
                int(tick),
                float(heat_delta),
            )
        )

    def set_tier(self, key: int, tier: int, device: int) -> int:
        return int(self.lib.ecs_set_tier(int(key), int(tier), int(device)))

    def remove(self, key: int) -> int:
        return int(self.lib.ecs_remove(int(key)))

    def choose_victim(
        self,
        tier: int,
        device: int,
        required_bytes: int,
        tick: int,
    ) -> int | None:
        out = ctypes.c_uint64(0)
        result = int(
            self.lib.ecs_choose_victim(
                int(tier),
                int(device),
                int(required_bytes),
                int(tick),
                ctypes.byref(out),
            )
        )
        return int(out.value) if result == 0 else None

    def lookup(self, key: int) -> dict[str, Any] | None:
        tier = ctypes.c_int32()
        device = ctypes.c_int32()
        size = ctypes.c_uint64()
        heat = ctypes.c_double()
        last_tick = ctypes.c_uint64()
        result = int(
            self.lib.ecs_lookup(
                int(key),
                ctypes.byref(tier),
                ctypes.byref(device),
                ctypes.byref(size),
                ctypes.byref(heat),
                ctypes.byref(last_tick),
            )
        )
        if result != 0:
            return None
        return {
            "tier": int(tier.value),
            "device": int(device.value),
            "bytes": int(size.value),
            "heat": float(heat.value),
            "last_tick": int(last_tick.value),
        }

    def entry_count(self) -> int:
        return int(self.lib.ecs_entry_count())

    def plan_chunks(
        self,
        expert_ids: list[int],
        counts: list[int],
        *,
        max_experts: int,
        max_tokens: int,
    ) -> list[tuple[int, int]]:
        if len(expert_ids) != len(counts):
            raise ValueError("expert_ids and counts must have equal length")
        count = len(expert_ids)
        if count == 0:
            return []
        ids = (ctypes.c_int64 * count)(*[int(value) for value in expert_ids])
        values = (ctypes.c_int32 * count)(*[int(value) for value in counts])
        capacity = count + 1
        starts = (ctypes.c_int32 * capacity)()
        lengths = (ctypes.c_int32 * capacity)()
        chunks = int(
            self.lib.ecs_plan_chunks(
                ids,
                values,
                count,
                int(max_experts),
                int(max_tokens),
                starts,
                lengths,
                capacity,
            )
        )
        if chunks < 0:
            raise RuntimeError("C++ ECS chunk planner failed")
        return [(int(starts[i]), int(lengths[i])) for i in range(chunks)]


__all__ = ["ECSCore", "available", "load"]
