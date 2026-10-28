"""Expert Coordination System (ECS) control plane.

The controller owns placement metadata and cache accounting.  Tensor data lives
in the existing GPU/CPU/GGUF stores; ECS only decides which tier should hold an
expert, which victim to evict, and how a token batch should be packed.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Iterable, Sequence

from deepinfer.ecs.core import ECSCore, available as _core_available


class Tier(IntEnum):
    L1 = 0  # GPU HBM
    L2 = 1  # host RAM / pinned staging
    L3 = 2  # NVMe / GGUF file


@dataclass(slots=True)
class ExpertKey:
    layer: int
    projection: str
    expert_id: int

    @property
    def stable_id(self) -> int:
        payload = f"{self.layer}:{self.projection}:{self.expert_id}".encode()
        return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


@dataclass(slots=True)
class Placement:
    key: int
    tier: Tier
    device: int
    bytes: int
    heat: float
    last_tick: int


class ECSController:
    """Policy engine for heat/LRU/tier placement and chunk packing."""

    def __init__(
        self,
        *,
        l1_capacity_bytes: int = 0,
        l2_capacity_bytes: int = 0,
        devices: Sequence[int] = (0,),
        use_cpp: bool = True,
    ) -> None:
        self.l1_capacity_bytes = max(0, int(l1_capacity_bytes))
        self.l2_capacity_bytes = max(0, int(l2_capacity_bytes))
        self.devices = tuple(int(device) for device in devices) or (0,)
        self.use_cpp = bool(use_cpp and _core_available())
        self._core = ECSCore() if self.use_cpp else None
        self._entries: dict[int, Placement] = {}
        self._tier_bytes: dict[Tier, int] = defaultdict(int)
        self._device_bytes: dict[int, int] = defaultdict(int)
        self._lock = threading.RLock()
        self._tick = 0
        self.metrics: dict[str, float] = {
            "l1_hits": 0.0,
            "l2_hits": 0.0,
            "l3_reads": 0.0,
            "l1_promotions": 0.0,
            "evictions": 0.0,
            "chunk_plans": 0.0,
            "cross_device_packets": 0.0,
            "cross_device_bytes": 0.0,
            "hot_replications": 0.0,
        }
        if self._core is not None:
            self._core.reset()

    @property
    def backend(self) -> str:
        return "cpp" if self._core is not None else "python"

    def _next_tick(self) -> int:
        self._tick += 1
        return self._tick

    @staticmethod
    def _key(key: ExpertKey | int) -> int:
        return key if isinstance(key, int) else key.stable_id

    def _capacity(self, tier: Tier) -> int:
        if tier == Tier.L1:
            return self.l1_capacity_bytes
        if tier == Tier.L2:
            return self.l2_capacity_bytes
        return 0

    def _remove_metadata(self, key: int) -> None:
        old = self._entries.pop(key, None)
        if old is not None:
            self._tier_bytes[old.tier] -= old.bytes
            self._device_bytes[old.device] -= old.bytes
        if self._core is not None:
            self._core.remove(key)

    def lookup(self, key: ExpertKey | int) -> Placement | None:
        key_id = self._key(key)
        with self._lock:
            placement = self._entries.get(key_id)
            if placement is None:
                return None
            return Placement(
                key=key_id,
                tier=placement.tier,
                device=placement.device,
                bytes=placement.bytes,
                heat=placement.heat,
                last_tick=placement.last_tick,
            )

    def touch(
        self,
        key: ExpertKey | int,
        *,
        tier: Tier,
        device: int = 0,
        bytes: int = 0,
        heat_delta: float = 1.0,
    ) -> Placement:
        key_id = self._key(key)
        tick = self._next_tick()
        with self._lock:
            old = self._entries.get(key_id)
            if old is not None:
                self._tier_bytes[old.tier] -= old.bytes
                self._device_bytes[old.device] -= old.bytes
                heat = old.heat + heat_delta
            else:
                heat = heat_delta
            size = max(0, int(bytes))
            placement = Placement(
                key=key_id,
                tier=Tier(tier),
                device=int(device),
                bytes=size,
                heat=heat,
                last_tick=tick,
            )
            self._entries[key_id] = placement
            self._tier_bytes[placement.tier] += size
            self._device_bytes[placement.device] += size
            if self._core is not None:
                self._core.touch(
                    key_id,
                    int(placement.tier),
                    placement.device,
                    size,
                    tick,
                    float(heat_delta),
                )
            return placement

    def record_tier_access(self, tier: Tier, bytes: int = 0) -> None:
        with self._lock:
            if tier == Tier.L1:
                self.metrics["l1_hits"] += 1.0
            elif tier == Tier.L2:
                self.metrics["l2_hits"] += 1.0
            else:
                self.metrics["l3_reads"] += 1.0
            if bytes:
                self.metrics["cross_device_bytes"] += 0.0

    def evict(
        self,
        *,
        tier: Tier,
        device: int = -1,
        required_bytes: int = 0,
    ) -> bool:
        with self._lock:
            victim_key: int | None = None
            if self._core is not None:
                victim_key = self._core.choose_victim(
                    int(tier),
                    int(device),
                    int(required_bytes),
                    self._next_tick(),
                )
            if victim_key is None:
                candidates = [
                    entry
                    for entry in self._entries.values()
                    if entry.tier == tier
                    and (device < 0 or entry.device == device)
                ]
                if not candidates:
                    return False
                victim = min(
                    candidates,
                    key=lambda entry: (entry.heat, entry.last_tick, entry.key),
                )
                victim_key = victim.key
            self._remove_metadata(victim_key)
            self.metrics["evictions"] += 1.0
            return True

    def place(
        self,
        key: ExpertKey | int,
        *,
        tier: Tier,
        device: int = 0,
        bytes: int = 0,
        heat_delta: float = 0.0,
    ) -> bool:
        key_id = self._key(key)
        size = max(0, int(bytes))
        with self._lock:
            capacity = self._capacity(tier)
            if capacity > 0:
                current = self._tier_bytes[tier]
                while current + size > capacity:
                    if not self.evict(tier=tier, device=device):
                        return False
                    current = self._tier_bytes[tier]
            self.touch(
                key_id,
                tier=tier,
                device=device,
                bytes=size,
                heat_delta=heat_delta,
            )
            if tier == Tier.L1:
                self.metrics["l1_promotions"] += 1.0
            return True

    def observe_routes(
        self,
        layer: int,
        expert_ids: Sequence[int],
        counts: Sequence[int],
    ) -> None:
        if len(expert_ids) != len(counts):
            raise ValueError("expert_ids and counts must have equal length")
        for expert_id, count in zip(expert_ids, counts, strict=True):
            self.touch(
                ExpertKey(layer, "route", int(expert_id)),
                tier=Tier.L3,
                bytes=0,
                heat_delta=float(count),
            )

    def heat(self, layer: int, expert_id: int) -> float:
        key = ExpertKey(layer, "route", int(expert_id))
        placement = self.lookup(key)
        return 0.0 if placement is None else float(placement.heat)

    def choose_cold_experts(
        self,
        layer: int,
        expert_ids: Sequence[int],
        *,
        count: int,
    ) -> set[int]:
        if count <= 0:
            return set()
        ranked = sorted(
            (int(expert_id) for expert_id in expert_ids),
            key=lambda expert_id: (self.heat(layer, expert_id), expert_id),
        )
        return set(ranked[: min(count, len(ranked))])

    def choose_device(self, key: ExpertKey | int, *, bytes: int = 0) -> int:
        key_id = self._key(key)
        with self._lock:
            current = self._entries.get(key_id)
            if current is not None:
                return current.device
            return min(self.devices, key=lambda device: self._device_bytes[device])

    def plan_chunks(
        self,
        expert_ids: Sequence[int],
        counts: Sequence[int],
        *,
        max_experts: int,
        max_tokens: int,
    ) -> list[tuple[int, int]]:
        if len(expert_ids) != len(counts):
            raise ValueError("expert_ids and counts must have equal length")
        self.metrics["chunk_plans"] += 1.0
        if self._core is not None:
            return self._core.plan_chunks(
                [int(value) for value in expert_ids],
                [int(value) for value in counts],
                max_experts=max(1, int(max_experts)),
                max_tokens=max(1, int(max_tokens)),
            )
        chunks: list[tuple[int, int]] = []
        start = 0
        while start < len(expert_ids):
            end = start
            tokens = 0
            while end < len(expert_ids) and end - start < max_experts:
                next_tokens = max(0, int(counts[end]))
                if end > start and tokens + next_tokens > max_tokens:
                    break
                tokens += next_tokens
                end += 1
            if end == start:
                end += 1
            chunks.append((start, end - start))
            start = end
        return chunks

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "backend": self.backend,
                "tier_bytes": {
                    tier.name: int(self._tier_bytes[tier])
                    for tier in (Tier.L1, Tier.L2, Tier.L3)
                },
                "device_bytes": {
                    str(device): int(size)
                    for device, size in sorted(self._device_bytes.items())
                },
                "entries": len(self._entries),
                "metrics": {key: float(value) for key, value in self.metrics.items()},
                "tick": self._tick,
            }


__all__ = ["ECSController", "ExpertKey", "Placement", "Tier"]
