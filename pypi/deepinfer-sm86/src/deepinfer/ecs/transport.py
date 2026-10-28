"""Cross-device ECS transport primitives.

The tensor math remains on the GPU.  This module only packs expert/token
routing metadata into device-addressed packets and provides a simulated
all-to-all for multi-GPU tests plus an optional torch.distributed backend.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from typing import Any, Iterable

from deepinfer.ecs.controller import ECSController, ExpertKey


@dataclass(slots=True, frozen=True)
class ExpertPacket:
    src_device: int
    dst_device: int
    layer: int
    expert_id: int
    token_ids: tuple[int, ...]
    expert_bytes: int = 0
    payload: bytes = b""

    @property
    def token_count(self) -> int:
        return len(self.token_ids)

    @property
    def transfer_bytes(self) -> int:
        return int(self.expert_bytes) + len(self.payload)


@dataclass(slots=True)
class RoutingPlan:
    packets: list[ExpertPacket]
    by_device: dict[int, list[ExpertPacket]] = field(default_factory=dict)
    cross_device_packets: int = 0
    cross_device_bytes: int = 0
    assigned_experts: dict[int, int] = field(default_factory=dict)
    queue_depths: dict[int, float] = field(default_factory=dict)
    hot_replications: int = 0

    def device_batches(self) -> dict[int, list[ExpertPacket]]:
        if self.by_device:
            return self.by_device
        grouped: dict[int, list[ExpertPacket]] = {}
        for packet in self.packets:
            grouped.setdefault(packet.dst_device, []).append(packet)
        self.by_device = grouped
        return grouped


@dataclass(slots=True)
class DeviceLoad:
    queue_depth: float = 0.0
    kv_used_ratio: float = 0.0
    inflight_bytes: int = 0
    healthy: bool = True


class ECSLoadBalancer:
    """Queue-depth + KV pressure aware device selection."""

    def __init__(self, devices: Iterable[int]) -> None:
        self.devices = tuple(int(device) for device in devices)
        self.states: dict[int, DeviceLoad] = {
            device: DeviceLoad() for device in self.devices
        }

    def update(self, device: int, **kwargs: Any) -> None:
        state = self.states[int(device)]
        for name, value in kwargs.items():
            if hasattr(state, name):
                setattr(state, name, value)

    def snapshot(self) -> dict[int, dict[str, Any]]:
        return {
            device: {
                "queue_depth": state.queue_depth,
                "kv_used_ratio": state.kv_used_ratio,
                "inflight_bytes": state.inflight_bytes,
                "healthy": state.healthy,
            }
            for device, state in sorted(self.states.items())
        }

    def choose(
        self,
        *,
        expert_bytes: int,
        preferred_device: int | None = None,
    ) -> int:
        candidates = [
            device for device in self.devices if self.states[device].healthy
        ] or list(self.devices)

        def score(device: int) -> tuple[float, int]:
            state = self.states[device]
            preference_penalty = (
                0.0
                if preferred_device is None or preferred_device == device
                else max(1.0, expert_bytes / 1_000_000.0)
            )
            return (
                state.queue_depth
                + state.kv_used_ratio * 8.0
                + state.inflight_bytes / 1_000_000_000.0
                + preference_penalty,
                device,
            )

        return min(candidates, key=score)


class ECSRouter:
    """Turn expert→token assignments into device-addressed packets."""

    def __init__(
        self,
        controller: ECSController,
        *,
        devices: Iterable[int] | None = None,
        replicate_hot: bool = True,
        hot_threshold: float = 8.0,
        load_balancer: ECSLoadBalancer | None = None,
    ) -> None:
        self.controller = controller
        self.devices = tuple(devices or controller.devices)
        self.replicate_hot = bool(replicate_hot)
        self.hot_threshold = float(hot_threshold)
        self.load_balancer = load_balancer or ECSLoadBalancer(self.devices)

    def plan(
        self,
        layer: int,
        assignments: dict[int, list[int]],
        *,
        expert_bytes: int = 0,
        src_device: int = 0,
        queue_depths: dict[int, float] | None = None,
        kv_pressure: dict[int, float] | None = None,
    ) -> RoutingPlan:
        for device, depth in (queue_depths or {}).items():
            self.load_balancer.update(device, queue_depth=float(depth))
        for device, pressure in (kv_pressure or {}).items():
            self.load_balancer.update(device, kv_used_ratio=float(pressure))

        loads = {device: 0 for device in self.devices}
        packets: list[ExpertPacket] = []
        assigned: dict[int, int] = {}
        hot_copies = 0
        for expert_id in sorted(assignments):
            token_ids = tuple(int(token) for token in assignments[expert_id])
            if not token_ids:
                continue
            key = ExpertKey(layer, "route", int(expert_id))
            preferred = self.controller.choose_device(key, bytes=expert_bytes)
            heat = self.controller.heat(layer, expert_id)
            if (
                self.replicate_hot
                and heat >= self.hot_threshold
                and len(self.devices) > 1
            ):
                targets = sorted(
                    self.devices,
                    key=lambda device: (
                        self.load_balancer.states[device].queue_depth
                        + self.load_balancer.states[device].kv_used_ratio * 8.0
                        + loads[device] / 1_000_000.0,
                        device,
                    ),
                )[:2]
                split = max(1, len(token_ids) // len(targets))
                shards = [
                    token_ids[index : index + split]
                    for index in range(0, len(token_ids), split)
                ]
                for target, shard in zip(targets, shards, strict=False):
                    if not shard:
                        continue
                    packets.append(
                        ExpertPacket(
                            src_device=int(src_device),
                            dst_device=int(target),
                            layer=int(layer),
                            expert_id=int(expert_id),
                            token_ids=tuple(shard),
                            expert_bytes=int(expert_bytes),
                        )
                    )
                    loads[target] += max(0, expert_bytes)
                assigned[int(expert_id)] = int(targets[0])
                hot_copies += max(0, len(shards) - 1)
                self.controller.metrics["hot_replications"] += 1.0
                continue

            dst = self.load_balancer.choose(
                expert_bytes=expert_bytes,
                preferred_device=preferred,
            )
            loads[dst] += max(0, expert_bytes)
            assigned[int(expert_id)] = dst
            packets.append(
                ExpertPacket(
                    src_device=int(src_device),
                    dst_device=int(dst),
                    layer=int(layer),
                    expert_id=int(expert_id),
                    token_ids=token_ids,
                    expert_bytes=int(expert_bytes),
                )
            )

        cross = [
            packet for packet in packets if packet.src_device != packet.dst_device
        ]
        plan = RoutingPlan(
            packets=packets,
            cross_device_packets=len(cross),
            cross_device_bytes=sum(packet.transfer_bytes for packet in cross),
            assigned_experts=assigned,
            queue_depths={
                device: float(self.load_balancer.states[device].queue_depth)
                for device in self.devices
            },
            hot_replications=int(hot_copies),
        )
        plan.by_device = plan.device_batches()
        self.controller.metrics["cross_device_packets"] += float(len(cross))
        self.controller.metrics["cross_device_bytes"] += float(
            plan.cross_device_bytes
        )
        return plan


class WorkerFailure(RuntimeError):
    """A destination worker failed during packet exchange."""

    def __init__(self, device: int) -> None:
        self.device = int(device)
        super().__init__(f"ECS worker device {self.device} failed")


class SimulatedAllToAll:
    """Deterministic in-process all-to-all used by tests and CPU simulation."""

    def __init__(
        self,
        devices: Iterable[int],
        failed_devices: Iterable[int] = (),
    ) -> None:
        self.devices = tuple(int(device) for device in devices)
        self.failed_devices = {int(device) for device in failed_devices}
        self.received: dict[int, list[ExpertPacket]] = {
            device: [] for device in self.devices
        }

    def set_failed(self, *devices: int) -> None:
        self.failed_devices = {int(device) for device in devices}

    def exchange(
        self, outbound: dict[int, list[ExpertPacket]]
    ) -> dict[int, list[ExpertPacket]]:
        self.received = {device: [] for device in self.devices}
        for source in self.devices:
            for packet in outbound.get(source, []):
                if packet.dst_device not in self.received:
                    raise ValueError(f"unknown destination device {packet.dst_device}")
                if packet.dst_device in self.failed_devices:
                    raise WorkerFailure(packet.dst_device)
                if packet.dst_device != source:
                    self.received[packet.dst_device].append(packet)
                else:
                    self.received[source].append(packet)
        return self.received


class ResilientAllToAll:
    """Retry/reroute wrapper for a simulated or real worker transport."""

    def __init__(
        self,
        backend: Any,
        *,
        fallback_device: int = 0,
        max_retries: int = 2,
    ) -> None:
        self.backend = backend
        self.fallback_device = int(fallback_device)
        self.max_retries = max(1, int(max_retries))
        self.failed_devices: set[int] = set()
        self.metrics = {
            "retries": 0,
            "failovers": 0,
            "failed_packets": 0,
        }

    def exchange(
        self, outbound: dict[int, list[ExpertPacket]]
    ) -> dict[int, list[ExpertPacket]]:
        current = {source: list(packets) for source, packets in outbound.items()}
        last_error: Exception | None = None
        for _ in range(self.max_retries):
            try:
                return self.backend.exchange(current)
            except WorkerFailure as exc:
                last_error = exc
                self.failed_devices.add(exc.device)
                self.metrics["retries"] += 1
                rerouted = 0
                for source, packets in list(current.items()):
                    updated = []
                    for packet in packets:
                        if packet.dst_device in self.failed_devices:
                            if packet.dst_device == self.fallback_device:
                                continue
                            from dataclasses import replace

                            updated.append(
                                replace(packet, dst_device=self.fallback_device)
                            )
                            rerouted += 1
                        else:
                            updated.append(packet)
                    current[source] = updated
                self.metrics["failed_packets"] += rerouted
                self.metrics["failovers"] += 1 if rerouted else 0
                if not rerouted:
                    raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("ECS retry loop did not execute")


class TorchCollectiveAllToAll:
    """Optional torch.distributed packet exchange.

    ``all_to_all_object`` is available in recent PyTorch builds.  When it is
    absent we fall back to ``all_gather_object`` and filter by destination;
    that path is correct but less bandwidth-efficient.
    """

    def __init__(self, process_group: Any = None) -> None:
        import torch.distributed as dist

        if not dist.is_available():
            raise RuntimeError("torch.distributed is unavailable")
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized first")
        self.dist = dist
        self.process_group = process_group
        self.rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)

    def exchange(
        self, outbound: list[ExpertPacket] | dict[int, list[ExpertPacket]]
    ) -> list[ExpertPacket]:
        if isinstance(outbound, list):
            by_dst: dict[int, list[ExpertPacket]] = {}
            for packet in outbound:
                by_dst.setdefault(packet.dst_device, []).append(packet)
        else:
            by_dst = outbound
        send = [by_dst.get(rank, []) for rank in range(self.world_size)]
        if hasattr(self.dist, "all_to_all_object"):
            received: list[list[ExpertPacket]] = [None] * self.world_size  # type: ignore[list-item]
            self.dist.all_to_all_object(
                received, send, group=self.process_group
            )
            return [packet for packets in received for packet in packets]
        gathered: list[list[ExpertPacket]] = [None] * self.world_size  # type: ignore[list-item]
        self.dist.all_gather_object(gathered, send, group=self.process_group)
        rank = self.rank
        return [
            packet
            for per_rank in gathered
            for packets in per_rank
            for packet in packets
            if packet.dst_device == rank
        ]


def encode_packet(packet: ExpertPacket) -> bytes:
    header = struct.pack(
        "<iiiqq",
        int(packet.src_device),
        int(packet.dst_device),
        int(packet.layer),
        int(packet.expert_id),
        int(len(packet.token_ids)),
    )
    token_blob = struct.pack(f"<{len(packet.token_ids)}q", *packet.token_ids)
    return header + token_blob + packet.payload


def decode_packet(data: bytes, *, expert_bytes: int = 0) -> ExpertPacket:
    if len(data) < 28:
        raise ValueError("packet is too short")
    src, dst, layer, expert_id, count = struct.unpack_from("<iiiqq", data, 0)
    offset = 28
    token_ids = struct.unpack_from(f"<{count}q", data, offset)
    offset += 8 * count
    return ExpertPacket(
        src_device=int(src),
        dst_device=int(dst),
        layer=int(layer),
        expert_id=int(expert_id),
        token_ids=tuple(int(token) for token in token_ids),
        expert_bytes=int(expert_bytes),
        payload=data[offset:],
    )


__all__ = [
    "ECSAllToAll",
    "ECSLoadBalancer",
    "ECSRouter",
    "ExpertPacket",
    "ResilientAllToAll",
    "RoutingPlan",
    "SimulatedAllToAll",
    "WorkerFailure",
    "TorchCollectiveAllToAll",
    "decode_packet",
    "encode_packet",
]

# Backwards-friendly alias for callers that prefer the full name.
ECSAllToAll = SimulatedAllToAll
