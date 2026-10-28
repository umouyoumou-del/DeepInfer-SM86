"""Startup memory planning for the paged KV cache.

The scheduler needs a physical KV block count before it can admit requests.
Guessing that number on a 4 GB card is dangerous, so the count is derived from
the already-resident model, the current CUDA free-memory snapshot, an explicit
reserve, and a utilization ceiling.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch


class InsufficientKvMemoryError(RuntimeError):
    """Raised when no usable KV block fits into the configured budget."""

    def __init__(self, report: MemoryBudgetReport) -> None:
        self.report = report
        super().__init__(
            "insufficient memory for a paged KV pool: "
            f"block={report.block_bytes} bytes, "
            f"kv_budget={report.kv_budget_bytes} bytes, "
            f"free_before_pool={report.free_memory_bytes_before_pool} bytes, "
            "reduce max_cache/max_num_seqs or lower reserve_memory_mb"
        )


@dataclass(frozen=True, slots=True)
class MemoryBudgetReport:
    mode: str
    device: str
    total_memory_bytes: int
    free_memory_bytes_before_pool: int
    model_parameter_bytes: int
    model_buffer_bytes: int
    gpu_memory_utilization: float
    reserve_memory_bytes: int
    kv_budget_bytes: int
    block_bytes: int
    block_size: int
    num_blocks: int
    max_cache: int
    max_num_seqs: int
    prefix_cache_blocks: int
    capacity_tokens: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def model_memory_bytes(model: torch.nn.Module) -> tuple[int, int]:
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    buffer_bytes = sum(
        buffer.numel() * buffer.element_size() for buffer in model.buffers()
    )
    return int(parameter_bytes), int(buffer_bytes)


def plan_kv_blocks(
    model: torch.nn.Module,
    *,
    device: str | torch.device,
    block_bytes: int,
    block_size: int,
    max_cache: int,
    max_num_seqs: int,
    prefix_cache_blocks: int,
    requested_blocks: int | None = None,
    gpu_memory_utilization: float = 0.80,
    reserve_memory_mb: int = 512,
    kv_budget_mb: int | None = None,
) -> MemoryBudgetReport:
    """Return a conservative physical KV block count and its accounting."""
    if block_bytes < 1 or block_size < 1 or max_cache < 1:
        raise ValueError("block_bytes, block_size and max_cache must be positive")
    if not 0 < gpu_memory_utilization <= 0.98:
        raise ValueError("gpu_memory_utilization must be in (0, 0.98]")
    if reserve_memory_mb < 0:
        raise ValueError("reserve_memory_mb must be non-negative")

    blocks_per_seq = math.ceil(max_cache / block_size)
    required_capacity = max_num_seqs * blocks_per_seq + prefix_cache_blocks
    parameter_bytes, buffer_bytes = model_memory_bytes(model)
    reserve_bytes = int(reserve_memory_mb) * 1024**2
    target_device = torch.device(device)

    if target_device.type != "cuda":
        blocks = (
            required_capacity if requested_blocks is None else int(requested_blocks)
        )
        if blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        report = MemoryBudgetReport(
            mode=str(target_device.type),
            device=str(target_device),
            total_memory_bytes=0,
            free_memory_bytes_before_pool=0,
            model_parameter_bytes=parameter_bytes,
            model_buffer_bytes=buffer_bytes,
            gpu_memory_utilization=gpu_memory_utilization,
            reserve_memory_bytes=reserve_bytes,
            kv_budget_bytes=blocks * block_bytes,
            block_bytes=block_bytes,
            block_size=block_size,
            num_blocks=blocks,
            max_cache=max_cache,
            max_num_seqs=max_num_seqs,
            prefix_cache_blocks=prefix_cache_blocks,
            capacity_tokens=blocks * block_size,
        )
        return report

    total_memory = int(torch.cuda.get_device_properties(target_device).total_memory)
    free_memory, _ = torch.cuda.mem_get_info(target_device)
    if kv_budget_mb is not None:
        kv_budget = int(kv_budget_mb) * 1024**2
    else:
        resident_model = parameter_bytes + buffer_bytes
        utilization_ceiling = int(total_memory * gpu_memory_utilization)
        after_model = max(0, utilization_ceiling - resident_model)
        kv_budget = max(0, min(int(free_memory), after_model) - reserve_bytes)
    by_memory = kv_budget // block_bytes
    if requested_blocks is not None:
        num_blocks = int(requested_blocks)
    else:
        num_blocks = min(required_capacity, by_memory)
    num_blocks = max(0, min(num_blocks, required_capacity))
    report = MemoryBudgetReport(
        mode="cuda",
        device=str(target_device),
        total_memory_bytes=total_memory,
        free_memory_bytes_before_pool=int(free_memory),
        model_parameter_bytes=parameter_bytes,
        model_buffer_bytes=buffer_bytes,
        gpu_memory_utilization=gpu_memory_utilization,
        reserve_memory_bytes=reserve_bytes,
        kv_budget_bytes=int(kv_budget),
        block_bytes=block_bytes,
        block_size=block_size,
        num_blocks=num_blocks,
        max_cache=max_cache,
        max_num_seqs=max_num_seqs,
        prefix_cache_blocks=prefix_cache_blocks,
        capacity_tokens=num_blocks * block_size,
    )
    if num_blocks < 1:
        raise InsufficientKvMemoryError(report)
    return report


__all__ = [
    "InsufficientKvMemoryError",
    "MemoryBudgetReport",
    "model_memory_bytes",
    "plan_kv_blocks",
]
