"""OLMoE experts driven by the pointer-grouped kernels on packed weights.

The fp16 reference path dequantises every expert, which costs 1.61 GB/token
against the 0.73 GB/token the Q4_K/Q6_K weights actually occupy.  This module
keeps them packed: each expert's bytes are a contiguous slab of the GGUF tensor
(verified in OLMOE_PORT_PLAN section 26), so a slab can be copied to the device
once and addressed by pointer with no gather and no dtype conversion.

Only the single-token decode shape is handled, where top-k experts each see the
same one token - that matches the kernel's (n_experts, max_count) layout with
max_count == 1 and every count equal to one.
"""

from __future__ import annotations

import numpy as np
import torch

from deepinfer.moe.ptr_grouped_moe import (
    fused_down_gemm_ptr_grouped,
    fused_gate_up_gemm_ptr_grouped,
)
from deepinfer.moe.olmoe_router import route_topk

Q4_K_ROW_BYTES_PER_256 = 144
Q6_K_ROW_BYTES_PER_256 = 210


def row_bytes(elements: int, qtype: str) -> int:
    per_block = Q4_K_ROW_BYTES_PER_256 if qtype == "Q4_K" else Q6_K_ROW_BYTES_PER_256
    if elements % 256:
        raise ValueError(f"{elements} elements is not a multiple of 256")
    return elements // 256 * per_block


def load_packed_experts(index, name: str, *, experts: int, out_features: int):
    """Copy every expert's packed slab to the device and build the pointer array.

    The quantisation type is read from the tensor itself: bartowski's OLMoE
    mixes types per layer (layer 0 keeps down in Q6_K, layer 2 in Q4_K), so
    hard-coding one is wrong.
    """

    info = index.tensor(name)
    qtype = info.qtype.name
    raw = np.asarray(index.read_raw(name))
    k_size = int(info.dims[0])
    stride = out_features * row_bytes(k_size, qtype)
    if raw.size != stride * experts:
        raise ValueError(
            f"{name}: {raw.size} bytes does not match {experts} x {stride}"
        )
    device = torch.device("cuda")
    storage = torch.from_numpy(np.ascontiguousarray(raw)).to(device)
    pointers = torch.tensor(
        [storage.data_ptr() + expert * stride for expert in range(experts)],
        dtype=torch.int64,
        device=device,
    )
    return storage, pointers, qtype


def moe_forward_packed(
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    packed: dict[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    top_k: int,
    gate_out: int,
    down_out: int,
    gate_k: int,
    down_k: int,
) -> torch.Tensor:
    """Weights stay packed; one pointer-grouped kernel per projection."""

    if hidden.shape[0] != 1:
        raise ValueError("packed path currently supports single-token decode only")
    expert_ids, weights = route_topk(hidden, gate_weight, top_k=top_k)
    ids = expert_ids[0]
    device = hidden.device

    x_all = hidden.to(torch.float16).repeat(top_k, 1).unsqueeze(1)  # [top_k, 1, k]
    counts = torch.ones(top_k, dtype=torch.int32, device=device)

    gate_storage, gate_ptrs, gate_qtype = packed["gate"]
    up_storage, up_ptrs, up_qtype = packed["up"]
    down_storage, down_ptrs, down_qtype = packed["down"]

    activation = fused_gate_up_gemm_ptr_grouped(
        x_all,
        gate_ptrs[ids].contiguous(),
        up_ptrs[ids].contiguous(),
        counts,
        n_experts=top_k,
        max_count=1,
        out_features=gate_out,
        k_size=gate_k,
        gate_qtype=gate_qtype,
        up_qtype=up_qtype,
        gate_row_bytes=row_bytes(gate_k, gate_qtype),
        up_row_bytes=row_bytes(gate_k, up_qtype),
    )
    expert_out = fused_down_gemm_ptr_grouped(
        activation,
        down_ptrs[ids].contiguous(),
        counts,
        n_experts=top_k,
        max_count=1,
        out_features=down_out,
        k_size=down_k,
        down_qtype=down_qtype,
        down_row_bytes=row_bytes(down_k, down_qtype),
    )
    flat = expert_out.reshape(top_k, down_out)
    return (flat * weights[0].unsqueeze(-1)).sum(dim=0, keepdim=True).to(hidden.dtype)
