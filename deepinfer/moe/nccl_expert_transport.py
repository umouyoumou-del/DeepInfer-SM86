"""NCCL transport for expert-sharded activation exchange.

The coordinator keeps attention, shared experts and the LM head.  A second rank
owns the remote routed-expert subset and executes the same grouped gate/up/down
kernels on its own GPU.  Unlike the first dual-GPU prototype, the remote rank
does not load the transformer model or duplicate non-routed weights.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.distributed as dist

from deepinfer.moe.deepseek_v2_runtime import GGUFExpertStore
from deepinfer.moe.gguf_index import GGUFIndex
from deepinfer.moe.grouped_moe import (
    fused_down_gemm_grouped,
    fused_gate_up_gemm_grouped,
)

_HEADER_BYTES = 1024
_DTYPE_NAMES = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}
_NAME_DTYPES = {value: key for key, value in _DTYPE_NAMES.items()}


def _pack_header(payload: dict[str, Any]) -> torch.Tensor:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    if len(encoded) > _HEADER_BYTES:
        raise ValueError(f"expert transport header too large: {len(encoded)} bytes")
    encoded += b" " * (_HEADER_BYTES - len(encoded))
    return torch.tensor(list(encoded), dtype=torch.uint8)


def _unpack_header(tensor: torch.Tensor) -> dict[str, Any]:
    data = bytes(int(value) for value in tensor.cpu().tolist())
    return json.loads(data.rstrip(b" \x00").decode("utf-8"))


class NCCLExpertClient:
    """Coordinator-side blocking request/reply client."""

    def __init__(self, *, dst_rank: int = 1, device: int = 0) -> None:
        self.dst_rank = int(dst_rank)
        self.device = int(device)
        self.requests = 0
        self.bytes_sent = 0
        self.bytes_received = 0
        self.latency_seconds = 0.0

    def compute_chunk(
        self,
        layer_idx: int,
        expert_ids: Sequence[int],
        tokens: torch.Tensor,
        counts: Sequence[int],
    ) -> torch.Tensor:
        tokens = tokens.contiguous()
        if not tokens.is_cuda:
            raise ValueError("NCCL expert transport requires CUDA tokens")
        dtype_name = _DTYPE_NAMES.get(tokens.dtype)
        if dtype_name is None:
            raise TypeError(f"unsupported expert activation dtype {tokens.dtype}")
        shape = tuple(int(value) for value in tokens.shape)
        header = _pack_header(
            {
                "stop": False,
                "layer": int(layer_idx),
                "experts": [int(value) for value in expert_ids],
                "counts": [int(value) for value in counts],
                "shape": list(shape),
                "dtype": dtype_name,
            }
        ).to(tokens.device)
        if os.environ.get("DEEPINFER_NCCL_DEBUG") == "1":
            print(
                f"[nccl-client] send layer={layer_idx} experts={len(expert_ids)} "
                f"shape={shape}",
                flush=True,
            )
        started = time.perf_counter()
        dist.send(header, dst=self.dst_rank)
        dist.send(tokens, dst=self.dst_rank)
        output = torch.empty(shape, dtype=tokens.dtype, device=tokens.device)
        dist.recv(output, src=self.dst_rank)
        if os.environ.get("DEEPINFER_NCCL_DEBUG") == "1":
            print(
                f"[nccl-client] recv layer={layer_idx} output={shape}",
                flush=True,
            )
        self.latency_seconds += time.perf_counter() - started
        self.requests += 1
        self.bytes_sent += int(tokens.numel() * tokens.element_size())
        self.bytes_received += int(output.numel() * output.element_size())
        return output

    def close(self) -> None:
        header = _pack_header({"stop": True}).to(f"cuda:{self.device}")
        dist.send(header, dst=self.dst_rank)

    def metrics(self) -> dict[str, float | int]:
        return {
            "requests": self.requests,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "latency_seconds": self.latency_seconds,
        }


class RemoteExpertWorker:
    """Second-rank routed-expert executor without a transformer model."""

    def __init__(
        self,
        gguf_path: str | Path,
        *,
        layer_indices: Sequence[int],
        device: int = 1,
        expert_cache_mb: int = 0,
        gpu_expert_cache_mb: int = 0,
        max_gpu_spill_mb: int = 512,
        parallel_read_workers: int = 8,
    ) -> None:
        torch.cuda.set_device(int(device))
        try:
            import cupy

            cupy.cuda.Device(int(device)).use()
        except Exception:
            pass
        self.device = torch.device(f"cuda:{int(device)}")
        self.index = GGUFIndex(Path(gguf_path))
        self.store = GGUFExpertStore(
            self.index,
            layer_indices=tuple(int(layer) for layer in layer_indices),
            device=self.device,
            dtype=torch.float16,
            max_gpu_spill_mb=int(max_gpu_spill_mb),
            parallel_read_workers=int(parallel_read_workers),
            expert_cache_mb=int(expert_cache_mb),
            gpu_expert_cache_mb=int(gpu_expert_cache_mb),
        )
        self.requests = 0
        self.bytes_received = 0
        self.bytes_sent = 0

    def compute_chunk(
        self,
        layer_idx: int,
        expert_ids: Sequence[int],
        tokens: torch.Tensor,
        counts: Sequence[int],
    ) -> torch.Tensor:
        tokens = tokens.to(self.device)
        n_experts = len(expert_ids)
        max_count = int(tokens.shape[1])
        count_tensor = torch.tensor(
            [int(value) for value in counts],
            dtype=torch.int32,
            device=self.device,
        )
        batch = self.store.load_packed_experts(
            layer_idx,
            expert_ids,
            device=self.device,
        )
        gate_spec, up_spec, down_spec = batch.specs
        batch.wait_for_gate_up()
        gate_activation = fused_gate_up_gemm_grouped(
            tokens,
            batch.packed[0],
            batch.packed[1],
            count_tensor,
            n_experts=n_experts,
            max_count=max_count,
            out_features=gate_spec.out_features,
            k_size=gate_spec.in_features,
            gate_qtype=gate_spec.quant_type,
            up_qtype=up_spec.quant_type,
            gate_row_bytes=gate_spec.row_bytes,
            up_row_bytes=up_spec.row_bytes,
        )
        batch.wait_for_down()
        return fused_down_gemm_grouped(
            gate_activation,
            batch.packed[2],
            count_tensor,
            n_experts=n_experts,
            max_count=max_count,
            out_features=down_spec.out_features,
            k_size=down_spec.in_features,
            down_qtype=down_spec.quant_type,
            down_row_bytes=down_spec.row_bytes,
        )

    def serve_forever(self, *, src_rank: int = 0) -> None:
        device = self.device
        header = torch.empty(_HEADER_BYTES, dtype=torch.uint8, device=device)
        while True:
            if os.environ.get("DEEPINFER_NCCL_DEBUG") == "1":
                print("[nccl-worker] wait header", flush=True)
            dist.recv(header, src=src_rank)
            meta = _unpack_header(header)
            if os.environ.get("DEEPINFER_NCCL_DEBUG") == "1":
                print(
                    f"[nccl-worker] header stop={meta.get('stop')} "
                    f"layer={meta.get('layer')} experts={len(meta.get('experts', []))}",
                    flush=True,
                )
            if meta.get("stop"):
                break
            dtype = _NAME_DTYPES[str(meta["dtype"])]
            shape = tuple(int(value) for value in meta["shape"])
            tokens = torch.empty(shape, dtype=dtype, device=device)
            dist.recv(tokens, src=src_rank)
            self.bytes_received += int(tokens.numel() * tokens.element_size())
            output = self.compute_chunk(
                int(meta["layer"]),
                [int(value) for value in meta["experts"]],
                tokens,
                [int(value) for value in meta["counts"]],
            )
            if os.environ.get("DEEPINFER_NCCL_DEBUG") == "1":
                print(
                    f"[nccl-worker] send layer={meta['layer']} shape={tuple(output.shape)}",
                    flush=True,
                )
            dist.send(output.contiguous(), dst=src_rank)
            self.bytes_sent += int(output.numel() * output.element_size())
            self.requests += 1

    def close(self) -> None:
        self.store.close()

    def metrics(self) -> dict[str, float | int]:
        return {
            "requests": self.requests,
            "bytes_received": self.bytes_received,
            "bytes_sent": self.bytes_sent,
        }


__all__ = ["NCCLExpertClient", "RemoteExpertWorker"]
