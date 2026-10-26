"""SM86 W4A16 backend for the DeepInfer Qwen2 model executor.

This module is the first native DeepInfer SM86 execution path for Qwen2:
dense Qwen2 weights are converted offline/in-memory to grouped symmetric int4
weights, and single-token decode uses a custom CUDA GEMV kernel compiled with
CuPy/NVRTC.  Prefill uses a per-layer dequantize + cuBLAS path until a tiled
W4A16 GEMM kernel is added.

The attention and sampling boundaries are unchanged from
``Qwen2ForCausalLM``; this is a model_executor-level backend, not a claim that
the DeepSeek SM100 engine/cache stack is complete.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from deepinfer.model_executor.models.qwen2 import Qwen2ForCausalLM

try:
    import cupy as cp
except ImportError:  # pragma: no cover - only the SM86 backend needs CuPy
    cp = None


_SM86_W4A16_KERNEL = r"""
#include <cuda_fp16.h>

__device__ __forceinline__ float int4_value(unsigned int value) {
    int v = (int)(value & 0x0f);
    if (v >= 8) v -= 16;
    return (float)v;
}

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

extern "C" __global__ void qwen2_w4a16_gemv(
    const __half* __restrict__ x,
    const unsigned char* __restrict__ packed,
    const __half* __restrict__ scales,
    __half* __restrict__ out,
    int n_rows,
    int k_size,
    int group_size)
{
    extern __shared__ __half shared_x[];
    for (int i = threadIdx.x; i < k_size; i += blockDim.x) {
        shared_x[i] = x[i];
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int row = blockIdx.x * warps_per_block + warp;
    if (row >= n_rows) return;

    const int groups_per_row = k_size / group_size;
    const unsigned char* row_packed = packed + (size_t)row * (k_size / 2);
    const __half* row_scales = scales + (size_t)row * groups_per_row;
    const int vec = 8;
    const int stride = 32 * vec;
    float acc = 0.0f;

    for (int k0 = lane * vec; k0 < k_size; k0 += stride) {
        const unsigned int word = *reinterpret_cast<const unsigned int*>(
            row_packed + (size_t)k0 / 2
        );
        const float scale = __half2float(row_scales[k0 / group_size]);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int kk = k0 + j * 2;
            acc += __half2float(shared_x[kk]) * scale * int4_value((word >> (j * 8)) & 0x0f);
            acc += __half2float(shared_x[kk + 1]) * scale * int4_value((word >> (j * 8 + 4)) & 0x0f);
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) out[row] = __float2half_rn(acc);
}
"""

_SM86_GQA_DECODE_KERNEL = r"""
#include <cuda_fp16.h>

#define QWEN2_GQA_MAX_SEQ 128

extern "C" __global__ void qwen2_gqa_decode(
    const __half* __restrict__ query,
    const __half* __restrict__ key_cache,
    const __half* __restrict__ value_cache,
    const unsigned char* __restrict__ mask,
    __half* __restrict__ output,
    int num_heads,
    int num_kv_heads,
    int max_seq,
    int head_dim,
    float scale)
{
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    if (head >= num_heads || max_seq > QWEN2_GQA_MAX_SEQ) return;

    const int kv_group = num_heads / num_kv_heads;
    const int kv_head = head / kv_group;
    const __half* q = query + (size_t)head * head_dim;
    const __half* k = key_cache + (size_t)kv_head * max_seq * head_dim;
    const __half* v = value_cache + (size_t)kv_head * max_seq * head_dim;

    __shared__ float scores[QWEN2_GQA_MAX_SEQ];
    __shared__ float probs[QWEN2_GQA_MAX_SEQ];
    __shared__ float reduce_buf[QWEN2_GQA_MAX_SEQ];

    const bool valid = (tid < max_seq) && (mask[tid] != 0);
    float dot = 0.0f;
    if (valid) {
        const __half* kt = k + (size_t)tid * head_dim;
        #pragma unroll 4
        for (int d = 0; d < head_dim; ++d) {
            dot += __half2float(q[d]) * __half2float(kt[d]);
        }
    }
    scores[tid] = valid ? dot * scale : -1.0e30f;
    reduce_buf[tid] = scores[tid];
    __syncthreads();

    for (int stride = QWEN2_GQA_MAX_SEQ / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_buf[tid] = fmaxf(reduce_buf[tid], reduce_buf[tid + stride]);
        }
        __syncthreads();
    }
    const float max_score = reduce_buf[0];
    const float probability = valid ? __expf(scores[tid] - max_score) : 0.0f;
    probs[tid] = probability;
    reduce_buf[tid] = probability;
    __syncthreads();

    for (int stride = QWEN2_GQA_MAX_SEQ / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_buf[tid] += reduce_buf[tid + stride];
        }
        __syncthreads();
    }
    const float inv_sum = 1.0f / reduce_buf[0];

    if (tid < head_dim) {
        float acc = 0.0f;
        for (int token = 0; token < max_seq; ++token) {
            if (mask[token] != 0) {
                acc += probs[token] * inv_sum * __half2float(v[(size_t)token * head_dim + tid]);
            }
        }
        output[(size_t)head * head_dim + tid] = __float2half_rn(acc);
    }
}
"""

_SM86_GQA_DECODE_KERNEL_256 = _SM86_GQA_DECODE_KERNEL.replace(
    "QWEN2_GQA_MAX_SEQ 128", "QWEN2_GQA_MAX_SEQ 256"
).replace("qwen2_gqa_decode", "qwen2_gqa_decode_256")

_RAW_KERNEL: Any = None
_GQA_KERNEL: Any = None
_GQA_KERNEL_256: Any = None
_BLOCK_SIZE = 256
_ROWS_PER_WARP = 1


def _torch_to_cupy(tensor: torch.Tensor) -> Any:
    """Zero-copy CuPy view without DLPack stream handshakes.

    DLPack calls Tensor.__dlpack__(), which synchronizes/wait-events PyTorch
    streams.  That is illegal during CUDA Graph capture.  A UnownedMemory view
    is safe here because the owning PyTorch tensor stays alive in the caller.
    """

    if cp is None:
        raise RuntimeError("CuPy is unavailable")
    if tensor.device.type != "cuda":
        raise ValueError(f"expected a CUDA tensor, got {tensor.device}")
    contiguous = tensor.contiguous()
    dtype = {
        torch.float16: cp.float16,
        torch.float32: cp.float32,
        torch.int64: cp.int64,
        torch.int32: cp.int32,
        torch.uint8: cp.uint8,
        torch.bool: cp.bool_,
    }.get(contiguous.dtype)
    if dtype is None:
        raise TypeError(f"unsupported CuPy bridge dtype: {contiguous.dtype}")
    memory = cp.cuda.UnownedMemory(
        contiguous.data_ptr(),
        contiguous.numel() * contiguous.element_size(),
        owner=contiguous,
        device_id=contiguous.device.index or 0,
    )
    return cp.ndarray(
        contiguous.shape, dtype=dtype, memptr=cp.cuda.MemoryPointer(memory, 0)
    )


def _raw_kernel() -> Any:
    global _RAW_KERNEL
    if cp is None:
        raise RuntimeError(
            "CuPy is required for the DeepInfer Qwen2 SM86 W4A16 backend; install cupy-cuda12x[ctk]"
        )
    if _RAW_KERNEL is None:
        _RAW_KERNEL = cp.RawKernel(
            _SM86_W4A16_KERNEL,
            "qwen2_w4a16_gemv",
            options=("--std=c++14", "--use_fast_math"),
        )
    return _RAW_KERNEL


def _gqa_kernel(max_seq: int) -> Any:
    global _GQA_KERNEL, _GQA_KERNEL_256
    if cp is None:
        raise RuntimeError("CuPy is required for the DeepInfer Qwen2 SM86 GQA kernel")
    if max_seq > 256:
        raise ValueError(f"native GQA decode supports max_seq <= 256, got {max_seq}")
    if max_seq > 128:
        if _GQA_KERNEL_256 is None:
            _GQA_KERNEL_256 = cp.RawKernel(
                _SM86_GQA_DECODE_KERNEL_256,
                "qwen2_gqa_decode_256",
                options=("--std=c++14", "--use_fast_math"),
            )
        return _GQA_KERNEL_256
    if _GQA_KERNEL is None:
        _GQA_KERNEL = cp.RawKernel(
            _SM86_GQA_DECODE_KERNEL,
            "qwen2_gqa_decode",
            options=("--std=c++14", "--use_fast_math"),
        )
    return _GQA_KERNEL


def _native_gqa_decode(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Decode one GQA query token against a static up-to-256-slot KV cache."""

    if cp is None:
        raise RuntimeError("CuPy is unavailable")
    if query.shape[0] != 1 or query.shape[2] != 1:
        raise ValueError(
            f"native GQA decode expects [1, heads, 1, dim], got {tuple(query.shape)}"
        )
    if attention_mask is None:
        raise ValueError("native GQA decode requires an attention mask")
    max_seq = int(attention_mask.shape[1])
    if max_seq > 256:
        raise ValueError(f"native GQA decode supports max_seq <= 256, got {max_seq}")
    num_heads = int(query.shape[1])
    head_dim = int(query.shape[3])
    if head_dim > 128:
        raise ValueError(f"native GQA decode supports head_dim <= 128, got {head_dim}")

    q = query.contiguous().to(torch.float16)
    k = key_cache.contiguous().to(torch.float16)
    v = value_cache.contiguous().to(torch.float16)
    mask = attention_mask.contiguous().to(torch.bool)
    output = torch.empty(
        (num_heads, head_dim), dtype=torch.float16, device=query.device
    )
    block_size = 256 if max_seq > 128 else 128
    stream = cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)
    with stream:
        _gqa_kernel(max_seq)(
            (num_heads,),
            (block_size,),
            (
                _torch_to_cupy(q),
                _torch_to_cupy(k),
                _torch_to_cupy(v),
                _torch_to_cupy(mask),
                _torch_to_cupy(output),
                num_heads,
                int(k.shape[1]),
                max_seq,
                head_dim,
                cp.float32(1.0 / (head_dim**0.5)),
            ),
        )
    return output.view(1, num_heads, 1, head_dim)


class W4A16Linear(nn.Module):
    """Grouped symmetric int4 weight + FP16 activation linear layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        *,
        group_size: int = 128,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str | None = None,
        prefill_chunk_rows: int | None = None,
        gemv_block_size: int = 256,
    ) -> None:
        super().__init__()
        self.prefill_chunk_rows = prefill_chunk_rows
        self.gemv_block_size = gemv_block_size
        if in_features % group_size:
            raise ValueError(
                f"in_features={in_features} must be divisible by group_size={group_size}"
            )
        if in_features % 8:
            raise ValueError(f"in_features={in_features} must be divisible by 8")
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self._packed_cp = None
        self._scales_cp = None
        # torch.get_default_device() does not follow a ``with torch.device``
        # context; allocating a zero-size tensor does.
        target_device = device or torch.empty(0).device
        self.register_buffer(
            "packed",
            torch.empty(
                (out_features, in_features // 2),
                dtype=torch.uint8,
                device=target_device,
            ),
        )
        self.register_buffer(
            "scales",
            torch.empty(
                (out_features, in_features // group_size),
                dtype=torch.float16,
                device=target_device,
            ),
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, dtype=dtype, device=target_device)
            )
        else:
            self.register_parameter("bias", None)

    def _apply(self, fn: Any, recurse: bool = True) -> Any:
        result = super()._apply(fn, recurse)
        self._packed_cp = None
        self._scales_cp = None
        return result

    def _bind_cupy(self) -> None:
        if cp is None:
            raise RuntimeError("CuPy is unavailable")
        self._packed_cp = _torch_to_cupy(self.packed)
        self._scales_cp = _torch_to_cupy(self.scales)

    def load_from_linear(self, source: nn.Linear) -> None:
        """Quantize a PEFT-merged dense linear layer into this module."""
        if tuple(source.weight.shape) != (self.out_features, self.in_features):
            raise ValueError(
                f"linear shape mismatch: target={(self.out_features, self.in_features)}, source={tuple(source.weight.shape)}"
            )
        with torch.no_grad():
            weight = source.weight.detach().to(
                device=self.packed.device, dtype=torch.float32
            )
            grouped = weight.view(
                self.out_features, self.in_features // self.group_size, self.group_size
            )
            scale = (grouped.abs().amax(dim=-1).clamp_min(1e-8) / 7.0).to(torch.float16)
            quantized = torch.clamp(
                torch.round(grouped / scale.float()[:, :, None]), -8, 7
            ).to(torch.int8)
            quantized = quantized.view(self.out_features, self.in_features)
            low = (quantized[:, 0::2] & 0x0F).to(torch.uint8)
            high = ((quantized[:, 1::2] & 0x0F) << 4).to(torch.uint8)
            self.packed.copy_(low | high)
            self.scales.copy_(scale)
            if self.bias is not None and source.bias is not None:
                self.bias.copy_(
                    source.bias.detach().to(
                        device=self.bias.device, dtype=self.bias.dtype
                    )
                )
            if cp is not None:
                self._bind_cupy()
            del weight, grouped, scale, quantized, low, high

    def load_from_packed(
        self,
        packed: torch.Tensor,
        scales: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> None:
        if tuple(packed.shape) != tuple(self.packed.shape):
            raise ValueError(
                f"packed shape mismatch: {tuple(packed.shape)} vs {tuple(self.packed.shape)}"
            )
        if tuple(scales.shape) != tuple(self.scales.shape):
            raise ValueError(
                f"scale shape mismatch: {tuple(scales.shape)} vs {tuple(self.scales.shape)}"
            )
        with torch.no_grad():
            self.packed.copy_(packed.to(device=self.packed.device, dtype=torch.uint8))
            self.scales.copy_(scales.to(device=self.scales.device, dtype=torch.float16))
            if self.bias is not None and bias is not None:
                self.bias.copy_(bias.to(device=self.bias.device, dtype=self.bias.dtype))
            if cp is not None:
                self._bind_cupy()

    def _dequantize_rows(self, start: int, end: int) -> torch.Tensor:
        rows = end - start
        packed = self.packed[start:end]
        scales = self.scales[start:end]
        low = (packed & 0x0F).to(torch.int8)
        high = ((packed >> 4) & 0x0F).to(torch.int8)
        low = torch.where(low >= 8, low - 16, low)
        high = torch.where(high >= 8, high - 16, high)
        quantized = torch.empty(
            (rows, self.in_features),
            dtype=torch.float16,
            device=self.packed.device,
        )
        quantized[:, 0::2] = low
        quantized[:, 1::2] = high
        grouped = quantized.view(
            rows,
            self.in_features // self.group_size,
            self.group_size,
        )
        return (grouped * scales[:, :, None]).view(rows, self.in_features)

    def _dequantize(self) -> torch.Tensor:
        return self._dequantize_rows(0, self.out_features)

    def _forward_gemv(self, inputs: torch.Tensor) -> torch.Tensor:
        if cp is None:
            raise RuntimeError("CuPy is unavailable")
        if self._packed_cp is None or self._scales_cp is None:
            self._bind_cupy()
        flat = inputs.reshape(1, self.in_features).contiguous()
        flat = flat.to(torch.float16)
        output = torch.empty(
            (1, self.out_features), dtype=torch.float16, device=flat.device
        )
        input_cp = _torch_to_cupy(flat)
        output_cp = _torch_to_cupy(output)
        kernel = _raw_kernel()
        rows_per_block = self.gemv_block_size // 32
        grid = ((self.out_features + rows_per_block - 1) // rows_per_block,)
        stream = cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)
        with stream:
            kernel(
                grid,
                (self.gemv_block_size,),
                (
                    input_cp,
                    self._packed_cp,
                    self._scales_cp,
                    output_cp,
                    self.out_features,
                    self.in_features,
                    self.group_size,
                ),
                shared_mem=self.in_features * 2,
            )
        if self.bias is not None:
            output.add_(self.bias)
        return output

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        original_shape = inputs.shape
        flat = inputs.reshape(-1, self.in_features)
        if flat.shape[0] == 1:
            output = self._forward_gemv(flat)
        else:
            chunk_rows = self.prefill_chunk_rows or self.out_features
            parts: list[torch.Tensor] = []
            for start in range(0, self.out_features, chunk_rows):
                end = min(start + chunk_rows, self.out_features)
                weight = self._dequantize_rows(start, end).to(flat.dtype)
                parts.append(F.linear(flat, weight, None))
                del weight
            output = torch.cat(parts, dim=-1)
            del parts
            if self.bias is not None:
                output = output + self.bias.to(flat.dtype)
        return output.reshape(*original_shape[:-1], self.out_features)


class Qwen2W4A16ForCausalLM(Qwen2ForCausalLM):
    """Qwen2 model_executor model with SM86 W4A16 linears.

    Single-token decode uses the native SM86 W4A16 GEMV and native SM86 GQA
    kernels.  The model still runs at model_executor level: the DeepSeek
    CacheEngine/Sampler and complete engine are not wired.
    """

    source_backend = "qwen2_w4a16_sm86_model_executor"
    native_w4a16_kernel = True
    native_attention_kernel = True
    native_runner_ready = False

    def __init__(
        self,
        cfg: Any,
        *,
        linear_factory: Any,
        fused_qkv_factory: Any | None = None,
        fused_gate_up_factory: Any | None = None,
        group_size: int = 128,
    ) -> None:
        self.group_size = group_size
        super().__init__(
            cfg,
            linear_factory=linear_factory,
            fused_qkv_factory=fused_qkv_factory,
            fused_gate_up_factory=fused_gate_up_factory,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        *,
        device: str | torch.device = "cuda",
        group_size: int = 128,
    ) -> Qwen2W4A16ForCausalLM:
        """Load a layered W4A16 checkpoint without materializing dense weights."""
        from safetensors import safe_open

        from deepinfer.model_spec import build_model_config

        root = Path(checkpoint_dir)
        hf = json.loads((root / "config.json").read_text(encoding="utf-8"))
        cfg = build_model_config(hf, root / "config.json")
        cfg = cfg.gpt if hasattr(cfg, "gpt") else cfg
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float16)
        try:
            with torch.device(device):
                model = cls(
                    cfg,
                    group_size=group_size,
                    linear_factory=lambda in_features, out_features, bias: W4A16Linear(
                        in_features,
                        out_features,
                        bias,
                        group_size=group_size,
                        dtype=torch.float16,
                    ),
                    fused_qkv_factory=lambda in_features, out_features, bias: (
                        W4A16Linear(
                            in_features,
                            out_features,
                            bias,
                            group_size=group_size,
                            dtype=torch.float16,
                        )
                    ),
                    fused_gate_up_factory=lambda in_features, out_features, bias: (
                        W4A16Linear(
                            in_features,
                            out_features,
                            bias,
                            group_size=group_size,
                            dtype=torch.float16,
                        )
                    ),
                )
        finally:
            torch.set_default_dtype(previous_dtype)
        model.requires_grad_(False)
        with safe_open(
            root / "embed.safetensors", framework="pt", device="cpu"
        ) as tensors:
            model.model.embed_tokens.weight.copy_(tensors.get_tensor("weight"))
        with safe_open(
            root / "norm.safetensors", framework="pt", device="cpu"
        ) as tensors:
            model.model.norm.weight.copy_(tensors.get_tensor("weight"))
        if not bool(cfg.get("tie_word_embeddings", False)):
            with safe_open(
                root / "lm_head.safetensors", framework="pt", device="cpu"
            ) as tensors:
                model.lm_head.weight.copy_(tensors.get_tensor("weight"))
        for layer_index, layer_file in enumerate(manifest["layer_files"]):
            layer_root = f"model.layers.{layer_index}"
            layer = model.get_submodule(layer_root)
            with (
                safe_open(root / layer_file, framework="pt", device="cpu") as tensors,
                torch.no_grad(),
            ):
                attention = layer.self_attn
                qkv = attention.fused_qkv
                q_rows = attention.num_heads * attention.head_dim
                kv_rows = attention.num_kv_heads * attention.head_dim
                qkv.packed[:q_rows].copy_(tensors.get_tensor("self_attn.q_proj.packed"))
                qkv.packed[q_rows : q_rows + kv_rows].copy_(
                    tensors.get_tensor("self_attn.k_proj.packed")
                )
                qkv.packed[q_rows + kv_rows :].copy_(
                    tensors.get_tensor("self_attn.v_proj.packed")
                )
                qkv.scales[:q_rows].copy_(tensors.get_tensor("self_attn.q_proj.scales"))
                qkv.scales[q_rows : q_rows + kv_rows].copy_(
                    tensors.get_tensor("self_attn.k_proj.scales")
                )
                qkv.scales[q_rows + kv_rows :].copy_(
                    tensors.get_tensor("self_attn.v_proj.scales")
                )
                if qkv.bias is not None:
                    qkv.bias[:q_rows].copy_(tensors.get_tensor("self_attn.q_proj.bias"))
                    qkv.bias[q_rows : q_rows + kv_rows].copy_(
                        tensors.get_tensor("self_attn.k_proj.bias")
                    )
                    qkv.bias[q_rows + kv_rows :].copy_(
                        tensors.get_tensor("self_attn.v_proj.bias")
                    )
                attention.o_proj.load_from_packed(
                    tensors.get_tensor("self_attn.o_proj.packed"),
                    tensors.get_tensor("self_attn.o_proj.scales"),
                )
                mlp = layer.mlp
                gate_up = mlp.fused_gate_up
                gate_rows = int(cfg.intermediate_size)
                gate_up.packed[:gate_rows].copy_(
                    tensors.get_tensor("mlp.gate_proj.packed")
                )
                gate_up.packed[gate_rows:].copy_(
                    tensors.get_tensor("mlp.up_proj.packed")
                )
                gate_up.scales[:gate_rows].copy_(
                    tensors.get_tensor("mlp.gate_proj.scales")
                )
                gate_up.scales[gate_rows:].copy_(
                    tensors.get_tensor("mlp.up_proj.scales")
                )
                mlp.down_proj.load_from_packed(
                    tensors.get_tensor("mlp.down_proj.packed"),
                    tensors.get_tensor("mlp.down_proj.scales"),
                )
                layer.input_layernorm.weight.copy_(
                    tensors.get_tensor("input_layernorm.weight")
                )
                layer.post_attention_layernorm.weight.copy_(
                    tensors.get_tensor("post_attention_layernorm.weight")
                )
            torch.cuda.empty_cache()
            print(
                f"loaded layer {layer_index + 1}/{len(manifest['layer_files'])}",
                flush=True,
            )
        for layer in model.model.layers:
            layer.self_attn.native_gqa_decode = _native_gqa_decode
            layer.self_attn.fused_qkv._bind_cupy()
            layer.mlp.fused_gate_up._bind_cupy()
        model.eval().requires_grad_(False)
        torch.cuda.empty_cache()
        gc.collect()
        return model

    @classmethod
    def from_dense_model(
        cls,
        dense_model: Qwen2ForCausalLM,
        *,
        device: str | torch.device = "cuda",
        group_size: int = 128,
    ) -> Qwen2W4A16ForCausalLM:
        cfg = dense_model.cfg
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float16)
        try:
            with torch.device(device):
                model = cls(
                    cfg,
                    group_size=group_size,
                    linear_factory=lambda in_features, out_features, bias: W4A16Linear(
                        in_features,
                        out_features,
                        bias,
                        group_size=group_size,
                        dtype=torch.float16,
                    ),
                )
        finally:
            torch.set_default_dtype(previous_dtype)
        model.requires_grad_(False)

        source_modules = dict(dense_model.named_modules())
        target_modules = dict(model.named_modules())
        with torch.no_grad():
            model.model.embed_tokens.weight.copy_(
                dense_model.model.embed_tokens.weight.to(
                    device=model.model.embed_tokens.weight.device,
                    dtype=model.model.embed_tokens.weight.dtype,
                )
            )
            model.model.norm.weight.copy_(
                dense_model.model.norm.weight.to(model.model.norm.weight)
            )
            for index, layer in enumerate(model.model.layers):
                source_layer = dense_model.model.layers[index]
                layer.input_layernorm.weight.copy_(
                    source_layer.input_layernorm.weight.to(layer.input_layernorm.weight)
                )
                layer.post_attention_layernorm.weight.copy_(
                    source_layer.post_attention_layernorm.weight.to(
                        layer.post_attention_layernorm.weight
                    )
                )
            for name, module in target_modules.items():
                if isinstance(module, W4A16Linear):
                    module.load_from_linear(source_modules[name])
        if bool(cfg.get("tie_word_embeddings", False)):
            model.lm_head.weight = model.model.embed_tokens.weight
        else:
            model.lm_head.weight.copy_(
                dense_model.lm_head.weight.to(model.lm_head.weight)
            )
        for layer in model.model.layers:
            layer.self_attn.native_gqa_decode = _native_gqa_decode
        model.eval().requires_grad_(False)
        torch.cuda.empty_cache()
        gc.collect()
        return model

    @classmethod
    def load_model(cls, *args: Any, **kwargs: Any) -> nn.Module:
        raise NotImplementedError(
            "Use Qwen2W4A16ForCausalLM.from_dense_model(...) with the PEFT-merged dense model from Qwen2ForCausalLM.load_model(...)"
        )


__all__ = [
    "Qwen2W4A16ForCausalLM",
    "W4A16Linear",
]
