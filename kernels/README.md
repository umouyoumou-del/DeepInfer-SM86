# SM86 CUDA Kernels

These `.cu` files are exported from the NVRTC source strings used by the
Python backend. The Python modules remain the source of truth for launch
arguments, kernel selection, and fallback conditions.

| File | Kernel(s) | Used by |
|---|---|---|
| `qwen2_w4a16_gemv.cu` | `qwen2_w4a16_gemv` | W4A16 batch=1 decode |
| `qwen2_w4a16_batched_gemv.cu` | `qwen2_w4a16_batched_gemv` | W4A16 decode batch 2..8, batch tile 8 |
| `qwen2_w4a16_tiled_prefill.cu` | `qwen2_w4a16_tiled_gemm` | W4A16 prefill `[M,K] @ [K,N]` |
| `qwen2_w3a16_gemv.cu` | `qwen2_w3a16_gemv`, `qwen2_w3a16_batched_gemv` | W3A16 decode; batched variant is experimental |
| `qwen2_w3a16_tiled_prefill.cu` | `qwen2_w3a16_tiled_gemm` | W3A16 prefill `[M,K] @ [K,N]` |
| `qwen2_gqa_decode.cu` | `qwen2_gqa_decode` | static dense KV decode, up to 128 slots |
| `qwen2_gqa_decode_256.cu` | `qwen2_gqa_decode_256` | static dense KV decode, up to 256 slots |
| `qwen2_paged_gqa_decode.cu` | `dp_paged_gqa_decode` | paged FP16 KV decode, ragged batch |
| `qwen2_paged_gqa_decode_int8.cu` | `dp_paged_gqa_decode_i8` | paged INT8 KV decode + FP16 scales |
| `qwen2_paged_gqa_decode_fp8.cu` | `dp_paged_gqa_decode_fp8` | paged FP8 E4M3/E5M2 KV decode |

## Batch support

- Static GQA decode kernels handle batch=1 only.
- Paged GQA decode kernels handle ragged batch sizes with per-row
  `valid_lengths`, GQA, sliding windows, and optional attention-logit softcap.
- W4A16 decode uses the shared-x GEMV at batch=1 and the batch-tile kernel at
  batch 2..8. The batch-tile kernel reuses one packed-weight read across up to
  8 tokens.
- W3A16 batch-tile GEMV is numerically correct but slower on the 4 GB RTX 3050
  because the larger accumulator tile reduces occupancy; it is not the default
  path.
- Prefill uses tiled GEMM for larger token counts; small batches may still use
  GEMV or dequantize + cuBLAS depending on backend and shape.

## Launch notes

### W4A16 GEMV

- batch=1 path: 256 threads per block, 8 rows per block, dynamic shared memory
  `K * sizeof(half)`, grid `ceil(N / 8)`;
- batch-tile path: 256 threads per block, one warp per output row, batch tile 8,
  grid `(ceil(N / 8), ceil(batch / 8))`, no dense weight tile.

### W3A16 GEMV

- gate/up: 512 threads, 16 rows per block;
- down: 768 threads, 24 rows per block;
- dynamic shared memory `K * sizeof(half)`;
- batched prototype uses a batch tile and is exported from the same source file.

### Tiled W4/W3 prefill GEMM

- CTA tile: `BM=32`, `BN=64`, `BK=32`;
- 256 threads per CTA;
- shared-memory quantized B tile and FP16 A tile;
- FP32 accumulation;
- W4 and W3 require `K % 32 == 0`; the Python wrappers validate shape before launch;
- current implementation is CUDA-core, not Tensor Core, and does not yet use
  `cp.async` or double buffering.

### Paged GQA decode

- grid `(num_query_heads, batch)`;
- one CUDA block per `(batch, query_head)`;
- block table is read directly; dense K/V is never materialized;
- FP16, INT8, and FP8 variants share the same softmax/layout contract;
- FP8 stores the byte bit pattern directly; the kernel decodes E4M3 or E5M2
  with a runtime `fp8_kind` flag and no scale sidecar;
- supports `head_dim <= 128` and `max_length <= 1024`;
- prefill, external attention masks, unsupported dtypes, and out-of-envelope
  shapes fall back to SDPA.

The files are generated with:

```powershell
python scripts\export_sm86_kernels.py
```

The exporter writes all sources with LF line endings.
