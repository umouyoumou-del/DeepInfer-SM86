# SM86 CUDA Kernels

These `.cu` files are exported from the NVRTC source strings used by the
Python backend. The Python modules remain the source of truth for launch
arguments and kernel selection.

| File | Kernel | Used by |
|---|---|---|
| `qwen2_w4a16_gemv.cu` | `qwen2_w4a16_gemv` | 4-bit Qwen2 linear decode |
| `qwen2_w3a16_gemv.cu` | `qwen2_w3a16_gemv` | 3-bit MLP/lm_head decode |
| `qwen2_gqa_decode.cu` | `qwen2_gqa_decode` | single-token GQA decode, up to 128 slots |
| `qwen2_gqa_decode_256.cu` | `qwen2_gqa_decode_256` | single-token GQA decode, up to 256 slots |

## Launch notes

### W4A16 GEMV

- 256 threads per block;
- 8 rows per block (one warp per row);
- dynamic shared memory: `K * sizeof(half)` bytes;
- grid: `ceil(N / 8)`.

### W3A16 GEMV

- gate/up: 512 threads, 16 rows per block;
- down: 768 threads, 24 rows per block;
- dynamic shared memory: `K * sizeof(half)` bytes;
- grid: `ceil(N / rows_per_block)`.

### GQA decode

- grid: `num_query_heads`;
- 128-slot variant: 128 threads;
- 256-slot variant: 256 threads, generated from the same source;
- the gateway uses 256 static cache slots;
- expects RoPE-applied K/V and a boolean validity mask.

The files are generated with:

```powershell
python scripts\export_sm86_kernels.py
```
