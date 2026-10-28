# deepinfer-sm86

Native DeepSeek-V2-Lite GGUF MoE runtime primitives for memory-constrained
CUDA GPUs.

This wheel contains the reusable library layer:

- packed GGUF expert index and streaming reader;
- fused Q4_K/Q5_0/Q8_0 grouped MoE kernels;
- fused paged MLA (FP16) and native INT8/FP8 E4M3/E5M2 paged MLA;
- CPU AVX2 packed GEMV/dequant sources, compiled at runtime;
- paged KV cache, memory budget and metrics primitives;
- generic paged continuous-batching scheduler.

Model weights, tokenizer metadata, benchmark scripts, the Qwen server and the
repository's application workspace are **not** included in the wheel.

## Install

```bash
pip install deepinfer-sm86
# GPU kernels also need CuPy matching the installed CUDA build:
pip install "deepinfer-sm86[cuda]"
```

`torch` must be installed with the CUDA build matching the target GPU; it is a
normal dependency so pip can be pointed at the PyTorch index when needed.

## Direct runtime

```python
from deepinfer.moe.deepseek_v2_runtime import DeepSeekV2LiteGGUFRuntime

runtime = DeepSeekV2LiteGGUFRuntime(
    "DeepSeek-V2-Lite-Chat-Q4_K_M.gguf",
    meta_dir="meta",
    max_cache=512,
)
try:
    result = runtime.generate(
        [{"role": "user", "content": "只回答数字：5-2=?"}],
        max_new_tokens=8,
        temperature=0.0,
    )
    print(result.text)
finally:
    runtime.close()
```

## Paged scheduler

```python
from deepinfer.scheduler import PagedBatchScheduler

scheduler = PagedBatchScheduler(
    runtime,
    max_num_seqs=2,
    block_size=16,
    prefill_chunk_tokens=8,
)
try:
    request = scheduler.submit(
        [{"role": "user", "content": "只回答数字：2+3=?"}],
        max_new_tokens=8,
        temperature=0.0,
    )
    print(request.future.result(timeout=120).text)
finally:
    scheduler.stop()
```

The scheduler is generic; it expects the runtime to expose `model`,
`tokenizer`, `device`, `max_cache`, `forward_logits` and `_sample`, which the
DeepSeek runtime does.

## Optional CPU AVX2 cold experts

The C++ sources are shipped in the wheel.  At runtime, `deepinfer.moe.cpu_avx2`
builds a small shared library with `g++ -O3 -mavx2 -mfma -fopenmp`; if the
compiler is unavailable, the runtime falls back to PyTorch CPU dequantization.
No binary DLL is bundled.

## License

Apache-2.0.  See `LICENSE` and `NOTICE`.
