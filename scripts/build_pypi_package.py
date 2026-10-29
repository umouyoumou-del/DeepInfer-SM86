"""Assemble the standalone `deepinfer-sm86` PyPI source tree.

The repository is an application/benchmark workspace.  This script selects the
reusable DeepSeek MoE, GGUF, paged-KV, quantized-MLA, AVX2 and generic paged
scheduler modules and copies them into an isolated PEP 517 project.
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "pypi" / "deepinfer-sm86"
SRC = DEST / "src" / "deepinfer"


def copy_file(relative: str, *, transform=None) -> None:
    source = ROOT / relative
    target = SRC / Path(relative).relative_to("deepinfer")
    target.parent.mkdir(parents=True, exist_ok=True)
    text = source.read_text(encoding="utf-8")
    if transform is not None:
        text = transform(text)
    target.write_text(text, encoding="utf-8")


def scheduler_transform(text: str) -> str:
    text = text.replace(
        "from apps.runtime import QwenSm86Runtime\n",
        "",
    )
    text = text.replace(
        "runtime: QwenSm86Runtime,",
        "runtime: Any,",
    )
    return text


def main() -> int:
    if DEST.exists():
        shutil.rmtree(DEST)
    (SRC / "moe").mkdir(parents=True)
    (SRC / "engine").mkdir(parents=True)
    (SRC / "ecs").mkdir(parents=True)
    (SRC / "scheduler").mkdir(parents=True)

    for name in ("__init__.py", "attrdict.py", "environ.py", "logger.py"):
        copy_file(f"deepinfer/{name}")

    moe_files = (
        "__init__.py",
        "deepseek_v2_runtime.py",
        "fused_moe.py",
        "gguf_index.py",
        "gpu_quant.py",
        "grouped_moe.py",
        "layout.py",
        "continuous_batching.py",
        "nccl_expert_transport.py",
        "nccl_request_service.py",
        "ptr_grouped_moe.py",
        "paged_adapter.py",
        "paged_mla_kernel.py",
        "paged_mla_quant.py",
        "cpu_avx2.py",
        "cpu_avx2_packed.cpp",
        "cpu_avx2_gemv.cpp",
    )
    for name in moe_files:
        copy_file(f"deepinfer/moe/{name}")

    for name in ("__init__.py", "kv_cache.py", "memory_budget.py", "metrics.py"):
        copy_file(f"deepinfer/engine/{name}")

    for name in (
        "__init__.py",
        "core.py",
        "controller.py",
        "transport.py",
        "worker_pool.py",
        "ecs_core.cpp",
    ):
        copy_file(f"deepinfer/ecs/{name}")

    scheduler = (ROOT / "apps" / "paged_scheduler.py").read_text(encoding="utf-8")
    (SRC / "scheduler" / "paged.py").write_text(
        scheduler_transform(scheduler), encoding="utf-8"
    )
    (SRC / "scheduler" / "__init__.py").write_text(
        '''"""Generic paged continuous-batching scheduler.

The scheduler uses only the runtime surface ``model``, ``tokenizer``,
``device``, ``max_cache``, ``forward_logits`` and ``_sample``; it is shared by
the DeepSeek MoE runtime and other DeepInfer runtimes.
"""

from deepinfer.scheduler.paged import (
    GenerationResult,
    PagedBatchScheduler,
    PagedScheduledRequest,
)

__all__ = ["GenerationResult", "PagedBatchScheduler", "PagedScheduledRequest"]
''',
        encoding="utf-8",
    )
    (SRC / "py.typed").write_text("", encoding="utf-8")

    for name in ("LICENSE", "NOTICE"):
        shutil.copy2(ROOT / name, DEST / name)
    (DEST / "README.md").write_text(
        """# deepinfer-sm86

Native DeepSeek-V2-Lite GGUF MoE runtime primitives for memory-constrained
CUDA GPUs.

This wheel contains the reusable library layer:

- packed GGUF expert index and streaming reader;
- fused Q4_K/Q5_0/Q8_0 grouped MoE kernels;
- fused paged MLA (FP16) and native INT8/FP8 E4M3/E5M2 paged MLA;
- CPU AVX2 packed GEMV/dequant sources, compiled at runtime;
- paged KV cache, memory budget and metrics primitives;
- generic paged continuous-batching scheduler;
- ECS control plane: heat/LRU tier placement, chunk packing and an
  all-to-all transport abstraction.

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
""",
        encoding="utf-8",
    )
    (DEST / "pyproject.toml").write_text(
        '''[build-system]
requires = ["setuptools>=77", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "deepinfer-sm86"
version = "0.2.0"
description = "Native streaming DeepSeek-V2-Lite GGUF MoE and paged MLA primitives for memory-constrained CUDA GPUs"
readme = "README.md"
requires-python = ">=3.10"
license = "Apache-2.0"
license-files = ["LICENSE", "NOTICE"]
authors = [{ name = "DeepInfer-SM86 contributors" }]
keywords = ["deepseek", "moe", "gguf", "paged-attention", "cuda", "sm86"]
classifiers = [
  "Development Status :: 3 - Alpha",
  "Intended Audience :: Developers",
  "Programming Language :: Python :: 3",
  "Programming Language :: Python :: 3.10",
  "Programming Language :: Python :: 3.11",
  "Programming Language :: Python :: 3.12",
  "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
  "torch>=2.5",
  "numpy>=1.24",
  "transformers>=4.49",
  "accelerate>=1.3",
  "gguf>=0.19",
  "typing-extensions>=4.8",
]

[project.optional-dependencies]
cuda = ["cupy-cuda12x[ctk]>=14.2,<15"]
dev = ["pytest>=8", "build>=1", "wheel>=0.43"]

[tool.setuptools]
package-dir = { "" = "src" }
include-package-data = true

[tool.setuptools.packages.find]
where = ["src"]
include = ["deepinfer*"]

[tool.setuptools.package-data]
"deepinfer.moe" = ["*.cpp"]
"deepinfer.ecs" = ["*.cpp"]
"deepinfer" = ["py.typed"]
''',
        encoding="utf-8",
    )
    (DEST / "Makefile").write_text(
        """PYTHON ?= python

.PHONY: build clean
build:
	$(PYTHON) -m build --no-isolation --sdist --wheel

clean:
	rm -rf build dist src/*.egg-info
""",
        encoding="utf-8",
    )
    print(f"assembled {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
