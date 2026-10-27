"""Extract embedded CuPy/NVRTC CUDA sources into standalone .cu files."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "qwen2_w4a16_gemv.cu": (
        ROOT / "deepinfer/model_executor/models/qwen2_w4a16.py",
        "_SM86_W4A16_KERNEL",
    ),
    "qwen2_w4a16_batched_gemv.cu": (
        ROOT / "deepinfer/model_executor/models/qwen2_w4a16.py",
        "_SM86_W4A16_BATCHED_KERNEL",
    ),
    "qwen2_w4a16_tiled_prefill.cu": (
        ROOT / "deepinfer/model_executor/models/qwen2_w4a16.py",
        "_SM86_W4A16_TILED_GEMM_KERNEL",
    ),
    "qwen2_paged_gqa_decode.cu": (
        ROOT / "deepinfer/engine/paged_attention.py",
        "_SM86_PAGED_GQA_KERNEL",
    ),
    "qwen2_paged_gqa_decode_int8.cu": (
        ROOT / "deepinfer/engine/paged_attention.py",
        "_SM86_PAGED_GQA_I8_KERNEL",
    ),
    "qwen2_paged_gqa_decode_fp8.cu": (
        ROOT / "deepinfer/engine/paged_attention.py",
        "_SM86_PAGED_GQA_FP8_KERNEL",
    ),
    "qwen2_gqa_decode.cu": (
        ROOT / "deepinfer/model_executor/models/qwen2_w4a16.py",
        "_SM86_GQA_DECODE_KERNEL",
    ),
    "qwen2_w3a16_gemv.cu": (
        ROOT / "deepinfer/model_executor/models/qwen2_mixed_sm86.py",
        "_W3_KERNEL",
    ),
    "qwen2_w3a16_tiled_prefill.cu": (
        ROOT / "deepinfer/model_executor/models/qwen2_mixed_sm86.py",
        "_SM86_W3A16_TILED_GEMM_KERNEL",
    ),
}


def main() -> int:
    output = ROOT / "kernels"
    output.mkdir(exist_ok=True)
    for name, (path, variable) in SOURCES.items():
        source = path.read_text(encoding="utf-8")
        match = re.search(rf'{variable} = r"""(.*?)"""', source, re.DOTALL)
        if match is None:
            raise RuntimeError(f"cannot find {variable} in {path}")
        body = match.group(1).lstrip("\n")
        banner = f"// Extracted from {path.relative_to(ROOT).as_posix()}:{variable}\n\n"
        (output / name).write_text(
            banner + body, encoding="utf-8", newline="\n"
        )
        print(f"wrote {name}")
    base = (output / "qwen2_gqa_decode.cu").read_text(encoding="utf-8")
    extended = base.replace("QWEN2_GQA_MAX_SEQ 128", "QWEN2_GQA_MAX_SEQ 256").replace(
        "qwen2_gqa_decode", "qwen2_gqa_decode_256"
    )
    (output / "qwen2_gqa_decode_256.cu").write_text(
        extended, encoding="utf-8", newline="\n"
    )
    print("wrote qwen2_gqa_decode_256.cu")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
