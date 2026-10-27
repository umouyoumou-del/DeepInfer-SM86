"""Import GPTQ, AWQ or GGUF weights into a native SM86 W4A16 checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepinfer.quantization import (
    UnsupportedQuantFormatError,
    convert_checkpoint,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a GPTQ/AWQ directory or single-file GGUF checkpoint into "
            "the DeepInfer-SM86 layered W4A16 format."
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--format", choices=("auto", "gptq", "awq", "gguf"), default="auto"
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="optional HF tokenizer directory; needed to serve GGUF through the gateway",
    )
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()

    try:
        summary = convert_checkpoint(
            args.source,
            args.output,
            group_size=args.group_size,
            tokenizer=args.tokenizer,
            overwrite=args.overwrite,
            source_format=None if args.format == "auto" else args.format,
        )
    except (UnsupportedQuantFormatError, ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"import failed: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
