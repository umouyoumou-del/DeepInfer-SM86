"""Report whether an HF/quantized model directory is supported by SM86."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepinfer.model_probe import probe_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    report = probe_model(args.model)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("supported") else 2


if __name__ == "__main__":
    raise SystemExit(main())
