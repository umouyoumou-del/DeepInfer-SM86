"""Shorten vLLM server logs for the repository: its "Received request" lines carry the whole prompt
(text and token ids, ~1 MB per 98K-token request); keep their first 200 characters."""

import sys
from pathlib import Path

for name in sys.argv[1:] or sorted(str(p) for p in Path(__file__).parent.glob("vllm_*.server.log")):
    path = Path(name)
    lines = path.read_text(encoding="utf-8", errors="replace").split("\n")
    cut = 0
    for i, line in enumerate(lines):
        if "Received request" in line and len(line) > 200:
            lines[i] = line[:200] + f" ... [prompt trimmed, {len(line):,} characters]"
            cut += 1
    before = path.stat().st_size
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"{path.name}: {cut} request lines trimmed, {before:,} -> {path.stat().st_size:,} bytes")
