"""Small CLI client for the local DeepInfer-SM86 gateway."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path


def post_chat(server: str, prompt: str, max_tokens: int, temperature: float) -> str:
    payload = {
        "model": "deepinfer-sm86-7b-mixed",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    request = urllib.request.Request(
        f"{server.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        data = json.loads(response.read().decode("utf-8"))
    return str(data["choices"][0]["message"]["content"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask the local DeepInfer-SM86 model")
    parser.add_argument("prompt", nargs="?", help="prompt text")
    parser.add_argument("--file", type=Path, help="read prompt from a text file")
    parser.add_argument("--server", default="http://127.0.0.1:18000")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--copy", action="store_true", help="copy result to clipboard on Windows"
    )
    args = parser.parse_args()

    if args.file:
        prompt = args.file.read_text(encoding="utf-8")
    elif args.prompt:
        prompt = args.prompt
    elif not sys.stdin.isatty():
        prompt = sys.stdin.read()
    else:
        parser.error("provide a prompt, --file, or stdin")
    answer = post_chat(args.server, prompt, args.max_tokens, args.temperature)
    print(answer)
    if args.copy:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        root.clipboard_clear()
        root.clipboard_append(answer)
        root.update()
        root.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
