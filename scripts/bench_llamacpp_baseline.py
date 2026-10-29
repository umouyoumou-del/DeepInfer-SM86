"""Measure the llama.cpp baseline for the 4 GB setup via a local Ollama server.

Ollama bundles llama.cpp, so it is the practical way to get an honest
offload baseline on the same machine and the same GGUF file without building
llama.cpp with CUDA from source.  The script:

1. starts `ollama serve` against the given model directory (if not running),
2. registers the GGUF with a Modelfile carrying `PARAMETER num_gpu N`,
3. runs the same greedy prompts the ECS benchmarks use,
4. reports decode tok/s, prefill tok/s and TTFT from the API timings,
5. samples `nvidia-smi` while each request runs to capture peak VRAM.

The result is written as JSON next to the other ECS result files so the paper
can cite it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_PROMPTS = [
    "只回答数字：5-2=?",
    "只回答数字：2+3=?",
    "只回答数字：9-4=?",
    "只回答数字：1+1=?",
]


def _request(url: str, payload: dict | None = None, *, timeout: float = 600.0):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body) if body else {}


def wait_for_server(base: str, timeout: float = 120.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            _request(f"{base}/api/version", timeout=5.0)
            return True
        except Exception:
            time.sleep(0.5)
    return False


class VramSampler:
    """Poll nvidia-smi in the background and keep the peak usage."""

    def __init__(self, index: int = 0, interval: float = 0.1) -> None:
        self.index = int(index)
        self.interval = float(interval)
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> int:
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.index}",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
            )
            return int(out.stdout.strip().splitlines()[0])
        except Exception:
            return 0

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_mib = max(self.peak_mib, self._sample())
            time.sleep(self.interval)

    def __enter__(self) -> "VramSampler":
        self.peak_mib = self._sample()
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)


def ensure_model(
    ollama: Path, gguf: Path, name: str, num_gpu: int, workdir: Path
) -> None:
    modelfile = workdir / f"Modelfile.{name}"
    modelfile.write_text(
        f'FROM {gguf.as_posix()}\nPARAMETER num_gpu {int(num_gpu)}\n'
        "PARAMETER temperature 0\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [str(ollama), "create", name, "-f", str(modelfile)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=3600,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ollama create failed ({result.returncode}): "
            f"{result.stderr[-2000:] or result.stdout[-2000:]}"
        )


def run_prompt(base: str, model: str, prompt: str, max_new_tokens: int) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": int(max_new_tokens)},
    }
    started = time.perf_counter()
    with VramSampler() as sampler:
        response = _request(f"{base}/api/generate", payload)
    wall = time.perf_counter() - started
    eval_count = int(response.get("eval_count") or 0)
    eval_ns = float(response.get("eval_duration") or 0.0)
    prompt_count = int(response.get("prompt_eval_count") or 0)
    prompt_ns = float(response.get("prompt_eval_duration") or 0.0)
    load_ns = float(response.get("load_duration") or 0.0)
    return {
        "prompt": prompt,
        "text": response.get("response", ""),
        "completion_tokens": eval_count,
        "prompt_tokens": prompt_count,
        "wall_seconds": wall,
        "decode_tokens_per_second": (
            eval_count / (eval_ns / 1e9) if eval_ns > 0 else 0.0
        ),
        "prefill_seconds": (prompt_ns + load_ns) / 1e9,
        "prefill_tokens_per_second": (
            prompt_count / (prompt_ns / 1e9) if prompt_ns > 0 else 0.0
        ),
        "peak_gpu_mib": sampler.peak_mib,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ollama", type=Path, default=Path(r"F:\Ollama\app\ollama.exe"))
    parser.add_argument("--models-dir", type=Path, default=Path(r"F:\Ollama\models"))
    parser.add_argument(
        "--gguf",
        type=Path,
        default=Path(
            r"E:\DeepInfer-MoE\models\DeepSeek-V2-Lite-Chat-GGUF"
            r"\DeepSeek-V2-Lite-Chat-Q4_K_M.gguf"
        ),
    )
    parser.add_argument("--base", default="http://127.0.0.1:11434")
    parser.add_argument("--num-gpu", type=int, nargs="+", default=[0, 8, 99])
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/results/llamacpp_baseline_4gb.json"),
    )
    args = parser.parse_args()

    if not args.gguf.exists():
        raise SystemExit(f"GGUF not found: {args.gguf}")

    env = dict(os.environ)
    env["OLLAMA_MODELS"] = str(args.models_dir)
    server = None
    try:
        _request(f"{args.base}/api/version", timeout=3.0)
    except Exception:
        server = subprocess.Popen(
            [str(args.ollama), "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if not wait_for_server(args.base):
            raise SystemExit("ollama server did not come up")

    workdir = Path("build") / "llamacpp_baseline"
    workdir.mkdir(parents=True, exist_ok=True)
    runs = []
    try:
        for num_gpu in args.num_gpu:
            name = f"dsv2lite-baseline-g{num_gpu}"
            print(f"[baseline] creating {name} (num_gpu={num_gpu})", flush=True)
            ensure_model(args.ollama, args.gguf, name, num_gpu, workdir)
            for repeat in range(args.repeats):
                for prompt in args.prompts:
                    record = run_prompt(
                        args.base, name, prompt, args.max_new_tokens
                    )
                    record.update(
                        {"num_gpu": num_gpu, "repeat": repeat, "model": name}
                    )
                    runs.append(record)
                    print(
                        f"[baseline] num_gpu={num_gpu} "
                        f"decode={record['decode_tokens_per_second']:.2f} tok/s "
                        f"peak={record['peak_gpu_mib']} MiB "
                        f"text={record['text'][:16]!r}",
                        flush=True,
                    )
    finally:
        summary = {}
        for record in runs:
            key = int(record["num_gpu"])
            summary.setdefault(key, []).append(record["decode_tokens_per_second"])
        payload = {
            "backend": "llama.cpp (via ollama)",
            "ollama_binary": str(args.ollama),
            "gguf": str(args.gguf),
            "max_new_tokens": args.max_new_tokens,
            "device": "RTX 3050 Laptop 4 GB",
            "num_gpu_values": list(args.num_gpu),
            "decode_tokens_per_second_by_num_gpu": {
                str(key): (sum(values) / len(values) if values else 0.0)
                for key, values in summary.items()
            },
            "records": runs,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[baseline] wrote {args.output}", flush=True)
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
