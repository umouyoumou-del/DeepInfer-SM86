"""deepinfer-bench: prompt processing and generation speed, in the style of llama-bench.

``pp N`` prefills N random tokens into a fresh context; ``tg M @ d N`` then generates M
tokens greedily after them and is timed from the first produced token to the last, so the
one-off graph capture is not counted.  After one untimed warm-up run each test runs ``-r``
times; the table gives mean +- spread.
"""

from __future__ import annotations

import random
import statistics
import sys
import time


def add_arguments(parser) -> None:
    parser.add_argument("-p", "--n-prompt", default="512",
                        help="prompt lengths, comma separated (default %(default)s)")
    parser.add_argument("-n", "--n-gen", type=int, default=128, help="tokens generated (default %(default)s)")
    parser.add_argument("-r", "--repetitions", type=int, default=3, help="runs per test (default %(default)s)")
    parser.add_argument("--no-warmup", action="store_true", help="skip the untimed first run")


def steady_rate(marks: list[tuple[float, int]]) -> float:
    """Tokens per second from ``(time, new tokens)`` marks, after the first two (argmax of the prompt, graph capture)."""

    if len(marks) >= 3 and marks[-1][0] > marks[1][0]:
        return sum(count for _, count in marks[2:]) / (marks[-1][0] - marks[1][0])
    if len(marks) >= 2 and marks[-1][0] > marks[0][0]:
        return sum(count for _, count in marks[1:]) / (marks[-1][0] - marks[0][0])
    return 0.0


def _mean(values: list[float]) -> str:
    if not values:
        return "-"
    spread = statistics.pstdev(values) if len(values) > 1 else 0.0
    return f"{statistics.mean(values):.2f} ± {spread:.2f}"


def run(args) -> None:
    import torch
    from transformers import AutoTokenizer

    from deepinfer.cli.models import resolve
    from deepinfer.moe.deepseek_longctx import DeepSeekLongContext
    from deepinfer.moe.deepseek_v2_runtime import DeepSeekV2LiteGGUFRuntime

    model = resolve(args.model, args.hf, args.meta, args.meta_hf)
    print(f"loading {model.gguf.name} ...", file=sys.stderr, flush=True)
    tok = AutoTokenizer.from_pretrained(str(model.meta), trust_remote_code=True)
    rt = DeepSeekV2LiteGGUFRuntime(model.gguf, meta_dir=model.meta, device=args.device, verbose=False)
    rng = random.Random(0)
    vocab = len(tok)

    def once(prompt: int) -> tuple[float, float]:
        ids = [rng.randrange(100, vocab - 100) for _ in range(prompt)]
        session = DeepSeekLongContext(rt, capacity=prompt + args.n_gen + 8, kv_dtype="fp16")
        torch.cuda.synchronize()
        began = time.perf_counter()
        session.prefill(ids[:-1])
        torch.cuda.synchronize()
        prefill = (prompt - 1) / (time.perf_counter() - began)
        marks: list[tuple[float, int]] = []
        session.generate(ids[-1:], max_new_tokens=args.n_gen,
                         on_tokens=lambda new: marks.append((time.perf_counter(), len(new))))
        return prefill, steady_rate(marks)

    prompts = [int(value) for value in str(args.n_prompt).split(",") if value.strip()]
    if not args.no_warmup:
        once(min(prompts))  # kernels compiled, caches filled: not part of any measurement
    rows = []
    for prompt in prompts:
        runs = [once(prompt) for _ in range(int(args.repetitions))]
        rows.append((f"pp{prompt}", _mean([prefill for prefill, _ in runs])))
        rows.append((f"tg{args.n_gen} @ d{prompt}", _mean([rate for _, rate in runs if rate])))
    name = torch.cuda.get_device_name(0)
    print("\n| model | gpu | test | t/s |\n|---|---|---:|---:|")
    for test, speed in rows:
        print(f"| {model.title} | {name} | {test} | {speed} |")
