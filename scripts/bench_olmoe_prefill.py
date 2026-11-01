"""OLMoE prompt processing through the batched prefill (deepinfer.moe.olmoe_prefill).

Measures what llama-bench's ``pp`` test measures: a prompt of N random tokens
into an empty cache, timed from the call until its result is on the host,
reported as N / time.  Each length is warmed up, then repeated; the median is
reported.  Launch overhead is included: it is part of what a prompt costs.

After the timed runs, one prefill of the fixed prompt (default "The capital of
France is") is decoded for a few tokens and compared with stepping the same
prompt through the fused decoder token by token.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

DEFAULT_GGUF = os.environ.get(
    "OLMOE_GGUF",
    r"E:\DeepInfer-MoE\models\OLMoE-1B-7B-GGUF\OLMoE-1B-7B-0924.Q4_K_M.gguf",
)
DEFAULT_PROMPT_IDS = [510, 5347, 273, 6181, 310]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gguf", default=DEFAULT_GGUF)
    parser.add_argument("--layers", type=int, help="truncate to the first N layers")
    parser.add_argument("--lengths", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--prompt-ids", type=int, nargs="+", default=DEFAULT_PROMPT_IDS)
    parser.add_argument("--check-tokens", type=int, default=16,
                        help="tokens decoded after the prefill check")
    parser.add_argument("--no-graphs", action="store_true",
                        help="launch every prefill kernel from Python instead of replaying a graph")
    parser.add_argument("--output", help="write all records as JSON")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import numpy as np
    import torch

    from deepinfer.moe.gguf_index import GGUFIndex
    from deepinfer.moe.olmoe_fused import OLMoEFusedDecoder, load_fused_model
    from deepinfer.moe.olmoe_prefill import OLMoEFusedPrefill
    from deepinfer.moe.olmoe_spec import spec_from_gguf

    index = GGUFIndex(args.gguf)
    spec = spec_from_gguf(index)
    layers = args.layers or spec.block_count
    loaded = time.perf_counter()
    model = load_fused_model(index, spec, layers=layers)
    torch.cuda.synchronize()
    print(f"loaded {layers} layers in {time.perf_counter() - loaded:.1f} s, "
          f"VRAM {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
    longest = max(*args.lengths, len(args.prompt_ids))
    kwargs = {"num_heads": spec.head_count, "eps": spec.rms_eps, "rope_base": spec.rope_base,
              "top_k": spec.expert_used_count, "max_len": longest + args.check_tokens + 8}
    decoder = OLMoEFusedDecoder(model, **kwargs)
    decoder.capture()
    engine = OLMoEFusedPrefill(decoder, max_tokens=longest, use_graphs=not args.no_graphs)
    environment = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                   "cuda": torch.version.cuda, "python": platform.python_version(),
                   "os": platform.platform(), "gguf": str(args.gguf), "layers": layers}
    print("environment:", json.dumps(environment, ensure_ascii=False), flush=True)

    rng = np.random.default_rng(0)
    vocab = model.lm_head.rows  # the GGUF's olmoe metadata carries no vocab size
    records = []
    for length in args.lengths:
        prompt = [int(t) for t in rng.integers(0, vocab, size=length)]
        for _ in range(args.warmup):
            engine.run(prompt)
        times = []
        for _ in range(args.repeats):
            torch.cuda.synchronize()
            began = time.perf_counter()
            engine.run(prompt)  # returns the first prediction, so it has synchronised
            times.append((time.perf_counter() - began) * 1e3)
        median = statistics.median(times)
        records.append({"length": length, "ms": times, "median_ms": median,
                        "tokens_per_second": length / median * 1e3})
        print(f"pp{length:<5d} {median:8.2f} ms  {length / median * 1e3:9.1f} tok/s"
              f"  (min {min(times):.2f}, max {max(times):.2f} ms, {len(times)} runs)", flush=True)

    # correctness: decode after a prefill against decode after stepping
    prompt = list(args.prompt_ids)
    first = engine.run(prompt)
    prefilled = [first] + [int(decoder.step()) for _ in range(args.check_tokens - 1)]
    stepped = decoder.generate(prompt, args.check_tokens)
    split = next((i for i, (a, b) in enumerate(zip(prefilled, stepped)) if a != b), None)
    print("prefill then decode vs stepping: tokens "
          + ("identical" if split is None else f"diverge at token {split}"), flush=True)
    print("  prefilled:", prefilled, flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(
            {"environment": environment, "args": vars(args), "records": records,
             "check": {"prompt": prompt, "prefilled": prefilled, "stepped": stepped,
                       "first_split": split}}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
