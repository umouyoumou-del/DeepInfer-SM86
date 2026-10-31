"""OLMoE decode on the packed-expert path, with layer-count and top_k sweeps.

The numbers in docs/OLMOE_PORT_PLAN.md sections 30-33 came from scripts that only
existed on the remote host.  This one reproduces that setup from the repository:
real GGUF weights, packed Q4_K/Q6_K experts addressed by pointer, fp16 dense
weights, the OLMoEKVKache, greedy decoding.

Each decode step is timed three ways, because a per-token number alone cannot
say *why* a step costs what it does:

  step   wall clock, synchronize to synchronize
  host   time for the host to enqueue the step (forward() returning)
  gpu    pure device time: the stream is held by a spin kernel while the host
         enqueues, so the GPU never waits for the host.  Only valid when the
         step contains no host synchronisation, which ``--sync-debug`` checks.

step ~= host >> gpu means the step is bound by host-side dispatch, not by the
device, and no change to kernels or bytes will move it.  ``--top-k`` changes
only device work and ``--host-delay-ms`` only host work, so sweeping each shows
which side a configuration is bound by.

Prefill runs one token at a time through the decode path.  The packed kernels
handle a single token, and a batched prefill would need fp16 experts at about
0.8 GB per layer.  Prefill is not what is measured, and it leaves the same cache.

Layer sweeps truncate the model to its first N layers; the text is then
meaningless but the per-layer work is unchanged.  Only the full model's text is
a correctness check.
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import os
import platform
import statistics
import sys
import time
import warnings
from pathlib import Path

DEFAULT_GGUF = os.environ.get(
    "OLMOE_GGUF",
    r"E:\DeepInfer-MoE\models\OLMoE-1B-7B-GGUF\OLMoE-1B-7B-0924.Q4_K_M.gguf",
)
# "The capital of France is" in the OLMoE (GPT-NeoX) vocabulary
DEFAULT_PROMPT_IDS = [510, 5347, 273, 6181, 310]


def cpu_name() -> str:
    """The CPU model, since host dispatch time depends on it.

    ``platform.processor()`` is only ``x86_64`` on Linux and a family/stepping
    string on Windows.
    """

    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    if sys.platform == "win32":
        import winreg

        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor()


def load_model(index, spec, layers: int, device):
    """Dense weights in fp16 (F32 norms/router kept as stored), experts packed."""

    import torch

    from deepinfer.moe.olmoe_packed import load_packed_experts
    from deepinfer.moe.olmoe_weights import layer_tensors

    def dense(name: str, dtype=torch.float16):
        array = index.read_tensor(name)
        return torch.from_numpy(array.copy()).to(device=device, dtype=dtype)

    def stored(name: str):
        # F32 in the GGUF; keeping it F32 avoids a cast kernel per call, since
        # rms_norm and route_topk call .float() on it
        return dense(name, dtype=torch.float32)

    weights = {
        "embed_tokens": dense("token_embd.weight"),
        "final_norm": stored("output_norm.weight"),
        "lm_head": dense("output.weight"),
    }
    stack = []
    for layer_index in range(layers):
        names = layer_tensors(spec, layer_index)
        layer = {role: dense(name) for role, name in names.attention.items()}
        layer.update({role: stored(name) for role, name in names.norms.items()})
        layer["router"] = stored(names.router)
        inter = int(index.tensor(names.expert_gate).numpy_shape[1])
        layer["packed"] = {
            "gate": load_packed_experts(
                index, names.expert_gate, experts=spec.expert_count, out_features=inter
            ),
            "up": load_packed_experts(
                index, names.expert_up, experts=spec.expert_count, out_features=inter
            ),
            "down": load_packed_experts(
                index, names.expert_down, experts=spec.expert_count,
                out_features=spec.hidden_size,
            ),
        }
        layer["packed_gate_out"] = inter
        stack.append(layer)
    return weights, stack


def bytes_per_token(weights, stack, *, top_k: int, experts: int) -> dict[str, float]:
    """Bytes a decode step reads: everything dense, plus top_k expert slabs."""

    def size(tensor) -> int:
        return tensor.numel() * tensor.element_size()

    dense = size(weights["lm_head"]) + size(weights["final_norm"])
    dense += weights["embed_tokens"].shape[1] * weights["embed_tokens"].element_size()
    expert = 0
    for layer in stack:
        dense += sum(size(value) for key, value in layer.items()
                     if key not in ("packed", "packed_gate_out"))
        for storage, _pointers, _qtype in layer["packed"].values():
            expert += top_k * size(storage) // experts
    return {"dense_gb": dense / 1e9, "expert_gb": expert / 1e9,
            "total_gb": (dense + expert) / 1e9}


def count_syncs(step) -> list[str]:
    """Locations of host synchronisations inside one decode step."""

    import torch

    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("warn")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            step()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    return [
        f"{Path(w.filename).name}:{w.lineno}"
        for w in caught
        if "synchronizing CUDA operation" in str(w.message)
    ]


def sleep_cycles_per_ms() -> float:
    """Calibrate torch.cuda._sleep, which spins for a cycle count, not a time."""

    import torch

    probe = 10_000_000
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda._sleep(probe)  # first call warms the kernel
    torch.cuda.synchronize()
    start.record()
    torch.cuda._sleep(probe)
    end.record()
    end.synchronize()
    return probe / max(start.elapsed_time(end), 1e-3)


def run_once(forward_step, prompt_ids, *, new_tokens: int, measure_gpu: bool,
             sleep_cycles: int):
    """Prefill token by token, then time ``new_tokens`` greedy decode steps.

    Returns the token the prefill produced (it precedes the timed ones), the
    decode-step tokens, and the per-step timings.
    """

    import torch

    cache_state = forward_step.new_cache()
    nxt = None
    for token in prompt_ids:
        nxt = forward_step(torch.tensor([[token]], device="cuda"), cache_state)
    torch.cuda.synchronize()
    first = int(nxt)

    generated, step_ms, host_ms, gpu_ms = [], [], [], []
    for _ in range(new_tokens):
        if measure_gpu:
            torch.cuda._sleep(sleep_cycles)  # hold the stream while the host enqueues
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        began = time.perf_counter()
        nxt = forward_step(nxt, cache_state)
        issued = time.perf_counter()
        end.record()
        end.synchronize()
        finished = time.perf_counter()
        host_ms.append((issued - began) * 1e3)
        if measure_gpu:
            gpu_ms.append(start.elapsed_time(end))
        else:
            step_ms.append((finished - began) * 1e3)
        generated.append(nxt)
    ids = [int(t) for t in torch.cat(generated).flatten().tolist()]
    return first, ids, step_ms, host_ms, gpu_ms


def install_host_delay(ms: float) -> None:
    """Spin ``ms`` on the host before every layer; the device work is unchanged.

    A spin rather than time.sleep, whose granularity on Windows is 1-15 ms.
    """

    from deepinfer.moe import olmoe_model
    from deepinfer.moe.olmoe_layer import decoder_layer_forward

    if not ms:
        olmoe_model.decoder_layer_forward = decoder_layer_forward
        return

    def delayed(*args, **kwargs):
        until = time.perf_counter() + ms / 1e3
        while time.perf_counter() < until:
            pass
        return decoder_layer_forward(*args, **kwargs)

    olmoe_model.decoder_layer_forward = delayed


class ForwardStep:
    """One greedy decode step; the next token stays on the device (no sync)."""

    def __init__(self, weights, stack, spec, *, layers: int, top_k: int):
        self.weights, self.stack, self.spec = weights, stack[:layers], spec
        self.top_k = top_k

    def new_cache(self):
        from deepinfer.moe.olmoe_layer import OLMoEKVKache

        return OLMoEKVKache()

    def __call__(self, token, cache):
        from deepinfer.moe.olmoe_model import forward

        logits = forward(
            token, self.weights, self.stack,
            num_heads=self.spec.head_count, eps=self.spec.rms_eps,
            rope_base=self.spec.rope_base, top_k=self.top_k, cache=cache,
        )
        return logits[:, -1].argmax(-1).view(1, 1)


def profile_host(step, prompt_ids, *, new_tokens: int, top: int = 12) -> None:
    """Where the host time of a decode step goes.  cProfile inflates small
    functions, so read the proportions, not the absolute numbers."""

    import cProfile
    import pstats

    import torch

    cache = step.new_cache()
    token = None
    for prompt_token in prompt_ids:
        token = step(torch.tensor([[prompt_token]], device="cuda"), cache)
    torch.cuda.synchronize()
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(new_tokens):
        token = step(token, cache)
    profiler.disable()
    torch.cuda.synchronize()
    stats = pstats.Stats(profiler).stats
    rows = sorted(
        ((cumulative / new_tokens * 1e3, calls / new_tokens, f"{Path(file).name}:{line}({name})")
         for (file, line, name), (_cc, calls, _tt, cumulative, _callers) in stats.items()),
        reverse=True,
    )
    print(f"  host profile, cumulative ms per step (cProfile, {new_tokens} steps):")
    for cumulative, calls, where in rows[:top]:
        print(f"    {cumulative:7.3f}  calls/step {calls:6.1f}  {where}")


def cpu_probe_ms() -> float:
    """Fixed pure-Python work with no GPU involvement.

    On the shared 4080 SUPER host the decode loop's CPU dropped into a slower
    state for whole runs, doubling host time (see docs/OLMOE_PORT_PLAN.md).
    Timing this before every repeat shows which repeats were taken in it.
    """

    began = time.perf_counter()
    total = 0
    for value in range(300_000):
        total += value * value % 7
    return (time.perf_counter() - began) * 1e3


def summarise(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    quartiles = statistics.quantiles(ordered, n=4) if len(ordered) > 1 else [ordered[0]] * 3
    return {"median": statistics.median(ordered), "q1": quartiles[0], "q3": quartiles[2],
            "min": ordered[0], "max": ordered[-1], "n": len(ordered)}


def detokenize(index, ids: list[int]) -> str:
    try:
        import gguf
    except ImportError:
        return ""
    reader = gguf.GGUFReader(str(index.path))
    field = reader.fields["tokenizer.ggml.tokens"]
    vocab = [bytes(field.parts[i]).decode("utf-8", "replace") for i in field.data]
    text = "".join(vocab[i] if 0 <= i < len(vocab) else "?" for i in ids)
    return text.replace("\u0120", " ").replace("\u010a", "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gguf", default=DEFAULT_GGUF)
    parser.add_argument("--layers", type=int, nargs="+",
                        help="truncate to the first N layers; several values sweep")
    parser.add_argument("--top-k", type=int, nargs="+",
                        help="experts per token; several values sweep (default: GGUF value)")
    parser.add_argument("--prompt-ids", type=int, nargs="+", default=DEFAULT_PROMPT_IDS)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup-tokens", type=int, default=4,
                        help="leading decode steps left out of the statistics")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--host-delay-ms", type=float, nargs="+", default=[0.0],
                        help="host-only spin per layer; several values sweep")
    parser.add_argument("--sync-debug", action="store_true",
                        help="list host synchronisations in one decode step")
    parser.add_argument("--profile-host", action="store_true",
                        help="print where the host time goes (cProfile) per configuration")
    parser.add_argument("--output", help="write all records as JSON")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import torch

    from deepinfer.moe.gguf_index import GGUFIndex
    from deepinfer.moe.olmoe_spec import spec_from_gguf

    index = GGUFIndex(args.gguf)
    spec = spec_from_gguf(index)
    layer_values = sorted(set(args.layers or [spec.block_count]))
    topk_values = sorted(set(args.top_k or [spec.expert_used_count]), reverse=True)
    if max(layer_values) > spec.block_count:
        parser.error(f"the checkpoint has {spec.block_count} layers")

    device = torch.device("cuda")
    loaded = time.perf_counter()
    weights, stack = load_model(index, spec, max(layer_values), device)
    torch.cuda.synchronize()
    print(f"loaded {len(stack)} layers in {time.perf_counter() - loaded:.1f} s, "
          f"VRAM {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
    cycles_per_ms = sleep_cycles_per_ms()

    environment = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "cpu": cpu_name(),
        "os": platform.platform(),
        "gguf": str(args.gguf),
    }
    print("environment:", json.dumps(environment, ensure_ascii=False), flush=True)
    records = []
    delay_values = sorted(set(args.host_delay_ms))
    for layers in layer_values:
        for top_k, delay in itertools.product(topk_values, delay_values):
            step = ForwardStep(weights, stack, spec, layers=layers, top_k=top_k)
            install_host_delay(delay)
            run_once(step, args.prompt_ids, new_tokens=4, measure_gpu=False,
                     sleep_cycles=0)  # compile kernels, fill allocator
            syncs = None
            if args.sync_debug:
                cache = step.new_cache()
                token = step(torch.tensor([[args.prompt_ids[0]]], device=device), cache)
                syncs = count_syncs(functools.partial(step, token, cache))

            steps, hosts, probes, first, ids = [], [], [], None, []
            for _ in range(args.repeats):
                probes.append(cpu_probe_ms())
                first, ids, step_ms, host_ms, _ = run_once(
                    step, args.prompt_ids, new_tokens=args.max_new_tokens,
                    measure_gpu=False, sleep_cycles=0)
                steps.append(statistics.median(step_ms[args.warmup_tokens:]))
                hosts.append(statistics.median(host_ms[args.warmup_tokens:]))

            # pure device time: only meaningful when nothing in the step syncs,
            # and only if the spin outlasts the enqueue, which is checked.  A
            # host delay leaves the device work unchanged, so it is not re-measured.
            gpus, hold_ms, worst_enqueue = [], 3 * max(hosts) + 20, 0.0
            if not syncs and not delay:
                for _ in range(args.repeats):
                    _, _, _, host_ms, gpu_ms = run_once(
                        step, args.prompt_ids, new_tokens=args.max_new_tokens,
                        measure_gpu=True, sleep_cycles=int(cycles_per_ms * hold_ms))
                    gpus.append(statistics.median(gpu_ms[args.warmup_tokens:]))
                    worst_enqueue = max(worst_enqueue, max(host_ms))
                if worst_enqueue >= hold_ms:
                    print(f"  gpu time invalid: enqueue {worst_enqueue:.1f} ms outlasted "
                          f"the {hold_ms:.1f} ms hold", flush=True)
                    gpus = []

            traffic = bytes_per_token(weights, stack[:layers], top_k=top_k,
                                      experts=spec.expert_count)
            record = {
                "layers": layers,
                "top_k": top_k,
                "host_delay_ms": delay,
                "step_ms": summarise(steps),
                "repeat_step_ms": steps,
                "repeat_cpu_probe_ms": probes,
                "host_ms": summarise(hosts),
                "gpu_ms": summarise(gpus),
                "tokens_per_second": 1e3 / statistics.median(steps),
                "bytes": traffic,
                "effective_gb_s": traffic["total_gb"] / (statistics.median(steps) / 1e3),
                "gpu_hold_ms": hold_ms if gpus else None,
                "syncs_per_step": None if syncs is None else len(syncs),
                "sync_sites": None if syncs is None else sorted(set(syncs)),
                "first_token_id": first,
                "generated_ids": ids,
            }
            records.append(record)
            gpu_text = (f"gpu {record['gpu_ms']['median']:6.2f}" if gpus
                        else "gpu    n/a")
            sync_text = "" if syncs is None else f"  syncs {len(syncs)} {sorted(set(syncs))}"
            delay_text = f" +{delay:.2f}ms/layer" if len(delay_values) > 1 or delay else ""
            print(
                f"L={layers:2d} top_k={top_k}{delay_text}"
                f"  step {record['step_ms']['median']:6.2f} ms"
                f"  host {record['host_ms']['median']:6.2f}  {gpu_text}"
                f"  {record['tokens_per_second']:6.2f} tok/s"
                f"  {traffic['total_gb']:.3f} GB/token"
                f" (dense {traffic['dense_gb']:.3f} + experts {traffic['expert_gb']:.3f})"
                f"  cpu-probe {min(probes):.1f}-{max(probes):.1f} ms"
                f"{sync_text}",
                flush=True,
            )
            if layers == spec.block_count:
                print("  text:", repr(detokenize(index, [first, *ids])), flush=True)
            if args.profile_host:
                profile_host(step, args.prompt_ids, new_tokens=args.max_new_tokens)

    if args.output:
        Path(args.output).write_text(json.dumps(
            {"environment": environment, "args": vars(args), "records": records},
            indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
