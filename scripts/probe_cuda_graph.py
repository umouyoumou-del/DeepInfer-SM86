"""Is CUDA Graph capture worth wiring into the decode path?

Captures a single fixed-shape fused expert chunk (6 experts, max_count=1) and
compares eager launch against graph replay.  If replay is not clearly faster
here, capturing the 27-layer loop will not pay off either.
"""

from __future__ import annotations

import argparse
import statistics
import time


def _time(fn, repeats: int) -> float:
    for _ in range(5):
        fn()
    import torch

    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1e6)
    return statistics.median(samples)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args(argv)

    import torch

    from deepinfer.moe.ptr_grouped_moe import (
        fused_down_gemm_ptr_grouped,
        fused_gate_up_gemm_ptr_grouped,
    )

    K, N, V, E = 2048, 1408, 2048, 6
    gate_row = K // 256 * 144
    down_row = N // 32 * 34
    gate = torch.empty((E, N, gate_row), dtype=torch.uint8, device="cuda")
    up = torch.empty((E, N, gate_row), dtype=torch.uint8, device="cuda")
    down = torch.empty((E, V, down_row), dtype=torch.uint8, device="cuda")
    gptr = torch.tensor([t.data_ptr() for t in gate], dtype=torch.int64, device="cuda")
    uptr = torch.tensor([t.data_ptr() for t in up], dtype=torch.int64, device="cuda")
    dptr = torch.tensor([t.data_ptr() for t in down], dtype=torch.int64, device="cuda")
    x = torch.randn((E, 1, K), dtype=torch.float16, device="cuda")
    counts = torch.ones(E, dtype=torch.int32, device="cuda")

    def gate_up() -> torch.Tensor:
        return fused_gate_up_gemm_ptr_grouped(
            x, gptr, uptr, counts, n_experts=E, max_count=1,
            out_features=N, k_size=K, gate_qtype="Q4_K", up_qtype="Q4_K",
            gate_row_bytes=gate_row, up_row_bytes=gate_row,
        )

    hidden = gate_up().detach()

    def down_proj() -> torch.Tensor:
        return fused_down_gemm_ptr_grouped(
            hidden, dptr, counts, n_experts=E, max_count=1,
            out_features=V, k_size=N, down_qtype="Q8_0", down_row_bytes=down_row,
        )

    eager_gu = _time(gate_up, args.repeats)
    eager_dn = _time(down_proj, args.repeats)
    print(f"eager   gate_up {eager_gu:7.1f} us   down {eager_dn:7.1f} us   sum {eager_gu + eager_dn:7.1f} us")

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            gate_up()
            down_proj()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    try:
        with torch.cuda.graph(graph):
            out_gu = gate_up()
            out_dn = down_proj()
    except Exception as error:  # noqa: BLE001
        print(f"capture failed: {type(error).__name__}: {error}")
        return 1

    def replay() -> None:
        graph.replay()

    graph_gu = _time(replay, args.repeats)
    print(f"graph   gate_up+down replayed together: {graph_gu:7.1f} us")
    speedup = (eager_gu + eager_dn) / graph_gu if graph_gu else 0.0
    print(f"per-chunk speedup: {speedup:.2f}x   (x27 layers: "
          f"{(eager_gu + eager_dn) * 27 / 1000:.1f} -> {graph_gu * 27 / 1000:.1f} ms/token)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
