"""Does ctpyes-based CPU expert compute actually overlap with GPU work?

Serial  : GPU loop, then CPU loop          T = Tg + Tc
Parallel: submit CPU loop to a thread, run GPU loop, join   T = max(Tg, Tc) + eps
"""
import ctypes, time, threading, sys
import torch

# --- load the real AVX2 library and call a real kernel entry point ---
from deepinfer.moe import cpu_avx2
lib = cpu_avx2._load()
print("avx2 available:", cpu_avx2.available())

def cpu_busy(seconds):
    """Burn CPU through a ctypes call so the GIL is released like in production."""
    # dequantize a real packed Q4_K buffer repeatedly
    import ctypes as C
    n = 1 << 20
    src = (C.c_uint8 * n)()
    dst = (C.c_float * (n * 256))()  # Q4_K: 256 values per block
    end = time.perf_counter() + seconds
    iters = 0
    while time.perf_counter() < end:
        lib.dp_dequant_q4_k(dst, src, C.c_int(n), C.c_int(n))
        iters += 1
    return iters

def gpu_busy(seconds):
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    b = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        for _ in range(20):
            a @ b
        torch.cuda.synchronize()
    return time.perf_counter() - t0

D = 1.0
gpu_busy(1.0); cpu_busy(1.0)     # warm up

t0 = time.perf_counter(); gpu_busy(D); tg = time.perf_counter() - t0
t0 = time.perf_counter(); cpu_busy(D); tc = time.perf_counter() - t0
t0 = time.perf_counter()
th = threading.Thread(target=cpu_busy, args=(D,)); th.start()
gpu_busy(D)
th.join()
tp = time.perf_counter() - t0

print(f"GPU only      : {tg:.3f} s")
print(f"CPU only      : {tc:.3f} s")
print(f"serial (sum)  : {tg + tc:.3f} s")
print(f"parallel      : {tp:.3f} s")
print(f"overlap gain  : {(tg + tc) / tp:.2f}x  (ideal = {(tg+tc)/max(tg,tc):.2f}x)")
print("VERDICT:", "OVERLAP WORKS" if tp < (tg + tc) * 0.85 else "NO REAL OVERLAP")
