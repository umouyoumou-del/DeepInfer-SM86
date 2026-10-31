#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/DeepInfer-SM86 || exit 1
cat > /tmp/p.py <<'PY'
import time, torch
from deepinfer.moe.ptr_grouped_moe import fused_gate_up_gemm_ptr_grouped, fused_down_gemm_ptr_grouped
from deepinfer.moe.fused_moe import fused_moe_supported
print("supported:", fused_moe_supported())
K=2048; N=1408; V=2048; E=6   # gate/up: [N,K]; down: [V,N]
rb = K//256*144          # Q4_K row bytes
vrb = N//256*144
g = torch.empty((E, N, rb), dtype=torch.uint8, device="cuda")
u = torch.empty((E, N, rb), dtype=torch.uint8, device="cuda")
d = torch.empty((E, V, vrb), dtype=torch.uint8, device="cuda")
gp = torch.tensor([t.data_ptr() for t in g], dtype=torch.int64, device="cuda")
up = torch.tensor([t.data_ptr() for t in u], dtype=torch.int64, device="cuda")
dp = torch.tensor([t.data_ptr() for t in d], dtype=torch.int64, device="cuda")
x = torch.randn((E, 1, K), dtype=torch.float16, device="cuda")
counts = torch.ones(E, dtype=torch.int32, device="cuda")
def bench(fn, n=200):
    for _ in range(10): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/n*1e6
t1 = bench(lambda: fused_gate_up_gemm_ptr_grouped(x, gp, up, counts, n_experts=E, max_count=1,
      out_features=N, k_size=K, gate_qtype="Q4_K", up_qtype="Q4_K",
      gate_row_bytes=rb, up_row_bytes=rb))
h = torch.randn((E,1,N), dtype=torch.float16, device="cuda")
t2 = bench(lambda: fused_down_gemm_ptr_grouped(h, dp, counts, n_experts=E, max_count=1,
      out_features=V, k_size=N, down_qtype="Q8_0", down_row_bytes=N//32*34))
mem = (E*(N*rb*2 + V*vrb))/1e6
print(f"gate_up grouped : {t1:8.1f} us")
print(f"down grouped    : {t2:8.1f} us")
print(f"total per layer : {t1+t2:8.1f} us   x27 layers = {(t1+t2)*27/1000:.1f} ms/token")
print(f"packed bytes    : {mem:.1f} MB -> at 736 GB/s = {mem/736*1000:.1f} us")
PY
PYTHONPATH=$PWD python /tmp/p.py 2>&1 | tail -8
echo PROF_DONE
