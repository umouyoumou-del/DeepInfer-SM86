# W4/W3 prefill: Tensor Core 尝试的状态

> 结论：**未完成，未启用，未验证。**

## 做了什么

`kernels/qwen2_w4a16_tiled_prefill_tc.cu` —— 一个与现有
`kernels/qwen2_w4a16_tiled_prefill.cu` 同构的 Tensor Core 版本：

- 保持 32×64 CTA tile、相同的 int4 分组反量化路径、相同的 shared memory 布局；
- 内积从"每线程 8 次 FMA + FP32 累加"换成 `wmma::mma_sync`
  （16×16×16，FP16 输入 / FP32 累加）；
- 8 个 warp 覆盖 2×4 的 fragment 网格，每个 warp 负责一个 16×16 输出块。

## 卡在哪

编译失败：

```
mma.h(55): catastrophic error: cannot open source file "crt/mma.h"
```

`<mma.h>` 会 include `crt/mma.h`，后者属于 **CUDA toolkit（nvcc）**，
而本机 CuPy 的 JIT 只依赖 pip 的 `nvidia-cuda-runtime-cu12` 包——
那个包里没有 `crt/` 目录。

尝试补装 `nvidia-cuda-nvcc-cu12` 时 pip 超时（本机到 PyPI 时断时续）。

## 解锁条件（三选一）

1. 安装完整 CUDA Toolkit（12.x），使 `CUDA_PATH/include/crt/mma.h` 存在；
2. `pip install nvidia-cuda-nvcc-cu12`（网络可用时）；
3. 改用内联 PTX `mma.sync.aligned.m16n8k16`，不依赖 `<mma.h>`——
   工作量更大，但能彻底摆脱头文件依赖。

## 启用前必须做的验证

该 kernel **从未执行过**，以下两项都未检查：

1. **数值**：与 `qwen2_w4a16_gemm` 逐元素对比，max abs error 应在 FP16 舍入范围内；
2. **性能**：至少覆盖 (M,N,K) = (32,64,256) 到 (512,896,2048) 的形状，
   并与 CUDA-core 版本对比。

验证脚本可参考 `scripts/` 下其它 microbench 的结构（CuPy `RawModule` + 双 kernel 对比）。

## 为什么这件事不一定值得继续

现有 prefill GEMM 是 **CUDA core + FP32 累加**，理论算力远低于 Tensor Core。
但本项目的瓶颈是**带宽**，不是算力：

| 路径 | 有效带宽 | 是否算力受限 |
|---|---:|---|
| 流式（4 GB 卡跑 MoE） | PCIe 约 7 GB/s | **否**，实测达成率仅 54% |
| 常驻（3B/7B 全在显存） | VRAM 约 192 GB/s | 可能受算力限制 |

见 `docs/CROSS_MODEL_COMPARISON.md`。因此：

- 在 **流式路径**上，Tensor Core 的收益预计很小——那块时间本来就花在等 PCIe；
- 只有在 **常驻型路径**（3B/7B）上才可能有明显作用。

**投入更多时间前应当先做一件事**：测一下 3B prefill 里 GEMM 占多少时间。
如果占比低于 30%，即使 Tensor Core 把 GEMM 提速 3 倍，端到端也只能快不到 10%。
