# 4 GB 卡上 MoE 解码的性能杠杆：全部试过一遍

> 结论：**没有任何一个软件杠杆能显著提速。**
> 这台机器上 MoE 流式解码的上限约 2.5–3.0 tok/s，是磁盘带宽决定的。

## 背景

流式路径每 token 要触达约 1.35 GB 的专家权重（激活 2.4B × Q4 4.5bit/8）。
实测 2.45 tok/s 对应 **3.3 GB/s 的 NVMe 读取**——这已经接近消费级 SSD 的顺序读上限。

## 逐个杠杆的实测

### 1. staging 流水深度（`DEEPINFER_STAGING_SLOTS`）

| slots | decode | prefill | 峰值 GPU |
|---:|---:|---:|---:|
| 2（默认） | 2.450 tok/s | 1.953 s | 2277.6 MiB |
| 4 | 2.499 tok/s | 1.961 s | 2277.6 MiB |
| 6 | **2.543 tok/s** | 2.030 s | 2277.6 MiB |

**+3.8%**，单调但很小。说明"同时有多少拷贝在飞"不是约束。

### 2. 专家缓存（`--expert-cache-mb`）

| 缓存 | decode |
|---:|---:|
| 0 MB | 2.479 tok/s |
| 256 MB | 2.505 tok/s |
| **512 MB** | **2.580 tok/s** |
| 1024 MB | 2.513 tok/s |
| 2048 MB | 2.462 tok/s |

极差 4.8%，**非单调**，落在噪声区间内。之前单独测 512 MB 时命中率只有 11%，
与这里一致：64 选 6 的路由太分散，工作集装不进缓存；
而且命中时"内存→显存"的拷贝成本 ≈ 重新读盘，收益被抵消。

### 3. CPU 冷专家（`--cpu-experts-per-layer`）

| 配置 | decode |
|---|---:|
| 全 GPU | 2.82 tok/s |
| 2 层 × 1 cold expert | 2.50 tok/s |
| 整层 offload（1 层） | 1.86 tok/s |

**变慢。** 省下 99.9% 的传输，但 CPU 算得比 PCIe 搬得慢。

### 4. Tensor Core prefill

阻塞于缺少 `crt/mma.h`，且**针对流式路径本来就没有意义**——
那条路是带宽受限，算力空闲（见 `docs/CROSS_MODEL_COMPARISON.md`，PCIe 达成率仅 54%）。

## 结论

四个方向全部试过，没有一个能显著提速。这不是调优问题：

```
每 token 需要 1.35 GB
2.45 tok/s × 1.35 GB ≈ 3.3 GB/s  ← 撞在 NVMe 上
```

**在 4 GB 显存 + 单条 NVMe 的配置下，这个数字基本就是硬上限。**

真正能改变它的只有两类：

1. **减少每 token 的字节数**——更激进的量化（2-bit）、或换激活参数更少的模型（如 OLMoE，0.73 GB/token）；
2. **换存储介质**——更快的盘，或让权重常驻显存（需要 ≥10 GB 显存）。

## 复现

```bash
# staging 深度
DEEPINFER_STAGING_SLOTS=4 python scripts/smoke_deepseek_v2_lite_moe.py ...

# 专家缓存
python scripts/smoke_deepseek_v2_lite_moe.py ... --expert-cache-mb 512
```

每点单次测量（n=1），但四个方向的量级差异足够大，结论不受噪声影响。
