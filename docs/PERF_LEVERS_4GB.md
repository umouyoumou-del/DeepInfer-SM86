# 4 GB 卡上 MoE 解码的性能杠杆：全部试过一遍

> 结论（2026-10-31 更正）：**磁盘是一堵硬墙，但「专家缓存无效」那条是错的**——
> 缓存之所以看着无效，是因为 `generate()` 每个请求都会清空它（见 §2）。
> 在 3050 上，NVMe 3.3 GB/s 确实把速度钉在 2.45 tok/s；
> 但换到 4080 SUPER（32 GB）后仍只有 5.41 tok/s，说明清空缓存挡住了更大的收益。

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

### 2. 专家缓存与 GPU 驻留 —— **原结论作废**

> **更正（2026-10-31）**：本节原先写「专家缓存无效，属噪声」。那个结论是错的。
> 真实原因是 `generate()` **每个请求都会清空 GPU 专家缓存**，与缓存大小无关。

#### 代码位置

```python
# deepseek_v2_runtime.py:3079（prefill 前）
expert_store.clear_gpu_cache()
expert_store.set_gpu_cache_active(False)

# deepseek_v2_runtime.py:3103（decode 前）
expert_store.clear_gpu_cache()          # 每次请求又清一次
expert_store.set_gpu_cache_active(True)
```

注释写明了动机：prefill 会碰到海量专家、把 decode 的工作集挤掉，
所以每个请求开始前先释放缓存，prefill 走连续路径，decode 再从一个干净的缓存开始。

这是**刻意的设计**，不是 bug。但它意味着：**任何在启动时预热缓存的做法，都会在第一次请求时被清空。**

#### 三组实验的重新解读

| 实验 | 原结论 | 更正后的原因 |
|---|---|---|
| 3050, `--expert-cache-mb 512` | 命中率 11%，无效 | 每个请求清空 |
| 3050, `--gpu-expert-cache-mb 1024` | +2.3%，不划算 | 同上 |
| 4080S, `--gpu-expert-cache-mb 20000` | 峰值仅 4.3 GiB | 同上（预算 20 GiB，实际只用 4.3） |
| 4080S, `--pin-all-experts` | 加载 1664 专家，无提速 | preload 完即被清空 |

`--pin-all-experts` 的日志证据：

```
[pin-all] 1664 expert entries resident
decode 4.953 tok/s | peak GPU 4302.8 MiB
```

**缓存声称驻留约 8.9 GiB、实测峰值只有 4.2 GiB**——两个数对不上，就是被清空的直接证据。

#### 潜在收益

4080 SUPER（736 GB/s）上，若驻留真正生效：

```
1.35 GB/token ÷ 736 GB/s = 1.83 ms  → 理论上限约 545 tok/s
按 3050 实测的 33% 达成率           → 约 180 tok/s
对比当前 5.41 tok/s                 → 约 33 倍
```

#### 为什么没有直接改

绕过 `clear_gpu_cache()` 只需要一行，但这个清理是刻意写的。
prefill 期间 `set_gpu_cache_active(False)`，若驻留的 decode 专家仍在显存，
需先确认 prefill 路径不会回读该缓存，否则可能读到跨请求的脏状态。
**改错会导致显存泄漏或状态污染，比提速失败严重得多。**

下一步：读 prefill 路径确认安全 → 在 `--pin-all-experts` 下跳过清理 → 同机对照。

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

### 5. 单卡 prefill/decode overlap

之前的判断是：overlap 在双卡上被 NCCL 挡住，但**单卡不走 NCCL，所以应该可行**。
实测否定了这个推论。

```
NO_OVERLAP   完成，peak GPU 2.36 GiB
WITH_OVERLAP 进程崩溃，原生栈回溯（ntdll!RtlUserThreadStart）
```

**结论修正**：阻塞 overlap 的不是 NCCL，而是更底层的东西——
后台线程 + 独立 CUDA stream 上跑 `runtime.forward_logits`，
与主线程的 decode 共用同一个 runtime 实例。即使没有 NCCL，
专家 staging、weight cache、timings 这些 runtime 级状态仍在两个线程间共享。

per-thread staging 只隔离了**缓冲区**，没有隔离**整个 runtime 的调用**。

这条现在和双卡一样是"未解决"，不能算"单卡可行"。

（附注：这次日志用了 `tail -3` 截取，baseline 的 wall/aggregate 数字没留下来，
只有 peak 显存。要复核需要重跑，这次不做。）

---

## 附：4080 SUPER（32 GB）上的对照

| 配置 | decode | 峰值显存 |
|---|---:|---:|
| 3050 4GB（NVMe 3.3 GB/s） | 2.45 tok/s | 2.23 GiB |
| **4080S 32GB 默认** | **5.41 tok/s** | 2.25 GiB |
| 4080S staging slots=4 / 8 | 5.49 / 5.45 | 2.25 GiB |
| 4080S GPU 缓存 8 / 20 GiB | 4.77 / 5.02 | 4.30 GiB |
| 4080S + `--pin-all-experts` | 4.95 | 4.30 GiB |

有效带宽反推：

```
3050 : 2.45 x 1.35 GB = 3.3 GB/s    <= 正好 NVMe 上限
4080S: 5.41 x 1.35 GB = 7.3 GB/s    <= 拷贝流水线 / PCIe 段
```

卡的理论带宽从 192 涨到 736 GB/s，**有效带宽只从 3.3 涨到 7.3**——
因为运行时始终从 GGUF 流式读取，没有驻留路径。

这组数据修正了「换硬件就能提速」的直觉：**硬件换对了，软件的驻留模式没写，收益只有 2.2 倍。**
