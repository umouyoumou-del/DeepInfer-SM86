# ECS: 面向显存受限设备的专家协调系统

**ECS: An Expert Coordination System for Memory-Constrained MoE Inference**

---

## 摘要

混合专家（MoE）模型的参数量与其激活量之间存在数量级差距，理论上适合在显存受限的设备上运行；
但把这一理论优势落地，需要在权重供给、KV 管理、请求调度和跨设备协同四个层面同时重构推理栈。
本文报告 ECS（Expert Coordination System）的设计、实现与实测结果。ECS 是一个非官方扩展，
建立在 DeepInfer 的 model_executor 之上，目标平台是单张 4 GB 笔记本 GPU，以及后续的双卡 RTX 3080 Ti。

在单卡 4 GB 上，ECS 以 CPU/GPU 分级存储和流式专家供给，使 9.65 GiB 的 DeepSeek-V2-Lite Q4_K_M
在 2.23 GiB 峰值显存下完成推理（2.64–3.01 tok/s）；同一套内核在 Qwen2.5-3B 上达到 42.7 tok/s 稳态。
在双卡配置上，ECS 将路由专家下放到第二张卡，远端 rank 不加载 transformer，
峰值占用仅 38 MiB，并在 8 个并发请求下达到 25.90 tok/s 的聚合解码吞吐（单卡的 6.17 倍）。

本文同时报告三类**否定性结果**，我们认为它们对该方向的研究者同样有价值：
（1）paged KV 的收益严格依赖上下文长度，在 17-token prompt 上与整块缓存无法区分；
（2）prefill/decode 重叠在 NCCL 上不可实现，根因是一个 process group 的点对点通信共享单条内部流，
而非调度器设计；（3）以专家为单位的分发在单 worker 下会退化为串行队列，
复制 worker 是提高上限的必要条件，但受硬件限制未能验证。

**关键词**：混合专家模型；显存受限推理；KV 缓存分页；专家并行；NCCL

---

## Abstract

Mixture-of-Experts (MoE) models separate total parameters from activated parameters by orders of
magnitude, which in principle suits memory-constrained hardware. Realising that advantage requires
rebuilding the inference stack at four layers simultaneously: weight supply, KV management, request
scheduling, and cross-device coordination. This report describes ECS (Expert Coordination System),
an unofficial extension built on DeepInfer's model_executor, targeting a single 4 GB laptop GPU and,
later, a pair of RTX 3080 Ti cards.

On one 4 GB device, tiered CPU/GPU storage with streamed expert supply runs the 9.65 GiB
DeepSeek-V2-Lite Q4_K_M checkpoint at 2.23 GiB peak (2.64-3.01 tok/s), while the same kernels reach
42.7 tok/s steady state on Qwen2.5-3B. On two cards, ECS places routed experts on the second device
without loading the transformer there (38 MiB peak) and reaches 25.90 tok/s aggregate decode at eight
concurrent requests, 6.17x the single-card figure.

We also report three negative results that we believe are useful to others pursuing this direction:
paged KV only pays off as context grows, prefill/decode overlap is blocked by NCCL's single internal
communication stream rather than by scheduler design, and per-expert dispatch degenerates into a serial
queue with a single worker.

**Keywords**: MoE; memory-constrained inference; paged KV cache; expert parallelism; NCCL

---

## 1. 引言

MoE 的卖点是"参数量大、激活量小"。DeepSeek-V2-Lite 有 15.7B 参数，但每个 token 只激活其中一小部分。
按这个逻辑，它应该能塞进比 15.7B 小得多的显存里。

实际做起来，这个"应该"卡在四个地方：

1. **权重供给**：即使只激活一部分，权重文件本身有 9.65 GiB（Q4_K_M），不可能整体装进 4 GB。
2. **KV 管理**：MoE 只压缩了 FFN，attention 的 KV 仍然随上下文线性增长。
3. **请求调度**：单请求跑通不等于能服务多个请求；批处理会引入新的内存压力。
4. **跨设备协同**：想把专家摊到多张卡上，就要解决激活值交换、负载均衡和故障恢复。

现有工作大多假设显存充足（≥24 GB），在这些假设下 MoE 推理已有成熟方案。
本文的问题是：**当显存预算比模型小一个数量级时，最小可行的推理栈长什么样？**

我们的定位是工程报告而非算法创新。三个贡献：

- **C1**：一套在 4 GB 上运行的 MoE 执行栈，包含分级存储、流式专家供给和 fused packed-weight 内核，
  使 9.65 GiB 权重在 2.23 GiB 峰值下完成推理（§4.1，§5.2）。
- **C2**：在双卡上用 ECS 完成专家分片执行，远端 rank 不加载 transformer，token 与单卡完全一致
  （§4.3，§5.3）。
- **C3**：在 ECS 之上实现 paged KV slot allocator 与前缀块共享，并给出它们的**适用范围边界**
  （§4.4，§5.4）。

此外，§6 用一整节报告否定性结果，因为我们认为它们比正面结果更容易被误判。

---

## 2. 背景与相关工作

**MoE 推理。** Switch Transformer 与 GShard 确立了稀疏激活的基本形式：router 为每个 token 选择
k 个专家，专家并行执行后加权合并。DeepSpeed-MoE 在训练侧讨论了专家并行与通信开销。
Mixtral 与 DeepSeek-V2 把这条路推进到开源模型。DeepSeek-V2 的另一处不同是 MLA
（Multi-head Latent Attention），它压缩 KV 而不牺牲太多质量，这对显存受限场景格外重要。

**KV 缓存分页。** PagedAttention 把 KV 切成固定大小的 block，用 block table 做间接寻址，
消除了连续分配带来的碎片与预分配浪费；vLLM 在此基础上实现了连续批处理。
本文的 paged KV 直接沿用这套抽象，但需要适配 MLA 的 head 维度与权重格式。

**推理系统的显存管理。** 主流做法包括权重量化（GPTQ/AWQ/GGUF 系列）、
CPU offload（按层或按专家）、以及分布式切分。ECS 的组合方式是：
**权重量化 + 按专家粒度的流式供给 + 跨设备专家并行**，并且把"不 OOM"作为硬约束而非优化目标。

**与本文的关系。** 上述工作大多假设设备显存不小于模型激活峰值。ECS 关注的是更极端的一端，
并且明确记录在这个区间里哪些优化会失效。

---

## 3. 系统设计

### 3.1 分层存储

ECS 把专家权重放在三级存储里：

| 层级 | 位置 | 用途 |
|---|---|---|
| L1 | GPU 显存 | 当前 chunk 的 packed 专家权重、attention、共享专家 |
| L2 | CPU 内存（pinned） | 预取槽位、专家缓存 |
| L3 | NVMe SSD | GGUF 全量权重，按需读取 |

对象粒度是 **expert chunk**（默认每 chunk 最多 8 个专家），而不是整层或整模型。
这个粒度的选择来自一个实测约束：整层 offload 会带来约 1.86 tok/s 的回退，
而专家粒度加冷热切分可以做到约 2.50 tok/s（§5.2）。

### 3.2 控制面与数据面分离

ECS 的控制器（`ECSController`，C++ 实现 heat/LRU/tier 策略）驻留在 CPU，负责：

- 路由热度的统计与持久化；
- 缓存的保留与淘汰决策；
- 跨设备的放置（`ECSRouter` 按队列深度与 KV 压力选择设备）。

GPU 侧只负责计算当前传入的专家。两者之间通过明确的回调接口通信，
使控制器可以独立测试，也可以在单测里用模拟的设备后端跑通。

### 3.3 流式专家供给

一次前向的执行序列：

1. router 产出每层的专家 ID 与 token 分配；
2. 按 chunk 切分，构造本次需要的 packed 权重读取请求；
3. 后台线程把下一 chunk 读入 pinned 槽位（双缓冲）；
4. 主线程做 H2D 拷贝，然后发 grouped kernel。

关键实现约束：**权重不做反量化展开**。fused kernel 直接消费 Q4_K/Q5_0/Q8_0 的 packed 行，
避免了"解量化到 FP16 → 读 HBM"的往返。这条改动是单卡路径从"能跑"到"能用"的分界点。

### 3.4 双卡专家分片

第二张卡（rank 1）是一个**不加载 transformer 的专家执行器**：它只打开 GGUF 中路由专家对应的字节区间，
接收激活值，执行 grouped gate/up/down，回传结果。协调卡（rank 0）保留 attention、共享专家和 LM head。

| 角色 | 持有 | 峰值显存 |
|---|---|---|
| rank 0（协调） | attention + shared experts + LM head + 部分专家 | 2.21–2.28 GiB |
| rank 1（专家执行器） | 仅专家权重与激活缓冲 | **38–62 MiB** |

### 3.5 请求调度

在 ECS 之上实现连续批处理：请求动态进入空闲 slot，完成后释放；KV 按 block 分配，
每个请求持有自己的 block 列表；解码时只重建 block table，不重排任何一行。

前缀共享通过 block 级累积 hash 实现：命中则 retain，之后只对后缀做前向，
写回时保证**已共享的 block 不可变**——block 对齐的 prompt 不会被交出最后一块，
因为那一块仍需写入本行尾部。

---

## 4. 实现

### 4.1 单卡执行栈（4 GB）

| 组件 | 实现 |
|---|---|
| 权重量化 | W4A16 / W3A16（group 128；3-bit 把 8 个权重压进 3 字节） |
| GEMV | 手写 CUDA，batch tile 8 |
| prefill GEMM | 32×64 CTA tile，CUDA core + FP32 累加（**未使用 Tensor Core**） |
| attention | 原生 GQA decode kernel（static）+ paged MLA（双卡路径） |
| KV | 静态 cache + CUDA Graph；paged block 池；FP16/INT8/FP8 |
| 调度 | 固定槽位（低延迟）与连续批处理（吞吐）两条路径 |

### 4.2 MoE 内核

三种 fused 路径：Q4_K、Q5_0、Q8_0。它们直接消费 packed expert 行，
每个 chunk 的 gate/up 与 down 各只发一次 kernel。作为回退保留 `--no-fused-moe` 的
"反量化 + grouped bmm"路径，用于数值对照。

### 4.3 传输层

协调卡与专家执行器之间使用 NCCL 点对点 request/reply：固定长度 JSON header + 激活张量。
一次请求对应一次 `isend`（header、tokens）与一次 `irecv`（输出）。

**该协议的实现约束（重要）**：协议是严格 FIFO、无请求 ID，worker 按到达顺序回包。
因此它只能由单一生产者驱动。我们后来把客户端改为「提交入队 + 单 I/O 线程排空」，
使调用方可以来自多线程而不破坏协议顺序（§6.2）。

### 4.4 Paged KV 与前缀共享

- `PagedKVCacheManager` 持有物理 block 池；每请求持有 `block_ids` 并按需逐块增长；
- decode 每步重建 `[batch, max_blocks]` block table；
- 两种 prefill：批量左填充 dense 前向后一次性写回（默认），或逐请求直接 paged 前向；
- 前缀共享仅在 paged prefill 路径可用（dense 批量路径从位置 0 写整行，无法从偏移开始）。

---

## 5. 评测

### 5.1 实验环境

| 配置 | 单卡 | 双卡 |
|---|---|---|
| GPU | RTX 3050 Laptop 4 GB (SM86) | 2× RTX 3080 Ti 12 GB（无 P2P） |
| CPU / RAM | Ryzen 5 5600H / 16 GB | Xeon Silver 4214R / 440 GiB |
| 系统 | Windows 11 | Ubuntu 22.04 |
| PyTorch / CUDA | 2.5.1+cu121 | 2.5.1+cu121 |

### 5.2 单卡：4 GB 上能跑到什么程度

| 模型 | 量化 | 峰值显存 | 解码 |
|---|---|---:|---:|
| Qwen2.5-3B | W4A16 + GQA + Graph | 2.49 GiB | **42.7 tok/s**（稳态） |
| Qwen2.5-3B（常驻网关） | W4A16 + 256 context | ~2.5 GiB | 44.8 tok/s / 0.895 s 每请求 |
| Qwen2.5-7B | 4-bit attention/embed + 3-bit MLP/head | 3.20 GiB | **25.96 tok/s**（均值） |
| Qwen2.5-7B | 整体 4-bit | 3.92 GiB | 7.94 tok/s（持续 paging） |
| DeepSeek-V2-Lite | Q4_K_M + grouped fused | 2.23 GiB | 2.64–3.01 tok/s |

MoE 路径的消融（同一份 Q4_K_M、贪心解码、单请求）：

| 变体 | 解码 |
|---|---:|
| 反量化 + grouped bmm | 2.43–2.77 tok/s |
| 逐专家 fused | 2.56–3.05 tok/s |
| **grouped fused + 双缓冲** | **2.64–3.01 tok/s** |
| 整层 CPU offload（1 层） | 1.86 tok/s |
| 冷热专家切分（2 层 × 1 cold） | 2.50 tok/s |

**对照基线**：同一份 Q4_K_M 在 llama.cpp CPU 路径为 9–13 tok/s。
因此单卡 MoE 路径的结论应表述为「在不 OOM 的前提下可运行」，而非性能优势。

### 5.3 双卡：专家分片与批处理

单请求与批量（贪心、短 prompt）：

| 路径 | 解码 | 相对单卡 |
|---|---:|---:|
| 单卡基线 | 4.20 tok/s | 1.00× |
| 双卡同步 NCCL 专家并行 | 3.44 tok/s | 0.82× |
| 双卡异步 NCCL | 4.90 tok/s | 1.17× |
| 双卡混合（prefill 批量 + decode 异步） | 4.71 tok/s | 1.12× |
| 固定 batch 2 | 6.32 tok/s（聚合） | 1.50× |
| 固定 batch 4 | 16.60 tok/s（聚合） | 3.95× |
| **固定 batch 8** | **25.90 tok/s（聚合）** | **6.17×** |

所有路径的输出 token 与单卡基线完全一致。

### 5.4 连续批处理、paged KV 与前缀共享

四个 staggered 请求（0.15 s 间隔），双卡：

| 配置 | wall | 聚合解码 |
|---|---:|---:|
| dense slot cache，整批 admission | 6.13 s | 4.24 tok/s |
| paged KV，17-token prompt | 5.13 s | 5.07 tok/s |
| dense，同 workload | 5.13 s | 5.07 tok/s |
| paged KV，**44-token prompt** | **5.97 s** | **4.53 tok/s** |
| dense，44-token prompt | 7.00 s | 3.86 tok/s |

前缀共享（4 个请求共享 89-token 前置说明）：

| 配置 | wall | 每请求 prefill |
|---|---:|---|
| 无前缀缓存 | 8.36 s | 2.52 / 1.17 / 1.22 / 1.21 s |
| **前缀缓存（复用 5 个 block）** | **6.15 s** | 2.47 / **0.58 / 0.52 / 0.54** s |

### 5.5 正确性

- 单卡与双卡 token 完全一致（`[207, 20, 12, 17, 28, 18, 100001]`）；
- paged 引擎在容量为 1 的串行配置下复现单请求输出；
- 每次 paged 运行结束 `used_blocks = 0`，KV 池完整归还；
- 全量单元测试 127 项通过（含 allocator、前缀不可变约束、引擎端到端、传输分发）。

---

## 6. 否定性结果

### 6.1 paged KV 的收益边界

我们最初的假设是：整块 KV 缓存在 admission 时的重排与拷贝是吞吐瓶颈。
实现 paged 分配器后，实测推翻了这个假设：

| prompt 长度 | dense | paged |
|---|---:|---:|
| 17 token | 5.13 s | 5.13 s |
| 44 token | 7.00 s | **5.97 s** |

在短上下文下两者无法区分。拷贝 `[rows, heads, tokens, dim]` 只有在 `tokens` 变大时才是真实开销。
真正的差距在 admission 同步和「每个调度迭代只推进一步 decode」。

### 6.2 prefill/decode 重叠在 NCCL 上不可行

我们尝试把 admission prefill 放到独立线程与独立 CUDA 流上。为此先把专家登台缓冲区
（pinned 槽位、copy 事件、copy 流）改为按线程隔离——这是任何并发方案的前提。
随后在双卡上运行，进程以 `SIGABRT` 终止。

把提交路径改为「单 I/O 线程排空」以排除发送端交织后，问题依旧。结论是：

> 一个 `ProcessGroupNCCL` 的点对点通信运行在单条内部通信流上。
> 两个前向同时需要专家交换时，无论由哪个线程发起调用，都会竞争同一条流。

这不是调度器层面的问题，客户端队列无法解决。

### 6.3 单 worker 是吞吐上限

把每个 chunk 整块发给某一个远端 worker，在只有一个 worker 时等价于一个串行队列。
扩展到 8 并发时，聚合吞吐受限于该 worker 的服务速率，而不是协调卡的调度能力。
复制 worker 是提高上限的必要条件；`PooledExpertClient` 已实现分发逻辑（按 in-flight 最少选择、
回包归属、不平衡指标），但实验环境只有两张卡，**从未承载真实流量**，因此不构成本文的实证贡献。

---

## 7. 讨论

**方法学教训。** 本研究过程中出现三次判断错误，其中两次可归为「测量不足」，
一次可归为「叙事替代验证」：

1. 未经测量就假定拷贝是瓶颈（§6.1）；
2. 改动调度循环后未配对测量，一个 0.5 s 的阻塞把收益吃成 47% 的回退；
3. 把自设护栏抛出的 `ValueError` 误读为 NCCL 崩溃，并将其写入文档作为结论，
   事后才通过完整日志纠正。

第 3 条的特殊性在于：失败信号来自我们自己插入的断言，而不是被测系统。
我们因此建议：**跨会话恢复或长时间实验时，应交叉验证三类信息**——
会话日志（意图，可能错误）、版本控制历史（代码实际状态）、测试套件（当前是否可运行）。

**适用边界的表述。** 我们认为"在 X 显存上跑 Y 模型"这类陈述如果没有限定词是没有意义的。
本文所有数字都标注了路径（静态 Graph / paged 连续批处理 / 固定 batch），
因为它们的数量级差异恰恰来自路径选择而非实现质量。

---

## 8. 结论与后续工作

ECS 证明了在显存比模型小一个数量级的设备上运行 MoE 是可行的：9.65 GiB 权重在 2.23 GiB 峰值显存下
以 2.64–3.01 tok/s 完成推理，双卡配置下 8 并发达到 25.90 tok/s 聚合吞吐且 token 与单卡一致。
同时，我们明确了三项在此环境下失效的优化，并给出失效原因。

后续工作按依赖顺序排列：

1. **worker 复制与监督**：需要至少三个 rank；包含单 worker 心跳与死 worker 摘除。
2. **激活分片**：按专家归属切分 chunk，配每 worker 路由表，而非整块投递。
3. **collective all-to-all**：替换点对点 request/reply；这也是 prefill/decode 重叠的前置条件。
4. **在飞请求续跑**：当前 worker 崩溃导致整请求重试，KV 状态不恢复。

---

## 附录 A：可复现命令

```bash
# 单卡 MoE 冒烟
PYTHONPATH=$PWD python scripts/smoke_deepseek_v2_lite_moe.py \
  --prompt "只回答数字：5-2=?" --max-new-tokens 8

# 双卡连续批处理（paged KV）
PYTHONPATH=$PWD torchrun --standalone --nproc_per_node=2 \
  scripts/bench_ecs_paged_continuous_batch.py \
  --max-batch-size 4 --max-new-tokens 8 --arrival-delay 0.15 --max-cache 256

# 双卡前缀共享
PYTHONPATH=$PWD torchrun --standalone --nproc_per_node=2 \
  scripts/bench_ecs_paged_continuous_batch.py \
  --max-batch-size 4 --max-new-tokens 8 --arrival-delay 0.15 --max-cache 256 \
  --prefill-mode paged --prefix-cache-blocks 64
```

## 附录 B：数据结构与结果文件

- 完整实验日志（含失败实验）：`docs/REMOTE_ECS_2X3080TI.md`
- 二卡定版范围：`docs/ECS_TWO_GPU_RELEASE.md`
- 结构化结果：`docs/results/ecs_*.json`
- 实现记录（十篇）：`docs/ARTICLE_1..10_*.md`

## 参考文献

1. Fedus W., Zoph B., Shazeer N. *Switch Transformers.* JMLR, 2022.
2. Lepikhin D., et al. *GShard.* ICLR, 2021.
3. Shazeer N., et al. *Outrageously Large Neural Networks.* ICLR, 2017.
4. Rajbhandari S., et al. *DeepSpeed-MoE.* ICML, 2022.
5. Kwon W., et al. *Efficient Memory Management for LLM Serving with PagedAttention.* SOSP, 2023.
6. Yu G.-I., et al. *Orca.* OSDI, 2022.
7. Dao T., et al. *FlashAttention.* NeurIPS, 2022.
8. DeepSeek-AI. *DeepSeek-V2.* arXiv:2405.04434, 2024.
9. Jiang A. Q., et al. *Mixtral of Experts.* arXiv:2401.04088, 2024.
10. Frantar E., et al. *GPTQ.* ICLR, 2023.
11. Lin J., et al. *AWQ.* MLSys, 2024.
