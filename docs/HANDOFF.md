# 长上下文工作交接（DeepInfer-SM86）

> ## ⚠️ 2026-11-01 更正（接手后复核）
>
> 本文 §1 的速度数字与 §1/§4 的若干结论不成立，原文保留作记录，复核结果见
> **`docs/LONG_CONTEXT_1M.md` §1**。要点：
>
> - **"128K 108 tok/s"、"1M 31.8 / 1.2 tok/s" 是一层注意力内核的单次计时**
>   （`make_views(...)[0]` 只取 27 层里的第 0 层，无预热），不是解码速度；
>   "int8 内核离下限 11×、fp16 293×" 是拿一层的时间对 27 层的字节，也不成立。
> - **"原生 160,384、128K 不需要 YaRN、质量不打折"**：163,840 是 YaRN 的位置容量；
>   模型卡写的训练长度是 V2-Lite-Chat **32K**、Coder-V2-Lite **128K**；YaRN 一直由
>   `modeling_deepseek` 按 config 生效。
> - **"MLA 逐头存储才正确、单卡上限 128K"**：MLA 应缓存共享 latent（512+64），
>   **30.4 KiB/token（fp16）/ 17.3 KiB/token（fp8）**，1M 只要 17.3 GiB。
> - **"2M 全量召回 8.6 年"**：FlashAttention 是算力受限，2M 约 2–2.5 小时。
> - `yarn_rope.py` 的 ramp 下标错了一倍（已修）。
>
> 远程机器的 root 密码曾写在本文并推到公开仓库，已轮换；凭据不入库。

---

## 0. 硬件与基线

| 项 | 值 |
|---|---|
| 机器 | RTX 4080 SUPER 32 GiB（SM89），503 GB 主机内存 |
| 模型 | DeepSeek-V2-Lite Q4_K_M，27 层，MLA，16 heads，YaRN 位置容量 163,840（训练长度：Chat 32K / Coder 128K，见顶部更正） |
| 远程 | `ssh -p 35868 root@connect.westc.seetacloud.com`（凭据不入库，向用户索取） |
| 仓库 | `/root/autodl-tmp/DeepInfer-SM86` |
| python | `/root/miniconda3/bin/python` |

---

## 1. 最重要的实测结果

### 128K 上下文，单卡全驻留

| 项 | 值 |
|---|---|
| 上下文 | 131,072（128K） |
| KV | **int8，20.46 GiB（164 KiB/token）** |
| 权重 | fp16，9 GiB |
| 合计 | **29.5 / 31.47 GiB，全部驻留显存，无分层** |
| YaRN | 不需要（128K < 原生 160,384） |

| 配置 | ms/token | tok/s |
|---|---:|---:|
| 全量注意力 | 548.9 | 1.82 |
| **window=4096, sink=4** | **9.2** | **108.78** |
| window=8192, sink=4 | 14.0 | 71.46 |
| window=16384, sink=4 | 38.0 | 26.29 |

结果文件：`docs/results/mla_128k_int8_window.txt`

### 容量表（含 9.65 GB 权重 + 2 GiB 余量）

| 上下文 | repo fp16 | repo int8 | 最小布局 int8 |
|---:|---:|---:|---:|
| 128K | ✗ 49.5 GiB | **✅ 29.2** | ✅ 22.7 |
| 160K | ✗ 58.4 | ✗ 33.7 | ✅ 25.7 |
| 256K | ✗ 90.0 | ✗ 49.5 | ✗ 36.4 |
| 2M | ✗ 648 | ✗ | ✗ 219（仅主机内存） |

**单卡上限是 128K（用现有布局 + int8）。**

---

## 2. ⚠️ 被撤回的结论（务必不要沿用）

| 撤回的主张 | 为什么错 |
|---|---|
| "MLA latent 所有 head 共享，每 head 存一份是 16× 浪费" | **错。** `blk.N.attn_kv_b.weight` 是 **(4096, 512)**，512 维 latent 上投影成 16×(128+128) **逐头不同**的 K/V。一份 per head 是正确的。真实冗余只有 1.48×（K/V 补齐到 192/head） |
| "修掉之后 1M 只需 20.25 GiB" | 那是**不可能的布局**。1M 真实成本 324 GiB（repo）/ 219 GiB（最小） |
| "1M 下 31.8 tok/s" | 在 20.25 GiB 的 pool 上测得，那个 pool 装不下真实 KV。只证明"内核在该 pool 大小下能跑多快" |
| "`write()` 仍写 16 份是缺口" | 不是缺口，逐头 KV 本来就正确 |
| "int8 scale pool 未透传是集成缺口" | 不是。`make_views` 和 `attend()` 一直在传，是我的测试脚本漏传参数 |
| "内核并行度差是 MLA 的普遍问题" | 只是 **fp16 内核**的问题。int8 内核 4K 窗口离下限 11×，fp16 是 293× |

**共性：三次把"我的假设/我的调用"错记成"代码的问题"。** 接手时请对每条结论问一次"哪个实验能区分它和它的反面"。

---

## 3. 已实现并通过测试的九项（274 项测试全绿）

| 层 | 项 | 位置 | 状态 |
|---|---|---|---|
| 容量 | Paged KV | `engine/kv_cache.py` | 已有 |
| | Attention Sink + 滑动窗口 | `moe/paged_mla_kernel.py`、`moe/paged_mla_quant.py` | **新增**（fp16 与 int8 都有） |
| | Prefix Caching | `engine/kv_cache.py` | 已有 |
| 检索 | block 级注意力质量 | `moe/block_mass.py` | **新增** |
| | H2O 驱逐 | `moe/block_mass.py` 的 `BlockMassHeat` + `PrefixCache(heat_fn=)` | **新增** |
| | 语义检索（检索原文→重算） | `moe/span_index.py` | **新增** |
| 计算 | Chunked prefill | `moe/paged_continuous_batching.py` 的 `prefill_chunk_tokens` | **新增** |
| | 重算 vs 重载边界 | `LONG_CONTEXT_PLAN.md` §8 | 实测 |
| 模型 | YaRN RoPE | `moe/yarn_rope.py` + 运行时 `rope_scaling` 参数 | **新增** |

测试：`tests/moe/test_mla_kv_layout.py`、`test_yarn_rope.py`、`test_block_mass.py`、
`test_h2o_eviction.py`、`test_span_index.py`、`test_paged_continuous_batching.py`

---

## 4. 关键设计结论

### 4.1 检索必须搬文本，不能搬 KV

KV 带**绝对位置标签**（RoPE）。把位置 90 万的段落搬进位置 4000 的上下文，
**数值上不成立**。所以：索引文本 → 取原文 → 在当前位置**重算** KV。

### 4.2 重算 vs 重载

| tokens | 重载（H2D 13 GB/s） | 重算（fused prefill） | 重载便宜 |
|---:|---:|---:|---:|
| 64 | 0.104 ms | 11.61 ms | 112× |
| 512 | 0.819 ms | 22.23 ms | 27× |

**成本上重载永远赢**；重算只在"KV 已被丢弃"或"必须搬到新位置"时才用。

### 4.3 2M 到底能不能做

| 定义 | 成本 | 可行 |
|---|---|---|
| **2M 全量召回** | KV 219 GiB(int8 最小) + prefill 读取 2.25e17 B（O(N²)） | ❌ **计算上不可行**（736 GB/s 下 8.6 年） |
| **2M 流式**（sink+窗口，丢弃中间） | **O(1) 内存** | ✅ 现在就能做 |
| 2M 流式 prefill（窗口 4K） | 读取 9.2e14 B → 约 **21 分钟** | ✅ 可行 |

**prefill 的 O(N²) 才是真正的墙，不是显存。**
所以正确表述是"处理过 2M token"可以，"能召回 2M 中任意位置"不行。

---

## 5. 下一步（按价值排序）

### 5.1 端到端真实文本验证（唯一缺的证据）

现在**所有数字都是内核层面**。真实的 128K 文本 prefill + generate + 检查答案**没跑过**。

注意：**不要先量 token 数就开跑**（我犯过，直接 OOM）。先打印 token 数、再算 pool、再定超参。

### 5.2 流式 2M 摄取

用 `prefill_chunk_tokens` + `sliding_window` 做增量摄取：
处理 2M token、内存恒定、中间内容靠 `span_index` 检索召回。
预期约 21 分钟填充 + 之后 decode 与窗口同速。

### 5.3 修布局的 1.48× 冗余

repo 把 K/V 都补齐到 192/head。最小布局是 K(16×128+64) + V(16×128)。
修好后 160K 可从 33.7 GiB 降到 25.7 GiB（int8），**160K 原生长度就装得下了**。
价值明确、改动局部。

### 5.4 int8 量化对生成质量的影响

**完全没测。** 128K int8 的输出与 fp16 差多少，未知。

### 5.5 fp16 内核的并行度

int8 内核 4K 窗口离带宽下限 11×，fp16 是 293×。若还要用 fp16 路径，
需 split-K 跨 block 并行 + 在线 softmax 两遍归约。

---

## 6. 环境坑（都踩过）

| 坑 | 表现 | 解法 |
|---|---|---|
| CRLF | `nohup: failed to run command` | `sed -i 's/\r$//'` 或用 `bash script.sh` |
| 僵尸进程占显存 | 自由显存少 12.7 GiB，1M 分配 OOM | `nvidia-smi --query-compute-apps=pid,used_memory` 然后 kill |
| CUDA Graph 空图 | `The CUDA Graph is empty` | 用 `cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)` 包住 |
| 机器慢态 | 同代码同参数慢 2×（19.9 → 10.5 tok/s） | 记录 `throttled_periods`，多会话取中位数 |
| git push 失败 | `Recv failure: Connection was reset` | 重试，间隔 10-20 s |
| 只跑新测试 | 既有测试在别的入口挂掉 | **每次改动跑全量**（274 项） |
