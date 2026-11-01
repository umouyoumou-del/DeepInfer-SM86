# 交接：DeepInfer-SM86 长上下文工作

> 接手者：Claude
> 仓库 `github.com/umouyoumou-del/DeepInfer-SM86`，分支 `main`，最新提交 `88beb19`
> 本文只写**做完了什么、实测数字、哪些结论被撤回、现在卡在哪、下一步的确切改法**。

---

## 0. 一句话状态

**128K 上下文在单卡 32 GiB 上跑通了（int8 KV，全驻留，108 tok/s）。**
**端到端真实文本的任务测试失败了，原因是我的测试脚本有 bug，不是代码。**

---

## 1. 环境

| 项 | 值 |
|---|---|
| 机器 | RTX 4080 SUPER 32 GiB（SM89），503 GB 主机内存，128 核 |
| 远程 | 见本机 `~/.ssh/config` 或密码管理器；**凭据一律不入库** |
| 仓库 | `/root/autodl-tmp/DeepInfer-SM86` |
| 模型 | `/root/autodl-tmp/models/DeepSeek-V2-Lite-Chat-Q4_K_M.gguf`，meta `/root/autodl-tmp/meta` |
| python | `/root/miniconda3/bin/python`（torch 2.5.1+cu124，CuPy 14.2） |
| 本地 | Windows，Git Bash，`F:` 盘常满，用 `E:` |

**模型事实**：DeepSeek-V2-Lite，27 层，MLA，16 heads，`qk_nope_head_dim=128`，
`qk_rope_head_dim=64`，`v_head_dim=128`，`kv_lora_rank=512`，
**原生上下文 160,384**，rope base 10000。

---

## 2. 唯一的强结果：128K 上下文，单卡全驻留

| 项 | 值 |
|---|---|
| 上下文 | 131,072（128K） |
| KV | **int8，20.46 GiB（164 KiB/token）** |
| 权重 | fp16，9 GiB |
| 合计 | **29.5 / 31.47 GiB——全部驻留显存，无分层** |
| YaRN | **不需要**（128K < 原生 160,384） |

| 配置 | ms/token | tok/s |
|---|---:|---:|
| 全量注意力 | 548.9 | 1.82 |
| **window=4096, sink=4** | **9.2** | **108.78** |
| window=8192, sink=4 | 14.0 | 71.46 |
| window=16384, sink=4 | 38.0 | 26.29 |

结果文件：`docs/results/mla_128k_int8_window.txt`

### 容量表（含 9.65 GB 权重 + 2 GiB 余量，31.47 GiB 卡）

| 上下文 | repo fp16 | repo int8 | 最小布局 int8 |
|---:|---:|---:|---:|
| 128K | ✗ 49.5 GiB | **✅ 29.2** | ✅ 22.7 |
| 160K | ✗ 58.4 | ✗ 33.7 | ✅ 25.7 |
| 256K | ✗ 90.0 | ✗ 49.5 | ✗ 36.4 |
| 1M | ✗ 324 | ✗ | ✗ 110 |
| 2M | ✗ 648 | ✗ | ✗ 219 |

**单卡上限 128K（现有布局 + int8）。256K 装不下。**

---

## 3. ⚠️ 被撤回的结论——不要在这些之上继续搭

| 撤回的主张 | 为什么错 |
|---|---|
| "MLA latent 所有 head 共享，每 head 一份是 16× 浪费" | `blk.N.attn_kv_b.weight` 是 **(4096, 512)**：512 维 latent 上投影成 16×(128+128) **逐头不同**的 K/V。一份 per head 是正确的。真实冗余只有 **1.48×**（K/V 补齐到 192/head） |
| "1M 只需 20.25 GiB" | 那个布局装不下真实 KV。1M = 324 GiB（repo）/ 219 GiB（最小） |
| "1M 下 31.8 tok/s" | 在不可能的 pool 上测的；只证明"内核在该 pool 大小能跑多快" |
| "`write()` 仍写 16 份是缺口" | 不是缺口，逐头 KV 本来就正确 |
| "int8 scale pool 未透传是集成缺口" | 不是。`make_views` 和 `attend()` 一直在传，是测试脚本漏传参数 |
| "内核并行度差是 MLA 普遍问题" | 只是 **fp16 内核**。int8 内核 4K 窗口离带宽下限 11×，fp16 是 293× |

**共性：六条里至少四条是"把我的假设或我的调用错误记成代码的问题"。**
接手后对每条结论问一次：**哪个实验能区分它和它的反面？**

---

## 4. 已实现并通过测试的九项（205+ 项测试）

| 层 | 项 | 位置 | 状态 |
|---|---|---|---|
| 容量 | Paged KV | `engine/kv_cache.py` | 已有 |
| | **Attention Sink + 滑动窗口** | `moe/paged_mla_kernel.py`、`moe/paged_mla_quant.py` | **新增，fp16 与 int8 都有** |
| | Prefix Caching | `engine/kv_cache.py` | 已有 |
| 检索 | block 级注意力质量 | `moe/block_mass.py` | **新增** |
| | H2O 驱逐 | `block_mass.BlockMassHeat` + `PrefixCache(heat_fn=)` | **新增** |
| | 语义检索（检索原文→重算） | `moe/span_index.py` | **新增** |
| 计算 | Chunked prefill | `paged_continuous_batching.py` 的 `prefill_chunk_tokens` | **新增** |
| | 重算 vs 重载边界 | `LONG_CONTEXT_PLAN.md` §8 | 实测 |
| 模型 | YaRN RoPE | `moe/yarn_rope.py` + 运行时 `rope_scaling` | **新增** |

测试：`tests/moe/test_mla_kv_layout.py`、`test_yarn_rope.py`、`test_block_mass.py`、
`test_h2o_eviction.py`、`test_span_index.py`、`test_paged_continuous_batching.py`

**运行时旋钮**：`rt.paged_sliding_window`、`rt.paged_attention_sink`、
`rt.rope_scaling`、`engine.prefill_chunk_tokens`

---

## 5. 现在卡在哪（最重要的一节）

### 5.1 多事实召回实验：0/5，**结果无效**

8,062 token 真实 haystack，5 个码字插在 10/30/50/70/90% 深度，问"列出所有码字"。

| 配置 | prefill | 召回 |
|---|---:|---:|
| 全量注意力 | 18 s | **0/5** |
| window=1024, sink=4 | 16 s | **0/5** |

两者都输出退化的 `topicNavTypeId:` 重复。

**全量注意力也 0/5 就是 harness 坏了的证据**——8K 全量下五处都应可见。

### 5.2 Bug 的确切位置与改法

`/tmp/needle.py`（实验脚本，未入库）的 prefill 循环：

```python
# 现在（错）
for start in range(0, total, CH):
    blk = mgr.allocate_tokens(end - start)       # 每块新分配
    v = mgr.make_views([blk], [start], ...)      # 只含本块

# 应该
blocks = mgr.allocate_tokens(total)              # 一次性分配
for start in range(0, total, CH):
    end = min(start + CH, total)
    need = (end + BLOCK - 1) // BLOCK
    v = mgr.make_views([blocks[:need]], [start], max_length=total+64,
                       new_tokens=end - start)
```

**原因**：`past_length=start` 声称前面有历史，但 `block_table` 里根本没有那些块，
模型永远只看到最后一个 512-token chunk。

### 5.3 修完必须先立基线

**自检：全量注意力必须召回 5/5。** 达不到就说明 harness 仍然坏，不要往下做。

基线立起来之后才有意义做：窗口 vs 全量的召回曲线（深度 10/30/50/70/90% ×
窗口 1024/4096/全量），预期 sink 区与窗口区召回、中段丢失。

---

## 6. 今天修掉的两个真代码缺口

| 缺口 | 表现 | 已修 |
|---|---|---|
| **int8 走不了 paged 路径** | `native_attention` 门限只认 `dtype == torch.float16`，int8 回退到 dense 并报 `Attention weights should be of size (1,16,512,512), but is (1,16,512,8126)` | 放宽到 int8 / fp8_e4m3 / fp8_e5m2 |
| **窗口/sink 到不了 serve 路径** | 内核有参数，上层拿不到 | `DeepSeekPagedCache` 持有并转发，运行时暴露 `rt.paged_*` |

---

## 7. 关键设计结论（可以沿用的）

### 7.1 检索必须搬文本，不能搬 KV

KV 带**绝对位置标签**（RoPE）。位置 90 万的段落搬进位置 4000 的上下文，**数值上不成立**。
所以：索引文本 → 取原文 → 在当前位置**重算** KV。（`moe/span_index.py` 按此实现）

### 7.2 重算 vs 重载（实测）

| tokens | 重载（H2D 13 GB/s） | 重算 | 重载便宜 |
|---:|---:|---:|---:|
| 64 | 0.104 ms | 11.61 ms | 112× |
| 512 | 0.819 ms | 22.23 ms | 27× |

**成本上重载永远赢**；重算只用于"KV 已丢弃"或"必须搬到新位置"。

### 7.3 2M 是计算墙，不是显存墙

| 定义 | 成本 | 可行 |
|---|---|---|
| 2M 全量召回 | KV 219 GiB(int8 最小) + prefill 读取 **2.25e17 B**（O(N²)） | ❌ 736 GB/s 下 **8.6 年** |
| **2M 流式**（sink+窗口，丢弃中间） | **O(1) 内存** | ✅ 现在能做 |
| 2M 流式 prefill（4K 窗口） | 9.2e14 B → **约 21 分钟** | ✅ |

**正确表述：能"处理过 2M token"，不能"召回 2M 中任意位置"。**

---

## 8. 建议的推进顺序

| 顺序 | 事项 | 理由 |
|---|---|---|
| **1** | **修 harness（§5.2）+ 立全量 5/5 基线** | 唯一缺的证据；不修则所有召回数字无效 |
| 2 | 召回 vs 深度曲线（窗口 1024/4096/全量） | 量化 §7.3 的取舍 |
| 3 | 流式 2M 摄取（`prefill_chunk_tokens` + `sliding_window`） | 用现有零件拼，约 21 分钟填充 |
| 4 | **修 1.48× 填充冗余** | K/V 都补齐到 192/head；最小布局是 K(16×128+64) + V(16×128)。修好后 **160K 原生就装得下**（33.7 → 25.7 GiB） |
| 5 | int8 量化对生成质量的影响 | **完全没测** |
| 6 | fp16 内核并行度（split-K + 两遍 softmax 归约） | int8 内核 11×、fp16 293×，差距在 fp16 |

---

## 9. 环境坑（全部踩过）

| 坑 | 表现 | 解法 |
|---|---|---|
| CRLF | `nohup: failed to run command` | `sed -i 's/\r$//'` 或 `bash script.sh` |
| **僵尸进程占显存** | 自由显存少 12.7 GiB，1M 分配 OOM | `nvidia-smi --query-compute-apps=pid,used_memory` 后 kill |
| CUDA Graph 空图 | `The CUDA Graph is empty` | `cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)` |
| **机器慢态 2×** | 同代码同参数 19.9 → 10.5 tok/s | 记录 `throttled_periods`，多会话取中位数 |
| git push 失败 | `Recv failure: Connection was reset` | 重试，间隔 10–20 s |
| **只跑新测试** | 既有测试在别的入口挂掉（发生过 2 次） | **每次改动跑全量** |
| 旁路 harness 未验证 | 4 次错误结果都归因错了 | 先立已知答案的基线 |

---

## 10. 文档索引

| 文件 | 内容 |
|---|---|
| `docs/LONG_CONTEXT_PLAN.md` | 12 节，含 §0 严重更正、§10-12 的容量表与失败记录 |
| `docs/LONG_CONTEXT_ARCHITECTURE.md` | 九项算法对现状的映射、检索的三个坑、实施顺序 |
| `docs/HANDOFF.md` | 本文 |
| `docs/results/mla_128k_int8_window.txt` | 128K 实测原始输出 |
| `docs/results/mla_kv_reload_tiers.txt` | 重载带宽实测 |
| `docs/OLMOE_PORT_PLAN.md` | OLMoE 移植（另一条线，38 节） |
| `docs/PAPER_TODO.md` | 论文缺口清单 |
