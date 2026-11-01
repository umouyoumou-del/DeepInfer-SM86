# 交接：DeepInfer-SM86 长上下文工作

> 接手者：Claude
> 仓库 `github.com/umouyoumou-del/DeepInfer-SM86`，分支 `main`；本文写于 `88beb19`，2026-11-01 晚的复核见下
> 本文只写**做完了什么、实测数字、哪些结论被撤回、现在卡在哪、下一步的确切改法**。

---

> ## ⚠️ 2026-11-01 晚，接手复核（Claude）
>
> 按本文 §3 自己的要求——每条结论问一次"哪个实验能区分它和它的反面"——逐条复核。新实测见 §2、§5.4，
> 细节在 `docs/LONG_CONTEXT_PLAN.md` §13，原始数据在 `docs/results/needle_paged/`、`docs/results/paged_decode/`。
>
> - **§5 的阻塞已解除。** harness 按 §5.2 修好（`scripts/needle_paged.py`），两道自检都过：分块不变性
>   （2K 前缀 NLL 3.0092 vs 3.0093；旧的逐块分配在同一前缀上 NLL 8.38）；paged fp16 对 dense 参考
>   （模型自带注意力 + DynamicCache）文档 NLL 差 +0.0006、top-1 98.6%、回答逐字相同。
> - **"全量必须 5/5"不是 harness 的性质。** "列出全部码字"dense 也 0/5，且与 paged 逐字相同。逐个提问后
>   dense / paged fp16 / paged int8 全量都是 4/5（30% 那一问三者都答成别的码字：混淆，不是丢失）；
>   window 4096 → 3/5、window 1024 → 1/5，正好是窗口里的那几个。
> - **"128K、108.78 tok/s"是一层注意力内核的时间（1000 ÷ 9.2 ms），不是解码速度。** 整模型实测：
>   112K int8 专家驻留 window 4096 **2.56 tok/s**、全量 0.15；128K int8 只有专家改为流式才放得下
>   （专家驻留时模型本体 11.14 GiB，不是 9 GiB），window 4096 **1.92 tok/s**、全量 0.13。
> - **§3 第 1 条撤回本身是错的。** 逐头 K/V 不同 ≠ 必须逐头缓存：上投影可以并进 query 和输出。区分实验：
>   `tests/moe/test_latent_mla.py::test_one_shared_latent_gives_the_per_head_attention_at_the_model_shape`
>   （16 头、kv_b (4096, 512)，参考实现与 Triton 融合内核都与逐头展开的定义一致）；端到端 1M、2M 召回 9/9
>   （`docs/LONG_CONTEXT_1M.md`，latent fp8 17.3 KiB/token）。所以"256K 装不下""1M 110–324 GiB"只对逐头展开布局成立。
> - **"原生 160,384、不需要 YaRN"：** `config.json` 是 `max_position_embeddings 163840 = 4096 × YaRN factor 40`、
>   `rope_scaling.type = yarn`——YaRN 就是模型自己的位置编码，任何长度都生效。
> - **§7.3"2M 全量召回 8.6 年"** 是按每个 query 从显存重读全部 KV 估算的；1M 全量 prefill 实测 35.1 min，
>   2M 有界编码 20.2 min，召回 9/9。
> - **新发现：paged 路径本身慢。** 同一进程 8K decode：dense 41.8 ms/token，paged fp16 160.6，paged int8 417.7（decode 部分已由下面的 split-K 解决）。
>   int8 文档 NLL 只多 0.0046 nat，但 6.5% 的位置 top-1 变了（§8 第 5 项的第一份实测）。
> - **paged decode 内核已换成 split-K（flash-decoding）**：`deepinfer/moe/paged_mla_split.py`。原内核每个
>   (head, query) 一个 warp、从 token 0 走到底，decode 时全卡只有 16 个 warp。一层 128K 全量 276 → 2.76 ms，
>   4K 窗口 9.0 → 0.14 ms；整模型 128K int8（专家流式）window 4096 1.92 → **5.41 tok/s**，全量 0.13 → **4.53 tok/s**；
>   8K paged decode 161 → 51 ms/token（dense 42）。8K 的 25 个回答、召回、NLL 与换内核前逐字相同。
> - **32K（Chat 的训练长度）**：全量 dense / fp16 / int8 都是 4/5；window 16384 → 2/5，window 4096 → 1/5。
> - 安全：本文曾写入远程 root 密码并推到公开仓库；已强制推送去除、密码已轮换。**凭据一律不入库。**

## 0. 一句话状态

**int8 paged 路径端到端可用，与 dense 参考逐字一致；128K int8 放得下（专家需流式），decode 内核换成 split-K 后整模型 4.5–5.4 tok/s（全量 / window 4096），原来 0.13–1.9。**
**召回实验已修好并立了基线：8K、32K 逐个提问全量 4/5（唯一的错 dense 也错），窗口只召回窗口内的事实。**

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

## 2. 128K int8：放得下；decode 内核换成 split-K 前后（2026-11-01 晚复测）

| 项 | 值 |
|---|---|
| 上下文 | 131,072（128K） |
| KV | int8，20.47 GiB（164 KiB/token，逐头展开、K/V 都补齐到 192） |
| 模型本体（实测） | 专家全部驻留 **11.14 GiB**；专家流式 2.16 GiB |
| 合计 | 专家驻留：11.14 + 20.47 > 31.47，**放不下**（112K 可以）；专家流式：放得下，剩 8.48 GiB |
| 位置编码 | YaRN（模型 config 自带：factor 40 × 4096 = 163,840），一直生效 |

| 配置 | 一层注意力内核：原 → split-K | 整模型 ms/token：原 → split-K | 整模型 tok/s：原 → split-K |
|---|---:|---:|---:|
| 128K，专家流式，全量 | 276.2 → **2.76** ms | 7,617 → **221** | 0.13 → **4.53** |
| 128K，专家流式，**window 4096 + sink 4** | 9.00 → **0.14** ms | 520 → **185** | 1.92 → **5.41** |
| 112K，专家驻留，全量 | 241.5 → **2.34** ms | 6,595 → **119** | 0.15 → **8.43** |
| 112K，专家驻留，window 4096 + sink 4 | 8.49 → **0.23** ms | 390 → 118\* | 2.56 → 8.46\* |
| 112K，专家驻留，window 8192 + sink 4 | 13.4 → **0.17** ms | 613 → 65–103\* | 1.63 → 9.7–15.4\* |
| 112K，专家驻留，window 16384 + sink 4 | 37.5 → **0.40** ms | 1,085 → **66** | 0.92 → **15.1** |
| 短上下文（past 256），专家驻留 / 流式 | — | 129 / 211 → 118\* / 212 | 7.8 / 4.7 → 8.5 / 4.7 |

\* 这台机器有 CPU 慢态，主机侧时间翻倍：window 8192 那组 10 步里出现过 103 → 65 ms 的跳变。快态下整步基线约
55 ms，112K 全量再加 27 × 2.34 = 63 ms，正好 118 ms。跨配置比较以一层内核时间为准。split-K 之后 decode 的主要
成本已经不在注意力：128K 专家流式时是专家搬运（短上下文也要 212 ms）。

原表"window 4096：9.2 ms / 108.78 tok/s"是第一列（一层、单次、未预热）换算成 tok/s；整步要过 27 层再加 MoE。
数据：`docs/results/paged_decode/112k_int8_resident.json`、`128k_int8_streamed.json`（原内核）与 `*_split.json`（split-K），脚本 `scripts/bench_paged_decode.py`。
原写的 `docs/results/mla_128k_int8_window.txt` 不在仓库里。

### 容量表（只算 KV，单位 GiB；模型本体另计：专家驻留 11.14，专家流式 2.16；卡 31.47）

| 上下文 | 逐头展开 fp16（repo） | 逐头展开 int8（repo） | 逐头最小布局 int8 | **共享 latent fp8（已实现）** |
|---:|---:|---:|---:|---:|
| 128K | 40.5 | 20.5 | 13.9 | **2.2** |
| 160K | 50.6 | 25.6 | 17.4 | **2.7** |
| 256K | 81.0 | 40.9 | 27.8 | **4.3** |
| 1M | 324 | 164 | 111 | **17.3**（单卡实测 9/9） |
| 2M | 648 | 327 | 223 | **34.6**（显存 1.51M + 主机 0.58M，实测 9/9） |

**逐头展开 int8：专家驻留时单卡到 112K，专家流式到 128K。共享 latent：单卡 1M，加主机内存 2M**（`docs/LONG_CONTEXT_1M.md`，模型 Coder-V2-Lite）。

---

## 3. ⚠️ 被撤回的结论——以及对撤回本身的复核（2026-11-01 晚）

| 撤回的主张 | 原撤回理由 | 复核：区分实验 → 结论 |
|---|---|---|
| "MLA latent 所有 head 共享，每 head 一份是 16× 浪费" | `attn_kv_b` 是 (4096, 512)，逐头 K/V 不同，一份 per head 正确；冗余只有 1.48× | **撤回错了。** 逐头 K/V 不同，不等于必须逐头缓存：W_UK 并进 query、W_UV 并进输出，缓存只需共享的 512 + 64。`test_one_shared_latent_gives_the_per_head_attention_at_the_model_shape`：16 头、kv_b (4096, 512)、逐头不同的上投影，单份 latent 的注意力与逐头展开的逐 (i, j) 定义一致（参考与 Triton 融合内核都过）；端到端 1M/2M 9/9。旧夹具"一份 latent 复制 16 份"确实把前提写进了夹具——这个测试没有 |
| "1M 只需 20.25 GiB" | 那个布局装不下真实 KV | latent fp8 实测 1M = **17.3 GiB**，单卡 9/9 |
| "1M 下 31.8 tok/s" | 在不可能的 pool 上测的 | 撤回对，但原因是**一层内核计时**；同类的"128K 108.78 tok/s"见 §2 |
| "`write()` 仍写 16 份是缺口" | 逐头 KV 本来就正确 | 对逐头展开的 paged 池成立；容量上真正的改法是换 latent 池（§8） |
| "int8 scale pool 未透传是集成缺口" | 测试脚本漏传参数 | 同意 |
| "内核并行度差是 MLA 普遍问题" | 只是 fp16 内核：int8 离下限 11×，fp16 293× | **"11×"是拿一层的时间比 27 层的字节。** int8 一层读 4,100 个 key（约 25 MB）实测 8.5 ms，约 3 GB/s，离 736 GB/s 约 250×；8K 全量整模型 int8 418 ms/token，比 fp16 的 161 还慢。根因是并行度（每个 query 一个 warp），fp16 也一样；split-K 后一层 0.23 ms、8K 整步 51 ms |

**共性：六条里至少四条是"把我的假设或我的调用错误记成代码的问题"。**
接手后对每条结论问一次：**哪个实验能区分它和它的反面？**——第三列就是照这句话做的。

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

## 5. 现在卡在哪——已解决（2026-11-01 晚，见 §5.4）

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

### 5.4 已解决：harness 修好，自检通过，基线与深度曲线（2026-11-01 晚）

`scripts/needle_paged.py`：一个序列只分配一次 block 列表，每个 chunk / decode step 都传这个列表与真实 past_length；
`dense:full`（模型自带注意力 + DynamicCache，与 paged 内核无关的实现）作参考。
数据：`docs/results/needle_paged/8k_each.json`、`8k_list.json`。

| 自检 | 结果 |
|---|---|
| 分块不变性（2K 前缀，一次 prefill vs 512 分块） | NLL 3.0092 vs 3.0093，最后 argmax 相同，top-1 98.7% ✅；**旧的逐块分配在同一前缀上 NLL 8.38**——自检抓得住原 bug |
| paged fp16 全量 vs dense | 文档 NLL +0.0006，top-1 98.6%，**五个回答逐字相同** ✅ |

**"全量必须 5/5"不是 harness 的性质。** "列出全部码字"这一问 dense 也 0/5（回答"1. DeepInfer 2. SM86 3. RTX…"，与 paged 逐字相同）：
8K 技术文档里模型做不了这个任务。改为逐个提问（5 道门各一个码字，同一次 prefill 上分支提问）：

| 配置（V2-Lite-Chat，8K） | 10% | 30% | 50% | 70% | 90% | 召回 | 文档 NLL 差 | top-1 一致 | decode ms/token |
|---|---|---|---|---|---|---:|---:|---:|---:|
| dense 全量（参考） | ✅ | ✗ 答 VESSEL | ✅ | ✅ | ✅ | 4/5 | — | — | 41.8 |
| paged fp16 全量 | ✅ | ✗ 答 VESSEL | ✅ | ✅ | ✅ | 4/5 | +0.0006 | 98.6% | 160.6 |
| paged int8 全量 | ✅ | ✗ 答 KESTREL | ✅ | ✅ | ✅ | 4/5 | +0.0046 | 93.5% | 417.7 |
| int8 window 4096 + sink 4 | ✗ 答 TALLOW | ✗ 答 VESSEL | ✅ | ✅ | ✅ | 3/5 | +0.0146 | 90.8% | 234.7 |
| int8 window 1024 + sink 4 | ✗ 编 NORTHGATE | ✗ 编 GATE_EAST | ✗ 编 SOUTH | ✗ 编 GATEWEST | ✅ | 1/5 | +0.146 | 80.7% | 106.9 |

- 30% 那一问 dense 也错，答成另一道门的码字：是混淆，不是丢失。
- 窗口召回的正好是窗口里的事实；窗口外的被编造的码字顶替——§7.3 的取舍，现在有了数。
- int8 KV：困惑度几乎不变（+0.46%），但 6.5% 的位置 top-1 变了（fp16 paged 对 dense 只有 1.4%）。
- 上表 decode 列是原内核；换成 split-K 后重跑（`8k_each_split.json`）：25 个回答、召回、NLL 逐字相同，
  decode fp16 160.6 → 50.6、int8 417.7 → 51.4、window 4096 234.7 → 49.8 ms/token（dense 42.0）。

**32K（Chat 的训练长度），同样的五问**（`docs/results/needle_paged/32k_each.json`，原内核）：

| 配置 | 10% | 30% | 50% | 70% | 90% | 召回 | 文档 NLL 差 | top-1 一致 | prefill | decode ms/token |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| dense 全量（参考） | ✗ 答 TALLOW | ✅ | ✅ | ✅ | ✅ | 4/5 | — | — | 69.0 s | 85.8 |
| paged fp16 全量 | ✗ 答 TALLOW | ✅ | ✅ | ✅ | ✅ | 4/5 | 0.0000 | 98.6% | 116.7 s | 597.8 |
| paged int8 全量 | ✅ | ✗ 答 TALLOW | ✅ | ✅ | ✅ | 4/5 | +0.0034 | 95.1% | 225.7 s | 1,736 |
| int8 window 16384 + sink 4 | ✗ 编 NORTHGATE | ✗ 编 GATE | ✅ | ✗ 答 HARROW | ✅ | 2/5 | +0.0156 | 93.1% | 185.8 s | 927.6 |
| int8 window 4096 + sink 4 | ✗ 编 HAMBURG | ✗ 编 DULING | ✗ 编 HAMBURG | ✗ 编 HASTE | ✅ | 1/5 | +0.204 | 84.9% | 99.9 s | 287.5 |

paged prefill 比 dense 慢 1.7–3.3×（32K：fp16 117 s、int8 226 s，dense 69 s）：prefill 仍走单遍内核，见 §8。

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

### 7.3 2M：逐头布局下是显存墙，latent 布局下已经做到（复核）

| 定义 | 成本 | 可行 |
|---|---|---|
| 2M，逐头展开 int8 | KV 223–327 GiB | ❌ 单卡装不下 |
| 2M，共享 latent fp8 | KV 34.6 GiB（显存 1.51M + 主机 0.58M） | ✅ **实测召回 9/9**（Coder-V2-Lite，`LONG_CONTEXT_1M.md`） |
| 2M 全量注意力 prefill | 608 PFLOP；分块 FlashAttention 是算力受限 | 约 2–2.5 h。原"读 2.25e17 B、8.6 年"按每个 query 从显存重读全部 KV 估算，不成立；1M 全量实测 35.1 min |
| 2M 有界编码（窗口 96K）prefill | 线性 | ✅ **实测 20.2 min** |
| 2M 流式（sink + 窗口，丢弃中间） | O(1) 内存 | 只能"处理过"，召回只剩窗口内（§5.4 的 8K 表就是它的小规模版） |

**正确表述：逐头布局只能"处理过 2M token"；latent 布局能召回 2M 中任意位置（定位 + 重读）。**

---

## 8. 建议的推进顺序（2026-11-01 晚更新）

| 顺序 | 事项 | 状态 / 理由 |
|---|---|---|
| ~~1~~ | 修 harness + 立基线 | ✅ §5.4：自检改为"paged 与 dense 一致"；逐个提问全量 4/5 |
| ~~2~~ | 召回 vs 深度曲线 | ✅ §5.4：8K、32K |
| ~~3~~ | paged MLA **decode** 内核 | ✅ split-K（`paged_mla_split.py`）：一层快 37–103×；128K 整模型 0.13–1.9 → 4.5–5.4 tok/s |
| **3′** | **paged MLA prefill**：仍是每个 (head, query) 一个 warp 走全部历史 | 32K prefill int8 226 s / fp16 117 s，dense 69 s；做法同 decode（按 key 切分 + 合并），或 prefill 走 flash 再把 K/V 写进页 |
| **4** | **serve 路径换 latent 池**（每 token 512 + 64，复用 `latent_mla_triton` 的 absorbed decode） | 取代原"修 1.48× 填充"：逐头最小布局 int8 111 KiB/token，latent fp8 17.3 KiB/token；容量差 9.5×（逐头 int8 164 KiB/token，latent fp8 17.3 KiB/token）；速度上 1M latent 全量 decode 230–370 ms/token，paged int8 128K 全量 split-K 后 221 ms/token |
| 5 | int8 对生成质量 | 第一份数：NLL +0.46%、top-1 93.5%；还缺长文本 / 任务级评测 |
| 6 | 流式 2M 摄取（sink + 窗口） | 只有"处理过"的语义；要召回走 latent 路径（已 9/9） |

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
| **一层计时当 tok/s** | "108.78 tok/s" = 1000 ÷ 一层 9.2 ms | 计时整步 `forward_logits`，预热后取中位数（`bench_paged_decode.py`） |
| 模型本体按估算 | 估 9 GiB，专家驻留实测 11.14 GiB → 128K int8 放不下 | 容量用实测（脚本会打印） |
| 自检设成模型做不到的事 | "列出全部码字"dense 也 0/5 | 自检比的是"paged 与 dense 一致"，模型能力单独报 |
| Git Bash 改写远程路径 | sftp `No such file` | `MSYS_NO_PATHCONV=1` |

---

## 10. 文档索引

| 文件 | 内容 |
|---|---|
| `docs/LONG_CONTEXT_PLAN.md` | 12 节，含 §0 严重更正、§10-12 的容量表与失败记录 |
| `docs/LONG_CONTEXT_ARCHITECTURE.md` | 九项算法对现状的映射、检索的三个坑、实施顺序 |
| `docs/HANDOFF.md` | 本文 |
| `docs/results/mla_128k_int8_window.txt`、`mla_kv_reload_tiers.txt` | **不在仓库里**（原文引用，未提交） |
| `scripts/needle_paged.py` | paged 路径召回实验，带分块不变性与 dense 参考两道自检 |
| `scripts/bench_paged_decode.py` | 整模型 decode vs 一层注意力内核 |
| `deepinfer/moe/paged_mla_split.py` | split-K paged MLA decode（fp16 / int8 / fp8 池，sink + window）；`DEEPINFER_PAGED_MLA_SPLIT=0` 关闭 |
| `tests/moe/test_paged_mla_split.py` | split-K 对逐查询的直接定义（24 组 + 边界） |
| `docs/results/needle_paged/` | 8K 逐个提问（`8k_each.json`，split-K 后 `8k_each_split.json`）、列表任务（`8k_list.json`）、32K（`32k_each.json`） |
| `docs/results/paged_decode/` | 112K 专家驻留 / 128K 专家流式的 decode，原内核与 `*_split.json` |
| `docs/LONG_CONTEXT_1M.md` | 共享 latent 布局：1M / 2M 召回 9/9 |
| `docs/OLMOE_PORT_PLAN.md` | OLMoE 移植（另一条线，38 节） |
| `docs/PAPER_TODO.md` | 论文缺口清单 |
