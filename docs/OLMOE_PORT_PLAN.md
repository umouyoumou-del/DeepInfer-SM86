# 移植 ECS 到 OLMoE-1B-7B：耦合点清单与工作量评估

> 状态：**计划文档，尚未实施**
> 触发原因：用户要求在同一套 ECS 栈上跑第二个 MoE 架构（OLMoE-1B-7B），
> 以验证 §8.2 的带宽模型是否跨架构成立。
> 结论先行：当前 ECS 的 MoE 路径是 **DeepSeek-V2 专用**的，
> 移植不是"换一个模型文件"，而是补齐一层架构抽象。

---

## 1. 两个模型的结构差异

| 维度 | DeepSeek-V2-Lite | OLMoE-1B-7B |
|---|---|---|
| 总参数 / 激活参数 | 15.7B / 2.4B | 7.0B / 1.3B |
| attention | **MLA**（compressed KV，`kv_lora_rank`） | **标准 GQA** |
| 专家路由 | top-k + shared experts | **top-8 / 64 experts**，无 shared experts |
| 层布局 | `first_k_dense_replace` + 稠密/MoE 交替 | 全部层同构 MoE |
| 量化文件 | 9.65 GiB（Q4_K_M） | 3.92 GiB（Q4_K_M） |
| 每 token 触达权重 | ≈ 1.35 GB | ≈ 0.73 GB |

最后一行是这次测试的核心：如果带宽模型成立，OLMoE 的吞吐应当约为 DeepSeek-V2-Lite 的 1.85 倍。

---

## 2. 代码里的耦合点

### 2.1 必须改动

**`deepinfer/moe/deepseek_v2_runtime.py`（3309 行，21 处 DeepSeek 字样，4 处 MLA 专用字段）**

- `_paged_forward_logits` 直接用 `config.qk_nope_head_dim` / `config.qk_rope_head_dim` / `config.v_head_dim`
  计算 MLA 的 key/value 维度 —— OLMoE 只需普通 `head_dim`；
- `install_native_paged_attention` 打的是 DeepSeek 的 `self_attn.forward`，OLMoE 的 attention
  签名不同（无 `kv_a_proj_with_mqa` / `kv_b_proj`）；
- 共享专家（shared experts）是 DeepSeek 特有，OLMoE 没有，对应分支需要短路。

**`deepinfer/moe/paged_adapter.py`**

- `DeepSeekPagedCache` 把 key 补齐到 `max(key_dim, value_dim)`（MLA 要求），
  OLMoE 的 K/V 维度相同，这个 padding 反而会浪费 2.7 倍显存；
- `attend()` 走 `paged_mla_*` 内核 —— OLMoE 需要普通 paged GQA 内核
  （好消息：仓库里 `deepinfer/engine/paged_attention.py` 已有 GQA 版本，Qwen 路径在用）。

**`deepinfer/moe/gguf_index.py`**

- `expert_tensor()` 已支持"从 3-D 专家张量里读单个专家"，这部分是通用的；
- 但张量命名（`blk.N.ffn_gate_exps` 之类）与专家轴位置需要在模型描述里参数化。

### 2.2 可以直接复用

| 组件 | 复用度 | 说明 |
|---|---|---|
| `fused_moe.py`（Q4_K/Q5_0/Q8_0 packed kernel） | **完全复用** | 只依赖量化格式，不依赖架构 |
| `grouped_moe.py` | **完全复用** | 同上 |
| `paged_continuous_batching.py` | **完全复用** | 只依赖 runtime 暴露的 `forward_logits` |
| `PrefixCache` / `BlockAllocator` | **完全复用** | 与模型无关 |
| `nccl_expert_transport.py` | **完全复用** | 传的是激活张量，与架构无关 |
| `ecs/`（控制面、分级存储） | **完全复用** | 只认 expert chunk，不认架构 |

**这是一个重要发现**：ECS 里真正与架构耦合的只有
"attention 适配"和"模型描述"两层，其余（量化内核、KV 分页、调度、传输、控制面）
都是架构无关的。

---

## 3. 需要新增的抽象

```python
class ModelSpec:                      # 已有雏形，dense 侧在用
    name: str
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int                  # GQA 需要，MLA 场景为 num_heads
    head_dim: int
    attention_kind: Literal["mla", "gqa"]
    num_experts: int
    top_k: int
    shared_experts: int                # DeepSeek 有，OLMoE 为 0
    expert_tensor_pattern: str         # GGUF 命名模板
    moe_layer_predicate: Callable[[int], bool]
```

- `runtime` 按 `attention_kind` 分派到 MLA 适配器或 GQA 适配器；
- 专家读取按 `expert_tensor_pattern` 定位；
- 共享专家为 0 时整条分支短路。

---

## 4. 工作量估计

| 步骤 | 内容 | 估计 |
|---|---|---:|
| 1 | `ModelSpec` 抽象 + 从 GGUF 元数据自动填充 | 1 天 |
| 2 | GQA paged 适配器（复用现有 Qwen 内核） | 1–2 天 |
| 3 | OLMoE 专家张量定位与读取 | 0.5 天 |
| 4 | 单请求 token 对齐验证（对照 llama.cpp） | 0.5 天 |
| 5 | 接入连续批处理 + 前缀共享 | 0.5 天 |
| 6 | 双卡专家分片验证 | 1 天 |
| | **合计** | **4.5–5.5 天** |

**风险点**：OLMoE 是 dropless MoE（top-8，无 capacity factor），
每层专家数 64 远多于 DeepSeek-V2-Lite 的 64（相同）但激活更多（8 vs 6）——
chunk 大小（当前默认 8 专家/chunk）需要重新调参，否则每个 chunk 会装不下一次路由的全部专家。

---

## 5. 为什么值得做

如果移植成功，能回答一个比"再测一个模型"更重要的问题：

> §8.2 的带宽模型（吞吐 ≈ 有效带宽 ÷ 每 token 触达字节数）是**普适规律**，
> 还是只是 DeepSeek-V2-Lite 一个样本的巧合？

- OLMoE 每 token 触达 0.73 GB，是 DeepSeek-V2-Lite 的 54%；
- 若实测吞吐约为后者的 1.85 倍 → 模型成立，可写进论文作为跨架构验证；
- 若明显偏离 → 说明模型漏掉了固定开销（路由、专家切换、kernel launch），
  论文里的算术需要加入常数项。

两种结果都有价值，后者甚至更有价值。

---

## 6. 当前阻塞

1. **模型下载**：`OLMoE-1B-7B-0924.Q4_K_M.gguf`（3.92 GiB）正在从 HuggingFace 拉取，
   速率约 1.1 MB/s，预计 1 小时；
2. **先做 A（llama.cpp 基线）**：下载完成后即可测，用于拿到对照数字；
3. **B（ECS 移植）**需要上述 5 天工作量，本轮不具备条件。

建议顺序：先完成 A 拿到 llama.cpp 基线，再决定是否投入 B。
如果 A 的结果显示带宽模型在 OLMoE 上明显不成立，B 的优先级应当降低。

---

## 7. 实测：GGUF 结构对比（2026-10-30 更新）

两部分都已在真实文件上核对，结论比 §4 的估计乐观。

### 专家张量布局：**完全兼容**

| | DeepSeek-V2-Lite | OLMoE-1B-7B |
|---|---|---|
| `ffn_gate_exps.weight` | `(64, 1408, 2048)` Q4_K | `(64, 1024, 2048)` Q4_K |
| `ffn_up_exps.weight` | `(64, 1408, 2048)` Q4_K | `(64, 1024, 2048)` Q4_K |
| `ffn_down_exps.weight` | `(64, 2048, 1408)` Q8_0 | `(64, 2048, 1024)` **Q6_K** |
| `ffn_gate_inp.weight` | `(64, 2048)` F32 | `(64, 2048)` F32 |

**同一套命名、同一个 `[experts, out, in]` 轴顺序。**
`GGUFIndex.expert_tensor(name, expert_index)` 是按 3-D 张量写的通用实现，
所以 **OLMoE 的专家读取不需要改一行代码**——这消掉了原计划 §4 第 3 步（0.5 天）。

### 三个真实缺口

1. **`ffn_down_exps` 是 Q6_K**，而 fused MoE kernel 只支持 Q4_K/Q5_0/Q8_0
   （`fused_moe.py:255` 会抛 `NotImplementedError`）。
   → 要么给 fused kernel 加 Q6_K，要么 down 走 dequant+bmm 回退。
   注意 DeepSeek 的 down 是 Q8_0（支持），所以这条路从没被触发过。

2. **attention 是标准 GQA，而且带 QK-norm**
   （`attn_q_norm.weight` / `attn_k_norm.weight` 各 2048 维）。
   DeepSeek 走 MLA（`attn_kv_a_mqa` / `attn_kv_b`），签名完全不同。
   → 需要接 `deepinfer/engine/paged_attention.py` 里已有的 GQA 内核，并补 QK-norm。

3. **无共享专家**（DeepSeek 有 `ffn_*_shexp`）→ 对应分支要短路。

### 修正后的工作量

| 步骤 | 原估计 | 修正 |
|---|---:|---:|
| 专家张量定位与读取 | 0.5 天 | **0（已兼容）** |
| Q6_K down 支持或回退 | — | **+0.5 天** |
| GQA + QK-norm 适配 | 1–2 天 | 1.5–2 天 |
| ModelSpec 抽象 | 1 天 | 1 天 |
| 单请求 token 对齐验证 | 0.5 天 | 0.5 天 |
| 连续批处理 + 前缀共享 | 0.5 天 | 0（复用） |
| 双卡验证 | 1 天 | 1 天 |
| **合计** | **4.5–5.5 天** | **4.5–5 天** |

省下的时间不多，但**风险最大的一步（专家读取）已经被证伪为风险**。

### 为什么这个移植值得做

按 §8.2 的带宽模型，OLMoE 每 token 触达 0.73 GB，
只有 DeepSeek 的 54%。在同一套流式栈上，理论上应该快 **1.85 倍**。
而这正好可以检验"吞吐 ≈ 带宽 ÷ 每 token 字节"是否是跨架构规律——
llama.cpp 侧已经验证（51.1 vs 24.5 tok/s），ECS 侧还没有。

---

## 8. 移植进度（2026-10-31）

### 已完成：Q6_K 支持

`fused_moe` 原本只支持 `Q4_K / Q5_0 / Q8_0`，OLMoE 的 `ffn_down_exps` 是 **Q6_K**。

已加入 `dp_q6_k_value`（`_QTYPE_CODES["Q6_K"] = 3`），按 llama.cpp 的
`block_q6_K` 布局实现：

```
ql[128] | qh[64] | int8 scales[16] | fp16 d   =>  210 bytes / 256 元素
```

`half = k >> 7`，`quarter = (k & 127) >> 5`，四条通路分别取
`(ql[l] & 0xF) | (qh[l] >> 0 & 3) << 4` 等，再乘 `sc[(l>>4) + quarter*2]` 与 `d`。

验证：`tests/moe/test_q6_k_dequant.py` 用**独立的纯 Python 参考实现**
逐元素比对随机块，`rtol=1e-3`，通过。

```
tests/moe    77 passed
全量         173 passed
```

### 剩余工作

| 项 | 状态 | 估计 |
|---|---|---|
| Q6_K dequant | ✅ 完成并验证 | — |
| GQA + QK-norm 适配 | 未开始 | 1.5–2 天 |
| 无共享专家的分支短路 | 未开始 | 0.5 天 |
| ModelSpec 抽象 | 未开始 | 1 天 |
| token 对齐验证 | 未开始 | 0.5 天 |
| 双卡验证 | 未开始 | 1 天 |

**专家读取层已确认零改动可用**（见 `OPTIMIZATION_GENERALITY.md`）。

---

## 9. 架构事实核对（2026-10-31，读 GGUF 元数据）

此前 §7 按张量命名推断"OLMoE 是 GQA + QK-norm"。**核对元数据后，GQA 是错的。**

```
olmoe.attention.head_count      = 16
olmoe.attention.head_count_kv   = 16      <-- 与 head_count 相等
olmoe.embedding_length          = 2048      => head_dim = 128
olmoe.block_count               = 16
olmoe.expert_count              = 64
olmoe.expert_used_count         = 8
olmoe.context_length            = 4096
olmoe.rope.freq_base            = 10000.0
olmoe.attention.layer_norm_rms_epsilon = 1e-5
```

`head_count_kv == head_count` 意味着 **OLMoE 是标准 MHA（多头注意力），不是分组查询注意力**。

### 这对移植的影响：比原估更简单

| 原判断 | 实际 | 影响 |
|---|---|---|
| GQA + QK-norm | **MHA + QK-norm** | `paged_attention.py` 的 GQA 内核在 `num_kv_heads == num_heads` 时退化为 MHA，**可直接复用** |
| 需要 KV 头映射 | 不需要 | 少一处易错逻辑 |

QK-norm 仍然要做：`attn_q_norm.weight` / `attn_k_norm.weight` 各 2048 维，
即 **16 头 × 128 维，逐头在 head_dim 上做 RMSNorm**（eps = 1e-5），
作用在 RoPE 之后、注意力之前。

### 修正后的工作量

| 项 | 状态 | 估计 |
|---|---|---|
| Q6_K dequant | ✅ 已实现并验证 | — |
| 专家读取 | ✅ 零改动可用 | — |
| **MHA 适配（复用 paged GQA 内核）** | 未开始 | **0.5–1 天**（原估 1.5–2） |
| **逐头 QK-norm** | 未开始 | **0.5 天** |
| 无共享专家分支短路 | 未开始 | 0.5 天 |
| Router（`ffn_gate_inp` F32 → softmax → top-8） | 未开始 | 0.5 天 |
| ModelSpec 抽象 + token 对齐 | 未开始 | 1 天 |
| **合计** | | **3–3.5 天**（原估 4.5–5） |

省下的一天来自"不用做 GQA 映射"。

### 教训

**按张量命名推断架构会出错。** `attn_q` / `attn_kv_b`（DeepSeek MLA）
与 `attn_q` / `attn_k`（标准 MHA）看起来相似，但语义完全不同。
元数据里有明确答案，应该先读它再排期。
