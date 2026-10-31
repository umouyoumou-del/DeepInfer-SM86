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

---

## 10. 移植进度：QK-norm 已完成

新增 `deepinfer/moe/olmoe_attention.py`：

```python
per_head_rms_norm(tensor, weight, num_heads=16, head_dim=128, eps=1e-5)
```

OLMoE 的 `attn_q_norm` / `attn_k_norm` 是 **2048 维 = 16 头 × 128**，
所以**每个头有自己独立的 128 个缩放值**（而不是所有头共享一个向量）——
这一点与 Qwen3 那种 `(head_dim,)` 的写法不同，容易搞错。

实现：reshape 成 `[..., heads, head_dim]`，在 `head_dim` 上求 RMS，
再乘以 `weight.reshape(heads, head_dim)`，最后按原 dtype 还原。

### 验证

`tests/moe/test_olmoe_attention.py`，4 个测试：

| 测试 | 内容 |
|---|---|
| 与朴素参考实现逐元素比对 | 双循环版本，`atol=1e-5` |
| 各头独立归一化 | 两个方差不同的头，归一化后 RMS 都等于 1 |
| 宽度校验 | 权重宽度或头数不匹配时抛 `ValueError` |
| dtype 保持 | fp16 进、fp16 出 |

```
tests/moe   81 passed
全量        177 passed
```

### 进度更新

| 项 | 状态 | 估计 |
|---|---|---|
| Q6_K dequant | ✅ 完成并验证 | — |
| 专家读取 | ✅ 零改动可用 | — |
| **逐头 QK-norm** | ✅ **完成并验证** | — |
| MHA 适配（复用 paged GQA 内核） | 未开始 | 0.5–1 天 |
| 无共享专家短路 | 未开始 | 0.5 天 |
| Router（F32 → softmax → top-8） | 未开始 | 0.5 天 |
| ModelSpec + token 对齐 | 未开始 | 1 天 |
| **剩余合计** | | **2.5–3 天** |

三个"格式/数学"类模块（Q6_K、专家读取、QK-norm）全部完成并有测试覆盖，
剩下的都是**接线工作**——把 MA 注意力、router、无共享专家这三条路径接进
一个 OLMoE 专用 runtime。

---

## 11. 移植进度：Router 已完成

新增 `deepinfer/moe/olmoe_router.py`：

```python
route_topk(hidden, gate_weight, top_k=8) -> (expert_ids, weights)
```

与 DeepSeek 的差异：

| | DeepSeek-V2 | **OLMoE** |
|---|---|---|
| 门控 | sigmoid + 分组限制 | **softmax（全部 64 专家）** |
| 分组 | 有（`n_group` / `topk_group`） | **无** |
| 权重归一化 | 有 | **不做**（top-8 之和 < 1 是正确行为） |

最后一条最容易搞错：**OLMoE 的 top-8 权重是原始 softmax 概率，不重新归一化。**

### 验证

`tests/moe/test_olmoe_router.py`，5 个测试：

| 测试 | 内容 |
|---|---|
| 与朴素参考实现比对 | 逐 token 循环 + 手写 exp/log 求和 |
| **不重归一化** | 均匀 logits 下 top-8 之和应为 `8/64 = 0.125`；若重归一化会得 1.0 |
| 选择有序且不重复 | 权重递减、每行 8 个不同专家 |
| `top_k == 专家数` | 退化为完整 softmax，和为 1 |
| 形状校验 | 门控转置或非 2-D 输入抛 `ValueError` |

> 第一版"不重归一化"测试用的是随机 logits，随机下 top-8 之和本就可能 >0.999，
> **测不出区别**；改为均匀 logits 后才具备判别力。

```
tests/moe   86 passed
全量        182 passed
```

### 进度更新

| 项 | 状态 | 估计 |
|---|---|---|
| Q6_K dequant | ✅ 完成并验证 | — |
| 专家读取 | ✅ 零改动可用 | — |
| 逐头 QK-norm | ✅ 完成并验证 | — |
| **Router** | ✅ **完成并验证** | — |
| MHA 适配（复用 paged GQA 内核） | 未开始 | 0.5–1 天 |
| 无共享专家短路 | 未开始 | 0.5 天 |
| ModelSpec + token 对齐 | 未开始 | 1 天 |
| **剩余合计** | | **2–2.5 天** |

**四个独立可验证的模块全部完成**，剩下一件真正的架构工作（MHA 接进 runtime）
加两件接线工作。

---

## 12. "复用 paged GQA 内核"：先被怀疑，最终证实成立

§9 曾据此把 MHA 适配估为 0.5–1 天，理由是"`head_count_kv == head_count` 时
现有 GQA 内核退化为 MHA，可直接复用"。

### 排查过程（三次）

**第一次：手搓 pool。** 形状被接受，但数值与 SDPA 差 2.56。

**第二次：改用 `PagedLayerKVCache.update()` 写入。** 仍差 2.44。
此时怀疑"内核契约未理解"或"MHA 不被支持"。

**关键实验：同一夹具跑 GQA（kv_heads=4 < heads=16）。**

```
GQA  heads=16 kv=4   maxdiff=2.456
MHA  heads=16 kv=16  maxdiff=2.437
```

**两者一样失败** → 问题在夹具，不在 MHA 路径。

**第三次：照 `scripts/validate_sm86_kernels.py` 的真实构造方式重写。**
用 `PagedKVCacheManager` + `write_tokens` + `make_views(max_length=PAST+1, new_tokens=1)`
+ `views[0].update(...)`，参考值取自 `views[0].materialize()`：

```
test_gqa_baseline_matches    PASSED   (heads=8, kv=2)
test_mha_matches_reference   PASSED   (heads=8, kv=8)
```

**MHA 复用成立。** 之前两次失败是因为自己拼缓存，没有走 manager 的
`block_table` / `max_length` / `new_tokens` 约定。

### 教训

第三次排查里唯一的"决定性实验"是**用 GQA 做对照**——它把"内核坏了"
和"夹具坏了"区分开了。前两次都在猜原因，而对照组一次就定位。

> **怀疑某个路径有问题时,先在同一夹具下跑一条已知正常的路径。**
> 如果它也失败,问题就在夹具,不在路径。

这与今天其他几次错误同源:先有假设,再找证据;
而正确顺序是先设计能区分两种可能的实验。

### 对排期的影响

MHA 适配回到 **0.5–1 天**，剩余总量 **2–2.5 天**（§11 的数字维持不变）。

测试现状：**184 passed**（含 GQA 基线 + MHA 各一例）。

---

## 13. 移植进度：MoE 块（含"无共享专家"）已完成

新增 `deepinfer/moe/olmoe_mlp.py`：

```python
moe_forward(hidden, gate_weight, expert_gate, expert_up, expert_down,
            top_k=8, activation="silu")
```

与 DeepSeek 的结构差异就这一点：**OLMoE 没有 shared expert，
输出只是被选中专家的加权和**。

### 验证（`tests/moe/test_olmoe_mlp.py`，5 个测试）

| 测试 | 内容 |
|---|---|
| 与独立公式比对 | 参考实现用 `sorted()` 选专家、按专家主序散射累加，与 `topk` 路径完全不同 |
| SiLU / GELU 两种激活 | 参数化 |
| **无共享专家** | 把专家权重全置零，输出必须**严格为零**——若存在隐藏的 shared-expert 项，这条会失败 |
| top_k=1 退化 | 必须等于单个专家的 SwiGLU 乘以它的路由权重 |
| 非法输入 | 未知激活、专家数不一致 → `ValueError` |

第二条是专门针对"OLMoE 没有共享专家"这个结构差异写的：
**它把"多出一条恒等通路"这种错误变成可检测的。**

```
tests/moe   91 passed
全量        189 passed
```

### 进度

| 项 | 状态 |
|---|---|
| Q6_K dequant | ✅ 验证 |
| 专家读取 | ✅ 验证 |
| 逐头 QK-norm | ✅ 验证 |
| Router | ✅ 验证 |
| MHA 复用 paged 内核 | ✅ 验证 |
| **MoE 块（无共享专家）** | ✅ **验证** |
| 组装 OLMoE runtime + ModelSpec | 未开始（1–1.5 天） |
| token 对齐验证 | 未开始（0.5 天） |
| **剩余** | **1.5–2 天** |

所有**可独立验证的数学与格式模块**都已就绪，剩下的是把它们组装成一个
runtime 并做端到端 token 对齐。

---

## 14. 移植进度：ModelSpec 已完成

新增 `deepinfer/moe/olmoe_spec.py`：

```python
spec = spec_from_gguf(GGUFIndex(path))
# block_count=16  head_count=head_count_kv=16  head_dim=128
# expert_count=64 expert_used_count=8  is_multi_head=True  kv_group_size=1
# has_shared_experts=False      moe_layer_indices=(0..15)
```

设计意图：**把结构事实从元数据读出来，而不是从张量名推断**——
这正是 §9 错误的直接补救。

### 自动校验的假设

| 检查 | 作用 |
|---|---|
| `general.architecture == "olmoe"` | 防止误喂 DeepSeek 的 GGUF |
| `hidden % head_count == 0` | head_dim 必须整除 |
| `head_count % head_count_kv == 0` | 保证 `kv_group_size` 有意义 |
| 必需字段存在 | 缺字段立即报错，而不是默认成 0 之后静默出错 |
| **扫描 `_shexp` 张量** | 若存在共享专家则 `has_shared_experts=True`——**用事实覆盖假设** |
| 扫描 `ffn_gate_exps` | 得出真正的 MoE 层列表（不假定每层都是 MoE） |

### 验证（6 个测试）

前 5 个用合成元数据覆盖正常路径与四种错误路径；
第 6 个在真实 GGUF 存在时运行，断言
`is_multi_head`、`has_shared_experts is False`、`head_dim == 128` 等**事实**。

```
tests/moe   97 passed
全量        195 passed
```

### 进度

| 项 | 状态 |
|---|---|
| Q6_K dequant / 专家读取 / QK-norm / Router / MHA 内核 / MoE 块 | ✅ 全部验证 |
| **ModelSpec** | ✅ **验证** |
| 组装 runtime | 未开始（1–1.5 天） |
| token 对齐 | 未开始（0.5 天） |
| **剩余** | **1.5–2 天** |

到这一步，OLMoE 需要的**每一个独立组件**都已存在且被独立测试覆盖。
剩下的工作是把它们串起来——一件需要连续时间的事。

---

## 15. 移植进度：权重命名与形状校验已完成

新增 `deepinfer/moe/olmoe_weights.py`：

```python
tensors = layer_tensors(spec, 0)
# attn_q / attn_k / attn_v / attn_output  -> [hidden, hidden]
# attn_norm / ffn_norm / attn_q_norm / attn_k_norm -> [hidden]
# ffn_gate_inp -> [experts, hidden]
# ffn_gate_exps / ffn_up_exps -> [experts, inter, hidden]
# ffn_down_exps -> [experts, hidden, inter]

report = validate_layer(index, spec, 0)   # 逐项校验，不匹配立即报错
```

### 真实 checkpoint 的两个新发现

| 张量 | 量化类型 | 说明 |
|---|---|---|
| `blk.0.attn_v.weight` | **Q6_K** | 又一处 Q6_K——§8 新增的 kernel 支持正好用得上 |
| 无 `blk.*.ffn_*_shexp` | — | 确认无共享专家 |

### 校验覆盖

| 检查 | 作用 |
|---|---|
| 所有期望张量存在 | 缺一个立刻报错，而不是在 forward 中途炸 |
| attention 形状 `[hidden, hidden]` | 与 spec 的 hidden 交叉验证 |
| norm 形状 `[hidden]` | |
| expert 张量为 3-D 且首维 = 专家数 | |
| gate/up 的末维 = hidden | |
| **down 的中间维 = gate/up 的中间维** | 最容易接错的一处 |
| 层号范围 | |

### 验证（6 个测试）

4 个合成场景覆盖正常与三种错误路径（缺张量、attention 形状错、down 中间维不一致），
2 个对**真实 GGUF 的第 0 层与第 15 层**做完整校验。

```
tests/moe  103 passed
全量       201 passed
```

### 进度

| 项 | 状态 |
|---|---|
| Q6_K / 专家读取 / QK-norm / Router / MHA / MoE 块 / ModelSpec | ✅ |
| **权重命名与形状校验** | ✅ **验证** |
| 组装 runtime 前向 | 未开始（1–1.5 天） |
| token 对齐 | 未开始（0.5 天） |
| **剩余** | **1.5–2 天** |

---

## 16. 整层组装已通过 transformers 对照（里程碑）

新增 `deepinfer/moe/olmoe_layer.py`：把 norm、QKV、RoPE、QK-norm、attention、
残差、MoE 块串成一层。

**验证方式**：与 `transformers.models.olmoe.modeling_olmoe.OlmoeDecoderLayer`
逐元素比对（小配置 + 随机权重，`atol=1e-4`）。这是移植过程中**最强的一次验证**
——norm 位置、RoPE 约定、QK-norm 作用范围、残差顺序、无共享专家，
五件事必须同时正确才能对上。

```
test_matches_transformers_layer   PASSED
test_no_shared_expert_path        PASSED   (专家置零后与 HF 同样置零一致)
全量 203 passed
```

### 过程中发现的两个真实错误

**错误 1：QK-norm 的位置。**
我按"RoPE 之后做 QK-norm"实现。transformers 是
`q_norm(q_proj(x))` —— **在 RoPE 之前**。看源码才能确认。

**错误 2（更严重）：QK-norm 的作用范围。**
我在 §10 实现了 `per_head_rms_norm`，并断定 OLMoE"Is 逐头在 head_dim 上做 RMSNorm"。
**错。** transformers 用 `OlmoeRMSNorm(hidden_size)`，
作用于**投影后的整个 hidden 向量**（2048），然后才 reshape 成 16 个头。

也就是说：**`attn_q_norm` 的 2048 维不是"16 头 × 128"，而就是一个 2048 维向量**
——这个数字和"16×128"巧合相等，正是它让我推出了错误结论。

`olmoe_attention.py` 的文档已更正，`per_head_rms_norm` 保留为通用工具但注明**不是** OLMoE 用的。

### 第三个问题在测试而非实现

对照测试一开始差 0.75，原因不是实现，而是**参考实现缺 causal mask**——
transformers 的 layer 由 model body 外部传入 mask，不传就是双向注意力。
按 causal 实际行为要求补上后一致。

> 这是今天第七次"问题在夹具或参考，而不在被测代码"。

### 进度

| 项 | 状态 |
|---|---|
| Q6_K / 专家读取 / Router / MHA 内核 / MoE 块 / ModelSpec / 权重校验 | ✅ |
| **QK-norm（范围已更正）** | ✅ |
| **整层组装** | ✅ **与 transformers 逐元素一致** |
| 组装完整 runtime（嵌入 + 16 层 + LM head） | 未开始（0.5–1 天） |
| token 对齐 | 未开始（0.5 天） |
| **剩余** | **1–1.5 天** |

---

## 17. 完整模型组装已通过 transformers 对照

新增 `deepinfer/moe/olmoe_model.py`：

```python
logits = forward(input_ids, weights, layer_weights,
                 num_heads=16, eps=1e-5, rope_base=10000.0, top_k=8,
                 tie_word_embeddings=False)
```

嵌入 → 16 层 → final norm → LM head，含 causal mask。

### 验证

与 `transformers.OlmoeForCausalLM` 整体对照（2 层小配置 + 随机权重，`atol=1e-4`）：

| 测试 | 结果 |
|---|---|
| 共享词嵌入（`tie_word_embeddings=True`） | PASSED |
| 独立 LM head（`tie_word_embeddings=False`，即真实 OLMoE 的配置） | PASSED |
| 层数为 0 时报错 | PASSED |

```
全量 206 passed
```

**两种词嵌入配置都对**，这一点重要：OLMoE-1B-7B 的 GGUF 里有独立的
`output.weight`（§15 已确认），所以实际走的是 `tie=False` 分支；
但把两种都测过，避免将来换 checkpoint 时踩坑。

### 进度

| 项 | 状态 |
|---|---|
| Q6_K / 专家读取 / Router / MHA / MoE 块 / ModelSpec / 权重校验 / QK-norm | ✅ |
| 整层组装 | ✅ 与 transformers 一致 |
| **完整模型组装** | ✅ **与 transformers 一致（两种词嵌入配置）** |
| 接 GGUF 真实权重跑通 | 未开始（0.5 天） |
| token 对齐验证 | 未开始（0.5 天） |
| **剩余** | **1 天** |

前面所有组件都是"与独立实现比对"，这一步是**第一次端到端整体比对**——
输入 token id、输出 logits，中间 16 层全部参与。它通过意味着整个组装链没有接错。

---

## 18. 真实 GGUF 权重已接入并跑通

`load_layer_tensor` 现在按实际 API 实现（`GGUFIndex.read_tensor(name)` 返回
反量化后的 numpy 数组，文档注明仅适用于非专家权重）。

新增测试对**真实 checkpoint 的第 0 层**执行完整链路：

```
GGUF 读取 → Q4_K/Q6_K 反量化 → attn_norm RMSNorm → q_proj
→ q_norm（全向量 RMSNorm）→ reshape 成 16 头 → RoPE
```

断言：

| 断言 | 作用 |
|---|---|
| dtype 为 float32 | 反量化正确落到目标类型 |
| **所有元素有限** | 没有 NaN/Inf —— 反量化出错时最典型的症状 |
| 形状 `(1, 3, 2048)` | 与 spec 一致 |
| **RoPE 保持每个头的范数** | 旋转是正交变换，范数必须不变；这条能抓住实现错位 |

```
tests/moe  108 passed
全量       207 passed
```

### 进度

| 项 | 状态 |
|---|---|
| 全部组件 + 整层 + 整模型（对照 transformers） | ✅ |
| **真实 GGUF 权重接入** | ✅ **验证** |
| 端到端 token 对齐（对照 llama.cpp 输出） | 未开始（0.5 天） |
| **剩余** | **0.5 天** |

### 与 llama.cpp 对齐需要补的最后一块

目前 `forward()` 接受**全部层权重一次性传入**。真实 checkpoint 是 7B，
fp32 下 28 GB，超出本机 16 GB 内存。要对齐 token，需要：

1. 逐层流式加载（本项目 `GGUFExpertStore` 已有该能力），或
2. 在 32 GB 的 4080 SUPER 机器上以 fp16 加载全部层（约 13 GB）。

后者更直接，且能顺带测出 OLMoE 在 ECS 栈上的真实吞吐——
这正是移植的最初目的（验证 0.73 GB/token 是否带来约 2 倍于 DeepSeek 的速度）。
