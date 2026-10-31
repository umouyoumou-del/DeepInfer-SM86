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

---

## 19. 当前阻塞点与下一位接手者的最短路径

移植的**代码部分已完成**（§16–§18）：整层与整模型与 transformers 逐元素一致，
真实 GGUF 权重可读、可反量化、可前向。

剩下的是**端到端 token 对齐**，卡在"把 3.92 GB 的 OLMoE GGUF 放到一台
有 32 GB 显存的机器上"。尝试记录：

| 途径 | 结果 |
|---|---|
| ModelScope `lmstudio-community` / `second-state` / `Oliverhzk` | 仓库不存在（404） |
| `hf-mirror.com` + `mav23/OLMoE-1B-7B-0924-GGUF` | 镜像可达（返回 404 而非超时），但文件名未猜对 |
| 直接从本机上传 3.92 GB | 可行但慢（约 1 小时），未做 |

### 最短路径（建议）

**从本机上传**，不要再猜远端仓库名：

```powershell
# 本地 F: 盘可用，先把模型打成可分片上传的包
# 远端用 tar 从 stdin 解包（本项目已验证该方式可行）
```

或用 `hf-mirror.com` 但**先列出仓库文件**再下载：

```bash
curl -s "https://hf-mirror.com/api/models/mav23/OLMoE-1B-7B-0924-GGUF" | head -c 2000
```

### 拿到模型之后的三步

1. 以 fp16 逐层加载（16 层 × 约 800 MB ≈ 13 GB，32 GB 卡够）；
2. 用 `olmoe_model.forward()` 生成 greedy token；
3. 与 `llama.cpp`（`ollama run`）对同一 prompt 的输出逐 token 比对。

**第 3 步通过 = 移植完成**，同时能拿到 OLMoE 在 ECS 栈上的真实吞吐，
用来验证 `0.73 GB/token` 是否带来约 2 倍于 DeepSeek 的速度。

---

## 20. 模型来源已解决

§19 记录的三条途径里，最后一条走通了：

```
https://hf-mirror.com/bartowski/OLMoE-1B-7B-0924-Instruct-GGUF/resolve/main/
  OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf
→ HTTP 200, 4,213,513,024 bytes
```

关键教训：**hf-mirror 的 API 需要认证（401），但直连文件路径不需要。**
之前 404 是因为仓库名猜错——`bartowski` 是实际可用的那个，
`mav23` / `tensorblock` / `lmstudio-community` 的路由都不通。

**正确顺序是先用搜索引擎确认仓库真实存在，再拼直连 URL**，
而不是从候选名字里逐个试。

### 自动化

远端已启动下载 + 完成后自动执行：

1. 校验大小与 GGUF 魔数
2. `spec_from_gguf` 读元数据（层数、专家数、是否 MHA、有无共享专家）
3. **对全部 16 层逐一 `validate_layer`**——张量缺失或形状不符立即报出

这一步通过，就说明真实 checkpoint 与 §15 的假设完全一致，
剩下的只有前向数值对齐。

---

## 21. 里程碑：OLMoE 端到端跑通

真实 checkpoint（OLMoE-1B-7B-0924-Instruct Q4_K_M，4.21 GB）
在 RTX 4080 SUPER 上完整加载并生成：

```
spec: 16 layers, 64 experts top 8 | MHA: True | shared: False | head_dim: 128
validated 16/16 layers

prompt: "The capital of France is"
generated ids: [7785, 15, 187, 187, 510, 14731, 273, 6181, 310, 14029, 15, 187]
text: " Paris.

The currency of France is Euro.
"
decode 253 ms/token (3.95 tok/s)
```

**输出语义正确**（先是 Paris，然后自发续写货币问题），
说明归一化、RoPE、QK-norm、路由、专家加权整条链路数值上是对的。

### 这条数字本身也有意义

| 栈 | 每 token 激活字节 | 实测 |
|---|---:|---:|
| DeepSeek-V2-Lite（ECS，MLA + 优化后） | 1.35 GB | 20.92 tok/s |
| **OLMoE（本移植，float16 全驻留 + 逐层 Python 循环）** | **0.73 GB** | **3.95 tok/s** |

OLMoE 更慢，因为这条路径是**逐层 Python 循环 + 每个专家单独 matmul**，
完全没有 DeepSeek 运行时那套优化（pin-all、指针数组、fused kernel）。

**这正是移植的价值**：现在有了第二个架构的可跑基线，
把 DeepSeek 侧那套优化应用上去，就能检验
"带宽模型是否跨架构成立"这个悬而未决的问题。

### 过程中修掉的两个环境问题

| 问题 | 原因 | 修法 |
|---|---|---|
| `ModuleNotFoundError: olmoe_spec` | 远端仓库是旧版，新模块没上传 | 重新打包上传 `deepinfer/ scripts/ tests/` |
| `nohup: failed to run /root/gen.sh` | **脚本 CRLF 换行**，`#!/bin/bash
` 使 shebang 失效 | `sed -i 's/
$//'` 或 `bash script.sh` |

第二个尤其值得记：**文件存在但报 "No such file or directory"** 时，
几乎总是 shebang 或换行符问题，不是路径问题。

### 移植完成度

| 项 | 状态 |
|---|---|
| 全部组件 + 整层 + 整模型（对照 transformers） | ✅ |
| 真实 GGUF 16 层全部校验 | ✅ |
| **端到端生成（真实权重、语义正确）** | ✅ |
| 与 llama.cpp 逐 token 对齐 | 未做（需要同 prompt 的 llama.cpp 输出） |
| 把 ECS 优化应用到 OLMoE | 未开始 |

**移植本身已完成**：OLMoE 能在本项目栈上加载并正确生成。
剩下的两项都是后续工作，不再是移植的组成部分。

---

## 22. 性能 profile：瓶颈不是 Python 执行层

有人（用户）提出"Python 执行层太慢，应转 C++ 直接操控显存里的专家"。
按今天反复得到的教训，**先 profile 再决定**。结果与直觉不同。

### 测量（4080 SUPER，真实 checkpoint，单次 forward）

```
wall = 154 ms      cpu = 96 ms      cuda = 253 ms（跨流累计）
总 op 数 = 9,176（约 574 op/层）
```

### 时间去向

| 项 | 每 forward | 次数 | 说明 |
|---|---:|---:|---|
| **`cudaStreamSynchronize`** | **66.8 ms** | **16** | **每层一次，单次 4.2 ms** |
| `aten::copy_`（CUDA） | 37.0 ms | 484 | dtype 转换 |
| `aten::_to_copy` | 36.9 ms | 436 | `.float()` / `.to()` |
| `aten::index`（CUDA） | 24.5 ms | 48 | **gather 选中的专家** |
| `cudaLaunchKernel`（CPU） | 7.0 ms | 1,549 | **Python 编排的全部开销** |

### 关键结论

1. **Python 只占 7 ms / 154 ms（4.5%）**。把执行层改写成 C++ 最多省这么多，
   **不会带来数量级提升**——反而要付出整个 runtime 重写的代价。
2. **最大单项是每层一次的 CUDA 同步，67 ms**（43%）。这是某处 `.item()`
   或等价的显式同步，与语言无关，C++ 里同样会同步。
3. **gather + dtype 转换合计 61 ms**（40%），根因是**把 fp16 专家 gather 出来再转 fp32**，
   这一步既复制又加宽。

### 正确的优化顺序（与"C++ 重写"无关）

| 优先级 | 手段 | 预计 |
|---|---|---|
| 1 | 找出并消除每层的同步点 | 最多省 67 ms（43%） |
| 2 | 用**打包量化权重 + 指针数组 kernel** 取代 gather+转换 | 省约 61 ms（40%） |
| 3 | 才轮到减少 Python 编排（7 ms） | ≤5% |

第 2 项正是 DeepSeek 运行时已有的
`fused_gate_up_gemm_ptr_grouped` / `fused_down_gemm_ptr_grouped`——
它们直接接受**每个专家的指针**，不复制权重。
OLMoE 的 Q4_K / Q6_K 已经被 §8 的 kernel 支持覆盖，
所以这条路是通的，只是还没接。

### 教训（第七次同类）

**"用 C++ 重写会更快"是一个假设，不是结论。**
154 ms 里 Python 只占 7 ms——如果按这个假设动手，
会花数周重写、然后发现只快 4%。

**先 profile，再决定改什么语言。**

---

## 23. 消除同步：只快了 5%，不是 43%

§22 的 profile 指出每层一次同步占 43%。定位方式：`torch.cuda.set_sync_debug_mode("warn")`，
它直接指出行号：

```
deepinfer/moe/olmoe_layer.py:31
inverse = 1.0 / torch.pow(torch.tensor(base, device=device), exponent)
```

`torch.tensor(base, device=...)` 从 Python float 造 GPU 标量 → H2D 拷贝 → 同步。
而 `build_rope` **每层调用一次** → 16 次同步。

### 两次修改

| 改动 | 结果 |
|---|---|
| 第一次：把角度计算搬到 CPU（`positions.cpu()`） | **更慢**（240 ms）——只是把同步从 H2D 换成了 D2H |
| 第二次：**缓存 RoPE 表**，按 (tokens, head_dim, base, device, dtype) | 223 → **212 ms/token** |

### 诚实的结果：5%，不是 43%

```
基线（逐专家循环）      253 ms/token   3.95 tok/s
向量化（bmm 批量）      223 ms/token   4.48 tok/s
+ 缓存 RoPE 表          212 ms/token   4.72 tok/s
```

**profile 里的 43% 没有转化成 43% 的墙钟收益。**
原因：那些同步与 GPU 工作**是重叠的**——CPU 在等同步时，GPU 队列里还有活可干。
所以"某操作占 X% 时间"不等于"消掉它能省 X% 墙钟"。

**这是今天第八次同类教训，而且这次是我自己的 profile 数字被误读了。**
正确表述应该是："同步占 CPU 时间 43%，但它与 GPU 工作重叠，因此墙钟收益取决于
CPU 是否真的是关键路径。"

### 剩下的最大一项

§22 的 profile 里 `aten::copy_` + `_to_copy` 合计 74 ms、`aten::index` 24.5 ms。
根因是**把 fp16 专家 gather 出来再转 fp32**。

正解是用 **打包量化权重 + 指针数组 kernel**（`fused_*_ptr_grouped`），
权重不复制、不转换。OLMoE 的 Q4_K/Q6_K 已被 §8 的 kernel 支持覆盖。

这一步还没做。

---

## 24. 为什么 OLMoE 这么慢：三个叠加的原因

| 栈 | 每 token 字节 | 耗时 | tok/s | 有效带宽 | 每层 op 数 |
|---|---:|---:|---:|---:|---:|
| DeepSeek-V2-Lite（ECS 优化后） | 1.35 GB | 47.8 ms | 20.92 | **28.2 GB/s（3.8%）** | **111** |
| OLMoE（本移植） | **1.61 GB** | 212 ms | 4.72 | **7.6 GB/s（1.0%）** | **574** |

### 原因 1：fp16 反量化把字节数放大 2.2 倍

移植为了先验证正确性，把专家**反量化成 fp16 稠密张量**常驻显存：

```
每个专家 (gate+up+down) = (1024x2048 + 1024x2048 + 2048x1024) x 2 字节 = 12.58 MB
x 8 个 top-k = 101 MB / 层
x 16 层 = 1.61 GB / token
```

而打包的 Q4_K 只有 **0.73 GB/token**——**反量化让每 token 多读 2.2 倍的字节**。

这直接否定了移植之初的预期："OLMoE 每 token 0.73 GB，应比 DeepSeek 的 1.35 GB 快约 2 倍"。
**那个 0.73 GB 是打包后的数字；换成 fp16 就不再成立。**

### 原因 2：每层 574 个 op，是 DeepSeek 的 5 倍

```
DeepSeek  2,994 / 27 层 = 111 op/层   （fused 指针数组 kernel，每层一次 MoE 调用）
OLMoE     9,176 / 16 层 = 574 op/层   （gather + .float() + bmm + 逐元素）
```

`expert_gate[expert_ids]`（gather）+ `.float()`（加宽）+ `bmm` + `sigmoid` + 乘法 + 求和，
在 8 个专家上展开后 op 数急剧膨胀。

### 原因 3：有效带宽只有 1.0%

两者都远低于 736 GB/s 的显存带宽：
DeepSeek 3.8%、OLMoE 1.0%。**瓶颈都不是带宽，而是编排与 kernel 效率。**

### 结论

**慢不是 OLMoE 架构的问题，是这条移植路径还很原始。**

它是一天内写出的"参考实现"，目标是把数值跑对（已验证），
完全没有应用 DeepSeek 运行时已有的三项优化：

1. **权重保持打包量化**（省 2.2 倍字节）
2. **指针数组 kernel**（gather 与 dtype 转换消失）
3. **pin-all 常驻**（DeepSeek 已有，OLMoE 尚未接入该机制）

把这三项接上，字节数回到 0.73 GB/token、op 数降到接近 DeepSeek 水平，
**届时才能公平检验"带宽模型是否跨架构成立"**——那是移植最初的目的，
目前还没有数据可以回答。

### 一句话

> **现在的 4.72 tok/s 不能用来评价 OLMoE，只能用来评价这条移植路径的当前成熟度。**

---

## 25. 去掉 fp32 加宽：212 → 104 ms/token（2.03 倍）

§24 指出 gather 之后 `.float()` 会把选中的专家加宽一倍。改动很小：
**bmm 直接用权重的 dtype（fp16）**，只在最后把结果转回输入 dtype。

fp16 的 `bmm` 在张量核心上内部仍以 fp32 累加，所以精度损失很小。

### 结果

| 版本 | ms/token | tok/s | 相对基线 |
|---|---:|---:|---:|
| 逐专家 Python 循环 | 253 | 3.95 | 1× |
| bmm 批量 | 223 | 4.48 | 1.13× |
| + 缓存 RoPE 表 | 212 | 4.72 | 1.19× |
| **+ 去掉 fp32 加宽** | **104** | **9.59** | **2.43×** |

**输出逐字相同**：`" Paris.

The currency of France is Euro.
"`
—— 且 `test_olmoe_layer.py` / `test_olmoe_model.py` 与 transformers 的
逐元素对照仍然通过（`atol=1e-4`）。

### 有效带宽

```
OLMoE  1.61 GB / 0.104 s = 15.5 GB/s   （占 736 GB/s 的 2.1%，原先 1.0%）
DeepSeek 1.35 GB / 0.0478 s = 28.2 GB/s（3.8%）
```

差距从 3.7 倍收窄到 **1.8 倍**。

### 为什么这一步收益这么大

§22 的 profile 里 `aten::copy_` + `_to_copy` 合计 74 ms、`aten::index` 24.5 ms。
`.float()` 既复制 8 个专家的权重又加宽，而且让 `bmm` 走 FP32 CUDA core
而不是张量核心。**一处改动同时消掉了"多复制一倍"和"放弃张量核心"两个问题。**

而 `aten::index`（gather）本身没有消除——它仍在，只是不再附带加宽。

### 下一步（真正的根因）

现在的 1.61 GB/token 仍然是**打包权重的 2.2 倍**。
要回到 0.73 GB/token 必须让权重**保持 Q4_K/Q6_K 打包**，
用 `fused_*_ptr_grouped` 直接吃每个专家的指针——gather 和 dtype 转换都不再需要。

那一步可以把字节数减半、op 数再降一个量级。

---

## 26. 打包权重路径已验证可行（下一步的最后一块拼图）

§25 之后仍有一个结构性浪费：移植把专家反量化成 fp16 稠密，每 token 读 1.61 GB，
而打包权重只有 0.73 GB。要消掉它必须**保持 Q4_K/Q6_K 打包**，
用 `fused_*_ptr_grouped` 直接吃每个专家的指针。

### 验证 1：字节数是自洽的

| 张量 | dims | 每专家行字节 | 推导总量 | 实际 nbytes |
|---|---|---:|---:|---:|
| `ffn_gate_exps` Q4_K | (2048, 1024, 64) | 1024 × 1152 | 75,497,472 | **75,497,472** |
| `ffn_down_exps` Q6_K | (1024, 2048, 64) | 2048 × 840 | 110,100,480 | **110,100,480** |

（Q4_K 每 256 元素 144 字节；Q6_K 每 256 元素 210 字节。）

### 验证 2：专家块在文件中是连续的

`read_raw` 返回整张量的 uint8 视图。取 `raw[0 : OUT*RB]` 当作专家 0 的砖块，
用 numpy 复刻 Q4_K 解量化，与 `expert_tensor(name, 0)` 比对：

```
slab0 row0 first256  maxdiff = 0.00000
mine[:4] [-0.0124774   0.01464331  0.02006745 -0.01790154]
ref [:4] [-0.0124774   0.01464331  0.02006745 -0.01790154]
```

**逐元素完全一致** → 专家是**最外层连续存放**的，可以零拷贝切片：

```python
raw = index.read_raw("blk.L.ffn_gate_exps.weight")   # mmap 视图，不复制
slab_e = raw[e * out * row_bytes : (e + 1) * out * row_bytes]   # 连续
```

### 接线计划（decode 情形最简单）

单 token decode 时 top-8 恰好是"8 个专家各 1 个 token"，
与 `fused_gate_up_gemm_ptr_grouped` 的入参形状天然吻合：

```
n_experts = 8, max_count = 1, counts = [1]*8
x_all     = hidden 重复 8 次 -> [8, 1, 2048]
pointers  = 8 个专家的打包显存指针
gate_qtype/up_qtype = "Q4_K",  gate_row_bytes = up_row_bytes = 1152
down:             qtype "Q6_K", down_row_bytes = 840, out_features = 2048, k_size = 1024
```

然后加权求和即得输出。**gather 与 dtype 转换同时消失，字节数减半。**

### 前置条件

| 条件 | 状态 |
|---|---|
| Q6_K kernel 支持 | ✅ §8 已完成并验证 |
| 专家块连续可零拷贝 | ✅ 本节验证 |
| 指针数组 kernel | ✅ 已存在（DeepSeek 在用） |
| 剩余 | 把 4.2 GB 打包砖块上传显存 + 接线 |

预计打包路径可将 OLMoE 从 104 ms/token 推近 DeepSeek 的水平
（1.61 → 0.73 GB/token，且 op 数大幅下降）。

---

## 27. 打包权重指针路径：数值一致，MoE 快 1.77 倍

`deepinfer/moe/olmoe_packed.py` 已实现并实测（真实 checkpoint，第 0 层）：

```
packed load 0.05s, gate bytes 75497472     ← 打包砖块上传显存
packed vs fp16: maxdiff = 0.00000           ← 与 fp16 参考逐元素一致
MoE per layer: packed 0.31 ms   fp16 0.54 ms   speedup 1.77x
x16 layers:    packed 5 ms/token  fp16 9 ms/token
```

**数值完全一致**——说明 Q4_K/Q6_K 直接在打包字节上做 GEMV 与先反量化成 fp16
再算，结果相同（差 0 而非"在容差内"）。

### 实现要点

| 步骤 | 做法 |
|---|---|
| 取打包字节 | `read_raw(name)` 返回 mmap 视图，按 `raw[e*out*rb : (e+1)*out*rb]` 零拷贝切 |
| 上传 | 一次性 `torch.from_numpy(...).to("cuda")`，0.05 s |
| 指针 | `storage.data_ptr() + e*stride`，组成 int64 数组 |
| decode 形状 | 单 token 时 top-8 = 8 专家各 1 token → `n_experts=8, max_count=1, counts=[1]*8` |
| gate/up | `Q4_K`, `row_bytes = 1152` |
| down | `Q6_K`, `row_bytes = 840` |

`fused_gate_up_gemm_ptr_grouped` 返回的**已经是 SwiGLU 后的激活**
（直接喂给 `fused_down_gemm_ptr_grouped`），所以加权求和由本模块自己做，
不依赖 DeepSeek 那边的 scatter 约定。

### 收益与预期

每层 MoE 从 0.54 → 0.31 ms。按 16 层算，MoE 部分 9 → 5 ms/token。
但端到端目前是 104 ms/token，**MoE 只占一小部分**——
说明剩下的时间在 attention、norm 和逐层 Python 编排上。

### 尚未接线

`olmoe_layer.py` 仍走 fp16 路径。要吃到这 1.77 倍，需要：

1. 启动时为 16 层加载全部打包砖块（16 × 261 MB ≈ 4.2 GB 显存，32 GB 卡够）；
2. `decoder_layer_forward` 增加 packed 分支；
3. 端到端复测。

这一步没有做——单层已验证，全模型接线还需要一轮完整时间。

---

## 28. 打包路径接进全模型：端到端没有变化，且我 profile 错了阶段

### 结果

```
prompt "The capital of France is"
text   " Paris.

The currency of France is Euro.
"      ← 输出不变
decode 103 ms/token (9.71 tok/s)                          ← 之前 104 ms
peak VRAM 18.0 GiB                                        ← 同时加载 fp16 与打包两套
```

**单层测得 packed 比 fp16 快 1.77 倍，但端到端没变。**

### 原因：profile 的是 prefill，优化的是 decode

§27 的 profile 用的是 `forward(ids, ...)`，其中 `ids` 是**完整 prompt（多 token）**
——那是 **prefill**，不是单 token 的 decode 步骤。

所以 `aten::index` 那 24.4 ms **主要来自 prefill 的 gather**，
而 packed 路径只作用于 `flat.shape[0] == 1` 的 decode 步骤。

`101 ms`（prefill）与 `104 ms/token`（decode）数值接近纯属巧合，
我把它们当成同一个量了。

### 这是今天第九次同类错误

| # | 错误 | 性质 |
|---|---|---|
| 1 | 专家缓存无效 | 实验条件不成立 |
| 2 | 驻留无提升 | 单次未预热 |
| 3 | 94% 主机开销 | 分母漏项 |
| 4 | 分配是瓶颈 | 噪声内 |
| 5 | shared expert 重复上传 | 未核对日志 |
| 6 | 每 token 743 MB | 归一化对象不随分母变 |
| 7 | 33% attention | profiler 条目互相包含 |
| 8 | 同步占 43% | 与 GPU 工作重叠 |
| 9 | **packed 能提升端到端** | **profile 了错误的阶段（prefill vs decode）** |

**共同点始终相同：把"看起来是那个量"当成"确实是那个量"，而没有用一次
能区分的实验去验证。**

### 打包路径仍然是有价值的

- 单层数值完全一致（maxdiff 0.00000）；
- MoE 部分快 1.77 倍；
- 它读 0.73 GB/token 而不是 1.61 GB——**真正的收益要在 prefill 上也用打包路径时才体现**，
  因为 prefill 才是 gather 密集的阶段。

### 下一步

1. **给 prefill 也接打包路径**（需要支持多 token 的 `max_count > 1` 情形）；
2. 或者：**重新 profile decode 步骤**（`max_new_tokens=1` 循环），
   拿到真正的 decode 分解再决定优化什么。

第 2 步更便宜，且是今天反复证明有效的顺序——**先测对阶段，再动手。**

---

## 29. 真正的 decode profile（这一份才是对的）

前几次 profile 都拿完整 prompt 跑，那是 prefill。这次**只 profile 单 token 的
decode 步骤**（prefill 一次、预热三次、再 profile 十次）：

```
DECODE wall = 92.5 ms   ops = 7,752
```

| CUDA 项 | 每步 | 次数 |
|---|---:|---:|
| **`aten::index`** | **29.08 ms** | **48** |
| `aten::bmm` | 14.78 ms | 48 |
| cutlass wmma tensorop | 11.26 ms | 97 |
| cutlass tensorop | 4.73 ms | 16 |
| `aten::matmul` / `mm` | 1.34 ms | 81 |

| CPU 项 | 每步 | 次数 |
|---|---:|---:|
| `cudaDeviceSynchronize` | 14.66 ms | — |
| `cudaLaunchKernel` | 8.71 ms | 1,229 |

### 关键：packed 分支确实生效了

已确认远端 `olmoe_layer.py` 有 `use_packed` 与 `moe_forward_packed`（第 15/120/121 行）。
所以那 48 次 `aten::index` **不再是权重 gather**，
而是 `gate_ptrs[ids]` / `up_ptrs[ids]` / `down_ptrs[ids]` 的**8 元素指针 gather**
——每层 3 次 × 16 层 = 48 次。

**对 8 个 int64 元素做索引却报 0.6 ms/次，这个数字不合理。**
合理的解释是：profiler 把**等待前序工作的队列排空时间**归到了这些 op 头上。
也就是说这 29 ms 是"GPU 在等"，不是"这个 op 在算"。

要区分这两种情况，需要给每个 op 单独打 CUDA event 计时，
而不是依赖 `key_averages()` 的累计时长。

### 当前 decode 成本结构（可信部分）

```
cutlass tensor 类 kernel（注意力 + bmm 的实际计算）  约 16 ms
bmm 调度                                           14.8 ms
kernel launch（CPU）                                8.7 ms
总墙钟                                             92.5 ms
```

**剩余约 50 ms 仍无法归因**，需要更细的 event 级计时。

### 诚实结论

- OLMoE 从 3.95 提升到 9.71 tok/s（2.46 倍），全部来自**已完成的真实改动**
  （向量化、缓存 RoPE、去掉 fp32 加宽）；
- 打包指针路径在**单层**快 1.77 倍且数值完全一致，但**端到端未体现**；
- 剩下的瓶颈**还没有被可信地定位**——目前手上这份 profile 有归因歧义。

**下一轮第一件事：用 CUDA event 给每个阶段单独计时，代替 profiler 累加。**
在那之前不应再改代码——今天已经九次因为归因不可靠而走错方向。

---

## 30. CUDA event 计时：profiler 把 decode 高估了 4.7 倍

按 §29 的建议，改成**每个阶段一对 CUDA event**，
先 `synchronize()` 再取 `elapsed_time`（顺序反了会报 "device not ready"）。

### 结果（16 层 × 单 token decode，打包权重）

```
decode wall = 19.7 ms/token          ← 50.8 tok/s
  moe              8.78 ms   44.5%
  qkv_qknorm_rope  6.97 ms   35.3%
  attn_norm        1.76 ms    8.9%
  o_proj           0.91 ms    4.6%
  attention        0.67 ms    3.4%
  (sum of stages) 19.09 ms
```

**19.7 ms/token，而 profiler 报的是 92.5 ms/token。**

### 两个结论

**1. profiler 的数字不能用来看绝对值。**
它给 7,752 个 op 加采样开销，把 19.7 ms 放大到 92.5 ms（4.7 倍）。
**比例可以参考，绝对值完全不可信。**

这解释了 §29 里"8 元素指针 gather 要 0.6 ms"的荒谬数字——那不是 op 慢，
是 profiler 把采样与队列排空的时间摊到了每个 op 上。

**2. 同时驻留两套权重代价极大。**
本节的 19.7 ms 是在**只加载打包权重（4.2 GB）**时测的；
端到端生成脚本同时加载 fp16（13 GB）+ 打包（4.2 GB）= 18 GB 时，
decode 是 103 ms/token。**差 5.2 倍。**

所以 §28 "packed 接到全模型没有提升"的真实原因是
**两套权重同时驻留把显存挤满了**，而不是 packed 路径无效。

### 现在的真实成本结构

| 阶段 | ms/token | 占比 |
|---|---:|---:|
| MoE（打包指针 kernel） | 8.78 | 44.5% |
| QKV + QK-norm + RoPE | 6.97 | 35.3% |
| attn_norm | 1.76 | 8.9% |
| o_proj | 0.91 | 4.6% |
| attention | 0.67 | 3.4% |

**MoE 现在是最大项（44.5%）**，而 QKV 那 35.3% 也值得注意——
三个 2048×2048 的投影在 fp16 下每 token 要读 25 MB，
按 736 GB/s 应该 0.03 ms，实际 6.97 ms，**离带宽上限 200 倍**。

### 遗留问题

把生成脚本改成"只加载打包权重"后报 `KeyError: 'expert_gate'`——
说明该路径下 `use_packed` 判成了 False，**需要一次调试**。
19.7 ms 这个数字来自直接调用 `decoder_layer_forward` 的计时循环，
不经过那条有问题的分支判断，所以数字本身可信。

---

## 31. KeyError 的根因：**这个移植没有 KV cache**

`KeyError: 'expert_gate'` 不是分支判断的 bug。生成循环里：

```python
ids = torch.cat([ids, nxt[None]], dim=1)      # 上下文每步 +1
```

第一次调用时 `ids` 只有 1 个 token（走 packed），**第二次起 `flat.shape[0] == 2`**，
于是落到 fp16 分支——而脚本为了省显存已经不再加载 fp16 专家，因此 KeyError。

### 真正的问题：每步都在重算整个前缀

`decoder_layer_forward` 每次都把**全部 token** 送进注意力和 MoE，
没有 KV cache。所以第 N 步要重算 N 个位置：

| 生成到第几步 | 送入的 token 数 | 相对单步成本 |
|---:|---:|---:|
| 1 | 6（prompt） | 6× |
| 6 | 11 | 11× |
| 12 | 17 | 17× |

**这就解释了 103 ms/token 与 19.7 ms/token 的差距**——不是显存问题（§30 的推断），
而是**平均了逐渐变长的前缀**。

§30 里"两套权重挤满显存导致 5 倍变慢"的结论**作废**。
真实原因是缺少 KV cache。

### 所以 OLMoE 的真实速度是多少

| 指标 | 数值 | 含义 |
|---|---:|---|
| **19.7 ms/token（50.8 tok/s）** | **稳态 decode 成本** | 单个新 token 走完 16 层，无前缀重算 |
| 103 ms/token | 前缀重算的平均值 | **不是稳态性能**，随生成长度恶化 |

**如果加上 KV cache，生成速度应接近 50.8 tok/s。**

### 这是移植缺的最后一个大件

| 组件 | 状态 |
|---|---|
| Q6_K / 专家读取 / Router / QK-norm / MHA / MoE 块 | ✅ |
| 整层、整模型与 transformers 一致 | ✅ |
| 真实权重端到端生成 | ✅ |
| 打包指针路径 | ✅ 单层快 1.77×、数值一致 |
| **KV cache** | ❌ **缺失，导致每步重算前缀** |

有了 KV cache 才能：
1. 拿到真实的生成速度（预计约 50 tok/s）；
2. 让 packed 路径在多 token 情形下也生效（每步只有 1 个新 token）；
3. 与 DeepSeek 的 20.92 tok/s 做公平对比。

### 今天第十一次归因修正

`103 ms` 我先归因为"两套权重挤显存"，实际是**没有 KV cache 导致重算前缀**。

又一次是同一个模式：**看到一个数字，先给了一个合理的解释，而没有做能区分两种解释的实验**
（这里只需要把生成长度改成 6 / 12 / 24 各测一次，就能看出成本随前缀线性增长）。

---

## 32. 移植完成：OLMoE 47.31 tok/s，比 DeepSeek 快 2.26 倍

加上 KV cache 之后（§31 指出这是最后一处缺失），端到端实测：

```
after prefill, VRAM 4.5 GiB
text:   " Paris.

The currency of France is Euro.

"     ← 输出正确
decode  21.1 ms/token  (47.31 tok/s)
```

### 全部结果

| 配置 | ms/token | tok/s | 相对 |
|---|---:|---:|---:|
| 逐专家 Python 循环（起点） | 253.0 | 3.95 | 1× |
| bmm 批量 | 223.0 | 4.48 | 1.13× |
| + 缓存 RoPE 表 | 212.0 | 4.72 | 1.19× |
| + 去掉 fp32 加宽 | 104.0 | 9.71 | 2.46× |
| + 打包指针路径 + KV cache | **21.1** | **47.31** | **12.0×** |

### 移植最初的问题现在有答案了

> **带宽模型跨架构成立吗？**

| 模型 | 每 token 字节 | 实测 | 反推有效带宽 |
|---|---:|---:|---:|
| DeepSeek-V2-Lite | 1.35 GB | 20.92 tok/s | 28.2 GB/s |
| **OLMoE-1B-7B** | **0.73 GB** | **47.31 tok/s** | **34.5 GB/s** |

- 速度比 **2.26×**，带宽模型预测 **1.85×**；
- 两者反推的有效带宽落在同一区间（28.2 vs 34.5 GB/s）；
- **两个架构完全不同**（MLA vs MHA、共享专家 vs 无、top-6 vs top-8、
  Q8_0 down vs 混合 Q4_K/Q6_K），但都由"每 token 触达字节数"这一项决定。

**结论：带宽模型跨架构成立。** 这是移植最初想验证的事情，现在有了数据。

### 关键改动清单

| 改动 | 收益 | 验证方式 |
|---|---|---|
| bmm 批量替代逐专家循环 | 1.13× | 与循环版数值一致 |
| 缓存 RoPE 表（每层一次同步 → 一次） | 1.06× | 与 transformers 一致 |
| **bmm 保持 fp16（去掉 fp32 加宽）** | **2.06×** | 逐字相同输出 |
| **打包 Q4_K/Q6_K + 指针 kernel** | 字节减半 | maxdiff 0.00000 |
| **KV cache** | 消除前缀重算 | 缓存/非缓存 logits 一致 |

### 今天这条线上最重要的两条经验

**1. 收益最大的改动（fp16 不加宽，2.06×）来自顺着数据类型这条线查，
不是来自 profile 的百分比。** profile 说过 gather 占 24%、同步占 43%，
但那两项加起来也没这一条大。

**2. 十一次归因错误里，有十次是同一个模式**——看到一个数字，
给了一个"合理"的解释，然后没有做能区分两种解释的实验就动手。
唯一每次都有效的做法是：**设计一个能区分两种可能的实验**（GQA 对照、
改生成长度、CUDA event 分阶段计时、缓存 vs 非缓存 logits 对比）。

---

## 33. 控制变量实验：带宽模型不成立，真正驱动的是"层数 × 每层固定开销"

要证明加速不局限 DeepSeek，最干净的办法不是再加一个模型，
而是**在同一个模型上直接操纵每 token 字节数**：改 `top_k`。

### 结果

| top_k | ms/token | tok/s | GB/token | 有效带宽 |
|---:|---:|---:|---:|---:|
| 8（原始） | 28.2 | 35.51 | 0.453 | 16.1 GB/s |
| 6 | 28.1 | 35.62 | 0.340 | 12.1 GB/s |
| 4 | 27.6 | 36.22 | 0.226 | 8.2 GB/s |
| 2 | 27.9 | 35.85 | 0.113 | 4.1 GB/s |
| **1** | **27.4** | **36.48** | **0.057** | **2.1 GB/s** |

**每 token 字节数变化 8 倍，吞吐几乎不变（35.5–36.5 tok/s）。**

### 这意味着什么

如果带宽模型成立，top_k=1 应该比 top_k=8 快 8 倍。**实际只快了 2.7%。**

而且 top_k=1 时每 token 只读 0.057 GB，在 27.4 ms 内完成——
有效带宽 2.1 GB/s，**只有这块卡 736 GB/s 的 0.3%**。

**OLMoE 的解码完全不是带宽受限的。**

### 那 DeepSeek 与 OLMoE 的差异来自哪里

| 模型 | 层数 | ms/token | **每层成本** | tok/s |
|---|---:|---:|---:|---:|
| DeepSeek-V2-Lite | 27 | 47.8 | **1.77 ms** | 20.92 |
| OLMoE-1B-7B | 16 | 28.0 | **1.75 ms** | 35.8 |

- **每层成本几乎相同**（1.77 vs 1.75 ms）
- 层数比 27/16 = **1.69**
- 实测速度比 35.8/20.92 = **1.71**

**两者的差异完全由层数解释，与"每 token 字节数"无关。**

### §32 的结论作废

§32 说"OLMoE 47.31 tok/s 对应 34.5 GB/s，与 DeepSeek 的 28.2 GB/s 同区间，
所以带宽模型跨架构成立"。**那个推理是错的**——两个"有效带宽"数字接近纯属巧合，
因为两者的每层成本恰好接近，而层数不同。

真正成立的规律是：

> **在当前实现里，解码成本 ≈ 层数 × 每层固定开销（约 1.75 ms）。
> 带宽根本不是约束——top_k 从 8 降到 1，吞吐只提高 2.7%。**

### 这条教训比结果本身重要

前一节我用"两个模型的每 token 字节数与吞吐反比"来支持带宽模型，
但那**只是两个点，且两个模型的层数也恰好不同**——两个变量同时变化，
无法区分是哪一个在起作用。

**这个 top_k 扫描第一次只让一个变量变化**，结论立刻反转。

**第十二次归因错误，也是今天最严重的一次**，因为它差点写进论文当成核心论断。

---

## 34. 本机复测：§33 的现象成立，解释要改——decode 取"主机派发"与"GPU 计算"的较大者

§33 写进论文之前，按它自己的标准复查了一遍：**能不能只让一个变量变化？**
复查发现三个问题，并在本机（RTX 3050 Laptop 4 GB，Ryzen 5 5600H，Windows）
用同一模型做了两组单变量实验。

### 先修的三个问题

| 问题 | 影响 |
|---|---|
| **§33 的字节账漏了稠密权重。** 移植把 attention（q/k/v/o）和 `lm_head` 反量化成 fp16 常驻（这正是 §32"显存稳定在 4.5 GiB"的来源），每 token 都要读，共 **0.752 GB** | top_k 8→1 时每 token 字节数实际只降 **1.52 倍**，不是 8 倍 |
| **KV cache 那次提交（661e807）把每层一次的同步加了回来。** `build_rope` 对 CUDA 上的 `positions` 做 `int(positions.flatten()[0])`；缓存键又含起始位置，每个新位置在首层未命中，再加两次可分页 H2D 拷贝 | §32、§33 的数字都是在**每 token 16 + 2 次同步**下测得的 |
| **§30–§33 的脚本只在远端** | 仓库里无法复现这几节的任何数字 |

修正后的 §33 表（ms/token 沿用 §33 的 4080S 实测；字节数按本地 GGUF 计算——
专家 down 投影逐层为 `6644644644644666`（6 = Q6_K）；远端 bartowski 版本的
专家量化组合可能差几个百分点，稠密部分都是 fp16，相同）：

| top_k | ms/token | §33 写的 GB/token | **实际 GB/token** | **实际有效带宽** |
|---:|---:|---:|---:|---:|
| 8 | 28.2 | 0.453 | **1.239** | **44.0 GB/s（6.0%）** |
| 6 | 28.1 | 0.340 | 1.118 | 39.8 GB/s（5.4%） |
| 4 | 27.6 | 0.226 | 0.996 | 36.1 GB/s（4.9%） |
| 2 | 27.9 | 0.113 | 0.874 | 31.3 GB/s（4.3%） |
| 1 | 27.4 | 0.057 | **0.813** | **29.7 GB/s（4.0%）** |

"不受带宽限制"仍然成立（带宽模型预测 top_k=1 快 1.52 倍，实测 1.03 倍），
但"字节数变化 8 倍"应为 **1.52 倍**，"只用到带宽的 0.3%"应为 **4–6%**。

### 修复与工具

- `deepinfer/moe/olmoe_layer.py`：每个 (head_dim, base, device, dtype) 只建**一张整表**，
  一次搬上 GPU，按位置**切片**（视图：无 kernel、无拷贝、无同步）。
  `positions` 必须是 CPU 张量，传 CUDA 张量直接报错——这次回归正是因为它被静默接受。
- `tests/moe/test_olmoe_rope.py`（7 项）：任意偏移（含越过初始 256 行触发扩表）与独立公式一致；
  GPU 上带 cache 的 decode 步在 `set_sync_debug_mode("error")` 下运行，**任何同步都会抛错**。
  在旧代码上这两项 GPU 测试失败，报错位置正是 `olmoe_layer.py:40`。
- `scripts/bench_olmoe_decode.py`：真实 GGUF、打包专家、KV cache、贪心，仓库内可复现。
  每个 decode 步计三种时间：
  - **step**：墙钟，同步到同步；
  - **host**：主机把这一步排进队列的时间（`forward()` 返回）；
  - **gpu**：纯设备时间——先发一个自旋 kernel 占住流，主机排队期间 GPU 不会等主机。
    只在步内无同步、且自旋确实长于排队时有效，脚本会检查。

  扫描参数：`--layers`（截断到前 N 层）、`--top-k`（**只改 GPU 工作量**）、
  `--host-delay-ms`（每层主机自旋，**只改主机工作量**）、`--sync-debug`（列出同步位置）。
  prefill 逐 token 走 decode 路径（打包 kernel 只支持单 token），不计时。

全量测试 216 项通过（新增 7 项）。

### 结果 1：同一模型内，成本随层数严格线性

3050 显存最多放 16 层中的 8 层（`docs/results/olmoe_decode_sm86_{before,after}_rope_fix.json`，
L = 1/2/4/6/8 × top_k = 1/2/4/8，5 次中位数，32 token）。修复后最小二乘拟合（R² ≥ 0.996）：

| 量 | 拟合（ms/token） |
|---|---|
| GPU 时间 | **1.27 + (0.68 + 0.233·top_k)·L** |
| 主机时间 | **0.4–0.6 + (1.79–1.96)·L**，与 top_k 无关 |

GPU 每层 0.68 ms 与 top_k 无关（稠密 matmul 与小算子），每个专家每层 0.233 ms；
截距 1.27 ms 主要是 `lm_head`（206 MB fp16）。

### 结果 2：step ≈ max(host, gpu)

L=8，修复后：

| top_k | step | host | gpu | 区间 |
|---:|---:|---:|---:|---|
| 8 | 23.90 | 15.01 | **21.68** | GPU 受限 |
| 4 | 16.92 | 15.06 | 14.27 | 两者相当 |
| 2 | 17.13 | **15.87** | 10.47 | 主机受限 |
| 1 | 17.07 | **15.98** | 8.44 | 主机受限 |

两个单变量实验从两侧验证：

**只改 GPU 工作量（top_k）。** 主机受限时 top_k 2→1，GPU 少了 2.0 ms，step 不变（17.13 / 17.07）；
GPU 受限时 top_k 8→4，step 少 7.0 ms。

**只改主机工作量（每层自旋，GPU 工作量不变；`olmoe_decode_sm86_host_delay.json`）。**

| 每层增加 | 合计增加 | top_k=8 的 Δstep | top_k=1 的 Δstep |
|---:|---:|---:|---:|
| 0.25 ms | 2 ms | +0.12 | +2.08 |
| 0.50 ms | 4 ms | +0.33 | +4.10 |
| 0.75 ms | 6 ms | +1.13 | +6.49 |
| 1.00 ms | 8 ms | +3.47 | +9.32 |
| 1.50 ms | 12 ms | +8.07 | +12.79 |

主机受限的 top_k=1 **约 1:1 付出**（1.03–1.17 倍）；GPU 受限的 top_k=8 **吸收**了前 4–6 ms，
直到主机时间追上 GPU 时间才开始付出。预测拐点 (21.81 − 15.87) / 8 = 0.74 ms/层，
实测 Δstep 在 0.75 处开始上升、1.0 处明显。另用探针脚本跑了两次，趋势一致，单点波动约 ±1–2 ms。

### 修复本身的收益

| L=8 | 修复前 | 修复后 | |
|---|---:|---:|---:|
| top_k=8 | 29.67 | 23.90 | **1.24×** |
| top_k=4 | 21.06 | 16.92 | 1.24× |
| top_k=2 | 17.29 | 17.13 | 1.01× |
| top_k=1 | 17.50 | 17.07 | 1.03× |

每层一次同步让主机与 GPU 在每层串行一次。两者相当时损失最大；
一方远大于另一方时，被串行的那部分本来就小。
**所以在主机受限的机器上，这个修复预计只有几个百分点**——4080S 上需要实测。

### 对 §33 的修正

**现象成立，解释要改。**

1. §33 的 top_k 扫描本身已经证明：在 4080S 上 **GPU 工作量不在关键路径上**。
   剩下的只有主机侧（派发 + 同步）。本机主机派发 1.8–1.96 ms/层，与 4080S 的 1.75 ms/层量级一致——
   但这是跨机器的类比（CPU、操作系统、驱动都不同；Windows WDDM 的 launch 通常比 Linux 慢），
   **需要在 4080S 上用 host 列直接确认**，在那之前它是预测，不是结论。
2. 同一份代码在 3050 上 top_k 8→1 **快 1.40 倍**（23.90 → 17.07 ms），
   因为这里 GPU 每层时间（0.68 + 0.233·k ms）在 top_k=8 时超过主机时间。
   **"与字节数无关"不是这份代码的性质，而是"这份代码 × 这台机器"的性质。**
3. "成本 = 层数 × 每层固定开销"把一个**取最大值**的关系当成了**常数**。
   每层成本 = max(主机派发, GPU 计算) + 边界项，哪一边更大取决于硬件组合。
4. "层数解释了 DeepSeek 与 OLMoE 的差异"：本节在**同一模型内**证实了成本随层数线性，
   但跨模型"每层成本几乎相同（1.77 vs 1.75 ms）"仍然只有两个点——
   两者的每层 op 数、注意力结构都不同，相近可能是巧合。

### 主机时间去向（cProfile，L=8，top_k=1；cProfile 放大小函数，只看比例）

| 部分 | 每步 | 占比 | 构成 |
|---|---:|---:|---|
| `rms_norm` | 6.67 ms | 36% | 每层 4 次，每次约 10 个 aten op（两次 `.float()`、pow、mean、add、rsqrt、两次 mul、`.to`） |
| `moe_forward_packed` | 4.89 ms | 26% | router、3 次指针 gather、`repeat` / `ones`、两次 CuPy 调用（每次 `_validate` + 5 次 `_to_cupy`） |
| `apply_rope` | 2.76 ms | 15% | 切片、乘、减、加、`cat` |
| 层内其余（`@`、view、残差） | 2.05 ms | 11% | |

每层约 118 个顶层 aten 调用。没有单一大项——是大量小算子的派发成本。

复现：`python scripts/bench_olmoe_decode.py --layers 8 --top-k 1 --profile-host`
（再跑一次：rms_norm 37%、MoE 26%、RoPE 15%）。

### 下一步（按机器所处区间区分）

| 区间 | 有效的方向 | 无效的方向 |
|---|---|---|
| 主机受限（4080S 预计如此；本机 top_k ≤ 2） | 减少每层 op 数：融合 RMSNorm（36%）、RoPE；或整步 CUDA Graph | 改专家 kernel、减少字节 |
| GPU 受限（本机 top_k = 8） | 专家 kernel 效率：每专家每层 0.233 ms，约 16 GB/s，3050 峰值的 8% | 减少主机开销 |

在 4080S 上先跑这一条，用来检验本节的预测：

```bash
python scripts/bench_olmoe_decode.py --layers 4 8 12 16 --top-k 1 2 4 8 \
    --sync-debug --output docs/results/olmoe_decode_sm89.json
```

预测：host ≈ 1.75 ms/层且与 top_k 无关；gpu 每层明显小于 host；
step 的斜率等于 host 的斜率；RoPE 修复带来的提升在个位百分数。
**任何一条不成立，本节的解释就要再改。**

### 仍未解决

§32 端到端的 21.1 ms/token 与 §33 top_k=8 的 28.2 ms/token 是同一配置，差 34%。
两个脚本都不在仓库里，**本节无法解释，也不给猜测**。
上面那条命令的 L=16、top_k=8 一行可以作为仲裁。

### 教训

§33 指出 §32 犯了"两个变量同时变化"的错，自己却在同一类错误上：
"字节数变 8 倍"只数了专家字节，而它要推翻的带宽模型讲的是**总**字节。

另外，只改 GPU 工作量的实验（top_k）只能说明"GPU 不是瓶颈"；
要说明"主机是瓶颈"，需要另一个只改主机工作量的实验——本节的自旋注入。
**两个方向的单变量实验都做了，max 的两条边才都算确认。**
