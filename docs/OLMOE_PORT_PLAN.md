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
