# 这些优化是不是 DeepSeek 特有？

> 结论：**优化所在的层是模型无关的，但端到端只在 DeepSeek-V2 架构上验证过。**
> 本文区分这两件事，并给出可复现的证据。

---

## 1. 优化改在了哪一层

本轮全部有效改动（见 `CUDA_GRAPH_STUDY.md` §13–§15）：

| 改动 | 所在位置 | 是否模型相关 |
|---|---|---|
| `preload_gpu_experts()` | `GGUFExpertStore` | **否** |
| 自动探测显存并预热 | `DeepSeekV2LiteGGUFRuntime.__init__` | 逻辑通用，代码在 DeepSeek 运行时里 |
| prefill 期间保持缓存启用 | `generate()` | 逻辑通用，代码在 DeepSeek 运行时里 |
| 加载后预热 | `generate()` | 同上 |

**核心机制（专家常驻显存、指针路径、跳过 staging）全部在 `GGUFExpertStore` 里**，
而它只依赖 GGUF 的 `blk.N.ffn_{gate,up,down}_exps.weight` 命名约定，与模型架构无关。

---

## 2. 证据：同一个 store 零改动读 OLMoE

```python
from deepinfer.moe.deepseek_v2_runtime import GGUFExpertStore
from deepinfer.moe.gguf_index import GGUFIndex

idx = GGUFIndex("OLMoE-1B-7B-0924.Q4_K_M.gguf")
store = GGUFExpertStore(idx, layer_indices=list(range(16)), device=torch.device("cpu"))
```

输出（与 DeepSeek 并列）：

```
OLMoE-1B-7B  layers=16 experts=64 in=2048 out=1024 qtype=Q4_K  expert_total=3.63 GiB
DeepSeek-Lite layers= 3 experts=64 in=2048 out=1408 qtype=Q4_K  expert_total=1.06 GiB
```

实际读取同样正常：

```
OLMoE layer0 experts [0,5,63] packed read OK
  gate packed bytes=3538944 (expect 3538944)
  up   packed bytes=3538944 (expect 3538944)
  down packed bytes=5160960 (expect 5160960)
```

**字节数精确匹配，无任何代码修改。**

### 这证明了什么

- `load_packed_experts`（优化的核心路径）对 OLMoE 完全可用；
- `preload_gpu_experts` 只调用 `load_packed_experts`，
  因此对任何使用 `ffn_*_exps` 三阶张量的 MoE GGUF 都成立；
- 自动探测所需的 `bytes_per_expert × expert_count` 也是从这个层算出来的。

### 这没有证明什么

**端到端吞吐。** OLMoE 走不通 `DeepSeekV2LiteGGUFRuntime.generate()`，因为：

| 组件 | DeepSeek-V2 | OLMoE |
|---|---|---|
| attention | **MLA**（`attn_kv_a_mqa` / `attn_kv_b`） | **GQA + QK-norm** |
| 共享专家 | 有（`ffn_*_shexp`） | 无 |
| `ffn_down_exps` | Q8_0 | **Q6_K**（fused kernel 仅支持 Q4_K/Q5_0/Q8_0） |

移植估计 4.5–5 天（见 `OLMOE_PORT_PLAN.md`）。

---

## 3. 在现有架构内换模型

若要在**不改运行时**的前提下证明"不是某一个 checkpoint 特有"，
可换用同架构的另一模型：

| 模型 | 总参数 | 显存需求（Q4） | 32 GB 能否驻留 |
|---|---:|---:|:--:|
| DeepSeek-V2-Lite（当前） | 15.7B | 9.65 GB | ✅ |
| **DeepSeek-Coder-V2-Lite** | 16B | 约 9.8 GB | ✅ |
| DeepSeek-V2（236B） | 236B | 约 130 GB | ❌ |

DeepSeek-Coder-V2-Lite 只需下载约 9.8 GB，**无需改一行代码**即可复测，
是"同架构、不同 checkpoint"的最省力验证。

---

## 4. 结论的准确表述

可以说的：

> 专家常驻、指针路径与跳过 staging 的机制实现于 `GGUFExpertStore`，
> 与模型架构无关；已在 OLMoE-1B-7B 的 GGUF 上零改动验证了专家读取与字节布局。

不能说的：

> ~~这些优化在任意 MoE 模型上都能带来 8.5 倍。~~

后者需要 OLMoE 移植完成后实测，或至少在同架构的另一个 checkpoint 上复现。

---

## 5. 实测：DeepSeek-Coder-V2-Lite，零代码改动

在同一台 RTX 4080 SUPER（32 GB）上，用**完全相同**的代码与命令，
只把模型与 meta 目录换成 DeepSeek-Coder-V2-Lite-Instruct（Q4_K_M，10.36 GB）：

```bash
python scripts/bench_decode_repeats.py   --model .../DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf   --meta-dir .../meta_coder --repeats 5 --max-new-tokens 12 --warmup 2
```

| 模型 | 逐次 ms/token | 中位数 | 稳态 |
|---|---|---:|---:|
| DeepSeek-V2-Lite-Chat | 47.8 / 47.6 / 47.8 / 47.8 / 47.8 | 47.8 | **47.7** |
| **DeepSeek-Coder-V2-Lite** | 90.2 / 89.8 / 76.1 / 47.1 / 46.8 | 76.1 | **47.0** |

**稳态几乎相同（47.7 vs 47.0 ms/token，即 20.9 vs 21.3 tok/s）。**

### 中位数差异是预热，不是吞吐

Coder 的前三次仍在 90 / 90 / 76 ms，第 4 次才落到 47.1。
原因是它刚下载完，**页缓存是冷的**——首次读取 10.36 GB 走了真实磁盘 I/O。
两个模型的**热态吞吐一致**，说明自动 pin-all 对二者生效方式相同。

> 这也再次说明基准里 `--warmup` 次数应当按模型大小调整；
> 对刚下载的模型，2 次预热不足。

### 这证明了什么

- 自动 pin-all 的探测（`required ≤ free × 0.7`）在 10.36 GB 的模型上同样触发；
- prefill 走常驻缓存的改动对两个 checkpoint 都生效；
- 预热机制对二者都有效（都收敛到约 47 ms/token）；
- **一行代码未改。**

结合 §2 的 OLMoE 存储层验证，可以确认：

> 这套优化不依赖 DeepSeek-V2-Lite 这一个 checkpoint 的权重细节。

仍需注意：Coder-V2-Lite 与 V2-Lite **同架构**（MLA），
所以这是"同架构不同 checkpoint"的验证，**不是跨架构验证**。
跨架构要等 OLMoE 移植完成（见 `OLMOE_PORT_PLAN.md`，剩余 2–2.5 天）。
