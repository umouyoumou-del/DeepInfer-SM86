# DeepInfer-SM86

在 **RTX 3050 Laptop 4 GB（SM86 / Windows）** 上为 DeepInfer 增加面向
Qwen2.5 / Llama / Mistral / Gemma 等 dense decoder 的 model_executor 原生
推理后端，并在其上提供 paged KV、连续批处理和指标治理等 engine-core 原型。

这是一个**非官方扩展仓库**，不是 DeepInfer 官方发行版，也不是完整 DeepInfer
engine。当前重点是：

- SM86 可运行的 W4A16、W3A16 CUDA GEMV 与 tiled W4/W3 prefill GEMM；
- 原生 GQA decode kernel（static + paged FP16/INT8）与 dense static-cache CUDA Graph；
- paged / elastic KV cache、prefix cache、连续批处理调度器；
- chunked prefill 与 batched decode 的双 CUDA stream overlap；
- FP16 / INT8 / FP8 paged KV 与 CPU cold-block swap；
- Qwen2.5-3B / 7B 的量化、加载和测速；
- Qwen2 / Llama / Mistral / Gemma / Gemma2 dense 架构适配；
- 4 GB 显存下的混合精度（Q4/Q3）与整体 Q4 实验数据；
- DeepSeek-V2-Lite Q4_K_M 的 GGUF routed-expert 流式 MoE、MLA reference
  路径与 CPU/GPU 混合 offload（单请求 2.56–3.05 tokens/s，paged batch 已可运行）；
- fused Q4_K/Q5_0/Q8_0 MoE GEMV/GEMM 默认开启，直接消费 packed expert rows，
  去掉 FP16 expert 权重 materialize；grouped 版把 decode 每层 12 次 expert
  launch 合并为 2 次；`--no-fused-moe` 可回退 dequant+bmm；
- expert staging 使用双 pinned slot + per-slot down copy stream；
- 可选 packed expert LRU cache（`--expert-cache-mb`，默认 0）；
- 可选 GPU packed expert cache + pointer-array grouped kernel
  （`--gpu-expert-cache-mb`，默认 0）；
- 新增 mini 质量回归 suite：perplexity、MMLU-style、GSM8K-style、长链推理，
  并支持与 llama.cpp 同 prompt / 同 greedy 采样对照；
- DeepSeek MoE 已接入现有 paged scheduler：chunked prefill、batched decode、
  block-level KV 预算、prefix cache 和 CPU KV offload 接口；
- 新增 `--cpu-moe-layers`：把最后 N 个 routed-expert 层放到 CPU FP32
  （PyTorch oneDNN/AVX2）路径，GPU 保留 attention/shared/lm_head；
- 新增 `--cpu-experts-per-layer` + `--cpu-expert-layers`：按累计路由频次
  选 cold experts，CPU 用 fused AVX2 packed GEMV，GPU 保留 hot experts；
- 新增实验性 `--prefill-pipeline`：prefill 多 chunk 的 gate/up 也走 copy
  stream；本机 WDDM 上实测略慢，默认关闭；
- 新增默认开启的 prefill prefetch：后台线程预读下一 expert chunk 到双 pinned
  slot，主线程只做 H2D + grouped kernel；32-token prefill 3.43→2.03 s；
- 新增默认开启的 decode expert prefetch：router 出 expert IDs 后立即后台预读，
  与 assignment sort/padding 重叠；32-token decode 2.52→2.70 t/s；
- 新增实验性 predictive next-layer prefetch：用上一 token 路由预热下一层；
  实测额外读取/拷贝开销更大，默认关闭。
- 同一 runtime 支持 `first_k_dense_replace`、`moe_layer_freq`、dense-only
  层布局，dense MLP 可按显存自动放 GPU 或回退到 CPU FP32。

---

## 结果速览

测试环境：Windows、RTX 3050 Laptop 4 GB、Python 3.10.3、torch 2.5.1+cu121、CUDA 12.1、CuPy 14.2。

| 路径 | 量化 | 显存 | decode |
|---|---|---:|---:|
| Qwen2.5-3B 兼容层（NF4） | NF4 | 2.28 GiB reserved | 5.859 tokens/s |
| Qwen2.5-3B DeepInfer SM86 原生 | W4A16 + GQA + Graph | 2.49 GiB allocated | **42.7 tokens/s（稳态）** |
| Qwen2.5-3B 常驻网关 | W4A16 + 256 context | 约 2.5 GiB | **44.773 tokens/s / 0.895 秒一次请求** |
| Qwen2.5-7B 混合精度 | 4-bit attention/embed + 3-bit MLP/head | 3.20 GiB allocated | **25.962 平均 / 26.188 中位** |
| Qwen2.5-7B 整体 Q4 | 全 4-bit | 3.92 GiB allocated | 7.942 tokens/s（持续 paging） |
| DeepSeek-V2-Lite MoE Q4_K_M | Q4_K_M + grouped fused + double buffer | 2.23 GiB allocated | 2.64–3.01 tokens/s（64-token 2.63） |

> 这些结果均为 `model_executor` 级；`full_engine=false`、`native_runner_ready=false`。
> 不能把它们写成“完整 DeepInfer engine 已支持 Qwen2.5”。当前新增的是
> engine-core 原型，不是官方 engine 接入。上表是旧有单请求 / 固定槽位路径的
> 实测。

### DeepSeek-V2-Lite Q4_K_M 原生 MoE（2026-10-28）

目标不是把 15.7B 参数全部塞进 4 GB，而是让 DeepInfer 自己的执行层在
**RTX 3050 4 GB + 16 GB RAM** 上流式跑完 DeepSeek-V2-Lite：

- 9.65 GiB Q4_K_M GGUF 不整体加载；routed experts 由 GGUFIndex 按需读取；
- 每个 expert chunk 最多 8 个专家，Q4_K / Q5_0 / Q8_0 packed bytes 通过
  pinned staging buffer + 独立 copy stream 异步拷贝到 GPU；
- 默认走 grouped fused Q4_K/Q5_0/Q8_0 MoE GEMV/GEMM：一个 chunk 的 gate/up
  和 down 各只发一次 kernel，直接消费 packed expert rows，不再 materialize
  FP16 expert 权重；`--no-fused-moe` 保留 dequant + grouped bmm 回退路径；
- expert staging 使用双 pinned slot；down projection 在 per-slot copy stream
  上异步拷贝，prefill 当前 chunk 的读取可以准备下一个 chunk；
- 可选 packed expert cache：`--expert-cache-mb`，默认 0；512 MB 实测仅约 11%
  hit，且 CPU cache copy 比 NVMe 读更慢，因此默认关闭；
- GPU 不同时保留 64 个专家；
- attention / router / dense layer0 / shared experts / lm_head 按可用显存
  自动放在 GPU 或 CPU；嵌入留在 CPU；
- 最终 `peak GPU allocated ≈ 2.23 GiB`，加载后仍有约 1.0 GiB 空闲显存。

最终实测（greedy、Q4_K_M、单请求、短 prompt / max_cache=256）：

| 指标 | 结果 |
|---|---:|
| 模型加载 | 17.6 s |
| cold/warmup（4 token） | 3.6–16 s（取决于 OS page cache） |
| prefill（短 prompt） | 1.95–2.44 s |
| decode 稳态 | **2.64–3.01 tokens/s**（64-token 长生成 2.63） |
| peak GPU allocated / reserved | **2.23 GiB / 2.31 GiB** |
| 对照 dequant+bmm | decode 2.43–2.77；长生成 2.43 |
| 对照 per-expert fused | decode 2.56–3.05；长生成 2.63；launch 数 312/token |
| 对照 grouped 无 double buffer | decode 2.53–2.95；prefill 2.20–2.88 |
| shared experts / lm_head | GPU（auto 放置） |
| 质量抽检 | MoE 解释、首都、翻译、`5-2=3`、`9.9 > 9.11` 均正确 |

同一份 Q4_K_M 用 llama.cpp CPU 基线约 9–13 tokens/s，因此这条路径不是“最快”；
它证明的是 **DeepInfer 自己的 MoE/MLA 执行层可以在 4 GB 卡上不 OOM 地跑起来**，
而 fused packed MoE kernel 已经把 dequant materialize 往返去掉；后续还可以继续
通过多 expert 合并 kernel、专家 prefetch 和多流流水线提高。
复现：

```powershell
F:\QwenQLoRA\venv\Scripts\python.exe `
  scripts\smoke_deepseek_v2_lite_moe.py `
  --prompt "用一句话解释什么是混合专家模型（MoE）。" `
  --max-new-tokens 24 --warmup-tokens 4 `
  --output docs\results\deepseek_v2_lite_q4km_moe_sm86.json
```

详见 `docs/ARTICLE_9_DEEPSEEK_V2_LITE_MOE_CPU_GPU_OFFLOAD.md`。

### Paged continuous batching（3B W4A16，4 GB）

同一台 RTX 3050 Laptop 4 GB、Qwen2.5-3B W4A16，`max_cache=128`、
`prefill_chunk_tokens=32`、`max_new_tokens=16`、无 prefix cache：

| 路径 | concurrency | tokens/s | TTFT p50 | TPOT p50 |
|---|---:|---:|---:|---:|
| SDPA reference | 1 | 6.906 | 0.946 s | 0.0914 s |
| native paged GQA | 1 | 8.144 | 0.924 s | 0.0693 s |
| native paged + batched W4 GEMV | 1 | 8.592 | 0.712 s | 0.0766 s |
| SDPA reference | 4 | 12.049 | 3.510 s | 0.1398 s |
| native paged GQA | 4 | 12.466 | 3.475 s | 0.1299 s |
| native paged + batched W4 GEMV | 4 | 14.736 | 2.746 s | 0.1214 s |
| native paged + batched W4 + INT8 KV | 4 | 13.868 | 2.831 s | 0.1324 s |
| tiled + overlap + FP8 E4M3 KV | 4 | 22.527 | 1.345 s | 0.1061 s |

`max_num_seqs=8` 时，**旧的串行 prefill 路径**达到 15.931 tokens/s，但
TTFT p50 约 5.24 s。这是 tiled/overlap 之前的基线；新路径的 c8 结果见下节。
完整数据和长文档插入抖动实验见
`docs/P0_CHUNKED_PREFILL_AND_MEMORY_BUDGET.md`。

> 不要把上表的 8.592 tokens/s 和旧静态 Graph 路径的 42~44 tokens/s 直接比较：
> 前者是 eager 动态 batch + chunked prefill + paged KV，后者是固定 shape
> static cache + CUDA Graph 的单请求最低延迟路径。两条路径的目标不同。

### Tiled prefill + prefill/decode overlap（2026-10-27）

在上一节同一套 3B W4A16、`max_cache=128`、`prefill_chunk_tokens=32`、
`max_new_tokens=16`、无 prefix cache 条件下，本轮新增：

- W4/W3 tiled prefill GEMM；
- paged scheduler 的 prefill/decode 双 CUDA stream overlap。

新结果：

| 并发 | 旧串行 prefill tokens/s | 新 tiled + overlap tokens/s | 提升 |
|---:|---:|---:|---:|
| 1 | 8.592 | **11.189** | 1.30× |
| 2 | 12.227 | **17.484** | 1.43× |
| 4 | 14.736 | **22.779** | 1.55× |

同轮测试 TTFT p50：

| 并发 | 旧 | 新 |
|---:|---:|---:|
| 1 | 0.712 s | **0.339 s** |
| 2 | 1.159 s | **0.515 s** |
| 4 | 2.746 s | **1.337 s** |

overlap 开关对照（均启用 tiled prefill）：

| 并发 | no-overlap TPOT p50 | overlap TPOT p50 | no-overlap jitter | overlap jitter | actual overlap steps |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0728 s | 0.0727 s | 0.00725 | 0.00842 | 0 |
| 2 | 0.0796 s | **0.0717 s** | 0.0332 | **0.0087** | 1 |
| 4 | 0.1056 s | 0.1045 s | 0.1583 | 0.1570 | 3 |

结论：**tiled prefill 是主要吞吐收益来源；stream overlap 在 4 GB 带宽受限卡上
主要改善部分 TPOT/jitter，不是吞吐翻倍按钮。**
扩展到 `max_cache=256` 的 c8 sweep（tiled + overlap）：

| 并发 | tokens/s | TTFT p50 | TPOT p50 | actual stream overlap steps |
|---:|---:|---:|---:|---:|
| 8 | **25.912** | **2.537 s** | 0.1570 s | 7 |

相对旧串行 c8 的 15.931 tokens/s / 5.243 s，吞吐约 1.63×，TTFT 约减半。
原始数据：`docs/results/paged_concurrency_3b_tiled_overlap_c8.json`。

### KV FP8 与 CPU cold-block offload（2026-10-27）

- FP8 E4M3 / E5M2 KV 有 native paged decode kernel，不再走 SDPA fallback；
  block bytes 从 589824 降到 294912；
- FP8 c1 / c2 / c4 为 11.089 / 17.317 / 22.527 tokens/s，接近 FP16 tiled
  + overlap 的 11.189 / 17.484 / 22.779；
- CPU offload 是请求级 used-block swap：GPU 压力大时把旧请求 KV 移到
  pinned CPU，block 足够后自动 reload；
- 3B 强制 3 个 GPU block 的 5 轮测试中，10/10 请求完成，每轮 swap/resume
  各 1 次；FP16 offload/reload copy 约 1.46/0.26 ms，FP8 约 1.12/0.26 ms；
- swap 等待会拉长被换出请求的 TPOT，它是 OOM 压力释放阀，不是吞吐优化。

详见 `docs/ARTICLE_8_KV_FP8_AND_CPU_OFFLOAD.md`。

## 本次更新（2026-10-28）

今天新增 **DeepSeek-V2-Lite Q4_K_M 的原生 MoE/MLA 执行层**，并把 DeepSeek-V2
的 dense / hybrid 层布局补齐。

### 1. 原生 MoE 流式执行

- 新增 `deepinfer/moe/gguf_index.py`：
  - GGUF v2/v3 header/metadata/tensor 目录解析；
  - 加载前校验 tensor data 是否完整，拒绝半截文件；
  - 按 expert 读取 3D expert tensor 的 packed slice。
- 新增 `deepinfer/moe/gpu_quant.py`：
  - SM86 上的 Q4_K、Q5_0、Q8_0 GPU 反量化；
  - 与 `gguf.dequantize` 的 CPU 结果逐元素对照通过；
  - 同一 packed 输入重复调用 bitwise 一致。
- 新增 `deepinfer/moe/deepseek_v2_runtime.py`：
  - DeepSeek-V2-Lite Q4_K_M GGUF 直接加载；
  - routed experts 按 chunk 流式读取，GPU 不同时 materialize 64 个专家；
  - 并行 expert slice 读取 + pinned staging + 独立 copy stream；
  - CUDA Event 防止 pinned buffer 复用 race；
  - grouped bmm 替代逐 expert 小 GEMM；
  - deterministic assignment scatter + FP32 top-k 求和；
  - attention / router / dense MLP / shared experts / lm_head 自动放置；
  - embedding 留在 CPU，DynamicCache + MLA reference 生成循环。
- 新增 `scripts/smoke_deepseek_v2_lite_moe.py`：
  - 质量/吞吐 CLI；
  - 自动 warmup；
  - 输出结构化 JSON，包含 prefill/decode 时间、峰值显存、token IDs、
    expert read/H2D/dequant 累计时间和层布局。

### 2. Dense / Hybrid 层支持

- 新增 `deepinfer/moe/layout.py`：
  - 镜像官方 `DeepseekV2DecoderLayer` 的 MoE 判定；
  - 支持 `first_k_dense_replace`、`moe_layer_freq`；
  - 支持 `n_routed_experts=None/0` 的 dense-only 配置。
- `DeepSeekV2LiteGGUFRuntime` 现在会：
  - 对 dense 层加载 `blk.{i}.ffn_gate/up/down.weight`；
  - 只为真实 MoE 层创建 expert store/router/shared experts；
  - dense MLP 默认按显存 `auto` 放置：GPU 放得下用 FP16，否则回退 CPU
    FP32，并在回退前检查 host RAM 余量；
  - CLI 新增 `--dense-mlp-device auto|cuda|cpu`；
  - 结果 JSON 记录 `dense_layer_indices`、`moe_layer_indices`、
    `dense_mlp_devices`。

### 3. Fused packed-weight MoE GEMV/GEMM

- 新增 `deepinfer/moe/grouped_moe.py`：
  - 一个 gate/up kernel 处理整个 expert chunk；
  - 一个 down kernel 处理整个 chunk；
  - block 的 2D grid 为 `(output_row, expert)`，counts 数组控制每个 expert
    的真实 token 数；
  - decode 每层 launch 从 12 次降到 2 次，每 token 从 312 次降到 52 次；
  - 当前实测长生成 decode 与 per-expert fused 基本持平（2.64 vs 2.63 t/s），
    prefill 略慢；瓶颈已经从 launch 转向 packed expert 读取和 H2D。
- 新增 double-buffer staging：
  - 每个 projection 两个 pinned slot；
  - slot 的 H2D 事件按 slot 记录，复用前才等待；
  - down 使用 per-slot copy stream，gate/up 保留在 compute stream；
  - prefill 长 prompt 从 2.95 s 降到 2.56 s，5-prompt prefill 从 2.20–2.88 s
    降到 **1.95–2.44 s**，decode 基本不变。
- 可选 packed expert cache：
  - 以 `(tensor_name, expert_id)` 为 key 的 per-layer LRU；
  - 第一次访问只计数，第二次访问才 admission，避免随机 expert churn；
  - `--expert-cache-mb` 控制，默认 **0**；
  - 512 MB 实测 hit rate 约 **11%**，但因为 CPU cache copy 和 eviction 开销，
    长生成从 2.63 t/s 降到 1.61 t/s；因此建议保持默认关闭。
- 可选 GPU packed expert cache + pointer-array grouped kernel：
  - 新增 `deepinfer/moe/ptr_grouped_moe.py`，kernel 接受每个 expert 的
    packed 权重指针数组，不再要求连续 packed buffer；
  - cache 存在 GPU 上，命中时跳过文件读和 H2D；
  - prefill 阶段自动关闭 cache，decode 前清空并启用，避免 prefill 污染；
  - 1 GiB 实测 hit rate **24.8%**，64-token decode 从 2.63 提升到
    **2.69 t/s**，但 peak allocated 从 2.23 GiB 升到 **3.20 GiB**；
  - 512 MB 时只有 12.0% hit，长生成反而降到 2.40 t/s；
  - 由于 4 GB 卡的 OOM 风险，`--gpu-expert-cache-mb` 默认 **0**。
- 新增 `deepinfer/moe/fused_moe.py`：
  - `dp_fused_gate_up_gemm`：一个 CUDA block 处理一个中间维输出 row，
    线程先从 Q4_K/Q5_0/Q8_0 packed row 解码权重到寄存器，再对多个 token
    做 dot-product，并直接写 `silu(gate) * up`；
  - `dp_fused_down_gemm`：同样直接消费 packed down row，写回 hidden；
  - GEMV（decode 单 token）和 GEMM（prefill 多 token）共用同一套 kernel，
    多 token 时权重只解码一次；
  - packed bytes 直接进 kernel，不再生成/写入/读回 FP16 expert 权重；
  - `--no-fused-moe` 可切回旧的 dequant + grouped bmm 路径做 A/B 对照。

效果：

| 指标 | dequant+bmm | per-expert fused | grouped fused |
|---|---:|---:|---:|
| 64-token 长生成 decode | 2.43 t/s | 2.63 t/s | **2.64 t/s** |
| 5-prompt decode 范围 | 2.43–2.77 | **2.56–3.05** | 2.53–2.95 |
| peak GPU allocated | 2.36 GiB | 2.23 GiB | **2.23 GiB** |
| peak GPU reserved | 2.43 GiB | 2.31 GiB | **2.31 GiB** |
| expert launch / decode token | 0（cuBLAS 3/layer） | 312 | **52** |
| expert dequant launch 累计 | 非零 | **0** | **0** |

5 个质量 prompt 和 64-token 长生成的 token IDs 在 fused / dequant+bmm 两条路径
逐 token 完全一致。

### 4. 实测结果

RTX 3050 Laptop 4 GB + 16 GB RAM，DeepSeek-V2-Lite Q4_K_M：

| 指标 | 结果 |
|---|---:|
| 模型加载 | 17.6 s |
| prefill（短 prompt） | 1.95–2.44 s |
| decode 稳态 | **2.64–3.01 tokens/s**（64-token 长生成 2.63） |
| peak GPU allocated / reserved | **2.23 GiB / 2.31 GiB** |
| 加载后 GPU 空闲 | 约 1.0 GiB |
| 质量抽检 | MoE 解释、首都、翻译、`5-2=3`、`9.9 > 9.11` 均正确；grouped/per-expert/回退 token IDs 一致 |
| 可选 expert cache | 512 MB 时约 11% hit，但长生成 1.61 t/s，默认关闭 |
| 可选 GPU expert cache | 1 GiB 时 hit 24.8%，长生成 2.69 t/s，peak 3.20 GiB；默认关闭 |
| mini 质量回归 | MMLU-style 8/10、GSM8K-style 4/8、长链 3/5；llama.cpp 为 8/10、2/8、1/5 |
| full-category 质量回归 | MMLU 27/57（47.4%，57 科各 1 题）、GSM8K 9/32（28.1%）、长链 2/32（6.25%）、PPL 6.59 |
| paged scheduler | 2 请求 batched decode；prefix cache 复用 16 token；token IDs 与单请求一致 |
| paged prefill/decode overlap | 已开启；CuPy 跟随 torch stream；token IDs 一致 |
| DeepSeek OpenAI server | chat/completions、raw completions、SSE、health/models；paged scheduler 接入 |
| INT8/FP8 paged MLA | native quant kernel；KV 106→约 53 MiB；INT8/FP8/E5M2 单测通过 |
| ECS control plane | C++ heat/LRU/tier + Python placement；opt-in；模拟 2 卡 all-to-all |
| CPU layer offload（1 层） | decode 1.86 t/s；32-token token IDs/text 与 GPU 路径完全一致 |
| Hot/cold expert split（2 层 × 1 cold） | decode **2.50 t/s**；32-token IDs/text 一致 |
| prefill copy-stream pipeline | 可选；本机实测 prefill 慢约 6%，默认关闭 |
| prefill background prefetch | 默认开启；prefill 3.43→**2.03 s**，token IDs 一致 |
| decode background prefetch | 默认开启；decode 2.52→**2.70 t/s**，token IDs 一致 |
| predictive next-layer prefetch | 可选；实测 decode 降到 2.21 t/s，默认关闭 |

同一份 Q4_K_M 的 llama.cpp CPU 基线约 9–13 tokens/s；本实现的重点是 native
MoE/MLA 执行层和 4 GB 显存下的可控 offload，不是与 llama.cpp 比绝对速度。

### 5. 测试与文档

- `tests/moe/` 新增 GGUF 索引、截断拒绝、expert slice、Q4_K/Q5_0/Q8_0
  GPU 反量化、dense/MoE 层布局、CPU MLP fallback、per-expert fused 和
  grouped MoE kernel 对照测试；
- 全套测试：**103 passed**；
- 新增文章：
  `docs/ARTICLE_9_DEEPSEEK_V2_LITE_MOE_CPU_GPU_OFFLOAD.md`；
- 结构化结果：
  `docs/results/deepseek_v2_lite_q4km_moe_sm86.json`（fused）、
  `docs/results/deepseek_v2_lite_q4km_moe_dequant_bmm_sm86.json`（回退基线）、
  `docs/results/deepseek_v2_lite_q4km_moe_fused_vs_dequant.json`（fused vs 回退）、
  `docs/results/deepseek_v2_lite_q4km_moe_grouped_vs_per_expert.json`（grouped vs per-expert）、
  `docs/results/deepseek_v2_lite_q4km_moe_quality_eval.json`（mini 质量回归）、
  `docs/results/deepseek_v2_lite_q4km_moe_full_quality.json`（full-category 回归）、
  `docs/results/deepseek_v2_lite_q4km_moe_paged_scheduler.json`（paged scheduler）、
  `docs/results/deepseek_v2_lite_q4km_moe_paged_overlap.json`（prefill/decode overlap）、
  `docs/results/deepseek_v2_lite_q4km_moe_openai_server.json`（OpenAI server smoke）、
  `docs/results/deepseek_v2_lite_q4km_moe_paged_quantized.json`（INT8/FP8 paged MLA）、
  `docs/results/deepseek_v2_lite_q4km_moe_cpu_layer_offload.json`（CPU layer offload）、
  `docs/results/deepseek_v2_lite_q4km_moe_cpu_expert_offload.json`（oneDNN 版本）、
  `docs/results/deepseek_v2_lite_q4km_moe_cpu_expert_avx2.json`（fused AVX2 版本）、
  `docs/results/deepseek_v2_lite_q4km_moe_prefill_pipeline.json`（pipeline A/B）、
  `docs/results/deepseek_v2_lite_q4km_moe_prefill_prefetch.json`（prefill prefetch）、
  `docs/results/deepseek_v2_lite_q4km_moe_decode_prefetch.json`（decode prefetch）、
  `docs/results/deepseek_v2_lite_q4km_moe_decode_predictive.json`（预测式 prefetch）；
- PyPI 独立库：`pypi/deepinfer-sm86/`；PyPI 已发布 `0.1.0`，
  本地 ECS `0.2.0` wheel/sdist 已构建但尚未上传；
- 已知边界：
  - paged scheduler 已接入 chunked prefill / batched decode / KV 预算，但
    DeepSeek 暂不支持 prefill/decode stream overlap；
  - MLA 已有 blockwise paged attention，但不是 fused CUDA/FlashMLA kernel；
  - grouped GEMV/GEMM 是 block-per-row 的 2D-grid 实现；它把 decode launch
    降到 52/token，但 wall time 仍主要受 packed expert 读取和 H2D 限制；
  - 尚未做 Tensor Core tile、cp.async 或真正的跨层专家预取；
  - dense/hybrid 路径有单元测试，但本机没有 dense-only DeepSeek GGUF 做
    端到端 benchmark；
  - Q4_K_M 有损，不能宣传成 BF16 等价。

### 6. 质量回归（mini suite）

新增 `scripts/eval_deepseek_moe_quality.py` 和
`docs/quality/deepseek_v2_lite_quality_suite.json`。这不是完整 MMLU/GSM8K，
而是一个可重复的快速 release gate：

- 4 段 raw text 的 next-token perplexity；
- 10 道 MMLU-style 四选一；
- 8 道 GSM8K-style 数学应用题；
- 5 道多步长链推理；
- 同 prompt、同 greedy 采样（temperature=0 / top-k=0 / top-p=1）跑
  `llama.cpp` CPU 对照；
- 支持 `--skip-llama`、`--limit`、`--no-fused-moe`。

| 指标 | DeepInfer native | llama.cpp CPU |
|---|---:|---:|
| MMLU-style | **8/10（80%）** | 8/10（80%） |
| GSM8K-style | **4/8（50%）** | 2/8（25%） |
| 长链推理 | **3/5（60%）** | 1/5（20%） |
| raw-text PPL | 4.97 | 3.19 |

PPL 不是严格同口径：native 是每段文本从 BOS 开始独立打分后按 token 加权，
llama.cpp 是 `llama-perplexity` 对拼接语料按 `n_ctx=128` 分 chunk；tokenizer/
context boundary 不同，不能把这个差值直接解释为模型质量差。MMLU/GSM8K 也只是
mini suite，不是官方完整数据集。

质量路径验证：

- fused 与 `--no-fused-moe` 在 3 个代表性 prompt 上 token IDs 完全一致；
- native 在 GSM8K-style 和长链上高于同 prompt 的 llama.cpp；
- 两个 MMLU-style 失败是模型输出冗长/算术错误，不是 kernel 崩溃或重复 token。

运行：

```powershell
F:/QwenQLoRA/venv/Scripts/python.exe ^
  scripts/eval_deepseek_moe_quality.py ^
  --output docs/results/deepseek_v2_lite_q4km_moe_quality_eval.json
```

### 6.1 Full-category 回归（2026-10-28）

新增：

- `scripts/build_deepseek_full_quality_suite.py`：从 MMLU、官方 GSM8K test
  和 WikiText-2 构建固定抽样 suite；
- `docs/quality/deepseek_v2_lite_full_quality_suite.json`；
- `scripts/eval_deepseek_full_quality.py`；
- `docs/results/deepseek_v2_lite_q4km_moe_full_quality.json`。

这不是完整的 14k MMLU / 1.3k GSM8K 全量跑分，而是一个固定预算的
**full-category** 回归：MMLU 覆盖全部 57 个 subject，每科 1 题；GSM8K
固定抽样 32/1319；长链为确定性生成的 32 道 5–7 步整数链；PPL 使用
WikiText-2 validation 的前 512 token。

| 指标 | native Q4_K_M | 说明 |
|---|---:|---|
| MMLU | **27/57（47.4%）** | 57 科各 1 题，zero-shot，Wilson 95% CI 35.0–60.1% |
| GSM8K | **9/32（28.1%）** | 固定 test 抽样，greedy |
| 长链推理 | **2/32（6.25%）** | 合成 5–7 步整数链，zero-shot |
| WikiText-2 PPL | **6.59** | 512 token，native score_text |
| llama.cpp PPL | 8.04 | llama-perplexity `n_ctx=128` 分块，非严格同口径 |

llama.cpp 生成对照没有跑满：CPU 上 16B Q4 生成耗时过高；PPL 已单独跑了
同一段文本作为参考。质量结论只应解释为 native 路径的固定预算回归，不应
宣传成官方 MMLU/GSM8K 全量成绩。

运行：

```powershell
F:/QwenQLoRA/venv/Scripts/python.exe ^
  scripts/eval_deepseek_full_quality.py ^
  --skip-llama ^
  --output docs/results/deepseek_v2_lite_q4km_moe_full_quality.json
```

官方全量 split 的 builder 开关已经提供，但本轮没有跑：

```powershell
python scripts/build_deepseek_full_quality_suite.py ^
  --mmlu-all --gsm8k-all --long-chain-count 512 --ppl-tokens 2048 ^
  --output docs/quality/deepseek_v2_lite_full_quality_suite.all.json
```

按本机 2.7 tok/s 估算，14k MMLU + 1.3k GSM8K 需要数小时到十几小时；这是
测试预算问题，不是 harness 缺失。当前发布门槛仍是固定预算的 full-category
suite。

CUDA Graph 仍不接入 DeepSeek decode：top-k 路由会改变 expert 集合、packed
buffer 地址和 chunk 数，当前不是可安全 capture 的静态图。代码和 server 都
明确 `use_graph=False`；如果要做，只能限定在固定 routing/buffer 的静态子图。

### 6.2 PyPI 独立库打包

仓库主体是应用/benchmark workspace；可复用库已单独放在
`pypi/deepinfer-sm86/`，包名 `deepinfer-sm86`，import 名仍为 `deepinfer`。

包含：

- GGUF expert index / packed reader；
- grouped Q4_K/Q5_0/Q8_0 MoE kernels；
- FP16 fused paged MLA + INT8/FP8 E4M3/E5M2 native paged MLA；
- CPU AVX2 packed dequant/GEMV C++ source；
- paged KV cache、memory budget、metrics；
- 通用 `PagedBatchScheduler`；
- ECS control plane：C++ heat/LRU/tier policy、Python placement 和模拟
  all-to-all transport。

不包含：模型权重、tokenizer 文件、Qwen server、benchmark/application
workspace。server 仍是仓库 app，不是库 wheel 的一部分。

构建：

```powershell
python scripts/build_pypi_package.py
cd pypi\deepinfer-sm86
python -m build --no-isolation --sdist --wheel
```

本地已生成：

```text
pypi/deepinfer-sm86/dist/deepinfer_sm86-0.2.0-py3-none-any.whl
pypi/deepinfer-sm86/dist/deepinfer_sm86-0.2.0.tar.gz
pypi/deepinfer-sm86/dist/SHA256SUMS
```

安装方式：

```powershell
# PyPI 当前已发布版本（不含 ECS 0.2）
pip install deepinfer-sm86==0.1.0
# ECS 0.2 本地 wheel
pip install .\pypi\deepinfer-sm86\dist\deepinfer_sm86-0.2.0-py3-none-any.whl
# GPU kernels for the local wheel:
pip install "deepinfer-sm86[cuda]"

# GitHub 直接安装库（不走 PyPI）
pip install "git+https://github.com/umouyoumou-del/DeepInfer-SM86.git@main#subdirectory=pypi/deepinfer-sm86"

# 或 clone 后 editable
git clone https://github.com/umouyoumou-del/DeepInfer-SM86.git
pip install -e .\DeepInfer-SM86\pypi\deepinfer-sm86
```

注意：仓库根目录仍是 application/benchmark workspace，没有根级
`pyproject.toml`，因此 `pip install git+https://.../DeepInfer-SM86.git`
（不带 `subdirectory`）**不是**库安装方式；必须使用上面的
`#subdirectory=pypi/deepinfer-sm86`，或者直接安装 PyPI 包。

Wheel 已在隔离 target 中验证：

- `import deepinfer`；
- `DeepSeekV2LiteGGUFRuntime`；
- `PagedBatchScheduler`；
- `deepinfer.moe.cpu_avx2.available()`；
- wheel 内包含 MoE/ECS `.cpp` 源文件和 LICENSE/NOTICE；
- `ECSController`、`ECSRouter` 和 C++ policy core 导入通过。

发布前检查 PyPI 上 `deepinfer-sm86` 名称是否可用，然后：

```powershell
python -m twine upload dist/*
```

### 6.3 ECS：Expert Coordination System

ECS 是这一轮新增的控制面/数据面分层：

- 数据面：GPU 继续做 Attention、grouped MoE GEMM 和当前专家计算；
- 控制面：CPU 侧 `ECSController` + C++ `ecs_core.cpp` 负责 heat、LRU、
  tier placement 和 victim selection；
- 存储层：L1 GPU packed expert cache、L2 packed CPU cache、L3 GGUF/NVMe
  文件读取；
- 批处理：C++ `ecs_plan_chunks` 和现有 paged scheduler 共同完成
  expert/token chunk packing；
- 多设备：`ECSRouter` 生成 device-addressed `ExpertPacket`，提供
  `SimulatedAllToAll` 和可选 `TorchCollectiveAllToAll`。

模块：

```text
deepinfer/ecs/core.py          runtime C++ build + ctypes facade
deepinfer/ecs/controller.py    heat/LRU/tier placement + cold selection
deepinfer/ecs/transport.py     packets, router, simulated/torch all-to-all
deepinfer/ecs/ecs_core.cpp     CPU policy core
```

启用方式：

```powershell
python scripts/smoke_deepseek_v2_lite_moe.py `
  --ecs --ecs-l1-mb 32 --ecs-l2-mb 0 `
  --cpu-experts-per-layer 1 --cpu-expert-layers 2 ...
```

实测 smoke：

- ECS backend：**cpp**；
- 输出正确：`5-2=3`；
- decode 2.36 tok/s（该 smoke 启用了 cold-expert split，不是默认 2.7）；
- ECS 记录了 3642 次 L3 read、1908 个 placement entries；
- 默认 `--ecs` 关闭，不改变原 GPU 路径。

需要明确：本机只有一张 RTX 3050，因此 **真实 8 卡 P100 / All-to-All 未实机验证**；
多设备部分目前是 device abstraction + simulated transport + torch distributed
adapter。ECS 的 Level 2 仍需要多卡硬件和 NCCL/Gloo 环境才能验收。

结果文件：`docs/results/deepseek_v2_lite_q4km_moe_ecs_smoke.json`。

### 7. Paged scheduler 接入

DeepSeek-V2-Lite 已接到现有 `apps/paged_scheduler.py`：

- chunked prefill（`prefill_chunk_tokens=8`）；
- batched decode（`max_num_seqs=2`，实测 max decode batch = 2）；
- block-level KV allocator / KV 预算；
- prefix cache；
- CPU KV offload 接口保留。

新增 `deepinfer/moe/paged_adapter.py`：

- 把 DeepSeek MLA 的 K/V 写入通用 paged pool；
- 适配 MLA 的 `qk_nope + qk_rope = 192` 与 `v_head_dim = 128` 不同维度；
- `attend()` 直接按 block table 逐块读取 K/V，用 online softmax 累积；
- 不再 materialize 当前 batch 的 dense K/V；
- 官方 DeepSeek attention 已换成 paged forward：q_len=1 和 chunk prefill
  都走 blockwise attention；
- FP16 路径新增 `deepinfer/moe/paged_mla_kernel.py`：一个 warp 负责一个
  `(batch, head, query-token)`，直接在 CUDA 里读 block table + online
  softmax，不再经过 Python block loop；
- INT8 使用 native quant paged MLA：按 block/head/token 的 K/V FP16 scale
  直接反量化，不 materialize dense K/V；
- FP8 E4M3/E5M2 使用 native byte kernel，直接读取 FP8 storage；
- 只有不支持的形状/dtype 才回退 PyTorch block loop 或 dense materialize。

同一组 2 请求 + prefix cache smoke：

| 路径 | 耗时 |
|---|---:|
| dense materialize | 11.89 s |
| PyTorch block loop | 13.25 s |
| fused FP16 CUDA paged MLA（scalar） | **11.43 s** |

fused kernel 提供三种映射：

- `scalar`（默认）：q 缓存在寄存器，标量 coalesced K/V load；
- `half2`：half2 向量化；
- `cta`：4 warp 分摊 key range，再合并 partial softmax。

kernel 微基准（200 次，B2/H16/Q1/seq28）：

| variant | ms/call |
|---|---:|
| scalar | **0.0616** |
| half2 | 0.0676 |
| cta | 0.0666 |

scalar 变体在这个 smoke 里略快于 dense materialize；token IDs 仍完全一致。

INT8/FP8 native quant MLA（2 请求、prefix cache、20 blocks）：

| KV dtype | KV memory | token IDs vs FP16 | 说明 |
|---|---:|---:|---|
| FP16 | 106.2 MB | 基准 | fused FP16 kernel |
| INT8 | **53.6 MB** | 语义正确，算术输出空格/tokenization 改变 | per-head/token scale |
| FP8 E4M3 | **53.1 MB** | **完全一致** | 直接 FP8 byte decode |

INT8/FP8/E5M2 三种 storage 都有单元测试，与 materialize/dequant 后的 PyTorch
reference 对比通过。结果文件：
`docs/results/deepseek_v2_lite_q4km_moe_paged_quantized.json`。
不过它还不是 FlashMLA：主要收益来自去掉 dense K/V materialize，以及减少
Python block loop/launch 开销。

实测（2 请求、chunked prefill 8、prefix cache）：

| 指标 | 结果 |
|---|---:|
| 2 请求 token IDs 与单请求 baseline | **完全一致** |
| max decode batch | **2** |
| prefix cache 复用 | **16 token** |
| prefix 后 token IDs | **完全一致** |
| KV pool | 20 blocks，约 106 MB |
| native paged attention | **开启（27 层）** |
| 2 请求 + prefix 总耗时 | **11.43 s**（materialize 11.89 s） |
| prefill/decode stream overlap | **开启** |
| overlap 后 token IDs | **完全一致** |

prefill/decode overlap 已启用。关键修复不是简单开两个 CUDA stream，而是让
**CuPy raw kernels 跟随 torch 当前 stream**：scheduler 在 prefill/decode
stream 上下文里用 `cp.cuda.ExternalStream(torch_stream.cuda_stream)` 设置
CuPy current stream。否则 PyTorch attention/linear 在非默认 stream，而 MoE/
paged MLA 的 CuPy kernel 仍落在默认 stream，重叠时会读到未就绪数据并输出
乱码。

短 smoke A/B（2 请求、prefix cache、max_new=8）：

| 配置 | 总耗时 | overlap steps | token IDs |
|---|---:|---:|---:|
| serial | 10.07 s | 0 | 一致 |
| overlap | **9.73 s** | 1 | 一致 |

24-token 对照为 9.66 s serial vs 9.95 s overlap，说明这个 workload 只有
一步真实 overlap，收益在噪声内；当前结论是**正确性和 stream 布线已验证**，
不是宣称大幅吞吐提升。结果文件：
`docs/results/deepseek_v2_lite_q4km_moe_paged_overlap.json`。

运行：

```powershell
F:/QwenQLoRA/venv/Scripts/python.exe ^
  scripts/smoke_deepseek_v2_lite_moe_paged.py ^
  --enable-prefill-decode-overlap --prefix-reuse ^
  --output docs/results/deepseek_v2_lite_q4km_moe_paged_scheduler.json
```

### 8. CPU MoE layer offload

新增 `--cpu-moe-layers N`：把最后 N 个真实 MoE 层放到 CPU FP32 路径，GPU
继续执行 attention、shared experts 和 lm_head。

实现方式：

- runtime 计算 `moe_layer_indices[-N:]`；
- 这些层在 `_streaming_moe_forward` 中走 `_moe_cpu_fallback()`；
- CPU 路径复用 `GGUFExpertStore.load_experts(device='cpu', dtype=torch.float32)`；
- 每个 expert chunk 仍限制为 `expert_chunk_size=8`，避免 CPU FP32 权重爆内存；
- 默认 `--cpu-moe-layers 0`，不改变默认 GPU 路径。

实测（32-token 长故事）：

| 指标 | 全 GPU | 最后 1 层 CPU |
|---|---:|---:|
| decode | 2.50 t/s | **1.86 t/s** |
| prefill | 3.19 s | 4.39 s |
| peak GPU allocated | 2259 MiB | 2259 MiB |
| token IDs / text | 基准 | **完全一致** |

这次整层 CPU 数据是用 oneDNN fallback 测的；当前共享 CPU 后端已经增加 fused
AVX2 packed GEMV。层切分验证了 CPU 专家读取和 GPU/CPU 结果一致，代价是明显变慢。
下一节进一步做了热门/冷门 expert 级切分。

### 9. Hot/cold expert split + AVX2 packed CPU

在 CPU layer offload 之外，新增 expert 级切分：

- `--cpu-experts-per-layer N`：每层选择 N 个 cold experts；
- `--cpu-expert-layers N`：只对最后 N 个 MoE 层启用 expert 切分；
- 每层维护累计路由计数；从当前 selected experts 中选计数最低的 N 个；
- GPU 计算 hot experts，CPU 计算 cold experts；
- CPU packed Q4_K gate/up + Q5_0/Q8_0 down 由 `cpu_avx2_gemv.cpp` 的
  fused AVX2 GEMV 解码并直接累加，不再先 materialize FP32 权重；
- CPU expert 权重有 512 MB LRU cache（`--cpu-expert-cache-mb`）；
- 默认关闭，不影响现有 GPU 路径。

实测矩阵（每层 1 个 cold expert）：

| CPU 层数 | decode | prefill | CPU MoE | token IDs |
|---:|---:|---:|---:|---:|
| 2 | **2.50 t/s** | 3.33 s | 0.31 s | 一致 |
| 3 | 2.49 t/s | 3.39 s | 0.56 s | 一致 |
| 4 | 2.30 t/s | 3.82 s | 0.65 s | 一致 |

推荐 **2 层 × 1 cold expert**：几乎无速度损失，同时仍把一部分 routed expert
放到 CPU。CPU 路径现在使用 fused AVX2 packed GEMV（OpenMP 按输出 row 并行），
并用固定 token 累加数组减少 K-block 循环里的小分配。

这比整层 CPU offload 的 1.86 t/s 和 oneDNN 版本的 2.23 t/s 都好，说明少量
cold expert 放到 CPU 是可行的显存/速度折中。当前 CPU cold-expert 路径已经是
fused AVX2 packed GEMV；oneDNN 版本作为 fallback 保留。

### 10. Prefill copy-stream pipeline（实验，默认关闭）

新增 `--prefill-pipeline`：

- prefill 的 gate/up/down 三个 packed expert H2D 都发到该 slot 的 copy stream；
- compute stream 先等 gate/up，down 继续异步；
- 配合双 pinned slot，尝试让下一个 expert chunk 的读取/H2D 和当前 grouped
  kernel 重叠；
- decode 仍保持 gate/up 在 compute stream，避免事件开销。

A/B 实测（32-token 长故事）：

| 配置 | prefill | decode |
|---|---:|---:|
| pipeline off | **3.11 s** | **2.58 t/s** |
| pipeline on | 3.29 s | 2.56 t/s |

token IDs / text 两种配置完全一致。由于 Windows/WDDM 上 copy stream + event
调度开销，pipeline 在本机反而让 prefill 慢约 6%，因此默认关闭，仅保留为
实验开关。

另外试验了 `--prefill-all-experts`（把所有 routed expert 合并成一个 prefill
chunk）：token IDs 正确，但 prefill 3.59 s，也更慢，所以同样默认关闭。

### 10.1 显式 background prefetch（默认开启）

新的有效路径不是 copy-stream overlap，而是 **CPU 后台预读**：

- store 维护 2 个 prefetch pinned slot；
- 处理当前 expert chunk 前，先通过 `ThreadPoolExecutor` 提交下一 chunk 的
  packed expert 读取；
- 主线程只负责当前 chunk 的 H2D 和 grouped kernel；
- 下一 chunk 完成后，主线程消费 prefetch buffer；
- 用 Event 保证 prefetch slot 在旧 H2D 完成前不会被覆盖。

32-token A/B：

| 配置 | prefill | decode | token IDs |
|---|---:|---:|---:|
| prefetch off | 3.43 s | 2.55 t/s | 基准 |
| prefetch on | **2.03 s** | 2.51 t/s | **完全一致** |

prefill 降低约 **41%**，decode 在噪声范围内。该路径默认开启；需要对照时用
`--no-prefill-prefetch`。

### 10.2 Decode expert prefetch（默认开启）

decode 只有 top-6 一个 expert chunk，无法提前知道下一层路由，但有另一个可
重叠窗口：

- router 产出 expert IDs 后，立即在后台线程预读这 6 个 expert 的 packed bytes；
- 主线程同时完成 GPU sort / padding / assignment 准备；
- 之后只做 H2D 和 grouped kernel；
- 双 pinned prefetch slot + Event 防止覆盖未完成 H2D 的源 buffer。

32-token 对比：

| 配置 | decode | token IDs |
|---|---:|---:|
| decode prefetch off | 2.52 t/s | 基准 |
| decode prefetch on | **2.70 t/s** | **完全一致** |

paged scheduler 2 请求 + prefix cache smoke 从 11.07 s 降到 **9.70 s**，
token IDs 仍一致。该路径默认开启，`--no-decode-prefetch` 可关闭。

### 10.3 Predictive next-layer prefetch（实验，默认关闭）

实现了“用上一 token 路由预热下一层”：

- 在层 L 记录当前 token 的 top-6 路由；
- 处理下一层 L+1 时，用上一 token 在 L+1 的 top-6 路由，后台预读 packed
  gate/up/down 到 predictive cache；
- 当前层计算与这些预读重叠；
- 下一 token 真正进入 L+1 时，命中部分 expert 直接复用 cache，未命中再补读；
- cache 256 MB，key 带 layer/projection/expert，避免跨层串数据。

问题是指令 mix 并没有足够稳定：16-token probe 中相邻 token 的 top-6 集合
重叠约 12%–45%，exact match 几乎为 0。结果是：

| 配置 | decode | predictive cache hits | prefetch reads |
|---|---:|---:|---:|
| predictive off（decode prefetch on） | **2.70 t/s** | — | — |
| predictive on | 2.21 t/s | 8,822 | 14,850 |

浪费的预测读取和 partial pinned copy 抵消了命中收益，因此该路径只作为
`--predictive-prefetch` 实验开关，默认关闭。token IDs/text 仍然完全一致。




CUDA Graph 没有强行接入：DeepSeek 的 top-k 路由会改变 expert 集合、tensor
地址和 chunk 数量，当前不是可安全 capture 的静态图。

## 上一轮更新（2026-10-27）

- 新增 `deepinfer/engine/kv_cache.py`：
  - 引用计数 block allocator；
  - paged KV pool / block table / tensor-only validity mask；
  - 有界 LRU prefix cache，支持完整 prompt block 跨请求复用；
  - KV OOM 隔离：单个请求失败不会污染其他 batch 成员。
- 新增 `apps/paged_scheduler.py`：
  - 可变 batch 连续批处理，`max_num_seqs` 可配置；
  - chunked prefill：prefill 按 `--prefill-chunk-tokens` 分块，与 decode 交替；
  - 请求状态机 `PREFILL / DECODE`，多 prefill round-robin 公平调度；
  - 短请求按需分配 block，结束后立即回收；
  - admission 时预留 prompt + output block，KV OOM 请求级隔离；
  - runtime OOM / cancel 隔离；
  - prefix-cache 命中统计与 Prometheus 指标。
- 新增 `deepinfer/engine/memory_budget.py`：
  - 根据模型权重、CUDA 空闲显存、reserve、utilization 计算 KV block 数；
  - `/health` 输出 memory_budget / KV usage / prefill / decode / queued 状态。
- 网关新增 `/v1/completions`、stop 截断、`user` / `X-Tenant-ID` 租户隔离；
- 新增 `--kv-cache-dtype int8`：paged K/V 用 int8 存储，FP16 scale sidecar，
  并有独立 native paged decode kernel；
- 新增 native paged GQA decode kernel 与 batched W4A16 GEMV，详见
  `docs/ARTICLE_6_NATIVE_PAGED_KERNEL_AND_4GB_BENCH.md`；
- 新增 tiled W4A16 / W3A16 prefill GEMM，peer 对比旧 dequantize fallback：
  W4 2.24×–2.87×，W3 7.71×–9.45×；
- 新增 paged scheduler 的 prefill/decode 双 CUDA stream overlap：
  `_prefill_step` / `_decode_step` 拆成 prepare / launch / finish 三段，
  支持 `--no-prefill-decode-overlap` 回归开关；
- 新增 logical overlap 与 actual stream overlap 两个独立指标；
- 新增 FP8 E4M3/E5M2 paged KV 与 software-decode native paged kernel；
- 新增请求级 CPU KV swap：`--cpu-offload`、`--max-cpu-offload-mb`、
  `swapped` 状态、offload/reload 指标和 pinned CPU payload；
- 新增 `docs/ARTICLE_8_KV_FP8_AND_CPU_OFFLOAD.md` 和
  `scripts/bench_cpu_offload.py`；
- 新增 `scripts/bench_sm86_tiled_prefill.py` 和
  `docs/ARTICLE_7_PREFILL_DECODE_OVERLAP_AND_TILED_PREFILL.md`；
- 新增 `scripts/probe_model.py`，自动探测 HF/量化模型目录的支持状态；
- 新增 `docs/P0_CHUNKED_PREFILL_AND_MEMORY_BUDGET.md`。
- 通用 decoder 增加绝对 position 的 causal mask、paged mask 和 sliding-window mask；
- 新增 Mistral sliding-window 架构；
- 新增 Gemma / Gemma 2 dense 架构：
  - GeGLU、offset RMSNorm、embedding scale；
  - Gemma 2 sandwich norm、attention/final logit softcap、
    alternating sliding/full attention；
- W4A16 分层 checkpoint 导出/加载支持 Gemma 2 额外 FFN norms；
- 网关新增 `GET /metrics`、有界队列、流式请求取消和更完整的 `/health`；
- 新增 CPU 正确性回归：block/prefix/paged cache、paged batch logits、
  paged scheduler、Mistral/Gemma/Gemma2 HF logits 对齐；
- 新增 `scripts/bench_paged_scheduler.py`，用于在目标 GPU 上做
  1/2/4/8 并发 sweep，记录 TTFT/TPOT/吞吐/OOM/prefix 命中；
- 新增工程复盘文章 `docs/ARTICLE_4_ENGINE_CORE_POSTMORTEM.md`，记录实现过程和
  走过的弯路；
- 新增量化格式兼容层 `deepinfer/quantization/`：
  - GPTQ 2/4/8-bit、group quant、`g_idx`/desc_act；
  - AWQ AutoAWQ GEMM 4-bit 布局；
  - GGUF 单文件，支持 `gguf` 包可反量化的 ggml 类型（Q4_0、Q4_K 等）；
  - 统一转换为 SM86 layered W4A16 checkpoint；
  - 新增 CLI `scripts/import_quantized_checkpoint.py` 和端到端 CPU 测试；
  - 新增复盘文章 `docs/ARTICLE_5_QUANTIZATION_FORMAT_COMPATIBILITY.md`。

## 更早更新（2026-10-26）

- 新增 3B W4A16 分层 checkpoint 导出与常驻加载：
  - `scripts/export_qwen3b_w4_checkpoint.py`
  - `Qwen2W4A16ForCausalLM.from_checkpoint(...)`
  - `D:\Qwen3B-W4A16-SM86` 约 1.912 GiB，加载约 11.75 秒
- 网关新增 `--backend w4a16` / `--backend mixed`，3B 与 7B 都可作为 OpenAI 兼容服务；
- 新增 256-slot DeepInfer SM86 native GQA kernel；
- 新增 3B / 7B / Ollama 顺序基准脚本和原始结果：
  - `scripts/bench_productivity_backends.py`
  - `docs/results/productivity_backends_combined.json`
- 修复 `.gitignore` 中 `models/` 误忽略
  `deepinfer/model_executor/models/` 源码的问题；
- 补齐 `deepinfer/model_executor/`、`config.py`、`environ.py`、engine 类型和源码测试；
- 长文拆分为两篇独立文章：
  - 上篇：`docs/ARTICLE_1_DEEPINFER_SM86_KERNEL.md`
  - 下篇：`docs/ARTICLE_2_DEEPINFER_SM86_PRODUCTIVITY.md`
- 合并稿保留为：`docs/QWEN_DEEPINFER_4GB_ENGINEERING_ARTICLE.md`

### 7B 为什么不是纯 4-bit

Qwen2.5-7B 全 4-bit 权重约 **3.657 GiB**，但该卡实测只有 **3.237 GiB** CUDA
可用显存。纯 4-bit 会进入 WDDM paging：

- 预填充约 86 秒；
- decode 约 7.9 tokens/s。

7B 混合精度 checkpoint 约 **2.930 GiB**、有效位宽约 **3.30 bpw**，才有约
26 tokens/s 的速度；它不是 Q4，量化配置为：

```text
attention q/k/v/o : 4-bit, group 128
embedding         : 4-bit, group 128
MLP gate/up/down  : 3-bit, group 128
lm_head           : 3-bit, group 128
```

---

## 功能

- 统一 `ModelSpec` 抽象层，执行器不直接依赖 Qwen 字段；
- 已支持 Qwen2、Llama、Mistral、Gemma、Gemma 2 dense decoder；
- 支持标准 GQA/SwiGLU/GeGLU、RMSNorm、RoPE、sliding-window attention；
- Qwen2/Qwen2.5/Llama/Mistral/Gemma HF config、safetensors 分片、tied embedding；
- PEFT LoRA 合并；
- W4A16 GEMV（symmetric int4，group 128）；
- W3A16 GEMV（8 个 signed 3-bit 压进 3 bytes）；
- GQA decode kernel（支持 16/2、28/4 等 Q/KV head 组合）；
- 静态 dense KV cache（固定 slot、外部 mask）与 CUDA Graph replay；
- paged/elastic KV cache：按 token 分块分配、引用计数、OOM 隔离；
- 完整 prompt block 的 prefix cache（LRU、跨请求共享）；
- 可变 batch 连续批处理调度器（1..max_num_seqs，请求动态加入/退出）；
- chunked prefill + 双 CUDA stream prefill/decode overlap；
- native paged attention decode（FP16/INT8/FP8）、T>1 paged prefill、
  tensor-only validity mask、sliding-window mask，异常 shape 回退 SDPA；
- tiled W4A16 / W3A16 prefill GEMM（32×64 CTA tile，CUDA core，FP32 累加）；
- 启动时 KV block 显存预算、请求级 block 预留、admission/runtime OOM 隔离；
- 可选 `--kv-cache-dtype int8` 的 paged KV 存储（FP16 scale sidecar）；
- 可选 `--kv-cache-dtype fp8` / `fp8_e5m2` 的 paged KV 存储与 native decode；
- 可选 `--cpu-offload`：GPU block 不足时把 cold request 的整段 used KV
  swap 到 CPU，并在之后恢复；
- TTFT / TPOT / prefill step / decode batch / KV OOM / stream overlap /
  CPU swap Prometheus 指标；
- prefix cache 按 tenant hash 隔离（`user` / `X-Tenant-ID`）；
- OpenAI `/v1/chat/completions`、`/v1/completions`、SSE、stop、流式取消；
- `scripts/probe_model.py` 模型目录支持探测；
- 构造期 fused qkv / fused gate-up，避免加载后替换造成显存碎片；
- 7B 流式量化、按层 safetensors checkpoint；
- 3-bit lm_head；
- 3B/7B 测速 harness 和 CPU 复核工具；
- GPTQ / AWQ / GGUF 到原生 W4A16 的离线转换导入器；
- 本地 OpenAI 兼容网关（`/v1/models`、`/v1/chat/completions`、SSE 流式）；
- 固定槽位静态调度器（W4/W3 batch GEMV + batch CUDA Graph，低延迟回退）；
- 分页连续批处理调度器（前缀复用、block 回收、动态 batch、CPU swap）；
- 桌面剪贴板快捷操作脚本。

---

## 能力矩阵（当前真实状态）

| 能力 | 状态 | 说明 |
|---|---|---|
| Qwen2 / Qwen2.5 | ✅ | dense/GQA、W4A16、mixed W4/W3、prefix/paged batch |
| Llama | ✅ | 标准 GQA/RoPE；RoPE scaling 未实现 |
| Mistral | ✅ | sliding-window attention 已进入通用 mask |
| Gemma | ✅ | GeGLU、offset RMSNorm、embedding scale |
| Gemma 2 | ✅ | sandwich norm、softcap、alternating SWA |
| paged / elastic KV | ✅ | block table、引用计数、OOM 隔离 |
| prefix cache | ✅ | 完整 prompt block LRU 共享 |
| continuous batching | ✅ | 可变 batch decode + chunked prefill 交替 |
| prefill/decode stream overlap | ✅ | 双 CUDA stream launch；4 GB 带宽受限，收益按场景测量 |
| tiled W4/W3 prefill GEMM | ✅ | 32×64 CTA tile，shared-memory 权重量化 tile |
| KV memory budget | ✅ | 启动预算、block 预留、admission/runtime OOM 隔离 |
| OpenAI `/v1/completions` | ✅ | prompt、stream、stop、cancel、tenant |
| TTFT / TPOT metrics | ✅ | mean/p50/p95 benchmark，Prometheus summary |
| model directory probe | ✅ | supported / runtime_ready / 明确拒绝原因 |
| tenant prefix isolation | ✅ | `user` / `X-Tenant-ID` 命名空间 |
| native paged GQA kernel | ✅ | block-table FP16/INT8/FP8 decode，SDPA fallback |
| batched W4 GEMV | ✅ | batch tile 8，权重只读一次 |
| W3 batch-tile GEMV 实验 | ⚠️ | 数值正确，但 occupancy 下降，当前不作为默认路径 |
| KV INT8 paged cache | ✅ | native paged kernel，带 FP16 K/V scales |
| KV FP8 paged cache | ✅ | E4M3/E5M2，native paged decode，无 scale sidecar |
| CPU cold-block offload | ✅ | 请求级 used-block swap，pinned CPU payload |
| Prometheus / health | ✅ | block/prefix/queue/prefill/decode/swap 指标 |
| DeepSeek-V2-Lite MoE | ✅ | Q4_K_M routed-expert 流式执行、fused GEMV/GEMM、CPU/GPU offload；paged scheduler batched decode |
| Fused MoE GEMV/GEMM | ✅ | Q4_K/Q5_0/Q8_0 直接消费 packed rows；`--no-fused-moe` 回退 |
| DeepSeek-V2 dense / hybrid layers | ✅ | `first_k_dense_replace` / `moe_layer_freq` / dense-only；dense MLP 自动 GPU/CPU |
| DeepSeek-V2 MLA | ✅ | DynamicCache 单请求 + blockwise paged attention；不是 FlashMLA |
| GPTQ | ✅ | 2/4/8-bit、group quant、`g_idx`/desc_act -> W4A16 |
| AWQ | ✅ | AutoAWQ GEMM 4-bit -> W4A16 |
| GGUF | ✅ | 单文件 + Q4_0/Q4_K 等类型 -> W4A16 |
| GGUF 直接流式执行 | ⚠️ | DeepSeek-V2-Lite Q4_K_M 已直跑；通用 dense 路径仍需转换 |
| activation quant / SmoothQuant | ❌ | 仅权重量化 W4A16/W3A16 |
| tensor / pipeline parallel | ❌ | 单 GPU |
| 跨厂商后端 | ❌ | 当前 kernel 绑定 NVIDIA SM86 |

> “✅”只表示当前仓库有可运行的 dense/reference 实现或测试覆盖，不代表达到
> vLLM/TensorRT-LLM 的生产完备度。

---

## 目录

```text
deepinfer/
  hugging_face/qwen2_config.py          Qwen2 HF config -> DeepInfer AttrDict
  hugging_face/config.py                配置分派修改
  model_config_loader.py                允许纯文本 Qwen2/Llama/Mistral/Gemma
  model_spec.py                         ModelSpec 与架构 adapter
  quantization/formats.py                GPTQ/AWQ/GGUF 格式检测
  quantization/dequant.py                GPTQ/AWQ int32 解包与反量化
  quantization/gguf_source.py            GGUF metadata/tensor 映射
  quantization/convert.py               任意支持格式 -> layered W4A16
  moe/gguf_index.py                     GGUF header/index、完整性与 expert slice
  moe/gpu_quant.py                      Q4_K/Q5_0/Q8_0 SM86 GPU dequant
  moe/fused_moe.py                      per-expert fused Q4_K/Q5_0/Q8_0 GEMV/GEMM
  moe/grouped_moe.py                    grouped multi-expert fused GEMV/GEMM
  moe/ptr_grouped_moe.py                pointer-array grouped GEMV/GEMM
  moe/paged_adapter.py                  MLA K/V -> generic paged KV pool adapter
  moe/paged_mla_kernel.py               fused FP16 paged MLA attention kernel
  moe/cpu_avx2.py                       AVX2 packed-dequant runtime loader
  moe/cpu_avx2_packed.cpp               Q4_K/Q5_0/Q8_0 CPU AVX2 dequant
  moe/cpu_avx2_gemv.cpp                 fused AVX2 packed gate/up/down GEMV
  ecs/controller.py                      ECS heat/LRU/tier placement
  ecs/transport.py                       multi-device packet/all-to-all
  ecs/ecs_core.cpp                       C++ control-plane policy core
  moe/layout.py                         dense / MoE / hybrid 层布局判定
  moe/deepseek_v2_runtime.py            DeepSeek-V2-Lite MoE/MLA 流式执行与 offload
  engine/kv_cache.py                    block allocator、paged FP16/INT8/FP8、CPU payload
  engine/paged_attention.py             native paged GQA decode（FP16/INT8/FP8）
  engine/memory_budget.py               启动显存预算与 KV block 规划
  engine/metrics.py                     TTFT/TPOT/OOM/queue Prometheus 指标
  model_probe.py                        HF/量化目录能力探测
  model_executor/models/qwen2.py        通用 decoder、GQA、paged/static cache
  model_executor/models/qwen2_w4a16.py  W4A16 + GQA CUDA 后端与通用分层加载
  model_executor/models/mistral*.py     Mistral sliding-window adapter
  model_executor/models/gemma*.py       Gemma/GeGLU/softcap adapter
  model_executor/models/qwen2_mixed_sm86.py
                                        4-bit/3-bit 混合模型、3-bit kernel
kernels/
  qwen2_w4a16_gemv.cu
  qwen2_w4a16_batched_gemv.cu
  qwen2_w4a16_tiled_prefill.cu
  qwen2_w3a16_gemv.cu
  qwen2_w3a16_tiled_prefill.cu
  qwen2_gqa_decode.cu
  qwen2_gqa_decode_256.cu
  qwen2_paged_gqa_decode.cu
  qwen2_paged_gqa_decode_int8.cu
  qwen2_paged_gqa_decode_fp8.cu
apps/
  server.py                             OpenAI 兼容网关、/metrics、健康检查
pypi/deepinfer-sm86/                    standalone PyPI source tree + dist artifacts
  runtime.py                            常驻模型 + dense cache/Graph
  paged_scheduler.py                    分页连续批处理、双 stream overlap、CPU swap
  scheduler.py                          旧固定槽位静态调度器
  ask.py                                CLI 客户端
desktop/
  quick_ask.ps1                         剪贴板总结/改写/翻译/待办
scripts/
  run_qwen2_w4a16_gpu_harness.py        3B 原生测速入口
  export_w4_checkpoint.py               通用模型 W4A16 checkpoint
  export_qwen3b_w4_checkpoint.py        导出 3B 分层 W4A16 checkpoint
  bench_productivity_backends.py        3B/7B/Ollama 顺序基准
  run_qwen7b_mixed_gpu_harness.py       3B/7B/Q4 通用测速入口
  quantize_qwen7b_mixed.py              7B 逐层量化
  finish_qwen7b_mixed_quantize.py       分块量化 embedding/head 并补 manifest
  quantize_qwen7b_head3.py              将 lm_head 降到 3-bit
  quantize_qwen7b_q4.py                 生成全 4-bit Q4 checkpoint
  bench_w4a16_sm86.py                   W4A16 微基准
  bench_w3a16_sm86.py                   W3A16 微基准
  export_sm86_kernels.py                导出 .cu 源文件
  bench_paged_scheduler.py              分页连续批处理并发 sweep
  bench_tpot_jitter.py                  长 prefill 插入时的流式 TPOT 抖动
  bench_sm86_tiled_prefill.py           tiled W4/W3 prefill GEMM 微基准
  bench_w4a16_batched_gemv.py           batch-tile W4 GEMV 微基准
  bench_w4_w3_batched_gemv.py           W4/W3 batch-tile vs grid.y 对比
  bench_cpu_offload.py                  强制 CPU KV swap 与恢复验证
  validate_sm86_kernels.py              native kernel 数值校验
  import_quantized_checkpoint.py        GPTQ/AWQ/GGUF -> W4A16 CLI
  probe_model.py                        模型目录支持状态探测
  smoke_deepseek_v2_lite_moe.py         DeepSeek-V2-Lite 原生 MoE 质量/吞吐入口
tests/
  quantization/test_dequant.py          GPTQ/AWQ 解包与反量化
  quantization/test_convert_hf.py       GPTQ/AWQ 端到端转换
  quantization/test_convert_gguf.py     GGUF 端到端与 Q4_K block
  engine/test_kv_cache.py               block/prefix/paged cache 正确性
  apps/test_paged_scheduler.py          chunked prefill / decode / OOM 测试
  apps/test_server_api.py               OpenAI chat/completions API 测试
  moe/test_server_adapter.py            DeepSeek server adapter 测试
  ecs/test_ecs.py                       ECS heat/LRU/tier/transport 测试
  engine/test_memory_budget.py          KV 显存预算测试
  models/test_model_probe.py            模型目录探测测试
  models/test_qwen2_source_integration.py
  models/test_model_spec_mistral.py
  models/test_model_spec_gemma.py
  models/test_qwen2_w4a16_gpu.py
  apps/test_paged_scheduler_gpu.py      双 CUDA stream overlap GPU 测试
  moe/test_gguf_index.py                GGUF 索引、截断检测和 expert slice
  moe/test_gpu_quant.py                 Q4_K/Q5_0/Q8_0 GPU 反量化对照
  moe/test_fused_moe.py                 per-expert fused GEMV/GEMM 对照
  moe/test_grouped_moe.py               grouped multi-expert kernel 对照
  moe/test_ptr_grouped_moe.py           pointer-array grouped kernel 对照
  moe/test_paged_adapter.py             MLA paged cache 形状/写入测试
  moe/test_cpu_avx2.py                  AVX2 packed dequant 对照
  moe/test_full_quality_suite.py        full-category suite 结构与答案校验
  moe/test_dense_support.py             dense/MoE 层布局与 CPU MLP fallback
docs/
  ARTICLE_1_DEEPINFER_SM86_KERNEL.md          上篇：内核与量化工程
  ARTICLE_2_DEEPINFER_SM86_PRODUCTIVITY.md    下篇：生产力、网关与基准
  ARTICLE_3_DEEPINFER_SM86_GENERIC.md          扩展篇：通用化与 Llama
  QWEN_DEEPINFER_4GB_ENGINEERING_ARTICLE.md   合并稿存档
  QWEN_SM86_W4A16_NATIVE.md
  QWEN_7B_MIXED_SM86.md
  QWEN_7B_Q4_SM86.md
  QWEN_7B_4GB_FEASIBILITY.md
  QWEN_NATIVE_MODEL_EXECUTOR.md
  GENERIC_MODEL_ADAPTER.md                     通用模型抽象层
  DYNAMIC_BATCHING_SCHEDULER.md                 固定槽位动态批处理
  ENGINE_CORE_PAGED_CONTINUOUS_BATCHING.md      分页 KV、连续批处理与架构扩展
  ARTICLE_4_ENGINE_CORE_POSTMORTEM.md           engine-core 改造复盘与弯路记录
  QUANTIZATION_FORMAT_COMPATIBILITY.md          量化格式兼容与转换说明
  ARTICLE_5_QUANTIZATION_FORMAT_COMPATIBILITY.md 量化格式改造复盘与弯路
  P0_CHUNKED_PREFILL_AND_MEMORY_BUDGET.md       P0 低延迟与显存预算说明
  ARTICLE_6_NATIVE_PAGED_KERNEL_AND_4GB_BENCH.md native paged / 4GB 实测
  ARTICLE_7_PREFILL_DECODE_OVERLAP_AND_TILED_PREFILL.md overlap 与 tiled prefill
  ARTICLE_8_KV_FP8_AND_CPU_OFFLOAD.md           FP8 KV 与 CPU swap
  ARTICLE_9_DEEPSEEK_V2_LITE_MOE_CPU_GPU_OFFLOAD.md
                                                DeepSeek-V2-Lite MoE 流式 offload 复盘
  PERSONAL_PRODUCTIVITY.md                     从网关到桌面快捷操作
  PRODUCTIVITY_BENCHMARK_3B_7B_OLLAMA.md       3B/7B/Ollama 完整基准
  results/                                     选定的可复现结果
docs/results/
  qwen2_w4a16_native_gqa_gpu.json
  qwen7b_mixed_head3_final.json
  qwen7b_q4_gpu.json
  qwen7b_mixed_sm86_summary.json
  qwen7b_q4_sm86_summary.json
  ...
```

---

## 快速开始

### 1. 前置条件

- NVIDIA RTX 3050 或其它 SM86 GPU；
- Windows / Linux；
- Python 3.10（本仓库 GPU harness 的已验证环境）；
- CUDA 12.1 对应的 PyTorch；
- CuPy 含 CUDA 头文件（`cupy-cuda12x[ctk]`）；
- 一个可用的 DeepInfer checkout。

安装：

```powershell
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1
pip install -r requirements-sm86.txt
```

本仓库是在 DeepInfer 基线 `8b16149` 上开发的。  
DeepInfer 主包本身可能需要 Python 3.12；本仓库的 GPU harness 已验证 Python 3.10 +
轻量 package stub 的方式。

### 2. 叠加到 DeepInfer

先备份 DeepInfer 工作区，然后运行：

```powershell
cd D:\DeepInfer-SM86
.\install-overlay.ps1 -DeepInferRoot D:\DeepInfer
```

脚本会把 `deepinfer/` 下的 overlay 文件复制到现有 DeepInfer checkout。  
也可以手工复制，但要保留备份。

> 该 overlay 会覆盖 `deepinfer/__init__.py`、`logger.py`、
> `model_config_loader.py`、`hugging_face/config.py`、
> `model_executor/models/__init__.py` 等兼容性文件。

### 3. 跑 3B

```powershell
cd D:\DeepInfer
F:\QwenQLoRA\venv\Scripts\python.exe `
  scripts\run_qwen2_w4a16_gpu_harness.py `
  --model F:\QwenQLoRA\models\Qwen2.5-3B-Instruct-smallshards `
  --adapter F:\QwenQLoRA\train\maibot-catgirl-v3\checkpoint-120 `
  --max-new-tokens 24 --max-cache 128 `
  --output D:\DeepInfer\doc\results\qwen2_w4a16_native_gqa_gpu.json
```

3B 会先在 CPU 合并 dense 权重，再转换到 W4A16；CPU 加载较慢，正式测速请用
已量化 checkpoint，或接受一次性的转换时间。

### 4. 跑 7B 混合精度

```powershell
cd D:\DeepInfer
F:\QwenQLoRA\venv\Scripts\python.exe `
  scripts\run_qwen7b_mixed_gpu_harness.py `
  --checkpoint D:\Qwen7B-Mixed-SM86 `
  --max-new-tokens 24 --warmup-replays 3 --max-cache 64 `
  --output D:\DeepInfer\doc\results\qwen7b_mixed_head3_final.json
```

### 5. 跑整体 Q4

```powershell
F:\QwenQLoRA\venv\Scripts\python.exe `
  scripts\run_qwen7b_mixed_gpu_harness.py `
  --checkpoint D:\Qwen7B-Q4-SM86 `
  --max-new-tokens 4 --warmup-replays 1 --max-cache 64 `
  --output D:\DeepInfer\doc\results\qwen7b_q4_gpu.json
```

4 GB 上 Q4 会持续 paging；不要把这个结果当作生产吞吐。

### 6. 启动个人生产力网关（3B / 7B）

3B W4A16 网关：

```powershell
cd D:\DeepInfer-SM86
python apps\server.py `
  --checkpoint D:\Qwen3B-W4A16-SM86 `
  --backend w4a16 `
  --model-id deepinfer-sm86-3b-w4a16 `
  --max-cache 256 `
  --port 18001
```

3B 分页连续批处理（推荐）：

```powershell
python apps\server.py `
  --checkpoint D:\Qwen3B-W4A16-SM86 `
  --backend w4a16 `
  --max-cache 256 `
  --scheduler --scheduler-backend paged `
  --max-num-seqs 4 --block-size 16 --prefix-cache-blocks 16 `
  --prefill-chunk-tokens 64 --kv-cache-dtype fp16 `
  --gpu-memory-utilization 0.80 --reserve-memory-mb 512 `
  --port 18001
```

启用 FP8 KV 和 CPU cold-block swap：

```powershell
python apps\server.py `
  --checkpoint D:\Qwen3B-W4A16-SM86 `
  --backend w4a16 --max-cache 256 `
  --scheduler --scheduler-backend paged --max-num-seqs 4 `
  --kv-cache-dtype fp8 --cpu-offload --max-cpu-offload-mb 1024 `
  --port 18001
```

旧固定槽位路径仍可用于低延迟回退：

```powershell
python apps\server.py `
  --checkpoint D:\Qwen3B-W4A16-SM86 `
  --backend w4a16 `
  --max-cache 256 `
  --scheduler --scheduler-backend static --max-batch-size 2 `
  --port 18001
```

7B 混合网关：

```powershell
cd D:\DeepInfer-SM86
python apps\server.py `
  --checkpoint D:\Qwen7B-Mixed-SM86 `
  --backend mixed `
  --model-id deepinfer-sm86-7b-mixed `
  --max-cache 256 `
  --port 18000
```

DeepSeek-V2-Lite OpenAI 网关（推荐 paged scheduler）：

```powershell
cd D:\DeepInfer-SM86
python apps\server.py `
  --backend deepseek `
  --model-id deepseek-v2-lite-q4km `
  --scheduler --scheduler-backend paged `
  --max-cache 256 --max-num-seqs 2 `
  --block-size 16 --prefill-chunk-tokens 8 `
  --kv-blocks 20 --prefix-cache-blocks 4 `
  --port 18002
```

- DeepSeek 的 `/v1/chat/completions`、`/v1/completions`、`/v1/models`、
  `/health` 和 `/metrics` 都已接入；
- `--scheduler` 路径提供 true token SSE、取消和 prefill/decode overlap；
- 不带 scheduler 时仍可用，但 SSE 是生成完成后按文本切片发送；
- raw `/v1/completions` 不走 chat template，直接使用原始 prompt。

接口（3B 使用 18001，7B 使用 18000，DeepSeek 使用 18002）：

```text
GET  http://127.0.0.1:18000/health
GET  http://127.0.0.1:18000/metrics
GET  http://127.0.0.1:18000/v1/models
POST http://127.0.0.1:18000/v1/chat/completions
POST http://127.0.0.1:18000/v1/completions
```

OpenAI 客户端把 Base URL 设为：

```text
http://127.0.0.1:18000/v1
```

命令行：

```powershell
python apps\ask.py --server http://127.0.0.1:18000 "用一句话解释 CUDA Graph。"
```

流式请求：

```powershell
curl.exe -N http://127.0.0.1:18000/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d "{\"model\":\"deepinfer-sm86-7b-mixed\",\"messages\":[{\"role\":\"user\",\"content\":\"写三句话解释 CUDA Graph\"}],\"max_tokens\":64,\"stream\":true}"
```

Completion API（`/v1/completions`）：

```powershell
curl.exe http://127.0.0.1:18000/v1/completions `
  -H "Content-Type: application/json" `
  -d "{\"model\":\"deepinfer-sm86-7b-mixed\",\"prompt\":\"用一句话解释 PagedAttention\",\"max_tokens\":32,\"stop\":[\"\n\n\"]}"
```

按租户隔离 prefix cache：

```powershell
curl.exe http://127.0.0.1:18000/v1/chat/completions `
  -H "Content-Type: application/json" `
  -H "X-Tenant-ID: user-123" `
  -d "{\"model\":\"deepinfer-sm86-7b-mixed\",\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}]}"
```

模型目录能力探测：

```powershell
python scripts\probe_model.py D:\Qwen3B-W4A16-SM86
```

桌面剪贴板动作：

```powershell
.\desktop\quick_ask.ps1 -Task 总结
.\desktop\quick_ask.ps1 -Task 翻译成英文
.\desktop\quick_ask.ps1 -Task 提取待办
```

结果会写回剪贴板。建议用 PowerToys Keyboard Manager 绑定
`Win+Alt+Q` 等快捷键，详见 `desktop/README.md`。

当前网关的已验证参数：

| 指标 | 值 |
|---|---:|
| context 上限 | 256 tokens（prompt + output） |
| 3B W4A16 decode | 约 44.8 tokens/s |
| 3B 一次请求 wall | 约 0.90 秒 |
| 3B prefill | 约 0.46 秒 |
| 7B 混合 decode | 约 25.8 tokens/s（隔离环境） |
| 7B prefill | 3.06 秒（隔离）；53 秒以上（paging） |
| 无 scheduler | 单请求串行 |
| `--scheduler-backend static` | 固定槽位 2-way 连续批处理（旧路径） |
| `--scheduler-backend paged` | 可变 batch、chunked prefill、分页 KV、prefix cache |
| `--prefill-chunk-tokens` | 控制长 prompt 对 decode TPOT 的干扰 |
| `--kv-cache-dtype` | `fp16` / `int8` / `fp8` / `fp8_e5m2` |
| `--cpu-offload` | GPU block 不足时整请求 KV swap 到 pinned CPU |
| `--max-cpu-offload-mb` | CPU payload 上限，默认 2048 MB |
| `/v1/completions` | prompt / stream / stop / cancel / tenant |

默认 256 context 适合总结选中段落、改写、翻译、提取待办。分页调度器允许
`--max-cache > 256`，但 4 GB 显卡仍受物理显存、prefill 激活和模型权重约束；
长文档还需要配合 map-reduce / RAG。

---

## 3B W4A16 checkpoint

3B 每次启动都重新做 dense→W4A16 会花约 5 分钟，因此先导出分层 checkpoint：

```powershell
python scripts\export_qwen3b_w4_checkpoint.py `
  --model F:\QwenQLoRA\models\Qwen2.5-3B-Instruct-smallshards `
  --adapter F:\QwenQLoRA\train\maibot-catgirl-v3\checkpoint-120 `
  --output D:\Qwen3B-W4A16-SM86
```

产物约 1.912 GiB，加载约 11.75 秒。

---

## 社区量化格式导入

新增统一的离线转换器：

```text
GPTQ / AWQ / GGUF
        →  dense/dequantized layer weights
        →  SM86 symmetric int4 W4A16
```

安装依赖：

```powershell
pip install -r requirements-sm86.txt
```

GPTQ：

```powershell
python scripts\import_quantized_checkpoint.py `
  --source F:\models\Qwen2.5-3B-Instruct-GPTQ-Int4 `
  --output D:\Qwen3B-GPTQ-to-W4A16-SM86 `
  --format auto `
  --group-size 128
```

AWQ：

```powershell
python scripts\import_quantized_checkpoint.py `
  --source F:\models\Qwen2.5-3B-Instruct-AWQ `
  --output D:\Qwen3B-AWQ-to-W4A16-SM86 `
  --format awq `
  --group-size 128
```

GGUF：

```powershell
python scripts\import_quantized_checkpoint.py `
  --source F:\models\qwen2.5-3b-instruct-q4_k_m.gguf `
  --tokenizer F:\models\Qwen2.5-3B-Instruct `
  --output D:\Qwen3B-GGUF-to-W4A16-SM86 `
  --format gguf `
  --group-size 128
```

支持：

- GPTQ 2/4/8-bit、group quant、`qzeros`、`g_idx`/desc_act；
- AWQ AutoAWQ GEMM 4-bit 布局；
- GGUF 单文件，以及 `gguf` 包支持反量化的 Q4_0/Q4_K 等 ggml 类型；
- fused `attn_qkv` / `ffn_gate_up` 在 GGUF 导入时自动拆分。

限制：

- 这是**转换兼容**，不是社区格式零拷贝直跑；
- `lm_head` 仍写 FP16 dense；非 tied 的 7B 模型要额外评估显存/磁盘；
- AWQ GEMV/ExLlama 布局、GGUF split shards、MoE、量化 embedding/norm
  尚未支持；
- 完整说明见 `docs/QUANTIZATION_FORMAT_COMPATIBILITY.md`。

---

## 7B 量化流程

原始 7B BF16 checkpoint 不包含在本仓库中。

### 混合 Q4/Q3

```powershell
python scripts\quantize_qwen7b_mixed.py `
  --source F:\QwenQLoRA\models\Qwen2.5-7B-Instruct `
  --output D:\Qwen7B-Mixed-SM86
```

在低内存机器上，主脚本完成 28 层后可能在 embedding/lm_head 阶段 OOM；随后执行：

```powershell
python scripts\finish_qwen7b_mixed_quantize.py `
  --source F:\QwenQLoRA\models\Qwen2.5-7B-Instruct `
  --output D:\Qwen7B-Mixed-SM86
```

再把 lm_head 降到 3-bit：

```powershell
python scripts\quantize_qwen7b_head3.py `
  --source F:\QwenQLoRA\models\Qwen2.5-7B-Instruct `
  --checkpoint D:\Qwen7B-Mixed-SM86
```

### 整体 Q4

```powershell
python scripts\quantize_qwen7b_q4.py `
  --source F:\QwenQLoRA\models\Qwen2.5-7B-Instruct `
  --mixed D:\Qwen7B-Mixed-SM86 `
  --output D:\Qwen7B-Q4-SM86
```

---

## 内核说明

### W4A16

- 2 个 signed int4 压入 1 byte；
- group 128；
- 每层一个 warp 处理一个输出 row；
- x 放 shared memory；
- GEMV 支持批量 token（decode batch 可大于 1）；
- 固定 dense batch 路径可配合 CUDA Graph。

### W3A16

- 8 个 signed 3-bit 值压入 3 bytes；
- group 128；
- gate/up 使用 512 threads / 16 rows per block；
- down 使用 768 threads / 24 rows per block；
- 3-bit kernel 与显式反量化 matvec 的 cosine 为 1.0。

### GQA decode

- 一个 block 处理一个 query head；
- dense static-cache 路径支持最多 256 个 cache slot；
- 支持 16 Q / 2 KV 和 28 Q / 4 KV；
- 与 PyTorch SDPA 对照 cosine 约 1.0；
- 只用于单 token decode；prefill 仍走 PyTorch。

CUDA 源文件由：

```powershell
python scripts\export_sm86_kernels.py
```

从 Python 模块中的 NVRTC 字符串导出。

---

## 已知限制

- 这是 **model_executor + engine-core 原型**，不是完整 DeepInfer engine；
- 没有接入官方 `worker/cache_engine.py` 和原生 Sampler；
- native paged GQA 只覆盖 decode `T=1`、FP16 query/output、`D<=128`、
  `max_length<=1024`；prefill 和 external mask 仍走 SDPA；
- 可变 batch 下没有 CUDA Graph；dense static-cache 路径仍支持 Graph；
- prefill 与 decode 已可在同一 scheduler cycle 内用两条 CUDA stream 并发，但仍
  不是单次 forward 内的 mixed batch；4 GB 卡上二者争用内存带宽，overlap 的吞吐
  收益有限；
- 小 chunk 能压低 ITL 抖动，但会抬高插入期间的 TPOT；大 chunk 相反；
- 每个 scheduling iteration 只推进一个 prefill chunk；
- 没有 beam search 或 speculative decoding；
- CPU offload 是整请求 swap，不是按 block 热度的细粒度冷热迁移；swap 期间
  被换出请求的 TPOT 会包含整段等待时间；
- CPU payload 使用 pinned memory，预算由 `--max-cpu-offload-mb` 显式限制，
  不会自动按系统 RAM 调整；
- INT8 KV 已有 native decode kernel，但 dequant 有成本；3B c4 比 FP16 native
  慢约 6%，主要价值是 KV 容量减半；
- FP8 KV 直接 cast，无 per-token scale；E4M3/E5M2 的误差和动态范围不同，
  不能把两者视为同一个精度档；
- W4/W3 prefill 已切到 tiled GEMM；但当前是 CUDA core + FP32 累加，尚未做 Tensor Core / cp.async pipeline；
- GPU harness 在 Python 3.10 下验证；Python 3.12 GPU package 验证未完成；
- 3-bit MLP/lm_head 是有损量化，短输出可读不代表与 BF16 等价；
- 4 GB 上整体 Q4 只能作为质量/兼容实验，速度约 8 tokens/s；
- GPTQ / AWQ 走**离线转换**为 W4A16；DeepSeek-V2-Lite Q4_K_M GGUF 可直接流式执行；
- AWQ 仅支持 GEMM 布局；GGUF 暂不支持 split shards；
- DeepSeek-V2-Lite MoE 已接入 paged scheduler 的 chunked prefill / batched
  decode / block KV 预算和 prefill/decode stream overlap；CuPy kernels 跟随
  torch stream，短 smoke 已验证 token parity，但 4 GB 带宽下 overlap 收益
  仍按场景测量；
- CPU MoE layer offload 已实现，1 层整层 offload 约 1.86 t/s；
- cold-expert split 已实现；推荐 2 层 × 1 cold expert，约 2.50 t/s，
  CPU 路径已有 fused AVX2 packed gate/up/down，oneDNN 作为 fallback；
- prefill copy-stream pipeline 已实现但默认关闭；本机 WDDM 上比默认路径慢约
  6%；prefill/decode background prefetch 默认开启，prefill 3.43→2.03 s、
  decode 2.52→2.70 t/s；预测式 next-layer prefetch 已实现但默认关闭，
  路由重叠不足导致 2.70→2.21 t/s；CUDA Graph 仍因动态 MoE 路由不可安全 capture；
- dense/hybrid 层布局已实现并有单元测试，但本机没有 dense DeepSeek-V2 GGUF
  做端到端质量基准；dense-only 模型超出显存时会回退 CPU FP32，速度会很慢；
- DeepSeek MLA paged 模式通过 `DeepSeekPagedCache`/paged forward 直接按
  block 读取 K/V 并 online softmax；FP16 使用 fused scalar-register CUDA
  kernel，INT8/FP8 E4M3/E5M2 使用 native quant paged MLA，不再回退 dense
  materialize；但整体仍不是 FlashMLA 级优化；
- MoE 默认为 grouped fused GEMV/GEMM，`--no-fused-moe` 可回退到 dequant +
  grouped GEMM；block-per-row 通用实现尚未做 Tensor Core tile；
- 非 tied 的 lm_head 仍以 FP16 dense 保存，7B 转换后体积需要单独评估；
- dense 通用执行层仍未实现 activation quantization、SmoothQuant 或 MoE；
- 没有 tensor/pipeline parallel 或跨厂商后端抽象；
- 默认 context 上限仍为 256 token；paged scheduler 可提高 `--max-cache`，
  但 4 GB 物理显存、长 prompt prefill 和 chunk size 仍是硬约束；
- tenant 前缀隔离是 hash namespace 隔离，不是进程级/加密级隐私隔离；
- 3B 与 7B 网关不能同时常驻 4 GB；DeepSeek-V2-Lite MoE 使用同一张卡时也会
  与常驻 dense 网关互斥；
- DeepSeek-V2-Lite 原生 MoE 已用双 pinned slot；但当前仍是“读下一个 chunk”
  而不是跨层预测 prefetch；
- CPU packed expert cache 已实现但默认关闭：512 MB 实测约 11% hit，CPU cache
  copy 比 NVMe 读更慢；
- GPU packed expert cache + pointer-array kernel 已实现，但 1 GiB cache 只换来约
  2.3% decode 提升，且 peak 升到 3.20 GiB；当前 4 GB 卡上默认关闭；
- 桌面快捷操作依赖 Windows PowerShell 剪贴板和本地 HTTP 网关。

---

## 结果与文档

- 上篇（内核/量化）：`docs/ARTICLE_1_DEEPINFER_SM86_KERNEL.md`
- 下篇（生产力/基准）：`docs/ARTICLE_2_DEEPINFER_SM86_PRODUCTIVITY.md`
- 扩展篇（通用化/Llama）：`docs/ARTICLE_3_DEEPINFER_SM86_GENERIC.md`
- 合并稿存档：`docs/QWEN_DEEPINFER_4GB_ENGINEERING_ARTICLE.md`
- 3B：`docs/QWEN_SM86_W4A16_NATIVE.md`
- 7B 混合：`docs/QWEN_7B_MIXED_SM86.md`
- 7B Q4：`docs/QWEN_7B_Q4_SM86.md`
- 7B 可行性：`docs/QWEN_7B_4GB_FEASIBILITY.md`
- 个人生产力落地：`docs/PERSONAL_PRODUCTIVITY.md`
- 3B/7B/Ollama 基准：`docs/PRODUCTIVITY_BENCHMARK_3B_7B_OLLAMA.md`
- 通用模型抽象层：`docs/GENERIC_MODEL_ADAPTER.md`
- 固定槽位动态批处理：`docs/DYNAMIC_BATCHING_SCHEDULER.md`
- 分页 KV / 连续批处理 / 架构扩展：
  `docs/ENGINE_CORE_PAGED_CONTINUOUS_BATCHING.md`
- 工程复盘与弯路记录：`docs/ARTICLE_4_ENGINE_CORE_POSTMORTEM.md`
- 量化格式兼容与转换：`docs/QUANTIZATION_FORMAT_COMPATIBILITY.md`
- 量化格式改造复盘：`docs/ARTICLE_5_QUANTIZATION_FORMAT_COMPATIBILITY.md`
- native paged kernel / batched GEMV / 4GB 实测：`docs/ARTICLE_6_NATIVE_PAGED_KERNEL_AND_4GB_BENCH.md`
- prefill/decode overlap 与 tiled prefill 复盘：
  `docs/ARTICLE_7_PREFILL_DECODE_OVERLAP_AND_TILED_PREFILL.md`
- KV FP8 与 CPU cold-block offload 复盘：
  `docs/ARTICLE_8_KV_FP8_AND_CPU_OFFLOAD.md`
- tiled W4/W3 prefill 微基准：`scripts/bench_sm86_tiled_prefill.py`
- P0 低延迟/显存预算说明：
  `docs/P0_CHUNKED_PREFILL_AND_MEMORY_BUDGET.md`
- DeepSeek-V2-Lite MoE 复盘：
  `docs/ARTICLE_9_DEEPSEEK_V2_LITE_MOE_CPU_GPU_OFFLOAD.md`
- fused MoE vs dequant+bmm 长生成对照：
  `docs/results/deepseek_v2_lite_q4km_moe_fused_vs_dequant.json`
- grouped multi-expert vs per-expert fused 长生成对照：
  `docs/results/deepseek_v2_lite_q4km_moe_grouped_vs_per_expert.json`
- double-buffer 长生成：
  `docs/results/deepseek_v2_lite_q4km_moe_double_buffer_long.json`
- CPU packed expert cache 负结果：
  `docs/results/deepseek_v2_lite_q4km_moe_expert_cache_eval.json`
- GPU packed expert cache + pointer-array kernel：
  `docs/results/deepseek_v2_lite_q4km_moe_gpu_cache_eval.json`
- GPU TTFT/TPOT/并发 benchmarks：`scripts/bench_paged_scheduler.py`
- 长文档插入 TPOT 抖动：`scripts/bench_tpot_jitter.py`
- batched W4 GEMV 微基准：`scripts/bench_w4a16_batched_gemv.py`
- CPU KV offload 压测：`scripts/bench_cpu_offload.py`
- native kernel 数值校验：`scripts/validate_sm86_kernels.py`
- 模型目录探测：`scripts/probe_model.py`
- 结构化结果：`docs/results/`

---

## 许可证与来源

本项目基于 DeepInfer 的 Apache-2.0 代码与设计进行非官方扩展。

- 原始 DeepInfer：<https://gitee.com/mirrors/DeepInfer>
- 本仓库不是 DeepInfer 官方仓库；
- 原始代码和衍生代码遵循 Apache License 2.0；
- 详见 `LICENSE` 与 `NOTICE`。

如果你使用本仓库的结果，请同时说明：

1. 这是 **SM86 model_executor 级** 实现；
2. 不是完整 DeepInfer engine；
3. 4 GB 上整体 Q4 与几十 tokens/s 不能同时满足。
