# 第九篇：DeepSeek-V2-Lite MoE 在 4 GB SM86 上的极致的 CPU/GPU 混合 Offload

> 编号说明：仓库中的 `ARTICLE_8` 已经用于 KV FP8 / CPU cold-block offload；
> 这篇 MoE 专题按现有文章序列编号为 `ARTICLE_9`，内容就是这次 DeepSeek-V2-Lite
> 原生 MoE 的完整实现与踩坑复盘。

> 日期：2026-10-28  
> 平台：Windows、RTX 3050 Laptop 4 GB（SM86）、AMD Ryzen 5 5600H、16 GB RAM  
> 模型：DeepSeek-V2-Lite-Chat Q4_K_M GGUF（10,364,416,320 bytes）  
> 运行时：Python 3.10.3、torch 2.5.1+cu121、CUDA 12.1、CuPy 14.2  
> 目标：让 DeepInfer 自己的 MoE/MLA 执行层在 4 GB 卡上不 OOM 地跑起来；速度可以牺牲，但质量不能明显下降。

## 0. 验收结果

本次不是“把 DeepSeek-V2-Lite 塞进 4 GB”，而是让它在 4 GB 卡上以流式方式完成
DeepSeek-V2-Lite 的 MoE 路由、专家计算、MLA attention 和自回归 decode。

最终结果：

| 指标 | 实测 |
|---|---:|
| 模型加载 | 17.6 s |
| 冷启动 warmup（4 token） | 3.6–16 s，取决于 OS page cache |
| prefill（短 prompt） | 1.95–2.44 s |
| decode 稳态 | **2.64–3.01 tokens/s**（64-token 长生成 2.63） |
| peak GPU allocated | **2.23 GiB** |
| peak GPU reserved | **2.31 GiB** |
| 对照 grouped 无 double buffer | decode 2.53–2.95；prefill 2.20–2.88 |
| 可选 GPU expert cache 1 GiB | decode 2.69；prefill 2.93；peak 3.20 GiB |
| mini quality gate | MMLU-style 8/10；GSM8K-style 4/8；长链 3/5 |
| full-category quality | MMLU 27/57（47.4%）；GSM8K 9/32（28.1%）；长链 2/32；PPL 6.59 |
| OpenAI server | chat/completions、raw completions、SSE、health/models、paged scheduler |
| INT8/FP8 paged MLA | native quant kernel；KV 106→约 53 MiB；INT8/FP8/E5M2 单测通过 |
| ECS control plane | C++ heat/LRU/tier；opt-in；模拟 2 卡 all-to-all |
| paged scheduler | 2 请求 batched decode；native paged MLA；prefix 16 token |
| CPU layer offload | 最后 1 个 MoE 层 CPU FP32；decode 1.86 t/s；token IDs 一致 |
| hot/cold expert split | 推荐 2 层 × 1 cold expert；decode 2.50 t/s；token IDs 一致 |
| prefill pipeline | 可选；本机实测 prefill 慢约 6%，默认关闭 |
| prefill background prefetch | 默认开启；prefill 3.43→2.03 s，token IDs 一致 |
| decode background prefetch | 默认开启；decode 2.52→2.70 t/s，token IDs 一致 |
| predictive next-layer prefetch | 可选；实测 decode 2.70→2.21 t/s，默认关闭 |
| 对照 per-expert fused | decode 2.56–3.05；64-token 2.63 |
| 对照 dequant+bmm | decode 2.43–2.77；64-token 2.43 |
| GPU 加载后空闲 | 约 1.0 GiB |
| shared experts / lm_head | GPU（auto 放置） |
| 运行结果 | 5 个质量抽检全部可读、无明显退化 |
| 可复现性 | 修复 pinned-buffer race 后，同进程 greedy 多次输出 token IDs 完全一致 |

质量抽检（greedy、max_new_tokens=24 时截断；最终结果 JSON 见
`docs/results/deepseek_v2_lite_q4km_moe_sm86.json`）：

| prompt | 生成 |
|---|---|
| 用一句话解释什么是混合专家模型（MoE）。 | 混合专家模型（MoE）是一种人工智能模型，它通过结合多个专家系统来提高整体性能，每个… |
| 中国的首都是哪里？请只回答城市名。 | 中国的首都是北京。 |
| 把这句话翻译成英文：今天天气很好。 | The weather is beautiful today. |
| 只回答数字：5-2=? | 5-2=3 |
| 9.11和9.9哪个大？只回答更大的数字。 | 9.9大。 |

同一份 Q4_K_M 用 llama.cpp CPU 基线跑，decode 约 9–13 tokens/s。因此本实现不是
最快的推理路径；但它把 **DeepInfer 的执行边界** 从 dense Qwen 推到了
DeepSeek-V2-Lite 的 MoE + MLA，而且是在 4 GB 卡上、不 OOM、质量可用地跑完。
速度是后续可以继续榨的，模型执行层是这次必须补上的。

---

## 1. 失败复盘：之前为什么 OOM / 为什么走弯路

### 1.1 半截 GGUF 与“先跑起来再说”的代价

上一轮最先下载的是 Q2_K，原始 metadata 标记的大小是 6,430,464,320 bytes，但
下载被中断后本地文件只有 2,824,452,838 bytes。`GGUFReader` 在加载时才报：

```text
ValueError: cannot reshape array of size 26414784 into shape (64,1408,1152)
```

这不是模型太大，而是 **把未完成的文件当成了可加载 checkpoint**。更糟的是，
下载任务和工具进程混在一起，误杀了 WindowsTerminal / job runner，导致：

- 下载没有继续；
- 工具会话不稳定；
- 既没有确认文件完整性，也没有确认内存上限。

教训：

1. 加载任何 GGUF 前先解析 header、计算 `data_start + max(tensor.offset + nbytes)`，
   与文件实际大小比较；不完整就直接拒绝，不要先 mmap 或反量化。
2. 下载、量化、加载、推理使用不同的进程边界，不能让下载进程和工具主进程
   互相误杀。
3. 不要用“先跑一下看看”的方式绕开内存预算；4 GB 卡上任何“先全部加载”的
   尝试都会把问题推到更晚、更难定位的 OOM。

### 1.2 BF16/F16 原始模型根本不在这台机器的资源范围内

DeepSeek-V2-Lite 原始 BF16/F16 GGUF 是 31.4 GB。机器只有 16 GB RAM，
当时磁盘可用空间又只有 10 GB 左右；`from_pretrained`、`AutoModel` 或
`GGUFReader` 全量加载都会失败。即使磁盘够，16 GB RAM 也放不下 31 GB 权重，
更不用说 4 GB GPU。

### 1.3 不能为了省内存选 Q2_K

Q2_K 能让文件更小，但对一个 15.7B/2.4B active 的 MoE 模型来说，2-bit 量化的
质量损失不可接受。用户的要求是“速度可以牺牲，但不要明显牺牲质量/智力”，
所以这次直接选 **Q4_K_M**：

- 文件 9.65 GiB；
- 用 E 盘 NVMe 存放；
- 不把完整文件读入匿名内存，只按 expert slice 读；
- 不把 64 个专家同时反量化到 GPU。

空间决策：E 盘当时约 21.8 GB 可用，Q4_K_M 为 9.65 GiB，直接放在
`E:\DeepInfer-MoE\models\...`，最终仍剩约 13 GB。本次**没有卸载三角洲**；
用户允许在空间不够时卸载，但实测 Q4_K_M + 结果文件并不需要这一步。之前
遗留的半截 Q2_K 文件没有参与本次执行。

### 1.4 第一次 Naive MoE：等于制造了一个 1 GB 级别的显存炸弹

最初的实现直接按 `expert_ids` 把本层所有被选中的专家一次性反量化。decode 时
每层只有 6 个专家还可以；但 prefill 一个 20–30 token 的 prompt 会产生
120–180 个 assignment，通常会覆盖 26 层中每层的全部 64 个专家。一次
prefill 每层就可能需要：

```text
64 experts × (gate + up + down) × 约 5.7 MiB ≈ 1.1 GiB
```

这还没算 padded tokens、KV、CUDA 工作区和 PyTorch allocator 的碎片。即使
decode 能跑，prefill 也会 OOM。

最终改成 **expert chunk**：每层最多同时 materialize 8 个专家；prefill 时按
chunk 循环。decode 的 6 个专家正好落在一个 chunk 内。

### 1.5 速度上的弯路：从 1.02 到 2.7 tokens/s

第一版 native 路径能生成正确内容，但只有约 **1.02 tokens/s**。逐步拆解后发现：

- Python 逐 expert 做 `nonzero()` 和 `index_add_`：大量 kernel launch 和同步；
- 文件读取用 mmap 的 4K page fault 模式，随机 expert slice 读取约 1 GB/s；
- 每个 expert 单独 H2D，pageable copy 阻塞 CPU；
- 每层 18 次小 bmm，GPU 利用率很低。

最终优化：

| 改动 | 效果 |
|---|---|
| 不再 `nonzero()`，用 `argsort + bincount + index_copy_` 做 assignment 分组 | decode 0.98 → 0.81 s/token |
| 每个 expert slice 进线程池并行 `readinto` | expert read 0.389 → 约 0.14 s/token |
| 引入 pinned staging buffer | H2D 从 blocking、低带宽拷贝变成可异步 |
| 独立 copy stream 预先拷 down，与 gate/up 计算重叠 | 稳态再降到约 0.42 s/token |
| 修复暂停用的 pinned-buffer race | 输出从随机乱码变回可复现 |

### 1.6 最危险的弯路：pinned buffer 复用 race

为了把 H2D 变成异步，一开始每个 projection 只有一块 pinned staging buffer。
`load_experts()` 返回后，CPU 会很快进入下一个 expert chunk，而上一 chunk 的
H2D 仍可能没完成；于是新的 `readinto` 覆盖了正在被 GPU DMA 读取的源缓冲区。

症状非常具有欺骗性：

- 有时输出完全正常；
- 有时输出是乱码或纯标点；
- 有时同一个 prompt 第一次生成中文，第二次生成英文；
- 用 `CUDA_LAUNCH_BLOCKING=1` 时能看到：

```text
Indexing.cu:1308: block: [0,0,0], thread: [32,0,0]
Assertion `srcIndex < srcSelectDimSize` failed.
```

一开始怀疑是 `bincount`、`index_select` 或 bmm 的问题，后来逐层 hook 后看到：

- layer 0 输出完全相同；
- layer 1 第一次出现差异；
- 再对 layer 1 的 MoE 做重复调用，发现 `expert_store.load_experts()` 返回的
  gate/up/down 权重本身就不相同；
- 手工按 chunk 比较后，发现部分 chunk 的权重不同，部分相同。

根因不是数学，而是 **异步 copy 与 pinned 源 buffer 复用之间没有同步**。修复：

1. 每次使用 pinned buffer 前，`Event.synchronize()` 等上一次 H2D 完成；
2. 仍然保留独立 copy stream，让 down 的 H2D 尽量与 gate/up 计算重叠；
3. 在 debug 模式下用 `DEEPINFER_DISABLE_EXPERT_COPY_STREAM=1` 做串行基线。

修复后，同一进程连续 4 次 greedy 生成 20 token，token IDs 全部一致。

### 1.7 第二个确定性问题：`index_add_` 的原子累加

MoE 最后的 top-k 加权通常写成：

```python
out.index_add_(0, token_ids, contribution)
```

当同一个 token 同时命中多个专家时，`token_ids` 有重复索引；CUDA `index_add_`
可能使用 `atomicAdd`，累加顺序不固定，FP16 下会产生微小数值差异。对于 dense
模型这通常只是舍入差异；但对于 top-6 router，它会改变下一 token 的专家选择，
然后误差被自回归放大。

最终改法：每个 `(token, slot)` assignment 是唯一的，先把专家结果写到
`assignment_outputs[assignment_index]`，再在 FP32 下做：

```python
routed = assignment_outputs.view(tokens, top_k, hidden).float().sum(dim=1)
routed = routed.to(hidden_dtype)
```

这样不再依赖重复索引的原子累加，也避免把 top-k 求和留在 FP16。

---

## 2. 资源预算：为什么这个方案能放进 4 GB

DeepSeek-V2-Lite 的大致参数结构：

| 部分 | 参数量（约） | 说明 |
|---|---:|---|
| routed experts | 14.39 B | 26 个 MoE 层 × 64 experts × 3 个 2048×1408 矩阵 |
| shared experts | 0.45 B | 每层 2 个 shared expert 的等效宽度 |
| attention / MLA | 0.36 B | q/kv_a/kv_b/o |
| dense layer 0 MLP | 0.067 B | 第一层 dense |
| embedding + lm_head | 0.42 B | vocab 102400 × hidden 2048 |
| 总量 | 约 15.7 B | active 约 2.4 B/token |

Q4_K_M 不是单一格式，实际 tensor 类型混合：

- routed gate/up：Q4_K；
- routed down：Q8_0 或 Q5_0（不同层混用）；
- token embedding：Q4_K；
- output / lm_head：Q6_K；
- 部分 shared expert：Q6_K/Q4_K；
- attention：Q4_K；
- norm：F32。

本实现不把 9.65 GiB 文件整体读入匿名内存，而是：

1. `GGUFIndex` 只解析 header 和 tensor 目录；
2. 文件用只读 mmap 建立地址映射；
3. 每次只读取当前专家 chunk 的 packed bytes；
4. packed bytes 读取到 pinned CPU staging buffer；
5. H2D 后由 SM86 kernel 反量化；
6. 只保留当前 chunk 的反量化结果。

因此 GPU 驻留部分约为：

```text
attention (FP16)                  ≈ 0.72 GiB
dense layer0 + router + norms     ≈ 0.15 GiB
shared experts (FP16)             ≈ 0.90 GiB
lm_head (FP16)                    ≈ 0.40 GiB
expert chunk 工作集               ≈ 0.14 GiB
KV / attention 工作区             < 0.10 GiB
CUDA context / allocator 余量     ≈ 0.30 GiB
------------------------------------------------
约 2.4 GiB allocated / 2.43 GiB reserved
```

CPU 侧只保留：

- token embedding（400 MiB，FP16）；
- 一小块 pinned expert staging（gate/up/down 各一块）；
- Python/GGUF 映射与 worker thread 栈。

shared experts 和 lm_head 的放置是 `auto` 的：如果加载后 GPU 空闲不足，
它们会回退到 CPU（FP32）并用 `_cpu_mlp_forward` 计算。本次机器上都在 GPU。

---

## 3. 执行路径设计

### 3.1 总览

```text
GGUF file (10.36 GB, read-only mmap)
  │
  ├─ token_embd / attention / router / shared / lm_head
  │     → 加载时反量化一次 → 按预算放 GPU 或 CPU
  │
  └─ routed expert tensors
        → 每层根据 router top-k 选 expert
        → 每个 expert 的 packed bytes 分别 readinto 到 pinned buffer
        → copy stream H2D（gate/up 与 down 流水）
        → SM86 Q4_K/Q5_0/Q8_0 GPU dequant
        → grouped bmm × 3（gate/up/down）
        → 按 assignment 确定性 scatter
        → FP32 top-k 求和 + shared expert
```

### 3.2 GGUFIndex：先验证完整性，再建立索引

`deepinfer/moe/gguf_index.py` 做四件事：

1. 解析 GGUF v2/v3 header；
2. 解析 metadata，只保留标量/小数组，跳过 tokenizer 的 102400 个 token 字符串；
3. 解析 377 个 tensor 的 name、dims、GGML type、offset、nbytes；
4. 计算 `data_start = align(header_end)`，并检查：

```python
required_size = data_start + max(t.offset + t.nbytes for t in tensors)
if required_size > file_size:
    raise GGUFTruncatedError(...)
```

这一步把“半截文件加载”变成了加载前错误，而不是反量化到一半才失败。

GGML 的 dims 约定是输入维在前：

```text
blk.1.ffn_gate_exps.weight  dims = (2048, 1408, 64)
     = (in_features, out_features, n_experts)
```

对于单个 expert，按如下方式取 packed bytes：

```text
per_expert_bytes = in_features * out_features / block_size * type_size
raw = mmap[data_start + tensor.offset + expert_id * per_expert_bytes : ...]
```

反量化后形状是 `(out_features, in_features)`，正好对应
`torch.nn.functional.linear` 的权重形状。gate/up 的 3D 张量因此可以直接按
`[expert, out, in]` 组织。

### 3.3 并行 expert slice 读取

如果按整个 projection 顺序读 6 个 expert，每次只有 3 个 read task，
NVMe 队列深度不够。实现改成 **每个 expert slice 一个 read task**：

```python
for spec, buffer in zip(specs, pinned_buffers):
    view = memoryview(buffer.numpy())
    for position, expert_id in enumerate(expert_ids):
        destination = view[position * size : (position + 1) * size]
        read_futures.append(executor.submit(
            self._read_expert_slice, spec, expert_id, destination
        ))
for future in read_futures:
    future.result()
```

- 每个 worker 线程有自己的文件句柄；
- destination 是互不重叠的 `memoryview` 切片；
- worker 数量默认 8；
- 实测 26 层 × 3 projection × 6 expert 的 893 MB 读取从约 0.39 s/token
  降到约 0.14 s/token。

### 3.4 pinned staging + copy stream + Event

这是速度与正确性的核心。

```python
# 复用 pinned buffer 前，等待上一次 copy 完成
previous_copy = self._pinned_copy_events.get(projection)
if previous_copy is not None:
    previous_copy.synchronize()

# gate/up 在当前 compute stream
packed_gate = pinned_gate.to(device, non_blocking=True)
gate_ready = torch.cuda.Event()
gate_ready.record(torch.cuda.current_stream())
self._pinned_copy_events["gate"] = gate_ready

# down 在独立 copy stream 上先发，尽量与 gate/up 的反量化 + bmm 重叠
with torch.cuda.stream(self._copy_stream):
    packed_down = pinned_down.to(device, non_blocking=True)
    down_ready = torch.cuda.Event()
    down_ready.record(self._copy_stream)

# gate/up 先做 dequant + bmm
...

# down 的 dequant 前等待事件
torch.cuda.current_stream().wait_event(down_ready)
packed_down.record_stream(torch.cuda.current_stream())
```

注意两个细节：

1. **pinned buffer 不是普通 CPU tensor**。异步 H2D 的来源必须保持有效；如果
   下一 chunk 复用同一块 pinned 内存，旧 copy 未完成就会读脏数据；
2. **copy stream 的 tensor 要 `record_stream`**。`packed_down` 是在 copy stream
   上分配/使用的，后续却被 compute stream 的反量化 kernel 消费。`record_stream`
   让 PyTorch allocator 知道 compute stream 也用它，避免过早复用。

### 3.5 SM86 GPU 反量化

`deepinfer/moe/gpu_quant.py` 提供：

- `dequantize_q4_k`；
- `dequantize_q5_0`；
- `dequantize_q8_0`；
- `dequantize_matrix` 按 GGUF type 分派。

三者都用 CuPy `RawKernel` 编译到当前 CUDA 设备，kernel 输出 FP16。以 Q4_K
为例，每个 256-value super-block 是 144 bytes：

```text
d (FP16) + dmin (FP16) + 12 bytes scales/mins + 128 bytes nibbles
```

kernel 的每个 CUDA block 处理一个 super-block，每个 thread 计算 256 个输出值
中的一个；scale/min 的 6-bit 解包直接写在 kernel 里，避免在 Python 里产生
大量临时张量。

Q4_K_M 里 down 层不全是 Q8_0，有些层是 Q5_0；这是实测读 GGUF tensor 目录后
才发现的。Q5_0 是 22 bytes / 32 values（FP16 scale + 4-byte high-bit mask +
16 bytes low nibbles），也必须实现，否则会在 `dequantize_q8_0` 的 size check
处失败。

验证方式不是“看起来能跑”，而是把 GPU kernel 输出与 `gguf.dequantize` 的
CPU 结果逐元素对照：

```text
Q4_K  max_abs ≈ 5.7e-5, cosine ≈ 0.999991
Q5_0  max_abs ≈ 6.1e-5, cosine ≈ 1.000000
Q8_0  max_abs ≈ 6.0e-5, cosine ≈ 0.999991
```

误差来自最终 FP16 舍入；同一输入重复调用是 bitwise 相同的。

### 3.6 grouped MoE：把 18 次小 GEMM 变成 3 次 grouped bmm

decode 一个 token 时，每层有 6 个专家。逐专家 `F.linear` 会产生 18 次小 GEMM
和大量 launch 开销。实现改为：

1. 对 `topk_idx` 的 assignment 按 expert id 排序；
2. 用 `bincount + cumsum` 算出每个 expert 的 token 数；
3. 把每个 expert 的 token 收集到 `padded_tokens[expert, position, hidden]`；
4. 一次 `torch.bmm(gate_w, x)`、一次 `bmm(up_w, x)`、一次 `bmm(down_w, h)`；
5. 按 assignment index scatter 回每个 token/slot。

对 prefill，专家总数可能到 64，所以 `expert_chunk_size=8` 一组一组做，
而不是一次 materialize 1.1 GiB 的专家权重。

### 3.7 Attention / MLA 与模型加载

MLA attention 和 decoder 的数学实现没有重写，而是复用 ModelScope/HuggingFace
仓库中的 `modeling_deepseek.py`：

- `AutoConfig` 读取 `config.json`；
- `get_class_from_dynamic_module()` 加载 `DeepseekV2ForCausalLM`；
- `accelerate.init_empty_weights()` 在 meta device 上构造 15.7B 参数；
- 只把非 routed expert 权重从 GGUF 读出来替换 meta 参数；
- routed expert 的 meta 参数直接丢弃，`DeepseekV2MoE.forward` 替换成
  `_streaming_moe_forward`；
- attention / router / norms / dense layer0 / shared / lm_head 按预算加载；
- token embedding 放 CPU，避免 embedding + lm_head + attention + shared 同时
  占满 4 GB；
- KV cache 用 `DynamicCache`；短 context 下 KV 只有几十 MiB。

这样做的优点是 attention/MLA/RoPE/YaRN 的数学由官方实现负责，我们只需要专注
MoE offload 和显存管理。`config.max_position_embeddings` 从 163840 降到 2048，
因为官方实现会为每一层预生成 RoPE cos/sin cache；原值在 27 层上会浪费数 GiB
host RAM。

### 3.8 生成循环

- tokenizer 用本地 `tokenizer.json` 和 `tokenizer_config.json`；
- `apply_chat_template` 生成 `User: ...\n\nAssistant:`；
- prefill 一次 forward，得到第一 token；
- decode 每次只输入一个 token，`DynamicCache` 累积 K/V；
- greedy 默认 temperature=0；
- 生成 token IDs 写入 `GenerationResult`，便于复现实验和逐 token 比较。

---

## 4. 调试过程：那些“看起来是数学错误”的 race

这一节记录最重要的一轮排查，因为最终结果的可复现性就是在这里建立起来的。

### 4.1 现象

相同 prompt、greedy、相同模型，四个连续请求得到：

```text
0: 混合专家模型（MoE）是一种利用多个专家…
1: 混合专家模型（MoE）是一种人工智能技术…
2: Mixed Expert Model (MoE) is a sophisticated…
3: 混合专家模型（MoE）是一种人工智能技术…
```

有时甚至出现：

```text
": (  “  ”  “
```

这不是采样随机性（temperature=0）。也不是简单的 FP16 舍入，因为 layer 1 的
hidden state 最大差异达到 0.278，随后逐层放大。

### 4.2 逐层 hook

在每层 decoder layer 上注册 forward hook，连续跑两次相同 prefill：

```text
embed equal True
layer 0 equal True
layer 1 equal False max 0.278
layer 2 equal False max 25.65
...
final equal False
```

layer 0 输出完全相同，说明 embedding、第一层 dense MLP 和第一层 attention 没
问题；layer 1 是第一个 MoE 层，问题锁定在 `DeepseekV2MoE.forward`。

### 4.3 检查 router 和 shared expert

对同一个 layer1 输入重复调用：

```text
gate idx_eq True
gate w_eq True
shared eq True
```

router 和 shared expert 都是确定的，问题进一步缩小到 routed expert 的读取、
拷贝、反量化或 scatter。

### 4.4 手工展开 chunk 阶段比较

写一个 debug 版本，把 `_streaming_moe_forward` 的每个中间量保存下来：

```text
chunk 0 stage gw diff 0.4307
chunk 0 stage uw diff 0.2962
chunk 0 stage dw diff 0.5454
chunk 1 stage gw diff 0.4307
...
```

连反量化后的 expert 权重本身都不同，说明不是 bmm，而是 expert store 的
输入/拷贝路径。

### 4.5 根因和修复

prefill 的 6 个 chunk 连续调用 `load_experts()`，而 pinned staging buffer 是
复用的。`non_blocking=True` 的 H2D 只是把 copy 排入 stream，CPU 可以立刻返回；
下一个 chunk 的 `readinto` 可能在 copy 完成之前覆盖源 buffer。

修复见第 3.4 节：每个 projection 记录一个 CUDA Event，在复用 pinned buffer
之前执行 `event.synchronize()`。这会让 CPU 在最坏情况下等完上一 chunk 的
H2D，但 copy 本身很短；正确性和可复现性优先。

### 4.6 第二个修复：确定性 scatter

修复 pinned race 后，逐层比较已经一致，但连续生成仍可能出现微小差异。
最后把 `index_add_` 替换为“按 assignment 写入 + FP32 top-k 求和”。

修复后，同一进程连续四次 20-token greedy：

```text
0 tps 2.309 same True prefix 20
1 tps 2.726 same True prefix 20
2 tps 2.686 same True prefix 20
3 tps 2.713 same True prefix 20
```

这才是可以把结果写进文章、写进 JSON benchmark 的状态。

---

## 5. 性能数据

### 5.1 最终质量/吞吐基准

命令：

```powershell
F:\QwenQLoRA\venv\Scripts\python.exe `
  scripts\smoke_deepseek_v2_lite_moe.py `
  --prompt "用一句话解释什么是混合专家模型（MoE）。" `
  --prompt "中国的首都是哪里？请只回答城市名。" `
  --prompt "把这句话翻译成英文：今天天气很好。" `
  --prompt "只回答数字：5-2=?" `
  --prompt "9.11和9.9哪个大？只回答更大的数字。" `
  --max-new-tokens 24 --warmup-tokens 4 --max-cache 256 `
  --output docs\results\deepseek_v2_lite_q4km_moe_sm86.json
```

结果（同一进程，warmup 之后）：

| prompt | prefill | decode token 数 | decode t/s | peak GPU |
|---|---:|---:|---:|---:|
| MoE 解释 | 1.996 s | 23 | **2.644** | 2278.6 MiB |
| 首都 | 2.259 s | 6 | **2.931** | 2278.6 MiB |
| 翻译 | 1.991 s | 6 | **2.916** | 2278.2 MiB |
| 5-2 | 1.945 s | 6 | **3.007** | 2277.7 MiB |
| 9.11 vs 9.9 | 2.439 s | 6 | **2.967** | 2281.5 MiB |

上表是默认 grouped + double-buffer 路径。grouped 无 double buffer 时 decode 为
2.53–2.95、prefill 2.20–2.88；per-expert fused 时 decode 2.56–3.05；
`--no-fused-moe` 的 dequant+bmm 回退路径为 2.43–2.77。所有路径 token IDs 一致。

峰值：

```text
peak allocated ≈ 2.23 GiB
peak reserved  ≈ 2.31 GiB
加载后空闲     ≈ 1.0 GiB
```

模型加载 17.6 s。第一次进程内的冷启动 warmup 可能很慢（曾观察 16 s），
因为 Q4_K_M 的 expert 页还没有进入 OS page cache；后续 prefill/decode 会稳定。

### 5.2 为什么比 llama.cpp CPU 慢

同一份 Q4_K_M 在 llama.cpp CPU 上：

| 测试 | llama.cpp CPU |
|---|---:|
| MoE 解释 | prompt ~5.5 t/s，generation ~10.9 t/s |
| 9.11/9.9 | prompt ~9.1–9.3 t/s，generation ~9.2–12.9 t/s |

llama.cpp 有成熟的 AVX2/AVX512 CPU 量化 GEMV、mmap 权重和专门的 MoE 调度；
本实现的主要时间花在：

- 从 NVMe/OS cache 读取 packed expert slice；
- H2D；
- fused kernel 内的 packed 权重解码与 dot-product；
- Python 层 router/assignment。

所以 2.56–3.05 tokens/s 不是“SM86 的上限”，而是 **DeepInfer 原生单请求执行层
在经过 fused MoE 后的可用速度**。下一阶段可以通过多 expert 合并 kernel、专家
预热与多流流水线继续提升。

### 5.3 关键优化前后

| 版本 | decode 稳态 |
|---|---:|
| 逐 expert Python + nonzero + mmap 随机读 | 约 1.02 t/s |
| 并行 readinto + index_copy 分组 | 约 1.6–1.9 t/s |
| pinned H2D + down copy stream overlap | 约 2.4–2.7 t/s |
| 修复 race 前的“2.7” | 不可复现，不能作为结果 |
| 修复 race 后的 dequant+bmm 版本 | **2.40–2.70 t/s，可复现** |
| per-expert fused GEMV/GEMM | **2.56–3.05 t/s；64-token 2.63；可复现** |
| grouped fused GEMV/GEMM | 2.53–2.95 t/s；64-token 2.64；可复现 |
| grouped + double-buffer | **2.64–3.01 t/s；64-token 2.63；prefill 1.95–2.44 s** |

---

## 6. 质量边界

### 6.1 已验证什么

- 同一 GGUF 的 GPU 反量化 kernel 与 `gguf.dequantize` 数值对照通过；
- fused GEMV/GEMM 与“反量化后 matmul”的 GPU 参考逐元素对照通过；
- fused 与 `--no-fused-moe` 在 5 个 prompt、64-token 长生成上 token IDs 完全一致；
- 短问答、翻译、简单算术、MoE 概念解释可读且语义正确；
- greedy 多次生成 token IDs 一致（修复 race 后）；
- 没有观察到 OOM、崩溃或 CUDA device-side assert。

### 6.2 没有证明什么

- Q4_K_M 不是 BF16，不能声称“与原始模型完全等价”；
- 质量回归有 mini suite 和 full-category 固定预算 suite，但还没有跑官方
  完整 14k MMLU / 1.3k GSM8K split；perplexity 与
  llama.cpp 的 tokenizer/context 口径也不完全相同；
- 没有和 llama.cpp 做逐 logits 对齐；两者执行顺序和累加精度不同；
- 4-bit MoE 的 router 对数值误差很敏感，长链推理仍可能出现与 BF16 不同的分支；
- “9.11 vs 9.9”这类诱导性长解释，模型中间句可能不严谨；本次直接问“只回答
  更大的数字”得到 9.9，但不代表所有推理题都稳。

结论：**本次在抽检范围内没有明显质量退化，但 Q4_K_M + FP16 bmm 仍然是有损的，
不能宣传成无损 BF16。**

---

## 7. 复现步骤

### 7.1 文件布局

```text
E:\DeepInfer-MoE\
  meta\
    config.json
    configuration_deepseek.py
    modeling_deepseek.py
    tokenization_deepseek_fast.py
    tokenizer.json
    tokenizer_config.json
    generation_config.json
  models\DeepSeek-V2-Lite-Chat-GGUF\
    DeepSeek-V2-Lite-Chat-Q4_K_M.gguf
  logs\
  results\
```

Meta 文件来自 ModelScope 的 `deepseek-ai/DeepSeek-V2-Lite-Chat`；Q4_K_M 来自
`second-state/DeepSeek-V2-Lite-Chat-GGUF`。

### 7.2 运行

```powershell
F:\QwenQLoRA\venv\Scripts\python.exe `
  scripts\smoke_deepseek_v2_lite_moe.py `
  --prompt "用一句话解释什么是混合专家模型（MoE）。" `
  --max-new-tokens 24 --warmup-tokens 4 `
  --output docs\results\deepseek_v2_lite_q4km_moe_sm86.json
```

### 7.3 测试

```powershell
PYTHONPATH=D:\DeepInfer-SM86 `
F:\QwenQLoRA\venv\Scripts\python.exe -m pytest -q tests\moe
```

覆盖：

- GGUF header/metadata 解析；
- 截断文件拒绝；
- 3D expert tensor 的 expert slice 反量化；
- Q4_K/Q5_0/Q8_0 GPU 反量化与 `gguf.dequantize` 对照；
- GPU kernel 重复调用一致性。

### 7.4 显存观察

```powershell
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv
```

本次加载后 GPU 仍有约 1.0 GiB 空闲；如果机器上还有浏览器/游戏占用显存，
`shared_experts_device=auto` 会自动把 shared experts 或 lm_head 回退到 CPU。

---

## 8. 代码地图

```text
deepinfer/moe/gguf_index.py
  - GGUF v2/v3 header 解析
  - tensor 目录、完整性检查、read-only mmap
  - 按 expert 切片的 dequantize

deepinfer/moe/gpu_quant.py
  - Q4_K/Q5_0/Q8_0 CuPy RawKernel
  - PyTorch fallback dequantizer
  - 分派与尺寸检查

deepinfer/moe/fused_moe.py
  - per-expert fused Q4_K/Q5_0/Q8_0 gate/up GEMV/GEMM
  - per-expert fused down GEMV/GEMM
  - packed row 解码 + dot-product，直接写 activation/output

deepinfer/moe/grouped_moe.py
  - grouped multi-expert gate/up GEMV/GEMM
  - grouped multi-expert down GEMV/GEMM
  - 2D grid `(output_row, expert)` + per-expert counts

deepinfer/moe/ptr_grouped_moe.py
  - pointer-array grouped gate/up GEMV/GEMM
  - pointer-array grouped down GEMV/GEMM
  - 支持 cached/missed expert 混合，无需拼连续 packed buffer

deepinfer/moe/paged_adapter.py
  - DeepSeekPagedCache：MLA K/V -> 通用 paged pool
  - 适配 K head_dim 与 V head_dim 不同的 MLA cache

deepinfer/moe/server_adapter.py
  - OpenAI server facade over the DeepSeek runtime

deepinfer/ecs/core.py
  - runtime-built C++ policy core facade
deepinfer/ecs/controller.py
  - heat/LRU/tier placement, chunk packing and cold selection
deepinfer/ecs/transport.py
  - expert packets, ECS router and simulated/torch all-to-all
deepinfer/ecs/ecs_core.cpp
  - C++ control-plane policy core

pypi/deepinfer-sm86/
  - standalone PyPI source tree and built wheel/sdist artifacts

deepinfer/moe/paged_mla_kernel.py
  - fused FP16 CUDA paged MLA attention
deepinfer/moe/paged_mla_quant.py
  - native INT8/FP8 E4M3/E5M2 paged MLA
  - one warp per (batch, head, query-token)

deepinfer/moe/cpu_avx2_packed.cpp
  - AVX2 Q4_K/Q5_0/Q8_0 packed dequant
deepinfer/moe/cpu_avx2_gemv.cpp
  - fused AVX2 packed gate/up/down GEMV
deepinfer/moe/cpu_avx2.py
  - runtime g++ build + ctypes bridge

deepinfer/moe/deepseek_v2_runtime.py
  - GGUFExpertStore：并行读取、pinned staging、copy stream、Event
  - DeepSeekV2LiteGGUFRuntime：meta 加载、权重放置、streaming MoE、生成
  - 确定性 assignment scatter / FP32 top-k 求和
  - CPU fallback

scripts/smoke_deepseek_v2_lite_moe.py
  - 质量/吞吐 CLI，输出结构化 JSON
scripts/eval_deepseek_moe_quality.py
  - mini quality regression runner
scripts/build_deepseek_full_quality_suite.py
  - fixed-budget MMLU/GSM8K/WikiText-2 suite builder
scripts/eval_deepseek_full_quality.py
  - full-category quality regression runner
scripts/smoke_deepseek_v2_lite_moe_paged.py
  - paged scheduler smoke
docs/quality/...
  - Q4_K_M quality suite

tests/moe/test_gguf_index.py
tests/moe/test_gpu_quant.py
tests/moe/test_fused_moe.py
tests/moe/test_grouped_moe.py
tests/moe/test_ptr_grouped_moe.py
tests/moe/test_paged_adapter.py
tests/moe/test_cpu_avx2.py
tests/moe/test_full_quality_suite.py
tests/moe/test_server_adapter.py
tests/moe/test_dense_support.py
```

---

## 9. 已知限制和下一步

### 当前限制

- 已接入 paged scheduler 的 chunked prefill / batched decode / KV 预算和
  prefill/decode stream overlap；CuPy kernels 已通过 ExternalStream 跟随
  torch stream，短 smoke token parity 通过；
- grouped GEMV/GEMM 已把 decode launch 降到 52/token；pinned double-buffer
  已实现，显著改善 prefill，但还没有跨层预测 prefetch；
- packed expert cache 已实现但默认关闭：512 MB 时约 11% hit，CPU cache copy
  和 eviction 开销反而使长生成降到 1.61 t/s；要进一步做 GPU cache；
- prefill 会扫描大量专家，短 prompt prefill 约 2.1–2.5 s；
- MLA 已有 fused paged attention：FP16 走 scalar/CTA/half2 CUDA kernel，
  INT8/FP8 E4M3/E5M2 走 native quant kernel；不再回退 dense materialize，
  但仍不是 FlashMLA 级优化；
- Q4_K_M 是唯一验证过的 DeepSeek-V2-Lite GGUF 量化档；
- 已做 full-category 固定预算质量回归；官方完整 MMLU/GSM8K split 和
  长上下文仍未跑；builder 已提供 `--mmlu-all --gsm8k-all` 开关；
- CPU MoE layer offload 仍是 PyTorch CPU FP32/oneDNN；1 个整层 CPU 约
  1.86 t/s；
- cold-expert split 已实现；推荐 2 层 × 1 cold expert，约 2.50 t/s；
  CPU 路径已有 fused AVX2 packed gate/up/down，oneDNN 仅作 fallback；
- prefill copy-stream pipeline 已实现但默认关闭；本机 WDDM 上 prefill 慢约
  6%；prefill/decode background prefetch 默认开启，prefill 3.43→2.03 s、
  decode 2.52→2.70 t/s；预测式 next-layer prefetch 默认关闭，路由重叠不足
  导致 2.70→2.21 t/s；CUDA Graph 仍因动态 MoE 路由不可安全 capture；
- 只验证 Windows + Python 3.10 + torch 2.5.1+cu121。

### 下一步优先级

1. **完整 MMLU/GSM8K split**：builder 已支持 `--mmlu-all --gsm8k-all`，
   需要数小时到十几小时评测预算；当前 fixed-budget suite 仍是发布门槛；
2. **受限 static CUDA Graph**：只对固定 routing/buffer 的 decode 子图做
   capture；完整 DeepSeek decode 因 top-k 动态路由仍不可安全 graph；
3. **更大显存/CPU offload 组合**：GPU cache 已验证但 1 GiB 只带来约 2.3%
   提升；继续需要把 shared/lm_head 移到 CPU 腾出 VRAM；
4. **fused paged MLA 继续优化**：大 tile/多 query 复用、减少 block-table
   访存，朝 FlashMLA 级靠近；
5. **多请求/长上下文压测**：4+ 请求、256/512/1024 context、取消和 OOM 组合。

---

## 10. 后记：Dense / Hybrid 层支持

最初这版运行时为了尽快跑通 DeepSeek-V2-Lite，默认了“layer 0 dense，其余全部
MoE”。这对 DeepSeek-V2-Lite 的配置刚好成立，但不是一个完整的执行层：

- DeepSeek 配置里有 `first_k_dense_replace`，不同模型可能是 1、3 或更多；
- `moe_layer_freq` 允许 MoE 与 dense 层交错；
- dense-only 模型没有 routed expert，不应该被迫走 `DeepseekV2MoE`；
- 即使只有 leading dense 层，也要明确 dense MLP 放 GPU 还是 CPU。

因此补了一层很薄的 layout 抽象和 dense MLP 加载路径。

### 10.1 层布局判定

`deepinfer/moe/layout.py` 直接镜像官方 `DeepseekV2DecoderLayer` 的判定：

```python
def is_moe_layer(config, layer_idx):
    num_experts = routed_expert_count(config)
    if num_experts == 0:
        return False
    first_dense = int(getattr(config, "first_k_dense_replace", 0) or 0)
    frequency = max(1, int(getattr(config, "moe_layer_freq", 1) or 1))
    return layer_idx >= first_dense and layer_idx % frequency == 0
```

`split_layer_indices()` 返回：

```python
dense_layer_indices, moe_layer_indices = split_layer_indices(config)
```

对 DeepSeek-V2-Lite：

```text
dense = [0]
moe   = [1, 2, ..., 26]
```

对 `first_k_dense_replace=3` 的 hybrid 模型：

```text
dense = [0, 1, 2]
moe   = [3, 4, ...]
```

对 `n_routed_experts=None/0` 的 dense-only 模型：

```text
dense = [0, 1, ..., N-1]
moe   = []
```

如果 config 里写的是 `n_routed_experts=0`，在构造官方模型前会归一化成 `None`，
因为官方 `DeepseekV2DecoderLayer` 只把 `None` 当成 dense。

### 10.2 Dense MLP 的加载和放置

对每个 `dense_layer_indices` 中的层，按照 GGUF 的 dense FFN 名字加载：

```text
blk.{i}.ffn_gate.weight
blk.{i}.ffn_up.weight
blk.{i}.ffn_down.weight
```

MoE 层继续加载：

```text
blk.{i}.ffn_gate_inp.weight
blk.{i}.ffn_gate_shexp.weight / ffn_up_shexp / ffn_down_shexp
blk.{i}.ffn_gate_exps.weight / ffn_up_exps / ffn_down_exps
```

dense MLP 的设备选择由 `--dense-mlp-device auto/cuda/cpu` 控制。`auto` 会估算：

```text
dense_layer_bytes
+ 尚未加载的 attention 权重
+ 256 MiB 预留
```

如果 GPU 放得下，dense MLP 用 FP16 放在 GPU；否则：

1. 估算 CPU FP32 权重需要多少 RAM；
2. 用 `psutil.virtual_memory().available` 检查 host 余量；
3. 放 CPU FP32，并把该层 `forward` 替换成 `_cpu_mlp_forward`；
4. 如果 host 余量也不够，直接给出明确 `MemoryError`，而不是硬着头皮加载到 OOM。

这样：

- DeepSeek-V2-Lite 的 layer 0 dense 仍然在 GPU；
- 多个 leading dense 层或 hybrid dense 层不会被误当 MoE；
- dense-only 模型在 4 GB 卡上会自动把放不下的 dense MLP 回退到 CPU；
- routed expert store 只在 `moe_layer_indices` 非空时创建。

### 10.3 测试和边界

新增 `tests/moe/test_dense_support.py`：

- 默认 DeepSeek-V2-Lite layout；
- 多个 leading dense 层；
- `moe_layer_freq` 交错的 dense/MoE；
- `n_routed_experts=None/0` 的 dense-only；
- CPU MLP fallback 的 shape/dtype/数值对照。

当前 `tests/moe` 共 20 个测试（含 grouped 与 pointer-array GEMV/GEMM 对照），全套 79 个测试通过。

必须说明：本机没有 dense-only DeepSeek-V2 GGUF，因此 dense-only 的端到端
吞吐和质量没有像 Q4_K_M MoE 那样做真实 benchmark；“dense/hybrid 层可加载、
可回退、CPU MLP 数值正确”有单元测试，但没有冒充成完整 dense 模型基准。
dense-only 超出显存时回退 CPU FP32 也会很慢，这是后续可以继续做 dense
expert-style streaming 的地方。

## 11. Fused packed-weight MoE GEMV/GEMM

前面“反量化到 FP16 再 bmm”的路径已经能在 4 GB 卡上跑，但它仍然有一个明显
浪费：每个 decode token 需要把当前专家矩阵反量化成约 2.7 GiB 的 FP16 权重流量
（26 层累计），再让 cuBLAS 把这些权重从 HBM 读一遍。真正必要的计算量只有
0.75 GB 左右的 packed expert bytes 和少量 activation。

### 11.1 Kernel 设计

新增 `deepinfer/moe/fused_moe.py`，两个 kernel：

1. `dp_fused_gate_up_gemm`
   - 一个 CUDA block 处理一个 intermediate 输出 row；
   - 256 threads，每个 thread 从 Q4_K/Q5_0/Q8_0 packed row 解码最多 8 个权重
     到寄存器；
   - 对 n_tokens 个 token 分别做 dot-product，block reduction；
   - 直接写 `silu(gate) * up`，不写回 gate/up 的 FP16 权重矩阵。
2. `dp_fused_down_gemm`
   - 同样直接消费 packed down row；
   - 输入是上一步的 `[n_tokens, 1408]` activation；
   - 输出 `[n_tokens, 2048]`，不 materialize down 的 FP16 权重。

Q4_K_Q5_0_Q8_0 的 decode 逻辑写成 device function，kernel 按 qtype 分支；
gate/up 两个 packed 指针同时传入，因此 gate 和 up 共享同一个 x 读取循环。
GEMV（n_tokens=1）和 GEMM（n_tokens>1）复用同一个 kernel；多 token 时权重
只解码一次，token 维在 block 内循环。

### 11.2 内存与速度变化

旧路径每个 decode token 的核心流量：

```text
packed expert H2D          ≈ 0.75 GiB
dequant 写 FP16 expert     ≈ 2.7 GiB
grouped bmm 读 FP16 expert ≈ 2.7 GiB
```

新路径：

```text
packed expert H2D          ≈ 0.75 GiB
fused kernel 读 packed row ≈ 0.75 GiB
activation 写回             < 1 MiB
```

实测（同一组 5 个 prompt）：

| 指标 | dequant + bmm | fused MoE |
|---|---:|---:|
| decode 范围 | 2.43–2.77 t/s | **2.56–3.05 t/s** |
| 64-token 长生成 | 2.43 t/s | **2.63 t/s** |
| peak GPU allocated | 2.36 GiB | **2.23 GiB** |
| peak GPU reserved | 2.43 GiB | **2.31 GiB** |
| expert dequant launch 累计 | 非零 | **0** |

fused 之后 decode 的瓶颈更偏向 packed expert 读取和 H2D，而不是 GPU 反量化
写带宽；这正是下一阶段要做多 expert 合并和 pinned 预取的原因。

### 11.3 正确性验证

- `tests/moe/test_fused_moe.py`：
  - Q4_K gate+up，n_tokens=1 和 3；
  - Q5_0 down，n_tokens=4；
  - Q8_0 down，n_tokens=4；
  - 与反量化后的 GPU matmul 参考逐元素对照。
- `tests/moe/test_grouped_moe.py`：
  - 3 个 expert、不同 counts；
  - grouped gate/up 和 grouped down 与 per-expert 反量化参考对照。
- 模型级 A/B：
  - 5 个质量 prompt 的 token IDs 完全一致；
  - 64-token 长故事生成 token IDs 完全一致；
  - `--no-fused-moe` 可随时回退到旧路径。
- 全套测试：**112 passed**。

### 11.4 Grouped multi-expert kernel

per-expert fused 路径对每个 expert 分别 launch gate/up 和 down：decode top-6
时每层 12 次 kernel，26 层共 312 次/token。新增
`deepinfer/moe/grouped_moe.py`：

- grid 变成 2D `(output_row, expert)`；
- 一个 kernel 处理整个 expert chunk；
- `counts[expert]` 控制每个 expert 的真实 token 数；
- gate/up 一个 launch，down 一个 launch；
- decode 每层 launch 从 12 降到 2，每 token 从 312 降到 **52**。

结果：

| 指标 | per-expert fused | grouped fused |
|---|---:|---:|
| decode launch / token | 312 | **52** |
| 64-token decode | 2.63 t/s | **2.64 t/s** |
| 5-prompt decode 范围 | **2.56–3.05** | 2.53–2.95 |
| prefill | 2.81 s（长 prompt） | 2.95 s |
| peak allocated | 2.23 GiB | 2.23 GiB |
| token IDs | 基准 | 完全一致 |

结论要诚实：**launch 数确实降了 6 倍，但 wall time 基本不变，prefill 还略慢**。
在 4 GB 卡上当前瓶颈已经是 packed expert 读取和 H2D，不是 launch overhead。
grouped 版本为下一阶段的 double-buffer/多 copy stream 提供了更干净的接口。

### 11.5 当前 fused kernel 的边界

- block-per-row，Q4_K/Q5_0/Q8_0 packed row 解码在每行内完成；
- 还没有做多 expert 合并 launch、Tensor Core tile、cp.async prefetch；
- prefill 多 token 时按 token 串行 reduction，prefill 质量正确，速度与旧路径
  基本持平；
- decode 是主要收益场景；
- 仍然只验证了 DeepSeek-V2-Lite Q4_K_M 的 tensor 形状（K=2048/1408，
  intermediate=1408，hidden=2048）。

## 12. Double-buffer 与 packed expert cache

### 12.1 双 pinned slot

原实现对每个 projection 只有一块 pinned staging buffer。prefill 有多個 expert
chunk，CPU 进入下一 chunk 前必须等待上一 chunk 的 pinned buffer 不再被 H2D
读取；这会让 prefill 的读取和 GPU 计算之间出现不必要的串行化。

改成：

- 每个 projection 维护两个 pinned slot；
- 每次 `_stage_experts()` 轮换 slot；
- 每个 slot 的 gate/up/down 事件单独记录；
- 复用 slot 前只等待它自己的旧事件；
- gate/up 保持在 compute stream，down 使用 per-slot copy stream；
- grouped kernel 使用 `wait_for_gate_up()` / `wait_for_down()` 只在需要时
  等待跨 stream 事件。

结果：

| 指标 | grouped 无 double buffer | grouped + double buffer |
|---|---:|---:|
| 5-prompt prefill | 2.20–2.88 s | **1.95–2.44 s** |
| 5-prompt decode | 2.53–2.95 t/s | **2.64–3.01 t/s** |
| 64-token decode | 2.64 t/s | 2.63 t/s |
| token IDs | 基准 | **完全一致** |

decode 没有明显变化是合理的：decode 每层只有一个 expert chunk，主要受
下一层 attention/router 依赖限制；double-buffer 的收益主要在 prefill 的
多 chunk 读取。

### 12.2 packed expert cache：实现了，但默认关闭

实现了一个 per-layer LRU：

- key 为 `(tensor_name, expert_id)`，存 packed uint8 bytes；
- 第一次访问只计数，第二次访问才 admission，避免一次性 expert 污染缓存；
- `--expert-cache-mb` 控制大小，默认 **0**；
- 统计 `expert_cache_hits/misses/bytes/evictions`。

512 MB 实测（64-token 长生成）：

```text
cache hits       = 4056
cache misses     = 32700
hit rate         ≈ 11%
cache bytes      ≈ 515 MB
evictions        = 27595
decode           = 1.61 t/s
no-cache decode  = 2.63 t/s
```

根因不是命中率单独决定的：命中的 packed bytes 需要从页缓存/普通 CPU cache
再 memcpy 到 pinned staging；当前 Python/NumPy 路径的 copy 吞吐明显低于
NVMe 直接 read，且 admission/eviction 还有额外开销。因此 cache 目前只作为
opt-in 实验项保留，默认关闭。

如果下一步认真做 cache，应该：

1. 把 packed expert cache 放到 **GPU**，命中时直接给 grouped kernel 用；
2. grouped kernel 改为接受 **expert pointer array**，避免把 cached/missed
   expert 重新拼成连续 buffer；
3. 按实际路由统计做分层容量，而不是平均分配。

### 12.3 验证

- double-buffer 路径与 grouped、per-expert、dequant+bmm 的 token IDs 一致；
- 512 MB cache 路径 token IDs 也一致；
- 全套测试：**112 passed**。

## 13. GPU packed cache + pointer-array grouped kernel

### 13.1 为什么需要 pointer array

CPU cache 的瓶颈是命中后仍要 memcpy 到 pinned staging。要真正跳过数据搬运，
cache 必须放在 GPU；同时 grouped kernel 不能再假设所有 expert 的 packed
权重排成一块连续 buffer，因为一部分 expert 来自 cache，另一部分刚从文件
H2D 进来。

新增 `deepinfer/moe/ptr_grouped_moe.py`：

- kernel 参数从 `const __half* packed` 改成 `unsigned long long* ptrs`；
- 每个 block 通过 `ptrs[expert]` 取该 expert 的 packed 基址；
- 命中 expert 指向 cache tensor，miss expert 指向本 chunk 的 staging view；
- 不再需要把 cached/missed expert 重排成一个连续 packed tensor。

### 13.2 cache 策略

- per-layer LRU，存整个 expert 的 gate/up/down 三个 packed tensor；
- 第一次 miss 只计数，第二次 miss 才 admission；
- prefill 阶段关闭 cache，避免大量 prefill expert 污染 decode 工作集；
- prefill 结束、decode 开始前清空 cache 并启用；
- `--gpu-expert-cache-mb` 控制容量，默认 **0**。

### 13.3 实测

64-token 长生成：

| 配置 | decode | prefill | hit rate | peak allocated |
|---|---:|---:|---:|---:|
| 无 GPU cache | 2.63 t/s | 2.56 s | 0% | 2.23 GiB |
| 512 MB GPU cache | 2.40 t/s | 3.11 s | 12.0% | 2.70 GiB |
| 1 GiB GPU cache | **2.69 t/s** | 2.93 s | **24.8%** | **3.20 GiB** |

1 GiB 只换来约 **2.3%** decode 提升，却多占约 1 GiB VRAM；512 MB 反而更慢。
在 4 GB RTX 3050 上，OOM 风险比这点收益更重要，所以默认关闭。

### 13.4 验证

- pointer-array gate/up 和 down 都有独立 GPU 单测；
- GPU cache 路径与无 cache 路径 token IDs 完全一致；
- 全套测试：**112 passed**。

## 14. 质量回归（mini suite）

完整 MMLU/GSM8K 在 2.6 tok/s 的 native 路径上耗时不可接受，因此先落一个
可重复的 release gate：

- `docs/quality/deepseek_v2_lite_quality_suite.json`
  - 4 段 raw text perplexity；
  - 10 道 MMLU-style 四选一；
  - 8 道 GSM8K-style 数学应用题；
  - 5 道多步长链推理。
- `scripts/eval_deepseek_moe_quality.py`
  - 同 prompt、同 greedy 采样跑 native；
  - 可选调用 llama.cpp CPU 的 `llama-cli` + `llama-perplexity`；
  - 支持 `--skip-llama`、`--limit`、`--no-fused-moe`。

64-token 前的完整 mini suite 结果：

| 指标 | DeepInfer native | llama.cpp CPU |
|---|---:|---:|
| MMLU-style | **8/10（80%）** | 8/10（80%） |
| GSM8K-style | **4/8（50%）** | 2/8（25%） |
| 长链推理 | **3/5（60%）** | 1/5（20%） |
| raw-text PPL | 4.97 | 3.19 |

不能把这个表宣传成完整 MMLU/GSM8K。两个 MMLU-style 失败是模型回答冗长、
在 64-token 内没有给出选项，或者算错 `2^10`；GSM8K-style 的失败多数是模型
算术错误，llama.cpp 同样错，且 native 在 GSM8K-style / 长链上高于 llama.cpp。
PPL 的 4.97 vs 3.19 也不是严格同口径：native 是每段文本从 BOS 开始独立打分
后按 token 加权，llama.cpp 是对拼接语料按 `n_ctx=128` 分 chunk。

质量路径验证：

- fused 与 `--no-fused-moe` 在 3 个代表性 prompt 上 token IDs 完全一致；
- mini suite 中没有出现 kernel 崩溃、重复 token 或明显语言崩坏；
- 结果 JSON 保留每个任务的 prompt、response、token IDs 和通过/失败标记，
  便于下次逐项 diff。

运行：

```powershell
F:/QwenQLoRA/venv/Scripts/python.exe ^
  scripts/eval_deepseek_moe_quality.py ^
  --output docs/results/deepseek_v2_lite_q4km_moe_quality_eval.json
```

结果文件：`docs/results/deepseek_v2_lite_q4km_moe_quality_eval.json`。

### 14.1 Full-category 固定预算回归

为避免把 mini suite 当成完整跑分，2026-10-28 新增了固定预算的
full-category 回归：

- MMLU：57 个 subject 各抽样 1 题，覆盖全部科目；
- GSM8K：官方 test split 固定抽样 32/1319；
- 长链：确定性生成 32 道 5–7 步整数链；
- PPL：WikiText-2 validation 前 512 token；
- 同 greedy 参数运行 native，LLM 对照只录 llama.cpp PPL。

| 指标 | native Q4_K_M |
|---|---:|
| MMLU | **27/57（47.4%）**，Wilson 95% CI 35.0–60.1% |
| GSM8K | **9/32（28.1%）** |
| 长链推理 | **2/32（6.25%）** |
| WikiText-2 PPL | **6.59**（512 token） |
| llama.cpp PPL | 8.04（`n_ctx=128` 分块，非严格同口径） |

结论：

- native 路径没有出现崩溃、重复 token 或语言崩坏；
- MMLU zero-shot 每科 1 题，47.4% 只能作为回归基线，不能替代官方
  5-shot/full-split 分数；
- GSM8K 28.1% 和长链 6.25% 暴露出模型在受限输出预算下的算术/多步推理
  弱项，这是模型能力问题，不是 kernel 路径错误；
- llama.cpp 生成对照因 CPU 16B Q4 成本过高没有跑满，只保留了 PPL 参考。

结果文件：`docs/results/deepseek_v2_lite_q4km_moe_full_quality.json`。
Suite：`docs/quality/deepseek_v2_lite_full_quality_suite.json`。

## 15. Paged scheduler 接入

`apps/paged_scheduler.py` 原本只面向 Qwen GQA。DeepSeek-V2-Lite 的 MLA
K/V 维度不同（K = qk_nope + qk_rope = 192，V = v_head_dim = 128），而且
官方 attention 走 eager 路径，所以这次没有硬塞一个错误形状的 cache，而是
增加兼容桥 `deepinfer/moe/paged_adapter.py`：

- `DeepSeekPagedCache` 实现 HF `Cache` 接口；
- `write()` 把当前 MLA K/V padding 到统一 head_dim，写入通用 paged pool；
- `attend()` 直接按 block table 逐块读取 K/V，用 online softmax 累积，不再
  materialize 当前 batch 的 dense K/V；
- FP16 路径新增 `deepinfer/moe/paged_mla_kernel.py`：一个 warp 负责一个
  `(batch, head, query-token)`，在 fused CUDA kernel 里完成 Q.K、online
  softmax 和 V 累加；
- `install_native_paged_attention()` 在每个 DeepSeek attention 上安装 paged
  forward，q_len=1 和 chunk prefill 都走 blockwise attention；
- run-time `forward_logits(input_ids, positions, kv_cache=...)` 暴露给 scheduler；
- scheduler 自动从 DeepSeek config 推断 `num_attention_heads=16` 和
  `head_dim=max(192,128)=192`，并继续使用 block allocator、prefix cache 和
  KV 预算。

当前是 fused FP16 CUDA paged MLA kernel，不是 FlashMLA：

| 路径 | 2 请求 + prefix 耗时 |
|---|---:|
| dense materialize | 11.89 s |
| PyTorch block loop | 13.25 s |
| fused FP16 CUDA paged MLA（scalar） | **11.43 s** |

fused kernel 有三种 mapping：

- `scalar`（默认）：q 缓存在寄存器，标量 coalesced K/V load；
- `half2`：half2 向量化；
- `cta`：4 warp 分摊 key range，再合并 partial softmax。

kernel 微基准（B2/H16/Q1/seq28，200 次）：

| variant | ms/call |
|---|---:|
| scalar | **0.0616** |
| half2 | 0.0676 |
| cta | 0.0666 |

scalar 变体在这个 smoke 中略快于 dense materialize，token IDs 完全一致。
主要收益仍是去掉 dense K/V materialize 和 Python block loop/launch 开销；
它还不是 FlashMLA 级访存/CTA 优化。

INT8/FP8 native quant paged MLA（2 请求、prefix cache、20 blocks）：

| KV dtype | KV memory | token IDs vs FP16 | 说明 |
|---|---:|---:|---|
| FP16 | 106.2 MB | 基准 | fused FP16 kernel |
| INT8 | **53.6 MB** | 语义正确，算术输出空格/tokenization 改变 | per-head/token scale |
| FP8 E4M3 | **53.1 MB** | **完全一致** | 直接 FP8 byte decode |

INT8/FP8/E5M2 三条 storage 路径都有单元测试，与 materialize/dequant 后的
PyTorch reference 对比通过。INT8 在 smoke 中输出 `5 - 2 = 3` 而不是
`5-2=3`，是量化后 logits 改变导致的 tokenization 差异，不是 kernel 崩溃。
结果文件：`docs/results/deepseek_v2_lite_q4km_moe_paged_quantized.json`。

实测（`max_cache=128`、`block_size=16`、`prefill_chunk_tokens=8`）：

| 指标 | 结果 |
|---|---:|
| 2 请求 token IDs vs 单请求 baseline | **完全一致** |
| max decode batch | **2** |
| decode tokens total | 36 |
| prefix cache 复用 | **16 token** |
| prefix 后 token IDs | **完全一致** |
| KV pool | 20 blocks / 约 106 MB |
| native paged attention | **开启（27 层 + fused CUDA）** |
| 2 请求 + prefix 总耗时 | **11.43 s**（materialize 11.89 s） |
| prefill/decode stream overlap | **开启** |
| overlap 后 token IDs | **完全一致** |

第一版直接开两个 CUDA stream 时输出退化成 `!!!!`。根因不是 expert pinned
buffer，而是 **CuPy raw kernel 没有跟随 PyTorch 当前 stream**：attention/
linear 在非默认 stream，MoE/paged MLA 的 CuPy kernel 却仍落在默认 stream。
修复方式是在 scheduler 的 stream 上下文里设置
`cp.cuda.ExternalStream(torch_stream.cuda_stream)`，让 CuPy current stream
与 torch 一致。修复后：

| 配置 | 总耗时 | overlap steps | token IDs |
|---|---:|---:|---:|
| serial（max_new=8） | 10.07 s | 0 | 一致 |
| overlap（max_new=8） | **9.73 s** | 1 | 一致 |

24-token 对照为 9.66 s serial vs 9.95 s overlap，说明这个 2 请求 workload
只有一步真实 overlap，收益在噪声内。当前结论是**stream 布线和正确性已验证**，
不是宣称大幅吞吐提升。结果文件：
`docs/results/deepseek_v2_lite_q4km_moe_paged_overlap.json`。

运行：

```powershell
F:/QwenQLoRA/venv/Scripts/python.exe ^
  scripts/smoke_deepseek_v2_lite_moe_paged.py ^
  --enable-prefill-decode-overlap --prefix-reuse ^
  --output docs/results/deepseek_v2_lite_q4km_moe_paged_scheduler.json
```

结果文件：`docs/results/deepseek_v2_lite_q4km_moe_paged_scheduler.json`。

### 15.1 DeepSeek OpenAI-compatible gateway

`apps/server.py` 新增 `--backend deepseek`：

- 新增 `DeepSeekServerRuntime` adapter，把 native runtime 的 `generate()`、
  `forward_logits()`、`_sample()` 暴露成 server/scheduler 需要的接口；
- `/v1/chat/completions` 走 chat template；
- `/v1/completions` 走 raw prompt，不套 chat template；
- `/v1/models`、`/health`、`/metrics` 已接入；
- `--scheduler` + paged backend 提供 true token SSE、取消、prefix cache 和
  prefill/decode overlap；
- 不带 scheduler 时，SSE 退化为生成完成后按文本切片发送。

推荐启动：

```powershell
python apps/server.py `
  --backend deepseek `
  --scheduler --scheduler-backend paged `
  --max-cache 256 --max-num-seqs 2 `
  --prefill-chunk-tokens 8 --kv-blocks 20 `
  --prefix-cache-blocks 4 --port 18002
```

实测 `/v1/models`、`/health`、chat 非流式、chat SSE、raw completion 都返回
正确 JSON；scheduler SSE 按 token 到达，`5-2=3` 与 `2+3=5` 均正确。

## 16. CPU MoE layer offload

新增 `--cpu-moe-layers N`：把最后 N 个真实 MoE 层放到 CPU FP32 路径。

实现：

- runtime 选择 `moe_layer_indices[-N:]`；
- 这些层在 `_streaming_moe_forward` 内走 `_moe_cpu_fallback()`；
- CPU 路径使用 `GGUFExpertStore.load_experts(device='cpu', dtype=torch.float32)`；
- 每个 expert chunk 仍限制为 8，避免 CPU FP32 权重一次性爆内存；
- attention、shared experts 和 lm_head 仍在 GPU；
- 默认 `--cpu-moe-layers 0`。

32-token 长故事实测：

| 指标 | 全 GPU | 最后 1 层 CPU |
|---|---:|---:|
| decode | 2.50 t/s | **1.86 t/s** |
| prefill | 3.19 s | 4.39 s |
| peak GPU allocated | 2259 MiB | 2259 MiB |
| token IDs / text | 基准 | **完全一致** |

CPU 端是 PyTorch CPU FP32 bmm，实际使用 oneDNN/AVX2，但不是手写的 AVX2
packed GEMV kernel。这个实验验证了层切分和 CPU 专家读取的正确性，也给出了
明确代价：1 层 offload 就会降到约 1.86 tok/s。下一节进一步做了 expert 级
冷热切分。

## 17. Hot/cold expert split + AVX2 packed CPU

在整层 CPU offload 之外，新增 expert 级冷热切分：

- `--cpu-experts-per-layer N`：每层把 N 个 cold experts 放 CPU；
- `--cpu-expert-layers N`：只对最后 N 个 MoE 层启用；
- 每层维护累计路由计数，从当前 selected experts 中选计数最低的 N 个；
- GPU 继续计算 hot experts；
- CPU 使用 `cpu_avx2_gemv.cpp` 的 fused AVX2 packed GEMV：
  Q4_K gate/up + Q5_0/Q8_0 down；
- CPU expert packed 权重有 512 MB LRU cache；
- 不再先 materialize FP32 权重；oneDNN 版本仅作 fallback；
- token 累加器使用固定栈数组，减少 K-block 循环里的小分配。

实测矩阵（每层 1 个 cold expert，32-token 长故事）：

| CPU 层数 | decode | prefill | CPU MoE | token IDs |
|---:|---:|---:|---:|---:|
| 2 | **2.50 t/s** | 3.33 s | 0.31 s | 一致 |
| 3 | 2.49 t/s | 3.39 s | 0.56 s | 一致 |
| 4 | 2.30 t/s | 3.82 s | 0.65 s | 一致 |

推荐 **2 层 × 1 cold expert**：几乎无速度损失，同时仍保留 CPU/GPU 切分。

这比整层 CPU offload 的 1.86 t/s 好，说明少量 cold experts 放到 CPU 是可行的
折中。当前 CPU cold-expert 路径已经是 fused AVX2 packed GEMV；oneDNN 版本保留
为 fallback。2 层配置的 token IDs/text 与全 GPU 完全一致。

## 18. Prefill copy-stream pipeline（实验，默认关闭）

新增 `--prefill-pipeline`：

- prefill 多 token chunk 的 gate/up/down 三个 H2D 都发到该 slot 的 copy stream；
- compute stream 先等 gate/up，down 继续异步；
- 配合双 pinned slot，尝试重叠下一 expert chunk 的读取/H2D 与当前 grouped
  kernel；
- decode 仍保持 gate/up 在 compute stream，避免 decode 的事件开销。

A/B 实测（32-token 长故事）：

| 配置 | prefill | decode |
|---|---:|---:|
| pipeline off | **3.11 s** | **2.58 t/s** |
| pipeline on | 3.29 s | 2.56 t/s |

token IDs / text 两种配置完全一致。Windows/WDDM 上 copy stream + event 调度
开销使 pipeline 在本机反而慢约 6%，所以默认关闭，仅作为实验开关。

另外试验了 `--prefill-all-experts`（所有 routed experts 合并成一个 prefill
chunk）：token IDs 正确，但 prefill 3.59 s，也更慢，因此同样默认关闭。

CUDA Graph 没有强行接入：DeepSeek 的 top-k 路由会改变 expert 集合、tensor
地址和 chunk 数，当前不是可安全 capture 的静态图。更现实的方向是继续做
prefill chunk 的显式预取，而不是 graph capture。

结果文件：`docs/results/deepseek_v2_lite_q4km_moe_prefill_pipeline.json`。

## 19. Prefill background prefetch

真正有效的 prefill 优化是 CPU 后台预读，而不是 copy-stream overlap：

- `GGUFExpertStore` 新增两个 prefetch pinned slot；
- 当前 chunk 计算前，先通过已有 `ThreadPoolExecutor` 提交下一 chunk 的
  packed expert 读取；
- 主线程只负责当前 chunk 的 H2D 和 grouped kernel；
- 下一 chunk 的读取在后台完成，主线程消费 prefetch buffer；
- 每个 prefetch slot 用 Event 防止覆盖尚未完成 H2D 的 pinned buffer；
- 默认开启，`--no-prefill-prefetch` 可关闭。

32-token A/B：

| 配置 | prefill | decode | token IDs |
|---|---:|---:|---:|
| prefetch off | 3.43 s | 2.55 t/s | 基准 |
| prefetch on | **2.03 s** | 2.51 t/s | **完全一致** |

prefill 降低约 **41%**，decode 在噪声范围内。结果文件：
`docs/results/deepseek_v2_lite_q4km_moe_prefill_prefetch.json`。

### 19.1 Decode prefetch / double-buffered weights

decode 无法预知下一层路由，但可以在 **router 产出当前层 expert IDs 后**立即
启动后台预读：

- router → `expert_ids.cpu().tolist()`；
- 立即后台读取这 6 个 expert 的 packed gate/up/down；
- 主线程同时完成 GPU sort / padding / assignment layout；
- 读取完成后只做 H2D + grouped kernels；
- 双 pinned prefetch slot + Event 保护源 buffer。

32-token decode：

| 配置 | decode | token IDs |
|---|---:|---:|
| prefetch off | 2.52 t/s | 基准 |
| prefetch on | **2.70 t/s** | **完全一致** |

paged scheduler 2 请求 + prefix cache smoke 也从 11.07 s 降到 9.70 s，token
IDs 一致。该路径默认开启，`--no-decode-prefetch` 可关闭。结果文件：
`docs/results/deepseek_v2_lite_q4km_moe_decode_prefetch.json`。

### 19.2 Predictive next-layer prefetch（实验，默认关闭）

实现了用上一 token 路由预热下一层：

- 在层 L 记录当前 top-6 路由；
- 处理下一层时，用上一 token 在下一层的 top-6 路由后台预读 packed
  gate/up/down；
- 预读与当前层计算重叠，进入下一层后命中部分 expert 复用 256 MB
  predictive cache；
- cache key 带 layer/projection/expert，避免跨层串数据。

但路由集合不稳定：16-token probe 相邻 token top-6 重叠约 12%–45%，exact
match 几乎为 0。实测：

| 配置 | decode | predictive hits | prefetch reads |
|---|---:|---:|---:|
| predictive off | **2.70 t/s** | — | — |
| predictive on | 2.21 t/s | 8,822 | 14,850 |

浪费的预测读取和 partial pinned copy 抵消了命中收益，所以该路径只是
`--predictive-prefetch` 实验开关，默认关闭；token IDs/text 仍完全一致。
结果文件：`docs/results/deepseek_v2_lite_q4km_moe_decode_predictive.json`。

## 20. PyPI 独立库

可复用库单独放在 `pypi/deepinfer-sm86/`：

- PyPI 包名：`deepinfer-sm86`；
- import 名：`deepinfer`；
- 包含 GGUF reader、grouped MoE、FP16/INT8/FP8 paged MLA、CPU AVX2
  source、paged KV、memory budget、metrics 和通用 paged scheduler；
- 不包含模型权重、tokenizer 文件、Qwen server 和 benchmark workspace。

已构建：

```text
deepinfer_sm86-0.1.0-py3-none-any.whl
deepinfer_sm86-0.1.0.tar.gz
SHA256SUMS
```

隔离 target 安装验证通过：`DeepSeekV2LiteGGUFRuntime`、
`PagedBatchScheduler`、`deepinfer.moe.cpu_avx2.available()`，且 wheel 内含
两个 AVX2 `.cpp` 源文件和 LICENSE/NOTICE。

## 21. ECS control plane

新增 `deepinfer/ecs/`：

- `ECSController`：heat、tier placement、LRU victim、cold expert 选择；
- `ecs_core.cpp`：C++ CPU 控制面，负责 heat/LRU/victim 和 chunk packing；
- 真实 cache callback：C++ victim 会释放实际 GPU/CPU cache tensor；
- ECS chunk planner 已接入 runtime 的 expert chunk 切分；
- `ECSRouter`：queue depth/KV pressure 选设备，hot expert 可拆 token 到多卡；
- `ResilientAllToAll`：worker 失败后 retry/reroute；
- heat profile 可持久化，支持 Prometheus text snapshot。

cache/profile smoke 实测：

| 指标 | 结果 |
|---|---:|
| ECS backend | C++ |
| 输出 | `5-2=3` 正确 |
| L1 占用 | 约 125 MiB / 128 MiB |
| L2 占用 | 约 510 MiB / 512 MiB |
| evictions | 3048 |
| chunk plans | 234 |
| L1 hits | 1470 |
| L2 hits | 1722 |
| decode | 1.42 tok/s（cache + cold split smoke） |
| heat profile | 已写盘并可 load |
| 默认 ECS | 关闭 |

### 双卡 RTX 3080 Ti 实机验证

临时双卡机：2×RTX 3080 Ti 12GB、440GB RAM、Ubuntu 22.04、
PyTorch 2.5.1+cu121、CuPy 14.2、Python 3.12。

| 项目 | 结果 |
|---|---:|
| 单卡 17-token prompt prefill | 0.90 s |
| 单卡 decode | 4.20 tok/s |
| 单卡 peak | 2.20 GiB |
| 两层 pipeline（0–12 / 13–26） | 4.76 tok/s |
| pipeline token parity | 与单卡完全一致 |
| 双 runtime worker pool | 两请求并发，分别落 GPU0/GPU1 |
| NCCL ExpertPacket all-to-all | rank1 收到 2 个跨卡包 |
| hot expert token shard | 2/2 跨卡 |
| 单模型 expert sharding | GPU0/GPU1 1416/1415 dispatches |
| expert sharding token parity | 与单卡完全一致 |
| NCCL expert-parallel async | 4.90 tok/s（per-chunk） |
| NCCL hybrid | 4.71 tok/s，prefill 1.74 s |
| remote worker peak | 50.4 MiB hybrid |

Layer pipeline 的 token IDs 与单卡基线相同：
`[207, 20, 12, 17, 28, 18, 100001]`。这说明跨 GPU hidden-state
传输、两层 stage 的 KV cache、norm/LM head 与 ECS expert streaming 能
保持数值结果一致。

验证到的真实路径：

- Linux 上编译的 C++ ECS core；
- `TorchCollectiveAllToAll` + NCCL 的真实跨卡 package exchange；
- 每卡一个独立 DeepSeek runtime 的 queue-depth worker pool；
- embedding+layers0–12 / layers13–26+norm+LM head 的两卡 layer pipeline。

Expert-sharded 路径也已跑通：ECS route placement 把 routed expert
分给 GPU0/GPU1，expert chunk 按 owner 拆分并在 owner GPU 上执行，
输出回到协调 GPU。GPU0/GPU1 分别 dispatch 1416/1415 个 expert，
token IDs 与单卡完全一致；代价是跨 PCIe activation/output 拷贝，
decode 3.25 tok/s，暂时慢于单卡 4.20 tok/s。

进一步把 in-model activation exchange 改为 NCCL：rank0 保留 attention /
shared experts / LM head，rank1 只加载 routed expert bytes。异步版本先用
`isend/irecv` 提交 remote expert chunk，再计算 local expert chunk，最后等待
remote output。输出 token 与单卡完全一致，294 个 remote expert requests；
remote rank 峰值仅37.7 MiB。

| 路径 | Prefill | Decode |
|---|---:|---:|
| 单卡 | 0.90 s | 4.20 tok/s |
| 阻塞 NCCL expert | 2.62 s | 3.44 tok/s |
| 异步 NCCL per-chunk | 2.02 s | **4.90 tok/s** |
| 异步 NCCL hybrid | **1.74 s** | 4.71 tok/s |

Hybrid 模式在 prefill 时按层批量提交 remote expert group，decode 时
保留 per-chunk 异步路径。45-token long-context 也保持 token parity，
但 per-chunk decode 4.28 tok/s，略低于单卡4.50 tok/s。

Expert ownership 仍以 50/50 最优：67% remote expert 会使 prefill
退化到7.79 s、decode2.74 tok/s。

新增 `NCCLRequestService`：支持并发提交、队列深度/active/completed
指标和 heartbeat 文件监控。2个并发提交全部完成，queue depth=0、
failed=0，heartbeat age 0.44 s。但 coordinator 仍一次执行一个请求，
这是服务层并发提交，不是 continuous batching。

新增 supervisor：监控 rank 退出和 heartbeat，rank1 崩溃时杀掉并重启
两个 rank，重新加载模型后重试整个请求。故障注入 `worker_exit_42` 后
第2次尝试成功，输出 token 正确，heartbeat healthy。它恢复的是整个请求，
不是 in-flight KV/cache 续跑。

当前仍是 point-to-point request/reply，不是 collective all-to-all；
continuous batching、in-flight 请求续跑和 hybrid 长上下文吞吐仍待做。
该机没有 P2P/NVLink。

结果文件：
`docs/results/deepseek_v2_lite_q4km_moe_ecs_smoke.json`、
`docs/results/deepseek_v2_lite_q4km_moe_ecs_cache_profile.json`、
`docs/results/deepseek_v2_lite_q4km_moe_ecs_2x3080ti.json`。
复现细节：`docs/REMOTE_ECS_2X3080TI.md`。
PyPI 库本地0.2.0已构建，正式 PyPI 仍为0.1.0，待上传。

## 22. 一句话结论

这次最重要的不是“4 GB 跑 15.7B”这个标题，而是把 **MoE 的路由、专家流式读取、
GPU 反量化、pinned H2D、确定性 scatter、MLA attention 和生成循环** 真正接进了
DeepInfer 的执行层，并在 4 GB RTX 3050 上用 Q4_K_M 跑出了 **2.64–3.01 tokens/s**
（64-token 长生成 2.63 tokens/s）
的可复现结果；速度还不是 llama.cpp CPU 的对手，但它是 native 的、可控的、
可以继续往下做 fused kernel 和调度的基线。
