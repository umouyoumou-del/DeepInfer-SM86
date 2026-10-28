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
| 模型加载 | 17.27 s |
| 冷启动 warmup（4 token） | 3.5–16 s，取决于 OS page cache |
| prefill（短 prompt） | 2.22–2.77 s |
| decode 稳态 | **2.40–2.70 tokens/s** |
| peak GPU allocated | **2.41 GiB** |
| peak GPU reserved | **2.43 GiB** |
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
| MoE 解释 | 2.349 s | 23 | **2.399** | 2410.2 MiB |
| 首都 | 2.544 s | 6 | **2.618** | 2410.0 MiB |
| 翻译 | 2.309 s | 6 | **2.595** | 2409.6 MiB |
| 5-2 | 2.224 s | 6 | **2.704** | 2409.1 MiB |
| 9.11 vs 9.9 | 2.772 s | 6 | **2.637** | 2412.8 MiB |

峰值：

```text
peak allocated ≈ 2.41 GiB
peak reserved  ≈ 2.43 GiB
加载后空闲     ≈ 1.0 GiB
```

模型加载 17.27 s。第一次进程内的冷启动 warmup 可能很慢（曾观察 16 s），
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
- 反量化到 FP16；
- cuBLAS grouped bmm；
- Python 层 router/assignment。

所以 2.7 tokens/s 不是“SM86 的上限”，而是 **DeepInfer 原生单请求执行层的第一版
可用速度**。下一阶段可以通过 fused packed MoE GEMV、专家预热与多流流水线继续
提升。

### 5.3 关键优化前后

| 版本 | decode 稳态 |
|---|---:|
| 逐 expert Python + nonzero + mmap 随机读 | 约 1.02 t/s |
| 并行 readinto + index_copy 分组 | 约 1.6–1.9 t/s |
| pinned H2D + down copy stream overlap | 约 2.4–2.7 t/s |
| 修复 race 前的“2.7” | 不可复现，不能作为结果 |
| 修复 race 后的最终版本 | **2.40–2.70 t/s，可复现** |

---

## 6. 质量边界

### 6.1 已验证什么

- 同一 GGUF 的 GPU 反量化 kernel 与 `gguf.dequantize` 数值对照通过；
- 短问答、翻译、简单算术、MoE 概念解释可读且语义正确；
- greedy 多次生成 token IDs 一致（修复 race 后）；
- 没有观察到 OOM、崩溃或 CUDA device-side assert。

### 6.2 没有证明什么

- Q4_K_M 不是 BF16，不能声称“与原始模型完全等价”；
- 只做了少量短 prompt 抽检，没有跑 MMLU、GSM8K、perplexity 或长链推理；
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

deepinfer/moe/deepseek_v2_runtime.py
  - GGUFExpertStore：并行读取、pinned staging、copy stream、Event
  - DeepSeekV2LiteGGUFRuntime：meta 加载、权重放置、streaming MoE、生成
  - 确定性 assignment scatter / FP32 top-k 求和
  - CPU fallback

scripts/smoke_deepseek_v2_lite_moe.py
  - 质量/吞吐 CLI，输出结构化 JSON

tests/moe/test_gguf_index.py
tests/moe/test_gpu_quant.py
```

---

## 9. 已知限制和下一步

### 当前限制

- 单请求；没有接入 paged scheduler / continuous batching；
- 没有 fused packed-weight MoE kernel；当前是“GPU 反量化 + cuBLAS grouped bmm”；
- prefill 会扫描大量专家，短 prompt prefill 约 2.1–2.5 s；
- pinned staging 单缓冲，依赖 Event 串行复用；还没有双缓冲/多缓冲预取；
- MLA 是官方 eager/reference attention，不是 FlashMLA；
- Q4_K_M 是唯一验证过的 DeepSeek-V2-Lite GGUF 量化档；
- 没有长上下文、批量并发和严格 perplexity 回归；
- 只验证 Windows + Python 3.10 + torch 2.5.1+cu121。

### 下一步优先级

1. **fused Q4_K/Q5_0/Q8_0 MoE GEMV/GEMM**：把 dequant 与 dot-product 融合，
   去掉 materialize FP16 expert 的 HBM 往返；
2. **pinned double-buffer + 多 copy stream**：在等待当前层 bmm 时预取下一
   chunk/下一层可预测专家；
3. **专家缓存**：按层缓存热门 expert 的 packed 或 dequantized 权重；
4. **接入 paged scheduler**：把 MoE 层接入已有 continuous batching / KV 预算；
5. **量化质量回归**：perplexity、MMLU、GSM8K、长链推理，与 llama.cpp 同
   prompt / 同采样设置对照；
6. **更激进的 CPU/GPU 切分**：把部分层或部分 expert 放在 CPU AVX2 kernel，
   GPU 只算 attention + 热门 expert；
7. **CUDA Graph / prefill chunk pipeline**：降低 Python/launch 开销，保持确定性。

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

当前新增 5 个 dense 测试，`tests/moe` 共 11 个测试，全套 70 个测试通过。

必须说明：本机没有 dense-only DeepSeek-V2 GGUF，因此 dense-only 的端到端
吞吐和质量没有像 Q4_K_M MoE 那样做真实 benchmark；“dense/hybrid 层可加载、
可回退、CPU MLP 数值正确”有单元测试，但没有冒充成完整 dense 模型基准。
dense-only 超出显存时回退 CPU FP32 也会很慢，这是后续可以继续做 dense
expert-style streaming 的地方。

## 11. 一句话结论

这次最重要的不是“4 GB 跑 15.7B”这个标题，而是把 **MoE 的路由、专家流式读取、
GPU 反量化、pinned H2D、确定性 scatter、MLA attention 和生成循环** 真正接进了
DeepInfer 的执行层，并在 4 GB RTX 3050 上用 Q4_K_M 跑出了 **2.40–2.70 tokens/s**
的可复现结果；速度还不是 llama.cpp CPU 的对手，但它是 native 的、可控的、
可以继续往下做 fused kernel 和调度的基线。
