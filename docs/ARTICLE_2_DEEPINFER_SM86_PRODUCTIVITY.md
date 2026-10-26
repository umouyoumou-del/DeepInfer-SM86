# DeepInfer SM86 个人生产力：3B/7B 常驻网关、OpenAI API 与 Ollama 基准

> 系列下篇。上篇是 `docs/ARTICLE_1_DEEPINFER_SM86_KERNEL.md`，讲 W4A16/W3A16/GQA、静态 cache、CUDA Graph、7B 容量和 Q4。
>
> 本文只回答一个问题：**怎样把 DeepInfer SM86 从跑分内核变成每天可以调用的个人生产力工具。**

> 术语约定：本文中的 **native** 一律指 **DeepInfer SM86 native**
> （DeepInfer model_executor / W4A16 / W3A16 / GQA / CUDA Graph），
> 不是 Ollama、llama.cpp、Transformers 兼容层或第三方 native engine。

## 1. 产品形态：本地 OpenAI 兼容网关

跑分解决的是“能不能快”，个人生产力解决的是“每天会不会用”。

最终增加了一层本地网关：

```text
Chatbox / Obsidian / VS Code / 浏览器插件
                │
                ▼
      http://127.0.0.1:18000/v1
                │
                ▼
        apps/server.py
        /v1/models
        /v1/chat/completions
                │
                ▼
    3B W4A16 / 7B mixed Q4/Q3
    常驻模型 + 256-slot static KV + CUDA Graph
```

### 1.1 已实现

- OpenAI 兼容的 `/v1/models`；
- `/v1/chat/completions`；
- SSE 流式输出；
- EOS 停止；
- 256 token context；
- 单请求串行与显存保护；
- `apps/ask.py` CLI；
- `desktop/quick_ask.ps1` 剪贴板总结/改写/翻译/待办；
- 建议用 PowerToys 绑定 `Win+Alt+Q` 等快捷键。

### 1.2 实测

在 4 GB 3050、后台还有 MaiBot 占用显存的情况下：

| 指标 | 结果 |
|---|---:|
| prefill | 约 5.8–6.3 秒 |
| decode | 约 25.7 tokens/s |
| 64-token 请求总延迟 | 约 8.4 秒 |
| 24-token 快捷回答 | 约 7.3 秒 |

时间主要花在 prefill。对于“选中一段文字，总结/改写/翻译”这种短上下文任务，
这是可用水平；对于整篇长文，需要分段 map-reduce。

### 1.3 这一层新增的弯路

#### 512 context 退化

直接测试 512-slot GQA kernel 时 cosine 为 1.0；但接进 7B runtime 后，
256/512 context 的输出出现重复和 `!`。最终定位到 512-thread 路径在集成
环境下不稳定，而 256-slot / 256-thread kernel 输出正常。

处理方式不是把错误路径包装成“能用”，而是**收口到 256 context**，并在
README 和结果中明确标注。512-slot 只保留为后续研究项。

#### UTF-8 流式重复

Qwen 的 byte-level BPE 会在一个 UTF-8 汉字中间切断 token。直接对每个
token prefix 做 `decode` 会得到 `�`，下个 token 补齐后又会把整段文本重新
输出，造成重复。

修复方式：

- 流式时截断第一个 `U+FFFD` 之后的内容；
- 只发送已经稳定的文本前缀；
- 等下一个 token 补齐后再发送增量；
- 非流式则收集完整 token IDs，最后一次性 `decode`。

#### SSE 不关闭

最初发送 `[DONE]` 后仍保持 keep-alive，`curl` 会一直等待。改为
`Connection: close` 并在 `[DONE]` 后显式关闭连接。

#### 端口 8000 被占用

本机 8000 已被 MaiBot 使用。网关默认改到 `18000`，避免“服务启动成功但
请求打到别的进程”。

#### PowerShell 中文乱码

Windows PowerShell 5.1 会把无 BOM 的 UTF-8 脚本按 ANSI 读取，中文
`ValidateSet` 直接语法错误。`desktop/quick_ask.ps1` 必须保存为
UTF-8 with BOM。

### 1.4 当前边界

- 网关仍然只是 model_executor 级原生模型的 HTTP 包装；
- `full_engine=false`；
- 256 context；
- 单请求串行；
- 没有 RAG、工具调用、邮件/日历写操作；
- 3B 已通过 `--backend w4a16` 接入常驻网关，详见第2节；
- 3-bit 量化仍是有损的。

但至少现在，这个成果不再只是 `doc/results/*.json` 里的数字，而是可以：

1. 作为 OpenAI 兼容服务接入现有客户端；
2. 用剪贴板快捷键处理每天遇到的文字；
3. 在断网、本地、4 GB 显卡上完成总结、改写、翻译和待办提取。

---

## 2. 3B DeepInfer SM86 适配与常驻网关

### 2.1 3B 不能每次启动都重新量化

原来的 3B 原生 harness 每次启动都要：

```text
加载 3B dense 权重 + merge LoRA（CPU，约 5 分钟）
→ 搬到 GPU
→ 逐层转 W4A16
```

这只能做测速，不能做常驻服务。

新增：

```text
scripts/export_qwen3b_w4_checkpoint.py
deepinfer/model_executor/models/qwen2_w4a16.py::from_checkpoint
```

导出一个分层 checkpoint：

```text
D:\Qwen3B-W4A16-SM86
  embed.safetensors
  norm.safetensors
  layer_00.safetensors
  ...
  layer_35.safetensors
  manifest.json
```

大小 **1.912 GiB**。  
加载时间从约 5 分钟降到 **11.75 秒**。

3B 是 tied embedding，checkpoint 只保存一份 embedding；lm_head 共享它。

### 2.2 3B 网关结果

启动：

```powershell
python apps\server.py `
  --checkpoint D:\Qwen3B-W4A16-SM86 `
  --backend w4a16 `
  --model-id deepinfer-sm86-3b-w4a16 `
  --max-cache 256 `
  --port 18001
```

同一协议（system + 两个中文 prompt、max_tokens=64、temperature=0、两轮）：

| 指标 | 3B DeepInfer SM86 native |
|---|---:|
| prefill 中位 | 0.4559 秒 |
| decode 中位 | **44.773 tokens/s** |
| 一次请求 wall 中位 | **0.8948 秒** |
| completion 中位 | 19 tokens |

### 2.3 和 Ollama 3B 对比

Ollama 使用 `maibot-catgirl-v3:latest`，Q4_K_M，4096 context：

| 指标 | 3B DeepInfer SM86 native | Ollama 3B |
|---|---:|---:|
| prefill | 0.4559 s | **0.0603 s** |
| decode | 44.773 t/s | **57.285 t/s** |
| wall | 0.8948 s | **0.4637 s** |
| context | 256 | **4096** |

结论：

- 3B DeepInfer SM86 native 已经进入个人快捷操作可用区间；
- Ollama 在 3B 上仍然更快，主要优势在 prefill 和完整 engine；
- DeepInfer 的 W4A16 decode 本身并不慢，差距在服务层。

### 2.4 7B 当前环境 paging

桌面后台约 500 MiB GPU 占用时，7B 网关 prefill 超过 120 秒仍无法完成。  
用 DeepInfer SM86 native harness 在当前环境测得：

```text
prefill: 53.089 秒
decode:  25.141 tokens/s
峰值 allocated: 3.196 GiB
峰值 reserved: 3.357 GiB
```

同一 7B checkpoint 在更干净环境下是：

```text
prefill: 3.061 秒
decode:  25.962 平均 / 26.188 中位 tokens/s
```

所以 7B 不是 kernel 退化，而是 **4 GB + 后台 GPU 占用 + WDDM paging**。  
当前环境下的 7B 生成还出现退化，不能把该文本当作质量结果。

### 2.5 新的弯路

- 3B dense 加载本身就要约 5 分钟，不能直接做常驻；
- 3B tied embedding 如果重复保存 lm_head 会浪费0.6 GB；
- 7B 网关在前台 HTTP + static cache + CUDA Graph + 后台 GPU 占用下，
  prefill 从3秒变成53秒以上；
- 3B/7B 不能同时常驻4 GB；
- 512 context 仍然没有交付，当前上限是256。

### 2.6 最终建议

| 场景 | 选择 |
|---|---|
| 剪贴板总结/改写/翻译/待办 | 3B DeepInfer SM86 native，约0.9秒 |
| 想用最快3B | Ollama Q4_K_M，约0.46秒 |
| 更高质的短任务 | 7B DeepInfer SM86 native，但必须隔离显存 |
| 长上下文 | 当前 DeepInfer SM86 256 context 不足，Ollama 4096 更合适 |
| 生产级 | 仍需 CacheEngine、Sampler、continuous batching、tiled prefill |

三后端原始数据：

```text
docs/results/productivity_3b_only.json
docs/results/productivity_ollama_only.json
docs/results/qwen7b_mixed_head3_final.json
docs/results/qwen7b_current_paging_harness.json
docs/results/productivity_backends_combined.json
```

完整报告：

```text
docs/PRODUCTIVITY_BENCHMARK_3B_7B_OLLAMA.md
```

---

## 3. 3B / 7B / Ollama 基准：协议与原始数据

### 3.1 为什么要先定协议

将 3B DeepInfer SM86 native、7B DeepInfer SM86 native、Ollama 放在一起比较时，最容易犯的错误是：

- 一个用 24 token，另一个用 64 token；
- 一个带 system prompt，另一个不带；
- 一个把模型加载算进 decode，另一个不算；
- 一个看 `eval_duration`，另一个看 HTTP wall；
- 模型温度、top_p、seed 不一致；
- 3B 和 7B 直接拿质量做排名。

因此这次先固定协议，再跑数据。

### 3.2 统一协议

```text
system:
你是一个简洁、可靠的中文个人助理。只输出最终答案，不要复述任务。

prompt 1:
用一句话解释 CUDA Graph。

prompt 2:
把下面这段话压缩成一句待办：
明天下午三点前把实验报告发给导师，并附上结果表格。

max_tokens  = 64
temperature = 0
top_p       = 0.95
top_k       = 50
seed        = 20261026
warmup      = 1
repeats     = 2 × 2 prompts
```

3B DeepInfer SM86 native 与 Ollama 的 tokenizer 都报告：

```text
prompt 1 = 38 tokens
prompt 2 = 58 tokens
```

这说明两条路径确实用了同一组输入。

### 3.3 3B DeepInfer SM86 native 的逐轮数据

| 轮次 | prompt | wall | prefill | decode | completion | decode tps |
|---|---|---:|---:|---:|---:|---:|
| 0 | CUDA Graph | 0.863216 s | 0.456009 s | 0.400732 s | 18 | 44.918 |
| 0 | 待办压缩 | 1.004158 s | 0.549768 s | 0.448161 s | 20 | 44.627 |
| 1 | CUDA Graph | 0.861360 s | 0.454701 s | 0.399181 s | 18 | 45.092 |
| 1 | 待办压缩 | 0.926476 s | 0.455816 s | 0.450063 s | 20 | 44.438 |

中位：

```text
prefill = 0.4559 s
decode  = 44.773 tokens/s
wall    = 0.8948 s
```

### 3.4 Ollama 3B 的逐轮数据

| 轮次 | prompt | wall | prefill | decode | completion | decode tps |
|---|---|---:|---:|---:|---:|---:|
| 0 | CUDA Graph | 0.439561 s | 0.017359 s | 0.257574 s | 15 | 58.236 |
| 0 | 待办压缩 | 0.499255 s | 0.070977 s | 0.276193 s | 16 | 57.930 |
| 1 | CUDA Graph | 0.485539 s | 0.114966 s | 0.264832 s | 15 | 56.640 |
| 1 | 待办压缩 | 0.441871 s | 0.049554 s | 0.283564 s | 16 | 56.425 |

中位：

```text
prefill = 0.0603 s
decode  = 57.285 tokens/s
wall    = 0.4637 s
```

### 3.5 7B 的两组数据

隔离环境：

```text
prefill = 3.061 s
decode  = 25.962 平均 / 26.188 中位 tokens/s
峰值 allocated = 3.196 GiB
峰值 reserved  = 3.357 GiB
```

当前后台约 500 MiB GPU 占用时：

```text
prefill = 53.089 s
decode  = 25.141 tokens/s
峰值 allocated = 3.196 GiB
峰值 reserved  = 3.357 GiB
```

当前 paging 文本：

```text
CUDA Graph是一种编译时<decltypeoriouslylopenngen<decltypeoriouslylopenngen>>>>>>>>来优化计算图以
```

这不是一个可接受的质量输出，因此不能拿它做质量结论。

---

## 4. 3B DeepInfer SM86 的实现细节

### 4.1 以前的 3B 启动路径

```text
加载 3B BF16 分片
→ merge PEFT LoRA
→ CPU dense 模型
→ GPU float16
→ 逐层量化 W4A16
→ 创建静态 cache
→ 创建 CUDA Graph
```

这个流程有两个问题：

1. CPU 加载和 LoRA merge 约 5 分钟；
2. 每次启动都重复做一次转换。

作为测速 harness 可以，作为个人网关不行。

### 4.2 分层 checkpoint

新增导出脚本：

```text
scripts/export_qwen3b_w4_checkpoint.py
```

它做的事情：

```text
Qwen2ForCausalLM.load_model(..., device="cpu", adapter_path=...)
→ Qwen2W4A16ForCausalLM.from_dense_model(..., device="cuda")
→ 每层保存 packed/scales/bias + norm
→ 单独保存 embed + norm
→ 写 manifest
```

产物：

```text
D:\Qwen3B-W4A16-SM86
  config.json
  tokenizer.json
  tokenizer_config.json
  manifest.json
  embed.safetensors
  norm.safetensors
  layer_00.safetensors
  ...
  layer_35.safetensors
```

总大小：

```text
1.912 GiB
```

### 4.3 tied embedding

3B 配置是：

```json
"tie_word_embeddings": true
```

因此：

- `embed_tokens.weight` 和 `lm_head.weight` 是同一个张量；
- checkpoint 只保存一份 `embed.safetensors`；
- loader 不重复分配 lm_head；
- 如果重复保存，会多浪费约 0.6 GiB。

### 4.4 loader

新增：

```text
Qwen2W4A16ForCausalLM.from_checkpoint(...)
```

流程：

1. 读取 Qwen2 config；
2. 在 CUDA 上构造空白 W4A16 模型；
3. 加载 dense embedding；
4. 加载 norm；
5. 逐层加载 q/k/v/o + gate/up/down 的 packed/scales/bias；
6. 绑定 CuPy 零拷贝视图；
7. 挂上 `native_gqa_decode`；
8. 进入 eval 模式。

加载时间：

```text
约 11.75 秒（第一次服务器启动）
```

### 4.5 网关

3B：

```powershell
python apps\server.py `
  --checkpoint D:\Qwen3B-W4A16-SM86 `
  --backend w4a16 `
  --model-id deepinfer-sm86-3b-w4a16 `
  --max-cache 256 `
  --port 18001
```

7B：

```powershell
python apps\server.py `
  --checkpoint D:\Qwen7B-Mixed-SM86 `
  --backend mixed `
  --model-id deepinfer-sm86-7b-mixed `
  --max-cache 256 `
  --port 18000
```

4 GB 上两者不能同时常驻：

```text
3B 约 2.0–2.5 GiB
7B 约 3.0–3.4 GiB
```

---

## 5. 服务化阶段走过的弯路

这一节不只列“最后怎么解决”，还记录**当时为什么会判断错**。

### 5.1 把兼容层速度当成 DeepInfer 原生速度

**错误判断：**

> NF4 只有 5.9 tokens/s，所以 DeepInfer 的 Qwen 路径也很慢。

**实际情况：**

NF4 是 Transformers + bitsandbytes 的兼容层，不是 DeepInfer 的 W4A16 DeepInfer SM86 native
路径。3B 原生后来到 42.7 tokens/s 稳态。

**教训：**

比较之前先写清楚是“兼容层、Ollama、source 模型还是 DeepInfer SM86 native kernel”。

### 5.2 以为纯 4-bit 7B 可以塞进 4 GB

**错误判断：**

> 4-bit 就是 0.5 byte/weight，7B 应该只有 3.5 GB 左右，也许刚好能跑。

**实际情况：**

- 7.615B 参数；
- 加 group128 scales 后约 3.657 GiB；
- 可用显存只有 3.237 GiB；
- 还没算 KV、激活和 Graph。

最终全 Q4 能跑，但 prefill 86 秒、decode 7.9 tokens/s。

**教训：**

先算权重 + scales + KV + 激活，再看“能不能跑”。

### 5.3 加载后再融合 qkv/gate-up

**错误判断：**

> 先按普通层加载，再替换成 fused 模块，代码更简单。

**实际情况：**

旧模块和 fused 模块同时存在，PyTorch allocator 出现碎片：

```text
reserved 峰值 4.811 GiB
decode 掉到 3.9 tokens/s
```

**修复：**

在模型构造期直接创建 fused qkv / gate-up，checkpoint 直接往 fused 切片加载。

**教训：**

4 GB 边缘上，allocator 碎片和算法一样致命。

### 5.4 CUDA Graph capture 不等于执行

**错误判断：**

> capture 完之后直接读输出就行。

**实际情况：**

capture 只记录自定义 CUDA kernel，不执行；必须 replay 一次。

**修复：**

```python
graph.replay()
torch.cuda.synchronize()
```

**教训：**

Graph 的“第一次正确性”必须单独验证。

### 5.5 DLPack 在 capture 中报 stream 错误

**报错：**

```text
operation would make the legacy stream depend on a capturing blocking stream
```

**原因：**

`cp.from_dlpack(torch_tensor)` 会触发 stream wait event。

**修复：**

使用 `cp.cuda.UnownedMemory` 零拷贝视图。

### 5.6 GQA scale 变成 0

**现象：**

DeepInfer SM86 native GQA 能跑，但输出全错，softmax 近似均匀。

**原因：**

Python float 传给 RawKernel 的 `float` 参数被解释成 0。

**修复：**

```python
cp.float32(1.0 / math.sqrt(head_dim))
```

### 5.7 3-bit 打包 row padding 错位

**现象：**

3-bit kernel 余弦约 `-0.068`。

**原因：**

每行多放了一个 padding byte，但 kernel 按紧密 row stride 读取。

**修复：**

只在整块 tensor 末尾放一个 padding byte。

### 5.8 量化 embedding/head 时 CPU OOM

**现象：**

一次性把 545M 参数转 float32，RAM 不够。

**修复：**

按 8192 行分块量化，写 `finish_qwen7b_mixed_quantize.py`。

### 5.9 Python 3.12 CUDA wheel 解压卡死

**现象：**

`torch_cuda.dll` 解压超过 20 分钟。

**实际选择：**

使用 Python 3.10 CUDA harness 加载同一份 DeepInfer source。

**代价：**

这不是最终 Python 3.12 package 验证；文章和 README 都必须明确这一点。

### 5.10 512 context 的“直接 kernel 正确”不等于“runtime 正确”

**错误判断：**

> 512-slot GQA kernel 与 SDPA 的 cosine 是 1.0，所以 512 context 可以交付。

**实际情况：**

- 直接 kernel 测试 cosine 1.0；
- 但接进 7B runtime 后，256/512 context 输出重复和 `!`；
- 512-thread 路径在集成环境不稳定；
- 256-slot / 256-thread 路径输出正常。

**修复：**

收口到 256 context，不把错误路径包装成“能用”。

**教训：**

单 kernel 正确性 ≠ 模型级正确性。

### 5.11 7B 网关 paging 被低估

**错误判断：**

> 7B 混合模型在 4 GB 上能跑到 26 tokens/s，网关也应该一样。

**实际情况：**

后台约 500 MiB GPU 占用时：

- 7B 网关 prefill 超过 120 秒；
- 7B harness 当前环境 prefill 53.089 秒；
- 同一 checkpoint 在干净环境 prefill 3.061 秒；
- paging 下输出还会退化。

**修复：**

- 3B 作为默认；
- 7B 只在高价值短任务、显存隔离时使用；
- 文档同时保留“隔离最佳”和“当前 paging”两组结果，不混在一起。

### 5.12 3B 每次启动重新转换

**错误判断：**

> 先把 3B 原生 harness 跑通，再考虑服务化。

**实际情况：**

dense 加载 + LoRA merge + W4A16 转换约 5 分钟，根本不能做常驻。

**修复：**

导出 1.912 GiB 的分层 checkpoint，加载降到 11.75 秒。

### 5.13 清理 Ollama 时把基准环境弄坏

**现象：**

基准脚本报：

```text
WinError 10061 目标计算机积极拒绝
```

**原因：**

为了释放显存，把 Ollama 服务进程停掉了，但没有重新启动。

**修复：**

```powershell
ollama serve
```

再跑 `/api/chat`。

**教训：**

“释放显存”和“保持服务可用”要分开操作。

### 5.14 超过 4 GB 后仍试图让 7B 网关完成

**现象：**

7B server 的 warmup 请求：

```text
prefill_seconds = 147.510645
```

第二请求继续 paging。

**错误做法：**

继续等待，希望第一次之后会变快。

**正确做法：**

记录 paging 事实，改用 DeepInfer SM86 native harness 获取可完成的当前数据，同时保留干净环境的
历史 25.962/26.188 结果，不伪造“同协议”数字。

### 5.15 仓库 `.gitignore` 把源码目录也忽略了

**现象：**

GitHub 上只有文档、脚本和 `.cu` 文件，没有：

```text
deepinfer/model_executor/models/qwen2.py
deepinfer/model_executor/models/qwen2_w4a16.py
deepinfer/model_executor/models/qwen2_mixed_sm86.py
```

**原因：**

`.gitignore` 写了 `models/`，把 `deepinfer/model_executor/models/` 也匹配掉了。

**修复：**

```text
/models/
/checkpoints/
```

只忽略仓库根目录的权重目录。

**教训：**

推送后不能只看 README 的 200 响应，还要检查 tree 是否包含真正的源码。

**当前状态：**

这个修复和源码仍然在本地未提交状态；远端还没有新的内核提交。

### 5.16 PowerShell 中文脚本编码

**现象：**

```text
Missing closing ')' in expression
```

**原因：**

Windows PowerShell 5.1 把无 BOM 的 UTF-8 脚本按 ANSI 读取，中文
`ValidateSet` 直接语法错误。

**修复：**

`desktop/quick_ask.ps1` 保存为 UTF-8 with BOM。

### 5.17 SSE 不关闭导致 curl 一直等待

**现象：**

流式内容已经收完，但 `curl` 不退出。

**原因：**

发送 `[DONE]` 后仍然保持 keep-alive。

**修复：**

```text
Connection: close
self.close_connection = True
```

### 5.18 UTF-8 流式重复

**现象：**

流式文本出现：

```text
... CUDA �CUDA Graph 是一种...
```

**原因：**

byte-level BPE 在一个汉字中间切断 token，直接 decode 前缀得到 `�`，下一 token
补齐后又重复输出整段。

**修复：**

- 只发送 `U+FFFD` 之前的稳定前缀；
- 等下一个 token 补齐；
- 非流式收集完整 token IDs 后一次 decode。

---

## 6. 哪些结论可以说，哪些不能说

### 可以说

- 3B DeepInfer SM86 W4A16 在 RTX 3050 4 GB 上达到约 44.8 tokens/s；
- 3B DeepInfer SM86 短任务一次请求约 0.9 秒；
- 7B DeepInfer SM86 mixed 在隔离环境约 26 tokens/s；
- Ollama 3B 在相同输入下约 57.3 tokens/s；
- 当前 DeepInfer SM86 网关 context 是 256，Ollama 是 4096；
- 4 GB 上 3B/7B 不能同时常驻；
- 全 Q4 7B 能跑，但只有约 7.9 tokens/s。

### 不能说

- “DeepInfer 已经全面超过 Ollama”；
- “7B 在 4 GB 上可以稳定生产”；
- “3-bit MLP/head 与 BF16 等价”；
- “当前是完整 DeepInfer engine”；
- “512 context 已经交付”；
- “7B paging 下的退化文本是模型质量问题”；
- “兼容层 5.9 tokens/s 是 DeepInfer SM86 native 性能”。

---

## 7. 复现清单

### 7.1 3B DeepInfer SM86 checkpoint

```powershell
python scripts\export_qwen3b_w4_checkpoint.py `
  --model F:\QwenQLoRA\models\Qwen2.5-3B-Instruct-smallshards `
  --adapter F:\QwenQLoRA\train\maibot-catgirl-v3\checkpoint-120 `
  --output D:\Qwen3B-W4A16-SM86
```

产物：

```text
D:\Qwen3B-W4A16-SM86
约 1.912 GiB
```

### 7.2 3B DeepInfer SM86 网关

```powershell
python apps\server.py `
  --checkpoint D:\Qwen3B-W4A16-SM86 `
  --backend w4a16 `
  --model-id deepinfer-sm86-3b-w4a16 `
  --max-cache 256 `
  --port 18001
```

### 7.3 7B DeepInfer SM86 网关

```powershell
python apps\server.py `
  --checkpoint D:\Qwen7B-Mixed-SM86 `
  --backend mixed `
  --model-id deepinfer-sm86-7b-mixed `
  --max-cache 256 `
  --port 18000
```

### 7.4 三后端基准

```powershell
python scripts\bench_productivity_backends.py
```

或：

```powershell
python scripts\bench_productivity_backends.py --skip-7b
python scripts\bench_productivity_backends.py --skip-3b --skip-7b
```

### 7.5 源码测试

```powershell
D:\QwenNativeTest-venv\Scripts\python.exe -m pytest `
  tests\models\test_qwen2_source_integration.py
```

结果：

```text
6 passed
```

### 7.6 Ollama 服务

如果基准报：

```text
WinError 10061 目标计算机积极拒绝
```

先启动：

```powershell
ollama serve
```

---

## 8. 3B / 7B / Ollama 怎么选

| 场景 | 选择 | 理由 |
|---|---|---|
| 剪贴板总结/改写/翻译/待办 | 3B DeepInfer SM86 | 约 0.9 秒，本地、离线 |
| 最快 3B | Ollama Q4_K_M | 约 0.46 秒 |
| 更好的短文本质量 | 7B DeepInfer SM86 mixed | 但必须隔离显存 |
| 长上下文 | Ollama 4096 | DeepInfer SM86 网关当前只有 256 |
| 生产级 | 继续做 engine | 单请求、短上下文、早期 runtime |

如果只保留一句话：

> **3B DeepInfer SM86 已经可以每天用；7B DeepInfer SM86 还是“边缘可用”；Ollama 仍然是 3B 小模型最成熟的基线。**

---

## 9. 当前仓库与发布状态

本地工作区：

```text
D:\DeepInfer-SM86
```

远端：

```text
https://github.com/umouyoumou-del/DeepInfer-SM86
本次更新前 HEAD = 29ddbea；本次提交后以 git log 为准
```

当前本地状态：

- 3B W4A16 checkpoint、3B backend、256 GQA、基准脚本、报告全部在本地；
- `.gitignore` 已修复，`deepinfer/model_executor/models/` 源码已在本地；
- 3B 分层 checkpoint 在 `D:\Qwen3B-W4A16-SM86`；
- 7B 混合 checkpoint 在 `D:\Qwen7B-Mixed-SM86`；
- **这些最新改动已整理进本次提交并上传**；
- 远端还缺少 3B/基准这一版，也没有完整的 `deepinfer/model_executor/models/` 源码；
- 本次提交包含 3B、基准、源码修复和两篇拆分文章；模型权重仍不在仓库中。

这不是“已经全部上云”的状态。  
这篇文章的价值之一，就是把“做出来了”“能跑”“已经推送”三件事分开写清楚。

---

## 系列文章

- **上篇**：`docs/ARTICLE_1_DEEPINFER_SM86_KERNEL.md`  
  DeepInfer SM86 内核、量化、显存预算和生成正确性。
- **下篇（本篇）**：3B/7B 常驻网关、个人生产力入口和 Ollama 基准。

> 内核细节看上篇；服务、交互、基准和发布状态看本篇。
