# 从固定槽位到分页连续批处理：DeepInfer-SM86 engine-core 改造复盘

> 日期：2026-10-27
> 目标硬件：RTX 3050 Laptop 4 GB / SM86 / Windows
> 定位：`model_executor` 之上的 engine-core 原型，不是完整 DeepInfer engine
> 本文性质：工程复盘。重点讲清“做了什么”“为什么这样做”“哪些路走错了”，
> 不是性能榜单。

## 1. 结论先行

改造前的 DeepInfer-SM86 是一个很有价值的垂直优化样本：

- W4A16 / W3A16 GEMV；
- native GQA decode；
- dense static KV cache；
- CUDA Graph；
- 固定槽位动态批处理；
- Qwen2.5-3B / 7B 在 4 GB 上的实测数据。

但它离“通用推理引擎”还缺三层基础设施：

1. 动态批处理和请求调度；
2. 分页 / 弹性 KV cache 与前缀复用；
3. 更广的模型、attention mask 和算子覆盖。

这次改造优先补这三层，并顺带补上网关指标、队列治理和请求取消。

最终得到的是一个可运行的 engine-core 原型：

```text
ModelSpec
  ├─ Qwen2 / Llama / Mistral / Gemma / Gemma2 adapters
  └─ normalized model config

PagedKVCacheManager
  ├─ reference-counted block allocator
  ├─ block table + tensor-only validity mask
  └─ full-block immutable prefix cache with LRU eviction

PagedBatchScheduler
  ├─ bounded waiting queue
  ├─ variable decode batch size
  ├─ per-request OOM / cancellation isolation
  └─ Prometheus metrics

OpenAI-compatible gateway
  ├─ /v1/chat/completions
  ├─ /health
  └─ /metrics
```

它仍然不是 vLLM 或 TensorRT-LLM 的替代品。准确地说：

> 它从“给 Qwen2.5 修的一条高性能跑道”，变成了“有一段可运行的分页调度环路
> 和更多车型适配”的 engine-core 原型。

## 2. 为什么先做这三层

在 4 GB 显存上，单纯继续堆 kernel 优化会很快遇到两个物理上限：

- 权重本身占掉了大部分显存；
- 固定 KV cache 按 `max_cache` 预留，短请求也在支付长上下文的成本。

因此这次没有先做更多孤立 kernel，而是先解决“显存怎么按需分配”和“多个请求
怎么一起跑”：

```text
单请求性能
    ↓
固定槽位并发（已有）
    ↓
分页 KV + 可变 batch（本次）
    ↓
chunked prefill / native paged attention（下一步）
```

这也符合当前最迫切的三块缺口：动态批处理、分页/弹性 KV、模型与算子覆盖。

## 3. 做了什么

### 3.1 分页 KV cache

新增 `deepinfer/engine/kv_cache.py`。

物理池只分配一次：

```text
key/value:
  [num_layers, num_blocks, num_kv_heads, block_size, head_dim]
```

每个序列维护逻辑 block 列表和 block table：

```text
block_table: [batch, max_blocks_in_batch]
seq_lens:    [batch]
```

decode 写入：

```text
physical_block = block_table[b, position // block_size]
offset         = position % block_size
pool[physical_block, :, offset, :] = new_kv[b]
```

attention 前按当前 batch 内最大长度 gather 成临时 dense K/V，并生成：

```text
valid[b, t] = t < seq_lens[b] + 1
```

关键点是 mask 生成全部由 tensor 完成，没有 `.item()`。

### 3.2 引用计数 block allocator

block 的所有权是显式的：

```text
allocate(n) -> refcount = 1
retain(ids) -> refcount += 1
free(ids)   -> refcount -= 1，归零后回到 free list
```

这样 prefix cache 和请求可以同时引用同一个物理 block，而不会因为任意一方退出
就提前释放。

### 3.3 Prefix cache

只缓存**完整且不可变**的 prompt block。累计 hash：

```text
hash_i = cumulative_hash(tokens[0 : (i+1)*block_size])
```

请求 prefill 时匹配最长完整 block 前缀，读取旧 K/V 作为动态 prefill cache 的
前缀，只对 suffix 执行前向。新生成的完整 block 才注册进 cache，尾部不完整
block 不注册。

这是一个刻意保守的设计：它换来了“不需要 copy-on-write”的简单性。

### 3.4 可变 batch 连续批处理

新增 `apps/paged_scheduler.py`。调度循环大致是：

```text
while not stopped:
    admit queued requests while running < max_num_seqs
    if running:
        ensure next-token blocks
        pack active sequences into one batch
        one batched decode forward
        per-request sampling / EOS / metrics
        release finished blocks immediately
    else:
        sleep 10ms
```

和旧的固定槽位实现相比：

| 项目 | 旧 static scheduler | 新 paged scheduler |
|---|---|---|
| batch 大小 | 固定 `max_batch_size` | 当前 running 数，1..max_num_seqs |
| KV 容量 | 每 slot 固定 `max_cache` | 按 token 需求分块分配 |
| 请求加入 | 等固定 slot 空出 | 有 block 即可加入 |
| 请求结束 | 清空整槽 | 立即归还 block |
| 前缀复用 | 无 | 完整 block 共享 |
| 动态 batch 下 CUDA Graph | 固定 shape 时可用 | 不可用 |
| 长上下文 | 受 native GQA 256 限制 | paged 路径可超过 256 |
| 指标 / 取消 | 基础 | Prometheus + stream cancel |

### 3.5 Attention mask 与模型覆盖

通用 attention 现在统一处理：

- 绝对 position 的 causal mask；
- paged validity mask；
- sliding-window mask；
- 外部 attention mask；
- dense static cache / dynamic cache / paged cache 三种 cache。

模型覆盖从 Qwen2 + Llama 扩到：

- Mistral：全层 sliding-window GQA；
- Gemma：GeGLU、offset RMSNorm、embedding scale；
- Gemma 2：sandwich norm、attention/final logit softcap、
  alternating sliding/full attention、`query_pre_attn_scalar`。

W4A16 分层 checkpoint 的导出和加载也同步支持 Gemma 2 额外的
`pre_feedforward_layernorm` / `post_feedforward_layernorm`。

### 3.6 网关治理

- 新增 `GET /metrics`，输出 Prometheus 文本；
- `/health` 附带 scheduler、block、prefix cache 和 metrics 摘要；
- 请求队列有上限；
- 流式连接断开时触发取消；
- finish reason 区分 `stop` / `length`。

### 3.7 基准脚本

新增 `scripts/bench_paged_scheduler.py`，比较同一组 prompt 的：

- 串行提交；
- 并发提交；
- aggregate tokens/s；
- prefix 复用；
- block 使用量。

当前环境没有 CUDA，因此这次没有产生 GPU 数字。脚本存在的意义是让结果可以在
目标机器上复现，而不是用 CPU 测试冒充性能结论。

## 4. 验证了什么

当前 CPU 回归结果：

```text
18 passed, 1 skipped
ruff: All checks passed
```

覆盖范围：

- block allocator：引用计数、double free、OOM；
- prefix cache：retain、复用、LRU 淘汰、清理；
- paged cache：写入、追加、grow、mask materialize；
- 两个不同长度请求的 batched paged logits 与 dynamic cache 对齐；
- Gemma 2 sliding/softcap mask 下的 paged decode 与 dynamic cache 对齐；
- Qwen2 / Llama / Mistral / Gemma / Gemma 2 与 HF logits 对齐；
- paged scheduler：两个请求在同一步 decode、相同 prompt 的 prefix 复用；
- Prometheus 文本格式。

这些测试证明的是**数学路径和缓存语义**，不是 GPU 吞吐。GPU 实测量需要运行：

```powershell
python scripts\bench_paged_scheduler.py `
  --checkpoint D:\Qwen3B-W4A16-SM86 `
  --backend w4a16 `
  --max-cache 256 --max-num-seqs 4 --block-size 16 `
  --max-new-tokens 32 `
  --output docs\results\paged_scheduler_3b.json
```

## 5. 走过的弯路

下面这些不是“总结出来的漂亮经验”，而是这次实现里真实踩过的坑。

### 5.1 一开始以为“动态 batch”只是把 batch 维加上去

旧 scheduler 已经有固定槽位：两个请求可以在同一个 decode step 里执行。继续往
前走时很容易产生一个错觉：

> 把 batch 从 1 改成 2，不就完成连续批处理了吗？

实际不是。固定槽位仍然要求每个请求有完整的 `max_cache` KV 容量；两个短请求
会预占两份 256-token cache。更麻烦的是，请求动态加入/退出后：

- batch 大小会变化；
- 每个请求的 context 长度不同；
- 需要 paged/ragged attention；
- CUDA Graph 的固定 shape 假设失效；
- 新请求的 prefill 不能直接塞进已有的 graph replay。

最后必须同时引入 block allocator、block table、ragged mask 和可变 batch 调度，
而不是“只加一个 batch 维”。

### 5.2 一开始想把 paged cache 直接接 native GQA kernel

最直觉的方案是改写 `qwen2_gqa_decode`，让它直接读取 block table。但这条路的
风险很高：

- 当前 native kernel 是 batch=1、dense cache、固定 slot；
- paged kernel 要做 block gather、变长序列、ragged batch；
- 没有 CUDA 测试环境，无法验证 kernel 正确性；
- 一旦 kernel 出错，错误会伪装成“模型质量下降”，很难定位。

所以这次先实现 PyTorch SDPA reference path，把缓存语义、mask 语义和 batched
logits 对齐做扎实。native paged GQA 留到下一步在目标 GPU 上单独开发和验证。

这不是性能最优解，但它避免了一个更糟的结果：**为了追求 kernel 数量，把没有
验证过的错误路径写进主链路**。

### 5.3 一开始每一个 layer 都重建 block table

第一版 `make_views()` 对每个 layer 都做：

```python
table = torch.full((batch, max_blocks), -1, ...)
# 填充 block ids
```

对于 7B 模型，28 层就是 28 份相同的 block table，每 token 都重新分配。正确性
没问题，但 decode 环里会多出大量细碎 tensor 分配。

后来发现 block table 和 `seq_lens` 在一次 decode step 内是只读的，于是改成：

```text
构造一次 table/lengths
    ↓
所有 layer 共享同一份只读 tensor
```

这是典型的“先用正确但笨的版本，再让 profile 决定优化点”。

### 5.4 一开始在 paged update 里做负数 block 检查

为了防御“scheduler 忘记分配 block”，曾在 `PagedLayerKVCache.update()` 里写：

```python
if bool((block_ids < 0).any()):
    raise RuntimeError(...)
```

这行代码很安全，但它会在每个 layer 触发一次 GPU→CPU 同步。每 token 28/36 次
同步，足以把 paged 路径的收益吃掉。

最后把检查改到 scheduler 的 block 分配逻辑里，并在注释中明确“视图构建时
block 必须已经存在”的 invariant。正确性检查不能无脑塞进高频 kernel 路径。

### 5.5 Prefix cache 的第一版想缓存所有 block，并原地追加

最初的直觉是：prompt 的 block 都缓存，decode 往最后一个 block 里写。但这样
会引入 copy-on-write：

- 两个请求共享最后一个 block；
- 一个请求继续 decode 时不能修改共享 block；
- 必须复制 block、更新 block table、处理引用计数；
- 一旦 prefix 命中和 decode 增长交错，状态机非常复杂。

最后选择更保守的规则：

1. 只缓存完整 prompt block；
2. 尾部不完整 block 不缓存；
3. 匹配 prefix 时至少保留最后一个 token 重新计算；
4. 如果 prompt 刚好 block 对齐，decode 新 token 会分配新 block。

这样已注册的完整 prompt block 永远不被原地修改，完全不需要 COW。省掉的不是
实现工作量，而是未来很难复现的一类一致性 bug。

### 5.6 Prefix 复用量的公式错过一次

一开始想直接复用所有完整 block，结果漏了一个条件：

> 必须至少留一个 token 重新做前向，才能拿到当前请求的 logits。

正确上限应该是：

```python
max_reusable_blocks = (prompt_length - 1) // block_size
```

如果 prompt 长度是 32、block size 是 16，不能复用最后两个 block；最多复用
第一个 block，用后 16 个 token 重新算出最后一个 token 的 hidden state 和
logits。

这个边界在短 prompt 和 block 对齐 prompt 上特别容易漏，测试里专门加了
prompt 长度 12 / block size 4 与 Gemma 2 的缓存路径。

### 5.7 引用计数模型的第一版测试写错了

第一次写 allocator/prefix cache 测试时，出现了两个典型的理解错误：

- `retain` 一次再把请求自己的那个引用释放掉，refcount 只从 2 回到 1，不会变成
  0，也不应该立即回到 free list；
- prefix cache 自己持有一个引用，请求退出后 block 仍然被 cache 持有，直到 LRU
  淘汰或 `clear()` 才会释放。

测试一开始把“请求不再使用”误当成“block 应该空闲”。修正测试的过程也帮助把
owner 语义写清楚：

```text
cache 的引用不是请求的引用；
free 只释放调用方持有的那一份。
```

另一个错误是测试里让 7 个 token 占用了 3 个 block，但 block size=4 时实际只
需要 2 个。实现没有错，是测试对容量的直觉错了。

### 5.8 动态 batch 和 CUDA Graph 不能同时假装都存在

旧路径可以捕获固定 shape 的 batch CUDA Graph。新 scheduler 的 batch 大小和
context 长度都在变化，如果继续使用旧 graph，会得到错误结果。

本次明确拆成两条路径：

```text
低延迟单请求 / 固定槽位：
  dense static cache + CUDA Graph

动态连续批处理：
  paged cache + SDPA + eager 调度
```

`/health` 会明确返回当前模式。不会把“static graph 可用”宣传成“paged 动态
batch 也有 graph”。

### 5.9 OOM 处理一开始想“整批失败”

paged cache 的好处是容量可以按请求分配，但代价是 OOM 可能发生在运行中。第一
版想法是：只要有一个请求分配不到 block，就让整个 decode step 失败。

这会导致一个短请求被一个长请求拖死，不符合调度器的隔离目标。后来改成：

- decode 前逐个请求检查下一个 token 需要的 block；
- 失败请求单独异常返回并释放自己的 block；
- 其他请求继续组成下一批。

还没有做 preemption / recompute / CPU swap，所以这不是完整的弹性策略，但至少
不会把一个请求的 OOM 变成全局故障。

### 5.10 Mistral 的 sliding window 不靠 `use_sliding_window` 开关

最初按 Qwen2 的经验去找 `use_sliding_window`，但 Mistral 的 HF config 默认
根本没有这个字段。Mistral 的行为是：

```text
只要 config.sliding_window 存在，attention 就使用 sliding window
```

如果在 adapter 里等 `use_sliding_window=True`，就会静默退化成 full attention。
最后在 `MistralAdapter` 中显式地把 `sliding_window` 转成归一化的
`use_sliding_window=True`，并让通用 mask 使用绝对 position 判断窗口。

### 5.11 Gemma 的 tied embedding 默认值踩坑

第一次加载 tiny Gemma checkpoint 时，模型报告缺少：

```text
lm_head.weight
```

原因是 Gemma 的 tied embedding 默认值在 HF `save_pretrained()` 后的 config 中
不一定显式写出 `tie_word_embeddings`，而 `lm_head` 可能因为 tied 而不落盘。
最初的通用 parser 用了通用默认 `False`，于是去寻找一个不存在的 `lm_head`。

修正方式是让 Gemma family 的默认 tied 语义进入 ModelSpec，而不是把默认值硬塞
进 loader。这个坑说明：**架构适配不仅是字段改名，还包括 HF 配置序列化的默认
语义。**

### 5.12 Gemma 2 不是“GeGLU 版 Gemma”

一开始把 Gemma 2 当成 Gemma 加一句 GeGLU，结果很快发现至少有五处差异：

- RMSNorm 是 `1 + weight`；
- embedding 需要乘 `sqrt(hidden_size)`；
- 每层有 pre/post feedforward sandwich norm；
- attention logits 和 final logits 各有一个 softcap；
- 奇偶层交替使用 sliding/full attention；
- attention scaling 由 `query_pre_attn_scalar` 决定。

这些差异不能靠“注册一个类名”解决。最后分别进入：

- `Qwen2RMSNorm(weight_offset=...)`；
- `Qwen2DecoderLayer(sandwich_norm=True)`；
- attention 的 `sliding_window_pattern`；
- 显式 eager attention 的 logit softcap 路径；
- `query_pre_attn_scalar` 对应的 scaling。

在 tiny HF checkpoint 上做 logits 对齐后，才把它写进能力矩阵。

### 5.13 指标计数也重复了一次

第一版在请求完成时统计全部 generated tokens，同时 decode step 里又统计了一次，
于是吞吐指标被重复计算。后来改成：

```text
prefill 采样出的第一个 token：admit 时计数
后续 decode 每产生一个 token：decode step 内计数
finish 时只统计请求完成数
```

指标看起来是外围功能，但错误指标会直接误导调度和容量决策，所以也必须像
kernel 一样做回归。

### 5.14 “prefill 也一起批处理”没有贸然做

最自然的下一个优化是 chunked prefill / prefill-decode 混合批处理。但当前选择
先保留串行 prefill，原因是：

- 4 GB 上 prefill 激活和临时缓存更难预测；
- 新请求的 prompt 长度差异大，micro-batch 很容易制造峰值；
- 没有 GPU 环境时无法验证 paged prefill 的显存和吞吐边界。

这是一个明确的性能缺口，不把它伪装成“已完成”。下一步应该在固定 KV pool 和
观测指标上做 chunked prefill，而不是先写一个没有验证的 overlap 调度器。

## 6. 哪些事情仍然没有做

这次工作的边界必须写清楚：

| 能力 | 状态 |
|---|---|
| paged native GQA CUDA kernel | ❌ 仍是 SDPA reference path |
| 动态 batch CUDA Graph | ❌ 可变 shape 下不可用 |
| chunked prefill / prefill-decode overlap | ❌ prefill 仍串行 |
| preemption / CPU KV swap | ❌ 只做单请求 OOM 隔离 |
| MoE / MLA 执行层 | ❌ 未实现 |
| GPTQ / AWQ / GGUF 直接导入 | ❌ 仍需要转成本仓库 W4A16 分层格式 |
| activation quant / SmoothQuant | ❌ 只做 W4A16 / W3A16 权重量化 |
| tensor / pipeline parallel | ❌ 未实现 |
| AMD / Intel / 跨厂商 kernel | ❌ 当前绑定 NVIDIA SM86 |
| 官方 DeepInfer CacheEngine/Sampler 接入 | ❌ 仍是 engine-core 原型 |

还有一个容易误解的点：物理 KV pool 是启动时按容量预分配的。所谓“弹性”是指：

- 逻辑上按请求长度分配 block；
- 请求结束立即回收；
- 同样内容的完整 prompt block 可以共享；
- 不是请求到来时无限向 CUDA 申请新显存。

CUDA 本身不允许我们在 4 GB 卡上假装显存不存在。真正的弹性需要 pool 容量规划、
OOM 策略、preemption 和 swap 一起配合。

## 7. 下一步的优先级

如果继续沿着“通用推理引擎”方向走，我会按下面的顺序：

### P0：真实 GPU 数据

先跑 `bench_paged_scheduler.py`，对比：

- 串行 vs 并发；
- prefix cache on/off；
- 不同 `block_size` / `max_num_seqs`；
- 3B W4A16 与 7B mixed。

如果没有这组数据，下一步 kernel 优化没有意义。

### P1：paged native attention

实现读取 block table 的 native GQA decode kernel，并保留 SDPA fallback：

- batch=1 paged kernel 先落地；
- 再做 ragged batch；
- 用 cosine、mask、prefix、EOS 做端到端回归；
- 最后再考虑 CUDA Graph 或更复杂的固定 shape 变体。

### P2：chunked prefill

把长 prompt 拆成 token budget 内的 chunk，让已有请求的 decode 不会被超长 prefill
完全阻塞。这需要：

- prefill token budget；
- 请求状态机（prefill / decode / finished）；
- 与 paged block 增长的交互；
- 更细的 OOM / preemption 测试。

### P3：压缩格式和 MoE

GPTQ / AWQ / GGUF 导入和 MoE 支持应该建立在稳定的 cache/scheduler 之上。否则
每接一种格式，都要同时修改 attention、cache、scheduler 和 kernel，回归成本会
指数增长。

### P4：后端与并行

模板化 CUDA/HIP 后端、跨架构 capability 探测、tensor parallel，属于下一阶段。
没有前三点，单纯增加后端抽象很容易变成“接口层很漂亮，实际没有可运行路径”。

## 8. 如何复现

CPU 逻辑回归：

```powershell
python -m pytest -q tests
```

代码风格检查：

```powershell
python -m ruff check apps deepinfer tests scripts\bench_paged_scheduler.py
```

分页连续批处理服务：

```powershell
python apps\server.py `
  --checkpoint D:\Qwen3B-W4A16-SM86 `
  --backend w4a16 `
  --max-cache 256 `
  --scheduler --scheduler-backend paged `
  --max-num-seqs 4 `
  --block-size 16 `
  --prefix-cache-blocks 16 `
  --port 18001
```

查看指标：

```powershell
curl.exe http://127.0.0.1:18001/metrics
```

做 GPU 串行/并发对比：

```powershell
python scripts\bench_paged_scheduler.py `
  --checkpoint D:\Qwen3B-W4A16-SM86 `
  --backend w4a16 `
  --max-cache 256 --max-num-seqs 4 --block-size 16 `
  --max-new-tokens 32 `
  --output docs\results\paged_scheduler_3b.json
```

## 9. 最后的判断

这次改造最重要的成果不是“又多支持了两个模型”，也不是“多了一个调度器类”，
而是把几个原本混在一起的问题拆开了：

```text
模型规格      ModelSpec / adapter
KV 分配       block allocator + prefix cache
attention     dense / static / paged 三种 cache 语义
调度          request queue + variable batch + OOM/cancel 隔离
观测          health + Prometheus
性能验证      独立 benchmark 脚本
```

拆开之后，后面才能真正做 paged native kernel、chunked prefill、量化格式导入和
并行化，而不至于每加一个功能就破坏已有路径。

一句话总结：

> 这次把 DeepInfer-SM86 从“固定槽位的专用加速器”推进成了“有分页 KV、连续
> 批处理、前缀复用和更多 dense 架构适配的 engine-core 原型”。它解决了三块最
> 急迫的拼图，但没有把“原型”写成“生产引擎”。
