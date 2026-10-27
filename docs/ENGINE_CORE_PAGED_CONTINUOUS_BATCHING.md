# Engine Core：分页 KV Cache、连续批处理与架构扩展

> 时间：2026-10-27  
> 目标硬件：RTX 3050 Laptop 4 GB / SM86 / Windows  
> 定位：`model_executor` 之上的 engine-core 原型，不是完整 DeepInfer engine

## 1. 这次补了什么

上一版已经有固定槽位动态批处理，但每个槽位仍然预分配完整的
`[batch, kv_heads, max_cache, head_dim]` KV 张量。它解决了“两个请求一起
decode”，但没有解决“短请求不应该支付 256 token 的缓存成本”。

本次新增四块：

```text
engine/
  kv_cache.py       BlockAllocator + PrefixCache + PagedKVCacheManager
  metrics.py        Prometheus 文本指标
apps/
  paged_scheduler.py  可变 batch 的连续批处理调度器
  server.py           /metrics、队列上限、流式请求取消
```

同时把模型适配扩到：

```text
Qwen2 / Qwen2.5
Llama
Mistral（sliding-window GQA）
Gemma（GeGLU + offset RMSNorm + embedding scale）
Gemma 2（sandwich norm + logit softcap + alternating SWA）
```

这些是 dense/GQA 架构上的真实执行差异，不再是只改模型名字的注册表占位。

## 2. 总体结构

```text
HTTP /v1/chat/completions
          │
          ▼
PagedBatchScheduler
  ├─ waiting queue（有界）
  ├─ running sequences（可变 batch）
  ├─ serial prefill（每请求）
  ├─ batched decode（当前所有 running）
  ├─ per-request sampling / EOS / cancel
  └─ metrics
          │
          ▼
PagedKVCacheManager
  ├─ BlockAllocator（引用计数）
  ├─ PrefixCache（LRU + 共享不可变 block）
  └─ physical pool [layers, blocks, kv_heads, block, dim]
          │
          ▼
Qwen2 generic decoder
  ├─ paged attention reference path (SDPA)
  ├─ sliding-window / alternating SWA masks
  ├─ Gemma/Gemma2 norm、GeGLU、softcap
  └─ existing W4A16 / W3A16 batch GEMV
```

## 3. 分页 KV cache

### 3.1 物理池与逻辑 block table

物理池只在启动时分配一次：

```text
key/value: [num_layers, num_blocks, num_kv_heads, block_size, head_dim]
```

每个请求只拥有一个 block-id 列表：

```text
request.block_ids = [3, 7, 11, ...]
```

decode 时构造：

```text
block_table: [batch, max_blocks_in_batch]
seq_lens:    [batch]
```

`PagedLayerKVCache.update()` 写入：

```text
physical_block = block_table[b, position // block_size]
offset         = position % block_size
pool[physical_block, :, offset, :] = new_kv[b]
```

`materialize()` 再按 batch 内最大长度 gather 成临时 dense K/V，并生成：

```text
valid[b, t] = t < seq_lens[b] + 1
```

mask 的生成完全在张量上完成，不依赖 `.item()`，因此不会在每个 layer 触发
GPU→CPU 同步。

### 3.2 为什么能省显存

固定 slot 方案中，两个 32-token 请求在 256 context 下也会占用：

```text
2 × ceil(256 / block_size) 个逻辑块容量
```

分页方案只分配：

```text
ceil(32 / block_size) × 2 个块
```

短请求结束后 block 立即回到 free list；突发长请求按需增长。默认池容量仍是
可配置的，因为 CUDA 物理内存不能凭空增长，但逻辑分配不再和 `max_cache`
强绑定。

### 3.3 BlockAllocator

block 是引用计数的：

```text
allocate(n)  -> refcount = 1
retain(ids)  -> refcount += 1（prefix cache 共享）
free(ids)    -> refcount -= 1，归零时回 free list
```

这样前缀 block 可以被多个请求同时读取，但不会被某个请求的退出提前释放。

## 4. 前缀缓存

只缓存**完整且不可变**的 prompt block：

```text
hash_i = cumulative_hash(tokens[0 : (i+1)*block_size])
```

请求 prefill 时：

1. 计算 prompt 的累计 block hash；
2. 从前往后匹配 prefix cache，最长匹配必须 block 对齐；
3. 至少保留最后 1 个 token 重新计算，除非最后一个完整 block 前还有可复用块；
4. 把匹配到的旧 block 作为动态 prefill cache 的前缀；
5. 只对 suffix 执行模型前向；
6. 新 suffix K/V 写入新 block；
7. 只有“完整 prompt block”才注册到 prefix cache。

decode 阶段如果 prompt 长度刚好落在 block 边界，下一个 token 会分配新 block，
因此已注册的完整 prompt block 永远不会被原地追加修改，不需要 copy-on-write。
prompt 尾部不完整 block 不注册，避免共享后写入污染。

prefix cache 有界并使用 LRU 淘汰；淘汰只释放 cache 自己持有的引用。

## 5. 连续批处理调度器

`apps/paged_scheduler.py` 的循环：

```text
while not stopped:
    admit queued requests while running < max_num_seqs
    if running:
        ensure every sequence has a block for its next token
        pack active sequences into one batch
        one batched decode forward
        per-request sample / EOS / metrics
        release finished blocks immediately
    else:
        sleep 10ms
```

与旧固定槽位实现的区别：

| 项目 | 旧 static scheduler | 新 paged scheduler |
|---|---|---|
| batch 大小 | 固定 max_batch_size | 当前 running 数量，1..max_num_seqs |
| KV 容量 | 每 slot 固定 max_cache | 按 token 需求分块分配 |
| 新请求加入 | 固定 slot 空出 | block 可分配即可加入 |
| 请求结束 | 清空整槽 | 立即归还 blocks |
| 前缀复用 | 无 | 完整 block 共享 |
| CUDA Graph | 固定 shape 时可捕获 | 动态 batch 下不捕获 |
| 长上下文 | 受 GQA kernel 256 限制 | paged 路径可超过 256 |
| 指标/取消 | 基础 | Prometheus + stream cancel |

### 5.1 prefill 仍是串行

当前为了保证 4 GB 上峰值可预测，每个新请求先单独 prefill，然后加入 decode
batch。也就是说：

- decode 已经连续批处理；
- prefill 还没有 chunked prefill / prefill-decode 混合；
- 最坏情况下长 prompt prefill 会阻塞已有 decode 请求。

这是下一阶段最值得做的性能工作之一。

### 5.2 显存不足策略

decode 前为每个序列检查下一个 token 所需 block。若池耗尽：

- 该请求以明确的 `KV cache out of memory` 失败并释放自身 block；
- 其他请求继续 decode；
- 不会像固定 slot 一样因为一个超长请求直接污染整个 batch。

当前还没有 CPU swap / recompute preemption。

## 6. Attention 改动

`Qwen2Attention.forward()` 现在统一处理：

- 普通动态 KV cache；
- 固定 dense cache（CUDA Graph 路径）；
- paged cache view；
- 绝对 position 的 causal mask；
- sliding-window mask；
- paged validity mask；
- 外部 attention mask；
- Gemma 2 attention-logit softcap（SDPA 无此参数，走显式 eager 参考路径）。

mask 条件：

```text
valid(b, q, k) =
    k <= absolute_position(b, q)
    AND paged_valid(k)
    AND (q_abs - k < sliding_window if enabled)
    AND external_mask(k)
```

这样可以正确计算：

- cache continuation；
- prefix-cache suffix prefill；
- ragged batch decode；
- Mistral 全层 sliding window；
- Gemma 2 奇偶层 alternating sliding/full attention。

## 7. 模型架构扩展

### 7.1 Mistral

差异：所有层使用 sliding-window attention。  
实现：`ModelSpec.use_sliding_window=True` + 通用 attention window mask。

### 7.2 Gemma

差异：GeGLU、RMSNorm 的 `1 + weight`、embedding 乘以
`sqrt(hidden_size)`。  
实现：归一化字段进入 ModelSpec，通用 decoder 按配置选择。

### 7.3 Gemma 2

差异：

- sandwich normalization（attn/MLP 前后各一次 norm）；
- offset RMSNorm；
- GeGLU；
- attention logit softcap；
- final logit softcap；
- `query_pre_attn_scalar` 控制 attention scaling；
- 奇偶层 alternating sliding/full attention。

实现位置：

```text
deepinfer/model_spec.py
  GemmaAdapter / Gemma2Adapter
deepinfer/model_executor/models/qwen2.py
  Qwen2RMSNorm(weight_offset=...)
  Qwen2DecoderLayer(sandwich_norm=...)
  Qwen2Attention(sliding_window_pattern=..., attn_logit_softcapping=...)
deepinfer/model_executor/models/gemma*.py
```

W4A16 分层 checkpoint 的导出/加载也同步支持额外的
`pre_feedforward_layernorm` / `post_feedforward_layernorm`。

## 8. 网关治理

新增：

```text
GET /metrics
```

输出 Prometheus 文本，包括：

```text
deepinfer_requests_total
deepinfer_requests_finished_total
deepinfer_requests_failed_total
deepinfer_prompt_tokens_total
deepinfer_generated_tokens_total
deepinfer_prefix_reused_tokens_total
deepinfer_decode_steps_total
deepinfer_running_requests
deepinfer_queued_requests
deepinfer_kv_cache_blocks_used
deepinfer_kv_cache_blocks_free
deepinfer_kv_cache_blocks_total
deepinfer_decode_step_seconds_avg
```

`/health` 会附带 scheduler 状态、block 使用量、prefix hit/miss 和 metrics
snapshot。请求队列有上限；流式连接断开时会调用 scheduler cancel，避免请求
继续占用 KV block。

## 9. 启动方式

3B W4A16 + paged continuous batching：

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

旧固定槽位路径仍保留：

```powershell
python apps\server.py `
  --checkpoint D:\Qwen3B-W4A16-SM86 `
  --backend w4a16 `
  --scheduler --scheduler-backend static `
  --max-batch-size 2
```

不启用 scheduler 时仍使用单请求 dense cache + CUDA Graph，这是当前单请求
decode 最低延迟路径。

## 10. 验证与基准

CPU 正确性回归：

```text
18 passed, 1 skipped
```

覆盖：

- block 引用计数 / double free / OOM；
- prefix cache retain、复用、LRU 淘汰；
- paged 写入、grow、mask materialize；
- 两个不同长度请求的 batched paged logits 与 dynamic cache 对齐；
- Qwen2 / Llama / Mistral / Gemma / Gemma2 与 HF logits 对齐；
- Mistral sliding window 和 Gemma2 alternating SWA/softcap；
- paged scheduler 两请求共享 prefix、同一步 batched decode；
- Prometheus 文本格式。

GPU 吞吐基准脚本：

```powershell
python scripts\bench_paged_scheduler.py `
  --checkpoint D:\Qwen3B-W4A16-SM86 `
  --backend w4a16 `
  --max-cache 256 --max-num-seqs 4 --block-size 16 `
  --max-new-tokens 32 `
  --output docs\results\paged_scheduler_3b.json
```

脚本会比较串行提交与并发提交的 wall time、aggregate tokens/s、prefix 复用和
block 状态。**本仓库当前没有伪造这次 GPU 基准结果**；需要在目标机器上实际
运行后记录。

## 11. 还没解决的部分

必须明确，以下仍然是缺口：

- paged attention 目前是 PyTorch SDPA reference path，不是 paged native GQA
  CUDA kernel；
- 动态 batch 下没有 CUDA Graph；
- prefill 串行，没有 chunked prefill；
- 没有 preemption / CPU KV swap；
- 没有 beam search、speculative decoding；
- 没有 MoE / MLA 执行层；
- 没有 GPTQ / AWQ / GGUF 直接导入；
- 没有 a
ctivation quantization / SmoothQuant；
- 没有 tensor/pipeline parallel；
- 不是 DeepInfer 官方 `worker/cache_engine.py` / Sampler 的完整接入；
- 4 GB 上长上下文和 batch 大小仍然受物理显存硬约束。

因此这次工作的准确定位是：

> 在原来的 SM86 专用 model_executor 之上，补齐了可运行的 engine-core：
> 分页/弹性 KV、连续批处理、prefix cache、基础指标治理和更多 dense 架构。
> 它仍不是生产级 vLLM/TensorRT-LLM 替代品。
