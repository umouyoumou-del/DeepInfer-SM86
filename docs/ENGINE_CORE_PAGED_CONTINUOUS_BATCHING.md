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
  ├─ chunked prefill（prepare / launch / finish）
  ├─ dual CUDA stream prefill/decode overlap
  ├─ CPU cold-request swap（可选）
  ├─ batched decode（当前所有 running）
  ├─ per-request sampling / EOS / cancel
  └─ metrics
          │
          ▼
PagedKVCacheManager
  ├─ BlockAllocator（引用计数）
  ├─ PrefixCache（LRU + 共享不可变 block）
  ├─ physical pool [layers, blocks, kv_heads, block, dim] FP16/INT8/FP8
  └─ CPU pinned payload（swap 时）
          │
          ▼
Qwen2 generic decoder
  ├─ native paged GQA decode kernel (FP16/INT8/FP8)
  ├─ SDPA fallback for prefill / external mask
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

### 5.1 Chunked prefill（后续 P0 已补）

后续 P0 改造已把 prefill 改成固定 token chunk，并在 scheduler loop 中与 decode
交替执行：

- 每个请求有 `PREFILL` / `DECODE` 状态；
- prefill 直接写入 paged block，不再保留整段 dense prefill cache；
- 多个 prefill 请求 round-robin；
- 每个 loop 先执行 batched decode，再推进一个 prefill chunk；
- 详细说明见 `docs/P0_CHUNKED_PREFILL_AND_MEMORY_BUDGET.md`。

剩余缺口是：prefill 与 decode 还没有在**同一次 forward** 内真正混合。

### 5.2 显存不足策略

decode 前为每个序列检查下一个 token 所需 block。若池耗尽：

- 若启用 `--cpu-offload`，先选最久未活跃的 PREFILL/DECODE 请求，把已使用
  block 复制到 pinned CPU payload，释放其 GPU block，并标记为 `SWAPPED`；
- 若仍不够，该请求才以明确的 `KV cache out of memory` 失败并释放自身 block；
- 其他请求继续 decode；后续 block 足够时自动 reload 并恢复原状态；
- 不会像固定 slot 一样因为一个超长请求直接污染整个 batch。

当前是整请求 swap，不是按 block 热度的细粒度迁移；没有 recompute preemption。
CPU payload 预算由 `--max-cpu-offload-mb` 显式限制。

## 6. Attention 改动

`Qwen2Attention.forward()` 现在统一处理：

- 普通动态 KV cache；
- 固定 dense cache（CUDA Graph 路径）；
- paged cache view；
- 绝对 position 的 causal mask；
- sliding-window mask；
- paged validity mask；
- 外部 attention mask；
- Gemma 2 attention-logit softcap（native paged kernel 支持；SDPA fallback
  走显式 eager 参考路径）；
- FP16 / INT8 / FP8 paged KV；FP8 支持 E4M3/E5M2。

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
53 passed, 6 skipped（GPU-only tests skipped on CPU）
```

覆盖：

- block 引用计数 / double free / OOM；
- prefix cache retain、复用、LRU 淘汰；
- paged 写入、grow、mask materialize；
- 两个不同长度请求的 batched paged logits 与 dynamic cache 对齐；
- Qwen2 / Llama / Mistral / Gemma / Gemma2 与 HF logits 对齐；
- Mistral sliding window 和 Gemma2 alternating SWA/softcap；
- paged scheduler 两请求共享 prefix、同一步 batched decode；
- Prometheus 文本格式；
- W4/W3 tiled prefill GEMM 与 dequant 参考对齐；
- W4/W3 batched GEMV 与单行 GEMV 对齐；
- FP8 paged KV 与 CPU offload payload roundtrip；
- GPU-only 双 stream overlap、FP8 native attention 和 CPU swap smoke。

GPU 吞吐基准脚本：

```powershell
python scripts\bench_paged_scheduler.py `
  --checkpoint D:\Qwen3B-W4A16-SM86 `
  --backend w4a16 `
  --max-cache 128 --block-size 16 --prefill-chunk-tokens 32 `
  --max-new-tokens 16 --concurrency 1,2,4 `
  --reserve-memory-mb 384 --no-prefix-cache `
  --output docs\results\paged_concurrency_3b_native_batched_fp16.json
```

RTX 3050 Laptop 4 GB 上：

```text
native paged + batched W4, c1/c4: 8.592 / 14.736 tokens/s
tiled + overlap,          c1/c4: 11.189 / 22.779 tokens/s
tiled + overlap + FP8,    c1/c4: 11.089 / 22.527 tokens/s
CPU swap smoke: 10/10 requests complete, swap/resume once per iteration
```

长文档插入抖动、INT8/FP8 KV、CPU offload 和完整数字见
`docs/P0_CHUNKED_PREFILL_AND_MEMORY_BUDGET.md`、
`docs/ARTICLE_6_NATIVE_PAGED_KERNEL_AND_4GB_BENCH.md`、
`docs/ARTICLE_7_PREFILL_DECODE_OVERLAP_AND_TILED_PREFILL.md` 与
`docs/ARTICLE_8_KV_FP8_AND_CPU_OFFLOAD.md`。

## 11. 还没解决的部分

必须明确，以下仍然是缺口：

- native paged GQA 只覆盖 decode `T=1`、`D<=128`、`max_length<=1024`；
  prefill / external mask 仍走 SDPA；
- 动态 batch 下没有 CUDA Graph；
- prefill 与 decode 可以双 CUDA stream 并发，但仍不是同一次 forward 内混合；
- 长 prefill 插入时，固定 chunk size 无法同时压低 TPOT 中位数和 p95 尖峰；
- CPU swap 是整请求粒度，没有 block 级冷热迁移或 recompute preemption；
- CPU offload 的 swap 等待会拉长被换出请求的 token interval；
- tiled W4/W3 GEMM 仍是 CUDA core + FP32 累加，没有 Tensor Core / cp.async；
- 没有 beam search、speculative decoding；
- 没有 MoE / MLA 执行层；
- 已支持 GPTQ / AWQ / GGUF 离线转换导入，但没有原格式零拷贝直跑；
- 没有 activation quantization / SmoothQuant；
- 没有 tensor/pipeline parallel；
- 不是 DeepInfer 官方 `worker/cache_engine.py` / Sampler 的完整接入；
- 4 GB 上长上下文和 batch 大小仍然受物理显存硬约束。

因此这次工作的准确定位是：

> 在原来的 SM86 专用 model_executor 之上，补齐了可运行的 engine-core：
> 分页/弹性 KV、连续批处理、prefix cache、基础指标治理和更多 dense 架构。
> 它仍不是生产级 vLLM/TensorRT-LLM 替代品。
