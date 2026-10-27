# P0 改造：Chunked Prefill、批量 Decode 和 4GB 显存预算

> 日期：2026-10-27  
> 目标：把“能并发”推进到“4GB 上并发时更稳定、可观测、可被外部系统使用”。

## 1. 本次完成的 P0

```text
Chunked Prefill        ✅
Request-state scheduler ✅
Batched paged decode    ✅（SDPA reference path）
KV memory budget        ✅
Admission OOM isolation ✅
TTFT / TPOT metrics     ✅
OpenAI /v1/completions  ✅
Tenant prefix isolation ✅
KV INT8 paged cache     ✅（SDPA reference path）
```

仍未完成：

```text
paged native GQA CUDA kernel  ❌
prefill+decode 单次混合 forward ❌
KV FP8 / CPU offload           ❌
```

## 2. Chunked Prefill

### 2.1 之前的问题

旧 scheduler 的 prefill 是串行的：

```text
长 prompt prefill（可能几十到几百 token）
        ↓
已有 decode 请求全部等待
        ↓
prefill 完成后才恢复 decode
```

这会让流式请求的 TPOT 在长 prompt 插入时出现长尾。

### 2.2 现在的调度方式

Paged scheduler 中每个请求有明确状态：

```text
PREFILL
  ├─ 每次只处理 min(剩余 token, prefill_chunk_tokens)
  ├─ 直接在 paged block 中写 K/V
  └─ prefill 完成后采样第一个 token，转 DECODE

DECODE
  └─ 每个 scheduler iteration 进行一次 batched decode
```

`_run()` 的顺序是：

```text
admit queued requests
decode all DECODE requests
prefill one chunk for one PREFILL request
```

因此一个长 prompt 不再一次占满整个调度周期，decode 请求可以在 chunk 之间持续
获得执行机会。

### 2.3 公平性

多个 prefill 请求采用 round-robin：

- 不会让第一个长 prompt 永远独占 prefill 机会；
- 短 prompt 可以更早完成 prefill，开始 decode；
- 测试里同时提交 20-token 和 3-token prompt，短请求 TTFT 明显早于长请求。

### 2.4 Paged prefill 而不是 dense 临时 cache

本次没有为 chunked prefill 构造一个不断增长的 dense cache。每个 chunk 都：

1. 用已有 block table 构造 paged cache view；
2. 写入当前 chunk 的 K/V；
3. 用 causal + paged mask 做 attention；
4. 只保留当前 chunk 的 logits。

这避免长 prompt 在 CPU/GPU 上再保留一份完整 dense KV。

### 2.5 参数

```text
--prefill-chunk-tokens 64
```

`64` 是默认起点；4 GB 卡上应根据实测 TPOT 抖动和 prefill 速度调整。  
chunk 越大，TTFT 可能越低，但长 prompt 对已有 decode 的干扰越大；chunk 越小，
TPOT 更平滑，但调度和 kernel launch 次数更多。

## 3. Batched Decode / Paged Attention 覆盖

### 3.1 已保证的内容

- W4A16 / W3A16 GEMV 支持 batch token；
- scheduler 每个 decode step 将当前所有 DECODE 请求打成一个 batch；
- paged cache 支持：
  - 多 block table；
  - batch 内不同 seq_len；
  - causal mask；
  - paged validity mask；
  - sliding-window mask；
  - chunked prefill 的 `T > 1` cache 写入。
- `EngineMetrics.max_decode_batch_size` 记录实际达到的最大 decode batch；
- `GenerationResult.inter_token_seconds` 保存每个请求的逐 token interval，benchmark
  用它计算 TPOT p50/p95 和 jitter，而不只看整段平均值。

### 3.2 正确的验证

CPU 回归中：

- 两个不同长度请求的 batched paged logits 与 dynamic cache 对齐；
- chunked paged prefill 的末位 logits 与完整 prefill 对齐；
- Gemma 2 sliding/softcap 场景下 paged decode 与 dynamic cache 对齐；
- scheduler 记录到 `max_decode_batch_size >= 2`。

### 3.3 性能验证工具

`scripts/bench_paged_scheduler.py` 已改为并发 sweep：

```bash
python scripts/bench_paged_scheduler.py \
  --checkpoint D:/Qwen3B-W4A16-SM86 \
  --max-new-tokens 32 \
  --concurrency 1,2,4,8 \
  --prefill-chunk-tokens 64 \
  --output docs/results/paged_concurrency_3b.json
```

输出：

- aggregate tokens/s；
- TTFT mean/p50/p95；
- TPOT mean/p50/p95；
- per-token inter-token interval mean/p50/p95 与 p95-p50 jitter；
- failed requests；
- KV OOM delta；
- max decode batch；
- prefix reused tokens；
- memory budget。

### 3.4 还没做到的 native kernel

当前 paged attention 仍是：

```text
block gather → dense K/V → SDPA
```

不是：

```text
native paged GQA kernel
```

原因很现实：当前没有 CUDA 测试环境，不能把未验证的 block-table kernel 直接放进
主链路。下一步应在目标 3050 上用 cosine + 变长 batch + sliding-window 回归后，
再把 native kernel 设为默认路径。

## 4. KV INT8 缓存

新增 `--kv-cache-dtype int8`：

- paged K/V pool 使用 int8；
- 每个 token/head 保存 FP16 K scale 和 V scale；
- attention gather 后参考路径反量化回计算 dtype 再做 SDPA；
- block 大小估算包含 scale sidecar；
- CPU 测试中 chunked paged prefill 与完整 FP32 prefill 对齐（cosine > 0.999）。

它不是 native INT8 attention kernel，而是 paged reference path 的低显存存储优化。
下一步需要把 dequant 融合进 native paged kernel。

## 5. 显存预算和 OOM 策略

### 5.1 启动预算

`deepinfer/engine/memory_budget.py` 在启动时计算：

```text
model parameter bytes
+ model buffer bytes
+ KV block pool budget
+ reserve headroom
<= GPU utilization ceiling
```

输入：

- 模型实际参数/缓冲；
- `torch.cuda.mem_get_info()` 当前空闲显存；
- block 的字节数；
- `max_num_seqs`；
- `max_cache`；
- `gpu_memory_utilization`；
- `reserve_memory_mb`。

输出：

- `num_blocks`；
- `kv_budget_bytes`；
- `capacity_tokens`；
- 完整的 `memory_budget` report。

如果连一个 block 都放不下，启动直接失败并打印 report，而不是先启动再随机 OOM。

### 5.2 请求级 block 预留

admission 时一次性预留：

```text
prompt tokens + max_new_tokens
```

所以 decode 期间通常不会再申请 block。容量不足时：

- 请求在 admission 阶段失败；
- 不污染已有请求；
- API 返回 503 + `insufficient_kv_memory`；
- metrics 增加 `kv_cache_oom_total` 和 `request_rejected_total`。

### 5.3 Prefill/runtime OOM 隔离

prefill chunk 或 decode forward 抛 OOM 时：

- 只失败当前请求或当前 decode batch；
- 清除 CUDA cache；
- 增加 `runtime_oom_total`；
- 其他请求可以继续；
- 流式请求收到 sentinel，HTTP 连接可以正常关闭。

### 5.4 观测

`/health` 的 scheduler 状态包含：

```text
active / prefill / decode / queued
memory_budget
kv_cache.allocator
prefix_cache
```

`/metrics` 包含：

```text
kv_cache_blocks_used
kv_cache_blocks_free
kv_cache_blocks_total
kv_cache_usage_ratio
prefill_requests
decode_requests
kv_cache_oom_total
runtime_oom_total
prefill_steps_total
prefill_decode_overlap_steps
ttft_seconds_avg
tpot_seconds_avg
```

## 6. OpenAI API 补全

新增：

```text
POST /v1/completions
```

支持：

- 字符串 prompt；
- `stream`；
- `stop`（字符串或字符串数组）；
- `max_tokens` / `max_completion_tokens`；
- paged scheduler 取消；
- tenant：`user` 或 `X-Tenant-ID`。

`/v1/chat/completions` 也补上了 stop 和 tenant 传递。

非 scheduler fallback 也会应用 stop 过滤；但完整并发、取消和 KV 预算能力仍然
需要 `--scheduler --scheduler-backend paged`。

## 7. 前缀缓存跨租户隔离

`PrefixCache.block_hashes(..., namespace=tenant)` 将租户名混入初始 hash：

```text
tenant A + prompt P != tenant B + prompt P
```

因此：

- 即使两个租户 prompt 完全相同，也不会共享 prefix block；
- 同租户同 prompt 仍可正常命中；
- block 引用计数和 LRU 驱逐逻辑不变。

这解决的是隐私隔离，不是加密存储；如果 threat model 包含同一进程内恶意代码，
还需要更严格的进程/内存隔离。

## 8. 模型目录探测

新增：

```powershell
python scripts\probe_model.py D:\models\Qwen2.5-3B-Instruct
```

输出：

- 是否 supported；
- architecture/model_type；
- 是否 native W4A16 runtime_ready；
- 是否检测到 GPTQ/AWQ/GGUF；
- tokenizer 文件；
- sliding window / rope scaling / 层数/头数；
- 不支持时的明确原因。

## 9. 当前测试结果

```text
44 passed, 1 skipped
ruff: All checks passed
```

新增测试覆盖：

- chunked paged prefill 与完整 prefill logits 对齐；
- 多请求 decode 达到 batch width >= 2；
- long/short prompt 的 chunked prefill 交替与 TTFT 差异；
- prefill runtime OOM 只失败一个请求；
- admission KV OOM 只失败一个请求；
- stop sequence 截断；
- tenant prefix isolation 与命中后结果一致；
- `/v1/chat/completions` 和 `/v1/completions` 路由/流式；
- 模型目录 probe 的 supported/unsupported 分支；
- INT8 KV cache 的写入、读取、scale sidecar 和 chunked prefill logits 对齐。

## 10. 仍未完成

- native paged GQA CUDA kernel；
- prefill + decode 真正的单次 mixed forward；
- TPOT 抖动自动反馈调节 chunk size；
- native INT8 attention kernel、KV FP8、CPU cold-block offload；
- CUDA Graph 用于动态 batch；
- 真实 3B/7B 的 TTFT/TPOT/吞吐数据。

下一阶段应先在实际 RTX 3050 上跑 `bench_paged_scheduler.py` 的并发 sweep，
根据 TTFT/TPOT 和 OOM 数据决定是否调小 `prefill_chunk_tokens` 或降低
`max_num_seqs`，再开发 native paged kernel。
