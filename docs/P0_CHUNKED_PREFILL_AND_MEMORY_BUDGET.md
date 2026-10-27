# P0 改造：Chunked Prefill、批量 Decode 和 4GB 显存预算

> 日期：2026-10-27  
> 目标：把“能并发”推进到“4GB 上并发时更稳定、可观测、可被外部系统使用”。

## 1. 本次完成的 P0

```text
Chunked Prefill        ✅
Request-state scheduler ✅
Batched paged decode    ✅（native FP16 + SDPA fallback）
Native paged GQA kernel ✅
Batched W4 GEMV         ✅
KV memory budget        ✅
Admission OOM isolation ✅
TTFT / TPOT metrics     ✅
OpenAI /v1/completions  ✅
Tenant prefix isolation ✅
KV INT8 paged cache     ✅（native + SDPA fallback）
KV FP8 paged cache      ✅（E4M3/E5M2 native decode）
CPU KV cold offload     ✅（request-level swap）
```

仍未完成：

```text
prefill+decode 单次混合 forward ❌
dynamic-batch CUDA Graph        ❌
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

### 3.4 Native paged GQA decode kernel

新增 `deepinfer/engine/paged_attention.py`，decode 阶段直接读取：

```text
query [B, H, 1, D]
key/value pool [num_blocks, KVH, block_size, D]
block_table [B, max_blocks]
valid_lengths [B]
```

kernel 完成 score、softmax、value 加权，不再 materialize dense K/V，也不调用
SDPA。支持：

- 变长 batch；
- GQA；
- sliding window；
- Gemma2 attention logit softcap；
- FP16 KV 和 INT8 KV + FP16 scale；
- `D <= 128`、`max_length <= 1024`。

超出 envelope、prefill `T > 1`、存在 external attention mask 时自动回退
SDPA。环境变量 `DEEPINFER_DISABLE_NATIVE_PAGED_ATTENTION=1` 可强制回退。

同时新增 batched W4A16 GEMV：decode batch 2..8 时一个行 block 只读取一次 packed
weight，在寄存器里同时累加 8 个 batch，不再让 `grid.y=batch` 重复读权重。
`DEEPINFER_DISABLE_BATCHED_GEMV=1` 可回退旧 kernel。

实际验证（RTX 3050 Laptop 4 GB、Qwen2.5-3B W4A16）：

```text
paged attention vs reference: cosine 1.0, max abs 0.0009765625
batched W4 vs dequant ref:    cosine 0.99999994, max abs 0.0009765625
end-to-end logits vs SDPA:    cosine 0.9999988, argmax 一致
单次 decode: native 0.0659 s vs SDPA 0.0863 s，约 1.31x
```

## 4. KV 量化缓存：INT8 与 FP8

新增 `--kv-cache-dtype int8`：

- paged K/V pool 使用 int8；
- 每个 token/head 保存 FP16 K scale 和 V scale；
- native paged kernel 内读取 FP16 scale 并反量化；prefill 仍走 SDPA；
- block 大小估算包含 scale sidecar；
- CPU 测试中 chunked paged prefill 与完整 FP32 prefill 对齐（cosine > 0.999）。

INT8 现在有独立 native paged decode kernel：K/V 以 int8 存储，kernel 内读取
FP16 scale 并完成反量化；gather/softmax/value 累加不再 materialize dense K/V。
变长 batch、GQA、sliding window、softcap 与 FP16 kernel 一致。

新增 `--kv-cache-dtype fp8` / `fp8_e5m2`：

- FP8 直接 cast，无 per-token scale sidecar；
- block bytes 294912，是 FP16 的 1/2、INT8 的约 98%；
- native paged kernel 将 FP8 byte bit pattern 在 CUDA 内解码成 float；
- 支持 E4M3 和 E5M2 两种 FP8；
- 与 FP16/INT8 一样支持 ragged batch、GQA、sliding window、softcap。

实机 synthetic 校验：INT8/E4M3 cosine 1.0；E5M2 cosine 0.99999988。
3B c1/c2/c4：11.089 / 17.317 / 22.527 tokens/s，接近 FP16 native 的
11.189 / 17.484 / 22.779，但 KV block 只占一半。FP8 先验版本走 SDPA fallback
时 c4 只有 21.4 tokens/s；native FP8 kernel 把这部分差距基本补平。

CPU cold offload 见 10.5。它和 KV 数据类型正交：FP16/INT8/FP8 payload 都能
swap 到 CPU。

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
- INT8 KV cache 的写入、读取、scale sidecar 和 chunked prefill logits 对齐；
- FP8 paged KV 写入/读取和 block 预算；
- CPU payload 的 FP16/INT8/FP8 offload/reload roundtrip；
- PREFILL/DECODE 请求 swap/resume、CPU budget=0 拒绝路径。

实机 kernel 校验：

```powershell
python scripts\validate_sm86_kernels.py `
  --checkpoint D:\Qwen3B-W4A16-SM86 `
  --output docs\results\sm86_kernel_validation_3b.json
```

## 10. 实机结果：RTX 3050 Laptop 4 GB

模型：Qwen2.5-3B W4A16；`max_cache=128`、`block_size=16`、
`prefill_chunk_tokens=32`、`max_new_tokens=16`、无 prefix cache、
`gpu_memory_utilization=0.80`、`reserve_memory_mb=384`。

### 10.1 并发 sweep

| 路径 | concurrency | tokens/s | TTFT p50 | TPOT p50 | TPOT p95-p50 | max decode batch |
|---|---:|---:|---:|---:|---:|---:|
| SDPA reference | 1 | 6.906 | 0.946 s | 0.0914 s | 0.0066 s | 1 |
| SDPA reference | 2 | 9.845 | 1.486 s | 0.0865 s | 0.0114 s | 2 |
| SDPA reference | 4 | 12.049 | 3.510 s | 0.1398 s | 0.3735 s | 4 |
| native paged attn | 1 | 8.144 | 0.924 s | 0.0693 s | 0.0036 s | 1 |
| native paged attn | 2 | 10.709 | 1.500 s | 0.0695 s | 0.0159 s | 2 |
| native paged attn | 4 | 12.466 | 3.475 s | 0.1299 s | 0.3546 s | 4 |
| + batched W4 GEMV | 1 | 8.592 | 0.712 s | 0.0766 s | 0.0123 s | 1 |
| + batched W4 GEMV | 2 | 12.227 | 1.159 s | 0.0724 s | 0.0057 s | 2 |
| + batched W4 GEMV | 4 | 14.736 | 2.746 s | 0.1214 s | 0.2903 s | 4 |

完整数据：

- `docs/results/paged_concurrency_3b_fp16.json`
- `docs/results/paged_concurrency_3b_native_fp16.json`
- `docs/results/paged_concurrency_3b_native_batched_fp16.json`
- `docs/results/paged_concurrency_3b_native_batched_c8.json`
- `docs/results/paged_concurrency_3b_int8kv_native.json`
- `docs/results/sm86_kernel_validation_3b.json`
- `docs/results/w4_batched_gemv_microbench.json`
- `docs/results/paged_concurrency_3b_fp8kv.json`
- `docs/results/cpu_offload_3b_fp16.json`
- `docs/results/cpu_offload_3b_fp8.json`

### 10.2 长文档插入时的流式 TPOT 抖动

短流先生成，在首个 token 后插入 189-token 长 prompt。完整数据见
`docs/results/tpot_jitter_3b.json`。

| chunk | 基线 TPOT p50 | 插入后 TPOT p50 | 插入后 ITL p95 | p95-p50 | 最大 ITL | 长 prompt TTFT |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.0706 s | 0.1885 s | 0.1981 s | 0.0095 s | 0.1992 s | 4.481 s |
| 16 | 0.0706 s | 0.2659 s | 0.2943 s | 0.0285 s | 0.2959 s | 3.465 s |
| 32 | 0.0705 s | 0.0692 s | 0.4926 s | 0.4234 s | 0.4950 s | 2.927 s |
| 64 | 0.0726 s | 0.0722 s | 0.5336 s | 0.4615 s | 0.5368 s | 1.606 s |

结论：

- chunk 8 抖动最低，但长 prefill 持续期间 TPOT 从 0.074 s 抬到约 0.185 s；
- chunk 32/64 的中位 TPOT 不受影响，但每隔一个 prefill chunk 出现约 0.5 s 尖峰；
- 当前是“decode 后串行做一个 prefill chunk”，没有跨 CUDA stream 并行，
  因此 TTFT 和 TPOT 抖动存在结构性 trade-off；
- 生产默认推荐从 chunk 8 或 16 开始，按目标 SLA 在 `bench_tpot_jitter.py`
  上选择，而不是只看 aggregate tokens/s。

### 10.3 max_num_seqs=8 和 INT8 KV

`max_cache=256` 下 c8 也能完成 8 路 decode，没有 OOM：

| c | tokens/s | TTFT p50 | TPOT p50 | max decode batch |
|---:|---:|---:|---:|---:|
| 1 | 8.897 | 0.715 s | 0.0722 s | 1 |
| 4 | 14.756 | 2.720 s | 0.1222 s | 4 |
| 8 | 15.931 | 5.243 s | 0.1910 s | 8 |

c8 相对 c1 只有约 1.79x，说明瓶颈已经从“KV 放不下”转为“prefill/decode
串行 + launch/Python 开销”。

INT8 KV native 路径在 c4 为 13.868 tokens/s，低于 FP16 native 的 14.736；
block bytes 从 589824 降到 299520，约减半。当前 3B 的 KV 只占约 18 MiB，因此
INT8 主要价值是容量而不是速度。

更完整的弯路和数字见:
`docs/ARTICLE_6_NATIVE_PAGED_KERNEL_AND_4GB_BENCH.md`。

### 10.5 CPU cold-block offload

新增请求级 swap：

- GPU block 不足时，选最久未活跃的 PREFILL/DECODE 请求；
- 把它已经使用的 KV block 复制到 pinned CPU payload；
- 释放整段 GPU block，请求状态变为 `SWAPPED`；
- 后续 GPU block 足够时重新分配并 reload，恢复原 PREFILL/DECODE 状态；
- `--max-cpu-offload-mb` 限制 CPU payload 总量。

3B W4A16、强制只给 3 个 GPU block、每个请求 3 blocks、prompt 36 tokens、
max_new=8、连续 5 轮：

| dtype | 完成请求 | swap/resume 每轮 | offload copy | reload copy |
|---|---:|---:|---:|---:|
| FP16 | 10/10 | 1/1 | 1.46 ms | 0.26 ms |
| FP8 | 10/10 | 1/1 | 1.12 ms | 0.26 ms |

没有 OOM，也没有请求丢失。被换出的第一个请求 TPOT 约 0.18–0.19 s，第二个
请求约 0.07 s；这个差异不是 kernel 变慢，而是 token interval 包含了整个
swap 等待窗口。CPU offload 是 OOM 压力释放阀，不是吞吐优化。

原始数据：

- `docs/results/cpu_offload_3b_fp16.json`
- `docs/results/cpu_offload_3b_fp8.json`
- `docs/results/paged_concurrency_3b_fp8kv.json`

## 11. 仍未完成

- prefill + decode 真正的单次 forward mixed batch；
- tiled W4/W3 GEMM 的 Tensor Core / cp.async / double buffering；
- TPOT 抖动闭环、time-sliced prefill、decode stream 优先级；
- dynamic batch CUDA Graph；
- 按 block 热度的细粒度 cold/hot 迁移（当前是整请求 swap）；
- CPU offload 的 SLA 感知调度和后台预取；
- 7B 在 4 GB 下的稳定性 sweep（已有 mixed smoke，但性能不稳定）。
