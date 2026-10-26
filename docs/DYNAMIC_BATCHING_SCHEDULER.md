# DeepInfer SM86 动态批处理和调度器

> 系列实现说明：在 ModelSpec 通用化之后，把单请求 SM86 执行器扩展为固定槽位的
> 连续批处理调度器。

## 1. 结论

3B DeepInfer SM86 W4A16、`max_batch_size=2`、256 context：

| 指标 | 串行两个请求 | 两请求并发 | 加速 |
|---|---:|---:|---:|
| wall | 2.5005 s | **1.5764 s** | **1.586×** |
| prompt 1 | 1.4475 s | 1.5753 s | — |
| prompt 2 | 1.0526 s | 1.4870 s | — |

这不是“单请求更快”；而是两个请求同时在 decode 时共享权重读取和 kernel
launch，聚合吞吐提高。

## 2. 组件

```text
HTTP requests
    │
    ▼
DynamicBatchScheduler
  request queue
  fixed slot table
  serial prefill
  batched decode
  per-request sampling / EOS
    │
    ▼
Qwen2Mixed/Qwen2W4A16 model
  W4A16 batch GEMV
  W3A16 batch GEMV
  batched SDPA attention
  batched static KV cache
  batch CUDA Graph (max_batch_size slots)
```

文件：

```text
apps/scheduler.py
apps/server.py
apps/runtime.py
deepinfer/model_executor/models/qwen2.py
deepinfer/model_executor/models/qwen2_w4a16.py
deepinfer/model_executor/models/qwen2_mixed_sm86.py
```

## 3. 调度器行为

- 固定 `max_batch_size` 个 slot；
- 每个请求占用一个 slot；
- 请求先进入队列；
- prefill 目前逐个执行；
- 已 prefill 的请求进入 batched decode；
- 每个 slot 有自己的 prompt、生成 token、采样参数、EOS 状态；
- 完成后释放 slot；
- 新请求可以在旧请求 decode 期间进入，但 prefill 仍然串行；
- no priority / no preemption / no beam search。

## 4. 批处理内核

### W4A16

原 kernel 一个 block 行组只处理一个 token。  
现在用 `blockIdx.y` 选择 batch：

```text
grid.x = ceil(N / rows_per_block)
grid.y = batch_size
x     = x + batch * K
out   = out + batch * N
```

验证：

```text
W4 batch output cos = 1.000000
```

### W3A16

同样增加 batch 维：

```text
W3 batch output cos = 0.99999994
```

### Attention

`qwen2_gqa_decode` 仍然是 batch=1 的 native kernel。  
batch>1 时执行器改走 SDPA：

```python
if native_gqa_decode is not None and sequence == 1 and query.shape[0] == 1:
    attention = native_gqa_decode(...)
else:
    attention = F.scaled_dot_product_attention(...)
```

这是当前 batch attention 的性能上限之一；批量 native GQA 还未实现。

### Static KV cache

原 `index_copy_` 只能表达所有 batch 共用同一个 position。  
动态批处理需要每个 slot 写入自己的 position，因此改成张量索引：

```python
batch_indices = torch.arange(batch)
cache.key[batch_indices, :, positions, :] = key[:, :, 0, :]
```

**关键点：** 不能使用 `int(position)`，因为 `.item()` 会触发 GPU→CPU
同步，在 CUDA Graph capture 中非法。

## 5. Batch CUDA Graph

调度器分配固定 batch cache 后捕获一次：

```text
static_token     [B,1]
static_position  [B,1]
static_mask      [B,max_seq]
batch_cache      [B,KV,max_seq,D] × layers
```

replay 前只更新 tensor 值，不改变 shape：

```python
self.static_token.copy_(input_ids)
self.static_position.copy_(positions)
self.static_mask.copy_(mask)
self.graph.replay()
```

空闲 slot 仍然参与计算，但：

- token 用 EOS；
- position 用 0；
- mask 第一位为 true；
- cache 清零；
- logits 丢弃。

这样保持固定 shape，避免每个请求重新 capture。

## 6. 启动

```powershell
python apps\server.py `
  --checkpoint D:\Qwen3B-W4A16-SM86 `
  --backend w4a16 `
  --model-id deepinfer-sm86-3b-w4a16 `
  --max-cache 256 `
  --scheduler --max-batch-size 2 `
  --port 18001
```

健康检查会返回：

```json
{
  "scheduler": {
    "max_batch_size": 2,
    "graph_captured": true,
    "active": 0,
    "queued": 0,
    "slots": [null, null]
  }
}
```

## 7. 验证

### 静态 cache 每 batch position

```text
tests/models/test_batch_cache.py
```

验证两个 slot 分别写入 position 3 和 5，互不覆盖。

### W4/W3 batch GEMV

手工 GPU 对比显式反量化：

```text
W4 batch cosine = 1.000000
W3 batch cosine = 0.99999994
```

### 并发 HTTP

同一调度器：

```text
serial wall     2.5005 s
concurrent wall 1.5764 s
speedup         1.586x
```

原始结果：

```text
docs/results/dynamic_batching_3b.json
```

## 8. 走过的弯路

### 8.1 native GQA 只支持 batch=1

第一次并发请求直接报：

```text
native GQA decode expects [1, heads, 1, dim], got (2, 16, 1, 128)
```

修复：

- batch=1 继续用 native GQA；
- batch>1 走 SDPA；
- 没有假装 batch kernel 已经存在。

### 8.2 `index_copy_` 不能做 per-batch position

原实现：

```python
self.key.index_copy_(2, positions, key)
```

在 batch>1 时所有 batch 会写同一组 position。  
改成 batch 维张量索引。

### 8.3 `.item()` 破坏 CUDA Graph capture

第一版 batch cache 写入：

```python
position = int(positions[index])
```

非 graph 模式没问题，但 capture 时触发 GPU→CPU sync，CUDA context 进入
错误状态，随后 `_zero_slot()` 继续报：

```text
operation failed due to a previous error during capture
```

修复：

- 全部使用 tensor 索引；
- capture 中不出现 `.item()` / `.cpu()` / Python 条件依赖 GPU 值。

### 8.4 CPU 单测被 batch GEMV 分支打中

把 `flat.shape[0] <= 32` 改成 GEMV 后，CPU 上的 W4 单测也进入 GEMV，
报：

```text
CuPy is unavailable
```

修复：

```python
if flat.shape[0] <= 32 and flat.is_cuda:
```

CPU 继续走 dequant + F.linear。

### 8.5 server 参数一开始没接上

第一次启动报：

```text
unrecognized arguments: --scheduler --max-batch-size 2
```

因为 CLI 参数和 runtime 分支只接了一半。  
后来补齐：

- `--scheduler`
- `--max-batch-size`
- `single_request_cache=False`
- `use_graph=False`（scheduler 自己维护 batch graph）

### 8.6 eager batch 看起来“没有收益”

最初没有 batch graph 时，两请求并发约 3 秒，甚至比串行更慢：

- eager SDPA attention；
- 每 token Python/kernel launch；
- 没有 graph replay。

加入 batch CUDA Graph 后：

```text
3.0 s -> 1.58 s
```

这说明动态批处理不能只做 shape 拼接，必须同时处理 launch 和 attention 路径。

### 8.7 prefill 仍然是串行

当前只批处理 decode：

- prefill 逐请求执行；
- 因此短请求、大 prompt 场景，prefill 仍占主导；
- 下一步需要 padded prefill batch。

不能把当前结果说成“完整 continuous batching engine”。

## 9. 当前边界

- 固定槽位，不是动态变长 tensor batch；
- batch attention 走 SDPA，不是 native batch GQA；
- prefill 未批处理；
- 无优先级、抢占、beam search；
- 4 GB 上 3B batch=2 已可用；7B batch=2 还没有可靠验证；
- 256 context；
- 单机、单进程；
- 仍然不是完整 DeepInfer engine（无 CacheEngine/Sampler/scheduler/worker 全链路）。

## 10. 当前建议

- 3B 日常服务：`--scheduler --max-batch-size 2`；
- 7B：先用 batch=1，避免 4 GB paging；
- 需要低延迟单请求：关闭 scheduler，使用单请求 CUDA Graph；
- 需要聚合吞吐：开启 scheduler。

> 动态批处理解决的是并发吞吐；单请求延迟不一定下降。
