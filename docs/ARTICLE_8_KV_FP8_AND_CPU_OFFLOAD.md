# 第八篇：KV FP8 与 CPU cold-block offload

> 系列第八篇·KV 量化与冷块迁移
> 日期：2026-10-27
> 硬件：NVIDIA RTX 3050 Laptop GPU，4 GB，SM86
> 软件：Python 3.10.3、torch 2.5.1+cu121、CUDA 12.1、CuPy 14.2
> 模型：Qwen2.5-3B W4A16
> 定位：model_executor + engine-core 原型，不是完整 DeepInfer engine

---

## 0. 结论先行

上一篇留下了两个明确的 KV 缺口：

1. KV 只有 FP16 / INT8，没有 FP8；
2. GPU block 不足时只能拒绝请求，没有 CPU cold-block offload。

这一篇把两者都做了，并且没有把它们写成“已经生产可用”。

最终结果：

```text
FP8 E4M3 KV block bytes              294912
FP16 KV block bytes                  589824
FP8 native paged decode              cosine 1.0 vs reference
FP8 c1 / c2 / c4 tokens/s            11.089 / 17.317 / 22.527
FP16 tiled + overlap c1/c2/c4        11.189 / 17.484 / 22.779

CPU swap smoke（3 GPU blocks）       10/10 requests complete
每轮 swap/resume                     1/1
FP16 offload/reload copy             1.46 ms / 0.26 ms
FP8 offload/reload copy              1.12 ms / 0.26 ms
```

最重要的判断：

- FP8 在这次实现里是**真正的 native paged decode 存储格式**，不是只在
  PyTorch 侧存成 float8、decode 时再 materialize 成 dense；
- CPU offload 是**请求级整段 used-block swap**，不是 block 级别冷热迁移；
- offload 会显著拉长被换出请求的 token interval，它是 OOM 压力释放阀，
  不是吞吐优化；
- FP8 省的是 KV 容量，INT8 也省容量，但 INT8 有 scale sidecar 和反量化成本，
  FP8 没有 scale、直接 cast，两者的数值行为不同，不能混为一谈。

## 1. 起点和边界

上一阶段的 paged KV 支持：

```text
fp16 -> 模型 compute dtype
int8 -> int8 + 每 token/head 的 FP16 K/V scale
```

block 估算也只认识 `int8`，其它名字都退回 compute dtype。

这带来两个问题：

- 没有 FP8 路径，4 GB 上不能进一步压缩 KV；
- `kv_cache_dtype` 看起来是字符串，实际上只有很窄的合法集合，错误会被
  静默吃成“compute dtype”。

所以这次先把 dtype 解析做成显式的：

```text
fp16 / auto        -> compute dtype
int8 / i8          -> torch.int8
fp8 / e4m3         -> torch.float8_e4m3fn
fp8_e5m2 / e5m2    -> torch.float8_e5m2
其它               -> ValueError
```

`paged_kv_block_bytes()` 也改用同一个 resolver，因此显存预算和真实 pool
不会出现“预算按 FP16 算、实际按 FP8 分”的错位。

## 2. KV FP8

### 2.1 为什么先做直接 cast，而不是 scale sidecar

INT8 KV 的做法是：

```text
每 token/head 求 absmax
scale = absmax / 127
q = round(x / scale)
存储 q + scale
```

FP8 有两种完全不同的取舍：

- E4M3：约 3 位尾数，动态范围较小但精度比 E5M2 好；
- E5M2：约 2 位尾数，动态范围大但精度更粗。

第一版没有给 FP8 加 per-token scale，原因是：

1. FP8 本身有指数位，K/V 的常见范围不需要 INT8 那样强的 per-token 缩放；
2. 加 scale 会让 block bytes 增加 sidecar，并给 kernel 增加一次 scale load；
3. 目标先是“支持 FP8 存储和 native decode”，不是追求极致精度的 FP8 方案。

因此当前 FP8 是：

```text
pool[key/value] = key/value.to(float8_dtype)
decode: 直接读取 FP8 bit pattern，在 kernel 内转回 float
```

没有 K/V scale sidecar。这是当前实现的明确边界。

### 2.2 FP8 的 block bytes

`paged_kv_block_bytes()` 现在的行为：

| dtype | K/V 数据 | scale sidecar | block bytes（3B 配置） |
|---|---:|---:|---:|
| FP16 | 2 bytes | 无 | 589824 |
| INT8 | 1 byte | FP16 K/V scale | 299520 |
| FP8 | 1 byte | 无 | 294912 |

FP8 比 INT8 略小，因为 INT8 还有每个 token/head 的 scale。这个差异在
大 KV 池上会体现出来，但在 3B 短上下文里主要还是一个容量选项。

### 2.3 第一个坑：PyTorch CPU 不支持 float8 的 index/index_copy

FP8 storage 本身没有问题：

```python
torch.empty(..., dtype=torch.float8_e4m3fn)
tensor.to(torch.float8_e4m3fn)
```

但 paged cache 不是简单的连续张量，它需要：

```python
self.key_pool[safe_block_table]          # gather blocks
self.key_pool.index_select(...)          # offload
self.key_pool.index_copy_(...)           # reload
```

第一次跑 CPU 回归时直接炸：

```text
RuntimeError: "index_cpu" not implemented for 'Float8_e4m3fn'
RuntimeError: "index_copy_cpu" not implemented for 'Float8_e4m3fn'
```

这不是我们的 shape 写错，而是当时 torch 2.5.1 CPU 对 float8 的索引算子
没有实现。GPU 上有些算子可能可用，但代码不能只保证 CUDA。

解决办法不是把 FP8 退回 FP16，而是把 FP8 tensor 当成 **uint8 storage view**
来做索引：

```python
def _storage_view(tensor):
    if is_fp8_dtype(tensor.dtype):
        return tensor.view(torch.uint8)
    return tensor
```

gather/index_select 在 uint8 上执行，再 view 回 FP8：

```python
selected = _storage_view(tensor).index_select(dim, index)
return _restore_storage(selected, tensor.dtype)
```

reload 也是：

```python
_storage_view(destination).index_copy_(
    dim, index, _storage_view(source)
)
```

这个做法的前提是 FP8 和 uint8 的 element size 都是 1 byte，且 bit pattern
原样复制。它不改变 FP8 的数值逻辑，只是绕开当时不支持的索引算子。

### 2.4 第二个坑：原生 FP8 kernel 的 NVRTC 编译

为了让 FP8 不是“存储成 FP8、decode 再回 SDPA”，我们加了独立的
`dp_paged_gqa_decode_fp8` kernel。

第一版编译失败：

```text
error: identifier "INFINITY" is undefined
```

原因是 NVRTC 的默认 include 集合并不保证 `INFINITY` 宏可见。修复不是
乱加头文件，而是显式构造 IEEE754 inf：

```cuda
return sign ? -__int_as_float(0x7f800000)
            :  __int_as_float(0x7f800000);
```

E5M2 的 exponent=31, mantissa=0 是 inf；mantissa!=0 是 NaN。
E4M3FN 的 exponent=15, mantissa=7 是 NaN，mantissa 其余值是有限数。
FP8 转换函数在 kernel 内用 `fp8_kind` 区分：

```text
0 -> E4M3
1 -> E5M2
```

### 2.5 为什么不在 kernel 里用 torch.float8 指针

CuPy bridge 最初只为 FP16/INT8 准备了 dtype 映射。直接给 RawKernel 传
`torch.float8_e4m3fn` 指针会遇到两个问题：

- 当时 CuPy 对 float8 的 dtype 映射不完整；
- CUDA kernel 的 `__half` 转换函数也不接受 float8 指针。

所以 wrapper 做的是：

```python
key_pool = cache.key_pool.contiguous().view(torch.uint8)
value_pool = cache.value_pool.contiguous().view(torch.uint8)
```

kernel 接收 `const unsigned char*`，在 device 内逐字节 decode 成 float。
这比在 Python 侧先转 FP16 再进 kernel 少了一次大 tensor materialize。

### 2.6 FP8 数值验证

`scripts/validate_sm86_kernels.py` 现在同时验证：

- FP16 native paged attention；
- INT8 native paged attention；
- FP8 E4M3 native paged attention；
- FP8 E5M2 native paged attention；
- W4 batched GEMV；
- 真实 3B checkpoint 的 native vs SDPA 单 token logits。

当前结果：

| 项目 | cosine | max abs error |
|---|---:|---:|
| paged attention FP16 | 1.0 | 0.0009765625 |
| paged attention INT8 | 1.0 | 0.0009765625 |
| paged attention FP8 E4M3 | 1.0 | 0.0009765625 |
| paged attention FP8 E5M2 | 0.99999988 | 0.0009765625 |
| batched W4 GEMV | 0.99999994 | 0.0009765625 |
| 3B end-to-end native vs SDPA | 0.99999875 | 0.03515625 |

E5M2 的 cosine 略低于 E4M3，符合它只有 2 位尾数的预期。3B end-to-end
的 max abs 误差比 synthetic 大，是 FP16/softmax 采样链路和 logits 尺度共同
造成的；argmax 仍然一致。

复现：

```powershell
F:\QwenQLoRA\venv\Scripts\python.exe `
  scripts\validate_sm86_kernels.py `
  --checkpoint D:/Qwen3B-W4A16-SM86 `
  --output docs/results/sm86_kernel_validation_3b.json
```

### 2.7 FP8 端到端结果

同一套 3B W4A16、`max_cache=256`、`block_size=16`、`prefill_chunk_tokens=32`、
`max_new_tokens=16`、tiled prefill + dual-stream overlap：

| 并发 | tokens/s | TTFT p50 | TPOT p50 | TPOT p95-p50 |
|---:|---:|---:|---:|---:|
| 1 | 11.089 | 0.345 s | 0.0732 s | 0.0042 s |
| 2 | 17.317 | 0.513 s | 0.0730 s | 0.0059 s |
| 4 | 22.527 | 1.345 s | 0.1061 s | 0.1547 s |

对照 FP16 tiled + overlap：

| 并发 | FP16 tokens/s | FP8 tokens/s | 差异 |
|---:|---:|---:|---:|
| 1 | 11.189 | 11.089 | -0.9% |
| 2 | 17.484 | 17.317 | -1.0% |
| 4 | 22.779 | 22.527 | -1.1% |

这个结果比预期好。原因是：

- KV read 不是 3B decode 的唯一瓶颈，W4 权重读取仍占大头；
- FP8 没有 scale sidecar，kernel 内直接做 FP8 → float；
- FP8 native decode 避免了 SDPA 的 dense materialize。

在 native FP8 kernel 之前，FP8 走 SDPA fallback，c4 只有约 21.4 tokens/s；
native kernel 之后补到了 22.5，已经接近 FP16。这是本次实现里收益最明确的一块。

原始结果：

- `docs/results/paged_concurrency_3b_fp8kv.json`
- `docs/results/sm86_kernel_validation_3b.json`

## 3. CPU cold-block offload

### 3.1 为什么先做请求级 swap，而不是 block 级迁移

真正理想的 cold-block offload 应该是：

```text
按 block 热度选 victim
  -> 只把冷 block 移到 CPU
  -> hot block 原地保留
  -> 访问冷 block 时按需 page-in
```

这要求 attention 路径支持 block table 中混有“CPU block”的逻辑 id，并且
要处理 page fault、异步 prefetch、block 重映射和 partial block。

这次选择先做 **请求级整段 used-block swap**：

```text
GPU block 不够
  -> 选一个最久未活跃的 PREFILL/DECODE 请求
  -> 把它已使用的 KV block 复制到 pinned CPU
  -> 释放这个请求的全部 GPU block
  -> 请求状态 SWAPPED
  -> 后续 GPU block 足够时重新分配、reload、恢复原状态
```

它没有 block 级热度，但有几个优点：

- 不破坏现有 attention kernel 的 block table 假设；
- 可以复用现有的 `block_ids + seq_lens + prefill_done` 状态；
- 正确性容易用 CPU 测试覆盖；
- 在 4 GB 这种小池子上，先有压力释放阀比追求细粒度更重要。

### 3.2 CPU payload 的形态

`PagedKVCacheManager.offload_blocks()` 复制的是**已经使用的完整 block**：

```text
key   [num_layers, used_blocks, kv_heads, block_size, head_dim]
value [num_layers, used_blocks, kv_heads, block_size, head_dim]
scales（仅 INT8）
  [num_layers, used_blocks, kv_heads, block_size]
```

为什么要带 `num_layers`？

- 一个请求的 KV 分布在所有层；
- 如果按层分别存 Python list，恢复时要创建大量小对象；
- 一次 index_select + 连续 CPU tensor 更容易做 pinned memory 和批量
  GPU→CPU copy。

GPU→CPU copy 在 CUDA 上完成后再 `.pin_memory()`：

```python
key = key_pool.index_select(1, index).detach().to("cpu", copy=True)
key = key.pin_memory()
```

reload 时：

```python
key = payload.key.to(device, dtype=pool.dtype, non_blocking=True)
pool.index_copy_(1, index, key)
```

FP8 的 payload 仍然通过 uint8 storage view 完成 CPU copy，避免 CPU float8
index 不支持的问题。

### 3.3 Scheduler 状态：SWAPPED 不是 FINISHED

新增状态：

```text
PREFILL
DECODE
SWAPPED
FINISHED
```

请求被换出时保留：

```text
request_id
prompt_ids / prompt_length
generated_ids
cached_tokens
prefill_done
max_new_tokens
stop / stream_queue
state_before_swap
offload_payload
```

`state_before_swap` 很关键。

第一版只允许换出 DECODE 请求，后来发现这会在并发 admission 时卡住：

- 调度器在一个循环里连续 admission；
- 第一个请求刚被 admit，状态还是 PREFILL；
- 第二个请求立刻到来，GPU block 已被第一个请求预留；
- 如果只能换出 DECODE，第二个请求根本没有 victim 可选，只能 OOM。

最后改成 PREFILL 和 DECODE 都可以换出：

```text
PREFILL:
  used_tokens = request.prefill_done
  resume state = PREFILL

DECODE:
  used_tokens = request.cached_tokens
  resume state = DECODE
```

`prefill_done` 表示已经写进 paged KV 的 prompt token 数；恢复后从原位置继续
prefill，不需要重新算整段 prompt。

### 3.4 Admission、victim 选择和 resume

Admission 需要 `new_block_count` 个 block，但池子不够时：

```python
need = new_block_count - available_blocks
_make_room_for(need)
if available_blocks < new_block_count:
    raise CacheOutOfMemoryError(...)
```

`_make_room_for()` 按 `last_active_time` 从旧到新选 candidate：

```text
旧请求 -> 先 swap
最近产生 token 的请求 -> 后 swap
```

只考虑当前有 KV 数据的 PREFILL/DECODE 请求。`SWAPPED` 和 `FINISHED` 不会
再次被 swap。

换出一个请求时：

1. 用 `used_blocks` 复制 CPU payload；
2. 检查 CPU payload 是否超过 `--max-cpu-offload-mb`；
3. 只有 payload 被接受后才释放 GPU block；
4. 从 `running` 移除，放进 `swapped`；
5. 记录 swap 指标。

恢复时：

1. 先尝试 `_resume_swapped_one()`，再处理新到达的请求；
2. 重新计算 `required_blocks`；
3. 分配 GPU block；
4. 把 payload 的前 `used_blocks` 个 block reload 到新 block id；
5. 恢复 PREFILL/DECODE 状态和 `last_active_time`；
6. 清空 CPU payload 计数。

恢复顺序很重要：如果先处理新请求，swap 队列可能长期饥饿。当前策略是
swapped 优先，Waiting queue 后处理。

### 3.5 CPU budget 与指标

新增配置：

```text
--cpu-offload
--max-cpu-offload-mb 2048
```

预算不是按 GPU block 数算，而是按真实 payload 字节数算：

```text
payload.nbytes = key + value + optional INT8 scales
```

新增指标：

```text
requests_swapped_total
requests_resumed_total
kv_cpu_offload_blocks_total
kv_cpu_offload_bytes_total
kv_cpu_offload_failed_total
kv_cpu_offload_rejected_total
kv_cpu_reload_blocks_total
cpu_offload_seconds
cpu_reload_seconds
swapped_requests
cpu_offload_bytes
cpu_offload_budget_bytes
cpu_offload_enabled
max_swapped_requests
```

状态里也会出现：

```json
"swapped": 1,
"cpu_offload": {
  "enabled": true,
  "payload_bytes": 3538944,
  "budget_bytes": 268435456,
  "max_swapped_requests": 1
}
```

### 3.6 第一个 offload 坑：不能先 free 再 copy

正确顺序是：

```text
copy GPU -> CPU
  -> validate CPU budget
  -> free GPU blocks
```

如果先 free GPU block，allocator 可能立刻把 block 分给别的请求，随后
CPU copy 读到的就是被覆盖的数据。

当前实现只有在 `offload_blocks()` 成功并且 payload 通过 budget 检查后，才
执行 allocator.free。这个顺序在测试里专门被覆盖。

### 3.7 第二个坑：共享 prefix 会让 resume 需要更多 block

如果请求的 prompt 命中 prefix cache：

- 原来的 `block_ids` 里有共享 prefix block；
- 请求只拥有部分新 block 的独立引用；
- swap 时复制的是“已经使用的数据”，包括共享 prefix 的数据副本；
- reload 时重新分配的是完全私有的 block。

因此 reload 可能比原来多需要 block：

```text
原来: shared_prefix_blocks + new_blocks
恢复: ceil(cached_or_prefill_tokens / block_size) 私有 block
```

当前实现会在 resume 时重新计算 required blocks；如果池子放不下，就保持
SWAPPED 等待，不会硬塞。这个行为是正确的，但也意味着 prefix cache 和
CPU swap 同时使用时，容量规划要更保守。

### 3.8 第三个坑：offload 会让 TPOT 看起来“突然坏了”

被换出请求的 `inter_token_seconds` 会包含：

```text
swap copy
+ 等待其他请求释放 GPU block
+ reload
```

所以 offload 测试里：

```text
第一个请求 TPOT ≈ 0.18-0.20 s
第二个请求 TPOT ≈ 0.07-0.08 s
```

这不是 kernel 退化，而是第一个请求在 token 之间经历了整段 swap 停顿。
`cpu_offload_seconds` 和 `cpu_reload_seconds` 只统计 copy 本身，不能代表
请求总停顿；真正的 SLA 应该看 per-request inter-token interval。

如果以后要把 offload 做成性能特性而不是 OOM 兜底，需要：

- 后台 prefetch；
- decode 优先级；
- 允许 hot block 不迁移；
- 把 swap 从请求关键路径移走。

### 3.9 实测结果

测试条件：

```text
Qwen2.5-3B W4A16
max_cache=128
block_size=16
max_new_tokens=8
每个请求恰好 3 个 GPU block
总 GPU block = 3
max_cpu_offload_mb=256
5 iterations
```

每个 iteration 都先启动一个 streaming 请求，收到首个 token 后再提交第二个
请求，强制触发一次 swap。

| dtype | 完成请求 | swap/resume 每轮 | offload copy | reload copy |
|---|---:|---:|---:|---:|
| FP16 | 10/10 | 1/1 | 1.46 ms | 0.26 ms |
| FP8 | 10/10 | 1/1 | 1.12 ms | 0.26 ms |

热身之后的单轮 wall time 约 1.5–1.7 s。由于 GPU 只有 3 个 block，两个请求
不能同时驻留，调度顺序是：

```text
request A 生成首 token
  -> request B 到达，A swap 到 CPU
  -> B prefill + decode 完成
  -> A reload，继续 decode
  -> A 完成
```

被换出的 A TPOT：

```text
FP16: 0.179-0.186 s
FP8:  0.195-0.201 s
```

B 的 TPOT：

```text
FP16: 0.071-0.076 s
FP8:  0.078-0.082 s
```

这不是 FP8 独有的问题，而是 swap 等待窗口进入了 A 的 token interval。
如果只看 aggregate tokens/s，会把这类停顿藏掉。

原始数据：

- `docs/results/cpu_offload_3b_fp16.json`
- `docs/results/cpu_offload_3b_fp8.json`

## 4. 正确性验证

CPU 回归：

```text
53 passed, 6 skipped
ruff: All checks passed
```

新增覆盖：

- FP8 E4M3 paged cache 写入/读取/scale sidecar 不存在；
- INT8/FP8 CPU payload offload roundtrip；
- FP8 block bytes 是 FP16 的一半、且小于 INT8；
- scheduler FP8 KV 生成；
- DECODE 请求 swap/resume 后继续生成；
- PREFILL 请求 swap/resume 后继续 prefill；
- CPU budget=0 时拒绝 swap，原请求继续完成。

GPU 验证：

- FP8 E4M3 native paged attention vs reference；
- FP8 E5M2 native paged attention vs reference；
- FP8 c1/c2/c4 端到端 sweep；
- FP16/FP8 CPU offload 强制 swap smoke。

复现命令：

```powershell
# CPU
D:\QwenNativeGPU-venv\Scripts\python.exe -m pytest -q tests

# FP8 + native kernel
F:\QwenQLoRA\venv\Scripts\python.exe `
  scripts\validate_sm86_kernels.py `
  --checkpoint D:/Qwen3B-W4A16-SM86 `
  --output docs/results/sm86_kernel_validation_3b.json

# FP8 并发
F:\QwenQLoRA\venv\Scripts\python.exe `
  scripts\bench_paged_scheduler.py `
  --checkpoint D:/Qwen3B-W4A16-SM86 `
  --backend w4a16 `
  --max-cache 256 `
  --max-new-tokens 16 `
  --prefill-chunk-tokens 32 `
  --concurrency 1,2,4 `
  --kv-cache-dtype fp8 `
  --no-prefix-cache `
  --output docs/results/paged_concurrency_3b_fp8kv.json

# 强制 CPU offload（每个请求刚好用满 3 个 GPU block）
F:\QwenQLoRA\venv\Scripts\python.exe `
  scripts\bench_cpu_offload.py `
  --checkpoint D:/Qwen3B-W4A16-SM86 `
  --backend w4a16 `
  --max-cache 128 `
  --max-new-tokens 8 `
  --block-size 16 `
  --kv-cache-dtype fp16 `
  --max-cpu-offload-mb 256 `
  --iterations 5 `
  --output docs/results/cpu_offload_3b_fp16.json
```

FP8 offload 把命令里的 `--kv-cache-dtype` 换成 `fp8` 即可。

## 5. 当前限制

必须写清楚：

- FP8 是直接 cast，没有 per-token scale；
- E4M3 与 E5M2 精度/范围不同，不能当成一个 fixed-point 格式；
- FP8 native kernel 使用软件位解码，没有 Tensor Core FP8（SM86 本身也没有
  原生 FP8 矩阵单元）；
- CPU offload 是整请求 swap，不是 block 级 hot/cold migration；
- swap 会在被换出请求的 token 间产生长间隔；
- prefix cache 命中请求 swap 后 reload 成私有 block，可能多需要 block；
- CPU payload 使用 pinned memory，默认上限 2048 MB，需要按机器 RAM 调整；
- 没有 GPU->CPU 后台预取，没有 swap 队列优先级，没有 recompute preemption；
- 没有真实的 7B 4 GB offload 稳定性数据；本次 offload 压测聚焦 3B。

## 6. 文件地图

| 文件 | 作用 |
|---|---|
| `deepinfer/engine/kv_cache.py` | FP8 存储、CPU payload offload/reload、INT8 scale |
| `deepinfer/engine/paged_attention.py` | FP16/INT8/FP8 native paged decode |
| `kernels/qwen2_paged_gqa_decode_fp8.cu` | 导出的 FP8 paged decode 源 |
| `apps/paged_scheduler.py` | SWAPPED 状态、victim 选择、resume、CPU budget |
| `apps/server.py` | `--kv-cache-dtype fp8`、`--cpu-offload` CLI |
| `deepinfer/engine/metrics.py` | swap/reload/CPU 指标 |
| `scripts/bench_paged_scheduler.py` | FP8 并发 sweep |
| `scripts/bench_cpu_offload.py` | 强制 swap 的可复现实验 |
| `scripts/bench_tpot_jitter.py` | offload 场景 TPOT/ITL 对照 |
| `scripts/validate_sm86_kernels.py` | FP8/INT8/FP16 native kernel 校验 |
| `tests/engine/test_kv_cache.py` | FP8 + CPU payload roundtrip |
| `tests/engine/test_memory_budget.py` | FP8 block 预算 |
| `tests/apps/test_paged_scheduler.py` | DECODE/PREFILL swap、budget=0 |
| `docs/results/sm86_kernel_validation_3b.json` | native kernel 数值结果 |
| `docs/results/paged_concurrency_3b_fp8kv.json` | FP8 并发结果 |
| `docs/results/cpu_offload_3b_fp16.json` | FP16 swap 结果 |
| `docs/results/cpu_offload_3b_fp8.json` | FP8 swap 结果 |

## 7. 一句话总结

这次把 DeepInfer-SM86 的 KV 从：

```text
FP16 / INT8
GPU OOM 只能拒绝
```

推进到了：

```text
FP16 / INT8 / FP8 E4M3 / FP8 E5M2
native paged decode 覆盖三种 KV 类型
请求级 CPU cold-block swap + 自动 resume
可复现的容量、延迟、TPOT 代价数据
```

但也要明确：

> FP8 的主要收益是 KV 容量，不是魔法提速；CPU offload 的主要收益是避免
> OOM，不是提高吞吐。把这两个边界写清楚，比把它们包装成“全面优化”更重要。
