# 长上下文：1M 的账，以及真正的拦路虎

> 目标：用 ECS 的 KV 管理把上下文推到 1M。
> 结论：**当前实现里 KV 被浪费了 10.7 倍，这是唯一的硬阻断；修掉它 1M 才可能。**

---

## 1. 实测：paged MLA pool 每 token 324 KiB

```python
PagedKVCacheManager(num_layers=27, num_kv_heads=16, head_dim=192,
                    block_size=16, num_blocks=64, dtype=torch.float16)
# memory_bytes() / (64 * 16) = 331,776 bytes = 324 KiB per token
```

而 **MLA 的真实成本只有 30 KiB/token**：

```
27 层 × (kv_lora_rank 512 + qk_rope 64) × 2 字节 = 30 KiB
```

**比值 10.7 倍。**

### 浪费在哪

`paged_dims()` 返回 `head_dim = max(key_dim, value_dim) = 192`，而
`PagedKVCacheManager` 按 `[num_blocks, num_kv_heads=16, block_size, head_dim=192]`
分配**每个 head 一份**。

但 MLA 的定义就是**压缩后所有 head 共享同一份 latent**——
存 16 份等于把 MLA 又退化成 MHA 的存储量，而且比 MHA（head_dim 128）还大。

---

## 2. 这直接决定了 1M 是否可能

| 上下文 | 当前实现（324 KiB/tok） | 真 MLA（30 KiB/tok） |
|---:|---:|---:|
| 32K | 10.1 GiB ✅ | 0.9 GiB ✅ |
| 64K | 20.2 GiB ✅（+权重 9.7 = 30 GiB，极限） | 1.9 GiB ✅ |
| 256K | **81 GiB ❌** | 7.5 GiB ✅ |
| **1M** | **324 GiB ❌** | **29 GiB** ⚠️（+权重 9.7 = 38.6 > 32） |

**修掉浪费之后：**

| 目标 | KV | +权重 9.65 GB | 32 GiB 卡 |
|---|---:|---:|:--:|
| 512K fp16 | 14.6 GiB | 24.3 GiB | ✅ |
| 1M fp16 | 29.2 GiB | 38.8 GiB | ❌ |
| **1M int8** | **14.6 GiB** | **24.3 GiB** | ✅ |

**所以 1M 需要两件事：不作废的 KV 布局 + int8 存储。**

---

## 2.5 布局已修复（本轮）

`paged_mla_attend` 现在按 `kv_heads` 步进（由 `key_pool.shape[1]` 传入），
不再假定等于 query heads。调用方把 pool 按 `num_kv_heads=1` 分配即可。

### 验证

`tests/moe/test_mla_kv_sharing.py`（3 项）：

| 测试 | 内容 |
|---|---|
| 共享布局 == 复制布局 | 同一 latent 分别存 1 份与 16 份，输出逐元素一致（`atol=2e-3`）——**证明这是纯浪费，不是语义** |
| 体积 | 1 份池的元素数恰为 16 份的 1/16 |
| 部分分组 | `kv_heads=4` 时每个 latent 由 4 个 query head 共享，输出有限且形状正确 |

### 容量实测（4080 SUPER，31.47 GiB）

| 上下文 | 修复前 | 修复后 | 状态 |
|---:|---:|---:|---|
| 64K | 20.2 GiB | **1.3 GiB** | ✅ |
| 256K | 81 GiB ❌ | **5.1 GiB** | ✅ |
| **512K** | 162 GiB ❌ | **10.1 GiB** | ✅ **现在装得下** |
| 1M | 324 GiB ❌ | **20.2 GiB** | ❌ 仍差一点 |

`num_kv_heads=1` 时每 token 是 **20,736 bytes（20.25 KiB）**，与真 MLA 推算一致。

### 1M 的失败点精确定位

```
key_pool  10.12 GiB   ✅ 分配成功
value_pool 10.12 GiB  ❌ OOM（可用 18.8 GiB）
```

**1M 需要 20.24 GiB 的 KV，而卡上自由显存只有 18.8 GiB**——
即使不考虑 9 GiB 权重也不够。所以 **int8 是 1M 的唯一路径**：
int8 后 KV 为 10.12 GiB，加权重共约 19 GiB，在 31.47 GiB 内。

---

## 3. 要做的改动

### 3.1 pool 布局：按 head 共享

```
现在: key_pool [blocks, 16, block_size, 192]   ← 16 份冗余
目标: key_pool [blocks,  1, block_size, 576]   ← 512 latent + 64 rope
```

`paged_mla_attend` 里 `kv_head` 索引改成恒为 0，注意力按 head 广播。

### 3.2 int8 支持

`cuda_paged_mla_supported()` 现在要求 `dtype == torch.float16`，
int8 走不了融合 kernel。需要像 `_SM86_PAGED_GQA_I8_KERNEL` 那样加一条
带 per-block scale 的 MLA int8 路径。

### 3.3 验证方式

1. 小规模对照：同一批 token，新布局与旧布局的输出逐元素比对（应完全一致，
   因为只是不再复制同一份数据）；
2. 容量：分配 512K / 1M token 的 blocks，确认 `memory_bytes()` 与推算一致；
3. 速度：在该上下文下跑 decode，记录 ms/token。

---

## 4. 预期数字

按 736 GB/s 的 VRAM 带宽，decode 每 token 要读全部 KV：

| 上下文 | KV 读取 | 理论下限 |
|---:|---:|---:|
| 64K | 1.9 GiB | 2.6 ms |
| 256K | 7.5 GiB | 10.2 ms |
| **1M（int8）** | 14.6 GiB | **19.9 ms → 约 50 tok/s** |

加上权重与算力开销，实际大约在 **20–40 tok/s**。

**注意这不是"能跑"，而是"1M 上下文下仍有几十 tok/s"**——这是本次要证明的东西。

---

## 5. 为什么这是"狠活"

1M 上下文在 32 GiB 卡上，**关键不在 ECS 能搬多少，而在有没有把 MLA 的压缩真的用上**。
当前实现把 MLA 退化成 MHA 的存储量，于是 1M 需要 324 GiB，任何分层都救不回来
（DDR4 每秒 30 GB，324 GiB 一个 token 就要 11 秒）。

**修掉 10.7 倍浪费之后，1M 的 KV 只有 14.6 GiB（int8），可以全程驻留显存——
不需要分层，也就没有分层带来的带宽惩罚。**

---

## 6. 实测：1M 上下文跑通了，1.2 tok/s

布局修好、僵尸进程清掉（它占着 12.7 GiB，是之前 OOM 的真凶）之后：

```
pool 20.25 GiB (20.2 KiB/token)
filled 0.2 s, VRAM used 20.26 GiB
trial 0: attention over 1,048,575 tokens = 1023.4 ms -> 0.98 tok/s (finite)
trial 1: attention over 1,048,575 tokens =  810.5 ms -> 1.23 tok/s (finite)
```

**1M 上下文：容量成立，数值有限，真的跑通了。**

| 项 | 结果 |
|---|---|
| KV 显存 | 20.25 GiB（31.47 GiB 卡的 64%） |
| + 权重 9 GiB | 29.25 GiB，**装得下** |
| 1M 下 decode | **810–1023 ms/token，约 1.2 tok/s** |

### 瓶颈不在带宽，在并行度

```
20.25 GiB ÷ 0.81 s = 25 GB/s = 736 GB/s 的 3.4%
```

**只用到 3.4% 的显存带宽**——所以不是带宽受限。原因是
`paged_mla_attend` 的并行度是 `batch × heads × q_len` = **1 × 16 × 1 = 16 个 warp**，
每个 warp 串行扫过 1,048,575 个位置。128 个 SM 上有 16 个 warp 在跑，
其余全空着，而每个位置都要等一次显存延迟。

**如果按带宽做，1M 的读取下限是 20.25 GiB ÷ 736 GB/s ≈ 28 ms → 约 36 tok/s。**
所以内核还有**约 30 倍**的空间。

### 结论

**问题从"能不能装下"变成了"attention 怎么并行"。**

- 容量：✅ 解决（324 GiB → 20.25 GiB，16 倍）
- 速度：❌ 1.2 tok/s，离带宽下限差 30 倍

下一步是给 MLA attention 加**跨 block 并行**（split-K 风格：把 1M 个位置切成多段，
每段一个 block 并行算部分 softmax 统计量，再做一次两遍归约）。
这是标准做法，但要改内核的 softmax 归约结构，不是小改动。

**在此之前，1M 上下文的正确表述是：可行、正确、但慢——1.2 tok/s。**
