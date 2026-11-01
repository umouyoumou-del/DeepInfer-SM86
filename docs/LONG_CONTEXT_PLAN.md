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
