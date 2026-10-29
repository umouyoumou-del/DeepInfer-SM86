# ECS on 2×RTX 3080 Ti

This document records the first real multi-GPU ECS validation. The host is a
temporary two-GPU substitute for the unavailable 8×P100/NCCL machine.

## Hardware and software

| Item | Value |
|---|---|
| GPU | 2× NVIDIA GeForce RTX 3080 Ti, 12 GB, SM 8.6 |
| CPU | Intel Xeon Silver 4214R, 48 hardware threads |
| RAM | 440 GiB |
| OS | Ubuntu 22.04.4 |
| Python | 3.12.3 |
| PyTorch | 2.5.1+cu121 |
| CUDA runtime | 12.1 |
| CuPy | 14.2.0 |
| Transformers | 4.49.0 |
| Model | DeepSeek-V2-Lite-Chat Q4_K_M, 10,364,416,320 bytes |
| Model SHA-256 | `30b4fb4ab1fbe1c6a827303ea898296c9faf8f54d6cdf8b9fbdda7a7ebfb292a` |

The two GPUs do not expose peer-to-peer access (`can_device_access_peer=False`
in both directions), so cross-device hidden states and packet exchange go
through host/PCIe/NCCL as appropriate.

## Environment bootstrap

```bash
python3.12 -m venv /root/autodl-tmp/ecs-venv
/root/autodl-tmp/ecs-venv/bin/pip install --upgrade pip wheel setuptools
# torch 2.5.1+cu121 (use the cached/system CUDA libraries available on host)
/root/autodl-tmp/ecs-venv/bin/pip install \
  torch==2.5.1 typing_extensions filelock fsspec jinja2 networkx sympy
/root/autodl-tmp/ecs-venv/bin/pip install \
  cupy-cuda12x==14.2.0 transformers==4.49.0 accelerate gguf safetensors \
  numpy pytest psutil
apt-get install -y build-essential aria2
```

The GGUF was downloaded from the ModelScope mirror because direct
Hugging Face access timed out from the host:

```bash
aria2c -c -x16 -s16 -k1M --file-allocation=none \
  -o DeepSeek-V2-Lite-Chat-Q4_K_M.gguf \
  https://modelscope.cn/models/second-state/DeepSeek-V2-Lite-Chat-GGUF/resolve/master/DeepSeek-V2-Lite-Chat-Q4_K_M.gguf
```

## Results

Single GPU baseline:

| Metric | Value |
|---|---:|
| Output | `5-2=3` |
| Prompt tokens | 17 |
| Completion tokens | 7 |
| Prefill | 0.90 s |
| Decode | 1.43 s |
| Decode rate | 4.20 tok/s |
| Peak GPU allocated | 2.20 GiB |

Two-stage layer pipeline, split after layer 12:

- GPU0 executes embedding + layers 0–12.
- GPU1 executes layers 13–26 + final norm + LM head.
- Hidden state is copied between GPUs once per stage per token.
- Exact token parity with the single-GPU baseline:
  `[207, 20, 12, 17, 28, 18, 100001]`.

| Metric | Value |
|---|---:|
| Output | `5-2=3` |
| Prefill | 1.61 s |
| Decode | 1.26 s |
| Decode rate | 4.76 tok/s |
| Peak GPU0 / GPU1 | 2.22 / 2.22 GiB |
| Load time (both runtimes) | 32.89 s |

The current layer-pipeline smoke loads the full runtime on both GPUs and
executes only the assigned layer ranges. It validates two-GPU execution,
cross-GPU hidden-state transfer, and token parity, but it is not yet a
memory-optimized layer-sharded loader.

Dual-GPU ECS worker pool (one independent runtime per GPU, queue-depth
assignment):

| Metric | Value |
|---|---:|
| Two concurrent requests | `5-2=3`, `2+3=5` |
| Assigned devices | `[0, 1]` |
| Wall time | 4.77 s |
| Per-stream decode | 2.88 / 2.82 tok/s |
| Aggregate output | both correct |

NCCL ECS packet All-to-All:

- `torch.distributed` backend: NCCL.
- Hot expert 7 was token-sharded 2/2 across the two GPUs.
- Expert 8 was routed to rank 1.
- Rank 0 sent 3 packets; rank 1 received 2 packets
  (`[101, 102]` for expert 7 and `[201, 202]` for expert 8).
- Rank 0 received its local shard `[103, 104]`.

## Expert-sharded single-model execution

The runtime now supports `--expert-devices 0 1`.  With ECS enabled, routed
experts receive an ECS route-placement owner on GPU0 or GPU1.  Each expert
chunk is split by owner, the owner stages its packed expert bytes and runs the
grouped gate/up/down kernels on that GPU, and the chunk output is copied back
to the coordinator GPU.

Command:

```bash
PYTHONPATH=$PWD /root/autodl-tmp/ecs-venv/bin/python -u   scripts/smoke_deepseek_v2_lite_moe.py   --model /root/autodl-tmp/models/DeepSeek-V2-Lite-Chat-GGUF/DeepSeek-V2-Lite-Chat-Q4_K_M.gguf   --meta-dir /root/autodl-tmp/meta   --device cuda:0 --expert-devices 0 1 --ecs   --prompt '只回答数字：5-2=?' --max-new-tokens 8 --warmup-tokens 2   --max-cache 128 --output /root/autodl-tmp/results/ecs_expert_shard.json
```

| Metric | Value |
|---|---:|
| Output | `5-2=3` |
| Token IDs | `[207, 20, 12, 17, 28, 18, 100001]` |
| Token parity vs single GPU | exact |
| Expert dispatches on GPU0 / GPU1 | 1416 / 1415 |
| Expert chunk dispatches | 834 |
| Prefill | 1.42 s |
| Decode | 3.25 tok/s |
| Coordinator peak | 2.24 GiB |

This is slower than the single-GPU 4.20 tok/s because every expert chunk now
has cross-PCIe activation/output copies and the coordinator still runs all
attention/shared layers.  It proves correctness and balanced expert ownership;
it is not yet a throughput win.

The current path uses direct `Tensor.to()` copies between the two GPUs.  The
NCCL ExpertPacket All-to-All above is the transport path for the next revision
of in-model activation exchange.

## NCCL expert-parallel runtime

The next revision removes the duplicated non-routed weights from the remote
rank.  Rank 0 keeps attention, shared experts and the LM head; rank 1 is a
lightweight routed-expert executor that only opens the GGUF expert bytes.
Requests and activation tensors travel over NCCL send/receive.

Command:

```bash
PYTHONPATH=$PWD /root/autodl-tmp/ecs-venv/bin/torchrun   --standalone --nproc_per_node=2   scripts/bench_ecs_expert_parallel_nccl.py   --max-new-tokens 8 --max-cache 128   --output /root/autodl-tmp/results/ecs_expert_parallel_nccl.json
```

| Metric | Value |
|---|---:|
| Output | `5-2=3` |
| Token IDs | `[207, 20, 12, 17, 28, 18, 100001]` |
| Token parity vs single GPU | exact |
| Remote expert requests | 294 |
| Expert chunk dispatches | 591 |
| GPU0 / GPU1 expert dispatches | 996 / 984 |
| NCCL bytes each way | 21,344,256 |
| NCCL request/reply latency | 0.375 s total |
| Prefill | 2.62 s |
| Decode | 3.44 tok/s |
| Coordinator peak | 2.21 GiB |
| Remote worker peak | **37.7 MiB** |

The remote rank does not load the transformer model: its peak allocation is
only the expert staging buffer and CUDA context.  This addresses the previous
duplicate-non-routed-weight problem.

The exchange is currently blocking NCCL point-to-point request/reply, not a
pipelined collective all-to-all.  It is correct but slower than the single-GPU
baseline (4.20 tok/s); removing blocking and overlapping expert chunks is the
next throughput task.

## What is and is not validated

Validated on hardware:

- real dual-GPU PyTorch/CuPy execution;
- real C++ ECS core on Linux (`cpp` backend);
- real NCCL ExpertPacket all-to-all;
- hot-expert token sharding across two physical GPUs;
- real cross-GPU layer pipeline with exact token parity;
- queue-depth worker-pool request assignment.

Still not validated:

- NVLink/P2P (this host reports no P2P);
- 8×P100 or any multi-node fabric;
- a pipelined collective All-to-All as the in-model activation exchange
  (the current NCCL expert-parallel path uses blocking point-to-point
  send/receive; the separate ExpertPacket all-to-all smoke did pass);
- throughput above the single-GPU baseline (expert parallelism currently adds
  request/reply latency);
- production heartbeat/worker restart for the dual-GPU pool.

The next target is pipelining the NCCL expert requests (or switching to a
collective all-to-all) so expert chunks overlap with coordinator attention,
then measuring whether the two-GPU path beats the 4.20 tok/s single-GPU
baseline.
