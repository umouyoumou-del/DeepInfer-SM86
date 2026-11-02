"""Greedy decoding over a short latent-cache context, one CUDA graph per step.

:meth:`DeepSeekLongContext.decode` is exact but bound by the host: about 3,400
kernel launches a token, the token embedding looked up on the CPU, and the
runtime's MoE routes on the host (two device-to-host syncs per MoE layer) -
55 ms a token on the 4080 SUPER while the GPU is busy for ~16 of them, and
twice that when the shared host is loaded.  The answers of locate + re-read are
generated in a short context (the re-read lines, a few hundred to a few
thousand slots, fp16, all in VRAM, one position chunk), where a step can run
without any host round trip:

* the embedding row comes from a GPU copy of the table;
* attention is the absorbed form over every slot of the (small) cache with the
  slots past the current position masked, so shapes do not change per token;
* routing stays on the GPU: the gate's top-k ids index per-layer tables of
  pointers to the resident packed experts, which the runtime's pointer-grouped
  Q4 kernels read directly;
* the argmax is written to a device buffer and fed to the next step.

The step is captured once in a CUDA graph and replayed; the host reads the
produced ids every ``sync_every`` steps to look for a stop id (a few steps past
it may already have run - their slots are simply past the session's length).
Same math as ``decode`` in fp32 accumulation; only the rounding differs.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from deepinfer.moe.ptr_grouped_moe import (
    fused_down_gemm_ptr_grouped,
    fused_gate_up_gemm_ptr_grouped,
)

try:
    import cupy as cp
except Exception:  # noqa: BLE001  # pragma: no cover - CPU-only hosts
    cp = None


@dataclass
class _ExpertTable:
    gate: torch.Tensor  # [experts] int64 device pointers to packed rows
    up: torch.Tensor
    down: torch.Tensor
    specs: tuple
    keepalive: tuple  # the packed tensors the pointers point into


def _gpu_embedding(rt) -> torch.Tensor:
    table = getattr(rt, "_graph_decode_embedding", None)
    if table is None:
        weight = rt.model.model.embed_tokens.weight
        table = weight.detach().to(rt.device, torch.float16)
        rt._graph_decode_embedding = table
    return table


def _expert_tables(rt) -> dict[int, _ExpertTable] | None:
    """Per MoE layer, pointers to every routed expert's packed gate / up / down; ``None`` unless all are resident."""

    tables = getattr(rt, "_graph_decode_experts", None)
    if tables is not None:
        return tables
    store = rt.expert_store
    if store is None or not store.gpu_cache_enabled():
        return None
    tables = {}
    for layer_idx in sorted(rt.moe_layer_indices):
        specs = tuple(store._specs[(layer_idx, projection)] for projection in ("gate", "up", "down"))
        entries = store._gpu_expert_cache.get(layer_idx, {})
        count = int(specs[0].expert_count)
        if any(entries.get(expert) is None for expert in range(count)):
            return None
        packed = [entries[expert] for expert in range(count)]
        pointers = [
            torch.tensor([entry[index].data_ptr() for entry in packed], dtype=torch.int64, device=rt.device)
            for index in range(3)
        ]
        tables[layer_idx] = _ExpertTable(*pointers, specs=specs, keepalive=tuple(packed))
    rt._graph_decode_experts = tables
    return tables


def usable(session) -> bool:
    """Whether ``session`` can decode through :class:`GraphDecoder` (``DEEPINFER_GRAPH_DECODE=0`` turns it off)."""

    if os.environ.get("DEEPINFER_GRAPH_DECODE", "1") == "0" or cp is None:
        return False
    cache = session.cache
    if torch.device(session.device).type != "cuda" or cache.kv_dtype != "fp16" or cache.host_tokens:
        return False
    if session.retrieval is not None or session.logn_trained is not None:
        return False
    if session.positions.chunk < cache.capacity:  # every slot in the first position chunk
        return False
    if session.next_position != cache.length or any(p0 != l0 for p0, _, l0 in session.extents):
        return False
    return _expert_tables(session.rt) is not None


class GraphDecoder:
    """Greedy continuation of a short :class:`DeepSeekLongContext`, the step replayed as a CUDA graph."""

    def __init__(self, session, *, steps: int, sync_every: int = 8) -> None:
        if not usable(session):
            raise ValueError("session cannot use graph decoding (see graph_decode.usable)")
        self.session = session
        self.device = torch.device(session.device)
        self.embedding = _gpu_embedding(session.rt)
        self.tables = _expert_tables(session.rt)
        self.sync_every = max(1, int(sync_every))
        capacity = session.cache.capacity
        self.token = torch.zeros(1, dtype=torch.long, device=self.device)
        self.position = torch.zeros(1, dtype=torch.long, device=self.device)
        self.count = torch.zeros(1, dtype=torch.long, device=self.device)
        self.produced = torch.zeros(max(1, int(steps)) + self.sync_every, dtype=torch.long, device=self.device)
        self.slots = torch.arange(capacity, device=self.device)
        top_k = int(session.layers[min(self.tables)].mlp.gate.top_k)
        self.counts = torch.ones(top_k, dtype=torch.int32, device=self.device)
        self.graph: torch.cuda.CUDAGraph | None = None

    # ---------------------------------------------------------------- step

    def _moe(self, layer_idx: int, mlp, x: torch.Tensor) -> torch.Tensor:
        topk_idx, topk_weight, _ = mlp.gate(x.unsqueeze(0))
        ids = topk_idx.view(-1)
        top_k = ids.numel()
        table = self.tables[layer_idx]
        gate_spec, up_spec, down_spec = table.specs
        grouped = x.expand(top_k, x.shape[-1]).reshape(top_k, 1, x.shape[-1]).contiguous()
        activation = fused_gate_up_gemm_ptr_grouped(
            grouped, table.gate.index_select(0, ids), table.up.index_select(0, ids), self.counts,
            n_experts=top_k, max_count=1, out_features=gate_spec.out_features, k_size=gate_spec.in_features,
            gate_qtype=gate_spec.quant_type, up_qtype=up_spec.quant_type,
            gate_row_bytes=gate_spec.row_bytes, up_row_bytes=up_spec.row_bytes,
        )
        out = fused_down_gemm_ptr_grouped(
            activation, table.down.index_select(0, ids), self.counts,
            n_experts=top_k, max_count=1, out_features=down_spec.out_features, k_size=down_spec.in_features,
            down_qtype=down_spec.quant_type, down_row_bytes=down_spec.row_bytes,
        )
        routed = (out.view(top_k, -1).float() * topk_weight.view(top_k, 1).float()).sum(dim=0, keepdim=True)
        return routed.to(x.dtype) + mlp.shared_experts(x)

    def _step(self) -> None:
        """Feed ``self.token`` at ``self.position``; the argmax becomes the next ``self.token``."""

        s = self.session
        positions = s.positions
        pos = self.position
        key_position = positions.key_positions(pos)
        query_position = positions.query_positions(pos, positions.INTRA)
        future = self.slots > pos
        hidden = self.embedding.index_select(0, self.token)
        for index, layer in enumerate(s.layers):
            attention = layer.self_attn
            x = layer.input_layernorm(hidden)
            q = attention.q_proj(x).view(1, s.num_heads, s.nope_dim + s.rope_dim)
            q_nope, q_pe = torch.split(q, [s.nope_dim, s.rope_dim], dim=-1)
            latent, k_pe = torch.split(attention.kv_a_proj_with_mqa(x), [s.latent_dim, s.rope_dim], dim=-1)
            latent = attention.kv_a_layernorm(latent.contiguous())
            keys = s.cache.gpu_latent[index]
            rope_keys = s.cache.gpu_pe[index]
            keys.index_copy_(0, pos, latent.to(keys.dtype))
            rope_keys.index_copy_(0, pos, s.rope.apply(k_pe, key_position).to(rope_keys.dtype))
            weights = s.weights[index]
            q_abs = weights.absorb_query(q_nope[0]).float()
            q_rot = s.rope.apply(q_pe, query_position)[0].float()
            latent_keys = keys.float()
            scores = (q_abs @ latent_keys.t() + q_rot @ rope_keys.float().t()) * weights.softmax_scale
            probs = torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1)
            attended = weights.project_output(probs @ latent_keys)
            hidden = hidden + attention.o_proj(attended.reshape(1, -1))
            x = layer.post_attention_layernorm(hidden)
            if index in self.tables:
                hidden = hidden + self._moe(index, layer.mlp, x)
            else:
                hidden = hidden + layer.mlp(x)
        token = torch.argmax(s._logits(hidden[-1])).view(1)
        self.produced.index_copy_(0, self.count, token)
        self.token.copy_(token)
        self.position.add_(1)
        self.count.add_(1)

    def _run_step(self) -> None:
        # the pointer-grouped kernels are CuPy launches: put them on torch's current stream
        with cp.cuda.ExternalStream(torch.cuda.current_stream(self.device).cuda_stream, self.device.index or 0):
            self._step()

    def _capture(self) -> None:
        state = (self.token.clone(), self.position.clone(), self.count.clone())
        side = torch.cuda.Stream(self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):
            for _ in range(2):  # compile kernels and settle the allocator outside the capture
                self._run_step()
        torch.cuda.current_stream(self.device).wait_stream(side)
        for target, saved in zip((self.token, self.position, self.count), state, strict=True):
            target.copy_(saved)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._run_step()

    # ------------------------------------------------------------- decode

    @torch.inference_mode()
    def greedy(
        self,
        logits: torch.Tensor,
        *,
        max_new_tokens: int,
        stop_ids: Sequence[int] = (),
        on_tokens: Callable[[list[int]], None] | None = None,
    ) -> list[int]:
        """What ``DeepSeekLongContext.generate`` returns after its prompt: argmax of ``logits``, then greedy steps.

        ``on_tokens`` gets each batch of new ids as the host reads it (every ``sync_every`` steps).
        """

        s = self.session
        stops = {int(token) for token in stop_ids}
        produced = [int(torch.argmax(logits))]
        if on_tokens is not None:
            on_tokens(list(produced))
        limit = min(int(max_new_tokens), s.cache.capacity - s.next_position + 1)
        if produced[-1] in stops or limit <= 1:
            return produced
        start = s.next_position
        # every step reads all slots and masks the future ones to probability 0; their
        # memory must not hold inf / NaN bit patterns (0 * inf is NaN)
        s.cache.gpu_latent[:, start:].zero_()
        s.cache.gpu_pe[:, start:].zero_()
        self.token.fill_(produced[0])
        self.position.fill_(start)
        self.count.zero_()
        self._capture()
        done = 0
        while len(produced) < limit:
            batch = min(self.sync_every, s.cache.capacity - start - done)
            if batch <= 0:
                break
            for _ in range(batch):
                self.graph.replay()
            fresh = self.produced[done : done + batch].tolist()
            done += batch
            added = []
            for token in fresh:
                produced.append(int(token))
                added.append(int(token))
                if token in stops or len(produced) >= limit:
                    break
            if on_tokens is not None and added:
                on_tokens(added)
            if produced[-1] in stops or len(produced) >= limit:
                break
        # the session now holds everything fed: the produced ids but the last
        fed = produced[:-1]
        s.cache.length = start + len(fed)
        s.next_position = start + len(fed)
        if fed:
            s._add_extent(start, len(fed), start)
            s._record(start, fed)
        return produced
