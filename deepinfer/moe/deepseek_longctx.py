"""Million-token contexts for the DeepSeek-V2 GGUF runtime on one GPU.

Drives the runtime's loaded decoder layers directly, with three changes from
its own forward:

* KV lives in a :class:`~deepinfer.moe.latent_mla.LatentKVCache` - the 576-wide
  MLA latent per token and layer instead of 16 expanded heads, fp16 or fp8,
  spilling to pinned host memory past ``gpu_tokens``.
* Prefill runs in chunks; attention expands cached latent tiles and uses flash
  attention (every query sees every earlier token), and the MoE dequantizes a
  layer's experts once per chunk and multiplies on tensor cores - 15x faster
  than the per-token fused kernels at 16K tokens.
* Only the last position's logits are computed, and RoPE is formed per call
  from one shared inverse-frequency vector (no per-layer cos/sin caches, which
  would take 13.8 GiB at 2M tokens).

Decode attends over the whole cache with the absorbed form; the MoE of a single
token goes through the runtime's own decode path.  ``generate`` on a short fp16
context (a re-read answer) runs each step as one CUDA graph instead
(:mod:`deepinfer.moe.graph_decode`).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F

from deepinfer.moe.latent_mla import (
    DualChunkPositions,
    LatentKVCache,
    LatentRope,
    MLAWeights,
    decode_attention,
    extent_regions,
    prefill_attention,
)
from deepinfer.moe.latent_retrieval import (
    BlockBounds,
    RetrievalConfig,
    sparse_decode_attention,
)


def resident_experts(store, layer_idx: int, expert_ids: Sequence[int]):
    """fp16 ``(gate, up, down)`` ``[E, out, in]`` for ``expert_ids``.

    When the store keeps every routed expert packed in VRAM (the runtime pins
    them all when they fit), dequantize from those bytes; ``load_experts``
    stages its input from pinned host memory instead - 31 ms of PCIe per layer
    for 64 experts, which was most of a prefill block's MoE time.  Falls back to
    ``load_experts`` when any expert is not resident.
    """

    from deepinfer.moe.gpu_quant import dequantize_matrix

    entries = [store._gpu_cache_get(layer_idx, int(e)) for e in expert_ids] if store.gpu_cache_enabled() else []
    if not entries or any(entry is None for entry in entries):
        return store.load_experts(layer_idx, expert_ids)
    weights = []
    for index, projection in enumerate(("gate", "up", "down")):
        spec = store._specs[(layer_idx, projection)]
        packed = torch.cat([entry[index].reshape(-1) for entry in entries])
        matrix = dequantize_matrix(
            packed, spec.quant_type, out_features=len(entries) * spec.out_features, in_features=spec.in_features,
        )
        weights.append(matrix.view(len(entries), spec.out_features, spec.in_features))
    return tuple(weights)


@torch.inference_mode()
def moe_prefill(store, mlp, x: torch.Tensor, layer_idx: int, *, group: int = 16) -> torch.Tensor:
    """DeepseekV2MoE for many tokens: ``x [T, hidden] -> [T, hidden]``.

    Same math as the official ``moe_infer`` (routing weights applied and summed
    in fp32), but each expert is dequantized once per call and its tokens go
    through three cuBLAS GEMMs.  Each expert's weighted output is added into
    the fp32 sum as soon as it is computed: gathering every routed copy first
    held 2 GB of temporaries for a 16K-token block (top-6 copies in fp16 in and
    out, plus their fp32 product), enough to run out of memory beside a 2M cache.
    """

    tokens, hidden = x.shape
    topk_idx, topk_weight, _ = mlp.gate(x.unsqueeze(0))
    top_k = topk_idx.shape[1]
    flat = topk_idx.reshape(-1)
    order = torch.argsort(flat, stable=True)
    expert_count = int(mlp.gate.weight.shape[0])
    counts = torch.bincount(flat, minlength=expert_count).tolist()
    token_of = order // top_k
    weights = topk_weight.reshape(-1).index_select(0, order).float().unsqueeze(-1)
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    active = [expert for expert, count in enumerate(counts) if count]
    combined = torch.zeros((tokens, hidden), dtype=torch.float32, device=x.device)
    for first in range(0, len(active), group):
        ids = active[first : first + group]
        gate_w, up_w, down_w = resident_experts(store, layer_idx, ids)
        for slot, expert in enumerate(ids):
            lo, hi = offsets[expert], offsets[expert + 1]
            rows = token_of[lo:hi]
            segment = x.index_select(0, rows)
            activated = F.silu(segment @ gate_w[slot].t()) * (segment @ up_w[slot].t())
            combined.index_add_(0, rows, (activated @ down_w[slot].t()).float() * weights[lo:hi])
        del gate_w, up_w, down_w
    return combined.to(x.dtype) + mlp.shared_experts(x)


class DeepSeekLongContext:
    """One long sequence over a DeepSeekV2LiteGGUFRuntime's weights."""

    # Up to this many tokens a prefill block's MoE goes through the runtime's
    # fused kernels on the packed experts: moe_prefill dequantizes every routed
    # expert of the layer first, ~0.2 s over 26 layers whatever the block holds
    # (moe_prefill vs fused, 26 layers, 4080 SUPER, experts resident: 17 tokens
    # 179 vs 104 ms; 162: 226 vs 306; 274: 229 vs 487; 7684: 673 vs 13626).
    small_moe_tokens = 64

    def __init__(
        self,
        runtime,
        *,
        capacity: int,
        gpu_tokens: int | None = None,
        kv_dtype: str = "fp16",
        positions: DualChunkPositions | None = None,
        prefill_chunk: int = 16384,
        tile_tokens: int = 32768,
        retrieval: RetrievalConfig | None = None,
        logn_trained: int | None = None,
        encode_window: int | None = None,
        encode_sink: int = 64,
    ) -> None:
        self.rt = runtime
        self.model = runtime.model.model
        self.layers = list(self.model.layers)
        first = self.layers[0].self_attn
        self.num_heads = int(first.num_heads)
        self.nope_dim = int(first.qk_nope_head_dim)
        self.rope_dim = int(first.qk_rope_head_dim)
        self.v_dim = int(first.v_head_dim)
        self.latent_dim = int(first.kv_lora_rank)
        self.rope = LatentRope.from_attention(first, runtime.config)
        self.weights = [MLAWeights.from_attention(layer.self_attn) for layer in self.layers]
        self.positions = positions or DualChunkPositions.plain()
        self.prefill_chunk = int(prefill_chunk)
        self.tile_tokens = int(tile_tokens)
        self.cache = LatentKVCache(
            num_layers=len(self.layers),
            capacity=int(capacity),
            gpu_tokens=gpu_tokens,
            kv_dtype=kv_dtype,
            device=runtime.device,
        )
        self.device = runtime.device
        self.moe_layers = set(getattr(runtime, "moe_layer_indices", ()))
        self.timings: dict[str, float] = {}
        # Physical slot != logical position once something is inserted: each
        # extent is (first slot, end slot, logical position of the first slot).
        self.extents: list[list[int]] = []
        self.next_position = 0
        # token id held by each physical slot, so located blocks can be re-read
        self.slot_tokens: list[int] = []
        # decode attends to sink + best blocks + window instead of everything
        self.retrieval = retrieval
        # Queries past this position get their logits scaled by
        # log(position) / log(logn_trained), so attention spread over more keys
        # than the model was trained with keeps its entropy (log-n attention).
        self.logn_trained = None if logn_trained is None else int(logn_trained)
        # Bounded encoding: during prefill/insert a token attends only to the
        # first ``encode_sink`` positions and the ``encode_window`` before its
        # block.  Past the trained length, full attention over everything
        # computes every later token's KV out of distribution - the library
        # entries themselves degrade - while a window <= the trained length
        # keeps each entry in distribution and makes prefill linear.  Recall
        # over the whole sequence is then the query's job (retrieval).
        self.encode_window = None if encode_window is None else int(encode_window)
        self.encode_sink = int(encode_sink)
        self.bounds: BlockBounds | None = None

    # ----------------------------------------------------------- utilities

    @property
    def length(self) -> int:
        """Cached tokens (physical slots in use)."""

        return self.cache.length

    def snapshot(self) -> tuple[int, int, list[list[int]]]:
        """State to come back to with :meth:`restore` (the KV itself is not copied)."""

        return self.cache.length, self.next_position, [list(e) for e in self.extents]

    def restore(self, state: tuple[int, int, list[list[int]]]) -> None:
        """Drop everything written since ``state``; later writes overwrite it."""

        length, position, extents = state
        if length > self.cache.length:
            raise ValueError("cannot restore to a longer cache")
        self.cache.length = int(length)
        self.next_position = int(position)
        self.extents = [list(e) for e in extents]
        del self.slot_tokens[int(length) :]
        if self.bounds is not None:
            self.bounds.ready = min(self.bounds.ready, int(length) // self.bounds.block)

    def save(self, directory: str, *, meta: dict | None = None, spill: Sequence[str] = ()) -> None:
        """Persist the cache with its layout, so a later session skips the prefill.

        ``spill``: more directories (disks) to spread a cache over that does not
        fit in ``directory`` (see :meth:`LatentKVCache.save`).
        """

        import os

        import numpy as np

        self.cache.save(
            directory,
            meta={
                **(meta or {}),
                "extents": self.extents,
                "next_position": self.next_position,
                "trained": self.positions.trained,
                "chunk": self.positions.chunk,
                "far": self.positions.far,
            },
            spill=spill,
        )
        tokens = self.slot_tokens[: self.length]
        if len(tokens) == self.length:  # lets a loaded session re-read located text
            np.asarray(tokens, dtype=np.int32).tofile(os.path.join(directory, "tokens.bin"))

    @classmethod
    def load(
        cls,
        runtime,
        directory: str,
        *,
        capacity: int,
        gpu_tokens: int | None = None,
        far: int | None = None,
        **kwargs,
    ) -> tuple[DeepSeekLongContext, dict]:
        """A session over a cache written by :meth:`save` (same model required).

        ``far`` may differ from the prefill's: it only changes how queries see
        old keys (see :class:`DualChunkPositions`).
        """

        cache, meta = LatentKVCache.load(
            directory, capacity=capacity, gpu_tokens=gpu_tokens, device=runtime.device
        )
        positions = DualChunkPositions(
            int(meta["trained"]),
            int(meta["chunk"]),
            int(far if far is not None else meta.get("far", -1)),
        )
        session = cls(
            runtime,
            capacity=1,
            kv_dtype=cache.kv_dtype,
            positions=positions,
            **kwargs,
        )
        session.cache = cache
        session.extents = [list(e) for e in meta["extents"]]
        session.next_position = int(meta["next_position"])
        import os

        import numpy as np

        token_file = os.path.join(directory, "tokens.bin")
        if os.path.exists(token_file):
            session.slot_tokens = np.fromfile(token_file, dtype=np.int32).tolist()
        return session, meta

    def truncate(self, length: int) -> None:
        """Keep the first ``length`` slots (only for a cache without insertions)."""

        if length > self.cache.length:
            raise ValueError("cannot truncate past the current length")
        kept = []
        for p0, p1, l0 in self.extents:
            if p0 >= length:
                break
            kept.append([p0, min(p1, length), l0])
        if any(p0 != l0 for p0, _, l0 in kept):
            raise ValueError("truncate() needs slot == position; use snapshot/restore")
        self.extents = kept
        self.cache.length = int(length)
        self.next_position = int(length)
        del self.slot_tokens[int(length) :]
        if self.bounds is not None:
            self.bounds.ready = min(self.bounds.ready, int(length) // self.bounds.block)

    def _add_extent(self, slot: int, count: int, position: int) -> None:
        if self.extents:
            last = self.extents[-1]
            if last[1] == slot and last[2] + (last[1] - last[0]) == position:
                last[1] += count
                return
        self.extents.append([slot, slot + count, position])

    def _clip_logical(self, extents: list[list[int]], start: int) -> list[list[int]]:
        """Keep the sink and the encode window before logical ``start``."""

        keep = [(0, self.encode_sink), (max(0, start - self.encode_window), start)]
        clipped = []
        for p0, p1, l0 in extents:
            l1 = l0 + (p1 - p0)
            for lo, hi in keep:
                a, b = max(l0, lo), min(l1, hi)
                if a < b:
                    clipped.append([p0 + a - l0, p0 + b - l0, a])
        clipped.sort(key=lambda e: e[0])
        merged: list[list[int]] = []
        for extent in clipped:  # the two ranges can meet when start is small
            if merged and merged[-1][1] >= extent[0] and merged[-1][2] + merged[-1][1] - merged[-1][0] >= extent[2]:
                merged[-1][1] = max(merged[-1][1], extent[1])
            else:
                merged.append(extent)
        return merged

    @staticmethod
    def _visible(extents: list[list[int]], before: int) -> list[list[int]]:
        """The parts of ``extents`` at logical positions below ``before``."""

        visible = []
        for p0, p1, l0 in extents:
            count = min(p1 - p0, before - l0)
            if count > 0:
                visible.append([p0, p0 + count, l0])
        return visible

    def _tick(self, name: str, started: float) -> float:
        now = time.perf_counter()
        self.timings[name] = self.timings.get(name, 0.0) + now - started
        return now

    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.rt._embed_tokens(ids.view(1, -1).cpu())[0]

    def _logits(self, hidden_last: torch.Tensor) -> torch.Tensor:
        normed = self.model.norm(hidden_last.view(1, -1))
        weight = self.rt.model.lm_head.weight
        return F.linear(normed.to(weight.device, weight.dtype), weight).float()[0]

    def _project(self, layer, x: torch.Tensor, start: int):
        """q, latent and rotated k_pe for ``n`` tokens starting at ``start``."""

        attention = layer.self_attn
        n = x.shape[0]
        q = attention.q_proj(x).view(n, self.num_heads, self.nope_dim + self.rope_dim)
        q_nope, q_pe = torch.split(q, [self.nope_dim, self.rope_dim], dim=-1)
        compressed = attention.kv_a_proj_with_mqa(x)
        latent, k_pe = torch.split(compressed, [self.latent_dim, self.rope_dim], dim=-1)
        latent = attention.kv_a_layernorm(latent.contiguous())
        absolute = torch.arange(start, start + n, device=x.device)
        pe = self.rope.apply(k_pe, self.positions.key_positions(absolute))
        if self.logn_trained is not None and start + n > self.logn_trained:
            factor = torch.log(absolute.double() + 1) / math.log(self.logn_trained)
            factor = factor.clamp_min(1.0).to(q_nope.dtype).view(n, 1, 1)
            q_nope = q_nope * factor
            q_pe = q_pe * factor
        return q_nope, q_pe, latent, pe

    # ------------------------------------------------------------- prefill

    def _blocks(self, start: int, length: int) -> list[tuple[int, int]]:
        blocks = []
        for piece_start, piece_length in self.positions.split_block(start, length):
            for offset in range(0, piece_length, self.prefill_chunk):
                blocks.append(
                    (piece_start + offset, min(self.prefill_chunk, piece_length - offset))
                )
        return blocks

    @torch.inference_mode()
    def prefill(
        self,
        ids: Sequence[int] | torch.Tensor,
        *,
        progress: Callable[[int, int, float], None] | None = None,
        bounded: bool = True,
    ) -> torch.Tensor:
        """Append ``ids`` and return the logits after the last one (fp32 ``[vocab]``).

        ``bounded=False`` ignores ``encode_window``: for a question, which should
        see the whole sequence, rather than for text being encoded.
        """

        start = self.next_position
        logits = self._run(ids, start, context=None, progress=progress, bounded=bounded)
        self.next_position = start + len(ids)
        return logits

    @torch.inference_mode()
    def insert(self, ids: Sequence[int] | torch.Tensor, position: int) -> None:
        """Store ``ids`` as if they stood at logical ``position`` of the sequence.

        They attend to everything cached at logical positions below
        ``position`` (and to themselves), exactly as if the text had been there
        during prefill, and every later query sees them at that position.  What
        is *not* reproduced is the influence they would have had on the cached
        tokens after ``position``, which were computed without them.  The next
        appended token still follows the end of the sequence.
        """

        if position > self.next_position:
            raise ValueError("insert position lies past the end of the sequence")
        self._run(ids, position, context=self._visible(self.extents, position), progress=None)

    def _record(self, slot: int, ids: list[int]) -> None:
        if len(self.slot_tokens) < slot:
            self.slot_tokens.extend([-1] * (slot - len(self.slot_tokens)))
        self.slot_tokens[slot : slot + len(ids)] = ids

    def _run(
        self, ids, start: int, *, context, progress, observe=None, bounded: bool = True, attend=None
    ) -> torch.Tensor:
        """Forward ``ids`` at logical ``start`` into fresh slots.

        ``context`` limits which earlier extents they attend to (``None``: all).
        ``observe(layer, q_nope, q_pe_raw, block_start, regions)`` is called with
        each layer's queries, before attention.  ``attend(layer, q_nope, q_pe_raw,
        latent, pe, block_start, regions)``, if given, replaces
        :func:`prefill_attention` and returns the same ``[n, H, v_dim]``.
        """

        ids = torch.as_tensor(ids, dtype=torch.long).view(-1)
        slot0 = self.cache.length
        if slot0 + ids.numel() > self.cache.capacity:
            raise ValueError(
                f"{slot0 + ids.numel()} tokens exceed the cache capacity {self.cache.capacity}"
            )
        began = time.perf_counter()
        hidden = None
        store = self.rt.expert_store
        own: list[list[int]] = []
        for block_start, block_length in self._blocks(start, ids.numel()):
            local = block_start - start
            slot = slot0 + local
            seen = [list(e) for e in (self.extents if context is None else context)] + own
            if bounded and self.encode_window is not None:
                seen = self._clip_logical(seen, block_start)
            regions = extent_regions(seen, block_start, self.positions)
            hidden = self._embed(ids[local : local + block_length]).to(self.device)
            for index, layer in enumerate(self.layers):
                started = time.perf_counter()
                x = layer.input_layernorm(hidden)
                q_nope, q_pe, latent, pe = self._project(layer, x, block_start)
                self.cache.write(index, slot, latent, pe)
                if observe is not None:
                    observe(index, q_nope, q_pe, block_start, regions)
                if attend is not None:
                    attended = attend(index, q_nope, q_pe, latent, pe, block_start, regions)
                else:
                    attended = prefill_attention(
                        self.weights[index],
                        self.cache,
                        index,
                        q_nope,
                        q_pe,
                        latent,
                        pe,
                        block_start,
                        positions=self.positions,
                        rope=self.rope,
                        tile_tokens=self.tile_tokens,
                        regions=regions,
                    )
                hidden = hidden + layer.self_attn.o_proj(
                    attended.reshape(block_length, self.num_heads * self.v_dim)
                )
                del q_nope, q_pe, latent, pe, attended, x
                torch.cuda.synchronize()
                started = self._tick("prefill_attention", started)
                x = layer.post_attention_layernorm(hidden)
                if index in self.moe_layers and block_length > self.small_moe_tokens:
                    hidden = hidden + moe_prefill(store, layer.mlp, x, index)
                elif index in self.moe_layers:
                    hidden = hidden + layer.mlp(x.unsqueeze(0))[0]
                else:
                    hidden = hidden + layer.mlp(x)
                del x
                torch.cuda.synchronize()
                self._tick("prefill_mlp", started)
            self._add_extent(slot, block_length, block_start)
            self._record(slot, ids[local : local + block_length].tolist())
            if context is not None:
                own.append([slot, slot + block_length, block_start])
            if progress is not None:
                progress(local + block_length, ids.numel(), time.perf_counter() - began)
        return self._logits(hidden[-1])

    # -------------------------------------------------------------- decode

    @torch.inference_mode()
    def decode(self, token_id: int) -> torch.Tensor:
        """Append one token and return the logits after it."""

        position = self.next_position
        slot = self.cache.length
        layout = [tuple(e) for e in self.extents] + [(slot, slot + 1, position)]
        regions = extent_regions(layout, position, self.positions)
        if self.retrieval is not None and self.bounds is None:
            self.bounds = BlockBounds(
                num_layers=len(self.layers),
                capacity=self.cache.capacity,
                block=self.retrieval.block,
                device=self.device,
            )
        hidden = self._embed(torch.tensor([int(token_id)])).to(self.device)
        for index, layer in enumerate(self.layers):
            started = time.perf_counter()
            x = layer.input_layernorm(hidden)
            q_nope, q_pe, latent, pe = self._project(layer, x, position)
            self.cache.write(index, slot, latent, pe)
            if self.retrieval is None:
                attended = decode_attention(
                    self.weights[index],
                    self.cache,
                    index,
                    q_nope[0],
                    q_pe[0],
                    position,
                    positions=self.positions,
                    rope=self.rope,
                    regions=regions,
                )
            else:
                self.bounds.update(self.cache, index, slot + 1)
                attended = sparse_decode_attention(
                    self.weights[index],
                    self.cache,
                    self.bounds,
                    index,
                    q_nope[0],
                    q_pe[0],
                    position,
                    regions,
                    positions=self.positions,
                    rope=self.rope,
                    config=self.retrieval,
                    extents=layout,
                )
            hidden = hidden + layer.self_attn.o_proj(attended.reshape(1, -1))
            started = self._tick("decode_attention", started)
            x = layer.post_attention_layernorm(hidden)
            hidden = hidden + layer.mlp(x.unsqueeze(0))[0]
            self._tick("decode_mlp", started)
        self._add_extent(slot, 1, position)
        self._record(slot, [int(token_id)])
        self.next_position = position + 1
        if self.bounds is not None:
            self.bounds.commit(slot + 1)
        return self._logits(hidden[-1])

    @torch.inference_mode()
    def feed(self, ids: Sequence[int]) -> torch.Tensor:
        """Append ``ids`` one token at a time through :meth:`decode`.

        With retrieval on, this is how a question is read without full
        attention; returns the logits after the last token.
        """

        logits = None
        for token in ids:
            logits = self.decode(int(token))
        return logits

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: Sequence[int],
        *,
        max_new_tokens: int = 32,
        stop_ids: Sequence[int] = (),
        on_tokens: Callable[[list[int]], None] | None = None,
    ) -> list[int]:
        """Greedy continuation of ``prompt_ids`` appended to the cache.

        With retrieval on, the prompt is fed token by token so that it, too,
        only sees the retrieved blocks.  A short fp16 context in VRAM decodes
        through :class:`~deepinfer.moe.graph_decode.GraphDecoder` (one CUDA
        graph per step, no host round trip inside it).  ``on_tokens`` gets the new
        ids as they are produced.
        """

        from deepinfer.moe import graph_decode

        if self.retrieval is not None:
            logits = self.feed(prompt_ids)
        else:
            logits = self.prefill(prompt_ids, bounded=False)
        if int(max_new_tokens) > 1 and graph_decode.usable(self):
            decoder = graph_decode.GraphDecoder(self, steps=int(max_new_tokens))
            return decoder.greedy(
                logits, max_new_tokens=int(max_new_tokens), stop_ids=stop_ids, on_tokens=on_tokens
            )
        produced: list[int] = []
        stops = {int(token) for token in stop_ids}
        for _ in range(int(max_new_tokens)):
            token = int(torch.argmax(logits))
            produced.append(token)
            if on_tokens is not None:
                on_tokens([token])
            if token in stops:
                break
            logits = self.decode(token)
        return produced
