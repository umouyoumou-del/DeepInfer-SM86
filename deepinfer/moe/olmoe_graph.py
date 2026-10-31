"""OLMoE decode captured as one CUDA graph.

Eager decode is bound by host dispatch on fast GPUs: on the 4080 SUPER a token
needs 7.2 ms of GPU work but 16.4 ms for the host to enqueue its ~1,900 ops
(OLMOE_PORT_PLAN section 35).  A graph replays the whole step with one launch.

A graph needs fixed shapes and fixed addresses, which the eager path does not
have: its KV cache grows by ``torch.cat`` and RoPE slices its table by a
host-side offset.  Here the cache is preallocated to ``max_len``, the position
lives on the device, the new key/value are written with ``index_copy_``,
attention covers the whole buffer with the positions past the current one
masked out, and RoPE rows are gathered by the device position.  The step ends
by writing its argmax into its own input and advancing the position, so decode
is one ``replay()`` per token with no host involvement in between.

The CuPy expert kernels are launched on the capturing stream; on CuPy's own
null stream the graph would silently record nothing for them.
"""

from __future__ import annotations

import warnings

import torch

from deepinfer.moe.olmoe_layer import apply_rope, build_rope, rms_norm
from deepinfer.moe.olmoe_packed import moe_forward_packed

try:
    import cupy as cp
except Exception:  # pragma: no cover - CPU-only hosts
    cp = None


class OLMoEGraphDecoder:
    """Greedy single-token decode over packed experts, replayed from a graph."""

    def __init__(
        self,
        weights: dict[str, torch.Tensor],
        layer_weights: list[dict],
        *,
        num_heads: int,
        eps: float,
        rope_base: float,
        top_k: int,
        max_len: int,
        tie_word_embeddings: bool = False,
    ) -> None:
        if cp is None:
            raise RuntimeError("the packed expert kernels need CuPy")
        if not layer_weights or any("packed" not in layer for layer in layer_weights):
            raise ValueError("every layer needs packed experts")
        embed = weights["embed_tokens"]
        self.device = embed.device
        self.weights, self.layers = weights, layer_weights
        self.num_heads, self.eps, self.top_k = num_heads, eps, top_k
        self.hidden = embed.shape[1]
        self.head_dim = self.hidden // num_heads
        self.max_len = max_len
        self.lm_head = embed if tie_word_embeddings else weights["lm_head"]

        shape = (1, num_heads, max_len, self.head_dim)
        self.keys = [torch.zeros(shape, dtype=embed.dtype, device=self.device)
                     for _ in layer_weights]
        self.values = [torch.zeros(shape, dtype=embed.dtype, device=self.device)
                       for _ in layer_weights]
        self.cos, self.sin = build_rope(torch.arange(max_len), self.head_dim, rope_base,
                                        device=self.device, dtype=embed.dtype)
        self.slots = torch.arange(max_len, device=self.device)
        self.position = torch.zeros(1, dtype=torch.long, device=self.device)
        self.token = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        # history[p] is the token predicted after reading position p
        self.history = torch.zeros(max_len, dtype=torch.long, device=self.device)
        self.used = 0
        self.graph: torch.cuda.CUDAGraph | None = None
        self.logits: torch.Tensor | None = None

    def _layer(self, hidden, w, index, cos, sin, visible):
        heads, dim = self.num_heads, self.head_dim
        residual = hidden
        normed = rms_norm(hidden, w["attn_norm"], self.eps)
        query = rms_norm(normed @ w["q_proj"].t(), w["attn_q_norm"], self.eps)
        key = rms_norm(normed @ w["k_proj"].t(), w["attn_k_norm"], self.eps)
        value = normed @ w["v_proj"].t()
        query = apply_rope(query.view(1, 1, heads, dim).transpose(1, 2), cos, sin)
        key = apply_rope(key.view(1, 1, heads, dim).transpose(1, 2), cos, sin)
        value = value.view(1, 1, heads, dim).transpose(1, 2)
        self.keys[index].index_copy_(2, self.position, key)
        self.values[index].index_copy_(2, self.position, value)
        context = torch.nn.functional.scaled_dot_product_attention(
            query, self.keys[index], self.values[index], attn_mask=visible
        )
        hidden = residual + context.transpose(1, 2).reshape(1, 1, self.hidden) @ w["o_proj"].t()

        residual = hidden
        normed = rms_norm(hidden, w["ffn_norm"], self.eps).reshape(1, self.hidden)
        # runs only while capturing; the fallback kernels launch on CuPy's
        # current stream, so it has to be the capturing one
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            stream = cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)
        with stream:
            experts = moe_forward_packed(
                normed, w["router"].t(), w["packed"], top_k=self.top_k,
                gate_out=w["packed_gate_out"], down_out=self.hidden,
                gate_k=self.hidden, down_k=w["packed_gate_out"],
            )
        return residual + experts.reshape(1, 1, self.hidden)

    def _step(self) -> torch.Tensor:
        hidden = torch.nn.functional.embedding(self.token, self.weights["embed_tokens"])
        visible = (self.slots <= self.position).view(1, 1, 1, self.max_len)
        cos = self.cos.index_select(0, self.position)
        sin = self.sin.index_select(0, self.position)
        for index, layer in enumerate(self.layers):
            hidden = self._layer(hidden, layer, index, cos, sin, visible)
        hidden = rms_norm(hidden, self.weights["final_norm"], self.eps)
        logits = hidden @ self.lm_head.t()
        predicted = logits[:, -1].argmax(-1, keepdim=True)
        self.history.index_copy_(0, self.position, predicted.view(1))
        self.token.copy_(predicted)
        self.position.add_(1)
        return logits

    def capture(self, warmup: int = 3) -> None:
        """Warm up on a side stream (cuBLAS handles, workspaces), then capture."""

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                self._step()
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.logits = self._step()
        self.reset()

    def reset(self) -> None:
        """Start a new sequence.  Stale cache rows are masked until rewritten."""

        self.position.zero_()
        self.used = 0

    def step(self, token: torch.Tensor | int | None = None) -> torch.Tensor:
        """Read one token (default: the previous prediction) and predict the next.

        Returns the device tensor holding the prediction; it is overwritten by
        the next step, so clone it to keep it.
        """

        if self.graph is None:
            raise RuntimeError("call capture() first")
        if self.used >= self.max_len:
            raise RuntimeError(f"sequence exceeds max_len={self.max_len}")
        if isinstance(token, int):
            self.token.fill_(token)
        elif token is not None and token.data_ptr() != self.token.data_ptr():
            self.token.copy_(token.view(1, 1))
        self.graph.replay()
        self.used += 1
        return self.token

    def generate(self, prompt_ids: list[int], max_new_tokens: int) -> list[int]:
        """Greedy: prompt read one token at a time, then the graph feeds itself."""

        if not prompt_ids or max_new_tokens < 1:
            raise ValueError("need a prompt and at least one new token")
        self.reset()
        for token in prompt_ids:
            self.step(int(token))
        for _ in range(max_new_tokens - 1):
            self.step()
        start = len(prompt_ids) - 1
        return self.history[start:start + max_new_tokens].tolist()
