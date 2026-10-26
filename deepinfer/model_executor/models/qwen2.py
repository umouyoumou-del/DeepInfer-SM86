"""Source-level Qwen2/Qwen2.5 model integration for DeepInfer's model_executor.

Scope
-----
This module implements the dense Qwen2 decoder (GQA attention + SwiGLU MLP) in
the same model class layout that DeepInfer's runner expects:

* ``Qwen2ForCausalLM.load_model(...)`` constructs the model from an HF shard
  directory and maps the HF parameter names directly into the executor model;
* ``forward(kv_caches, input_metadata, hidden_states)`` returns
  :class:`CausalLMForwardOutput` and accepts a per-layer GQA KV cache;
* a reference PyTorch SDPA attention path is used when no fused kernel backend
  is installed.  This makes config/weight/forward/cache behavior testable on
  CPU without the SM100-only FlashMLA/DeepGEMM extensions.

SM100 integration boundary
--------------------------
The native DeepInfer engine still needs a full-attention cache layout and a
Qwen-specific input-metadata producer in ``worker/cache_engine.py`` /
``model_metadata_utils.py``.  Those pieces are intentionally not faked here.
On the target SM100/Linux environment the SDPA reference call should be
replaced by the repository's FlashAttention-4 GQA kernel while the rest of
this model layout and cache contract stays unchanged.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from deepinfer import AttrDict
from deepinfer.logger import init_logger
from deepinfer.model_executor.model_output import CausalLMForwardOutput

logger = init_logger(__name__)

_LORA_A_RE = re.compile(r"^(?P<module>.+)\.lora_A\.weight$")


@dataclass
class Qwen2LayerKVCache:
    """Per-layer standard GQA KV cache.

    Shapes are ``[batch, num_kv_heads, sequence, head_dim]``.  The native
    cache engine should expose the same logical tensors; the exact block-table
    materialization belongs to the future full-attention cache implementation.
    """

    key: torch.Tensor | None = None
    value: torch.Tensor | None = None

    @property
    def sequence_length(self) -> int:
        return 0 if self.key is None else int(self.key.shape[-2])


@dataclass
class Qwen2StaticLayerKVCache:
    """Preallocated GQA KV cache for CUDA-graph decode.

    The tensors keep a fixed ``max_sequence`` shape; each decode step writes the
    new key/value at ``cache_positions`` and attends over the full tensor with an
    external validity mask.
    """

    key: torch.Tensor
    value: torch.Tensor

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        cache_positions: torch.Tensor,
    ) -> None:
        positions = cache_positions.reshape(-1).to(torch.long)
        self.key.index_copy_(2, positions, key)
        self.value.index_copy_(2, positions, value)


@dataclass
class Qwen2InputMetadata:
    """Minimal common metadata used by source-level Qwen2 forwards.

    The native runner metadata object can be adapted to these three fields
    without changing the model math.
    """

    input_ids: torch.Tensor
    positions: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    is_prompt: bool = True


class Qwen2ModelMetadata:
    """Qwen2-specific cache/attention requirements for future engine wiring."""

    full_attention = True
    attention_type = "gqa_full"
    runner_ready = False
    """False until worker/cache_engine.py gains a full-attention cache layout."""

    def __init__(self, cfg: AttrDict) -> None:
        self.num_hidden_layers = int(cfg.num_hidden_layers)
        self.num_attention_heads = int(cfg.num_attention_heads)
        self.num_key_value_heads = int(cfg.num_key_value_heads)
        self.head_dim = int(cfg.head_dim)
        self.max_position_embeddings = int(cfg.max_position_embeddings)


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base

    def forward(
        self,
        positions: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(
                    0, self.head_dim, 2, dtype=torch.float32, device=positions.device
                )
                / self.head_dim
            )
        )
        freqs = positions.to(torch.float32).unsqueeze(-1) * inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


LinearFactory = Callable[[int, int, bool], nn.Module]
EmbeddingFactory = Callable[[int, int], nn.Module]


def _default_linear(in_features: int, out_features: int, bias: bool) -> nn.Module:
    return nn.Linear(in_features, out_features, bias=bias)


def _default_embedding(vocab_size: int, hidden_size: int) -> nn.Module:
    return nn.Embedding(vocab_size, hidden_size)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rotary(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # query/key: [batch, heads, sequence, head_dim]
    # cos/sin:   [batch, sequence, head_dim]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    query = query * cos + _rotate_half(query) * sin
    key = key * cos + _rotate_half(key) * sin
    return query, key


def _repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return hidden_states
    batch, heads, sequence, head_dim = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]
        .expand(batch, heads, repeats, sequence, head_dim)
        .reshape(
            batch,
            heads * repeats,
            sequence,
            head_dim,
        )
    )


class Qwen2Attention(nn.Module):
    """Qwen2 GQA attention with a PyTorch SDPA reference path."""

    backend = "torch_sdpa"

    def __init__(
        self,
        cfg: AttrDict,
        *,
        linear_factory: LinearFactory = _default_linear,
        fused_qkv_factory: LinearFactory | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = int(cfg.hidden_size)
        self.num_heads = int(cfg.num_attention_heads)
        self.num_kv_heads = int(cfg.num_key_value_heads)
        self.head_dim = int(cfg.head_dim)
        if self.num_heads % self.num_kv_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5

        qkv_bias = bool(cfg.get("attention_bias", True))
        q_out = self.num_heads * self.head_dim
        kv_out = self.num_kv_heads * self.head_dim
        if fused_qkv_factory is not None:
            self.fused_qkv = fused_qkv_factory(
                self.hidden_size, q_out + 2 * kv_out, qkv_bias
            )
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
        else:
            self.q_proj = linear_factory(self.hidden_size, q_out, qkv_bias)
            self.k_proj = linear_factory(self.hidden_size, kv_out, qkv_bias)
            self.v_proj = linear_factory(self.hidden_size, kv_out, qkv_bias)
            self.fused_qkv = None
        self.o_proj = linear_factory(q_out, self.hidden_size, False)
        # Optional native decode hook: (query, key_cache, value_cache, mask).
        self.native_gqa_decode: Any | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        layer_cache: Qwen2LayerKVCache | Qwen2StaticLayerKVCache | None = None,
        attention_mask: torch.Tensor | None = None,
        cache_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, sequence, _ = hidden_states.shape
        if self.fused_qkv is not None:
            qkv = self.fused_qkv(hidden_states)
            query, key, value = torch.split(
                qkv,
                [
                    self.num_heads * self.head_dim,
                    self.num_kv_heads * self.head_dim,
                    self.num_kv_heads * self.head_dim,
                ],
                dim=-1,
            )
        else:
            query = self.q_proj(hidden_states)
            key = self.k_proj(hidden_states)
            value = self.v_proj(hidden_states)
        query = query.view(batch, sequence, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        key = key.view(batch, sequence, self.num_kv_heads, self.head_dim).transpose(
            1, 2
        )
        value = value.view(batch, sequence, self.num_kv_heads, self.head_dim).transpose(
            1, 2
        )
        query, key = _apply_rotary(query, key, cos, sin)

        if isinstance(layer_cache, Qwen2StaticLayerKVCache):
            if cache_positions is None:
                raise ValueError("static KV cache requires cache_positions")
            layer_cache.update(key, value, cache_positions)
            key = layer_cache.key
            value = layer_cache.value
        elif layer_cache is not None:
            if layer_cache.key is None:
                layer_cache.key = key
                layer_cache.value = value
            else:
                assert layer_cache.value is not None
                layer_cache.key = torch.cat((layer_cache.key, key), dim=2)
                layer_cache.value = torch.cat((layer_cache.value, value), dim=2)
            key = layer_cache.key
            value = layer_cache.value

        if (
            self.native_gqa_decode is not None
            and isinstance(layer_cache, Qwen2StaticLayerKVCache)
            and sequence == 1
        ):
            attention = self.native_gqa_decode(query, key, value, attention_mask)
        else:
            key = _repeat_kv(key, self.num_key_value_groups)
            value = _repeat_kv(value, self.num_key_value_groups)
            key_length = key.shape[2]

            mask: torch.Tensor | None = None
            if sequence > 1:
                if isinstance(layer_cache, Qwen2StaticLayerKVCache):
                    if attention_mask is None:
                        raise ValueError(
                            "static KV cache prefill requires an explicit attention mask"
                        )
                else:
                    offset = key_length - sequence
                    mask = torch.ones(
                        (sequence, key_length), dtype=torch.bool, device=query.device
                    ).tril(diagonal=offset)
                    mask = mask[None, None, :, :]
            if attention_mask is not None:
                assert attention_mask.shape == (batch, key_length), (
                    f"attention_mask shape {attention_mask.shape=} does not match {(batch, key_length)=}"
                )
                valid = attention_mask.to(torch.bool)[:, None, None, :]
                mask = valid if mask is None else mask & valid

            if mask is None:
                attention = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    dropout_p=0.0,
                    scale=self.scaling,
                    is_causal=False,
                )
            else:
                attention = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=mask,
                    dropout_p=0.0,
                    scale=self.scaling,
                )

        attention = attention.transpose(1, 2).reshape(
            batch, sequence, self.num_heads * self.head_dim
        )
        return self.o_proj(attention)


class Qwen2MLP(nn.Module):
    def __init__(
        self,
        cfg: AttrDict,
        *,
        linear_factory: LinearFactory = _default_linear,
        fused_gate_up_factory: LinearFactory | None = None,
    ) -> None:
        super().__init__()
        hidden_size = int(cfg.hidden_size)
        intermediate_size = int(cfg.intermediate_size)
        if fused_gate_up_factory is not None:
            self.fused_gate_up = fused_gate_up_factory(
                hidden_size, 2 * intermediate_size, False
            )
            self.gate_proj = None
            self.up_proj = None
        else:
            self.gate_proj = linear_factory(hidden_size, intermediate_size, False)
            self.up_proj = linear_factory(hidden_size, intermediate_size, False)
            self.fused_gate_up = None
        self.down_proj = linear_factory(intermediate_size, hidden_size, False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.fused_gate_up is not None:
            gate, up = self.fused_gate_up(hidden_states).chunk(2, dim=-1)
        else:
            gate = self.gate_proj(hidden_states)
            up = self.up_proj(hidden_states)
        return self.down_proj(F.silu(gate) * up)


class Qwen2DecoderLayer(nn.Module):
    def __init__(
        self,
        cfg: AttrDict,
        *,
        linear_factory: LinearFactory = _default_linear,
        fused_qkv_factory: LinearFactory | None = None,
        fused_gate_up_factory: LinearFactory | None = None,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen2Attention(
            cfg,
            linear_factory=linear_factory,
            fused_qkv_factory=fused_qkv_factory,
        )
        self.mlp = Qwen2MLP(
            cfg,
            linear_factory=linear_factory,
            fused_gate_up_factory=fused_gate_up_factory,
        )
        self.input_layernorm = Qwen2RMSNorm(
            int(cfg.hidden_size), float(cfg.rms_norm_eps)
        )
        self.post_attention_layernorm = Qwen2RMSNorm(
            int(cfg.hidden_size), float(cfg.rms_norm_eps)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        layer_cache: Qwen2LayerKVCache | Qwen2StaticLayerKVCache | None = None,
        attention_mask: torch.Tensor | None = None,
        cache_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            cos,
            sin,
            layer_cache=layer_cache,
            attention_mask=attention_mask,
            cache_positions=cache_positions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen2Model(nn.Module):
    def __init__(
        self,
        cfg: AttrDict,
        *,
        linear_factory: LinearFactory = _default_linear,
        embedding_factory: EmbeddingFactory = _default_embedding,
        fused_qkv_factory: LinearFactory | None = None,
        fused_gate_up_factory: LinearFactory | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = embedding_factory(int(cfg.vocab_size), int(cfg.hidden_size))
        self.layers = nn.ModuleList(
            [
                Qwen2DecoderLayer(
                    cfg,
                    linear_factory=linear_factory,
                    fused_qkv_factory=fused_qkv_factory,
                    fused_gate_up_factory=fused_gate_up_factory,
                )
                for _ in range(int(cfg.num_hidden_layers))
            ]
        )
        self.norm = Qwen2RMSNorm(int(cfg.hidden_size), float(cfg.rms_norm_eps))
        self.rotary_emb = Qwen2RotaryEmbedding(int(cfg.head_dim), float(cfg.rope_theta))

    def forward_embeds(
        self,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        *,
        kv_cache: list[Qwen2LayerKVCache | Qwen2StaticLayerKVCache | None]
        | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if positions.shape != inputs_embeds.shape[:2]:
            raise ValueError(
                f"positions must have shape {inputs_embeds.shape[:2]}, got {positions.shape}"
            )
        cos, sin = self.rotary_emb(positions, inputs_embeds.dtype)
        hidden_states = inputs_embeds
        for layer_index, layer in enumerate(self.layers):
            cache = None if kv_cache is None else kv_cache[layer_index]
            if kv_cache is not None and cache is None:
                cache = Qwen2LayerKVCache()
                kv_cache[layer_index] = cache
            hidden_states = layer(
                hidden_states,
                cos,
                sin,
                layer_cache=cache,
                attention_mask=attention_mask,
                cache_positions=positions,
            )
        return self.norm(hidden_states)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        *,
        kv_cache: list[Qwen2LayerKVCache | None] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if positions is None:
            offset = 0
            if kv_cache is not None:
                offset = next(
                    (cache.sequence_length for cache in kv_cache if cache is not None),
                    0,
                )
            positions = torch.arange(
                offset,
                offset + input_ids.shape[1],
                dtype=torch.long,
                device=input_ids.device,
            ).unsqueeze(0)
            positions = positions.expand(input_ids.shape[0], -1)
        return self.forward_embeds(
            self.embed_tokens(input_ids),
            positions,
            kv_cache=kv_cache,
            attention_mask=attention_mask,
        )


class Qwen2ForCausalLM(nn.Module):
    """Qwen2 dense decoder with HF checkpoint and PEFT LoRA loading."""

    source_backend = "qwen2_reference_sdpa"
    native_runner_ready = False

    def __init__(
        self,
        cfg: AttrDict,
        *,
        linear_factory: LinearFactory = _default_linear,
        embedding_factory: EmbeddingFactory = _default_embedding,
        lm_head_factory: LinearFactory = _default_linear,
        fused_qkv_factory: LinearFactory | None = None,
        fused_gate_up_factory: LinearFactory | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = Qwen2Model(
            cfg,
            linear_factory=linear_factory,
            embedding_factory=embedding_factory,
            fused_qkv_factory=fused_qkv_factory,
            fused_gate_up_factory=fused_gate_up_factory,
        )
        self.lm_head = lm_head_factory(int(cfg.hidden_size), int(cfg.vocab_size), False)
        if bool(cfg.get("tie_word_embeddings", False)):
            self.lm_head.weight = self.model.embed_tokens.weight
        self.model_metadata = Qwen2ModelMetadata(cfg)

    def new_kv_cache(self) -> list[Qwen2LayerKVCache | None]:
        return [None] * int(self.cfg.num_hidden_layers)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def forward_logits(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        *,
        kv_cache: list[Qwen2LayerKVCache | None] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            kv_cache=kv_cache,
            attention_mask=attention_mask,
        )
        return self.lm_head(hidden_states)

    def forward(
        self,
        kv_caches: Sequence[Qwen2LayerKVCache | None] | None,
        input_metadata: Qwen2InputMetadata,
        hidden_states: torch.Tensor | None = None,
    ) -> CausalLMForwardOutput:
        """Native-executor-shaped forward.

        The current source-level path accepts :class:`Qwen2InputMetadata` or a
        compatible object exposing ``input_ids``, ``positions`` and
        ``attention_mask``.  Cache objects are the per-layer standard-GQA
        tensors described by :class:`Qwen2LayerKVCache`.
        """
        input_ids = getattr(input_metadata, "input_ids", None)
        if input_ids is None:
            raise ValueError("Qwen2 input_metadata must expose input_ids")
        positions = getattr(input_metadata, "positions", None)
        attention_mask = getattr(input_metadata, "attention_mask", None)
        cache = list(kv_caches) if kv_caches is not None else None

        if hidden_states is None:
            hidden_states = self.get_input_embeddings(input_ids)
        if positions is None:
            offset = 0
            if cache is not None:
                offset = next(
                    (item.sequence_length for item in cache if item is not None), 0
                )
            positions = (
                torch.arange(
                    offset,
                    offset + hidden_states.shape[1],
                    dtype=torch.long,
                    device=hidden_states.device,
                )
                .unsqueeze(0)
                .expand(hidden_states.shape[0], -1)
            )

        output = self.model.forward_embeds(
            hidden_states,
            positions,
            kv_cache=cache,
            attention_mask=attention_mask,
        )
        return CausalLMForwardOutput(hidden_states=output)

    def sample_greedy(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Reference helper for source tests; native runner still uses Sampler."""
        return self.lm_head(hidden_states).argmax(dim=-1)

    def allocate_output(self, num_tokens: int) -> CausalLMForwardOutput:
        parameter = next(self.parameters())
        return CausalLMForwardOutput(
            hidden_states=torch.empty(
                (num_tokens, int(self.cfg.hidden_size)),
                dtype=parameter.dtype,
                device=parameter.device,
            )
        )

    def sample(self, *args: Any, **kwargs: Any) -> Any:
        """Fail explicitly until the native Sampler/cache-engine wiring lands.

        Qwen2 model math, HF loading and the standard GQA KV contract are
        implemented and tested.  The existing DeepInfer Sampler and cache engine
        are DeepSeek-V4/MLA-specific; silently returning an incompatible object
        here would misrepresent source-level integration as a runnable engine.
        """
        del args, kwargs
        raise NotImplementedError(
            "Qwen2 native sampling is not wired yet: implement full-attention "
            "cache support in worker/cache_engine.py and bind the DeepInfer "
            "Sampler after a target SM100/Linux build is available"
        )

    @classmethod
    def load_model(
        cls,
        model_path: str,
        model_cfg: AttrDict,
        skip_load: bool = False,
        *,
        adapter_path: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> nn.Module:
        gpt = (
            model_cfg.gpt
            if isinstance(model_cfg, dict) and "gpt" in model_cfg
            else model_cfg
        )
        if not isinstance(gpt, AttrDict):
            gpt = AttrDict(gpt)
        dtype = model_cfg.get("dtype") if hasattr(model_cfg, "get") else torch.bfloat16
        if not isinstance(dtype, torch.dtype):
            dtype = torch.bfloat16

        # Construct directly in the requested dtype.  Building a 3B model in
        # float32 and then casting would temporarily require both copies in RAM
        # and can exceed the CPU/host memory budget before weights are loaded.
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            if device is not None:
                with torch.device(device):
                    model = cls(gpt)
            else:
                model = cls(gpt)
        finally:
            torch.set_default_dtype(previous_dtype)

        if not skip_load:
            _load_qwen2_state_dict(model, Path(model_path))

        selected_adapter = (
            Path(adapter_path)
            if adapter_path is not None
            else Path(
                str(model_cfg.get("adapter_path") or gpt.get("adapter_path") or "")
            )
        )
        if selected_adapter and str(selected_adapter) not in ("", "."):
            _merge_peft_adapter(model, selected_adapter)

        logger.info(
            "Qwen2 source model loaded: layers=%s hidden=%s heads=%s kv_heads=%s adapter=%s",
            gpt.num_hidden_layers,
            gpt.hidden_size,
            gpt.num_attention_heads,
            gpt.num_key_value_heads,
            selected_adapter if selected_adapter else "<none>",
        )
        return model.eval().requires_grad_(False)


def _iter_safetensor_weights(model_path: Path) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield HF safetensors shards without materializing the whole checkpoint."""

    from safetensors import safe_open

    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"invalid safetensors index: {index_path}")
        files = sorted({str(name) for name in weight_map.values()})
    else:
        files = ["model.safetensors"]
        if not (model_path / "model.safetensors").is_file():
            candidates = sorted(model_path.glob("*.safetensors"))
            if not candidates:
                raise FileNotFoundError(
                    f"no safetensors checkpoint found in {model_path}"
                )
            files = [path.name for path in candidates]

    for filename in files:
        path = model_path / filename
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint shard missing: {path}")
        with safe_open(str(path), framework="pt", device="cpu") as shard:
            # safe_open is iterable only through its explicit keys() API.
            for name in shard.keys():  # noqa: SIM118
                yield name, shard.get_tensor(name)


def _load_qwen2_state_dict(model: Qwen2ForCausalLM, model_path: Path) -> None:
    parameters = dict(model.named_parameters(remove_duplicate=False))
    loaded: set[str] = set()
    unexpected: list[str] = []
    with torch.no_grad():
        for name, tensor in _iter_safetensor_weights(model_path):
            parameter = parameters.get(name)
            if parameter is None:
                unexpected.append(name)
                continue
            if tuple(parameter.shape) != tuple(tensor.shape):
                raise ValueError(
                    f"shape mismatch for {name}: model={tuple(parameter.shape)}, checkpoint={tuple(tensor.shape)}"
                )
            parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
            loaded.add(name)

    tied_head = model.lm_head.weight is model.model.embed_tokens.weight
    missing = [
        name
        for name in parameters
        if name not in loaded and not (tied_head and name == "lm_head.weight")
    ]
    if missing:
        raise RuntimeError(
            f"checkpoint {model_path} missing {len(missing)} model parameters: {missing[:20]}"
        )
    if unexpected:
        logger.warning(
            "ignored %s checkpoint tensors not present in Qwen2 model: %s",
            len(unexpected),
            unexpected[:10],
        )


def _merge_peft_adapter(model: Qwen2ForCausalLM, adapter_path: Path) -> None:
    adapter_file = adapter_path / "adapter_model.safetensors"
    config_file = adapter_path / "adapter_config.json"
    if not adapter_file.is_file() or not config_file.is_file():
        raise FileNotFoundError(
            f"PEFT adapter must contain adapter_model.safetensors and adapter_config.json: {adapter_path}"
        )
    adapter_cfg = json.loads(config_file.read_text(encoding="utf-8"))
    if adapter_cfg.get("use_dora"):
        raise NotImplementedError(
            "DoRA adapters are not supported by the source-level Qwen2 merger"
        )
    rank = int(adapter_cfg.get("r") or 0)
    alpha = float(adapter_cfg.get("lora_alpha") or rank or 1)
    if rank <= 0:
        raise ValueError(f"invalid PEFT LoRA rank in {config_file}: {rank}")
    scaling = alpha / math.sqrt(rank) if adapter_cfg.get("use_rslora") else alpha / rank

    from safetensors import safe_open

    with safe_open(str(adapter_file), framework="pt", device="cpu") as tensors:
        names = set(tensors.keys())
        parameters = dict(model.named_parameters(remove_duplicate=False))
        merged = 0
        with torch.no_grad():
            for key in sorted(names):
                match = _LORA_A_RE.match(key)
                if match is None:
                    continue
                adapter_module_path = match.group("module")
                b_key = adapter_module_path + ".lora_B.weight"
                if b_key not in names:
                    continue
                module_path = adapter_module_path
                module_path = module_path.removeprefix("base_model.model.")
                base_key = module_path + ".weight"
                parameter = parameters.get(base_key)
                if parameter is None:
                    continue
                lora_a = tensors.get_tensor(key).to(torch.float32)
                lora_b = tensors.get_tensor(b_key).to(torch.float32)
                if (
                    lora_a.ndim != 2
                    or lora_b.ndim != 2
                    or lora_b.shape[1] != lora_a.shape[0]
                ):
                    raise ValueError(
                        f"invalid LoRA shapes for {module_path}: {tuple(lora_a.shape)}, {tuple(lora_b.shape)}"
                    )
                delta = (lora_b @ lora_a) * scaling
                if delta.shape != parameter.shape:
                    raise ValueError(
                        f"LoRA delta shape mismatch for {base_key}: {tuple(delta.shape)} vs {tuple(parameter.shape)}"
                    )
                parameter.add_(delta.to(device=parameter.device, dtype=parameter.dtype))
                merged += 1
    if merged == 0:
        raise RuntimeError(
            f"no compatible LoRA matrices were merged from {adapter_file}"
        )
    logger.info("merged %s PEFT LoRA matrices from %s", merged, adapter_path)


__all__ = [
    "Qwen2ForCausalLM",
    "Qwen2InputMetadata",
    "Qwen2LayerKVCache",
    "Qwen2ModelMetadata",
    "Qwen2StaticLayerKVCache",
]
