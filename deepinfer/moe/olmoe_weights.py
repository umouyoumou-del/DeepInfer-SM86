"""Load and shape-check OLMoE tensors against an OLMoESpec.

Every lookup goes through one place so a naming or shape surprise shows up here
rather than halfway through a forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from deepinfer.moe.gguf_index import GGUFIndex
from deepinfer.moe.olmoe_spec import OLMoESpec

_ATTENTION = {
    "q_proj": ("attn_q", (None, None)),
    "k_proj": ("attn_k", (None, None)),
    "v_proj": ("attn_v", (None, None)),
    "o_proj": ("attn_output", (None, None)),
}
_NORMS = ("attn_norm", "ffn_norm", "attn_q_norm", "attn_k_norm")


@dataclass(frozen=True)
class OLMoELayerTensors:
    """Tensor names for one layer, resolved once and reused."""

    layer_index: int
    attention: dict[str, str]
    norms: dict[str, str]
    router: str
    expert_gate: str
    expert_up: str
    expert_down: str


def layer_tensors(spec: OLMoESpec, layer_index: int) -> OLMoELayerTensors:
    if not 0 <= layer_index < spec.block_count:
        raise ValueError(f"layer {layer_index} outside 0..{spec.block_count - 1}")
    prefix = f"blk.{layer_index}."
    attention = {
        role: prefix + name + ".weight" for role, (name, _) in _ATTENTION.items()
    }
    norms = {name: prefix + name + ".weight" for name in _NORMS}
    return OLMoELayerTensors(
        layer_index=layer_index,
        attention=attention,
        norms=norms,
        router=prefix + "ffn_gate_inp.weight",
        expert_gate=prefix + "ffn_gate_exps.weight",
        expert_up=prefix + "ffn_up_exps.weight",
        expert_down=prefix + "ffn_down_exps.weight",
    )


def validate_layer(index: GGUFIndex, spec: OLMoESpec, layer_index: int) -> dict[str, tuple]:
    """Check every expected tensor exists with the shape the spec implies."""

    names = set(index.tensor_names())
    tensors = layer_tensors(spec, layer_index)
    hidden = spec.hidden_size
    inter = None
    expected: list[tuple[str, str, tuple | None]] = []

    for role, name in tensors.attention.items():
        expected.append((role, name, (hidden, hidden)))
    for name in ("attn_norm", "ffn_norm"):
        expected.append((name, tensors.norms[name], (hidden,)))
    for name in ("attn_q_norm", "attn_k_norm"):
        expected.append((name, tensors.norms[name], (hidden,)))

    expected.append(("router", tensors.router, (spec.expert_count, hidden)))
    for role, name in (
        ("expert_gate", tensors.expert_gate),
        ("expert_up", tensors.expert_up),
        ("expert_down", tensors.expert_down),
    ):
        expected.append((role, name, None))  # 3-D, checked below

    missing = [name for _role, name, _shape in expected if name not in names]
    if missing:
        raise ValueError(f"layer {layer_index} is missing {missing}")

    report: dict[str, tuple] = {}
    for role, name, want in expected:
        shape = tuple(index.tensor(name).numpy_shape)
        if want is not None and shape != want:
            raise ValueError(f"{role} {name} has shape {shape}, expected {want}")
        report[role] = shape

    # [experts, intermediate, hidden] for gate/up and [experts, hidden, intermediate] for down
    for role in ("expert_gate", "expert_up"):
        shape = report[role]
        if len(shape) != 3 or shape[0] != spec.expert_count:
            raise ValueError(f"{role} must be 3-D with {spec.expert_count} experts, got {shape}")
        if shape[2] != spec.hidden_size:
            raise ValueError(f"{role} inner dim must be hidden={spec.hidden_size}, got {shape}")
        inter = shape[1]
    down = report["expert_down"]
    if len(down) != 3 or down[0] != spec.expert_count or down[1] != spec.hidden_size:
        raise ValueError(f"expert_down shape {down} does not match the spec")
    if inter is not None and down[2] != inter:
        raise ValueError(
            f"expert_down intermediate {down[2]} disagrees with gate/up {inter}"
        )
    return report


def load_layer_tensor(
    index: GGUFIndex,
    spec: OLMoESpec,
    layer_index: int,
    role: str,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantise one non-expert tensor and return it as a torch tensor.

    ``GGUFIndex.read_tensor`` is documented for non-expert weights only, so this
    helper is restricted to attention projections, norms and the router.
    """

    tensors = layer_tensors(spec, layer_index)
    if role in tensors.attention:
        name = tensors.attention[role]
    elif role in tensors.norms:
        name = tensors.norms[role]
    elif role == "router":
        name = tensors.router
    else:
        raise KeyError(
            f"{role!r} is not a non-expert tensor; use GGUFIndex.expert_tensor"
        )
    return torch.from_numpy(index.read_tensor(name)).to(dtype)
