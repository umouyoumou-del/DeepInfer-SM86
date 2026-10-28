from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from deepinfer.moe.deepseek_v2_runtime import _cpu_mlp_forward
from deepinfer.moe.layout import is_moe_layer, split_layer_indices


def _config(**overrides):
    values = {
        "num_hidden_layers": 6,
        "n_routed_experts": 64,
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_default_deepseek_layout_has_one_leading_dense_layer() -> None:
    dense, moe = split_layer_indices(_config())
    assert dense == [0]
    assert moe == [1, 2, 3, 4, 5]


def test_multiple_leading_dense_layers() -> None:
    dense, moe = split_layer_indices(_config(first_k_dense_replace=3))
    assert dense == [0, 1, 2]
    assert moe == [3, 4, 5]


def test_interspersed_dense_layers_follow_official_predicate() -> None:
    dense, moe = split_layer_indices(_config(moe_layer_freq=2))
    assert dense == [0, 1, 3, 5]
    assert moe == [2, 4]
    assert is_moe_layer(_config(moe_layer_freq=2), 2)
    assert not is_moe_layer(_config(moe_layer_freq=2), 3)


def test_dense_only_config_has_no_moe_layers() -> None:
    for experts in (None, 0):
        dense, moe = split_layer_indices(_config(n_routed_experts=experts))
        assert dense == list(range(6))
        assert moe == []


def test_cpu_mlp_forward_is_dtype_and_device_preserving() -> None:
    module = nn.Module()
    module.gate_proj = nn.Linear(4, 8, bias=False)
    module.up_proj = nn.Linear(4, 8, bias=False)
    module.down_proj = nn.Linear(8, 4, bias=False)
    hidden = torch.randn(2, 3, 4, dtype=torch.float16)
    output = _cpu_mlp_forward(module, hidden)
    assert output.shape == (2, 3, 4)
    assert output.dtype == torch.float16
    assert output.device.type == "cpu"

    reference_input = hidden.float()
    reference = module.down_proj(
        torch.nn.functional.silu(module.gate_proj(reference_input))
        * module.up_proj(reference_input)
    )
    torch.testing.assert_close(output.float(), reference, rtol=2e-3, atol=2e-3)
