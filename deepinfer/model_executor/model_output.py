from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import torch


@dataclass(frozen=True, slots=True)
class CausalLMForwardOutput:
    hidden_states: torch.Tensor
    """Input of the lm_head. Shape: [num_tokens, hidden_size]."""

    def narrow(self, num_tokens: int) -> Self:
        assert 0 < num_tokens <= self.hidden_states.size(0), (
            f"{num_tokens=}, {self.hidden_states.size(0)=}"
        )
        return type(self)(
            hidden_states=self.hidden_states[:num_tokens],
        )

    def copy_(self, src: Self) -> None:
        self.hidden_states.copy_(src.hidden_states)
