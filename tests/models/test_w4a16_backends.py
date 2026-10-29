"""The W4A16 model classes are thin aliases of one shared base class.

A wrong ``source_backend`` string, or a class that stops deriving from the base,
breaks checkpoint loading silently rather than loudly, so pin both.
"""

from __future__ import annotations

import pytest

from deepinfer.model_executor.models import qwen2_w4a16
from deepinfer.model_executor.models.gemma2_w4a16 import (
    Gemma2W4A16ForCausalLM,
)
from deepinfer.model_executor.models.gemma_w4a16 import GemmaW4A16ForCausalLM
from deepinfer.model_executor.models.llama_w4a16 import LlamaW4A16ForCausalLM
from deepinfer.model_executor.models.mistral_w4a16 import (
    MistralW4A16ForCausalLM,
)

BASE = qwen2_w4a16.Qwen2W4A16ForCausalLM

BACKENDS = [
    (qwen2_w4a16.Qwen2W4A16ForCausalLM, "qwen2_w4a16_sm86_model_executor"),
    (LlamaW4A16ForCausalLM, "llama_w4a16_sm86_model_executor"),
    (MistralW4A16ForCausalLM, "mistral_w4a16_sm86_model_executor"),
    (GemmaW4A16ForCausalLM, "gemma_w4a16_sm86_model_executor"),
    (Gemma2W4A16ForCausalLM, "gemma2_w4a16_sm86_model_executor"),
]


@pytest.mark.parametrize("cls,expected", BACKENDS)
def test_w4a16_backend_declares_expected_source(cls, expected: str) -> None:
    assert issubclass(cls, BASE)
    assert cls.source_backend == expected


def test_w4a16_source_backends_are_unique() -> None:
    names = [source for _, source in BACKENDS]
    assert len(names) == len(set(names)), "duplicate source_backend strings"


def test_w4a16_backends_do_not_override_the_loader_contract() -> None:
    """Subclasses may only add a docstring; the base owns loading and forward."""

    for cls, _ in BACKENDS:
        if cls is BASE:
            continue
        assert set(cls.__dict__) <= {"source_backend", "__doc__", "__module__", "__qualname__", "__firstlineno__", "__static_attributes__"}
