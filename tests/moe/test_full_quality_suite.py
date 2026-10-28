"""Validation for the committed full-category quality regression suite."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "docs" / "quality" / "deepseek_v2_lite_full_quality_suite.json"


def _load() -> dict:
    return json.loads(SUITE.read_text(encoding="utf-8"))


def test_full_quality_suite_shape_and_sources() -> None:
    suite = _load()
    assert suite["counts"]["mmlu_subjects"] == 57
    assert suite["counts"]["mmlu_sampled"] == 57
    assert len(suite["mmlu"]) == 57
    assert len(suite["gsm8k"]) == 32
    assert len(suite["long_chain"]) == 32
    assert suite["counts"]["perplexity_tokens"] == 512
    assert suite["sources"]["mmlu"].startswith("https://github.com/")
    assert suite["sources"]["gsm8k"].startswith("https://github.com/")
    assert suite["perplexity"]["text"].strip()
    assert {item["subject"] for item in suite["mmlu"]}.__len__() == 57
    for item in suite["mmlu"]:
        assert item["answer"] in {"A", "B", "C", "D"}
        assert len(item["choices"]) == 4
    for item in suite["gsm8k"]:
        assert item["answer"]
        assert item["question"]


def test_long_chain_answers_recompute() -> None:
    suite = _load()
    operation = re.compile(
        r"(Add|Subtract|Multiply by|Divide by) (\d+)"
    )
    for item in suite["long_chain"]:
        question = item["question"]
        start = re.search(r"Start with (\d+)\.", question)
        assert start is not None
        value = int(start.group(1))
        for kind, operand_text in operation.findall(question):
            operand = int(operand_text)
            if kind == "Add":
                value += operand
            elif kind == "Subtract":
                value -= operand
            elif kind == "Multiply by":
                value *= operand
            elif kind == "Divide by":
                assert value % operand == 0, (item["id"], value, operand)
                value //= operand
        assert str(value) == item["answer"], item["id"]


def test_quality_suite_has_no_duplicate_ids() -> None:
    suite = _load()
    ids = [
        item["id"]
        for category in ("mmlu", "gsm8k", "long_chain")
        for item in suite[category]
    ]
    assert len(ids) == len(set(ids))
