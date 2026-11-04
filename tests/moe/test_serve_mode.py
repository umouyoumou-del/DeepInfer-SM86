"""Automatic full-KV vs hierarchical tier selection."""

from __future__ import annotations

from deepinfer.cli.serve import FULL_KV_MAX_TOKENS, resolve_mode


def test_auto_uses_full_kv_through_4m() -> None:
    assert resolve_mode("auto", 3_407_872) == "full"
    assert resolve_mode("auto", FULL_KV_MAX_TOKENS) == "full"


def test_auto_uses_hierarchical_above_4m() -> None:
    assert resolve_mode("auto", FULL_KV_MAX_TOKENS + 1) == "hierarchical"
    assert resolve_mode("auto", 10_000_000) == "hierarchical"


def test_explicit_mode_and_hierarchical_shortcut_win() -> None:
    assert resolve_mode("full", 10_000_000) == "full"
    assert resolve_mode("auto", 3_407_872, hierarchical=True) == "hierarchical"
