"""The suite's haystack: the standard library's source, read again from the top when a document needs more."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("torch")

from deepinfer.longctx_suite import haystack_tokens


class _Chars:
    """A tokenizer with one token per character."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]


def _library(root) -> str:
    (root / "a.py").write_text("x = 1\n", encoding="utf-8")
    (root / "pkg").mkdir()
    (root / "pkg" / "b.py").write_text("y = 2\n", encoding="utf-8")
    (root / "site-packages").mkdir()
    (root / "site-packages" / "c.py").write_text("z = 3\n", encoding="utf-8")
    return str(root)


def _text(ids: list[int]) -> str:
    return "".join(map(chr, ids))


def test_haystack_reads_the_library_again_when_it_runs_out(tmp_path) -> None:
    root = _library(tmp_path)
    once = f"\n# a.py\nx = 1\n\n# {os.path.join('pkg', 'b.py')}\ny = 2\n"  # installed packages are left out
    assert _text(haystack_tokens(_Chars(), root, 5)) == once[:5]
    assert _text(haystack_tokens(_Chars(), root, len(once))) == once
    assert _text(haystack_tokens(_Chars(), root, 2 * len(once) + 3)) == once + once + once[:3]


def test_haystack_needs_some_source(tmp_path) -> None:
    with pytest.raises(RuntimeError):
        haystack_tokens(_Chars(), str(tmp_path), 10)
