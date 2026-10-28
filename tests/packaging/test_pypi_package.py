"""Checks for the standalone PyPI source tree."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "pypi" / "deepinfer-sm86"


def test_pypi_project_and_package_data() -> None:
    pyproject = (PACKAGE / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "deepinfer-sm86"' in pyproject
    assert 'version = "0.1.0"' in pyproject
    assert 'include = ["deepinfer*"]' in pyproject
    assert 'package-dir = { "" = "src" }' in pyproject
    assert 'license-files = ["LICENSE", "NOTICE"]' in pyproject
    for relative in (
        "src/deepinfer/moe/cpu_avx2_packed.cpp",
        "src/deepinfer/moe/cpu_avx2_gemv.cpp",
        "src/deepinfer/moe/deepseek_v2_runtime.py",
        "src/deepinfer/moe/paged_mla_quant.py",
        "src/deepinfer/scheduler/paged.py",
        "src/deepinfer/py.typed",
        "README.md",
        "LICENSE",
        "NOTICE",
        "Makefile",
    ):
        assert (PACKAGE / relative).is_file(), relative


def test_pypi_scheduler_is_decoupled_from_apps() -> None:
    source = (PACKAGE / "src/deepinfer/scheduler/paged.py").read_text(
        encoding="utf-8"
    )
    assert "from apps." not in source
    assert "QwenSm86Runtime" not in source
    scheduler_init = (PACKAGE / "src/deepinfer/scheduler/__init__.py").read_text(
        encoding="utf-8"
    )
    assert "PagedBatchScheduler" in scheduler_init
