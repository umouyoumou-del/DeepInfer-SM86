"""Model resolution for the command-line tools: presets, -hf specs, GGUF choice, local config folders."""

from __future__ import annotations

import json

import pytest

from deepinfer.cli import models


def test_presets_and_repo_specs() -> None:
    assert models.parse_hf("coder") == ("second-state/DeepSeek-Coder-V2-Lite-Instruct-GGUF", "Q4_K_M")
    assert models.parse_hf("CHAT:Q8_0") == ("second-state/DeepSeek-V2-Lite-Chat-GGUF", "Q8_0")
    assert models.parse_hf("someone/Other-GGUF") == ("someone/Other-GGUF", "Q4_K_M")
    assert models.preset_for("DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf").name == "coder"
    assert models.preset_for("second-state/DeepSeek-V2-Lite-Chat-GGUF").name == "chat"
    assert models.preset_for("Qwen3-4B-Q4_K_M.gguf") is None


def test_the_single_file_gguf_of_a_quant_is_picked() -> None:
    files = [
        "README.md",
        "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf",
        "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_L.gguf",
        "DeepSeek-Coder-V2-Lite-Instruct-Q8_0.gguf",
        "split/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M-00001-of-00002.gguf",
    ]
    assert models.pick_gguf(files, "q4_k_m") == "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf"
    assert models.pick_gguf(files, "Q8_0") == "DeepSeek-Coder-V2-Lite-Instruct-Q8_0.gguf"
    with pytest.raises(SystemExit, match="Q4_K_L.gguf"):
        models.pick_gguf(files, "Q2_K")


def _meta(folder, **config) -> None:
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (folder / "tokenizer_config.json").write_text("{}", encoding="utf-8")


def test_a_local_gguf_finds_its_config_folder_and_trained_length(tmp_path) -> None:
    gguf = tmp_path / "models" / "my-model-Q4_K_M.gguf"
    gguf.parent.mkdir()
    gguf.write_bytes(b"GGUF")
    with pytest.raises(SystemExit, match="--meta"):
        models.resolve(str(gguf), None)
    _meta(tmp_path / "meta", max_position_embeddings=163840,
          rope_scaling={"type": "yarn", "original_max_position_embeddings": 4096})
    model = models.resolve(str(gguf), None)
    assert (model.gguf, model.meta, model.trained, model.title) == (
        gguf, tmp_path / "meta", 4096, "my-model-Q4_K_M",
    )


def test_an_explicit_config_folder_and_a_preset_name(tmp_path) -> None:
    gguf = tmp_path / "DeepSeek-V2-Lite-Chat-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF")
    _meta(tmp_path / "chat-meta", max_position_embeddings=163840)
    model = models.resolve(str(gguf), None, meta=str(tmp_path / "chat-meta"))
    assert (model.meta, model.trained, model.title) == (tmp_path / "chat-meta", 32768, "DeepSeek-V2-Lite-Chat")
    with pytest.raises(SystemExit, match="not found"):
        models.resolve(str(tmp_path / "missing.gguf"), None)


def test_a_cached_gguf_needs_no_network(monkeypatch, tmp_path) -> None:
    import huggingface_hub

    monkeypatch.setenv("DEEPINFER_CACHE", str(tmp_path))
    monkeypatch.setenv("HF_ENDPOINT", "https://mirror.invalid")
    folder = tmp_path / "models" / "org--Model-GGUF"
    (folder / ".cache").mkdir(parents=True)
    (folder / ".cache" / "Model-Q8_0.gguf").write_bytes(b"partial")  # huggingface_hub's bookkeeping
    (folder / "Model-Q4_K_M.gguf").write_bytes(b"GGUF")

    def offline(*args, **kwargs):
        raise OSError("429 Client Error: Too Many Requests for url: https://mirror.invalid/...\nWe had to rate limit you")

    monkeypatch.setattr(huggingface_hub, "list_repo_files", offline)
    assert models.download_gguf("org/Model-GGUF", "Q4_K_M") == folder / "Model-Q4_K_M.gguf"
    with pytest.raises(models.DownloadError) as error:  # not cached: one line, no traceback for the user
        models.download_gguf("org/Model-GGUF", "Q8_0")
    assert str(error.value) == (
        "could not download org/Model-GGUF from https://mirror.invalid: "
        "429 Client Error: Too Many Requests for url: https://mirror.invalid/..."
    )


def test_the_cache_folder_and_a_chosen_endpoint_are_respected(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DEEPINFER_CACHE", str(tmp_path))
    assert models.cache_dir() == tmp_path
    monkeypatch.setenv("HF_ENDPOINT", "https://example.invalid")
    assert models.use_mirror_if_needed() == "https://example.invalid"  # no probe when one is chosen
