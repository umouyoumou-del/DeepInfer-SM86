"""Which model to run: a local GGUF (``-m``) or one fetched from Hugging Face (``-hf``).

The runtime reads the weights from the GGUF and the model's config and tokenizer from a
small directory of Hugging Face files next to it (``config.json``, ``tokenizer.json``,
``modeling_deepseek.py`` ...).  ``-hf`` downloads both: the GGUF from a quantized repo and
those files from the original repo.  Two presets cover the models this project is built
and measured on.  Downloads go to ``~/.cache/deepinfer`` (``DEEPINFER_CACHE`` overrides)
and are used from there without the network afterwards; when huggingface.co does not
answer and ``HF_ENDPOINT`` is unset, the hf-mirror.com mirror is used.
"""

from __future__ import annotations

import contextlib
import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Preset:
    name: str
    title: str
    gguf_repo: str
    meta_repo: str
    trained: int  # context length the model was trained to


# second-state's Q4_K_M files are the ones every documented result was measured on (sha256-checked)
PRESETS = {
    "coder": Preset(
        "coder", "DeepSeek-Coder-V2-Lite-Instruct", "second-state/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
        "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", 131072,
    ),
    "chat": Preset(
        "chat", "DeepSeek-V2-Lite-Chat", "second-state/DeepSeek-V2-Lite-Chat-GGUF",
        "deepseek-ai/DeepSeek-V2-Lite-Chat", 32768,
    ),
}
META_FILES = (
    "config.json", "configuration_deepseek.py", "modeling_deepseek.py", "generation_config.json",
    "tokenizer.json", "tokenizer_config.json",
)
DEFAULT_QUANT = "Q4_K_M"
MIRROR = "https://hf-mirror.com"


@dataclass
class Model:
    gguf: Path
    meta: Path
    trained: int
    title: str


def cache_dir() -> Path:
    return Path(os.environ.get("DEEPINFER_CACHE") or Path.home() / ".cache" / "deepinfer")


def preset_for(text: str) -> Preset | None:
    """The preset a name, repo id or file name refers to."""

    lowered = text.lower()
    if lowered in PRESETS:
        return PRESETS[lowered]
    if "coder-v2-lite" in lowered:
        return PRESETS["coder"]
    if "v2-lite-chat" in lowered:
        return PRESETS["chat"]
    return None


def parse_hf(spec: str) -> tuple[str, str]:
    """``repo[:QUANT]`` or a preset name -> ``(repo, quant)``."""

    repo, _, quant = spec.partition(":")
    preset = PRESETS.get(repo.lower())
    return (preset.gguf_repo if preset else repo), (quant or DEFAULT_QUANT)


def _matching(files: list[str], quant: str) -> list[str]:
    """Single-file GGUFs of ``quant``, the plainest name first."""

    tag = quant.lower()
    matches = [
        name for name in files
        if name.lower().endswith(".gguf") and tag in name.lower() and "-of-" not in name.lower()
    ]
    return sorted(matches, key=len)


def pick_gguf(files: list[str], quant: str) -> str:
    """The single-file GGUF of ``quant`` among a repo's files."""

    matches = _matching(files, quant)
    if not matches:
        found = sorted({name.rsplit("/", 1)[-1] for name in files if name.lower().endswith(".gguf")})
        raise SystemExit(f"no {quant} GGUF in that repo; it has: {', '.join(found) or 'no GGUF files'}")
    return matches[0]


def use_mirror_if_needed(timeout: float = 4.0) -> str:
    """Point ``HF_ENDPOINT`` at the mirror when huggingface.co does not answer (call before importing huggingface_hub)."""

    if os.environ.get("HF_ENDPOINT"):
        return os.environ["HF_ENDPOINT"]
    try:
        urllib.request.urlopen("https://huggingface.co/api/models?limit=1", timeout=timeout).close()
        return "https://huggingface.co"
    except Exception:  # noqa: BLE001 - any failure means: try the mirror
        os.environ["HF_ENDPOINT"] = MIRROR
        print(f"huggingface.co did not answer; downloading from {MIRROR} (set HF_ENDPOINT to choose)", flush=True)
        return MIRROR


class DownloadError(RuntimeError):
    """A model's files could not be fetched: no network, a rate limit, a missing repo or file."""


@contextlib.contextmanager
def _downloading(what: str):
    endpoint = use_mirror_if_needed()
    try:
        yield
    except Exception as error:  # one line for the user instead of a traceback
        text = str(error).strip()
        raise DownloadError(
            f"could not download {what} from {endpoint}: {text.splitlines()[0] if text else type(error).__name__}"
        ) from error


def download_gguf(repo: str, quant: str) -> Path:
    """The repo's ``quant`` GGUF, downloaded once; later runs use the cached file without the network."""

    target = cache_dir() / "models" / repo.replace("/", "--")
    if target.is_dir():
        cached = [path.relative_to(target).as_posix() for path in target.rglob("*.gguf") if ".cache" not in path.parts]
        if _matching(cached, quant):
            return target / _matching(cached, quant)[0]
    with _downloading(repo):
        from huggingface_hub import hf_hub_download, list_repo_files

        filename = pick_gguf(list_repo_files(repo), quant)
        print(f"model: {repo}/{filename} -> {target}", flush=True)
        return Path(hf_hub_download(repo, filename, local_dir=target))


def download_meta(repo: str) -> Path:
    target = cache_dir() / "meta" / repo.replace("/", "--")
    if all((target / name).is_file() for name in META_FILES):
        return target
    with _downloading(f"{repo}'s config and tokenizer"):
        from huggingface_hub import snapshot_download

        print(f"config and tokenizer: {repo} -> {target}", flush=True)
        snapshot_download(repo, allow_patterns=list(META_FILES), local_dir=target)
    return target


def find_meta(gguf: Path) -> Path | None:
    """A config/tokenizer directory already next to a local GGUF, where the runtime looks for one."""

    for candidate in (gguf.parent / "meta", gguf.parent.parent / "meta", gguf.parent):
        if (candidate / "config.json").is_file() and (candidate / "tokenizer_config.json").is_file():
            return candidate
    return None


def trained_length(meta: Path, preset: Preset | None) -> int:
    if preset is not None:
        return preset.trained
    config = json.loads((meta / "config.json").read_text(encoding="utf-8"))
    scaling = config.get("rope_scaling") or {}
    return int(scaling.get("original_max_position_embeddings") or config.get("max_position_embeddings") or 4096)


def resolve(model: str | None, hf: str | None, meta: str | None = None, meta_hf: str | None = None) -> Model:
    """Turn the command line's model options into files on disk, downloading what is missing."""

    if not model and not hf:
        hf = "coder"
    if model:
        gguf = Path(model).expanduser()
        if not gguf.is_file():
            raise SystemExit(f"model file not found: {gguf}")
        preset = preset_for(gguf.name)
    else:
        repo, quant = parse_hf(hf)
        preset = preset_for(hf) or preset_for(repo)
        gguf = download_gguf(repo, quant)
    if meta:
        meta_dir = Path(meta).expanduser()
    elif meta_hf:
        meta_dir = download_meta(meta_hf)
    elif preset is not None:
        # the preset's own files (small, cached); a folder next to the GGUF may belong to another model
        try:
            meta_dir = download_meta(preset.meta_repo)
        except DownloadError:  # offline: fall back to what is on disk
            meta_dir = find_meta(gguf)
            if meta_dir is None:
                raise
            print(f"offline: using the config and tokenizer in {meta_dir}", flush=True)
    else:
        meta_dir = find_meta(gguf)
        if meta_dir is None:
            raise SystemExit(
                "no config/tokenizer files for this GGUF: pass --meta DIR (a folder with config.json and "
                "tokenizer files) or --meta-hf ORG/REPO (the original model's Hugging Face repo)"
            )
    return Model(gguf=gguf, meta=meta_dir, trained=trained_length(meta_dir, preset),
                 title=preset.title if preset else gguf.stem)
