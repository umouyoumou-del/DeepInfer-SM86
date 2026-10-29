"""Multi-part (split) GGUF support.

Large checkpoints are commonly published as ``model-00001-of-00003.gguf``.
:class:`~deepinfer.moe.gguf_index.GGUFIndex` reads a single file, so a model
that spans shards could not be opened at all -- every consumer (the MoE expert
reader and the dense loader) simply saw the first shard's tensors.

:class:`SplitGGUFIndex` presents several shards as one index with the same
surface (``tensor_names`` / ``tensor`` / ``read_tensor`` / ``expert_tensor``),
so callers do not need to know whether the model arrived in one file or six.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from deepinfer.moe.gguf_index import GGUFError, GGUFIndex

_SHARD_RE = re.compile(r"^(?P<stem>.+)-(?P<number>\d+)-of-(?P<total>\d+)\.gguf$")


class SplitGGUFError(GGUFError):
    """Raised when a shard set is incomplete or inconsistent."""


def discover_split_shards(path: str | Path) -> list[Path]:
    """Return every shard that belongs to the same model as ``path``.

    Accepts any one of the shards (or the directory containing them) and
    returns them in order, after checking the set is complete.
    """

    candidate = Path(path)
    directory = candidate if candidate.is_dir() else candidate.parent
    if candidate.is_file():
        match = _SHARD_RE.match(candidate.name)
        if match is None:
            return [candidate]
        stem, total = match.group("stem"), int(match.group("total"))
    else:
        matches = [
            (entry, _SHARD_RE.match(entry.name))
            for entry in directory.glob("*.gguf")
        ]
        matches = [(entry, match) for entry, match in matches if match]
        if not matches:
            raise SplitGGUFError(f"no split GGUF shards found in {directory}")
        stem = matches[0][1].group("stem")
        total = int(matches[0][1].group("total"))
        if any(match.group("stem") != stem for _, match in matches):
            raise SplitGGUFError(
                f"{directory} contains shards from more than one model"
            )

    shards: dict[int, Path] = {}
    for entry in directory.glob(f"{stem}-*-of-*.gguf"):
        match = _SHARD_RE.match(entry.name)
        if match is None or match.group("stem") != stem:
            continue
        if int(match.group("total")) != total:
            continue
        number = int(match.group("number"))
        if number in shards:
            raise SplitGGUFError(f"duplicate shard number {number} in {directory}")
        shards[number] = entry
    missing = sorted(set(range(1, total + 1)) - set(shards))
    if missing:
        raise SplitGGUFError(
            f"{stem}: missing shard(s) {missing} of {total}"
        )
    return [shards[number] for number in sorted(shards)]


class SplitGGUFIndex:
    """One index over an ordered list of shards."""

    def __init__(
        self,
        paths: Sequence[str | Path],
        *,
        use_mmap: bool = True,
    ) -> None:
        if not paths:
            raise SplitGGUFError("at least one shard is required")
        self.paths = [Path(path) for path in paths]
        self.parts = [GGUFIndex(path, use_mmap=use_mmap) for path in self.paths]
        self.metadata = dict(self.parts[0].metadata)
        self.version = self.parts[0].version
        self._owner: dict[str, int] = {}
        for index, part in enumerate(self.parts):
            for name in part.tensors:
                if name in self._owner:
                    raise SplitGGUFError(
                        f"tensor {name!r} appears in more than one shard"
                    )
                self._owner[name] = index
        self._validate_split_metadata()

    def _validate_split_metadata(self) -> None:
        counts = {
            int(part.metadata.get("split.count", len(self.parts)))
            for part in self.parts
        }
        if len(counts) > 1:
            raise SplitGGUFError(f"shards disagree on split.count: {sorted(counts)}")
        declared = counts.pop()
        if declared != len(self.parts):
            raise SplitGGUFError(
                f"split.count declares {declared} shards, opened {len(self.parts)}"
            )
        numbers = {int(part.metadata.get("split.no", -1)) for part in self.parts}
        if -1 not in numbers and len(numbers) != len(self.parts):
            raise SplitGGUFError("shards disagree on split.no")

    # ---------------------------------------------------------------- queries

    @property
    def shard_count(self) -> int:
        return len(self.parts)

    def tensor_names(self, prefix: str = "") -> list[str]:
        names = sorted(self._owner)
        if not prefix:
            return names
        return [name for name in names if name.startswith(prefix)]

    def tensor(self, name: str) -> Any:
        try:
            part = self.parts[self._owner[name]]
        except KeyError:
            raise KeyError(f"tensor {name!r} is not present in any shard") from None
        return part.tensor(name)

    def read_tensor(self, name: str) -> Any:
        self.tensor(name)  # raises KeyError with a shard-aware message
        return self.parts[self._owner[name]].read_tensor(name)

    def expert_tensor(self, name: str, expert_index: int, **kwargs: Any) -> Any:
        self.tensor(name)
        return self.parts[self._owner[name]].expert_tensor(
            name, expert_index, **kwargs
        )

    def tensor_owner(self, name: str) -> int:
        try:
            return self._owner[name]
        except KeyError:
            raise KeyError(f"tensor {name!r} is not present in any shard") from None

    def close(self) -> None:
        for part in self.parts:
            close = getattr(part, "close", None)
            if close is not None:
                close()

    def stats(self) -> dict[str, Any]:
        return {
            "shards": self.shard_count,
            "tensors": len(self._owner),
            "paths": [str(path) for path in self.paths],
        }

    def __len__(self) -> int:
        return len(self._owner)


def open_gguf(path: str | Path, **kwargs: Any) -> Any:
    """Open a single file, or a whole shard set, transparently."""

    candidate = Path(path)
    if candidate.is_dir():
        return SplitGGUFIndex(discover_split_shards(candidate), **kwargs)
    if _SHARD_RE.match(candidate.name):
        shards = discover_split_shards(candidate)
        return SplitGGUFIndex(shards, **kwargs)
    return GGUFIndex(candidate, **kwargs)


__all__ = [
    "SplitGGUFError",
    "SplitGGUFIndex",
    "discover_split_shards",
    "open_gguf",
]
