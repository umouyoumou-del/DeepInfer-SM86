"""One table from the needle_longctx.py result files.

    python scripts/summarize_needles.py results/*.json
"""

from __future__ import annotations

import json
import sys


def _as_list(value) -> list[float]:
    """Early result files stored one prefill time, later ones a list."""

    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


def _attention(data: dict, retrieval: dict | None) -> str:
    if data.get("reread"):
        return f"locate+reread top{data['reread']}"
    if retrieval:
        return f"top{retrieval['top_blocks']}" + ("+repos" if retrieval.get("reposition") else "")
    return "full"


def main(paths: list[str]) -> None:
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        hits = [needle["hit"] for needle in data["needles"]]
        retrieval = data.get("retrieval")
        rows.append(
            (
                data["tokens"],
                data.get("mode", "multi"),
                data["positions"],
                data["kv"],
                " ".join(
                    part
                    for part in (
                        f"window {data['encode_window'] // 1024}K" if data.get("encode_window") else "",
                        "logn" if data.get("logn") else "",
                        f"far={data['far']}"
                        if data.get("far") and data.get("trained") and data["far"] != data["trained"] - 1
                        else "",
                    )
                    if part
                ),
                _attention(data, retrieval),
                f"{sum(hits)}/{len(hits)}",
                "".join("x" if hit else "." for hit in hits),
                _as_list(data.get("prefill_seconds")),
                data.get("cache_load_seconds"),
                path.rsplit("/", 1)[-1],
            )
        )
    rows.sort(key=lambda row: (row[0], row[1], row[5], row[4]))
    print(
        "| tokens | mode | positions | kv | encode / scaling | question | recall | by depth | prefill | load | file |"
    )
    print("|---:|---|---|---|---|---|---:|---|---:|---:|---|")
    for tokens, mode, positions, kv, scaling, attention, recall, pattern, prefill, load, name in rows:
        prefill_text = f"{prefill[0] / 60:.1f} min" if prefill else ""
        load_text = f"{load:.0f} s" if load else ""
        print(
            f"| {tokens:,} | {mode} | {positions} | {kv} | {scaling} | {attention} | {recall} "
            f"| `{pattern}` | {prefill_text} | {load_text} | {name} |"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
