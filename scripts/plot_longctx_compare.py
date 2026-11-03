"""Figures 2 and 3 of docs/PAPER_LONG_CONTEXT.md from the comparison results: accuracy and prefill time
against context length (log scale), one SVG each, drawn by hand (no plotting library on this machine).

    python scripts/plot_longctx_compare.py docs/figures
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# measured points: context length (tokens) -> value; see docs/results/longctx/baselines/
ACCURACY = {  # questions fully right, of 21
    "deepinfer": {16384: 21, 65536: 21, 98304: 20, 1048576: 21, 2097152: 21, 3145728: 21, 4194304: 19},
    "llama": {16384: 15, 65536: 7, 98304: 3, 1048576: 0},
    "plain": {16384: 16, 65536: 6, 98304: 0},
}
PREFILL = {  # seconds to read the document
    "deepinfer": {16384: 2.2, 65536: 13.7, 98304: 26.4, 1048576: 524.9, 2097152: 16.8 * 60, 3145728: 25.3 * 60, 4194304: 2030.4},
    "llama": {16384: 4.4, 65536: 50.5, 98304: 104.5, 1048576: 5957.5},
}
SERIES = {  # palette slots 1-3, validated all-pairs (light surface)
    "deepinfer": ("#2a78d6", "本系统：定位 + 重读"),
    "llama": ("#eb6834", "llama.cpp：整段直接回答"),
    "plain": ("#1baf7a", "本系统：整段直接回答（对照）"),
}
SHORT = {"deepinfer": "定位 + 重读", "llama": "llama.cpp", "plain": "本系统直接回答"}
SURFACE, GRID, AXIS, MUTED, INK, INK2 = "#fcfcfb", "#e1e0d9", "#c3c2b7", "#898781", "#0b0b0b", "#52514e"
FONT = "Microsoft YaHei, PingFang SC, sans-serif"
WIDTH, HEIGHT = 760, 420
LEFT, RIGHT, TOP, BOTTOM = 72, 132, 70, 64  # plot margins; the right one leaves room for end labels
X_TICKS = [(16384, "16K"), (65536, "64K"), (262144, "256K"), (1048576, "1M"), (3145728, "3M"), (4194304, "4M")]


def x_of(tokens: float) -> float:
    lo, hi = math.log2(16384), math.log2(4194304)
    return LEFT + (math.log2(tokens) - lo) / (hi - lo) * (WIDTH - LEFT - RIGHT)


def frame(title: str, subtitle: str, y_ticks: list[tuple[float, str]], y_of) -> list[str]:
    parts = [
        (f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" '
         f'font-family="{FONT}">'),
        f'<rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="{SURFACE}"/>',
        f'<text x="{LEFT}" y="24" font-size="15" fill="{INK}">{title}</text>',
        f'<text x="{LEFT}" y="42" font-size="12" fill="{INK2}">{subtitle}</text>',
    ]
    for value, label in y_ticks:  # hairline grid, solid
        y = y_of(value)
        parts.append(f'<line x1="{LEFT}" y1="{y:.1f}" x2="{WIDTH - RIGHT}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(f'<text x="{LEFT - 8}" y="{y + 4:.1f}" font-size="11" fill="{MUTED}" text-anchor="end">{label}</text>')
    base = HEIGHT - BOTTOM
    parts.append(f'<line x1="{LEFT}" y1="{base}" x2="{WIDTH - RIGHT}" y2="{base}" stroke="{AXIS}" stroke-width="1"/>')
    for tokens, label in X_TICKS:
        x = x_of(tokens)
        parts.append(f'<line x1="{x:.1f}" y1="{base}" x2="{x:.1f}" y2="{base + 4}" stroke="{AXIS}" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{base + 18}" font-size="11" fill="{MUTED}" text-anchor="middle">{label}</text>')
    parts.append(f'<text x="{(LEFT + WIDTH - RIGHT) / 2:.1f}" y="{HEIGHT - 18}" font-size="12" fill="{INK2}" '
                 f'text-anchor="middle">上下文长度（token，对数刻度）</text>')
    return parts


def series(parts: list[str], data: dict, y_of, order: list[str], end_labels: dict[str, str]) -> None:
    ends = []
    for name in order:
        color = SERIES[name][0]
        points = [(x_of(t), y_of(v)) for t, v in sorted(data[name].items()) if v is not None]
        if not points:
            continue
        if len(points) > 1:
            path = " ".join(f"{'M' if i == 0 else 'L'} {x:.1f} {y:.1f}" for i, (x, y) in enumerate(points))
            parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2" '
                         f'stroke-linejoin="round" stroke-linecap="round"/>')
        for x, y in points:  # 8px markers with a 2px surface ring
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{SURFACE}"/>')
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        ends.append((points[-1], name))
    # end labels in ink beside each series' last point (left of it when the right edge is near),
    # nudged apart only where two would overlap
    placed: list[tuple[float, float]] = []
    for (x, y), name in sorted(ends, key=lambda item: item[0][1]):
        text = end_labels[name]
        width = sum(12 if ord(ch) > 0x2E80 else 6.5 for ch in text)
        right_side = x + 10 + width <= WIDTH - 8
        lx = x + 10 if right_side else x - 10
        label_y = y if y < HEIGHT - BOTTOM - 8 else y - 12  # not on the baseline
        for px, py in placed:
            if abs(label_y - py) < 14 and abs(lx - px) < width + 20:
                label_y = py + 14
        placed.append((lx, label_y))
        anchor = "start" if right_side else "end"
        parts.append(f'<text x="{lx:.1f}" y="{label_y + 4:.1f}" font-size="11" fill="{INK}" text-anchor="{anchor}">{text}</text>')


def legend(parts: list[str], order: list[str]) -> None:
    x = LEFT
    for name in order:
        color, label = SERIES[name]
        parts.append(f'<line x1="{x}" y1="{TOP - 10}" x2="{x + 18}" y2="{TOP - 10}" stroke="{color}" stroke-width="2" stroke-linecap="round"/>')
        parts.append(f'<circle cx="{x + 9}" cy="{TOP - 10}" r="4" fill="{color}"/>')
        parts.append(f'<text x="{x + 24}" y="{TOP - 6}" font-size="11" fill="{INK2}">{label}</text>')
        x += 24 + 12 * len(label) + 18


def accuracy_svg() -> str:
    top, base = TOP + 18, HEIGHT - BOTTOM

    def y_of(value: float) -> float:
        return base - value / 21 * (base - top)

    parts = frame("同一套 39 针题，答对的题数（满分 21）", "每个长度用逐 token 相同的文档；llama.cpp 在 1M 以上显存装不下；vLLM 最长 160K，98K 同为 3/21",
                  [(0, "0"), (7, "7"), (14, "14"), (21, "21")], y_of)
    legend(parts, ["deepinfer", "llama", "plain"])
    labels = {}
    for name in ("deepinfer", "llama", "plain"):
        last = [(t, v) for t, v in sorted(ACCURACY[name].items()) if v is not None]
        if last:
            labels[name] = f"{SHORT[name]} {last[-1][1]}/21"
    series(parts, ACCURACY, y_of, ["plain", "llama", "deepinfer"], labels)
    parts.append("</svg>")
    return "\n".join(parts)


def prefill_svg() -> str:
    top, base = TOP + 18, HEIGHT - BOTTOM
    lo, hi = math.log10(2), math.log10(3 * 3600)

    def y_of(seconds: float) -> float:
        return base - (math.log10(seconds) - lo) / (hi - lo) * (base - top)

    ticks = [(10, "10 s"), (60, "1 min"), (600, "10 min"), (3600, "1 h")]
    parts = frame("读入文档用的时间", "llama.cpp：全量注意力；本系统：有界编码（每个 token 看前 96K）",
                  ticks, y_of)
    names = {"deepinfer": ("#2a78d6", "本系统：有界编码"), "llama": ("#eb6834", "llama.cpp：全量注意力")}
    saved = dict(SERIES)
    SERIES.update(names)
    legend(parts, ["deepinfer", "llama"])
    labels = {}
    for name in ("deepinfer", "llama"):
        last = [(t, v) for t, v in sorted(PREFILL[name].items()) if v is not None]
        if last:
            value = last[-1][1]
            labels[name] = f"{names[name][1].split('：')[0]} {value / 60:.1f} min" if value >= 120 else \
                f"{names[name][1].split('：')[0]} {value:.0f} s"
    series(parts, PREFILL, y_of, ["llama", "deepinfer"], labels)
    SERIES.clear()
    SERIES.update(saved)
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/figures")
    (target / "fig11_longctx_accuracy.svg").write_text(accuracy_svg(), encoding="utf-8")
    (target / "fig12_longctx_prefill.svg").write_text(prefill_svg(), encoding="utf-8")
    print("wrote", target / "fig11_longctx_accuracy.svg", target / "fig12_longctx_prefill.svg")


if __name__ == "__main__":
    main()
