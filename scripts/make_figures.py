"""Generate the blog figures as dependency-free SVG.

matplotlib is not installable in this environment (pip times out), and the
figures are simple enough that emitting SVG directly is both faster and
sharper.  Text uses a font stack that resolves to Microsoft YaHei on Windows
and to any reasonable CJK sans elsewhere.
"""

from __future__ import annotations

from pathlib import Path

OUT = Path("docs/figures")
FONT = "'Microsoft YaHei','PingFang SC','Noto Sans CJK SC',sans-serif"
INK = "#1f2328"
MUTED = "#57606a"
BLUE = "#2f6feb"
ORANGE = "#d2691e"
GREEN = "#2da44e"
GREY = "#8c959f"


def _svg(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="{FONT}">\n'
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>\n'
        f"{body}\n</svg>\n"
    )


def _text(x, y, s, size=14, fill=INK, weight="normal", anchor="start"):
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
        f'font-weight="{weight}" text-anchor="{anchor}">{s}</text>\n'
    )


def bandwidth_chart() -> str:
    rows = [
        ("手机 SoC  LPDDR5X", 60, GREEN, "统一内存"),
        ("笔记本 CPU  DDR4-3200", 35, BLUE, "统一内存"),
        ("独立 GPU  PCIe 3.0 实际", 7, ORANGE, "分离，要搬运"),
        ("手机 UFS 4.0 闪存", 3, GREY, "按页流式"),
    ]
    w, h = 860, 380
    left, top, bar_h, gap = 250, 80, 44, 26
    scale = 7.4  # px per GB/s
    body = _text(left, 42, "各平台的有效带宽（GB/s）", 20, INK, "bold")
    body += _text(left, 64, "同一份 MoE 每生成一个 token 需触达约 1.35 GB 权重", 13, MUTED)
    for i, (label, value, color, note) in enumerate(rows):
        y = top + i * (bar_h + gap)
        body += _text(left - 16, y + bar_h * 0.68, label, 14, INK, anchor="end")
        body += (
            f'<rect x="{left}" y="{y}" width="{value * scale:.0f}" height="{bar_h}" '
            f'rx="4" fill="{color}" opacity="0.85"/>\n'
        )
        body += _text(left + value * scale + 10, y + bar_h * 0.68, f"{value}", 15, INK, "bold")
        body += _text(left + value * scale + 52, y + bar_h * 0.68, note, 12, MUTED)
    body += _text(
        left, h - 26,
        "数据口径：DDR4/PCIe 为实测区间中点；LPDDR5X、UFS 4.0 为公开规格的典型值。",
        12, MUTED,
    )
    return _svg(w, h, body)


def data_path_chart() -> str:
    w, h = 900, 420
    body = _text(40, 42, "同一个 token 的两条数据通路", 20, INK, "bold")

    # row 1: discrete GPU
    body += _text(40, 96, "① 独立显卡架构（本文实测）", 15, ORANGE, "bold")
    boxes = [
        ("NVMe", "9.65 GB 全量"),
        ("主机内存", "pinned 缓冲"),
        ("PCIe 3.0", "6–8 GB/s"),
        ("显存 4 GB", "2.23 GiB 峰值"),
    ]
    x, y, bw, bh = 40, 116, 180, 72
    for i, (title, sub) in enumerate(boxes):
        bx = x + i * (bw + 26)
        stroke = ORANGE if i == 2 else GREY
        body += (
            f'<rect x="{bx}" y="{y}" width="{bw}" height="{bh}" rx="8" fill="#fff" '
            f'stroke="{stroke}" stroke-width="{2 if i == 2 else 1.4}"/>\n'
        )
        body += _text(bx + bw / 2, y + 32, title, 15, INK, "bold", "middle")
        body += _text(bx + bw / 2, y + 54, sub, 12, MUTED, anchor="middle")
        if i < len(boxes) - 1:
            body += (
                f'<path d="M {bx + bw + 3} {y + bh / 2} L {bx + bw + 22} {y + bh / 2}" '
                f'stroke="{INK}" stroke-width="1.6" marker-end="url(#a)"/>\n'
            )
    body += _text(
        40, y + bh + 30,
        "每 token 都要搬 1.35 GB 过 PCIe → 上限约 5 tok/s（实测 2.64–3.01）",
        13, ORANGE,
    )

    # row 2: unified memory
    y2 = 286
    body += _text(40, y2 - 12, "② 统一内存架构（手机 SoC）", 15, GREEN, "bold")
    body += (
        f'<rect x="40" y="{y2}" width="{800}" height="86" rx="10" fill="{GREEN}" '
        f'opacity="0.08" stroke="{GREEN}" stroke-width="1.6"/>\n'
    )
    body += _text(60, y2 + 34, "统一内存（LPDDR5X，50–70 GB/s）", 15, INK, "bold")
    body += _text(60, y2 + 58, "CPU · GPU · NPU 共用同一块物理内存：不需要 pinned 缓冲、不需要 H2D、不需要双缓冲槽位", 12, MUTED)
    body += _text(640, y2 + 34, "UFS 4.0", 14, INK, "bold", "middle")
    body += _text(640, y2 + 58, "冷门专家按需读", 12, MUTED, anchor="middle")
    body += (
        f'<path d="M 560 {y2 + 34} L 556 {y2 + 34}" stroke="{INK}" stroke-width="1.6" '
        f'marker-end="url(#a)"/>\n'
    )
    body += _text(40, h - 18, "热专家常驻 LPDDR5X → 理论上限 30–35 tok/s；全冷走 UFS → 1.5–3 tok/s", 13, GREEN)

    defs = (
        '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{INK}"/></marker></defs>\n'
    )
    return _svg(w, h, defs + body)


def throughput_chart() -> str:
    rows = [
        ("llama.cpp 热态\n（页缓存已热）", 23.7, 25.5, BLUE, "实测"),
        ("ECS\n（本文，PCIe 流式）", 2.64, 3.01, ORANGE, "实测"),
        ("手机预测 · 全冷\n（走 UFS 4.0）", 1.5, 3.0, GREY, "推算"),
        ("手机预测 · 热专家常驻\n（LPDDR5X）", 30.0, 35.0, GREEN, "推算"),
    ]
    w, h = 900, 430
    left, top, row_h, gap = 250, 86, 34, 40
    scale = 17.0
    body = _text(left, 42, "解码吞吐区间（token/s）", 20, INK, "bold")
    body += _text(left, 64, "实测 vs 推算；条形表示区间，非均值", 13, MUTED)
    for i, (label, lo, hi, color, kind) in enumerate(rows):
        y = top + i * (row_h + gap)
        for j, part in enumerate(label.split("\n")):
            body += _text(left - 16, y + 14 + j * 17, part, 13.5, INK, anchor="end")
        x0, x1 = left + lo * scale, left + hi * scale
        body += (
            f'<rect x="{x0:.0f}" y="{y}" width="{max(3, x1 - x0):.0f}" height="{row_h}" '
            f'rx="5" fill="{color}" opacity="0.85"/>\n'
        )
        body += _text(x1 + 12, y + row_h * 0.7, f"{lo:g} – {hi:g}", 13.5, INK, "bold")
        body += _text(x1 + 96, y + row_h * 0.7, kind, 12, MUTED)
    body += (
        f'<line x1="{left + 5 * scale}" y1="{top - 18}" x2="{left + 5 * scale}" '
        f'y2="{h - 70}" stroke="{ORANGE}" stroke-width="1.4" stroke-dasharray="5 4"/>\n'
    )
    body += _text(left + 5 * scale + 8, h - 74, "预测的可证伪阈值：5 tok/s", 12, ORANGE)
    body += _text(
        left, h - 26,
        "llama.cpp 与 ECS 为同机同模型实测（RTX 3050 Laptop 4 GB + Ryzen 5 5600H）；手机两行为按 1.35 GB/token 推算的区间。",
        11.5, MUTED,
    )
    return _svg(w, h, body)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    figures = {
        "fig1_bandwidth.svg": bandwidth_chart(),
        "fig2_data_path.svg": data_path_chart(),
        "fig3_throughput.svg": throughput_chart(),
    }
    for name, content in figures.items():
        (OUT / name).write_text(content, encoding="utf-8")
        print(f"wrote {OUT / name} ({len(content)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
