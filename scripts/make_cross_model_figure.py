"""fig9: throughput vs bytes touched per token, with iso-bandwidth curves."""

from pathlib import Path

OUT = Path("docs/figures")
FONT = "'Microsoft YaHei','PingFang SC',sans-serif"
INK, MUTED = "#1f2328", "#57606a"
BLUE, ORANGE, GREEN, PURPLE, GREY = "#2f6feb", "#d2691e", "#2da44e", "#8250df", "#8c959f"

POINTS = [
    (1.50, 42.7, "Qwen2.5-3B (ECS, VRAM)", PURPLE, 0.0, -14),
    (3.15, 26.0, "Qwen2.5-7B (ECS, VRAM)", PURPLE, 0.0, -14),
    (1.35, 24.5, "DeepSeek-V2-Lite (llama.cpp, DDR4)", BLUE, 8, 18),
    (0.73, 51.1, "OLMoE-1B-7B (llama.cpp, DDR4)", GREEN, 8, -8),
    (1.35, 2.82, "DeepSeek-V2-Lite (ECS, PCIe)", ORANGE, 10, 18),
]
CURVES = [(3.8, ORANGE, "PCIe 实测 ≈3.8 GB/s"), (34.0, BLUE, "DDR4 实测 ≈34 GB/s"),
          (72.0, PURPLE, "VRAM 实测 ≈72 GB/s")]

W, H = 900, 520
L, R, T, B = 90, 40, 80, 80
xmax, ymax = 3.6, 60.0


def sx(v):
    return L + (v / xmax) * (W - L - R)


def sy(v):
    return H - B - (v / ymax) * (H - T - B)


body = (f'<text x="{L}" y="42" font-size="20" font-weight="bold" fill="{INK}" '
        f'font-family="{FONT}">吞吐 vs 每 token 触达字节数</text>\n'
        f'<text x="{L}" y="64" font-size="13" fill="{MUTED}" font-family="{FONT}">'
        '同一条虚线 = 同一个有效带宽；点越靠近哪条线，说明瓶颈就是那条通路</text>\n')

for bw, color, label in CURVES:
    pts = []
    v = 0.15
    while v <= xmax:
        y = bw / v
        if y <= ymax:
            pts.append(f"{sx(v):.1f},{sy(y):.1f}")
        v *= 1.04
    body += (f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" '
             f'stroke-width="1.6" stroke-dasharray="6 5" opacity="0.75"/>\n')
    ylab = min(bw / xmax, ymax * 0.96)
    body += (f'<text x="{sx(xmax) - 6}" y="{sy(ylab) - 6:.0f}" font-size="11.5" fill="{color}" '
             f'text-anchor="end" font-family="{FONT}">{label}</text>\n')

body += (f'<line x1="{L}" y1="{H - B}" x2="{W - R}" y2="{H - B}" stroke="{INK}" stroke-width="1.2"/>\n'
         f'<line x1="{L}" y1="{T}" x2="{L}" y2="{H - B}" stroke="{INK}" stroke-width="1.2"/>\n')
for v in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5):
    body += (f'<line x1="{sx(v):.0f}" y1="{H - B}" x2="{sx(v):.0f}" y2="{H - B + 5}" stroke="{INK}"/>\n'
             f'<text x="{sx(v):.0f}" y="{H - B + 20}" font-size="11.5" fill="{MUTED}" '
             f'text-anchor="middle" font-family="{FONT}">{v:g}</text>\n')
for v in (0, 10, 20, 30, 40, 50, 60):
    body += (f'<line x1="{L - 5}" y1="{sy(v):.0f}" x2="{L}" y2="{sy(v):.0f}" stroke="{INK}"/>\n'
             f'<text x="{L - 10}" y="{sy(v) + 4:.0f}" font-size="11.5" fill="{MUTED}" '
             f'text-anchor="end" font-family="{FONT}">{v}</text>\n')
body += (f'<text x="{(L + W - R) / 2:.0f}" y="{H - 24}" font-size="13" fill="{INK}" '
         f'text-anchor="middle" font-family="{FONT}">每 token 触达字节数（GB）</text>\n'
         f'<text x="30" y="{(T + H - B) / 2:.0f}" font-size="13" fill="{INK}" text-anchor="middle" '
         f'font-family="{FONT}" transform="rotate(-90 30 {(T + H - B) / 2:.0f})">实测吞吐（tok/s）</text>\n')

for x, y, label, color, dx, dy in POINTS:
    body += (f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="6" fill="{color}" '
             f'stroke="#ffffff" stroke-width="1.5"/>\n')
    anchor = "start" if dx >= 0 else "end"
    body += (f'<text x="{sx(x) + dx:.1f}" y="{sy(y) + dy:.1f}" font-size="11.5" fill="{INK}" '
             f'text-anchor="{anchor}" font-family="{FONT}">{label}</text>\n')

body += (f'<text x="{L}" y="{H - 6}" font-size="11" fill="{MUTED}" font-family="{FONT}">'
         '同一个 DeepSeek-V2-Lite 在 llama.cpp（DDR4）与 ECS（PCIe）上相差 8.7×，'
         '因为两点分属不同的带宽曲线；两个不同架构的 MoE 落在同一条 DDR4 曲线上。</text>\n')

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n'
       f'<rect width="{W}" height="{H}" fill="#ffffff"/>\n{body}</svg>\n')
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "fig9_cross_model.svg").write_text(svg, encoding="utf-8")
print("wrote", OUT / "fig9_cross_model.svg")
