"""Experimental figures for the ECS paper, emitted as dependency-free SVG."""

from __future__ import annotations

from pathlib import Path

OUT = Path("docs/figures")
FONT = "'Microsoft YaHei','PingFang SC','Noto Sans CJK SC',sans-serif"
INK, MUTED = "#1f2328", "#57606a"
BLUE, ORANGE, GREEN, PURPLE, GREY = "#2f6feb", "#d2691e", "#2da44e", "#8250df", "#8c959f"


def svg(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" font-family="{FONT}">\n'
            f'<rect width="{w}" height="{h}" fill="#ffffff"/>\n{body}\n</svg>\n')


def t(x, y, s, size=13, fill=INK, weight="normal", anchor="start"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{s}</text>\n')


def rect(x, y, w, h, color, op=0.85, r=4):
    return (f'<rect x="{x:.0f}" y="{y:.0f}" width="{max(2, w):.0f}" height="{h:.0f}" '
            f'rx="{r}" fill="{color}" opacity="{op}"/>\n')


def hbars(title, sub, rows, *, scale, unit="", w=900, note="", bar_h=30, gap=16):
    """rows: (label, lo, hi, color, value_text)"""
    top = 92
    h = top + len(rows) * (bar_h + gap) + 60
    left = 300
    body = t(left, 42, title, 20, INK, "bold") + t(left, 64, sub, 13, MUTED)
    for i, (label, lo, hi, color, value_text) in enumerate(rows):
        y = top + i * (bar_h + gap)
        body += t(left - 16, y + bar_h * 0.7, label, 13.5, INK, anchor="end")
        x0, x1 = left + lo * scale, left + hi * scale
        body += rect(x0, y, x1 - x0, bar_h, color)
        body += t(x1 + 12, y + bar_h * 0.7, value_text, 13, INK, "bold")
    body += t(left, h - 24, note, 11.5, MUTED)
    return svg(w, h, body)


def fig4_single_gpu():
    rows = [
        ("Qwen2.5-3B  W4A16 + GQA + Graph", 0, 42.7, GREEN, "42.7 tok/s  ·  2.49 GiB"),
        ("Qwen2.5-7B  混合 4/3 bit", 0, 25.96, BLUE, "26.0 tok/s  ·  3.20 GiB"),
        ("Qwen2.5-7B  整体 Q4", 0, 7.94, GREY, "7.9 tok/s  ·  3.92 GiB（持续 paging）"),
        ("DeepSeek-V2-Lite  MoE Q4_K_M", 0, 2.82, ORANGE, "2.64–3.01 tok/s  ·  2.23 GiB"),
    ]
    return hbars("单卡实测：4 GB RTX 3050 Laptop",
                 "四条路径的目标不同，不能直接互相比较；括号内为峰值显存",
                 rows, scale=17, note="数据来源：README《结果速览》、docs/results/")


def fig5_moe_ablation():
    rows = [
        ("整层 CPU offload（1 层）", 0, 1.86, GREY, "1.86"),
        ("反量化 + grouped bmm", 2.43, 2.77, MUTED, "2.43–2.77"),
        ("冷热专家切分（2 层 × 1 cold）", 0, 2.50, PURPLE, "2.50"),
        ("逐专家 fused", 2.56, 3.05, BLUE, "2.56–3.05"),
        ("grouped fused + 双缓冲", 2.64, 3.01, GREEN, "2.64–3.01"),
    ]
    return hbars("MoE 消融：同一份 Q4_K_M，单请求贪心解码",
                 "单位 tok/s；条形表示实测区间",
                 rows, scale=150, note="数据来源：docs/ARTICLE_9_DEEPSEEK_V2_LITE_MOE_CPU_GPU_OFFLOAD.md")


def fig6_batch_scaling():
    batches = [(1, 4.20, 1.00), (2, 6.32, 1.50), (4, 16.60, 3.95), (8, 25.90, 6.17)]
    w, h, left, top, bw, gap, scale = 860, 420, 110, 90, 110, 60, 20
    body = t(left, 42, "双卡固定批扩展（聚合解码吞吐）", 20, INK, "bold")
    body += t(left, 64, "批 1 为单请求基线 4.20 tok/s；柱上标注相对单卡加速比", 13, MUTED)
    base = h - 90
    for i, (b, tps, sp) in enumerate(batches):
        x = left + i * (bw + gap)
        bh = tps * scale
        color = GREEN if b == 8 else BLUE
        body += rect(x, base - bh, bw, bh, color)
        body += t(x + bw / 2, base - bh - 26, f"{tps:.2f}", 15, INK, "bold", "middle")
        body += t(x + bw / 2, base - bh - 8, f"{sp:.2f}×", 12, GREEN if sp > 1 else MUTED, "bold", "middle")
        body += t(x + bw / 2, base + 22, f"batch {b}", 14, INK, anchor="middle")
    body += f'<line x1="{left - 20}" y1="{base}" x2="{w - 40}" y2="{base}" stroke="{INK}" stroke-width="1"/>\n'
    body += t(left - 20, base + 52, "所有批大小的输出 token 与单卡基线完全一致", 12, MUTED)
    return svg(w, h, body)


def fig7_paged_vs_dense():
    w, h, left, base, bw, gap = 860, 440, 130, 330, 130, 90
    groups = [("17-token prompt", 5.13, 5.13), ("44-token prompt", 7.00, 5.97)]
    scale = 30
    body = t(left, 42, "paged KV 与整块 KV 缓存的耗时对比", 20, INK, "bold")
    body += t(left, 64, "四个 staggered 请求的 wall time（秒），越低越好", 13, MUTED)
    for i, (label, dense, paged) in enumerate(groups):
        gx = left + i * (2 * bw + gap + 40)
        for j, (value, color, name) in enumerate(
            [(dense, GREY, "整块缓存"), (paged, GREEN, "paged KV")]
        ):
            x = gx + j * (bw + 12)
            bh = value * scale
            body += rect(x, base - bh, bw, bh, color)
            body += t(x + bw / 2, base - bh - 10, f"{value:.2f} s", 14, INK, "bold", "middle")
            body += t(x + bw / 2, base + 22, name, 12.5, MUTED, anchor="middle")
        body += t(gx + bw + 6, base + 52, label, 14, INK, "bold", "middle")
    body += f'<line x1="{left - 40}" y1="{base}" x2="{w - 40}" y2="{base}" stroke="{INK}" stroke-width="1"/>\n'
    body += t(left - 40, h - 34, "短上下文下两者无差别；44-token 时 paged 快 1.17×（5.97 s vs 7.00 s）", 12.5, ORANGE)
    body += t(left - 40, h - 14, "数据来源：docs/results/ecs_paged_dense_long.json / ecs_paged_paged_long.json", 11, MUTED)
    return svg(w, h, body)


def fig8_prefix_cache():
    w, h, left, base, bw = 860, 470, 140, 330, 90
    body = t(left, 42, "前缀块共享：4 个请求共享 89-token 前置说明", 20, INK, "bold")
    body += t(left, 64, "上：总耗时；下：每个请求的 prefill 耗时（秒）", 13, MUTED)
    for i, (value, color, name) in enumerate([(8.36, GREY, "无前缀缓存"), (6.15, GREEN, "有前缀缓存")]):
        x = left + i * (bw * 2 + 40)
        bh = value * 26
        body += rect(x, base - bh, bw, bh, color)
        body += t(x + bw / 2, base - bh - 10, f"{value:.2f} s", 14, INK, "bold", "middle")
        body += t(x + bw / 2, base + 22, name, 12.5, MUTED, anchor="middle")
    body += f'<line x1="{left - 40}" y1="{base}" x2="{w - 60}" y2="{base}" stroke="{INK}" stroke-width="1"/>\n'
    per = [(2.52, 1.17, 1.22, 1.21), (2.47, 0.58, 0.52, 0.54)]
    base2 = h - 40
    for i, values in enumerate(per):
        gy = base + 70 + i * 62
        body += t(left - 20, gy + 16, "无缓存" if i == 0 else "有缓存", 12, MUTED, anchor="end")
        for j, v in enumerate(values):
            x = left + j * 105
            bh = v * 16
            color = GREEN if (i == 1 and j > 0) else GREY
            body += rect(x, gy + 26 - bh, 88, bh, color)
            body += t(x + 44, gy + 22 - bh, f"{v:.2f}", 11, INK, anchor="middle")
    body += t(left - 40, h - 12, "命中后 3 个请求的 prefill 从 1.17 s 降到 0.52–0.58 s（−53%）；第 1 个请求仍需全价", 12, ORANGE)
    return svg(w, h, body)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    figs = {
        "fig4_single_gpu.svg": fig4_single_gpu(),
        "fig5_moe_ablation.svg": fig5_moe_ablation(),
        "fig6_batch_scaling.svg": fig6_batch_scaling(),
        "fig7_paged_vs_dense.svg": fig7_paged_vs_dense(),
        "fig8_prefix_cache.svg": fig8_prefix_cache(),
    }
    for name, content in figs.items():
        (OUT / name).write_text(content, encoding="utf-8")
        print(f"wrote {OUT / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
