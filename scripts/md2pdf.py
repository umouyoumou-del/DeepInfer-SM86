"""Render a markdown document to PDF using headless Chrome (no pandoc/markdown)."""

from __future__ import annotations

import html as H
import re
import subprocess
import sys
from pathlib import Path

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: 'Microsoft YaHei','PingFang SC',sans-serif; font-size: 10.5pt;
       line-height: 1.75; color: #1f2328; max-width: 100%; }
h1 { font-size: 19pt; border-bottom: 2.5px solid #1f2328; padding-bottom: 8px; margin: 0 0 18px; }
h2 { font-size: 14pt; margin: 26px 0 10px; border-left: 4px solid #2f6feb; padding-left: 10px; }
h3 { font-size: 12pt; margin: 18px 0 8px; color: #2f6feb; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 9.5pt; }
th, td { border: 1px solid #c9ccd1; padding: 5px 8px; text-align: left; }
th { background: #f2f4f7; font-weight: 600; }
td:has(+ td), th:has(+ td) { }
pre { background: #f6f8fa; border: 1px solid #d8dee4; border-radius: 4px;
      padding: 10px; font-size: 9pt; overflow-x: auto; white-space: pre-wrap; }
code { font-family: Consolas, monospace; background: #f2f4f7; padding: 1px 4px;
       border-radius: 3px; font-size: 9pt; }
pre code { background: none; padding: 0; }
blockquote { margin: 12px 0; padding: 8px 14px; border-left: 4px solid #d0d7de;
             background: #f8f9fb; color: #57606a; }
img { max-width: 100%; display: block; margin: 14px auto; }
hr { border: none; border-top: 1px solid #d8dee4; margin: 20px 0; }
ul, ol { padding-left: 24px; }
li { margin: 4px 0; }
"""


def inline(text: str) -> str:
    text = H.escape(text)
    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r'<img alt="\1" src="\2">', text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!`)`([^`]+)`(?!`)", r"<code>\1</code>", text)
    return text


def convert(md: str) -> str:
    lines = md.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            i += 1
            buf = []
            while i < len(lines) and not lines[i].startswith("```"):
                buf.append(H.escape(lines[i]))
                i += 1
            i += 1
            out.append("<pre><code>" + "\n".join(buf) + "</code></pre>")
            continue
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            level = len(m.group(1))
            out.append(f"<h{level}>{inline(m.group(2))}</h{level}>")
            i += 1
            continue
        if line.strip().startswith("|") and i + 1 < len(lines) and re.match(
            r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]
        ):
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            i += 2
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            out.append("<table><thead><tr>")
            out.extend(f"<th>{inline(c)}</th>" for c in header)
            out.append("</tr></thead><tbody>")
            for row in rows:
                out.append("<tr>")
                out.extend(f"<td>{inline(c)}</td>" for c in row)
                out.append("</tr>")
            out.append("</tbody></table>")
            continue
        if line.strip() in {"---", "***", "___"}:
            out.append("<hr>")
            i += 1
            continue
        if line.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].startswith(">"):
                buf.append(inline(lines[i].lstrip("> ").rstrip()))
                i += 1
            out.append("<blockquote>" + "<br>".join(buf) + "</blockquote>")
            continue
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)", line)
        if m:
            ordered = m.group(2)[0].isdigit()
            tag = "ol" if ordered else "ul"
            items = []
            while i < len(lines):
                mm = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)", lines[i])
                if not mm:
                    if lines[i].startswith("  ") and items:
                        items[-1] += "<br>" + inline(lines[i].strip())
                        i += 1
                        continue
                    break
                items.append(inline(mm.group(3)))
                i += 1
            out.append(f"<{tag}>" + "".join(f"<li>{x}</li>" for x in items) + f"</{tag}>")
            continue
        if not line.strip():
            i += 1
            continue
        buf = []
        while i < len(lines) and lines[i].strip() and not re.match(
            r"^(#{1,4}\s|```|\||>|\s*([-*]|\d+\.)\s|---)", lines[i]
        ):
            buf.append(inline(lines[i].strip()))
            i += 1
        if buf:
            out.append("<p>" + "<br>".join(buf) + "</p>")
    return "\n".join(out)


def main() -> int:
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    body = convert(src.read_text(encoding="utf-8"))
    page = (
        '<!doctype html><html><head><meta charset="utf-8">'
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )
    tmp = dst.with_suffix(".html")
    tmp.write_text(page, encoding="utf-8")
    subprocess.run(
        [
            CHROME, "--headless=new", "--disable-gpu", "--no-sandbox",
            f"--user-data-dir={dst.parent / ('.chromepdf_' + dst.stem)}",
            "--no-pdf-header-footer",
            f"--print-to-pdf={dst.resolve()}",
            tmp.resolve().as_uri(),
        ],
        check=False,
        timeout=300,
        capture_output=True,
    )
    size = dst.stat().st_size if dst.exists() else 0
    print(f"{dst} -> {size} bytes")
    return 0 if size > 10000 else 1


if __name__ == "__main__":
    raise SystemExit(main())
