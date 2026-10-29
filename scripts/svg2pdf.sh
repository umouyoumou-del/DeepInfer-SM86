#!/bin/bash
cd /d/DeepInfer-SM86/docs/figures || exit 1
CHROME="/c/Program Files/Google/Chrome/Application/chrome.exe"
for f in fig*.svg; do
  name="${f%.svg}"
  w=$(grep -o 'width="[0-9]*"' "$f" | head -1 | tr -dc '0-9')
  h=$(grep -o 'height="[0-9]*"' "$f" | head -1 | tr -dc '0-9')
  {
    echo '<!doctype html><html><head><meta charset="utf-8"><style>'
    echo "@page { size: ${w}px ${h}px; margin: 0; } html,body{margin:0;padding:0} svg{display:block}"
    echo '</style></head><body>'
    cat "$f"
    echo '</body></html>'
  } > "${name}_print.html"
  timeout 90 "$CHROME" --headless=new --disable-gpu --no-sandbox --disable-dev-shm-usage \
    --user-data-dir="/tmp/chromeprof_$name" --no-pdf-header-footer \
    --print-to-pdf="$(pwd)/${name}.pdf" "file:///D:/DeepInfer-SM86/docs/figures/${name}_print.html" \
    >/dev/null 2>&1
  sz=$(stat -c %s "${name}.pdf" 2>/dev/null || echo 0)
  echo "$name -> ${sz} bytes"
done
echo ALL_DONE
