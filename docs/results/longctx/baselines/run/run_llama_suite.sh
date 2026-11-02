#!/bin/sh
# The 39-needle suite through llama.cpp's server: start it with the given context options, wait,
# run scripts/longctx_baseline.py, stop it.  Usage: run_llama_suite.sh NAME TOKENS server-options...
set -u
NAME=$1; TOKENS=$2; shift 2
B=/root/autodl-tmp/llamacpp-vs-deepinfer/llama.cpp-ddf48f85c933c1f438212a02eb21e7b001d2d37d/build/bin
M=${MODEL:-/root/autodl-tmp/models/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf}
OUT=/root/autodl-tmp/llamacpp-vs-deepinfer/longctx
SRC=/root/di-bench/DeepInfer-SM86-main
mkdir -p "$OUT"
"$B/llama-server" -m "$M" --port 8090 -np 1 -ngl 99 "$@" > "$OUT/$NAME.server.log" 2>&1 &
pid=$!
for _ in $(seq 1 120); do
    curl -s -m 2 http://127.0.0.1:8090/health | grep -q '"ok"' && break
    kill -0 "$pid" 2>/dev/null || { echo "server exited"; tail -5 "$OUT/$NAME.server.log"; exit 1; }
    sleep 5
done
echo "server ready: $* (VRAM $(nvidia-smi --query-gpu=memory.used --format=csv,noheader))"
cd "$SRC" && /root/miniconda3/bin/python scripts/longctx_baseline.py --engine llama.cpp --url http://127.0.0.1:8090 \
    --tokens "$TOKENS" --label "llama.cpp b12024 (ddf48f8), $(basename "$M"), $*" --out "$OUT/$NAME.json"
kill "$pid"; wait "$pid" 2>/dev/null
echo "done: $OUT/$NAME.json"
