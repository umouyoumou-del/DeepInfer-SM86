#!/bin/sh
# Can llama.cpp's server hold a context of N tokens for DeepSeek-Coder-V2-Lite Q4_K_M on this GPU?
# Each argument is "name llama-server-options..."; starts the server, waits for /health (ready) or
# exit, records VRAM and the KV / allocation log lines.
B=/root/autodl-tmp/llamacpp-vs-deepinfer/llama.cpp-ddf48f85c933c1f438212a02eb21e7b001d2d37d/build/bin
M=${MODEL:-/root/autodl-tmp/models/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf}
OUT=/root/autodl-tmp/llamacpp-vs-deepinfer/longctx
mkdir -p "$OUT"
probe() {
    name=$1; shift
    "$B/llama-server" -m "$M" --port 8090 -np 1 -ngl 99 "$@" > "$OUT/$name.log" 2>&1 &
    pid=$!
    state="timeout"
    for _ in $(seq 1 120); do
        if ! kill -0 "$pid" 2>/dev/null; then state="exited"; break; fi
        if curl -s -m 2 http://127.0.0.1:8090/health | grep -q '"ok"'; then state="READY"; break; fi
        sleep 5
    done
    echo "== $name ($*): $state, VRAM $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    grep -iE "n_ctx|kv|alloc|fail|error|overflow|out of memory|MiB" "$OUT/$name.log" \
        | grep -viE "^print_info|llama_model_loader|load_tensors:  *CPU" | tail -8
}
for spec in "$@"; do
    # shellcheck disable=SC2086
    probe $spec
done
