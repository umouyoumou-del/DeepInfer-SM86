#!/bin/sh
# vLLM with the AWQ checkpoint of the same model: start the OpenAI server at MAX_LEN, wait for it
# (or for it to fail), and run the suite at TOKENS when TOKENS > 0.
# Usage: run_vllm.sh NAME MAX_LEN TOKENS [vllm options...]
set -u
NAME=$1; MAX_LEN=$2; TOKENS=$3; shift 3
OUT=/root/autodl-tmp/llamacpp-vs-deepinfer/longctx
PY=/root/vllm073-venv/bin/python
while pgrep -f 'build/bin/llama-serve[r]|needle_comple[x].py' > /dev/null; do sleep 15; done
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 "$PY" -m vllm.entrypoints.openai.api_server --model /dev/shm/awq \
    --served-model-name coder --trust-remote-code --port 8000 --max-num-seqs 1 \
    --gpu-memory-utilization 0.95 --max-model-len "$MAX_LEN" "$@" > "$OUT/$NAME.server.log" 2>&1 &
pid=$!
state=timeout
for _ in $(seq 1 180); do
    if ! kill -0 "$pid" 2>/dev/null; then state=exited; break; fi
    if curl -s -m 2 http://127.0.0.1:8000/health > /dev/null 2>&1 && curl -s -m 2 http://127.0.0.1:8000/v1/models | grep -q coder; then
        state=READY; break
    fi
    sleep 5
done
echo "== $NAME max_model_len $MAX_LEN $*: $state, VRAM $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
grep -iE "KV cache|cache blocks|GPU blocks|maximum concurrency|max seq len|larger than|Error|error" "$OUT/$NAME.server.log" | grep -v Warning | tail -8 | cut -c1-300
if [ "$state" = READY ] && [ "$TOKENS" -gt 0 ]; then
    cd /root/di-bench/DeepInfer-SM86-main && /root/miniconda3/bin/python scripts/longctx_baseline.py --engine vllm \
        --url http://127.0.0.1:8000 --model coder --tokens "$TOKENS" \
        --label "vLLM 0.7.3, TechxGenus/DeepSeek-Coder-V2-Lite-Instruct-AWQ, max_model_len $MAX_LEN $*" --out "$OUT/$NAME.json"
fi
kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
sleep 5
