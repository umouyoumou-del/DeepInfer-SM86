#!/bin/sh
# The 39-needle suite through DeepInfer at TOKENS, in each MODE (plain = full-attention control,
# feedback = locate + reread), with the 3M run's settings.  Waits for the GPU to be free first.
# Usage: run_deepinfer_suite.sh TOKENS MODE...
set -u
TOKENS=$1; shift
OUT=/root/autodl-tmp/llamacpp-vs-deepinfer/longctx
cd /root/di-bench/DeepInfer-SM86-main || exit 1
while pgrep -f 'build/bin/llama-serve[r]' > /dev/null; do sleep 10; done
for MODE in "$@"; do
    NAME=deepinfer_$((TOKENS / 1024))k_$MODE
    /root/miniconda3/bin/python scripts/needle_complex.py --tokens "$TOKENS" --near-window 98304 --mode "$MODE" \
        --out "$OUT/$NAME.json" > "$OUT/$NAME.out" 2>&1
    echo "$NAME exit $?"
done
