#!/bin/sh
# llama.cpp's best case: the same weights in its MLA layout (/dev/shm, scripts/gguf_mla_layout.py).
# Speed of each FA / cache-type setting, then how long a context fits on the GPU.
cd /root/autodl-tmp/llamacpp-vs-deepinfer || exit 1
B=llama.cpp-ddf48f85c933c1f438212a02eb21e7b001d2d37d/build/bin
export MODEL=/dev/shm/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M-mla.gguf
while pgrep -f 'needle_comple[x].py' > /dev/null; do sleep 10; done
echo "== speed (MLA layout)"
for cfg in "-fa 1 -ctk f16" "-fa 1 -ctk q8_0" "-fa 0 -ctk f16"; do
    # shellcheck disable=SC2086
    timeout 300 "$B/llama-bench" -m "$MODEL" -ngl 99 $cfg -p 2048 -n 0 -d 0,16384 -r 1 -o md 2>/dev/null | grep pp2048
    echo "[$cfg] exit $?"
done
echo "== capacity (MLA layout, FlashAttention)"
sh llama_capacity.sh "mla_3m_f16 -fa on -c 3145728" "mla_3m_q8 -fa on -ctk q8_0 -c 3145728" \
    "mla_1m_q8 -fa on -ctk q8_0 -c 1050624" "mla_1m_f16 -fa on -c 1050624" "mla_512k_f16 -fa on -c 526336" \
    2>&1 | grep -E '^==|alloc|buffer size'
