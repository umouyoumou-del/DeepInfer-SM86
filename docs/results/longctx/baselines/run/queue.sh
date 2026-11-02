#!/bin/sh
# After llama.cpp's 1M run: DeepInfer's pipeline at 1M, then vLLM (once installed and downloaded):
# capacity at 3M and at the model's 160K, the suite at 98K and at 1M.
cd /root/autodl-tmp/llamacpp-vs-deepinfer || exit 1
L=longctx/queue.log
while pgrep -f 'run_llama_suit[e]' > /dev/null; do sleep 30; done
echo "$(date +%T) llama.cpp 1M done; DeepInfer 1M" >> $L
sh run_deepinfer_suite.sh 1048576 feedback >> longctx/deepinfer_1m.runs 2>&1
echo "$(date +%T) DeepInfer 1M done; waiting for vLLM install + AWQ download" >> $L
until grep -qE '^exit [0-9]+' longctx/vllm073_install.log 2>/dev/null; do sleep 30; done
grep -q '^exit 0' longctx/vllm073_install.log || { echo "$(date +%T) vLLM install failed" >> $L; exit 1; }
until grep -q 'awq download done' longctx/awq_download.log 2>/dev/null; do sleep 30; done
echo "$(date +%T) vLLM runs" >> $L
sh run_vllm.sh vllm_cap_3m 3145728 0 >> longctx/vllm.out 2>&1
sh run_vllm.sh vllm_cap_160k 163840 0 >> longctx/vllm.out 2>&1
sh run_vllm.sh vllm_98k 100352 98304 --enable-prefix-caching >> longctx/vllm.out 2>&1
sh run_vllm.sh vllm_1m 1050624 1048576 --enable-prefix-caching >> longctx/vllm.out 2>&1
echo "$(date +%T) queue done" >> $L
