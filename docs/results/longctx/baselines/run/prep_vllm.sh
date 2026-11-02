#!/bin/sh
# Scratch for the vLLM comparison, all in RAM-backed /dev/shm (removed afterwards):
# the AWQ checkpoint of the same model, and the latest vLLM in its own venv.
LOG=/root/autodl-tmp/llamacpp-vs-deepinfer/longctx
HF_ENDPOINT=https://hf-mirror.com /root/miniconda3/bin/python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('TechxGenus/DeepSeek-Coder-V2-Lite-Instruct-AWQ', local_dir='/dev/shm/awq'))
" > "$LOG/awq_download.log" 2>&1 &
/root/miniconda3/bin/python -m venv /dev/shm/vllm-venv \
    && /dev/shm/vllm-venv/bin/pip install --no-cache-dir -q --upgrade pip \
    && /dev/shm/vllm-venv/bin/pip install --no-cache-dir vllm==0.32.0 > "$LOG/vllm_install.log" 2>&1
echo "vllm install exit $?" >> "$LOG/vllm_install.log"
wait
echo "awq download done" >> "$LOG/awq_download.log"
