#!/bin/sh
# after queue.sh: DeepInfer's control and pipeline at 16K and 64K, so every engine has the same lengths
cd /root/autodl-tmp/llamacpp-vs-deepinfer || exit 1
until grep -q 'queue done\|vLLM install failed' longctx/queue.log 2>/dev/null; do sleep 60; done
echo "$(date +%T) queue2: DeepInfer 16K / 64K" >> longctx/queue.log
sh run_deepinfer_suite.sh 16384 plain feedback >> longctx/deepinfer_small.runs 2>&1
sh run_deepinfer_suite.sh 65536 plain feedback >> longctx/deepinfer_small.runs 2>&1
echo "$(date +%T) queue2 done" >> longctx/queue.log
