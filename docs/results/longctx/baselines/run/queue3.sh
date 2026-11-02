#!/bin/sh
# After queue2: vLLM again, configured so long prompts are not dropped. vLLM 0.7.3 turns chunked prefill on
# for max_model_len > 32K with a 2048-token batch, then MLA turns chunked prefill off but keeps the 2048
# batch, so every prompt over 2048 tokens was rejected ("Input prompt ... exceeds limit of 2048").
# With --max-num-batched-tokens >= the prompt the whole prompt is prefilled in one batch.
cd /root/autodl-tmp/llamacpp-vs-deepinfer || exit 1
L=longctx/queue.log
until grep -q 'queue2 done' $L 2>/dev/null; do sleep 30; done
for f in vllm_98k.json vllm_98k.server.log; do
    [ -f "longctx/$f" ] && mv "longctx/$f" "longctx/$(echo $f | sed 's/vllm_98k/vllm_98k_rejected/')"
done
echo "$(date +%T) queue3: vLLM with --max-num-batched-tokens" >> $L
sh run_vllm.sh vllm_98k 100352 98304 --max-num-batched-tokens 100352 >> longctx/vllm.out 2>&1
sh run_vllm.sh vllm_cap_256k_batch 262144 0 --max-num-batched-tokens 262144 >> longctx/vllm.out 2>&1
sh run_vllm.sh vllm_cap_512k_batch 524288 0 --max-num-batched-tokens 524288 >> longctx/vllm.out 2>&1
echo "$(date +%T) queue3 done" >> $L
