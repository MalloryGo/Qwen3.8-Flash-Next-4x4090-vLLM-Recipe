#!/usr/bin/env bash
set -euo pipefail

secret_file=/run/secrets/vllm-api-key
[[ -r "$secret_file" ]] || { echo "missing API key secret file" >&2; exit 2; }
api_key="$(cat "$secret_file")"
[[ -n "$api_key" ]] || { echo "API key is empty" >&2; exit 2; }

exec vllm serve /model \
  --host "${AB_BIND_HOST:-127.0.0.1}" \
  --port "${VLLM_PORT:-8000}" \
  --api-key "$api_key" \
  --served-model-name qwen3.8-flash \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 2 \
  --enable-expert-parallel \
  --max-model-len 262144 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.94 \
  --max-num-batched-tokens 1024 \
  --kv-cache-dtype fp8_e4m3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --engram-config '{"cpu_offload":true}' \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --disable-custom-all-reduce
