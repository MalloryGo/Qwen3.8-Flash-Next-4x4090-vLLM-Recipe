#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_DIR:?set MODEL_DIR to the separately downloaded Merlin checkpoint}"
: "${API_KEY_FILE:?set API_KEY_FILE to a readable 0600 secret file}"

CONFIG_DIR="${CONFIG_DIR:-$PWD/configs}"
RUNTIME_IMAGE="${RUNTIME_IMAGE:-qwen38-merlin-vllm:0.30.0}"
GPU_ORDER="${GPU_ORDER:-0,1,2,3}"

[[ -f "$MODEL_DIR/model.safetensors.index.json" ]]
[[ -f "$MODEL_DIR/qsa_kv_scales_262k.json" ]]
[[ -r "$API_KEY_FILE" && "$(stat -c '%a' "$API_KEY_FILE")" == 600 ]]
[[ -f "$CONFIG_DIR/entrypoint-prefix-off.sh" ]]

exec docker run -d --name qwen38-prefix-off \
  --gpus all --ipc=host --network=host \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_VISIBLE_DEVICES="$GPU_ORDER" \
  -e NCCL_P2P_DISABLE=1 \
  -e VLLM_PP_LAYER_PARTITION=25,23 \
  -e VLLM_ENABLE_RESPONSES_API_STORE=1 \
  -e VLLM_QSA_KV_SCALES=/model/qsa_kv_scales_262k.json \
  -e AB_BIND_HOST="${AB_BIND_HOST:-127.0.0.1}" \
  -e VLLM_PORT="${VLLM_PORT:-8000}" \
  --mount "type=bind,src=$MODEL_DIR,dst=/model,readonly" \
  --mount "type=bind,src=$CONFIG_DIR,dst=/recipe-config,readonly" \
  --mount "type=bind,src=$API_KEY_FILE,dst=/run/secrets/vllm-api-key,readonly" \
  --entrypoint /bin/bash "$RUNTIME_IMAGE" /recipe-config/entrypoint-prefix-off.sh