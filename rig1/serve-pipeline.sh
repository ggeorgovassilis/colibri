#!/usr/bin/env bash
# Run the 2-GPU pipeline server (pipeline_serve_cuda) behind the OpenAI gateway
# on rig1. Builds the binary inside the image first (no-op when up to date).
#
# The two stages overlap: while stage1 decodes token t, stage0 is already
# decoding token t+1. That overlap needs >=2 concurrent requests, so KV_SLOTS
# defaults to 4. (--kv-slots 1 degrades the pipeline to serial.)
#
# Tier contract (matches the pipeline's layer ranges):
#   COLI_GPUS=0,1       devices available to the CUDA tier
#   QT_HOME=layer       home each stage's layer tier to a device
#   QT_LAYER_SPLIT=20   stage0 owns [0,20), stage1 owns [20,40) -- must equal
#                       num_layers/2 (40 layers in qwen36_i4_gs64)
#
# Security defaults mirror rig1/run.sh (fully-open LAN endpoint). Re-enable
# specific controls with:
#   PORT      host port to publish (default 8000)
#   API_KEY   set -> require a Bearer/x-api-key token
#   HOSTS     comma-separated -> accept only those Host headers (default '*')
#   CORS      comma-separated -> accept only those origins (default '*')
#   SPLIT     QT_LAYER_SPLIT (default 20)
#   KV_SLOTS  concurrent request slots (default 4)
set -euo pipefail

IMG="colibri:2.0.0-cuda11.8"
PORT="${PORT:-8000}"
MODEL="${MODEL:-/models/colibri/qwen36_i4_gs64}"
SPLIT="${SPLIT:-20}"
KV_SLOTS="${KV_SLOTS:-4}"
HOSTS="${HOSTS:-*}"
CORS="${CORS:-*}"

TIER_ENV="COLI_GPUS=0,1 QT_HOME=layer QT_LAYER_SPLIT=${SPLIT}"

# `coli serve` has no --engine flag (it resolves the per-family qwen36 binary),
# so launch openai_server.py directly -- exactly what `coli serve` does
# internally, but pointing --engine at the pipeline server.
SERVE="python3 openai_server.py --model ${MODEL} --engine ./pipeline_serve_cuda --host 0.0.0.0 --port 8000 --kv-slots ${KV_SLOTS}"

# Accept the Host header(s) the clients will send (--allowed-host may repeat).
# Iterate via a quoted array so a literal '*' is not glob-expanded.
IFS=',' read -ra HOST_ARR <<< "$HOSTS"
for h in "${HOST_ARR[@]}"; do
  [ -n "$h" ] && SERVE="${SERVE} --allowed-host ${h}"
done

# Allow the browser origin(s) (--cors-origin may repeat).
IFS=',' read -ra CORS_ARR <<< "$CORS"
for c in "${CORS_ARR[@]}"; do
  [ -n "$c" ] && SERVE="${SERVE} --cors-origin ${c}"
done

if [ -n "${API_KEY:-}" ]; then
  SERVE="${SERVE} --api-key ${API_KEY}"
  GUARD=""
else
  GUARD="COLI_ALLOW_INSECURE_BIND=1"
fi

docker run --rm --gpus all -p "${PORT}:8000" \
  -v "$HOME/colibri:/src" \
  -v "$HOME/models:/models" \
  -w /src/c \
  "${IMG}" \
  bash -lc "set -f; make pipeline_serve_cuda CUDA=1 CUDA_ARCH=sm_60 && ${GUARD} ${TIER_ENV} ${SERVE}"
