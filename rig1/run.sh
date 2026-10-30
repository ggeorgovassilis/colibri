#!/usr/bin/env bash
# Run colibri qwen36 in OpenAI-compatible server mode on rig1.
# Builds the engine inside the image first (no-op when it is up to date).
#
# The container is pinned to the name "colibri" so it can be managed with the
# usual docker verbs: `docker logs colibri`, `docker stop colibri`, etc. Re-running
# this script first removes any stale container under that name, then starts a
# fresh one. (A container started earlier under an auto-generated name still
# holds the published port -- stop it first, e.g. `docker rm -f <name>`.)
#
# Experiment defaults (2x Tesla P100 + qwen36_i4_gs64, native `coli serve`):
#   KV_SLOTS=4            concurrent KV slots -- matches the pipeline baseline
#   CUDA_EXPERT_GB=14     per-device expert budget: keep experts in VRAM
#   COLI_TRUNK_PROBE=0    skip the single-GEMV probe: keep the trunk in VRAM
#   COLI_TIMERS=1         print the per-phase timing breakdown to stderr
#
# By default EVERY security check / auth that colibri can disable is disabled,
# for a fully-open LAN endpoint:
#   1. no --api-key                -> no bearer/x-api-key auth
#   2. COLI_ALLOW_INSECURE_BIND=1  -> allow a non-loopback bind with no key
#   3. --allowed-host '*'          -> accept any Host header (no DNS-rebind guard)
#   4. --cors-origin '*'           -> allow any browser origin
#
# Overridable env vars:
#   NAME         container name (default "colibri")
#   PORT         host port to publish (default 8000)
#   MODEL        model dir inside the container (default /models/colibri/qwen36_i4_gs64)
#   GPU          device list for --gpu (default 0,1)
#   KV_SLOTS     concurrent request slots (default 4)
#   CAP          cache slots/layer (default: unset -> engine auto)
#   MAX_QUEUE    scheduler queue length (default 8)
#   QUEUE_TIMEOUT seconds a request may wait for a slot (default 300)
#   CUDA_EXPERT_GB  per-device expert budget in GB (default 14)
#   COLI_TRUNK_PROBE skip the trunk GEMV probe when 0 (default 0)
#   COLI_TIMERS  print the phase timing breakdown when 1 (default 1)
#   API_KEY      set -> require a Bearer/x-api-key token (re-enables auth)
#   HOSTS        comma-separated -> accept only those Host headers (default '*')
#   CORS         comma-separated -> accept only those origins (default '*')
set -euo pipefail

NAME="${NAME:-colibri}"
IMG="colibri:2.0.0-cuda11.8"
PORT="${PORT:-8000}"
MODEL="${MODEL:-/models/colibri/qwen36_i4_gs64}"
GPU="${GPU:-0,1}"
KV_SLOTS="${KV_SLOTS:-4}"
CAP="${CAP:-}"
MAX_QUEUE="${MAX_QUEUE:-8}"
QUEUE_TIMEOUT="${QUEUE_TIMEOUT:-300}"
CUDA_EXPERT_GB="${CUDA_EXPERT_GB:-14}"
COLI_TRUNK_PROBE="${COLI_TRUNK_PROBE:-0}"
COLI_TIMERS="${COLI_TIMERS:-1}"
HOSTS="${HOSTS:-*}"
CORS="${CORS:-*}"

SERVE="python3 ./coli serve --model ${MODEL} --gpu ${GPU} --host 0.0.0.0 --port 8000 \
  --kv-slots ${KV_SLOTS} --max-queue ${MAX_QUEUE} --queue-timeout ${QUEUE_TIMEOUT}"

# Cache slots/layer: only pass --cap when set, so the engine keeps its auto default.
[ -n "$CAP" ] && SERVE="${SERVE} --cap ${CAP}"

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

# CUDA tier placement. The startup GEMV probe times a SINGLE int8 GEMV, which
# is memory-latency-bound on a P100 and measures slower than the CPU; the
# auto-placer then withdraws the trunk AND the experts back to the CPU. Both
# are wrong for this workload (a batched topk-8 GEMM, not one GEMV), so the
# defaults above force trunk AND experts to stay on the device.
ENGINE_ENV="CUDA_EXPERT_GB=${CUDA_EXPERT_GB} COLI_TRUNK_PROBE=${COLI_TRUNK_PROBE} COLI_TIMERS=${COLI_TIMERS}"

# Pin the container name. Remove any stale container under that name first so
# re-runs are idempotent (`docker run --name` fails if the name is still taken).
docker rm -f "${NAME}" >/dev/null 2>&1 || true

docker run --rm --name "${NAME}" --gpus all -p "${PORT}:8000" \
  -v "$HOME/colibri:/src" \
  -v "$HOME/models:/models" \
  -w /src/c \
  "${IMG}" \
  bash -lc "set -f; make qwen36 CUDA=1 CUDA_ARCH=sm_60 && ${ENGINE_ENV} ${GUARD} ${SERVE}"
