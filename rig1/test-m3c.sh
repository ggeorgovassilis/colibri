#!/usr/bin/env bash
# M3c: run the 2-GPU concurrent pipeline overlap test on rig1.
# Builds and runs tests/test_pipeline_cuda_overlap inside the CUDA image:
#   stage 0 (layers [0,20))  -> device 0
#   stage 1 (layers [20,40)) -> device 1
#   two concurrent requests through the 2-stage ring, checked bit-identical to
#   the serial full-range CPU decode; prints "overlap peak 2" when the stages
#   overlap.
#
# Watch the GPUs in a second shell while it runs:
#   ssh rig1 'nvidia-smi dmon -s u -d 1'
set -euo pipefail

IMG="colibri:2.0.0-cuda11.8"
MODEL="${MODEL:-/models/colibri/qwen36_i4_gs64}"

docker run --rm --gpus all \
  -v "$HOME/colibri:/src" \
  -v "$HOME/models:/models" \
  -w /src/c \
  "${IMG}" \
  bash -lc "set -f; make pipeline-cuda-overlap CUDA=1 CUDA_ARCH=sm_60 QWEN_PIPELINE_MODEL=${MODEL}"
