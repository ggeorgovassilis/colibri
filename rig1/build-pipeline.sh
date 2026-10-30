#!/usr/bin/env bash
# Build the 2-GPU pipeline server (pipeline_serve_cuda) on rig1.
#
# pipeline_serve.c is a standalone server that runs the Qwen3.6 two-stage layer
# pipeline -- edge -> stage0 [0,split) -> stage1 [split,n) -> edge -- over two
# worker threads, and speaks the same serve_codec.h wire protocol as qwen36.
# It is NOT wired into `coli serve` (which picks the per-family qwen36 binary),
# so it is launched by openai_server.py with --engine ./pipeline_serve_cuda.
set -euo pipefail

IMG="colibri:2.0.0-cuda11.8"

docker run --rm --gpus all \
  -v "$HOME/colibri:/src" \
  -v "$HOME/models:/models" \
  -w /src/c \
  "${IMG}" \
  bash -lc "set -f; make pipeline_serve_cuda CUDA=1 CUDA_ARCH=sm_60"
