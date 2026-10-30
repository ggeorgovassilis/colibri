#!/usr/bin/env bash
# Build the colibri CUDA 11.8 toolchain image on rig1.
# Run once, or after editing rig1/Dockerfile. Needs internet for the
# nvidia/cuda base image + apt packages.
set -euo pipefail

IMG="colibri:2.0.0-cuda11.8"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> building ${IMG}"
docker build -t "${IMG}" -f "${DIR}/Dockerfile" "${DIR}"
echo "==> done: ${IMG}"
