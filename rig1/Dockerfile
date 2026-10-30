# colibri CUDA 11.8 toolchain image — used to compile and run the engine on
# rig1 (2x Tesla P100, sm_60). This reproduces the colibri:2.0.0-cuda11.8
# image: NVIDIA's CUDA 11.8 dev base + the build deps the engine needs.
#
# The colibri source is NOT baked in — it is mounted at /src at run time so
# the engine can be rebuilt in place (see run.sh).

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ARG COLIBRI_VERSION=2.0.0
LABEL org.opencontainers.image.title="colibri CUDA toolchain"
LABEL org.opencontainers.image.version=${COLIBRI_VERSION}

# Force IPv4 for apt (some mirrors are IPv6-flaky behind NAT).
RUN echo 'Acquire::ForceIPv4 "true";' > /etc/apt/apt.conf.d/99force-ipv4

# build-essential = gcc/make for the engine; git + python3 = the `coli`
# launcher; numactl = NUMA topology for layer/expert placement.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential git python3 numactl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src
