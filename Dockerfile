# Vendor GPU runtime stage: the host-built backend binaries are compiled for
# exactly one vendor and NEEDED-link its GPU runtime at LOAD time (llama-server
# -> libhipblas.so.3 + libamdhip64.so.7, audiocpp_server additionally
# libhipblaslt.so.1), so those libs must be inside the image regardless of the
# gpu overlay. Pinned to the ROCm release the host binaries were built against
# (ROCm 7.2.3 — the rocm-smi driver version of this deployment). For an
# NVIDIA build, replace this stage with
#   FROM nvidia/cuda:<ver>-runtime-ubuntu24.04 AS cuda-libs
# and stamp the CUDA libs from it instead (libcudart/libcublas/libcublasLt).
ARG ROCM_VERSION=7.2.3
FROM rocm/dev-ubuntu-24.04:${ROCM_VERSION}-complete AS rocm-libs

# Base: the host-built backend binaries are compiled on the deployment host
# (Ubuntu 26.04, glibc 2.43) and refuse to load on older glibc, so the image
# bases on the same distro instead of python:slim (noble, glibc 2.39 —
# "version `GLIBC_2.43' not found"). Python runs in a venv at /opt/venv
# (first on PATH), which also keeps `pip install` PEP 668-clean.
FROM ubuntu:26.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LLAMASWAP_HOST=0.0.0.0 \
    LLAMASWAP_PORT=11434

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        libgomp1 \
        libstdc++6 \
        libssl3t64 \
        libelf1t64 \
        libdrm2 \
        libdrm-amdgpu1 \
        libnuma1 \
        zlib1g \
        libzstd1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

# The Dockerfile.audio overlay (docker-compose.audio.yml) stamps the
# audio.cpp / whisper.cpp binaries into this image; see README.

RUN mkdir -p /opt/rocm/lib /opt/llama.cpp/build/bin

# The ROCm runtime closure the stamped binaries load: the direct NEEDED libs
# (libamdhip64.so.7, libhipblas.so.3, libhipblaslt.so.1) plus their
# transitive deps from the pinned stage. AMD's official builds link their
# sysdeps (elf/drm/numa/z/zstd) straight against the system — the apt
# packages above provide those. The HIP libs carry $ORIGIN RUNPATHs, so they
# must keep sitting in one flat dir (/opt/rocm/lib). Verified: full GPU
# inference runs with ONLY these baked libs (the official libs embed their
# per-GPU kernels, no host mount needed) — the gpu-rocm overlay therefore
# exposes only the device nodes; bind-mounting the host's /opt/rocm would
# shadow these copies with dangling /etc/alternatives symlinks and break
# binaries whose RPATH lacks the host's versioned lib dir (radiance).
COPY --from=rocm-libs /opt/rocm/lib/libamdhip64.so.7          /opt/rocm/lib/
COPY --from=rocm-libs /opt/rocm/lib/libhsa-runtime64.so.1     /opt/rocm/lib/
COPY --from=rocm-libs /opt/rocm/lib/librocprofiler-register.so.0 /opt/rocm/lib/
COPY --from=rocm-libs /opt/rocm/lib/libamd_comgr.so.3         /opt/rocm/lib/
COPY --from=rocm-libs /opt/rocm/lib/librocroller.so.1         /opt/rocm/lib/
COPY --from=rocm-libs /opt/rocm/lib/libroctx64.so.4           /opt/rocm/lib/
COPY --from=rocm-libs /opt/rocm/lib/libhipblas.so.3           /opt/rocm/lib/
COPY --from=rocm-libs /opt/rocm/lib/libhipblaslt.so.1         /opt/rocm/lib/
COPY --from=rocm-libs /opt/rocm/lib/librocblas.so.5           /opt/rocm/lib/
COPY --from=rocm-libs /opt/rocm/lib/librocsolver.so.0         /opt/rocm/lib/
# LLVM OpenMP runtime: the radiance CPU kernel plugins
# (share/radiance/kernels/libavx.so, libref.so) NEEDED the unversioned
# libomp.so, which the pinned ROCm stage does not ship. Stamped from the
# host's ROCm LLVM dir (same 7.2.3 release the host binaries were built
# against) via the rocm-llvm-libs build context.
COPY --from=rocm-llvm-libs ./libomp.so                        /opt/rocm/lib/

RUN printf '/opt/rocm/lib\n' > /etc/ld.so.conf.d/rocm.conf && ldconfig

COPY --from=llamacpp-bin ./llama-server /opt/llama.cpp/build/bin/llama-server
# The Prism fork is optional: when the host has no llama_prism.cpp build,
# scripts/deploy.sh stamps the plain llama-server into the prism context
# instead, so this COPY always resolves and llama-prism-server exists
# (it then lacks the fork's low-precision quantization support).
COPY --from=llamacpp_prism-bin ./llama-server /opt/llama.cpp/build/bin/llama-prism-server

RUN chmod 0755 /opt/llama.cpp/build/bin/llama-server
RUN chmod 0755 /opt/llama.cpp/build/bin/llama-prism-server

# Radiance engine (managed backend for the qwen3.8-27b model): engine binary
# + runtime plugin tree + front launcher, stamped at /opt/radiance.
# bin/radiance, start.sh and qwen.yaml sit side by side; share/radiance
# carries the kernel plugins the engine loads at startup (kernels/ +
# architectures/ + quantizers/ + vectors/ + the precompiled .rkb blobs) —
# without it the engine stops with "plugin directory .../kernels: No such
# file or directory". start.sh is the front binary on purpose: the radiance
# CLI refuses unknown flags, so start.sh swallows the --host/--port the
# process manager appends and execs `radiance --config qwen.yaml` from its
# own dir. Inside the container start.sh's power-cap step (sudo amd-smi)
# degrades to a warning — the cap is applied on the host by the
# radiance-powercap.service systemd unit. qwen.yaml pins the weights at
# /opt/rad_models (bind-mounted by docker-compose.yml) and the entry port
# at 8000.
COPY --from=radiance-bin bin/radiance /opt/radiance/bin/
COPY --from=radiance-bin share/radiance /opt/radiance/share/radiance
COPY --from=radiance-src start.sh qwen.yaml /opt/radiance/
RUN chmod 0755 /opt/radiance/start.sh

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Entrypoint: sniffes deploy-time Strata pins off the mounted install when
# present (docker-compose.hub.yml); a no-op passthrough otherwise. CMD is
# unchanged — the script `exec`s it.
COPY scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod 0755 /usr/local/bin/docker-entrypoint.sh

COPY app ./app
COPY backend ./backend

# opencode integration: the provider fragment + local skills installed by
# scripts/install-opencode.py. deploy.sh uses the checkout on the host; the
# Hub image entrypoint uses this copy when an opencode config dir is mounted
# (docker-compose.hub.yml).
COPY opencode ./opencode
COPY scripts/install-opencode.py ./scripts/install-opencode.py
RUN chmod 0755 ./scripts/install-opencode.py

# The container listens on 11434 only; docker-compose.yml publishes it on
# the host as 11434 (default, LLAMASWAP_PORT-overridable), 9090 (legacy
# graphify port) and 1234 (llama.cpp default port). EXPOSE documents all
# three aliases for tooling/UX; they are the same listener.
EXPOSE 1234 11434 9090

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "11434"]
