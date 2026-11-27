# GPU runtime: the host-built backend binaries are compiled for exactly one
# vendor and NEEDED-link its GPU runtime at LOAD time (llama-server ->
# libhipblas.so.3 + libamdhip64.so.7, audiocpp_server additionally
# libhipblaslt.so.1), so those libs must be inside the image regardless of the
# gpu overlay. They are stamped from the HOST's ROCm install via the
# 'rocm-closure' build context (build/rocm-closure/, materialized by
# scripts/stamp-rocm-closure.sh — deploy.sh calls it before every build).
# The host build is custom (TensileLite hipBLASLt, librocm_kpack, compressed
# .dat.zlib kernel data under hipblaslt/library/gfx1201/) and shares only the
# version number with the stock rocm/dev stage — mixing the two aborts
# llama-server at the first GEMM ("Cannot read TensileLibrary_lazy_gfx1201.dat"
# -> CUBLAS_STATUS_INTERNAL_ERROR). Every GPU lib in the image must come from
# the same host build as the binaries. Re-stamp on ROCm upgrades. For an
# NVIDIA build, stage the closure from a CUDA runtime image instead
# (libcudart/libcublas/libcublasLt + device kernels).

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

# The ROCm runtime closure the stamped binaries load — flat libs plus the
# vendored $ORIGIN-relative dirs (rocm_sysdeps/, llvm/lib, host-math/lib)
# their RPATHs resolve. Staged from the HOST by scripts/stamp-rocm-closure.sh
# (see the header there for the custom-build vs stock-stage rationale).
COPY --from=rocm-closure ./                                     /opt/rocm/lib/
# The radiance CPU kernel plugins (share/radiance/kernels/libavx.so,
# libref.so) NEEDED the unversioned libomp.so and resolve it via ldconfig —
# which only scans the flat /opt/rocm/lib, not the llvm/lib subdir the HIP
# libs' RPATH uses. Keep a flat copy for them.
COPY --from=rocm-closure ./llvm/lib/libomp.so                   /opt/rocm/lib/
# The HIP libs carry $ORIGIN RUNPATHs, so they keep sitting in one flat dir
# (/opt/rocm/lib) next to their vendored subdirs. AMD's official builds link
# their sysdeps (elf/drm/numa/z/zstd) through rocm_sysdeps — staged above;
# the apt zlib1g/zstd1 packages stay as a fallback for system-linked libs.

# GEMM kernel data for gfx1200/gfx1201 (RDNA4): rocBLAS and hipBLASLt do NOT
# embed their kernels in the .so — they load TensileLibrary*.dat/.hsaco/.co
# from these libdir subdirs at first GEMM (the model warmup run). Without
# them llama-server dies during startup: "rocblaslt error: Cannot read
# TensileLibrary_lazy_gfx1201.dat" + "rocBLAS error: Cannot read .../library/
# TensileLibrary.dat" + "directory iterator cannot open directory
# [/opt/rocm/lib/rocblas/library]" -> exit -6 -> llama-swap 503. Stamped from
# the host's ROCm 7.2.3 install (same release as the .so closure above; the
# contexts point at the host dirs, see docker-compose.yml additional_contexts).
# rocblas/library carries the per-type fallback kernels + per-arch lazy index
# (~11 MB), hipblaslt/library the gfx1201 contraction kernels (~163 MB);
# rocisa is hipBLASLt's kernel-DSL runtime module. Re-stamp on ROCm upgrades,
# same as the host-built binaries.
COPY --from=rocm-rocblas-tensile   ./library /opt/rocm/lib/rocblas/library/
COPY --from=rocm-hipblaslt-tensile ./library /opt/rocm/lib/hipblaslt/library/
COPY --from=rocm-hipblaslt-tensile ./rocisa  /opt/rocm/lib/hipblaslt/rocisa/

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
