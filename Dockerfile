FROM nvidia/cuda:13.3.1-runtime-ubuntu24.04 AS cuda-libs

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LLAMASWAP_HOST=0.0.0.0 \
    LLAMASWAP_PORT=11434

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        libstdc++6 \
        libssl3 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# The Dockerfile.audio overlay (docker-compose.audio.yml) stamps the
# audio.cpp / whisper.cpp binaries into this image; see README.

RUN mkdir -p /usr/local/cuda/lib64 /opt/llama.cpp/build/bin

COPY --from=cuda-libs /usr/local/cuda/lib64/libcudart.so* /usr/local/cuda/lib64/
COPY --from=cuda-libs /usr/local/cuda/lib64/libcublas.so* /usr/local/cuda/lib64/
COPY --from=cuda-libs /usr/local/cuda/lib64/libcublasLt.so* /usr/local/cuda/lib64/

RUN printf '/usr/local/cuda/lib64\n' > /etc/ld.so.conf.d/cuda.conf && ldconfig

COPY --from=llamacpp-bin ./llama-server /opt/llama.cpp/build/bin/llama-server
COPY --from=llamacpp_prism-bin ./llama-server /opt/llama.cpp/build/bin/llama-prism-server

RUN chmod 0755 /opt/llama.cpp/build/bin/llama-server
RUN chmod 0755 /opt/llama.cpp/build/bin/llama-prism-server

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Entrypoint: sniffes deploy-time Strata pins off the mounted install when
# present (docker-compose.hub.yml); a no-op passthrough otherwise. CMD is
# unchanged — the script `exec`s it.
COPY scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod 0755 /usr/local/bin/docker-entrypoint.sh

COPY app ./app
COPY backend ./backend

# The container listens on 11434 only; docker-compose.yml publishes it on
# the host as 11434 (default, LLAMASWAP_PORT-overridable), 9090 (legacy
# graphify port) and 1234 (llama.cpp default port). EXPOSE documents all
# three aliases for tooling/UX; they are the same listener.
EXPOSE 1234 11434 9090

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "11434"]
