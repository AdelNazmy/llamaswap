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
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /usr/local/cuda/lib64 /opt/llama.cpp/build/bin

COPY --from=cuda-libs /usr/local/cuda/lib64/libcudart.so* /usr/local/cuda/lib64/
COPY --from=cuda-libs /usr/local/cuda/lib64/libcublas.so* /usr/local/cuda/lib64/
COPY --from=cuda-libs /usr/local/cuda/lib64/libcublasLt.so* /usr/local/cuda/lib64/

RUN printf '/usr/local/cuda/lib64\n' > /etc/ld.so.conf.d/cuda.conf && ldconfig

COPY --from=llamacpp-bin ./llama-server /opt/llama.cpp/build/bin/llama-server
RUN chmod 0755 /opt/llama.cpp/build/bin/llama-server

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY backend ./backend

EXPOSE 11434

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "11434"]
