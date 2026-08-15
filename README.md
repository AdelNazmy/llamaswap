# llamaswap

An OpenAI-compatible proxy in front of `llama-server` (llama.cpp) with a
YAML-driven model registry. Point any OpenAI client at it and switch chat
models per request — the backend chat `llama-server` is stopped and
relaunched transparently with the launch command of the requested model.

A dedicated embedding `llama-server` can also run persistently on its own
port, so `/v1/embeddings` does not disturb the currently loaded chat model.

## How it works

```
OpenAI client (port 11434)
        │  /v1/chat/completions, /v1/embeddings, /v1/models
        ▼
llamaswap (FastAPI)
  • model registry        ← backend/*.yaml
  • LLM process manager   ← start / stop / swap chat llama-server
  • embedding manager     ← persistent embedding llama-server
        │  HTTP (streaming pass-through)
        ├── llama-server chat  (127.0.0.1:8101, one model at a time)
        └── llama-server embed (127.0.0.1:8102, persistent)
```

- **One chat LLM at a time.** A 13 GB model does not fit twice on a 16 GB
  GPU, so when a request names a different chat model than the one currently
  loaded, the proxy gracefully stops the running chat `llama-server`, spawns
  a new one from that model's YAML config, and waits for its `/health`
  endpoint before forwarding the request.
- **Persistent embeddings.** If the registry contains a model with
  `role: embedding`, llamaswap launches it at startup and keeps it running
  for the lifetime of the proxy. `/v1/embeddings` goes directly to that
  server, leaving the loaded chat model untouched.
- **Resource fallback.** If a chat model fails to load while the embedding
  server is running, llamaswap stops the embedding server, retries the chat
  load once, and then best-effort relaunches embeddings in the background.
- **Registry.** Every file in `backend/` is one model: the exact
  `llama-server` binary + arguments + env + port. Add a file, reload, done.
- **OpenAI compatible.** llama-server already speaks the OpenAI protocol; the
  proxy passes it through (SSE streams included) and rewrites response
  `model` fields to the model named in the request.

## Layout

```
llamaswap/
├── backend/                  # model registry — one YAML per model
│   ├── octen-embedding.yaml  # persistent embedding server
│   ├── qwen3.8-27b.yaml
│   ├── qwen3.6-35b.yaml
│   └── qwen3.6-35b-vision.yaml
├── app/
│   ├── config.py             # settings (env prefix LLAMASWAP_)
│   ├── registry.py           # YAML → validated ModelConfig objects
│   ├── process_manager.py    # chat llama-server lifecycle + health checks
│   ├── embedding_manager.py  # persistent embedding llama-server lifecycle
│   ├── proxy.py              # async reverse proxy (stream / non-stream)
│   └── main.py               # FastAPI routes (OpenAI API)
├── requirements.txt
├── docs/images/              # screenshots referenced by this README
└── README.md
```

## Model config format

```yaml
name: qwen3.8-27b                    # model id used in API requests
description: "human readable"
role: llm                            # llm (default) or embedding
command:
  binary: /opt/llama.cpp/build/bin/llama-server
  args:
    - "--model"
    - "/opt/models/Qwen3.8-27B-UD-Q3_K_XL.gguf"
    - "--ctx-size"
    - "8192"
    - "--n-gpu-layers"
    - "99"
    - "--jinja"
  env:                               # optional, merged over os.environ
    CUDA_VISIBLE_DEVICES: "0"
host: 127.0.0.1                      # internal binding (enforced)
port: 8101                           # internal port (enforced)
meta:
  context_length: 8192
  family: qwen
  capabilities: [chat]
```

`role: llm` models are managed by the normal swap path. A single
`role: embedding` model is managed by the persistent embedding path and
should use its own port, usually `8102`, plus `--embedding` in `args`.

`host`/`port` are always enforced by the proxy (any `--host`/`--port` in
`args` is stripped) so models never fight over the internal port.

## Getting and compiling llama.cpp

llamaswap does not bundle llama.cpp — each model YAML in `backend/` points at
a `llama-server` binary you compile (or install) yourself. The default
configs expect it at `/opt/llama.cpp/build/bin/llama-server`.

Prerequisites for both backends: git, cmake (≥ 3.13) and a C++17 compiler
(`g++`/`clang++`).

### CUDA (NVIDIA)

Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
(`nvcc`) and your GPU driver, then:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build --config Release -j
```

### ROCm (AMD)

Install [ROCm](https://rocm.docs.amd.com/) (`hipcc`) and your GPU driver,
then:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_HIPBLAS=ON
cmake --build build --config Release -j
```

Notes:

- The binary ends up at `build/bin/llama-server`; either run llama.cpp from
  its clone location or move it and update `binary:` in the model YAMLs.
- Verify the GPU backend before pointing llamaswap at it — the server logs
  `CUDA is initialized` (or the ROCm equivalent) on startup. The proxy
  mirrors those lines into its own log, so a missing/wrong backend shows up
  there as `llama-server[<model>]: ...`.
- The Docker image in this repo ships a prebuilt CUDA `llama-server`
  instead, so the steps above are only for building it locally.

## Run

```bash
cd ~/llamaswap
~/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 11434
```

Or with an explicit python:

```bash
~/.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 11434
```

Environment overrides (prefix `LLAMASWAP_`): `LLAMASWAP_PORT`,
`LLAMASWAP_BACKEND_DIR`, `LLAMASWAP_STARTUP_TIMEOUT`,
`LLAMASWAP_STOP_TIMEOUT`, `LLAMASWAP_LOG_LEVEL`.

## API

| Endpoint | Description |
|---|---|
| `GET /v1/models` | List models from the registry (OpenAI shape) |
| `GET /v1/models/{model}` | One model entry |
| `POST /v1/chat/completions` | Chat; `stream: true` for SSE |
| `POST /v1/completions` | Text completions |
| `POST /v1/embeddings` | Embeddings; uses the persistent embedding server if configured, otherwise the normal model-swap path |
| `POST /v1/registry/reload` | Re-read `backend/*.yaml` (admin) |
| `GET /health` | Proxy, current chat llama-server, and persistent embedding status |

### Examples

```bash
# list models
curl -s localhost:11434/v1/models | jq

# chat (non-streaming) — first call loads the model (can take minutes)
curl -s localhost:11434/v1/chat/completions -d '{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": "Say hello in one word."}],
  "max_tokens": 32
}'

# streaming
curl -N localhost:11434/v1/chat/completions -d '{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": "Count to five."}],
  "stream": true
}'

# switch model — the proxy swaps the backend for you
curl -s localhost:11434/v1/chat/completions -d '{
  "model": "qwen3.6-35b",
  "messages": [{"role": "user", "content": "Who are you?"}],
  "max_tokens": 64
}'

# vision
curl -s localhost:11434/v1/chat/completions -d '{
  "model": "qwen3.6-35b-vision",
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "What do you see?"},
    {"type": "image_url", "image_url": {"url": "http://.../cat.jpg"}}
  ]}]
}'

# embeddings — served by the persistent embedding server
curl -s localhost:11434/v1/embeddings -d '{
  "model": "octen-embedding-0.6b",
  "input": "hello world"
}'
```

OpenAI SDK:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:11434/v1", api_key="unused")
resp = client.chat.completions.create(
    model="qwen3.8-27b",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)

emb = client.embeddings.create(
    model="octen-embedding-0.6b",
    input="hello world",
)
print(emb.data[0].embedding[:8])
```

## Notes

- The persistent embedding server starts with the proxy. If it fails to
  start, the proxy still starts and reports the embedding state in
  `/health`.
- Chat model first request loads it (13 GB → VRAM); expect 1–3 minutes.
  Subsequent requests for the *same* model are instant.
- Switching back and forth re-loads the model each time (that is the
  price of one-model-at-a-time on 16 GB VRAM).
- If a chat model cannot load while embeddings are running, the embedding
  server is stopped and the chat load is retried once. If the retry
  succeeds, embeddings are relaunched in the background; if it fails, the
  error is returned and embeddings remain stopped until a later successful
  chat load or an embedding request restarts them.
- Model load failures are surfaced as `503` with the last llama-server
  log lines in the server console; the proxy stays up.
- `llama-server` stderr is mirrored to the proxy log as
  `llama-server[<model>]: ...`; embedding server lines are tagged
  `llama-server[embed <model>]: ...`.

## Utilization and logs
  ![llama-server starting with the CUDA backend](docs/images/llama-server-cuda.png)
