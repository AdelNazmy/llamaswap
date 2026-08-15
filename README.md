# custom_ollama

An OpenAI-compatible proxy in front of `llama-server` (llama.cpp) with a
YAML-driven model registry. Point any OpenAI client at it and switch models
per request — the backend `llama-server` is stopped and relaunched
transparently with the launch command of the requested model.

## How it works

```
OpenAI client (port 11434)
        │  /v1/chat/completions, /v1/embeddings, /v1/models
        ▼
custom_ollama (FastAPI)
  • model registry        ← backend/*.yaml
  • process manager       ← start / stop / swap llama-server
        │  HTTP (streaming pass-through)
        ▼
llama-server (127.0.0.1:8101, one model at a time)
```

- **One model at a time.** A 13 GB model does not fit twice on a 16 GB GPU,
  so when a request names a different model than the one currently loaded,
  the proxy gracefully stops the running `llama-server`, spawns a new one
  from that model's YAML config, and waits for its `/health` endpoint before
  forwarding the request.
- **Registry.** Every file in `backend/` is one model: the exact
  `llama-server` binary + arguments + env + port. Add a file, reload, done.
- **OpenAI compatible.** llama-server already speaks the OpenAI protocol; the
  proxy passes it through (SSE streams included) and rewrites the `model`
  field to the registry name.

## Layout

```
custom_ollama/
├── backend/                  # model registry — one YAML per model
│   ├── qwen3.8-27b.yaml
│   ├── qwen3.6-35b.yaml
│   └── qwen3.6-35b-vision.yaml
├── app/
│   ├── config.py             # settings (env prefix CUSTOM_OLLAMA_)
│   ├── registry.py           # YAML → validated ModelConfig objects
│   ├── process_manager.py    # subprocess lifecycle + health checks
│   ├── proxy.py              # async reverse proxy (stream / non-stream)
│   └── main.py               # FastAPI routes (OpenAI API)
├── requirements.txt
└── README.md
```

## Model config format

```yaml
name: qwen3.8-27b                    # model id used in API requests
description: "human readable"
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

`host`/`port` are always enforced by the proxy (any `--host`/`--port` in
`args` is stripped) so models never fight over the internal port.

## Run

```bash
cd ~/custom_ollama
~/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 11434
```

Or with an explicit python:

```bash
~/.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 11434
```

Environment overrides (prefix `CUSTOM_OLLAMA_`): `CUSTOM_OLLAMA_PORT`,
`CUSTOM_OLLAMA_BACKEND_DIR`, `CUSTOM_OLLAMA_STARTUP_TIMEOUT`,
`CUSTOM_OLLAMA_STOP_TIMEOUT`, `CUSTOM_OLLAMA_LOG_LEVEL`.

## API

| Endpoint | Description |
|---|---|
| `GET /v1/models` | List models from the registry (OpenAI shape) |
| `GET /v1/models/{model}` | One model entry |
| `POST /v1/chat/completions` | Chat; `stream: true` for SSE |
| `POST /v1/completions` | Text completions |
| `POST /v1/embeddings` | Embeddings (needs an embedding model in the registry) |
| `POST /v1/registry/reload` | Re-read `backend/*.yaml` (admin) |
| `GET /health` | Proxy + current llama-server status |

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
```

## Notes

- First request for a model loads it (13 GB → VRAM); expect 1–3 minutes.
  Subsequent requests for the *same* model are instant.
- Switching back and forth re-loads the model each time (that is the
  price of one-model-at-a-time on 16 GB VRAM).
- Model load failures are surfaced as `503` with the last llama-server
  log lines in the server console; the proxy stays up.
- `llama-server` stderr is mirrored to the proxy log as
  `llama-server[<model>]: ...`.
