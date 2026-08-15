A 27B dense model. 16GB of VRAM. 65,000-token context. All on a single consumer GPU.
I've been running Qwen3.8-27B on an RTX 4070 Ti (16GB) and the intelligence-to-watts ratio is absurd for a card from a few years ago. It codes, it reasons, it holds a 65k-token conversation without breaking a sweat.
The current llama.cpp config:
• Q3_K_XL quant — the sweet spot for 27B on 16GB
• 65k context, 100% GPU offload (999 layers)
• Q8_0 KV cache + flash attention
• Speculative decoding (n-gram modified)
• 1,024-token reasoning budget
• mlock + mmap so the model never pokes the page cache
The honest part: juggling multiple local models on one 16GB GPU is painful. Ollama is a great starting point, but when your models don't all fit in VRAM and you need exact launch flags per model, you quickly hit its ceiling.
So I built llamaswap — an OpenAI-compatible proxy in front of llama-server with a YAML-driven model registry.
What it does:
1. One model = one YAML file. Exact llama-server binary, args, env, port. Drop a file in the registry, hot-reload, done. No Modelfiles, no black box.
2. Per-request model swapping. Ask for "qwen3.8-27b" then "qwen3.6-35b" on consecutive requests — the proxy gracefully stops the running chat server, launches the requested model from its config, waits for a health check, then forwards your request. Streaming (SSE) passes through untouched.
3. Persistent embedding server. A small 0.6B embedding model runs on its own port for the lifetime of the proxy, so RAG and vector search never evict your chat model from VRAM.
4. Graceful fallback. If a chat model can't load while embeddings are running, llamaswap stops the embedding server, retries the chat load, and best-effort relaunches embeddings in the background. The proxy never dies.
5. One endpoint to rule them all. Point any OpenAI SDK at port 11434 — /v1/chat/completions, /v1/embeddings, /v1/models. Response model fields are rewritten so clients always see the model they asked for.
Right now the registry is running three models on one 4070 Ti: Qwen3.8-27B (dense, Q3_K_XL, 65k ctx), Qwen3.6-35B MoE (IQ3_XXS, 200k ctx), and the embedding model. One card, one port, any model per request.
The local LLM story is no longer "pick one model and pray." It's "ship a registry, swap per request, and let the hardware do what it does best."
Curious what others are running on 16GB cards in 2026 — I'd love to compare configs.
#LocalLLM #llama.cpp #Qwen #OpenSource #AIInfrastructure #GPU #16GBVRAM #DevTools
Notes:
- Kept your naming (Qwen3.8-27B) and the real config flags from backend/qwen3.8-27b.yaml:1-42.
- If the card is a 4070 Ti Super (16GB) vs base (12GB), tweak line 4 accordingly — the 65k ctx strongly suggests the Super.
- No emojis per your style; add a few (🔥 💾 ⚡) if you want more LinkedIn punch — just say the word.
