I got a dense 27B model doing 90–122 tokens/sec on a single 16 GB GPU.

Not by buying more VRAM — by making the model I run actually fit inside my orchestration.

I run a local LLM stack on one 16 GB card (WSL2), fronted by llamaswap — an OpenAI-compatible proxy that swaps models in and out of VRAM on demand. Chat LLMs, embeddings, TTS, ASR, and image models share the same card without fighting for memory.

llama.cpp models were always first-class there: load on request, unload on idle, swap transparently.

EXL3 quantized models — the best quality-per-bit quantization format available right now — were the exception. They required a separate, always-on inference server that permanently held VRAM and sat outside the orchestration.

So I fixed that:

→ A minimal OpenAI-compatible server around exllamav3's AsyncGenerator — the same engine the big local-LLM servers run — registered in llamaswap exactly like a llama.cpp model.
→ Same swap semantics: SIGTERM = unload, VRAM freed, next model takes the card.
→ Tuned for the card: 65k context, Q4 KV cache, speculative decoding via the checkpoint's built-in multi-token-prediction head (no draft model needed), and system-RAM overflow tiers.

Same 27B model, two quantization stacks:

EXL3 3.5bpw (exllamav3): 90–122 tok/s generation (avg ~105), 750–1,400 tok/s prefill
Q3_K_XL (llama.cpp): ~35 tok/s generation

Prefill is a wash between the two stacks. Generation is where exllamav3's kernels pull away — 90+ tok/s even at a 63k-token context. Long documents get read at the same speed; the answer just gets written 3× faster.

Why EXL3 wins at the same bit budget:

→ Trellis coding packs 3.5 effective bits per weight with less quality loss than a classic Q3 K-quant
→ The checkpoint's own MTP head drafts tokens — no separate draft model or draft KV cache
→ Q4 KV cache at 65k context: half the cache memory of Q8, doubling usable context on the same card
→ Dense 27B fully resident on 16 GB — weights, KV, and MTP head — with system-RAM tiering absorbing overflow

The hard part wasn't the server. It was the fit. At 65k context the card was within ~50 MiB of the wire. Three levers made it fit: the engine's load reserve (500 MB → 96 MB), the autosplit safety margin (256 MB → 128 MB), and halving the prefill chunk to shrink transient workspace.

Two bonus wins:

→ The exl3 backend emits llama.cpp's exact slot timing log format — uniform tokens/sec across every backend
→ The embedding server dropped to lazy loading: the whole stack now boots using 9 MiB of GPU

Everything is OpenAI-API-compatible end to end. One endpoint, any client, zero VRAM waste.

If you're running a local stack on a single card — what's your biggest VRAM headache: always-on models, context bloat, or something else?

#LocalLLM #LLM #AIInfrastructure #SelfHosted #OpenSource