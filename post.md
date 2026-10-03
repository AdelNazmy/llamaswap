# EXL3 quantized models just became first-class citizens in my local LLM stack

I run a local LLM stack on a single 16 GB GPU (WSL2), fronted by **llamaswap** — an OpenAI-compatible proxy I built that swaps models in and out of VRAM on demand, so chat LLMs, embedding, TTS, ASR, and image models never fight for memory. llama.cpp models have always been first-class there: load on request, unload on idle, swap transparently.

But EXL3 quantized models — the best quality-per-bit quantization format available right now — required a separate, always-on inference server that permanently held VRAM and sat outside that orchestration.

So I fixed that: a minimal OpenAI-compatible server (~300 lines) around exllamav3's AsyncGenerator, registered in llamaswap exactly like a llama.cpp model.

## How it works

- Streaming chat completions via exllamav3's async generator — the same engine the big local-LLM servers use, minus the always-on footprint
- Swap semantics identical to llama.cpp: SIGTERM = unload, VRAM freed, next model takes the card
- Tuned for the card: 65k context, Q4 KV cache, speculative decoding via the checkpoint's built-in multi-token-prediction head (no draft model needed), system-RAM overflow tiers

## Benchmark — same 27B model, two quantization stacks

| | EXL3 3.5bpw (exllamav3) | Q3_K_XL (llama.cpp) |
|---|---|---|
| Generation | **90–122 tok/s** (avg 105) | ~35 tok/s |
| Prefill | 750–1,400 tok/s | 900–1,400 tok/s |
| Context window | 65k, Q4 KV cache | — |

## Why EXL3 wins at the same bit budget

- **~3× generation speed** — prefill is a wash between the two stacks; generation is where exllamav3's kernels pull away, holding 90+ tok/s even at a 63k-token context. Long documents get read at the same speed — EXL3 writes the answer 3× faster
- **More quality per bit** — EXL3's trellis coding packs 3.5 effective bpw with less quality loss than a classic Q3 K-quant
- **Free speculative decoding** — the checkpoint's own MTP head drafts tokens, no separate draft model or draft KV cache
- **Q4 KV cache at 65k context** — half the cache memory of Q8, doubling usable context on the same card
- **Dense 27B fully resident on 16 GB** — weights, KV, and MTP head, with system-RAM tiering absorbing overflow

## Bonus touches

- The exl3 backend emits llama.cpp's exact `slot print_timing` log format — prefill and generation tokens/sec for every request, uniform across all backends
- Embedding server dropped to lazy loading: the stack now boots using 9 MiB of GPU

The lesson from the trenches: the card was within ~50 MiB of the wire at 65k context. Three levers made it fit — the engine's load reserve (500 MB → 96 MB), the autosplit safety margin (256 MB → 128 MB), and halving the prefill chunk to shrink transient workspace.

Everything is OpenAI-API-compatible end to end — one endpoint, any client, zero VRAM waste.

#LocalLLM #llama #exllamav3 #Quantization #AIInfrastructure #SelfHosted #OpenSource #LLM #MLOps
