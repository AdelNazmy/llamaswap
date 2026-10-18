#!/usr/bin/env sh
# Download the TTS/ASR model weights used by the llamaswap audio backends
# into the shared model root (/opt/models by default — the same directory
# docker-compose mounts into the container).
#
# Existing files are skipped ("download the *missing* models"), so this
# script is idempotent and safe to re-run.
#
# Usage:
#   ./scripts/download-models.sh [bundle ...]
#
# Bundles (no args = default set: tts asr whisper):
#   tts      OuteTTS 1.0 1B Q8_0 GGUF  (audio.cpp -> /opt/audio.cpp
#            family `outetts`, used by backend/outetts-tts.yaml)
#   tts-qwen3 Qwen3-TTS 12Hz 0.6B Q8_0 (audio.cpp family `qwen3_tts`,
#            alternative TTS backend)
#   asr      Qwen3-ASR 0.6B Q8_0 GGUF  (audio.cpp family `qwen3_asr`,
#            used by backend/qwen3-asr.yaml)
#   whisper  ggml-base.en (whisper.cpp, used by backend/whisper-asr.yaml)
#   whisper-multi  ggml-base (multilingual whisper.cpp; enables real
#            translation for /v1/audio/translations — point whisper-asr.yaml's
#            -m at ggml-base.bin to use it)
#   flux-schnell FLUX.1-schnell q8_0 GGUF  (stable-diffusion.cpp,
#            used by backend/flux-schnell.yaml)
#   flux-vae  FLUX VAE ae.safetensors     (shared FLUX auxiliary)
#   flux-clip_l FLUX clip_l text encoder  (shared FLUX auxiliary)
#   flux-t5xxl FLUX t5xxl text encoder    (shared FLUX auxiliary, ~10 GB)
#   qwen-image-2.1-uc  Qwen-Image 2.1 Uncensored diffusion Q4_K_M GGUF
#            (stable-diffusion.cpp, used by backend/qwen-image-2.1-uc.yaml)
#   qwen-image-2.1-te  Qwen3-VL-8B int8 ConvRot text encoder (~9.35 GB)
#   qwen-image-2.1-vae Qwen Image 2.1 VAE bf16
#   music3-config   MiniMax Music 3 package metadata (config.json)
#   music3-cfg-*    MiniMax Music 3 per-component config JSONs
#   music3-tok-*    MiniMax Music 3 tokenizer files
#   music3-lm       MiniMax Music 3 Qwen3-8B language model q4_0 (~6 GB)
#   music3-depth    MiniMax Music 3 RVQ depth decoder q8_0
#   music3-cond     MiniMax Music 3 condition encoder
#   music3-tr       MiniMax Music 3 flow transformer q4_0
#   music3-vocoder  MiniMax Music 3 vocoder
#            (all from hf.co/audio-cpp/MiniMax-Music3-GGUF, used by
#            backend/minimax-music3.yaml)
#
# Env:
#   MODEL_ROOT  model directory (default: /opt/models)
#   NO_RESUME   set to 1 to restart a partial download instead of resuming
#
# Requires: aria2 (the aria2c binary). (No huggingface_hub / hf CLI needed.)

set -u

MODEL_ROOT="${MODEL_ROOT:-.}"

# bundle -> "url|relative/path"
BUNDLES='
tts|https://huggingface.co/mirek190/audio.cpp/resolve/main/Text%20to%20audio%20(TTS)/Llama-OuteTTS-1.0-1B_Q8.gguf|Llama-OuteTTS-1.0-1B_Q8/Text to audio (TTS)/Llama-OuteTTS-1.0-1B_Q8.gguf
tts-qwen3|https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main/Qwen3-TTS-12Hz-0.6B-Base-GGUF/qwen3-tts-12hz-0.6b-base-q8_0.gguf|Qwen3-TTS-12Hz-0.6B-Base-GGUF/qwen3-tts-12hz-0.6b-base-q8_0.gguf
asr|https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main/Qwen3-ASR-0.6B-GGUF/qwen3-asr-0.6b-q8_0.gguf|Qwen3-ASR-0.6B-GGUF/qwen3-asr-0.6b-q8_0.gguf
whisper|https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin|ggml-base.en.bin
whisper-multi|https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin|ggml-base.bin
nemotron_asr|https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main/Nemotron-3.5-ASR-Streaming-0.6B-GGUF/nemotron-3.5-asr-streaming-0.6b-q8_0.gguf|Nemotron-3.5-ASR-Streaming-0.6B-GGUF/nemotron-3.5-asr-streaming-0.6b-q8_0.gguf
supertonic_3|https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main/Supertonic-3-GGUF/supertonic-3-q8_0.gguf|Supertonic-3-GGUF/supertonic-3-q8_0.gguf
flux-schnell|https://huggingface.co/leejet/FLUX.1-schnell-gguf/resolve/main/flux1-schnell-q8_0.gguf|FLUX.1-schnell-gguf/flux1-schnell-q8_0.gguf
flux-vae|https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors|FLUX.1-common/ae.safetensors
flux-clip_l|https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors|FLUX.1-common/clip_l.safetensors
flux-t5xxl|https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors|FLUX.1-common/t5xxl_fp16.safetensors
qwen-image-2.1-uc|https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-Quantized/resolve/main/qwen-image-2.1-UC-Q4_K_M.gguf|Qwen-Image-2.1-Uncensored-Quantized/qwen-image-2.1-UC-Q4_K_M.gguf
qwen-image-2.1-te|https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-Quantized/resolve/main/text_encoders/qwen3vl_8b_int8_convrot.safetensors|Qwen-Image-2.1-Uncensored-Quantized/text_encoders/qwen3vl_8b_int8_convrot.safetensors
qwen-image-2.1-vae|https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-Quantized/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors|Qwen-Image-2.1-Uncensored-Quantized/vae/qwen_image_2.1_vae_bf16.safetensors
music3-config|https://huggingface.co/audio-cpp/MiniMax-Music3-GGUF/resolve/main/config.json|MiniMax-Music3-GGUF/config.json
music3-cfg-lm|https://huggingface.co/audio-cpp/MiniMax-Music3-GGUF/resolve/main/config/language_model.json|MiniMax-Music3-GGUF/config/language_model.json
music3-cfg-depth|https://huggingface.co/audio-cpp/MiniMax-Music3-GGUF/resolve/main/config/rvq_depth_decoder.json|MiniMax-Music3-GGUF/config/rvq_depth_decoder.json
music3-cfg-cond|https://huggingface.co/audio-cpp/MiniMax-Music3-GGUF/resolve/main/config/condition_encoder.json|MiniMax-Music3-GGUF/config/condition_encoder.json
music3-cfg-tr|https://huggingface.co/audio-cpp/MiniMax-Music3-GGUF/resolve/main/config/transformer.json|MiniMax-Music3-GGUF/config/transformer.json
music3-cfg-voc|https://huggingface.co/audio-cpp/MiniMax-Music3-GGUF/resolve/main/config/vocoder.json|MiniMax-Music3-GGUF/config/vocoder.json
music3-tok|https://huggingface.co/audio-cpp/MiniMax-Music3-GGUF/resolve/main/tokenizer/tokenizer.json|MiniMax-Music3-GGUF/tokenizer/tokenizer.json
music3-tok-cfg|https://huggingface.co/audio-cpp/MiniMax-Music3-GGUF/resolve/main/tokenizer/tokenizer_config.json|MiniMax-Music3-GGUF/tokenizer/tokenizer_config.json
music3-lm|https://huggingface.co/audio-cpp/MiniMax-Music3-GGUF/resolve/main/language_model_q4_0.gguf|MiniMax-Music3-GGUF/language_model_q4_0.gguf
music3-depth|https://huggingface.co/audio-cpp/MiniMax-Music3-GGUF/resolve/main/rvq_depth_decoder_q8_0.gguf|MiniMax-Music3-GGUF/rvq_depth_decoder_q8_0.gguf
music3-cond|https://huggingface.co/audio-cpp/MiniMax-Music3-GGUF/resolve/main/condition_encoder.gguf|MiniMax-Music3-GGUF/condition_encoder.gguf
music3-tr|https://huggingface.co/audio-cpp/MiniMax-Music3-GGUF/resolve/main/transformer_q4_0.gguf|MiniMax-Music3-GGUF/transformer_q4_0.gguf
music3-vocoder|https://huggingface.co/audio-cpp/MiniMax-Music3-GGUF/resolve/main/vocoder.gguf|MiniMax-Music3-GGUF/vocoder.gguf
'

resolve() { # name -> url|relpath
    printf '%s\n' "$BUNDLES" | awk -F'|' -v n="$1" '$1 == n { print $2 "|" $3; exit; }'
}

# is_present <relpath> -> 0 if the file exists and is non-empty
is_present() {
    [ -f "$MODEL_ROOT/$1" ] && [ -s "$MODEL_ROOT/$1" ]
}

# list_missing <bundle...> -> print each bundle whose file is missing.
# Exits 0 so callers can capture the list via command substitution.
list_missing() {
    for name in "$@"; do
        spec=$(resolve "$name")
        if [ -z "$spec" ]; then
            echo "  [error] unknown bundle '$name'" >&2
            continue
        fi
        rel="${spec#*|}"
        if ! is_present "$rel"; then
            printf '%s\n' "$name"
        fi
    done
    return 0
}

download_one() {
    url="$1"; rel="$2"
    dest="$MODEL_ROOT/$rel"
    if [ -f "$dest" ] && [ -s "$dest" ]; then
        echo "  [skip] $rel (already present)"
        return
    fi
    mkdir -p "$(dirname "$dest")"
    echo "  [get ] $rel"
    aria2c -x 16 -s 16 -m 3 --retry-wait=5 \
    ${NO_RESUME:+-c} \
    -o "$dest" "$url" \
        || { echo "  [error] failed downloading $rel" >&2; rm -f "$dest"; return 1; }
    echo "  [done] $(du -h "$dest" | cut -f1) -> $dest"
}

main() {
    missing_only=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --missing) missing_only=1; shift ;;
            --) shift; break ;;
            -*) echo "download-models: unknown option '$1'" >&2; exit 2 ;;
            *) break ;;
        esac
    done

    if [ $# -eq 0 ]; then
        set -- tts asr whisper whisper-multi tts-qwen3 nemotron_asr supertonic_3 \
            music3-config music3-cfg-lm music3-cfg-depth music3-cfg-cond \
            music3-cfg-tr music3-cfg-voc music3-tok music3-tok-cfg \
            music3-lm music3-depth music3-cond music3-tr music3-vocoder
    fi

    if [ "$missing_only" = 1 ]; then
        list_missing "$@"
        return 0
    fi

    echo "Model root: $MODEL_ROOT"
    rc=0
    for name in "$@"; do
        spec=$(resolve "$name")
        if [ -z "$spec" ]; then
            echo "  [error] unknown bundle '$name' (known: tts tts-qwen3 asr whisper whisper-multi nemotron_asr supertonic_3 flux-schnell flux-vae flux-clip_l flux-t5xxl qwen-image-2.1-uc qwen-image-2.1-te qwen-image-2.1-vae music3-config music3-cfg-lm music3-cfg-depth music3-cfg-cond music3-cfg-tr music3-cfg-voc music3-tok music3-tok-cfg music3-lm music3-depth music3-cond music3-tr music3-vocoder)" >&2
            rc=1
            continue
        fi
        url="${spec%%|*}"; rel="${spec#*|}"
        download_one "$url" "$rel" || rc=1
    done
    if [ "$rc" -eq 0 ]; then
        echo "All requested models are in place under $MODEL_ROOT"
    else
        echo "One or more downloads failed." >&2
    fi
    exit "$rc"
}

main "$@"
