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
#
# Env:
#   MODEL_ROOT  model directory (default: /opt/models)
#   NO_RESUME   set to 1 to restart a partial download instead of resuming
#
# Requires: curl. (No huggingface_hub / hf CLI needed.)

set -u

MODEL_ROOT="${MODEL_ROOT:-.}"

# bundle -> "url|relative/path"
BUNDLES='
tts|https://huggingface.co/mirek190/audio.cpp/resolve/main/Text%20to%20audio%20(TTS)/Llama-OuteTTS-1.0-1B_Q8.gguf|Llama-OuteTTS-1.0-1B_Q8/Text to audio (TTS)/Llama-OuteTTS-1.0-1B_Q8.gguf
tts-qwen3|https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main/Qwen3-TTS-12Hz-0.6B-Base-GGUF/qwen3-tts-12hz-0.6b-base-q8_0.gguf|Qwen3-TTS-12Hz-0.6B-Base-GGUF/qwen3-tts-12hz-0.6b-base-q8_0.gguf
asr|https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main/Qwen3-ASR-0.6B-GGUF/qwen3-asr-0.6b-q8_0.gguf|Qwen3-ASR-0.6B-GGUF/qwen3-asr-0.6b-q8_0.gguf
whisper|https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin|ggml-base.en.bin
whisper-multi|https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin|ggml-base.bin
nemotron_asr|https://huggingface.co/audio-cpp/audio.cpp-gguf/blob/main/Nemotron-3.5-ASR-Streaming-0.6B-GGUF/nemotron-3.5-asr-streaming-0.6b-q8_0.gguf|Nemotron-3.5-ASR-Streaming-0.6B-GGUF/nemotron-3.5-asr-streaming-0.6b-q8_0.gguf
supertonic_3|https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main/Supertonic-3-GGUF/supertonic-3-q8_0.gguf|Supertonic-3-GGUF/supertonic-3-q8_0.gguf
'

resolve() { # name -> url|relpath
    printf '%s\n' "$BUNDLES" | awk -F'|' -v n="$1" '$1 == n { print $2 "|" $3; exit; }'
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
    if [ $# -eq 0 ]; then
        set -- tts asr whisper whisper-multi tts-qwen3 nemotron_asr supertonic_3
    fi
    echo "Model root: $MODEL_ROOT"
    rc=0
    for name in "$@"; do
        spec=$(resolve "$name")
        if [ -z "$spec" ]; then
            echo "  [error] unknown bundle '$name' (known: tts tts-qwen3 asr whisper whisper-multi)" >&2
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
