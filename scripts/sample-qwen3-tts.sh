#!/usr/bin/env sh
# sample-qwen3-tts.sh — generate a speech sample from the Qwen3-TTS
# voice-clone backend through the llamaswap OpenAI audio API.
#
# Qwen3-TTS 12Hz 0.6B *Base* is a voice-clone model: it has no preset
# voices and synthesizes from a short reference clip (``voice_ref``) plus its
# exact transcript (``reference_text`` — required in ICL clone mode). llamaswap already
# wires a reference voice into backend/qwen3-tts.yaml
# (``meta.reference_voice`` + ``meta.reference_text`` → data/script_scarlette_ref.wav),
# so the plain OpenAI ``input`` field is enough to sample that voice.
#
# Two modes:
#   * default           — send only ``input``; the server injects the
#                         configured reference (script_scarlette).
#   * --reference-audio — clone a DIFFERENT voice from a local WAV passed
#                         in (inline base64), with its transcript in
#                         --reference-text.
#
# Requires: curl, jq (and base64 when --reference-audio is used).

set -eu

BASE_URL="${LLAMASWAP_BASE_URL:-http://localhost:11434/v1}"
MODEL="qwen3-tts"
OUT="data/qwen3-tts-sample.wav"
RESPONSE_FORMAT="wav"
TEXT=""
REF_AUDIO=""
REF_TEXT=""

usage() {
    cat <<'USAGE'
Usage: sample-qwen3-tts.sh [options] [TEXT]

Generate a speech sample from the Qwen3-TTS voice-clone backend through the
llamaswap OpenAI audio API (/v1/audio/speech).

Options:
  -t, --text <text>             Text to synthesize (positional TEXT also works).
  -a, --reference-audio <wav>   Clone this local WAV instead of the voice wired
                                into backend/qwen3-tts.yaml (inline base64;
                                <= 5 MiB, aim for 3-10 s of clean speech).
  -T, --reference-text <text>   Exact transcript of --reference-audio
                                (required with -a; ICL clone mode throws
                                without it).
  -o, --out <path>              Output path (default: data/qwen3-tts-sample.wav).
  -m, --model <id>              Model id (default: qwen3-tts).
  -b, --base-url <url>          llamaswap base URL (default:
                                http://localhost:11434/v1).
  -f, --response-format <fmt>   wav|mp3|opus|aac|flac|pcm (default: wav).
  -h, --help                    Show this help.

Examples:
  # sample the voice already configured in backend/qwen3-tts.yaml
  ./scripts/sample-qwen3-tts.sh "Hello from the cloned voice"

  # clone a different local reference clip and sample it
  ./scripts/sample-qwen3-tts.sh \
      --reference-audio data/my_voice_ref.wav \
      --reference-text "The exact words spoken in that clip." \
      --out /tmp/my_clone.wav \
      "Say this in the new voice."
USAGE
}

# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------
while [ "$#" -gt 0 ]; do
    case "$1" in
        -h|--help)           usage; exit 0 ;;
        -t|--text)           TEXT="${2:-}"; shift 2 ;;
        -a|--reference-audio) REF_AUDIO="${2:-}"; shift 2 ;;
        -T|--reference-text) REF_TEXT="${2:-}"; shift 2 ;;
        -o|--out)            OUT="${2:-}"; shift 2 ;;
        -m|--model)          MODEL="${2:-}"; shift 2 ;;
        -b|--base-url)       BASE_URL="${2:-}"; shift 2 ;;
        -f|--response-format) RESPONSE_FORMAT="${2:-}"; shift 2 ;;
        --) shift; [ "$#" -gt 0 ] && TEXT="$*"; break ;;
        -*) echo "error: unknown option: $1" >&2; usage >&2; exit 2 ;;
        *)  if [ -z "$TEXT" ]; then TEXT="$1"; else TEXT="$TEXT $1"; fi; shift ;;
    esac
done

if [ -z "$TEXT" ]; then
    echo "error: no text to synthesize (pass TEXT or -t/--text)" >&2
    usage >&2
    exit 2
fi
if [ -z "$REF_AUDIO" ] && [ -n "$REF_TEXT" ]; then
    echo "error: --reference-text requires --reference-audio" >&2
    exit 2
fi
if [ -n "$REF_AUDIO" ] && [ ! -f "$REF_AUDIO" ]; then
    echo "error: reference audio not found: $REF_AUDIO" >&2
    exit 2
fi

command -v curl >/dev/null 2>&1 || { echo "error: curl is required" >&2; exit 2; }
command -v jq   >/dev/null 2>&1 || { echo "error: jq is required" >&2; exit 2; }

# ---------------------------------------------------------------------------
# build the request body
# ---------------------------------------------------------------------------
# The JSON is written to a temp file and curl reads it with --data-binary @file
# (never through argv). This matters for --reference-audio, where the inline
# base64 can push the body well past Linux's 128 KiB single-argument limit.
# jq only escapes the (small) text fields; the reference WAV is base64-encoded
# with the `base64` binary (byte-exact for binary) into a temp file, then read
# by jq as raw ASCII over stdin.
PAYLOAD_FILE=$(mktemp)
REF_B64_FILE=""
cleanup() {
    rm -f "$PAYLOAD_FILE"
    [ -n "$REF_B64_FILE" ] && rm -f "$REF_B64_FILE"
    return 0  # keep a failed [ -n ] above from flipping the exit status
}
trap cleanup EXIT

if [ -n "$REF_AUDIO" ]; then
    command -v base64 >/dev/null 2>&1 \
        || { echo "error: base64 is required with --reference-audio" >&2; exit 2; }
    REF_B64_FILE=$(mktemp)
    # base64 output wrapping differs across GNU/macOS; strip line breaks.
    base64 < "$REF_AUDIO" | tr -d '\n' > "$REF_B64_FILE"
    jq -n -R \
        --arg model "$MODEL" \
        --arg input "$TEXT" \
        --arg fmt "$RESPONSE_FORMAT" \
        --arg reftext "$REF_TEXT" \
        '{
            model: $model,
            input: $input,
            response_format: $fmt,
            voice_ref: { type: "base64", data: input }
          }
        | if $reftext != "" then .reference_text = $reftext else . end' \
        < "$REF_B64_FILE" > "$PAYLOAD_FILE"
else
    jq -n \
        --arg model "$MODEL" \
        --arg input "$TEXT" \
        --arg fmt "$RESPONSE_FORMAT" \
        '{
            model: $model,
            input: $input,
            response_format: $fmt
          }' \
        > "$PAYLOAD_FILE"
fi

# ---------------------------------------------------------------------------
# call the API
# ---------------------------------------------------------------------------
echo "Sampling qwen3-tts voice clone..."
echo "  model:       $MODEL"
echo "  input:       $TEXT"
if [ -n "$REF_AUDIO" ]; then
    echo "  reference:   $REF_AUDIO (inline base64)"
else
    echo "  reference:   server-configured (backend/qwen3-tts.yaml)"
fi

OUT_DIR=$(dirname "$OUT")
if [ "$OUT_DIR" != "." ]; then
    mkdir -p "$OUT_DIR"
fi

HTTP_CODE=$(
    curl -sS -o "$OUT" -w '%{http_code}' \
        -H 'Content-Type: application/json' \
        --data-binary @"$PAYLOAD_FILE" \
        "${BASE_URL%/}/audio/speech"
)

if [ "$HTTP_CODE" != "200" ]; then
    echo "error: request failed (HTTP $HTTP_CODE) for ${BASE_URL%/}/audio/speech" >&2
    if [ -s "$OUT" ]; then
        head -c 1000 "$OUT" | tr -d '\000' >&2
        echo >&2
    fi
    rm -f "$OUT"
    exit 1
fi

# Sanity-check WAV output (audio.cpp returns WAV; transcoded formats are
# produced by the proxy and can't be cheaply magic-checked here).
if [ "$RESPONSE_FORMAT" = "wav" ]; then
    case "$(head -c 4 "$OUT" 2>/dev/null)" in
        RIFF) ;;
        *)
            echo "error: response is not a WAV file; body:" >&2
            head -c 1000 "$OUT" | tr -d '\000' >&2
            echo >&2
            rm -f "$OUT"
            exit 1
            ;;
    esac
fi

if command -v ffprobe >/dev/null 2>&1; then
    DUR=$(ffprobe -v error -show_entries format=duration \
        -of default=noprint_wrappers=1:nokey=1 "$OUT" 2>/dev/null || true)
    if [ -n "$DUR" ]; then
        echo "  duration:    ${DUR}s"
    fi
fi
echo "Wrote $(du -h "$OUT" | cut -f1) -> $OUT"
