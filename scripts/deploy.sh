#!/usr/bin/env sh
# deploy.sh — build and deploy the complete llamaswap stack (all backends).
#
# Deploys every backend, strictly in the order the images must be built:
#
#   1. base   — llama.cpp `llama-server`                (chat LLM + embedding)  [always]
#   2. audio  — audio.cpp `audiocpp_server`               (TTS/ASR)
#             + whisper.cpp `whisper-server`              (ASR)
#   3. image  — stable-diffusion.cpp `sd-server`          (text-to-image)
#
# The order matters: Dockerfile.audio and Dockerfile.image are BOTH
# `FROM llamaswap:local` and stamp their host-built binaries into the SAME
# image tag, so each overlay must be layered on top of the previous image
# (base -> audio -> image). Rebuilding the base out of order silently drops
# the audio/overlay layers already in the tag.
#
# Nothing is compiled inside the image — every backend is built on the HOST
# first and stamped in via compose `additional_contexts` (see the README
# "Building ..." sections). A missing binary aborts the matching build.
#
# Model weights are NOT downloaded here: on `up -d` the one-shot init
# services (models-init for audio, image-models-init for image) download any
# missing weights into /opt/models before llamaswap starts. To fetch them
# manually instead, run ./scripts/download-models.sh [bundle ...].
#
# Usage:
#   ./scripts/deploy.sh                       # build all + start (up -d)
#   ./scripts/deploy.sh --build-only          # build all, do not start
#   ./scripts/deploy.sh --no-audio            # skip audio (base + image)
#   ./scripts/deploy.sh --no-image            # skip image (base + audio)
#   ./scripts/deploy.sh --no-audio --no-image # base only
#   ./scripts/deploy.sh --verify              # also smoke-test after start
#
# Env:
#   LLAMASWAP_PORT        proxy port to health-check (default: 11434)
#   LLAMASWAP_DEPLOY_WAIT max seconds to wait for /health during --verify
#                         (default: 180)

set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT"

INCLUDE_AUDIO=1
INCLUDE_IMAGE=1
DO_UP=1
VERIFY=0
PORT="${LLAMASWAP_PORT:-11434}"
WAIT="${LLAMASWAP_DEPLOY_WAIT:-180}"

usage() {
    cat <<'EOF'
Usage: ./scripts/deploy.sh [options]

Deploy the full llamaswap stack. All backends are enabled by default.
./deploy.sh                       # build all + start (up -d)

Options:
  --no-audio      skip the audio overlay (base + image only)
  --no-image      skip the image overlay (base + audio only)
  --build-only    build the images without starting the stack
  --verify        smoke-test /health and /v1/models after starting
  -h, --help      show this help

Backends: base (llama.cpp), audio (audio.cpp + whisper.cpp),
          image (stable-diffusion.cpp).
EOF
    exit "${1:-0}"
}

# --- flag parsing -----------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        --no-audio)   INCLUDE_AUDIO=0 ;;
        --no-image)   INCLUDE_IMAGE=0 ;;
        --build-only) DO_UP=0 ;;
        --verify)     VERIFY=1 ;;
        -h|--help)    usage 0 ;;
        *) echo "deploy: unknown option '$1' (try --help)" >&2; exit 2 ;;
    esac
    shift
done

# --- prerequisites: tooling -------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
    echo "deploy: docker is not installed" >&2
    exit 1
fi
if ! docker compose version >/dev/null 2>&1; then
    echo "deploy: 'docker compose' (Docker Compose v2) is required" >&2
    exit 1
fi

# --- prerequisites: host-built backend binaries -----------------------------
# These are NOT compiled by the build; they are stamped in from these paths.
MISSING=0
require_x() {
    if [ ! -x "$1" ]; then
        echo "deploy: missing host-built binary: $1" >&2
        MISSING=1
    fi
}

require_x /opt/llama.cpp/build/bin/llama-server

if [ "$INCLUDE_AUDIO" = 1 ]; then
    require_x /opt/audio.cpp/build/bin/audiocpp_server
    require_x /opt/whisper.cpp/build/bin/whisper-server
    if [ ! -d /opt/audio.cpp/model_specs ]; then
        echo "deploy: missing audio schema dir: /opt/audio.cpp/model_specs" >&2
        MISSING=1
    fi
fi

if [ "$INCLUDE_IMAGE" = 1 ]; then
    require_x /opt/stable-diffusion.cpp/build/bin/sd-server
fi

if [ "$MISSING" = 1 ]; then
    echo "deploy: build the missing backend(s) on the host first" \
        "(see README 'Building ...' sections)" >&2
    exit 1
fi

# GPU visibility is a runtime concern (nvidia container runtime), warn only.
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L >/dev/null 2>&1 || \
        echo "deploy: warning: no NVIDIA GPU visible; GPU backends will not run" >&2
else
    echo "deploy: warning: nvidia-smi not found; GPU backends need the NVIDIA" \
        "container runtime" >&2
fi

# --- build in order: base -> audio -> image ---------------------------------
echo "==> [1/3] build base image (llama.cpp)"
docker compose -f docker-compose.yml build

if [ "$INCLUDE_AUDIO" = 1 ]; then
    echo "==> [2/3] build audio overlay (audio.cpp + whisper.cpp)"
    docker compose -f docker-compose.yml -f docker-compose.audio.yml build
else
    echo "==> [2/3] audio overlay skipped (--no-audio)"
fi

if [ "$INCLUDE_IMAGE" = 1 ]; then
    echo "==> [3/3] build image overlay (stable-diffusion.cpp)"
    docker compose -f docker-compose.yml -f docker-compose.image.yml build
else
    echo "==> [3/3] image overlay skipped (--no-image)"
fi

# --- start ------------------------------------------------------------------
FILES="-f docker-compose.yml"
[ "$INCLUDE_AUDIO" = 1 ] && FILES="$FILES -f docker-compose.audio.yml"
[ "$INCLUDE_IMAGE" = 1 ] && FILES="$FILES -f docker-compose.image.yml"

if [ "$DO_UP" = 1 ]; then
    echo "==> start stack (docker compose up -d)"
    echo "    (first run may block while the init services download missing"
    echo "     model weights into /opt/models)"
    # shellcheck disable=SC2086  # fixed, space-free file names -> intentional split
    docker compose $FILES up -d

    echo
    echo "llamaswap is starting. Endpoints:"
    echo "  API:     http://localhost:$PORT/v1   (also :9090/v1 and :1234/v1)"
    echo "  health:  http://localhost:$PORT/health"
    echo "  models:  http://localhost:$PORT/v1/models"
    echo "  logs:    docker compose $FILES logs -f llamaswap"
else
    echo "==> build-only requested; skipping 'up -d'"
    echo "    start with: docker compose $FILES up -d"
fi

# --- optional smoke test ----------------------------------------------------
if [ "$DO_UP" = 1 ] && [ "$VERIFY" = 1 ]; then
    BASE="http://localhost:$PORT"
    echo
    echo "==> waiting for /health (up to ${WAIT}s) ..."
    i=0
    while [ "$i" -lt "$WAIT" ]; do
        if curl -fsS -m 3 "$BASE/health" -o /tmp/llamaswap-health.json 2>/dev/null; then
            break
        fi
        i=$((i + 1))
        sleep 1
    done
    if [ "$i" -ge "$WAIT" ]; then
        echo "deploy: /health did not respond within ${WAIT}s" >&2
        echo "       check: docker compose $FILES logs llamaswap" >&2
        exit 1
    fi

    echo "==> /health"
    if command -v python3 >/dev/null 2>&1; then
        python3 -m json.tool /tmp/llamaswap-health.json
    else
        cat /tmp/llamaswap-health.json; echo
    fi

    echo "==> /v1/models (by role)"
    if command -v python3 >/dev/null 2>&1; then
        curl -fsS -m 5 "$BASE/v1/models" | python3 -c '
import sys, json
from collections import defaultdict
data = json.load(sys.stdin)["data"]
by = defaultdict(list)
for m in data:
    by[m.get("role", "?")].append(m["id"])
for role in sorted(by):
    print(role + ":")
    for name in by[role]:
        print("  - " + name)
'
    else
        curl -fsS -m 5 "$BASE/v1/models" | head -c 1200; echo
    fi
    echo
    echo "verify OK"
fi
