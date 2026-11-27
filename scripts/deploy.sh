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
# Strata (docker-compose.strata.yml) is different: it builds nothing — it
# only bind-mounts the Strata install and the model disk (read-only) and
# raises the startup timeout. It is deployed by default like every other
# backend; a machine without the Strata files skips it automatically
# (--no-strata silences the check, --strata makes missing files an error).
#
# GPU: docker-compose.yml reserves no GPU itself — GPU exposure is a
# vendor-specific overlay. deploy.sh detects the host's GPU (nvidia-smi /
# rocm-smi) and stacks docker-compose.gpu-nvidia.yml (device reservations,
# needs nvidia-container-toolkit) or docker-compose.gpu-rocm.yml (/dev/kfd +
# /dev/dri only — the ROCm userland is baked into the image from the pinned
# stage, so the overlay must NOT mount the host's /opt/rocm over it).
# Direct `docker compose` users add the matching -f; without an overlay the
# GPU backends start CPU-only.
#
# Nothing is compiled inside the image — every backend is built on the HOST
# first and stamped in via compose `additional_contexts` (see the README
# "Building ..." sections). A missing binary aborts the matching build —
# except the optional Prism fork (llama_prism.cpp): when it is not built,
# the plain llama-server is stamped in as llama-prism-server instead.
#
# Model weights: before starting (skipped by --build-only and --no-download),
# deploy.sh checks
# the shared model root (MODEL_ROOT, default /opt/models) and downloads any
# missing TTS/ASR/FLUX weights by running ./scripts/download-models.sh. The
# one-shot init services (models-init / image-models-init) still run on
# `up -d` as a safety net for direct `docker compose up` users, but they'll
# find everything already present. The base chat LLM + embedding weights are
# still installed separately via `hf` (see README) and are not managed here.
#
# Usage:
#   ./scripts/deploy.sh                       # build all + start (up -d)
#   ./scripts/deploy.sh --build-only          # build all, do not start
#   ./scripts/deploy.sh --no-audio            # skip audio (base + image)
#   ./scripts/deploy.sh --no-image            # skip image (base + audio)
#   ./scripts/deploy.sh --no-strata           # skip the Strata engine
#   ./scripts/deploy.sh --no-audio --no-image # base only
#   ./scripts/deploy.sh --verify              # also smoke-test after start
#
# Env:
#   LLAMASWAP_PORT        proxy port to health-check (default: 11434)
#   LLAMASWAP_DEPLOY_WAIT max seconds to wait for /health during --verify
#                         (default: 180)
#   MODEL_ROOT            shared model dir to check/download into
#                         (default: /opt/models)
#   STRATA_HOME           Strata install to mount with --strata
#                         (default: $HOME/.local/share/strata; the
#                         container side is always /opt/strata/...)
#   STRATA_MODELS         model disk to mount with --strata
#                         (default: $MODEL_ROOT)
#   STRATA_TERMINAL_VERSION / STRATA_ENGINE_DIR
#                         version pins the model YAMLs expand (auto-sniffed
#                         off the install; see app/registry.py)
#
# An optional ./.env is sourced before any of these defaults are applied —
# precedence matches docker compose: shell environment beats .env.

set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT"

# --- optional .env (host-side deploy configuration) --------------------------
# Source ./.env (if present) so THIS script's own logic — the path
# validations, version sniff, health-check port — agrees with what
# `docker compose` substitutes into the services. Compose reads .env itself;
# sourcing only aligns deploy.sh with it. Precedence mirrors compose:
# a variable already in the shell environment wins over the file, so
# `STRATA_HOME=/x ./scripts/deploy.sh` always beats the .env value.
# Recognized lines: comments, blanks, KEY=VALUE, optional `export ` prefix,
# optional surrounding quotes, inline ` # comment` on unquoted values,
# CRLF line endings.
if [ -f .env ]; then
    cr=$(printf '\r')
    while IFS= read -r line || [ -n "$line" ]; do
        line=${line%"$cr"}
        case $line in
            ''|\#*) continue ;;
            export\ *) line=${line#export } ;;
        esac
        case $line in
            *=*) ;;
            *) continue ;;
        esac
        key=${line%%=*}
        value=${line#*=}
        # only valid shell identifiers become variables
        case $key in
            ''|*[!A-Za-z0-9_]*) continue ;;
        esac
        case $value in
            # quoted: cut at the CLOSING quote — anything after it (e.g.
            # ` # comment`) is dropped, and `#` inside the quotes survives
            \"*) value=${value#\"}; value=${value%%\"*} ;;
            \'*) value=${value#\'}; value=${value%%\'*} ;;
            # unquoted: strip an inline comment
            *) value=${value%%' #'*} ;;
        esac
        # shell environment wins: skip keys already set (even to empty) —
        # `${VAR+x}` is "x" when set, empty when unset; the key was
        # validated as an identifier above, so the eval is safe.
        if [ -z "$(eval "printf '%s' \"\${$key+x}\"")" ]; then
            export "$key=$value"
        fi
    done < .env
fi

INCLUDE_AUDIO=1
INCLUDE_IMAGE=1
INCLUDE_STRATA=1
STRATA_EXPLICIT=0
DO_UP=1
# Check/download model weights before starting (LLAMASWAP_NO_DOWNLOAD=1 or
# --no-download skips — assumes weights are already in place).
DO_DOWNLOAD=1
case "${LLAMASWAP_NO_DOWNLOAD:-0}" in
    1|true|yes|on) DO_DOWNLOAD=0 ;;
esac
VERIFY=0
# Install/refresh llamaswap's provider + skills in the user's opencode
# config (LLAMASWAP_INSTALL_OPENCODE=0 or --no-opencode disables).
case "${LLAMASWAP_INSTALL_OPENCODE:-1}" in
    0|false|no|off) INSTALL_OPENCODE=0 ;;
    *) INSTALL_OPENCODE=1 ;;
esac
PORT="${LLAMASWAP_PORT:-11434}"
WAIT="${LLAMASWAP_DEPLOY_WAIT:-180}"
# Shared model root — the single source of truth for STRATA_MODELS below
# and for download-models.sh (exported so the `sh` subprocess sees it).
MODEL_ROOT="${MODEL_ROOT:-/opt/models}"
export MODEL_ROOT

usage() {
    cat <<'EOF'
Usage: ./scripts/deploy.sh [options]

Deploy the full llamaswap stack. All backends are enabled by default.
./deploy.sh                       # build all + start (up -d)

Options:
  --no-audio      skip the audio overlay (base + image only)
  --no-image      skip the image overlay (base + audio only)
  --no-strata     skip the Strata overlay (mounts the Strata install +
                  model disk; runtime-only, builds nothing)
  --strata        require the Strata overlay (missing files abort instead
                  of auto-skipping)
  --build-only    build the images without starting the stack
  --no-download   skip the model-weights check/download (weights assumed
                  present under MODEL_ROOT)
  --verify        smoke-test /health and /v1/models after starting
  --no-opencode   skip installing the llamaswap provider + skills into the
                  default opencode config (~/.config/opencode)
  -h, --help      show this help

Backends: base (llama.cpp), audio (audio.cpp + whisper.cpp),
          image (stable-diffusion.cpp), strata (runtime-only, no image;
          auto-skipped on machines without the files).
EOF
    exit "${1:-0}"
}

# --- flag parsing -----------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        --no-audio)   INCLUDE_AUDIO=0 ;;
        --no-image)   INCLUDE_IMAGE=0 ;;
        --strata)     INCLUDE_STRATA=1; STRATA_EXPLICIT=1 ;;
        --no-strata)  INCLUDE_STRATA=0 ;;
        --build-only) DO_UP=0 ;;
        --no-download) DO_DOWNLOAD=0 ;;
        --verify)     VERIFY=1 ;;
        --no-opencode) INSTALL_OPENCODE=0 ;;
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
# The daemon must be reachable BY THIS USER: 'command -v docker' only proves
# the client is installed. Without socket access every docker call below
# dies with 'permission denied ... /var/run/docker.sock'.
if ! docker info >/dev/null 2>&1; then
    if docker info 2>&1 | grep -q 'permission denied'; then
        echo "deploy: cannot access the Docker socket as '$USER'" >&2
        echo "       add your user to the 'docker' group (then log out/in):" >&2
        echo "         sudo usermod -aG docker $USER" >&2
        echo "       or run this script with sudo" >&2
    else
        echo "deploy: cannot reach the Docker daemon (is it running?)" >&2
    fi
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

# The Prism fork (llama_prism.cpp) is OPTIONAL. The Dockerfile always copies
# a llama-server from the prism context (docker-compose*.yml point it at
# /opt/llama_prism.cpp/build/bin), so that context must resolve — when the
# fork is not built on this host, stamp the plain llama-server in instead,
# so the image's llama-prism-server exists (it simply lacks the fork's
# low-precision quantization support).
PRISM_CTX=/opt/llama_prism.cpp/build/bin
if [ ! -x "$PRISM_CTX/llama-server" ]; then
    echo "deploy: note: llama_prism.cpp is not built on this host; stamping the"
    echo "        plain llama-server as llama-prism-server instead"
    if ! mkdir -p "$PRISM_CTX" || ! cp /opt/llama.cpp/build/bin/llama-server "$PRISM_CTX/llama-server"; then
        echo "deploy: could not populate $PRISM_CTX (permissions?); build" >&2
        echo "        llama_prism.cpp (README 'Manual build') or create it manually" >&2
        MISSING=1
    fi
fi

# Radiance (qwen3.8-27b backend): the base Dockerfile stamps the engine
# binary + runtime plugin tree from its install dir and the front launcher +
# fleet config from its source tree (the radiance-bin / radiance-src build
# contexts in docker-compose.yml). All five must exist or the base build
# fails mid-COPY.
require_x /opt/radiance/bin/radiance
if [ ! -d /opt/radiance/share/radiance ]; then
    echo "deploy: missing radiance plugin tree: /opt/radiance/share/radiance" >&2
    MISSING=1
fi
for f in start.sh qwen.yaml; do
    if [ ! -f "$HOME/radiance/$f" ]; then
        echo "deploy: missing radiance source file: $HOME/radiance/$f" >&2
        MISSING=1
    fi
done

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

# Strata: runtime-only overlay — no image to build, but the bind mounts and
# the pack files must exist on the host (docker-compose.strata.yml mounts
# STRATA_HOME at the fixed container path /opt/strata/... and STRATA_MODELS
# at /opt/models). A default deploy auto-skips strata when they are
# missing; an explicit --strata makes that an error instead.
if [ "$INCLUDE_STRATA" = 1 ]; then
    STRATA_HOME="${STRATA_HOME:-$HOME/.local/share/strata}"
    STRATA_MODELS="${STRATA_MODELS:-$MODEL_ROOT}"
    STRATA_PACK="$STRATA_MODELS/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M"

    # Strata misses are tracked in their OWN flag: the auto-skip below must
    # not fire (or clear MISSING) on errors from other backends, and a
    # missing binary above must not be masked by strata being skipped.
    STRATA_MISSING=0
    if [ ! -x "$STRATA_HOME/terminal/current/bin/strata" ]; then
        echo "deploy: missing host-built binary: $STRATA_HOME/terminal/current/bin/strata" >&2
        STRATA_MISSING=1
    fi
    for shard in 00001 00002; do
        f="$STRATA_PACK/models/coder-IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-$shard-of-00002.gguf"
        [ -f "$f" ] || {
            echo "deploy: missing strata weights shard: $f" >&2
            STRATA_MISSING=1
        }
    done
    [ -d "$STRATA_PACK/packs/coder-iq1_m" ] || {
        echo "deploy: missing strata pack dir: $STRATA_PACK/packs/coder-iq1_m" >&2
        STRATA_MISSING=1
    }
    [ -d "$STRATA_PACK/mtp/rt" ] || {
        echo "deploy: missing strata MTP dir: $STRATA_PACK/mtp/rt" >&2
        STRATA_MISSING=1
    }

    if [ "$STRATA_MISSING" = 1 ]; then
        if [ "$STRATA_EXPLICIT" = 1 ]; then
            echo "deploy: --strata requested but the files above are missing" >&2
            exit 1
        fi
        echo "deploy: strata files missing; continuing WITHOUT the strata" >&2
        echo "        overlay (--no-strata silences this check, --strata makes" >&2
        echo "        it an error)" >&2
        INCLUDE_STRATA=0
    fi

    # A standalone strata instance holds the same ~15.2 GiB the managed
    # pack needs — warn (it is this deployment's own main model, so only
    # the user can decide to stop it).
    if [ "$INCLUDE_STRATA" = 1 ] && command -v pgrep >/dev/null 2>&1 \
            && pgrep -f "strata serve" >/dev/null 2>&1; then
        echo "deploy: warning: a standalone 'strata serve' is running and holds" >&2
        echo "        the VRAM the managed pack needs; stop it (pkill -f 'strata serve')" >&2
        echo "        and repoint its clients at llamaswap's published port" >&2
    fi
fi

# Strata version pins, sniffed off the install: the model YAMLs reference
# the versioned terminal dir and engine folder via ${STRATA_TERMINAL_VERSION}
# / ${STRATA_ENGINE_DIR} (expanded by app/registry.py from the environment
# docker-compose.strata.yml passes into the container). Run AFTER the file
# checks: they guarantee both sources exist. Overrides stay respected.
if [ "$INCLUDE_STRATA" = 1 ]; then
    : "${STRATA_TERMINAL_VERSION:=$(basename "$(readlink -f "$STRATA_HOME/terminal/current")")}"
    : "${STRATA_ENGINE_DIR:=$(ls -1 "$STRATA_HOME/components/engine" 2>/dev/null | head -n1)}"
    export STRATA_TERMINAL_VERSION STRATA_ENGINE_DIR
    echo "==> strata version pins: terminal=$STRATA_TERMINAL_VERSION engine=$STRATA_ENGINE_DIR"
fi

if [ "$MISSING" = 1 ]; then
    echo "deploy: build the missing backend(s) on the host first" \
        "(see README 'Building ...' sections)" >&2
    exit 1
fi

# GPU exposure is a per-vendor overlay (docker-compose.gpu-nvidia.yml /
# docker-compose.gpu-rocm.yml); detect the vendor and remember the overlay
# for the FILES assembly below. Probe with the canonical SMI of each vendor:
# 'amd-smi -L' does not exist (ROCm's amd-smi takes subcommands, and a venv
# 'amd-smi' may shadow it on PATH), so use rocm-smi for AMD. No GPU found
# -> no overlay; the GPU backends would start CPU-only.
GPU_OVERLAY=
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    GPU_OVERLAY=gpu-nvidia
elif command -v rocm-smi >/dev/null 2>&1 \
        && rocm-smi --showproductname >/dev/null 2>&1; then
    GPU_OVERLAY=gpu-rocm
elif command -v amd-smi >/dev/null 2>&1 && amd-smi list >/dev/null 2>&1; then
    GPU_OVERLAY=gpu-rocm
else
    echo "deploy: warning: no GPU visible; GPU backends will run CPU-only" >&2
fi

# --- ROCm runtime closure staging --------------------------------------------
# The image's GPU runtime libs are stamped from the HOST's ROCm install — the
# host build is custom (TensileLite hipBLASLt, librocm_kpack, compressed
# kernel data) and does NOT match the stock rocm/dev stage of the same
# version number. scripts/stamp-rocm-closure.sh materializes the ldd closure
# into build/rocm-closure/ (the 'rocm-closure' build context). Non-ROCm
# hosts stage an empty context; builds stay green.
if [ -d /opt/rocm/lib ]; then
    echo "==> staging ROCm runtime closure"
    sh "$ROOT/scripts/stamp-rocm-closure.sh"
else
    echo "==> no host ROCm; skipping runtime closure staging"
    mkdir -p "$ROOT/build/rocm-closure"
fi

# --- build in order: base -> audio -> image ---------------------------------
# (strata builds nothing — it is a runtime-only overlay, see --strata)
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

# --- privilege / dependency helpers -----------------------------------------
# Called only when a model download is actually needed, so `sudo` is only
# requested when model weights are missing (never for a no-op deploy).

need_escalation() { # -> 0 if we can run commands as root (root, or sudo)
    [ "$(id -u)" -eq 0 ] || command -v sudo >/dev/null 2>&1
}

run_root() { # run "$@" as root (sudo when not already root)
    if [ "$(id -u)" -eq 0 ]; then
        "$@"
    else
        sudo "$@"
    fi
}

ensure_model_root() { # dir -> 0 once it exists and is writable by this user
    root="$1"
    if [ -d "$root" ] && [ -w "$root" ]; then
        return 0
    fi
    echo "deploy: model root '$root' is missing or not writable; requesting sudo" >&2
    if ! need_escalation; then
        echo "deploy: cannot sudo; run as root, or create a writable '$root' manually" >&2
        return 1
    fi
    run_root mkdir -p "$root" || return 1
    run_root chown "$(id -u):$(id -g)" "$root" 2>/dev/null || true
    if [ -d "$root" ] && [ -w "$root" ]; then
        return 0
    fi
    echo "deploy: still unable to write '$root'; check its permissions" >&2
    return 1
}

ensure_aria2c() { # -> 0 once aria2c is on PATH
    if command -v aria2c >/dev/null 2>&1; then
        return 0
    fi
    echo "deploy: aria2c is required to download model weights" >&2
    if ! need_escalation; then
        echo "deploy: cannot sudo; install 'aria2' manually and re-run" >&2
        return 1
    fi
    if command -v apt-get >/dev/null 2>&1; then
        echo "deploy: installing aria2 via apt-get ..."
        run_root apt-get update && run_root apt-get install -y aria2
    elif command -v dnf >/dev/null 2>&1; then
        run_root dnf install -y aria2
    elif command -v yum >/dev/null 2>&1; then
        run_root yum install -y aria2
    elif command -v apk >/dev/null 2>&1; then
        run_root apk add --no-cache aria2
    elif command -v pacman >/dev/null 2>&1; then
        run_root pacman -S --noconfirm aria2
    else
        echo "deploy: no supported package manager found; install 'aria2' manually and re-run" >&2
        return 1
    fi
    command -v aria2c >/dev/null 2>&1
}

# --- model weights ---------------------------------------------------------
# Runtime concern only; --build-only and --no-download skip this.
# download-models.sh is idempotent (skips files that already exist), but we
# ask it which of our required bundles are missing and only invoke the
# download when there is work to do.
if [ "$DO_UP" = 1 ] && [ "$DO_DOWNLOAD" = 1 ]; then
    NEEDED=
    if [ "$INCLUDE_AUDIO" = 1 ]; then
        NEEDED="$NEEDED tts asr whisper whisper-multi tts-qwen3 nemotron_asr supertonic_3"
    fi
    if [ "$INCLUDE_IMAGE" = 1 ]; then
        NEEDED="$NEEDED flux-schnell flux-vae flux-clip_l flux-t5xxl"
    fi

    if [ -z "$NEEDED" ]; then
        echo "==> no downloadable model bundles for this configuration"
    else
        # shellcheck disable=SC2086  # intentional word split into bundle names
        missing="$(sh "$ROOT/scripts/download-models.sh" --missing $NEEDED)"
        if [ -n "$missing" ]; then
            echo
            echo "==> missing model weights detected; downloading into $MODEL_ROOT"
            # shellcheck disable=SC2086  # newline/space-separated bundle names
            for b in $missing; do echo "      - $b"; done
            ensure_model_root "$MODEL_ROOT" || exit 1
            ensure_aria2c || exit 1
            # shellcheck disable=SC2086
            sh "$ROOT/scripts/download-models.sh" $NEEDED
        else
            echo "==> all required model weights present under $MODEL_ROOT"
        fi
    fi
fi

# --- opencode integration ---------------------------------------------------
# Merge the llamaswap provider + skills into the user's default opencode
# config so `opencode` can select the local models (and the artist /
# music-composer skills) without manual setup. Non-fatal: a missing python3
# or an unwritable config never blocks the deployment.
if [ "$INSTALL_OPENCODE" = 1 ]; then
    echo "==> install opencode integration (llamaswap provider + skills)"
    if command -v python3 >/dev/null 2>&1; then
        python3 "$ROOT/scripts/install-opencode.py" \
            --config-dir "${OPENCODE_CONFIG_DIR:-$HOME/.config/opencode}" \
            --base-url "${LLAMASWAP_OPENCODE_BASE_URL:-http://localhost:9090/v1}" \
            || echo "deploy: warning: failed to update the opencode config" >&2
    else
        echo "deploy: python3 not found; skipping opencode integration" >&2
    fi
fi

# --- start ------------------------------------------------------------------
FILES="-f docker-compose.yml"
[ -n "$GPU_OVERLAY" ] && FILES="$FILES -f docker-compose.$GPU_OVERLAY.yml"
[ "$INCLUDE_AUDIO" = 1 ] && FILES="$FILES -f docker-compose.audio.yml"
[ "$INCLUDE_IMAGE" = 1 ] && FILES="$FILES -f docker-compose.image.yml"
[ "$INCLUDE_STRATA" = 1 ] && FILES="$FILES -f docker-compose.strata.yml"

if [ "$DO_UP" = 1 ]; then
    echo "==> start stack (docker compose up -d --force-recreate)"
    echo "    (first run may block while the init services download missing"
    echo "     model weights into /opt/models)"
    # --force-recreate: `up` never re-points an existing container at a
    # same-tag image that was just rebuilt (it compares the image STRING,
    # not the digest), so a deploy after an image change would silently
    # keep the old image running. The stack keeps no container-local state
    # (everything is host-mounted), so a forced recreate is always safe.
    # shellcheck disable=SC2086  # fixed, space-free file names -> intentional split
    docker compose $FILES up -d --force-recreate

    echo
    echo "llamaswap is starting. Endpoints:"
    echo "  API:     http://localhost:$PORT/v1   (also :9090/v1 and :1234/v1)"
    echo "  health:  http://localhost:$PORT/health"
    echo "  models:  http://localhost:$PORT/v1/models"
    echo "  logs:    docker compose $FILES logs -f llamaswap"
    if [ "$INCLUDE_STRATA" = 1 ]; then
        echo
        echo "  strata:  first request loads the pack (~2.5 min; the first"
        echo "           boot after a reboot can take ~13 min); it stays"
        echo "           resident for 4 h idle, then unloads"
    fi
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
