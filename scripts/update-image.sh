#!/usr/bin/env sh
# update-image.sh — refresh the published Docker Hub image with the
# checkout's current code, without a full rebuild.
#
# The published image carries the compiled backend binaries
# (llama-server, audiocpp_server, whisper-server, sd-server) and the
# Python dependency layers from the machine that built it. This script
# layers only what changes between releases — app/, backend/, opencode/,
# the opencode installer and the entrypoint (Dockerfile.update, the tail
# of the base Dockerfile) — on top of the existing image, so a release
# needs no /opt builds and no host toolchain.
#
# Steps:
#   1. refuse a dirty work tree — the image must match a git commit
#      (a branch that is not pushed is a warning, not a blocker)
#   2. pull the base image when it is missing locally (--pull re-pulls
#      even when present, to build on the newest published state)
#   3. build Dockerfile.update on top of it (tag: $IMAGE:$TAG and
#      llamaswap:local)
#   4. smoke-test the built image itself (no mounts): import the app and
#      boot uvicorn long enough to answer /health
#   5. push to Docker Hub (--no-push skips; push needs docker login)
#
# Usage:
#   ./scripts/update-image.sh              # build + smoke + push
#   ./scripts/update-image.sh --no-push    # build + smoke only
#   ./scripts/update-image.sh --pull       # re-pull the base image first
#
# Env:
#   LLAMASWAP_IMAGE   repo/name to publish (default: adelnazmy2002/llamaswap)
#   LLAMASWAP_TAG     tag to publish (default: latest)
#   LLAMASWAP_DOCKERFILE  update Dockerfile (default: Dockerfile.update)

set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT"

IMAGE="${LLAMASWAP_IMAGE:-adelnazmy2002/llamaswap}"
TAG="${LLAMASWAP_TAG:-latest}"
DOCKERFILE="${LLAMASWAP_DOCKERFILE:-Dockerfile.update}"
PULL=0
PUSH=1
for arg in "$@"; do
    case "$arg" in
        --no-push) PUSH=0 ;;
        --pull) PULL=1 ;;
        -h|--help) sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "update-image: unknown option '$arg' (try --help)" >&2; exit 2 ;;
    esac
done

if ! command -v docker >/dev/null 2>&1; then
    echo "update-image: docker is not installed" >&2
    exit 1
fi

# --- the image must match a commit ------------------------------------------
if [ -n "$(git status --porcelain)" ]; then
    echo "update-image: work tree is dirty; commit first — the published" >&2
    echo "               image must be reproducible from a commit" >&2
    exit 1
fi
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if git fetch -q origin "$BRANCH" 2>/dev/null \
        && [ -n "$(git rev-list "origin/$BRANCH..HEAD" 2>/dev/null)" ]; then
    echo "update-image: warning: $BRANCH has commits not pushed to origin;" >&2
    echo "               publishing an image ahead of the repo is confusing" >&2
fi

# --- base image ---------------------------------------------------------------
if [ "$PULL" = 1 ] || ! docker image inspect "$IMAGE:$TAG" >/dev/null 2>&1; then
    echo "==> pulling base image $IMAGE:$TAG"
    docker pull "$IMAGE:$TAG"
fi

# --- build the refresh layers -------------------------------------------------
echo "==> building $IMAGE:$TAG (+ llamaswap:local) from $DOCKERFILE"
docker build -f "$DOCKERFILE" \
    --build-arg "BASE_IMAGE=$IMAGE:$TAG" \
    -t "$IMAGE:$TAG" -t llamaswap:local "$ROOT"

# --- smoke: the built image itself, no mounts ---------------------------------
echo "==> smoke: import + boot + /health inside the built image"
docker run --rm --entrypoint sh "$IMAGE:$TAG" -c "
python -c 'import app.main, app.token_store; print(\"    import ok\")' || exit 1
python -m uvicorn app.main:app --host 127.0.0.1 --port 11499 >/tmp/boot.log 2>&1 &
up=\$!
sleep 5
python -c 'import urllib.request; print(\"    /health\", urllib.request.urlopen(\"http://127.0.0.1:11499/health\", timeout=5).status)' || { cat /tmp/boot.log; exit 1; }
kill \$up 2>/dev/null
"

# --- publish -------------------------------------------------------------------
if [ "$PUSH" = 1 ]; then
    echo "==> pushing $IMAGE:$TAG"
    docker push "$IMAGE:$TAG"
    digest="$(docker image inspect "$IMAGE:$TAG" --format '{{index .RepoDigests 0}}' 2>/dev/null || true)"
    [ -n "$digest" ] && echo "    pushed: $digest" || true
else
    echo "==> --no-push: image built locally; publish with: docker push $IMAGE:$TAG"
fi
