#!/bin/sh
# docker-entrypoint.sh — llamaswap container entrypoint.
#
# Launches the proxy (CMD) after sniffing deploy-time values for the
# optional Strata engine off the mounted install. This is what lets
# docker-compose.hub.yml (prebuilt image, single file) enable Strata with
# zero configuration: when ~/.local/share/strata is bind-mounted at the
# fixed container path (see docker-compose.hub.yml), the pins the backend
# YAMLs expand via ${...} (app/registry.py) are detected here; without the
# mount nothing changes.
#
# Precedence: variables set through compose `environment` (deploy.sh
# exports them for the stacked deploy) always win — the sniff only fills
# in what is unset.
set -eu

STRATA_INSTALL=/opt/strata/.local/share/strata

# Gate on real directories: components/engine and terminal/. Everything
# under terminal/current is OFF LIMITS here — that symlink carries an
# absolute HOST path (<home>/<user>/...) which does not exist inside the
# container, so it can neither be traversed (cat, -x) nor resolved
# (readlink -f fails). The version comes from the terminal/ listing
# instead: it contains the versioned install dir(s) plus `current`.
if [ -d "$STRATA_INSTALL/components/engine" ] && [ -d "$STRATA_INSTALL/terminal" ]; then
    : "${STRATA_TERMINAL_VERSION:=$(ls -1 "$STRATA_INSTALL/terminal" | grep -vx current | head -n1)}"
    : "${STRATA_ENGINE_DIR:=$(ls -1 "$STRATA_INSTALL/components/engine" | head -n1)}"
    export STRATA_TERMINAL_VERSION STRATA_ENGINE_DIR
    # Sanity: the versioned dir we just derived must exist and carry the
    # matching VERSION marker (guards against a stray non-version entry).
    if [ -f "$STRATA_INSTALL/terminal/$STRATA_TERMINAL_VERSION/VERSION" ]; then
        # The IQ1_M pack's first load after a reboot can take ~13 min over
        # a cold page cache — beyond the 600 s default ceiling.
        : "${LLAMASWAP_STARTUP_TIMEOUT:=1800}"
        export LLAMASWAP_STARTUP_TIMEOUT
        echo "entrypoint: strata detected — terminal=$STRATA_TERMINAL_VERSION engine=$STRATA_ENGINE_DIR startup_timeout=$LLAMASWAP_STARTUP_TIMEOUT"
    else
        echo "entrypoint: warning: strata terminal version '$STRATA_TERMINAL_VERSION' not found under $STRATA_INSTALL/terminal — set STRATA_TERMINAL_VERSION/STRATA_ENGINE_DIR explicitly" >&2
    fi
fi

exec "$@"
