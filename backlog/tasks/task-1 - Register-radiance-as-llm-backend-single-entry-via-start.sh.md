---
id: TASK-1
title: Register radiance as llm backend (single entry via start.sh)
status: In Progress
assignee: []
created_date: '2026-11-22T09:14:33Z'
labels:
  - radiance
  - integration
milestone: m-0
dependencies: []
references:
  - /home/adnathan/radiance/start.sh
  - /home/adnathan/radiance/core/main.cpp
priority: high
type: feature
ordinal: 1000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Step 1 of the radiance front-door plan: zero code changes to llamaswap.

Create backend/qwen3.8-27b-iq4r-v4.yaml:
- binary: /home/adnathan/radiance/start.sh — start.sh does `exec radiance --config qwen.yaml` and swallows the `--host/--port` that build_argv (app/registry.py) always appends. Radiance's CLI accepts only --help/--check/--config; any other flag hits refuse_flags() in core/main.cpp and exits 2, so the launcher must be the front binary.
- host: 127.0.0.1, port: 8000 — the port MUST equal server.port in qwen.yaml, since the health poll targets the entry port.
- meta.health_path: /v1/health
- meta.vram_gb: ~30.5 (weights 16.4 + KV pool 13.77 + driver) so the VRAM guards can rank it
- meta.idle_unload_seconds: 0 for resident (each stop/start cycle = 13 s cold boot + prefix cache loss) or 300 to free ~30 GiB VRAM for the audio roles after idle.

Stop contract verified: SIGTERM makes radiance drain in-flight requests then run ~Engine (core/fleet/fleet.cpp); llamaswap's 30 s timeout -> SIGKILL is the documented force path. start.sh re-applies the 230 W power cap idempotently on every launch.

Handover: kill the manually started radiance instance so llamaswap owns port 8000; the entry boots on the first chat request.

Verify: first chat request via llamaswap boots radiance; /v1/health 200 on :8000; cap reads 230 W (amd-smi static -g 0 -l); response model field passes through unchanged.
<!-- SECTION:DESCRIPTION:END -->

## Implementation

Built 2026-11-23:
- Created `backend/qwen3.8-27b-iq4r-v4.yaml` per spec: `start.sh` front binary,
  host 127.0.0.1, port 8000 (must equal qwen.yaml's server.port),
  `meta.health_path: /v1/health`, `meta.ready_path: /admin/fleet` (TASK-5),
  `meta.vram_gb: 30.5`, `meta.idle_unload_seconds: 0` (resident).
- Verified: `tests/test_process_manager.py::test_radiance_registry_entry`
  loads the real `backend/` directory and asserts the entry parses intact
  (host/port/meta) and that `build_argv()` is exactly
  `start.sh --host 127.0.0.1 --port 8000`.

Handover (run from another terminal once the session that the manual
instance is serving has ended — that instance serves the session that
wrote these notes):
    pkill -TERM -f "radiance --config qwen.yaml"   # SIGTERM drains, then exits
    ss -ltnp | grep :8000                          # must come up empty
    # then start llamaswap the usual way:
    ~/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 11434
    curl -s localhost:11434/v1/models              # the new entry is listed

Remaining (live verification): first chat request through llamaswap boots
radiance via start.sh (re-applying the 230 W cap); /v1/health:8000 answers
200 once resident; `amd-smi static -g 0 -l` reads 230 W; the response
model field passes through unchanged.
