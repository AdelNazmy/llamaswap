---
id: TASK-5
title: 'Radiance readiness: /v1/health 200 before model is resident'
status: Done
assignee: []
created_date: '2026-11-22T09:15:03Z'
labels:
  - radiance
  - proxy
milestone: m-0
dependencies: []
references:
  - /home/adnathan/radiance/core/main.cpp
priority: low
type: enhancement
ordinal: 5000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Radiance's /v1/health returns 200 as soon as the HTTP server binds, which can be seconds before the fleet finishes loading the model(s). llamaswap will therefore mark the backend READY early; the first request after a stop simply waits out the load (600 s startup timeout >> ~120 s worst-case load) — no failure mode, only latency on that first request.

Optional improvement: before declaring READY, poll /admin/fleet (or /stats) until the requested model's residency is loaded, or have radiance report per-model readiness in /v1/health.
<!-- SECTION:DESCRIPTION:END -->

## Implementation

Built 2026-11-23:
- New opt-in `meta.ready_path` (`app/registry.py` ModelMeta, default empty =
  off). When set, the process manager additionally polls this path on the
  same host:port after /v1/health answers 200, and declares READY only
  when the JSON body's `models` array carries an entry with `id` equal to
  the model name and `state: "resident"` (Radiance's /admin/fleet shape).
  The radiance entry sets `ready_path: /admin/fleet`.
- `app/process_manager.py`: `_wait_healthy` runs the readiness probe inside
  its existing startup-timeout loop; `_fleet_resident` parses the fleet
  status. /admin/fleet is guarded by radiance's local-operator check
  (loopback peer + local Host), which llamaswap's 127.0.0.1 poll satisfies.
- Why both gates: a single-model fleet already gates /v1/health on
  residency (`required = pin || models.size() == 1`, front.cpp), so the
  probe is the safety net for a multi-model fleet (TASK-3), where
  /v1/health can answer 200 before a given model is resident.
- Tests: `tests/test_process_manager.py::test_fleet_resident` (resident /
  loading / unknown id / parked / non-JSON / empty body / empty array /
  wrong key) and `test_radiance_registry_entry` (asserts `ready_path` on
  the real entry).
