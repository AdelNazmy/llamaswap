---
id: TASK-5
title: 'Radiance readiness: /v1/health 200 before model is resident'
status: To Do
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
