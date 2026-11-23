---
id: TASK-2
title: 'ensure_model: no swap when argv identical (multi-model backend aliases)'
status: Done
assignee: []
created_date: '2026-11-22T09:14:50Z'
labels:
  - radiance
  - proxy
milestone: m-0
dependencies:
  - TASK-1
references:
  - app/process_manager.py
  - app/registry.py
priority: high
type: feature
ordinal: 2000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Structural gap in the front-door plan. ensure_model (app/process_manager.py:238-254) decides "same server?" by MODEL NAME ONLY (line 245). Two registry entries that launch the identical process — radiance is one process serving N models, selected by the client model field — would SIGTERM + cold-restart radiance on every switch: 13 s load plus total prefix-cache loss per switch.

Fix: after the name check, if the current model is READY and its build_argv equals the requested entry's build_argv, return (name, cfg.port) without stop/start — the two entries are aliases of one server and model switching happens inside radiance (park to RAM ~2.5 s, load requested model).

Then register one registry entry per radiance fleet model (entry name == radiance model name: proxy_json forwards the client model field unmodified on OpenAI routes, and radiance honours it).

Add a unit test: alias switch must not send SIGTERM / must not restart the process.
<!-- SECTION:DESCRIPTION:END -->

## Implementation

Built 2026-11-23 in `app/process_manager.py`:
- `ensure_model` now checks, for a READY current process, whether the
  requested entry's `build_argv` equals the running entry's argv
  (`_same_argv`, which rebuilds the running argv with the rendered config
  path of the process actually spawned). On a match it only updates the
  bookkeeping name — the entry name follows the alias — and returns
  `(name, cfg.port)`: no SIGTERM, no restart, no swap metric/event.
- Tests: `tests/test_process_manager.py` — same name (no restart), alias
  switch (no SIGTERM / no restart / name follows / same process object),
  alias back-and-forth, genuinely different argv (still SIGTERM + restart),
  non-READY (LOADING) current (still replaced).
