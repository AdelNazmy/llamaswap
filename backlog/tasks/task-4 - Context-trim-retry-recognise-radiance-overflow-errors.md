---
id: TASK-4
title: 'Context-trim retry: recognise radiance overflow errors'
status: Done
assignee: []
created_date: '2026-11-22T09:14:59Z'
labels:
  - proxy
milestone: m-0
dependencies: []
references:
  - app/proxy.py
priority: low
type: enhancement
ordinal: 4000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
proxy.py's context-trim retry logic matches strata/llama-server error wordings only. Radiance's context-overflow errors use different wording, so an oversized request to a radiance-backed model gets a raw error instead of the trim-and-retry path. Add radiance's overflow wording (capture the actual message from a forced overflow) or generalise the matcher to detect context-length errors across backends.
<!-- SECTION:DESCRIPTION:END -->

## Implementation

Built 2026-11-23 in `app/proxy.py`:
- Verified against the radiance source (core/server/oai.cpp): the
  request-too-long error reuses llama-server's numbers beside "maximum
  context length is M tokens" ("the request (N tokens) exceeds the
  available context size (M tokens)") — the existing llama pattern already
  matches it, so no change was needed for that wording.
- Added a third `OVERFLOW_RES` pattern for the KV-pool-full refusal
  (scheduler.cpp `pool_refusal_`), which names only the cap:
  `maximum context length is (\d+) tokens`, appended last. `overflow_info`
  maps its 1-capture match to `(0, ctx)` so `trim_messages` falls back to
  its existing chars/4 estimate for the missing prompt count.
- Tests: `tests/test_context_trim.py::test_overflow_info_radiance_variants`
  covers both wordings raw and wrapped in the OpenAI-style error JSON
  (`code: context_length_exceeded`).
