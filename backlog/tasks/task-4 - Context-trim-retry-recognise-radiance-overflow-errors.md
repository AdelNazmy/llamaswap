---
id: TASK-4
title: 'Context-trim retry: recognise radiance overflow errors'
status: To Do
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
