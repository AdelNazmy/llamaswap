---
id: TASK-3
title: Move qwen3.6-35b into the radiance fleet (pattern A coexistence)
status: To Do
assignee: []
created_date: '2026-11-22T09:14:51Z'
labels:
  - radiance
  - models
milestone: m-0
dependencies:
  - TASK-1
references:
  - /home/adnathan/radiance/docs/MULTIMODEL.md
priority: medium
type: feature
ordinal: 3000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Co-existence plan "pattern A": radiance owns the big LLMs (at most one resident, others parked in pinned RAM, ~2.5 s restore from the 42 GB free host RAM), llamaswap keeps its small models. Single GPU: VRAM is zero-sum (live: 32,174 / 32,624 MB used). Loading a 35B next to the resident 27B does not fit, and an oversized-model load failure while radiance holds the card can trigger the documented REMOVE_QUEUE/mode1 card reset that wipes ALL VRAM (docs/MULTIMODEL.md §3.1) — so the 35B must live inside radiance's fleet, never as a second llamaswap big-LLM process.

Steps:
1. rad-convert qwen3.6-35b via q36-35b-a3b-iq8r.recipe (rad-convert refuses llama.cpp k-quants; only GPTQ/CT-INT4 integer grids + FP8 import).
2. Add the fleet model to the radiance config alongside qwen3.8-27b-iq4r-v4.
3. Set gpu-headroom-mib: 4608 in qwen.yaml so the llamaswap small models (embedding ~0.6B + TTS/ASR) still fit: KV pool 13.77 -> ~9.8 GiB, 395K -> ~281K tokens; one full 262K session still fits (current session at 263K makes margin thin).
4. Validate with radiance --check and a live /admin/fleet park/restore cycle.

Blocked on the TASK-2 alias patch if model switching through the llamaswap front door is wanted.
<!-- SECTION:DESCRIPTION:END -->
