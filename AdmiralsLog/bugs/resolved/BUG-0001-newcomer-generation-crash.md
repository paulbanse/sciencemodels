---
title: "BUG-0001 - Newcomer generation crashed for agent_generation_time > 0"
author: "Hermes Agent"
date: "2026-11-13"
tags: [bug, demographics, supervisor-entry, mesa-model]
project: "sciencemodels"
---

- **Status:** resolved
- **Date:** 2026-11-13
- **Related:** ADR-0001 (field-execution hygiene), `mesa_model.py`, `docs/conceptual_degrees_of_freedom.md` (demographics level)
- **Spawned:** none (the fix landed inside the W1 measurement-hygiene round, 2026-11-13)

## Symptom

Any model constructed with `agent_generation_time > 0` died at construction
with `TypeError: 'numpy.float64' object cannot be interpreted as an integer`
in `__generate_new_agents`. The supervisor-entry channel --- the literal
collaborator-choice mechanism --- was not merely switched off by
configuration; it was unrunnable.

## Cause

`agents_to_generate = np.floor(steps * t) - np.floor((steps - 1) * t)` is an
`np.float64`; `range(agents_to_generate)` requires an int
(`mesa_model.py`, `__generate_new_agents`).

## Fix

`range(int(agents_to_generate))` --- one-token repair inside the W1 hygiene
round (2026-11-13, `mesa_model.py:537`). No existing configuration changes
behavior: `agent_generation_time = 0` returns before the line, and positive
values previously crashed.

## Resolution

Verified first-hand by the dispatching session on 2026-11-13: seeded run
(`agent_generation_time=0.1`, `step_limit=20`, `agent_seed=1`) completes
end-to-end with 3 newcomers spawned at supervisors (6 scientists total),
all 6 curiosities distinct via `agent_generation_function`; the same-seed
baseline remains bit-identical across reruns (report dict compare). With
this fix the supervisor-proportional entry of `ADR-0001`'s collaborator
channel is executable for the first time.
