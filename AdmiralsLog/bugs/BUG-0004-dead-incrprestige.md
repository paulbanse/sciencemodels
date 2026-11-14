---
title: "BUG-0004 - Dead incrPrestige definition swallowed by comment on agents.py distance line"
author: "Hermes Agent"
date: "2026-11-14"
tags: [bug, dead-code, agents]
project: "sciencemodels"
---

- **Status:** open
- **Date:** 2026-11-14
- **Related:** `sciencemodels/agents.py:61-62`, `docs/conceptual_degrees_of_freedom.md` (v3.0)
- **Spawned:** none

## Symptom

On `sciencemodels/agents.py` the trailing comment `# Manhattan distance on a
torus` sits on the same line as a dead `def incrPrestige(self, value):`
header, so the definition is visually swallowed and syntactically a no-op
body carried by the following line. A reader (or a grep for prestige
methods) misses it entirely.

## Cause

Historical leftover from the pre-restructure `mesa_model.py`, preserved by
the pure-move rule; pinned by the shim/catalogue round (2026-11-14).

## Fix

Not yet applied. Candidate: delete the dead definition (it has no callers),
keeping the comment on its own line.
