---
title: "BUG-0004 - Dead incrPrestige definition swallowed by comment on agents.py distance line"
author: "Hermes Agent"
date: "2026-11-14"
tags: [bug, dead-code, agents]
project: "sciencemodels"
---

- **Status:** resolved (2026-11-14, direct fix with proof)
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

Applied 2026-11-14 (session round): the comment tail `def incrPrestige(self,
value):` was stripped from the distance-method line; the comment now ends at
"Manhattan distance on a torus".

## Resolution

Verified first-hand 2026-11-14: byte-checked edit (`sed ... | cat -A`), and
the full smoke test passes unchanged after the edit (canary green before and
after).
