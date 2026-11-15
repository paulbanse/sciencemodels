---
title: "UE-0001 - Author x step panel reconstruction for Markov C club-closure moments"
author: "Antigravity"
date: "2026-11-15"
tags: [unexplored, path-b, markov-closure, calibration-moments, panel]
project: "sciencemodels"
status: unexplored
---

- **Status:** unexplored
- **Date:** 2026-11-15
- **Related:** `output/reports/path_b_probe_v1.md`, `output/reports/path_b_probe_v2.md` (Line 275), `docs/calibration_moments.yaml`, `docs/connections_niklas.md`

## The unexplored path

Both `path_b_probe_v1` and `path_b_probe_v2` evaluated Gate 2 against the scalar co-harvest moments (`mean_team_size` $2.65$, `repeated_collab_share` $0.34$, and `link_prestige_assortativity`), leaving the **Markov C club-closure transition matrix** (top-decile prestige club retention $0.3995$, non-member entry $0.0619$; and Markov A top-10% transition $0.71 / 0.054$) uncomputed because `MyModel`'s step reporters only store aggregate scalar moments rather than an `author x step` longitudinal panel.

Reconstructing or recording the step-by-step agent prestige rank panel across windows $t \to t+\Delta t$ would allow direct testing of whether dyadic collaborator choice (`dyadic_mode != "off"`) increases elite club persistence ($P(\text{top-10\%}_{t+1} \mid \text{top-10\%}_t)$) relative to the solo-search baseline.
