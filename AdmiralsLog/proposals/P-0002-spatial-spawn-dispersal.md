---
title: "P-0002 - Spatial spawn dispersal experiment: disentangle birth-tile depletion from visibility herding"
author: "Antigravity"
date: "2026-11-14"
tags: [proposal, operative, experiment, spatial-herding, depletion, entry]
project: "sciencemodels"
status: open
---

- **Status:** open
- **Class:** operative
- **Date:** 2026-11-14
- **Spark:** `CONT1-depletion-counterforces-refills.md` (Section 2, Item 1.1); in `sciencemodels/model.py:336-337`, newcomers spawn at `coords = sup.pos`, fusing prestige-proportional supervision with birth-tile starvation.
- **Outcome target:** a ruling on whether to run the 18-cell spawn-dispersal decomposition sweep (`r_spawn` in `{0, 1, random}`).
- **Ruling:** pending
- **Spawned:** none

## The proposal

Run a clean 18-cell experimental probe (`scripts/interventions/`, 20 seeds per cell, $H = 200$) varying a single birth-placement parameter $r_{\text{spawn}} \in \{0 \text{ (exact supervisor tile)}, 1 \text{ (von Neumann ring around supervisor)}, \text{uniform random grid tile}\}$ across curiosity $c \in \{0.3, 0.5, 0.7\}$ and landscape $\in \{\text{mgstd5}, \text{noisypr5}\}$ at entry rate $g = 0.1$.

## The stated reason

In Waves 1–3 (`output/reports/sensitivity_map_v1.md`, `output/reports/career_flows_v1.md`), activating prestige-proportional entry flips `Corr_prestige_ideal_merit` negative and prevents newcomers from ever closing the prestige gap with incumbents. However, because `model.py:336` places every newcomer directly on `sup.pos` (which the prestigious supervisor has already multiplicatively depleted via `farm()`), two distinct mechanisms are confounded:

1. **Mechanical birth-tile starvation:** Newcomers harvest crumbs on an exhausted birth tile while their supervisor retains accumulated historical prestige.
2. **Social/attentional herding:** Visibility-weighted movement ($1-c$) pulls agents toward crowded prestige clusters even when they do not start on the same tile.

Varying $r_{\text{spawn}}$ separates the two mechanisms quantitatively without altering any other model rule.

## Experimental design & cost

- **Grid:** 3 `r_spawn` levels (`0`, `1`, `random`) $\times$ 3 curiosities (`0.3`, `0.5`, `0.7`) $\times$ 2 landscapes (`mgstd5`, `noisypr5`) = 18 cells $\times$ 20 seeds = **360 runs** (~5 minutes at `n_jobs=3`).
- **Implementation:** Driver-side wrapper with a bit-identity proof that `r_spawn = 0` matches `MyModel` to `0.0` diff.
- **Deliverable:** Decomposition of the entry-induced ideal-merit correlation drop and newcomer-incumbent prestige gap into birth-tile starvation vs. attentional herding.

Do we proceed with this 360-run spawn-dispersal probe?
