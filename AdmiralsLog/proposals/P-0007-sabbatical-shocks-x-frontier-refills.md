---
title: "P-0007 - Sabbatical shocks x frontier paradigm refills: unlocking unclaimed frontier paradigms (Path C v2)"
author: "Antigravity"
date: "2026-11-15"
tags: [proposal, operative, experiment, interventions, sabbatical-shocks, frontier-refills, path-c]
project: "sciencemodels"
status: open
---

- **Status:** open
- **Class:** operative
- **Date:** 2026-11-15
- **Spark:** `output/reports/interventions_v1.md` (Results B & C) and `CONT1-depletion-counterforces-refills.md` (Sections 3–4): in Arm B, sabbatical relocation shocks (`p_reloc = 0.02`) broke spatial lock-in and cut the high-visibility welfare penalty by $40\%$; in Arm C, `frontier` paradigm refills restored positive quality alignment ($c_{\text{ideal}} = +0.078$) but accumulated **unclaimed** (`mean_grid_knowledge = 0.1895` vs. $0.0006$ baseline) because visibility-locked herds never left depleted clusters.
- **Outcome target:** a ruling on whether to run a 36-cell interaction probe crossing `p_reloc` with supply-matched `refill_mode`.
- **Ruling:** pending
- **Spawned:** none

## The proposal

Run a 36-cell interaction experiment (`scripts/interventions/run_interventions_v2.py`, 20 seeds per cell, $H = 200$) crossing:
1. **Sabbatical / Maverick Relocation Rate (`p_reloc` $\in \{0.00, 0.02, 0.05\}$):** Triggered either by tile depletion ($k < 0.025$, matching v1) or by spatial stagnation (`steps_stationary >= 10`, `IE-0005`).
2. **Supply-Matched Paradigm Refill Regime (`refill_mode` $\in \{\text{off}, \text{occupied}, \text{frontier}\}$):** With a fixed exogenous knowledge arrival budget per step (resolving the v1 supply caveat noted in [interventions_v1.md](file:///Users/joko/Git/sciencemodels/output/reports/interventions_v1.md#L159-L183) where `occupied` refills were consumed immediately while `frontier` refills accumulated).
3. **Curiosity / Visibility Weight ($c \in \{0.3, 0.5\}$)** and **Landscape ($\{\text{mgstd5}, \text{noisypr5}\}$)** at entry rate $g = 0.1$.

## The stated reason

In `interventions_v1`, Arms B and C were run separately:
- Arm C showed that when new scientific paradigms open on unoccupied frontier tiles (`frontier`), a community governed by visibility-weighted movement leaves them sitting on the table (`mean_grid_knowledge` rises $315\times$ while `explored` drops below the no-refill baseline).
- Arm B showed that a $2\%$ sabbatical relocation shock (`p_reloc = 0.02`) is the only lever that breaks spatial status-quo lock-in.

Crossing `p_reloc` $\times$ `refill_mode` under a supply-matched refill budget tests the central pluralist policy hypothesis of the project (Dobusch & Kapeller 2009, 2012; Zollman 2010): **does transient maverick mobility allow a prestige-stratified scientific community to discover and harvest emerging frontier paradigms instead of stagnating on exhausted orthodoxy?**

## Experimental design & cost

- **Grid:** 3 `p_reloc` (`0.00`, `0.02`, `0.05`) $\times$ 3 `refill_mode` (`off`, `occupied`, `frontier`) $\times$ 2 curiosities (`0.3`, `0.5`) $\times$ 2 landscapes (`mgstd5`, `noisypr5`) = 36 cells $\times$ 20 seeds = **720 runs** at $H = 200$ (~7 minutes at `n_jobs=3`).
- **Deliverable:** `output/reports/interventions_v2.md` + frontier-harvest efficiency and welfare-recovery figures.

Do we proceed with the 36-cell sabbatical $\times$ frontier-refill interaction probe?
