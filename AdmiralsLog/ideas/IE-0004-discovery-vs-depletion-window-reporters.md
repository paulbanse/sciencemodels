---
title: "IE-0004 - Discovery-phase vs. depletion-phase window reporters (early100 vs. late100)"
author: "Antigravity"
date: "2026-11-15"
tags: [idea, metrics, time-windows, discovery-phase, depletion-phase, turnover-v1]
project: "sciencemodels"
status: idea
---

- **Status:** idea
- **Date:** 2026-11-15
- **Related:** `output/reports/turnover_v1.md` (Lines 138–171, 282–287), `AdmiralsLog/contributions/CONT5-generational-turnover-and-paradigm-lockin.md`, `AdmiralsLog/proposals/P-0009-core-turnover-and-steady-state-interventions.md`

## The idea

`output/reports/turnover_v1.md` uncovered a fundamental sign split between whole-run time averages and terminal steady states across all three demographic regimes (`none`, `fixed_span`, `age_hazard`):
- **Whole-run time mean `Corr_prestige_ideal_merit` is positive in every mode:** $+0.1141$ (`none`), $+0.1042$ (`fixed_span`), $+0.1112$ (`age_hazard`).
- **Final-step `Corr_prestige_ideal_merit` is negative in every mode:** $-0.0860$ (`none`), $-0.2401$ (`fixed_span`), $-0.5678$ (`age_hazard`).

**Why:** Every run passes through two distinct dynamical regimes:
1. **Discovery Phase ($t \in [1, 100]$):** Founders and early entrants discover virgin Gaussian peaks; prestige accrues to genuine pioneers and positively tracks ideal merit.
2. **Depletion / Herding Phase ($t \in [H-100, H]$):** Initial peaks are exhausted; prestige hierarchies lock in around depleted clusters and anti-track ideal merit.

Reporting only a whole-run `time_mean` mixes these two opposite phases into a diluted positive number, while reporting only the final step (`fin`) is noisy at a single step. Adding explicit **windowed reporters** — `c_ideal_early100` (mean over steps $1\text{–}100$) and `c_ideal_late100` (mean over the final 100 steps, matching `turnover_v1`'s `last100` convention) — captures both regimes cleanly without single-step variance.
