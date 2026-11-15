---
title: "IE-0001 - Birth-tile starvation as the main driver of the entry-induced prestige-merit wedge"
author: "Antigravity"
date: "2026-11-15"
tags: [idea, rejected, mechanism, spawn-dispersal, entry, interventions-v1]
project: "sciencemodels"
status: rejected
---

- **Status:** rejected
- **Date:** 2026-11-14 (captured in `CONT1` v1 / `P-0002`); 2026-11-15 (rejected after `output/reports/interventions_v1.md` Arm A)
- **Related:** `AdmiralsLog/proposals/P-0002-spatial-spawn-dispersal.md`, `output/reports/interventions_v1.md` (Result A), `AdmiralsLog/contributions/CONT1-depletion-counterforces-refills.md`, `AdmiralsLog/contributions/CONT4-inert-and-invariant-degrees-of-freedom.md` (Section 3.1)

## The idea

Because `sciencemodels/model.py:389-420` spawns every newcomer directly on its supervisor's current tile (`coords = sup.pos`), and high-prestige supervisors have already multiplicatively depleted their tile via `farm()`, we hypothesized that **mechanical birth-tile starvation** (newcomers harvesting crumbs on exhausted birth tiles) accounted for a large share of the negative `Corr_prestige_ideal_merit` and the newcomer-incumbent prestige gap under entry, and that dispersing newcomer birth tiles (`r_spawn` $\in \{\text{ring1}, \text{random}\}$) would substantially repair both.

## Why it was rejected (`output/reports/interventions_v1.md`, Result A)

Across 18 cells / 360 runs (`3 r_spawn` $\times$ `3 curiosity` $\times$ `2 landscapes` at $g = 0.1$):
1. **Dispersing birth tiles makes `Corr_prestige_ideal_merit` *more* negative at every curiosity level:**
   - $c = 0.3$: $-0.154$ (`exact`) $\to -0.187$ (`ring1`) / $-0.171$ (`random`), starvation share **$-5.7\%$**.
   - $c = 0.5$: $-0.057$ (`exact`) $\to -0.075$ (`ring1`) / $-0.145$ (`random`), starvation share **$-21.5\%$**.
   - $c = 0.7$: $+0.028$ (`exact`) $\to -0.003$ (`ring1`) / $-0.095$ (`random`), starvation share **$-53.5\%$**.
2. **Dispersing birth tiles *widens* the newcomer-incumbent prestige gap everywhere:** by $0.08\%\text{–}2.6\%$ on `mgstd5` and $5.0\%\text{–}12.0\%$ on `noisypr5`.

**Mechanical explanation:** High-prestige founders sit on or near Gaussian knowledge peaks. Spawning at `sup.pos` (`exact`) at least starts advisees near high-value terrain; dispersing their birth tile while keeping prestige-proportional supervision and visibility herding leaves founders with an unchallenged harvest monopoly on the peak while advisees herd in lower-value areas. Consequently, **$105.7\%$ to $153.5\%$ of the entry-induced quality drop is driven by prestige-proportional recruitment and evaluative herding**, not birth-tile starvation.
