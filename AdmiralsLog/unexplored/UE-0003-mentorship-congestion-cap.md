---
title: "UE-0003 - Endogenous supervision capacity and mentorship congestion caps"
author: "Antigravity"
date: "2026-11-15"
tags: [unexplored, mentorship, supervision-cap, entry, dynasty-lockin]
project: "sciencemodels"
status: unexplored
---

- **Status:** unexplored
- **Date:** 2026-11-15
- **Related:** `output/reports/interventions_v1.md` (Result A), `output/reports/turnover_v1.md`, `AdmiralsLog/contributions/CONT1-depletion-counterforces-refills.md`

## The unexplored path

`output/reports/interventions_v1.md` (Arm A) proved that $105.7\%\text{–}153.5\%$ of the entry-induced quality drop comes from prestige-proportional recruitment plus visibility herding, and `output/reports/turnover_v1.md` showed that under `age_hazard` turnover, top-10% prestige concentration surges from $26.1\%$ to **$40.9\%$** as elite dynasties replace retiring cohorts.

In `sciencemodels/model.py:389-420`, a single high-prestige incumbent can supervise an **unlimited** number of newcomers with zero mentorship congestion cost or capacity cap, while `interventions_v1` Arm B showed that pure random lotteries ($\alpha = 0$) fail to reduce welfare loss because entrants still herd toward dense visibility clusters. An unexplored intermediate institutional design is a **mentorship capacity cap or congestion penalty** (e.g., an incumbent can hold at most $K_{\max}$ active advisees within radius $1$, or supervisor roulette weights decay in the number of recent advisees $P(\text{sup}_i) \propto \text{prestige}_i / (1 + n_{\text{advisees}, i})^\beta$), preventing a single depleted tile from becoming a demographic mega-cluster.
