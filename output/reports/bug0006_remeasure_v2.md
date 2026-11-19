---
title: "BUG-0006 Post-Fix Re-Measure & Staggered Turnover Report"
author: "Coordination Session (bug0006_remeasure_v2 + turnover_v2)"
version: 1
date: 2026-11-19
tags: [bug-0006, remeasure, turnover, staggered-cohorts, proxy-failure, not-calibrated]
project: sciencemodels
revision-vehicle: strikethrough
revision-renders: "md, pdf, html"
---

**Version 1** (November 2026)

> **MODEL-INTERNAL, NOT CALIBRATED** (owner ruling 2026-11-13). Every number below
> is computed by `scripts/bug0006/analyse_bug0006.py` from
> `data/bug0006_remeasure_v2/` (1,720 runs) and `data/turnover_v2/` (1,520 runs)
> and round-trip checked against `data/bug0006_remeasure_v2/tables/*.csv`.

# 1. Executive Summary: What BUG-0006 Was, and What Survives

## 1.1 Root Cause & Fix
`MyModel.end_loop_update()` runs `self.updateIdealMerit()` (line 516) **before** `self.__generate_new_agents()` (line 517) and `self.datacollector.collect(self)` (line 554). Previously, `Scientist.__init__` preset `self.ideal_Merit = 0` (`sciencemodels/agents.py:48`). Because `ideal_Merit <= 0` (`0` = omniscient planner optimum), every newborn spawned on step $t$ entered `datacollector.collect()` with `(prestige = 0.0, ideal_Merit = 0.0)` — artificially pairing the lowest possible prestige with the highest possible merit on every birth step (including steps 200 and 400).

**Fix (`BUG-0006`):** `sciencemodels/agents.py:48` now initializes `self.ideal_Merit = np.nan`, and `sciencemodels/metrics.py:308-316` (`_finite_pair_vectors`) filters `correlation_reporter` to finite `(prestige, merit)` pairs. Dynamics, RNG draws, harvest, local merit, and prestige concentration are unchanged.

## 1.2 Pristine-Channel Invariance Proofs
Across **all 3,240 re-measured runs**, every dynamics and non-ideal channel reproduces the pre-fix archives to floating-point precision:
- **`T1` `g = 0.0` control arm (360 runs):** max absolute difference across all 9 metrics (including `c_ideal_fin` and `c_ideal_tm`) = `4.12e-15`.
- **`W_v05` `g = 0.0` control arm (400 runs):** max absolute difference across all 9 metrics = `9.58e-15`.
- **Non-ideal channels under `g > 0` (`T1`, `W_v05`, `M2_xi`, `turnover_v2` `none`/`fixed_span`/`age_hazard_sync`):** `c_local_fin`, `c_local_tm`, `top10_fin`, `harvested_fin`, `ideal_loss_cum`, `team_tm`, `links_tm`, and `gap_recent` all differ by `<= 1e-11` (`0.0` modulo IEEE roundoff).

## 1.3 What Changes vs. What Survives (Headline Scientific Verdict)
1. **The massive negative birth-step spike at step 200/400 was the `BUG-0006` artifact:**
   - In `T1` (`g = 0.1`), pooled terminal `c_ideal_fin` shifts from **`-0.1438`** (pre-fix, `17/18` negative cells) to **`+0.0327`** (post-fix, **`5/18`** negative cells).
   - **Important landscape split in `T1` post-fix:** on clean `mgstd5`, `M1` `c_ideal_fin` is **`+0.0638`** (`0/9` cells negative; time-mean **`+0.1172`**, down sharply from `M0`'s `+0.2200` / `+0.4101`), whereas on noisy `noisypr5`, `M1` collapses to **`+0.0017`** pooled with **`5/9` cells negative** (vs time-mean `+0.3549`).
   - Across `T1` (`g = 0.1`), the time-mean `c_ideal_tm` moves modestly upward (`+0.2130` -> `+0.2360`), confirming a **1.74x** collapse in time-mean merit tracking from `M0` (`+0.4101`) to `M1` (`+0.2360`).
2. **Turnover (`turnover_v2`) separates cleanly once `BUG-0006` + cohort synchronization are removed:**
   - Under `age_hazard_sync`, removing `BUG-0006` alone shifts step-400 `c_ideal_fin` from **`-0.5678`** to **`+0.0173`** (a paired shift of **`+0.5851`** caused purely by 21 uncomputed `(0, 0)` replacements at $t=400$).
   - Even with `BUG-0006` fixed, synchronized replacement (`age_hazard_sync`) still creates a step-200/400 concentration spike (`top10(400) = 0.4086` vs `top10(399) = 0.1801`).
   - **Staggering founder retirements (`age_hazard_staggered`) eliminates both artifacts:** `c_ideal(400) = +0.0106`, `c_ideal(301-400) = +0.0128`, `top10(400) = 0.2700` (vs `none` `0.2605` and `fixed_span` `0.2327`), while **preserving the arrival-gap shrinkage** (`gap_recent`: `-0.01362` -> `-0.00612` [`-55.0%`] under `fixed_span`, `-0.00233` [`-82.9%`] under `age_hazard_staggered`, negative in `1440/1440` runs).
3. **Every core structural finding survives intact:**
   - The `M0` -> `M1` Discovery-to-Depletion lifecycle collapse survives across all 4 turnover modes (pooled trajectory mean `c_ideal`: `+0.4176` in Discovery $t \le 80$ vs `+0.0310` in Depletion $t \ge 150$, `+0.0234` in $t \ge 301$).
   - The `M2` hierarchy at $\xi = 0.5$ (`M0 +0.2881 > M2_lambda1 +0.1970 > M1 +0.1324 > M2_lambda0 +0.0603` on time-mean `c_ideal_tm`) and the Stage-2 harvest/alignment collapse at $\xi = 0.9$ (`M2_lambda1` harvest `0.764` -> `0.389`, `c_ideal_tm` -> `+0.0434`) survive post-fix.

---

# 2. `T1` Sensitivity Sweep (`data/bug0006_remeasure_v2/t1_runs.csv`, 720 runs)

| Scope | Metric | n | Pre Mean | Post Mean | Delta | CI lo | CI hi | Max |Diff| | Pre <0 Cells | Post <0 Cells |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| g=0.0 (control) | c_ideal_fin | 360 | +0.2200 | +0.2200 | -0.0000 | -0.0000 | +0.0000 | 2.13e-15 | 2 | 2 |
| g=0.0 (control) | c_ideal_tm | 360 | +0.4101 | +0.4101 | +0.0000 | -0.0000 | +0.0000 | 3.33e-16 | 0 | 0 |
| g=0.0 (control) | c_local_fin | 360 | +0.2499 | +0.2499 | -0.0000 | -0.0000 | +0.0000 | 4.12e-15 | 2 | 2 |
| g=0.0 (control) | wedge_local_minus_ideal_fin | 360 | +0.0299 | +0.0299 | +0.0000 | -0.0000 | +0.0000 | 4.12e-15 | 6 | 6 |
| g=0.0 (control) | top10_fin | 360 | +0.1353 | +0.1353 | -0.0000 | -0.0000 | +0.0000 | 1.11e-16 | 0 | 0 |
| g=0.0 (control) | harvested_fin | 360 | +0.4570 | +0.4570 | -0.0000 | -0.0000 | -0.0000 | 2.78e-16 | 0 | 0 |
| g=0.1 (all) | c_ideal_fin | 360 | -0.1438 | +0.0327 | +0.1765 | +0.1674 | +0.1857 | 5.10e-01 | 17 | 5 |
| g=0.1 (all) | c_ideal_tm | 360 | +0.2130 | +0.2360 | +0.0231 | +0.0224 | +0.0237 | 4.22e-02 | 0 | 0 |
| g=0.1 (all) | c_local_fin | 360 | +0.0383 | +0.0383 | +0.0000 | -0.0000 | +0.0000 | 4.02e-16 | 5 | 5 |
| g=0.1 (all) | wedge_local_minus_ideal_fin | 360 | +0.1821 | +0.0056 | -0.1765 | -0.1858 | -0.1672 | 5.10e-01 | 0 | 8 |
| g=0.1 (all) | top10_fin | 360 | +0.2282 | +0.2282 | -0.0000 | -0.0000 | +0.0000 | 1.11e-16 | 0 | 0 |
| g=0.1 (all) | harvested_fin | 360 | +0.5543 | +0.5543 | -0.0000 | -0.0000 | -0.0000 | 2.22e-16 | 0 | 0 |
| g=0.1 (mgstd5) | c_ideal_fin | 180 | -0.0849 | +0.0638 | +0.1486 | +0.1363 | +0.1618 | 4.75e-01 | 8 | 0 |
| g=0.1 (mgstd5) | c_ideal_tm | 180 | +0.0920 | +0.1172 | +0.0251 | +0.0241 | +0.0263 | 4.22e-02 | 0 | 0 |
| g=0.1 (mgstd5) | c_local_fin | 180 | +0.0562 | +0.0562 | +0.0000 | -0.0000 | +0.0000 | 4.02e-16 | 1 | 1 |
| g=0.1 (mgstd5) | wedge_local_minus_ideal_fin | 180 | +0.1410 | -0.0076 | -0.1486 | -0.1617 | -0.1358 | 4.75e-01 | 0 | 6 |
| g=0.1 (mgstd5) | top10_fin | 180 | +0.2123 | +0.2123 | -0.0000 | -0.0000 | +0.0000 | 1.11e-16 | 0 | 0 |
| g=0.1 (mgstd5) | harvested_fin | 180 | +0.7545 | +0.7545 | -0.0000 | -0.0000 | +0.0000 | 1.11e-16 | 0 | 0 |
| g=0.1 (noisypr5) | c_ideal_fin | 180 | -0.2026 | +0.0017 | +0.2043 | +0.1929 | +0.2159 | 5.10e-01 | 9 | 5 |
| g=0.1 (noisypr5) | c_ideal_tm | 180 | +0.3339 | +0.3549 | +0.0210 | +0.0205 | +0.0216 | 3.19e-02 | 0 | 0 |
| g=0.1 (noisypr5) | c_local_fin | 180 | +0.0204 | +0.0204 | -0.0000 | -0.0000 | +0.0000 | 2.01e-16 | 4 | 4 |
| g=0.1 (noisypr5) | wedge_local_minus_ideal_fin | 180 | +0.2231 | +0.0188 | -0.2043 | -0.2161 | -0.1926 | 5.10e-01 | 0 | 2 |
| g=0.1 (noisypr5) | top10_fin | 180 | +0.2442 | +0.2442 | -0.0000 | -0.0000 | +0.0000 | 1.11e-16 | 0 | 0 |
| g=0.1 (noisypr5) | harvested_fin | 180 | +0.3541 | +0.3541 | -0.0000 | -0.0000 | -0.0000 | 2.22e-16 | 0 | 0 |

---

# 3. `W_v05` Welfare Sweep & $\xi^*$ Kink (`data/bug0006_remeasure_v2/w_v05_runs.csv`, 800 runs)

| Scope | Metric | n | Pre Mean | Post Mean | Delta | CI lo | CI hi | Max |Diff| |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| g=0.0 (control) | c_ideal_fin | 400 | +0.3018 | +0.3018 | -0.0000 | -0.0000 | -0.0000 | 4.79e-15 |
| g=0.0 (control) | c_ideal_tm | 400 | +0.3752 | +0.3752 | -0.0000 | -0.0000 | -0.0000 | 5.00e-16 |
| g=0.0 (control) | c_local_fin | 400 | +0.2952 | +0.2952 | -0.0000 | -0.0000 | +0.0000 | 4.79e-15 |
| g=0.0 (control) | wedge_local_minus_ideal_fin | 400 | -0.0065 | -0.0065 | +0.0000 | -0.0000 | +0.0000 | 9.58e-15 |
| g=0.0 (control) | top10_fin | 400 | +0.1208 | +0.1208 | -0.0000 | -0.0000 | +0.0000 | 1.11e-16 |
| g=0.0 (control) | harvested_fin | 400 | +0.4679 | +0.4679 | -0.0000 | -0.0000 | -0.0000 | 2.22e-16 |
| g=0.0 (control) | ideal_loss_cum | 400 | +0.5045 | +0.5045 | +0.0000 | +0.0000 | +0.0000 | 4.44e-16 |
| g=0.1 (all) | c_ideal_fin | 400 | -0.0784 | +0.0866 | +0.1650 | +0.1555 | +0.1745 | 4.73e-01 |
| g=0.1 (all) | c_ideal_tm | 400 | +0.1479 | +0.1738 | +0.0260 | +0.0251 | +0.0269 | 4.87e-02 |
| g=0.1 (all) | c_local_fin | 400 | +0.0777 | +0.0777 | +0.0000 | -0.0000 | +0.0000 | 8.05e-16 |
| g=0.1 (all) | wedge_local_minus_ideal_fin | 400 | +0.1561 | -0.0089 | -0.1650 | -0.1746 | -0.1556 | 4.73e-01 |
| g=0.1 (all) | top10_fin | 400 | +0.2117 | +0.2117 | -0.0000 | -0.0000 | +0.0000 | 1.11e-16 |
| g=0.1 (all) | harvested_fin | 400 | +0.5704 | +0.5704 | -0.0000 | -0.0000 | -0.0000 | 2.78e-16 |
| g=0.1 (all) | ideal_loss_cum | 400 | +0.6701 | +0.6701 | +0.0000 | +0.0000 | +0.0000 | 8.88e-16 |
| g=0.1 (mgstd5) | c_ideal_fin | 200 | -0.0990 | +0.0620 | +0.1610 | +0.1478 | +0.1742 | 4.73e-01 |
| g=0.1 (mgstd5) | c_ideal_tm | 200 | +0.0845 | +0.1099 | +0.0254 | +0.0241 | +0.0267 | 4.15e-02 |
| g=0.1 (mgstd5) | c_local_fin | 200 | +0.0595 | +0.0595 | +0.0000 | -0.0000 | +0.0000 | 8.05e-16 |
| g=0.1 (mgstd5) | wedge_local_minus_ideal_fin | 200 | +0.1584 | -0.0026 | -0.1610 | -0.1743 | -0.1478 | 4.73e-01 |
| g=0.1 (mgstd5) | top10_fin | 200 | +0.2110 | +0.2110 | +0.0000 | -0.0000 | +0.0000 | 1.11e-16 |
| g=0.1 (mgstd5) | harvested_fin | 200 | +0.6921 | +0.6921 | -0.0000 | -0.0000 | +0.0000 | 2.78e-16 |
| g=0.1 (mgstd5) | ideal_loss_cum | 200 | +0.8628 | +0.8628 | +0.0000 | -0.0000 | +0.0000 | 8.88e-16 |
| g=0.1 (mg6) | c_ideal_fin | 200 | -0.0579 | +0.1111 | +0.1690 | +0.1554 | +0.1822 | 4.25e-01 |
| g=0.1 (mg6) | c_ideal_tm | 200 | +0.2113 | +0.2378 | +0.0265 | +0.0253 | +0.0278 | 4.87e-02 |
| g=0.1 (mg6) | c_local_fin | 200 | +0.0959 | +0.0959 | -0.0000 | -0.0000 | +0.0000 | 5.00e-16 |
| g=0.1 (mg6) | wedge_local_minus_ideal_fin | 200 | +0.1538 | -0.0152 | -0.1690 | -0.1827 | -0.1556 | 4.25e-01 |
| g=0.1 (mg6) | top10_fin | 200 | +0.2124 | +0.2124 | -0.0000 | -0.0000 | +0.0000 | 1.11e-16 |
| g=0.1 (mg6) | harvested_fin | 200 | +0.4487 | +0.4487 | -0.0000 | -0.0000 | -0.0000 | 2.78e-16 |
| g=0.1 (mg6) | ideal_loss_cum | 200 | +0.4773 | +0.4773 | +0.0000 | +0.0000 | +0.0000 | 4.44e-16 |

## 3.1 Breakpoint $\xi^*$ Stability (Pre vs. Post BUG-0006)

| Arm | Landscape | Entry | Vis Reward | xi* (terminal sum) | xi* (cum loss) | n |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| pre_BUG0006 | mg6 | 0.0 | False | 0.6 | 0.6 | 100 |
| pre_BUG0006 | mg6 | 0.0 | True | 0.6 | 0.6 | 100 |
| pre_BUG0006 | mg6 | 0.1 | False | 0.6 | 0.6 | 100 |
| pre_BUG0006 | mg6 | 0.1 | True | 0.6 | 0.6 | 100 |
| pre_BUG0006 | mgstd5 | 0.0 | False | 0.6 | 0.6 | 100 |
| pre_BUG0006 | mgstd5 | 0.0 | True | 0.6 | 0.6 | 100 |
| pre_BUG0006 | mgstd5 | 0.1 | False | 0.8 | 0.6 | 100 |
| pre_BUG0006 | mgstd5 | 0.1 | True | 0.8 | 0.6 | 100 |
| post_BUG0006 | mg6 | 0.0 | False | 0.6 | 0.6 | 100 |
| post_BUG0006 | mg6 | 0.0 | True | 0.6 | 0.6 | 100 |
| post_BUG0006 | mg6 | 0.1 | False | 0.6 | 0.6 | 100 |
| post_BUG0006 | mg6 | 0.1 | True | 0.6 | 0.6 | 100 |
| post_BUG0006 | mgstd5 | 0.0 | False | 0.6 | 0.6 | 100 |
| post_BUG0006 | mgstd5 | 0.0 | True | 0.6 | 0.6 | 100 |
| post_BUG0006 | mgstd5 | 0.1 | False | 0.8 | 0.6 | 100 |
| post_BUG0006 | mgstd5 | 0.1 | True | 0.8 | 0.6 | 100 |

---

# 4. `M2_xi` Dyadic Extension (`data/bug0006_remeasure_v2/m2_xi_runs.csv`, 200 runs)

| Scope | Metric | n | Pre Mean | Post Mean | Delta | CI lo | CI hi | Max |Diff| |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | c_ideal_tm | 200 | +0.0795 | +0.1043 | +0.0248 | +0.0234 | +0.0260 | 4.32e-02 |
| all | c_local_tm | 200 | +0.1004 | +0.1004 | +0.0000 | -0.0000 | +0.0000 | 2.22e-16 |
| all | wedge_local_minus_ideal_tm | 200 | +0.0209 | -0.0039 | -0.0248 | -0.0261 | -0.0234 | 4.32e-02 |
| all | team_tm | 200 | +9.6495 | +9.6495 | +0.0000 | +0.0000 | +0.0000 | 0.00e+00 |
| all | links_tm | 200 | +12.6348 | +12.6348 | +0.0000 | +0.0000 | +0.0000 | 0.00e+00 |
| all | top10_fin | 200 | +0.2108 | +0.2108 | -0.0000 | -0.0000 | +0.0000 | 1.11e-16 |
| all | harvested_fin | 200 | +0.6887 | +0.6887 | -0.0000 | -0.0000 | +0.0000 | 1.67e-16 |
| lambda=0.0 | c_ideal_tm | 100 | +0.0235 | +0.0468 | +0.0234 | +0.0214 | +0.0254 | 4.19e-02 |
| lambda=0.0 | c_local_tm | 100 | +0.0842 | +0.0842 | -0.0000 | -0.0000 | +0.0000 | 2.22e-16 |
| lambda=0.0 | wedge_local_minus_ideal_tm | 100 | +0.0607 | +0.0374 | -0.0234 | -0.0254 | -0.0214 | 4.19e-02 |
| lambda=0.0 | team_tm | 100 | +10.3366 | +10.3366 | +0.0000 | +0.0000 | +0.0000 | 0.00e+00 |
| lambda=0.0 | links_tm | 100 | +2.2492 | +2.2492 | +0.0000 | +0.0000 | +0.0000 | 0.00e+00 |
| lambda=0.0 | top10_fin | 100 | +0.2075 | +0.2075 | -0.0000 | -0.0000 | +0.0000 | 1.11e-16 |
| lambda=0.0 | harvested_fin | 100 | +0.6928 | +0.6928 | -0.0000 | -0.0000 | +0.0000 | 1.67e-16 |
| lambda=1.0 | c_ideal_tm | 100 | +0.1355 | +0.1617 | +0.0261 | +0.0245 | +0.0277 | 4.32e-02 |
| lambda=1.0 | c_local_tm | 100 | +0.1166 | +0.1166 | +0.0000 | -0.0000 | +0.0000 | 1.67e-16 |
| lambda=1.0 | wedge_local_minus_ideal_tm | 100 | -0.0190 | -0.0451 | -0.0261 | -0.0277 | -0.0245 | 4.32e-02 |
| lambda=1.0 | team_tm | 100 | +8.9624 | +8.9624 | +0.0000 | +0.0000 | +0.0000 | 0.00e+00 |
| lambda=1.0 | links_tm | 100 | +23.0204 | +23.0204 | +0.0000 | +0.0000 | +0.0000 | 0.00e+00 |
| lambda=1.0 | top10_fin | 100 | +0.2142 | +0.2142 | +0.0000 | -0.0000 | +0.0000 | 1.11e-16 |
| lambda=1.0 | harvested_fin | 100 | +0.6846 | +0.6846 | +0.0000 | -0.0000 | +0.0000 | 1.11e-16 |

---

# 5. Staggered Turnover Decomposition (`data/turnover_v2/`, 1,520 runs)

## 5.1 Main Grid Comparison Across All Four Exit Modes (360 runs/mode, $H = 400$)

| Mode | N(400) | c_ideal(400) | c_ideal(399) | c_ideal(301-400) | c_ideal(tm) | wedge_fin | top10(400) | top10(399) | top10(301-400) | gap_recent | gap_ratio | harvested |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 67.7 | +0.0037 | +0.0101 | +0.0153 | +0.1319 | +0.0013 | 0.2605 | 0.2272 | 0.2311 | -0.01362 | -0.986 | 0.7869 |
| fixed_span | 20.0 | +0.0390 | +0.0251 | +0.0349 | +0.1362 | -0.0227 | 0.2327 | 0.2262 | 0.2244 | -0.00612 | -0.918 | 0.6106 |
| age_hazard_sync | 67.7 | +0.0173 | +0.0198 | +0.0305 | +0.1345 | -0.0018 | 0.4086 | 0.1801 | 0.2077 | -0.00281 | -0.947 | 0.7875 |
| age_hazard_staggered | 67.7 | +0.0106 | +0.0094 | +0.0128 | +0.1010 | -0.0051 | 0.2700 | 0.2311 | 0.2647 | -0.00233 | -0.937 | 0.7865 |

## 5.2 Entry-Rate Gradient ($g \in {0.05, 0.10, 0.20}$)

| Mode | Entry g | c_ideal(400) | c_ideal(399) | c_ideal(301-400) | c_ideal(tm) | gap_recent | top10(400) | top10(301-400) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| age_hazard_staggered | 0.05 | +0.0047 | +0.0044 | +0.0155 | +0.1118 | -0.00295 | 0.2728 | 0.2474 |
| age_hazard_staggered | 0.10 | +0.0034 | +0.0175 | +0.0145 | +0.1090 | -0.00237 | 0.2663 | 0.2678 |
| age_hazard_staggered | 0.20 | +0.0237 | +0.0062 | +0.0084 | +0.0821 | -0.00168 | 0.2708 | 0.2788 |
| age_hazard_sync | 0.05 | +0.0222 | +0.0464 | +0.0411 | +0.1640 | -0.00377 | 0.5104 | 0.1757 |
| age_hazard_sync | 0.10 | +0.0372 | +0.0039 | +0.0291 | +0.1320 | -0.00281 | 0.3835 | 0.2087 |
| age_hazard_sync | 0.20 | -0.0076 | +0.0093 | +0.0212 | +0.1077 | -0.00187 | 0.3318 | 0.2387 |
| fixed_span | 0.05 | +0.0027 | -0.0095 | +0.0032 | +0.1404 | -0.01243 | 0.2698 | 0.2485 |
| fixed_span | 0.10 | +0.0528 | +0.0422 | +0.0454 | +0.1366 | -0.00408 | 0.2176 | 0.2189 |
| fixed_span | 0.20 | +0.0614 | +0.0425 | +0.0562 | +0.1314 | -0.00185 | 0.2107 | 0.2058 |
| none | 0.05 | +0.0193 | +0.0332 | +0.0267 | +0.1639 | -0.01792 | 0.2271 | 0.1902 |
| none | 0.10 | -0.0014 | +0.0061 | +0.0180 | +0.1284 | -0.01352 | 0.2554 | 0.2306 |
| none | 0.20 | -0.0068 | -0.0091 | +0.0012 | +0.1035 | -0.00941 | 0.2991 | 0.2723 |

## 5.3 Paired Seed-Matched Deltas (`turnover_v2` vs. `turnover_v1`)

| Contrast | Metric | Pairs | Base Mean | Comp Mean | Delta | CI lo | CI hi | Max |Diff| |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none_v2_vs_none_v1 | c_ideal_fin | 360 | -0.0860 | +0.0037 | +0.0897 | +0.0816 | +0.0984 | 4.69e-01 |
| none_v2_vs_none_v1 | c_ideal_tm | 360 | +0.1141 | +0.1319 | +0.0178 | +0.0172 | +0.0185 | 3.81e-02 |
| none_v2_vs_none_v1 | c_local_fin | 360 | +0.0050 | +0.0050 | +0.0000 | -0.0000 | +0.0000 | 5.97e-16 |
| none_v2_vs_none_v1 | top10_fin | 360 | +0.2605 | +0.2605 | -0.0000 | -0.0000 | +0.0000 | 1.11e-16 |
| none_v2_vs_none_v1 | harvested_fin | 360 | +0.7869 | +0.7869 | -0.0000 | -0.0000 | +0.0000 | 2.22e-16 |
| none_v2_vs_none_v1 | gap_recent_mean_last100 | 360 | -0.0136 | -0.0136 | -0.0000 | -0.0000 | +0.0000 | 1.01e-16 |
| fixed_span_v2_vs_fixed_span_v1 | c_ideal_fin | 360 | -0.2401 | +0.0390 | +0.2791 | +0.2646 | +0.2936 | 8.56e-01 |
| fixed_span_v2_vs_fixed_span_v1 | c_ideal_tm | 360 | +0.1042 | +0.1362 | +0.0320 | +0.0301 | +0.0339 | 7.55e-02 |
| fixed_span_v2_vs_fixed_span_v1 | c_local_fin | 360 | +0.0163 | +0.0163 | -0.0000 | -0.0000 | +0.0000 | 6.11e-16 |
| fixed_span_v2_vs_fixed_span_v1 | top10_fin | 360 | +0.2327 | +0.2327 | -0.0000 | -0.0000 | -0.0000 | 1.11e-16 |
| fixed_span_v2_vs_fixed_span_v1 | harvested_fin | 360 | +0.6106 | +0.6106 | -0.0000 | -0.0000 | -0.0000 | 2.22e-16 |
| fixed_span_v2_vs_fixed_span_v1 | gap_recent_mean_last100 | 360 | -0.0061 | -0.0061 | -0.0000 | -0.0000 | +0.0000 | 9.97e-17 |
| age_hazard_sync_v2_vs_age_hazard_v1 | c_ideal_fin | 360 | -0.5678 | +0.0173 | +0.5851 | +0.5647 | +0.6062 | 1.23e+00 |
| age_hazard_sync_v2_vs_age_hazard_v1 | c_ideal_tm | 360 | +0.1112 | +0.1345 | +0.0233 | +0.0226 | +0.0241 | 4.41e-02 |
| age_hazard_sync_v2_vs_age_hazard_v1 | c_local_fin | 360 | +0.0154 | +0.0154 | -0.0000 | -0.0000 | -0.0000 | 4.16e-16 |
| age_hazard_sync_v2_vs_age_hazard_v1 | top10_fin | 360 | +0.4086 | +0.4086 | -0.0000 | -0.0000 | +0.0000 | 2.22e-16 |
| age_hazard_sync_v2_vs_age_hazard_v1 | harvested_fin | 360 | +0.7875 | +0.7875 | -0.0000 | -0.0000 | -0.0000 | 1.11e-16 |
| age_hazard_sync_v2_vs_age_hazard_v1 | gap_recent_mean_last100 | 360 | -0.0028 | -0.0028 | +0.0000 | +0.0000 | +0.0000 | 0.00e+00 |
| age_hazard_staggered_v2_vs_age_hazard_v1 | c_ideal_fin | 360 | -0.5678 | +0.0106 | +0.5784 | +0.5564 | +0.6003 | 1.16e+00 |
| age_hazard_staggered_v2_vs_age_hazard_v1 | c_ideal_tm | 360 | +0.1112 | +0.1010 | -0.0102 | -0.0176 | -0.0033 | 3.91e-01 |
| age_hazard_staggered_v2_vs_age_hazard_v1 | c_local_fin | 360 | +0.0154 | +0.0055 | -0.0099 | -0.0273 | +0.0069 | 5.10e-01 |
| age_hazard_staggered_v2_vs_age_hazard_v1 | top10_fin | 360 | +0.4086 | +0.2700 | -0.1386 | -0.1469 | -0.1305 | 4.54e-01 |
| age_hazard_staggered_v2_vs_age_hazard_v1 | harvested_fin | 360 | +0.7875 | +0.7865 | -0.0010 | -0.0024 | +0.0004 | 8.67e-02 |
| age_hazard_staggered_v2_vs_age_hazard_v1 | gap_recent_mean_last100 | 360 | -0.0028 | -0.0023 | +0.0005 | +0.0004 | +0.0005 | 5.27e-03 |
| age_hazard_staggered_v2_vs_sync_v2 | c_ideal_fin | 360 | +0.0173 | +0.0106 | -0.0067 | -0.0299 | +0.0165 | 8.40e-01 |
| age_hazard_staggered_v2_vs_sync_v2 | c_ideal_tm | 360 | +0.1345 | +0.1010 | -0.0335 | -0.0408 | -0.0267 | 4.05e-01 |
| age_hazard_staggered_v2_vs_sync_v2 | top10_fin | 360 | +0.4086 | +0.2700 | -0.1386 | -0.1468 | -0.1307 | 4.54e-01 |
| age_hazard_staggered_v2_vs_sync_v2 | gap_recent_mean_last100 | 360 | -0.0028 | -0.0023 | +0.0005 | +0.0004 | +0.0005 | 5.27e-03 |
| age_hazard_staggered_v2_vs_sync_v2 | harvested_fin | 360 | +0.7875 | +0.7865 | -0.0010 | -0.0023 | +0.0004 | 8.67e-02 |

---

# 6. Post-BUG-0006 Four-Tier Overview Across $\xi \in [0.1, 0.9]$ (`mgstd5`)

| Tier | xi | van | H | harvested | c_ideal(fin) | c_ideal(tm) | wedge(fin) | wedge(tm) | top10(fin) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M0_closed_Wv05 | 0.1 | 0.5 | 200 | 0.6654 | +0.3755 | +0.3493 | -0.0914 | -0.0330 | 0.1164 |
| M0_closed_Wv05 | 0.3 | 0.5 | 200 | 0.6650 | +0.3493 | +0.3336 | -0.0242 | -0.0275 | 0.1163 |
| M0_closed_Wv05 | 0.5 | 0.5 | 200 | 0.6573 | +0.2653 | +0.2881 | +0.0191 | -0.0092 | 0.1165 |
| M0_closed_Wv05 | 0.7 | 0.5 | 200 | 0.6176 | +0.3378 | +0.2058 | +0.0051 | +0.0069 | 0.1155 |
| M0_closed_Wv05 | 0.9 | 0.5 | 200 | 0.3300 | -0.0287 | +0.0437 | -0.0000 | +0.0080 | 0.1144 |
| M1_lineage_Wv05 | 0.1 | 0.5 | 200 | 0.7684 | +0.1624 | +0.1405 | -0.0048 | -0.0053 | 0.2153 |
| M1_lineage_Wv05 | 0.3 | 0.5 | 200 | 0.7682 | +0.0539 | +0.1366 | -0.0047 | -0.0098 | 0.2151 |
| M1_lineage_Wv05 | 0.5 | 0.5 | 200 | 0.7655 | +0.0885 | +0.1324 | -0.0142 | -0.0008 | 0.2142 |
| M1_lineage_Wv05 | 0.7 | 0.5 | 200 | 0.7361 | +0.0253 | +0.1007 | +0.0371 | +0.0067 | 0.2083 |
| M1_lineage_Wv05 | 0.9 | 0.5 | 200 | 0.4325 | -0.0030 | +0.0436 | +0.0032 | +0.0101 | 0.2028 |
| M1_turnover_staggered | 0.1 | 0.9 | 400 | 0.9385 | +0.0404 | +0.0933 | -0.0493 | -0.0120 | 0.2956 |
| M1_turnover_staggered | 0.3 | 0.9 | 400 | 0.9384 | -0.0054 | +0.0803 | +0.0078 | -0.0049 | 0.2971 |
| M1_turnover_staggered | 0.5 | 0.9 | 400 | 0.9376 | -0.0377 | +0.0768 | +0.0438 | -0.0043 | 0.2936 |
| M1_turnover_staggered | 0.7 | 0.9 | 400 | 0.9140 | -0.0416 | +0.0507 | +0.0671 | +0.0030 | 0.2778 |
| M1_turnover_staggered | 0.9 | 0.9 | 400 | 0.6949 | +0.0000 | +0.0216 | +0.0000 | +0.0051 | 0.2721 |
| M1_turnover_sync | 0.1 | 0.9 | 400 | 0.9382 | +0.0628 | +0.1050 | -0.0601 | -0.0007 | 0.4187 |
| M1_turnover_sync | 0.3 | 0.9 | 400 | 0.9381 | +0.0181 | +0.1023 | +0.0003 | -0.0058 | 0.4193 |
| M1_turnover_sync | 0.5 | 0.9 | 400 | 0.9374 | +0.0791 | +0.0931 | -0.0439 | -0.0030 | 0.4200 |
| M1_turnover_sync | 0.7 | 0.9 | 400 | 0.9213 | +0.0891 | +0.0566 | -0.0406 | +0.0044 | 0.4019 |
| M1_turnover_sync | 0.9 | 0.9 | 400 | 0.6660 | -0.0110 | +0.0205 | +0.0187 | +0.0065 | 0.3975 |
| M2_lambda0 | 0.1 | 0.9 | 200 | 0.7701 | -0.0487 | +0.0458 | +0.0780 | +0.0644 | 0.2112 |
| M2_lambda0 | 0.3 | 0.9 | 200 | 0.7692 | -0.0998 | +0.0297 | +0.1011 | +0.0526 | 0.2099 |
| M2_lambda0 | 0.5 | 0.9 | 200 | 0.7674 | -0.0620 | +0.0603 | +0.0437 | +0.0251 | 0.2116 |
| M2_lambda0 | 0.7 | 0.9 | 200 | 0.7295 | +0.0656 | +0.0668 | -0.1264 | +0.0079 | 0.2095 |
| M2_lambda0 | 0.9 | 0.9 | 200 | 0.4278 | -0.0038 | +0.0316 | +0.0016 | +0.0369 | 0.1951 |
| M2_lambda1 | 0.1 | 0.9 | 200 | 0.7693 | +0.1314 | +0.2167 | -0.0853 | -0.1029 | 0.2148 |
| M2_lambda1 | 0.3 | 0.9 | 200 | 0.7679 | +0.0306 | +0.2002 | -0.0272 | -0.0834 | 0.2151 |
| M2_lambda1 | 0.5 | 0.9 | 200 | 0.7642 | +0.1549 | +0.1970 | -0.0770 | -0.0547 | 0.2172 |
| M2_lambda1 | 0.7 | 0.9 | 200 | 0.7325 | +0.1005 | +0.1510 | -0.0127 | -0.0203 | 0.2160 |
| M2_lambda1 | 0.9 | 0.9 | 200 | 0.3893 | +0.0242 | +0.0434 | +0.0214 | +0.0358 | 0.2079 |

---

# 7. Reproduction & Provenance

- `bug0006_remeasure_v2` gross/net: `T1` 720/720, `W_v05` 800/800, `M2_xi` 200/200 (`sources_moved_during_sweep = False`).
- `turnover_v2` gross/net: 1520/1520 runs across 76/76 cells (`sources_moved_during_sweep = False`).
- Summary CSVs: `data/bug0006_remeasure_v2/tables/*.csv`.
