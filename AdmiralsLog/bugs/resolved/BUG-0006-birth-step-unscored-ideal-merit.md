---
title: "BUG-0006 - Unscored step-t newborns (ideal_Merit=0) distort correlation_reporter"
author: "Coordination Session"
date: "2026-11-19"
tags: [bug, measurement, ideal-merit, newcomer, correlation-reporter]
project: "sciencemodels"
---

- **Status:** resolved (2026-11-19, direct fix with regression test and 3,240-run paired proof)
- **Date:** 2026-11-19
- **Related:** `sciencemodels/agents.py:48`, `sciencemodels/model.py:516-554`, `sciencemodels/metrics.py:308-316`, `tests/test_smoke.py`, `output/reports/bug0006_remeasure_v2.md`
- **Spawned:** `scripts/bug0006/run_bug0006_remeasure.py`, `scripts/turnover/run_turnover_v2.py`, `scripts/bug0006/analyse_bug0006.py`

## Symptom

Whenever demographic entry is active (`agent_generation_time > 0`), `Corr_prestige_ideal_merit` (and its vanishing/visibility variants) exhibits sharp negative sawtooth spikes on every step when new scientists are born—including terminal steps $t = 200$ and $t = 400$ for $g \in \{0.05, 0.10, 0.20\}$—and plunges artificially to `-0.5678` at $t = 400$ under synchronized cohort replacement (`turnover_v1` `age_hazard`, where 21 replacements enter simultaneously).

## Cause

In `MyModel.end_loop_update()` (`sciencemodels/model.py:516-554`), `self.updateIdealMerit()` runs at line 516 **before** `self.__generate_new_agents()` at line 517 and `self.datacollector.collect(self)` at line 554. Because `Scientist.__init__` (`sciencemodels/agents.py:48`) initialized `self.ideal_Merit = 0`, and `ideal_Merit <= 0` (`0` is the omniscient-planner optimum), every newborn spawned on step $t$ entered `correlation_reporter` at line 554 at `(prestige = 0.0, ideal_Merit = 0.0)`—artificially pairing zero prestige with maximum merit before the agent had ever farmed or been evaluated.

## Fix

1. `sciencemodels/agents.py:48`: Changed `self.ideal_Merit= 0` to `self.ideal_Merit = np.nan` so unscored newborns are explicitly marked non-finite until `update_ideal_merit` scores them on the following step.
2. `sciencemodels/metrics.py:308-316` + `_finite_pair_vectors` (appended after line 337, zero line shifts on existing CDF anchors): Updated `correlation_reporter` to extract matched finite `(prestige, merit)` pairs, excluding unscored step-$t$ newborns without touching dynamics, RNG state, harvest, or `ideal_merit_cum`.

## Resolution

Verified first-hand on 2026-11-19:
- **Canary & regression (`tests/test_smoke.py` Check 11):** `ALL SMOKE TESTS PASSED` (synthetic 3+1 test `synth_r = 1.000`, live step-20 birth-step `step20_unscored = 1`, step-21 non-birth `step21_unscored = 0`, `g=0` `g0_nans = 0`; 7 non-ideal Path B v2 anchors bit-identical).
- **CDF anchors (`scripts/cdf_reanchor/reanchor_cdf_refs.py`):** `PASS: every catalogue anchor resolves to the token it quotes` (109/109 resolvable callouts).
- **Paired invariance & re-measure (`output/reports/bug0006_remeasure_v2.md`, 3,240 runs across `data/bug0006_remeasure_v2/` + `data/turnover_v2/`):**
  - `g = 0.0` controls (`T1` 360 runs, `W_v05` 400 runs): max diff across all 9 metrics `<= 9.58e-15`.
  - Non-ideal channels under `g > 0` (`c_local_fin`, `c_local_tm`, `top10_fin`, `harvested_fin`, `ideal_loss_cum`, `team_tm`, `links_tm`, `gap_recent`): max diff `<= 8.88e-16`.
  - `T1` (`g = 0.1`, 360 runs): terminal `c_ideal_fin` shifts from `-0.1438` (`17/18` cells $< 0$) to `+0.0327` (`5/18` cells $< 0$, all on `noisypr5`), while time-mean `c_ideal_tm` moves modestly (`+0.2130 -> +0.2360`, still a `1.74x` collapse relative to `M0`'s `+0.4101`).
  - `turnover_v2` (`age_hazard_sync` vs `turnover_v1`): terminal `c_ideal_fin` shifts from `-0.5678` to `+0.0173` (`+0.5851` artifact removed); staggering founder ages (`age_hazard_staggered`) also removes the step-200/400 `top10` spike (`0.4086 -> 0.2700`).
