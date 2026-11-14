---
title: "P-0006 - Demographic turnover and retirement probe: test career flows under stationary population"
author: "Antigravity"
date: "2026-11-14"
tags: [proposal, operative, experiment, demographics, turnover, career-flows]
project: "sciencemodels"
status: open
---

- **Status:** open
- **Class:** operative
- **Date:** 2026-11-14
- **Spark:** `output/reports/career_flows_v1.md` (Lines 257–295) and `docs/conceptual_degrees_of_freedom.md` (Line 134: `age` is a recorded-but-dead lever; no exit/retirement exists, so population grows unboundedly as $21 + \lfloor g \cdot t \rfloor$).
- **Outcome target:** a ruling on whether to run the demographic turnover probe (`turnover_v1`) to verify that the career-flow and wedge findings hold under a stationary population ($N = 20$) with finite career spans.
- **Ruling:** pending
- **Spawned:** none

## The proposal

Run a checkpointed sweep (`scripts/turnover/`, `data/turnover_v1/`, `output/reports/turnover_v1.md`) comparing the baseline unbounded-entry model (`exit_mode = "none"`) against two stationary-population turnover rules at $N = 20$ (`exit_mode = "fixed_span"`, where each newcomer arrival retires the oldest incumbent, and `exit_mode = "age_hazard"`, where scientists retire after $\tau_{\max}$ steps and are replaced via the supervisor roulette).

## The stated reason

In `output/reports/career_flows_v1.md`, founders live forever (`mean_age` reaches 293 at step 400) and sit permanently on the peaks they depleted early in the run. A natural referee objection is that both the persistent newcomer-incumbent prestige gap and the negative `Corr_prestige_ideal_merit` under entry are artifacts of founder immortality rather than structural properties of prestige-mediated reproduction. Testing stationary generational turnover at horizon $H = 400$ directly answers this objection: if advisees inherit their retiring supervisor's depleted location and prestige weight, the dynasty lock-in persists even after every founder has retired.

## Experimental design & cost

- **Grid:** 3 `exit_mode` (`none`, `fixed_span`, `age_hazard`) $\times$ 3 entry/replacement rates $g \in \{0.05, 0.10, 0.20\}$ $\times$ 3 curiosities $c \in \{0.3, 0.5, 0.7\}$ $\times$ 2 landscapes (`mgstd5`, `noisypr5`) = 54 cells $\times$ 20 seeds = **1,080 runs** at $H = 400$ (~18 minutes at `n_jobs=3`).
- **Implementation:** Driver-side wrapper in `scripts/turnover/` with bit-identity proof for `exit_mode = "none"`.

Do we proceed with the `turnover_v1` stationary-population probe?
