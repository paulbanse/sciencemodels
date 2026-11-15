---
title: "P-0009 - Core demographic turnover promotion and steady-state counterforce probe (Demography v2)"
author: "Antigravity"
date: "2026-11-15"
tags: [proposal, operative, demographics, turnover, steady-state, sabbatical-shocks, model-core]
project: "sciencemodels"
status: open
---

- **Status:** open
- **Class:** operative
- **Date:** 2026-11-15
- **Spark:** `output/reports/turnover_v1.md` and `CONT5-generational-turnover-and-paradigm-lockin.md` (Section 2): stationary turnover deepened terminal proxy failure $2.79\times$ (`fixed_span`, $c_{\text{ideal}} = -0.2401$) to $6.61\times$ (`age_hazard`, $c_{\text{ideal}} = -0.5678$), exposed a simultaneous step-200 cohort-replacement shock when all founders share identical initial tenure $\tau_0 = 0$, and required a rolling tenure-window gap (`gap_recent`) because `is_newcomer` becomes $100\%$ of active agents once founders retire.
- **Outcome target:** a ruling on (a) promoting `exit_mode` (`none`, `fixed_span`, `age_hazard`, `staggered_age_hazard`) and `gap_recent` from the `scripts/turnover/` wrapper into `sciencemodels/`, and (b) testing whether sabbatical shocks (`p_reloc`) can mitigate steady-state proxy failure.
- **Ruling:** pending
- **Spawned:** none

## The proposal

1. **Promote `exit_mode` and `gap_recent` into `sciencemodels/` core (default `exit_mode = "none"`, bit-identical to current `MyModel`):**
   - Add `exit_mode` $\in \{\text{"none"}, \text{"fixed\_span"}, \text{"age\_hazard"}, \text{"staggered\_age\_hazard"}\}$ and `max_tenure = 200` to `MyModel.__init__`, turning `agent.age` / `agent.tenure` from an inert attribute (`docs/conceptual_degrees_of_freedom.md:134`) into a validated first-class model lever.
   - Under `staggered_age_hazard`, initialize the 20 founders with uniform initial tenure $\tau_0 \in \{0, 10, \dots, 190\}$ so founder retirements are spread smoothly across $[10, 200]$ rather than occurring simultaneously at step 200 ([turnover_v1.md](file:///Users/joko/Git/sciencemodels/output/reports/turnover_v1.md#L237-L256)).
   - Add `recent_arrival_prestige_gap` (`gap_recent`, tenure $< 20$ vs. $\ge 20$) and phase-split correlations (`c_ideal_early100`, `c_ideal_late100`, `IE-0004`) to `sciencemodels/metrics.py`.
2. **Run a 36-cell Steady-State Counterforce Probe (`turnover_v2`, $H = 400$):**
   Cross `exit_mode` $\in \{\text{none}, \text{fixed\_span}, \text{staggered\_age\_hazard}\}$ $\times$ sabbatical relocation shock `p_reloc` $\in \{0.00, 0.02\}$ $\times$ curiosity $c \in \{0.3, 0.5, 0.7\}$ $\times$ landscape $\in \{\text{mgstd5}, \text{noisypr5}\}$ at $g = 0.10$.

## The stated reason

`turnover_v1` established that the strongest form of proxy failure in the entire project ($-0.240$ to $-0.568$) lives in the turnover regime once founders retire. Promoting `exit_mode` and `gap_recent` into `sciencemodels/` removes the brittle monkey-patch wrapper in `scripts/turnover/`, eliminates the step-200 synchronous founder-extinction shock via staggered initial ages, and answers whether the sabbatical mobility shock (`p_reloc = 0.02`, which cut loss by $40\%$ under immortal founders in `interventions_v1`) can also rescue a stationary multi-generational community from dynastic lock-in.

## Experimental design & cost

- **Grid:** 3 `exit_mode` $\times$ 2 `p_reloc` $\times$ 3 curiosities $\times$ 2 landscapes = 36 cells $\times$ 20 seeds = **720 runs** at $H = 400$ (~18 minutes at `n_jobs=3`).
- **Deliverable:** Core `sciencemodels/` support + smoke-test canary + `output/reports/turnover_v2.md`.

Do we promote `exit_mode` / `gap_recent` into `sciencemodels/` and run the 36-cell steady-state counterforce probe?
