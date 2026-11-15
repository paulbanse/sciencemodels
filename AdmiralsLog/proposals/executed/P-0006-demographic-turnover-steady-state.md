---
title: "P-0006 - Demographic turnover and retirement probe: test career flows under stationary population"
author: "Antigravity"
date: "2026-11-15"
tags: [proposal, operative, experiment, demographics, turnover, career-flows, exercised]
project: "sciencemodels"
status: exercised
---

- **Status:** exercised
- **Class:** operative
- **Date:** 2026-11-14 (proposed); 2026-11-15 (exercised in `output/reports/turnover_v1.md`)
- **Spark:** `output/reports/career_flows_v1.md` (Lines 257–295) and `docs/conceptual_degrees_of_freedom.md` (Line 134: `age` is a recorded-but-dead lever; no exit/retirement exists, so population grows unboundedly as $21 + \lfloor g \cdot t \rfloor$).
- **Outcome target:** a ruling on whether to run the demographic turnover probe (`turnover_v1`) to verify that the career-flow and wedge findings hold under a stationary population ($N = 20$) with finite career spans.
- **Ruling:** exercised (2026-11-15, `output/reports/turnover_v1.md`, 54 cells / 1,080 runs at $H = 400$). Both headline claims survive in sign, with a major discovery: stationary turnover *strengthens* negative `Corr_prestige_ideal_merit` by $2.79\times$ (`fixed_span`, `-0.0860 -> -0.2401`) to $6.61\times$ (`age_hazard`, `-> -0.5678`, negative in 100% of runs), while shrinking the arrival prestige gap (`gap_recent`) by $55.0\%$ to $79.3\%$ (still negative in 1,080/1,080 runs).
- **Spawned:** `output/reports/turnover_v1.md`, `P-0009`, `CONT3` (v2), `CONT4` (Section 3.3), `CONT5` (Section 2), `IE-0004`, `IE-0005`

## The proposal

Run a checkpointed sweep (`scripts/turnover/`, `data/turnover_v1/`, `output/reports/turnover_v1.md`) comparing the baseline unbounded-entry model (`exit_mode = "none"`) against two stationary-population turnover rules (`exit_mode = "fixed_span"` at $N \equiv 20$, where each newcomer arrival retires the oldest incumbent, and `exit_mode = "age_hazard"`, where scientists retire after $\tau_{\max} = 200$ steps and are replaced via the supervisor roulette).

## Exercise Summary (`output/reports/turnover_v1.md`)

- **Executed:** 54 cells $\times$ 20 seeds = 1,080 runs at $H = 400$ (`data/turnover_v1/`).
- **Key Findings:**
  1. **Proxy failure deepens $2.8\times\text{–}6.6\times$ after founders retire:** Terminal $c_{\text{ideal}}$ drops from $-0.0860$ (`none`) to $-0.2401$ (`fixed_span`, negative in $89.2\%$ of runs) and $-0.5678$ (`age_hazard`, negative in $100\%$ of runs). Immortal founders who discovered genuine peaks at $t=0$ were masking the severity of dynasty-driven proxy failure.
  2. **The arrival gap survives in sign ($1,080/1,080$ runs) and shrinks $55\%\text{–}79\%$ in magnitude:** Measured via a 20-step tenure window (`gap_recent`, since `is_newcomer` becomes $100\%$ of active agents after founders retire), `gap_recent` moves from $-0.01362$ (`none`) to $-0.00612$ (`fixed_span`, $-55.0\%$) and $-0.00282$ (`age_hazard`, $-79.3\%$).
  3. **Entry-rate dilution disappears under stationary population (`fixed_span`):** Terminal $c_{\text{ideal}}$ is flat across $g \in \{0.05, 0.10, 0.20\}$ ($-0.231, -0.247, -0.243$).
  4. **Follow-up spawned (`P-0009`):** Promote `exit_mode` (with staggered initial founder tenure under `age_hazard` to remove the step-200 simultaneous cohort-extinction shock) and `gap_recent` into `sciencemodels/`, and test whether sabbatical shocks (`p_reloc`) can rescue steady-state proxy failure.
