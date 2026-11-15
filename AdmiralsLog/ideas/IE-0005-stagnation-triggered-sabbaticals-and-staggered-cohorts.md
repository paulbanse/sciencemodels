---
title: "IE-0005 - Stagnation-triggered sabbaticals (steps_stationary) and staggered initial founder tenure"
author: "Antigravity"
date: "2026-11-15"
tags: [idea, interventions, turnover, sabbatical-shocks, cohort-shock, model-refinement]
project: "sciencemodels"
status: idea
---

- **Status:** idea
- **Date:** 2026-11-15
- **Related:** `output/reports/interventions_v1.md` (Line 172), `output/reports/turnover_v1.md` (Lines 239–256), `AdmiralsLog/proposals/P-0007-sabbatical-shocks-x-frontier-refills.md`, `AdmiralsLog/proposals/P-0009-core-turnover-and-steady-state-interventions.md`

## The idea

Two clean micro-specification refinements surfaced in `F2` (`interventions_v1`) and `F4` (`turnover_v1`):

1. **Position-History Stagnation Trigger for Sabbatical Shocks (`steps_stationary >= k`):**
   In `interventions_v1` Arm B, "trapped in a stationary cluster" (`P-0003`) was operationalized as standing on a tile with knowledge $k_{xy} < 0.025$ because `Scientist` does not currently track how many steps it has stayed on the same tile. Tracking a single integer counter `self.steps_stationary` on `Scientist` (incremented when `best_pos == self.pos`, reset to `0` on move) allows sabbatical shocks (`p_reloc`) to trigger on true behavioral immobility (`steps_stationary >= 10`) even when a tile is kept above $0.025$ by `occupied` paradigm refills (`P-0007`).
2. **Staggered Initial Founder Tenure Under `age_hazard` ($\tau_0 \sim \text{Uniform}(0, \tau_{\max})$):**
   In `turnover_v1`, all 20 founders started with tenure $\tau_0 = 0$ and hit $\tau_{\max} = 200$ at the **exact same step ($t = 200$)**, causing a synchronous 20-agent extinction shock ($c_{\text{ideal}}$ plunged to $-0.6212$ at step 200 before recovering to $-0.2061$ at step 300 and settling to $-0.5678$ at step 400). Initializing founders with staggered tenures $\tau_0 \in \{0, 10, \dots, 190\}$ spreads founder retirements smoothly at one per 10 steps across $[10, 200]$, removing the synchronous cohort shock (`P-0009`).
