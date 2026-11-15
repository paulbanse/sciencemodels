---
title: "P-0004 - Endogenous paradigm refills: contrast incumbent hot-spot bailouts against frontier emergence (new_questions)"
author: "Antigravity"
date: "2026-11-15"
tags: [proposal, operative, experiment, new-questions, paradigm-shifts, lock-in, exercised]
project: "sciencemodels"
status: exercised
---

- **Status:** exercised
- **Class:** operative
- **Date:** 2026-11-14 (proposed); 2026-11-15 (exercised in `output/reports/interventions_v1.md`)
- **Spark:** `CONT1-depletion-counterforces-refills.md` (Section 4, Item 1.3); the `new_questions` lever was split from `knowledge_normalization` in Path 0 (`33421ff`) but pinned to `0` in every Wave 1–3 sweep.
- **Outcome target:** a ruling on whether to probe the dormant `new_questions` channel under occupied vs. frontier refill rules.
- **Ruling:** exercised (2026-11-15, Arm C of `output/reports/interventions_v1.md`, 18 cells / 360 runs). `occupied` refills collapse exploration by 84% (`0.377 -> 0.061`) and widen the newcomer gap $11\times$ (`-0.451`), whereas `frontier` refills restore positive quality correlation under entry (`+0.078`) but accumulate unclaimed on the grid (`0.1895` vs. `0.0006` baseline).
- **Spawned:** `output/reports/interventions_v1.md` (Result C), `P-0007`, `CONT1` (v2), `CONT5` (Section 4)

## The proposal

Run an 18-cell probe comparing three epistemic regeneration regimes when tiles drop below the $0.025$ depletion threshold (`sciencemodels/model.py:422-429`):
1. `refill_mode = "off"` (`new_questions = False`, current baseline).
2. `refill_mode = "occupied"` (`new_questions = True` as currently coded in `model.py`: only *occupied* depleted tiles refill to $U(0,1)$ — "incumbent hot-field bailout").
3. `refill_mode = "frontier"` (when an occupied tile drops below $0.025$, the new question $U(0,1)$ spawns on a random *unoccupied* tile — "frontier paradigm emergence").

## Exercise Summary (`output/reports/interventions_v1.md`, Result C)

- **Executed:** 18 cells $\times$ 20 seeds = 360 runs (`data/interventions_v1/cells/cellC_*`).
- **Result:**
  - `occupied` refills induce hard exploitation lock-in (`explored` drops $84\%$ from $0.377$ to $0.061$; newcomer gap widens $11\times$ to $-0.451$; $c_{\text{ideal}} = -0.045$).
  - `frontier` refills are the only regime with positive quality alignment under entry ($c_{\text{ideal}} = +0.078$), but visibility-locked herds fail to migrate to the frontier (`mean_grid_knowledge` accumulates to $0.1895$ vs. $0.0006$ baseline while `explored` drops to $0.330$).
  - **Follow-up spawned (`P-0007`):** Cross `frontier` refills with sabbatical relocation shocks (`p_reloc > 0`) under a supply-comparable refill rule to test whether mobility shocks unlock unclaimed frontier paradigms.
