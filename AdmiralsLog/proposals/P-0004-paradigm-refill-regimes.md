---
title: "P-0004 - Endogenous paradigm refills: contrast incumbent hot-spot bailouts against frontier emergence (new_questions)"
author: "Antigravity"
date: "2026-11-14"
tags: [proposal, operative, experiment, new-questions, paradigm-shifts, lock-in]
project: "sciencemodels"
status: open
---

- **Status:** open
- **Class:** operative
- **Date:** 2026-11-14
- **Spark:** `CONT1-depletion-counterforces-refills.md` (Section 4, Item 1.3); the `new_questions` lever was split from `knowledge_normalization` in Path 0 (`33421ff`) but pinned to `0` in every Wave 1–3 sweep.
- **Outcome target:** a ruling on whether to probe the dormant `new_questions` channel under occupied vs. frontier refill rules.
- **Ruling:** pending
- **Spawned:** none

## The proposal

Run an 18-cell probe comparing three epistemic regeneration regimes when tiles drop below the $0.025$ depletion threshold (`sciencemodels/model.py:342-349`):
1. `refill_mode = "off"` (`new_questions = False`, current baseline).
2. `refill_mode = "occupied"` (`new_questions = True` as currently coded in `model.py:344`: only *occupied* depleted tiles refill to $U(0,1)$ — "incumbent hot-field bailout").
3. `refill_mode = "frontier"` (when an occupied tile drops below $0.025$, the new question $U(0,1)$ spawns on a random *unoccupied* tile — "frontier paradigm emergence").

## The stated reason

1. **Unexplored Path 0 lever:** Path 0 disentangled `new_questions` from `knowledge_normalization` specifically so refills could be varied independently (`docs/conceptual_degrees_of_freedom.md:105`), yet no sweep has tested `new_questions = True`.
2. **Theoretical link to Neoclassical Lock-In (Dobusch & Kapeller 2009) and Hot-Field Bias (Ioannidis 2005):** The current code in `model.py:344` refills only tiles where scientists are already standing. Under high visibility pressure ($1-c$), herds that exhaust a tile receive an immediate exogenous knowledge refill right under their feet. Contrasting `"occupied"` against `"frontier"` tests when visibility-driven communities adapt to new paradigms versus when they remain locked into exhausted orthodoxy.

## Experimental design & cost

- **Grid:** 3 `refill_mode` (`off`, `occupied`, `frontier`) $\times$ 3 curiosities (`0.3`, `0.5`, `0.7`) $\times$ 2 entry rates (`0.0`, `0.1`) on `mgstd5` = 18 cells $\times$ 20 seeds = **360 runs** at $H = 200$ (~5 minutes at `n_jobs=3`).
- **Implementation:** Driver-side subclass/wrapper overriding `_generate_new_questions`, with bit-identity verification for `off` and `occupied`.

Do we proceed with the 18-cell `new_questions` paradigm-refill probe?
