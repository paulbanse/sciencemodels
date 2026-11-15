---
title: "P-0003 - Institutional counterforce experiments: prestige-damped entry lotteries and transient-diversity shocks (Path C)"
author: "Antigravity"
date: "2026-11-15"
tags: [proposal, operative, experiment, counterforces, lotteries, welfare, path-c, exercised]
project: "sciencemodels"
status: exercised
---

- **Status:** exercised
- **Class:** operative
- **Date:** 2026-11-14 (proposed); 2026-11-15 (exercised in `output/reports/interventions_v1.md`)
- **Spark:** `CONT1-depletion-counterforces-refills.md` (Section 3, Item 1.2) and `AdmiralsLog/way_forward.md` (Path C: test Smaldino et al. 2019 interventions against the welfare collapse threshold).
- **Outcome target:** a ruling on whether to run the Path C institutional intervention sweep against the $\xi^* \in [0.6, 0.8]$ welfare kink.
- **Ruling:** exercised (2026-11-15, Arm B of `output/reports/interventions_v1.md`, 36 cells / 720 runs). Sabbatical relocation shocks (`p_reloc = 0.02`) cut the high-visibility welfare loss penalty by ~40% (`0.00929 -> 0.00557`) and convexity by 27%, whereas entry lotteries ($\alpha = 0.0$) reduce the penalty by only `0.9%`.
- **Spawned:** `output/reports/interventions_v1.md` (Result B), `P-0007`, `CONT1` (v2), `CONT4` (Section 2.3), `CONT5` (Section 3)

## The proposal

Extend the Path C welfare probe (`output/reports/welfare_probe_v1.md`) with two minimal institutional counterforce experiments scored against `ideal_merit_cum`, `Corr_prestige_ideal_merit`, and the $\xi^* \in [0.6, 0.8]$ collapse breakpoint:

1. **Modified Entry Lotteries / Prestige Damping (Smaldino et al. 2019 analogue):**
   Generalize the supervisor roulette weight in `model.py:389-420` to $P(\text{sup}_i) \propto \text{prestige}_i^\alpha$ for $\alpha \in \{1.0 \text{ (baseline linear roulette)}, 0.5 \text{ (concave damping)}, 0.0 \text{ (pure placement lottery)}\}$.
2. **Sabbatical / Maverick Relocation Shocks (Zollman 2010 transient diversity):**
   With probability $p_{\text{reloc}} \in \{0.0, 0.02\}$ per step, relocate an agent trapped in a stationary cluster to a random unexplored tile, breaking status-quo lock-in (`agents.py:166`).

## Exercise Summary (`output/reports/interventions_v1.md`, Result B)

- **Executed:** 36 cells $\times$ 20 seeds = 720 runs (`data/interventions_v1/cells/cellB_*`).
- **Result:** Sabbatical shocks (`p_reloc = 0.02`) flatten the high-visibility loss penalty by $40\%$ (`0.009286 -> 0.005573`), cut convexity by $27\%$ (`0.005517 -> 0.004037`), cut terminal loss at weight $0.9$ by $41\%$ (`0.010330 -> 0.006047`), and improve quality correlation at weight $0.9$ (`-0.2572 -> -0.2391`). Entry lotteries ($\alpha = 0.0$) leave the loss penalty essentially untouched (`-0.9%`), proving that the welfare kink is an attention-side lock-in phenomenon.
