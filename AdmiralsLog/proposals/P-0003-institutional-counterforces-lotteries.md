---
title: "P-0003 - Institutional counterforce experiments: prestige-damped entry lotteries and transient-diversity shocks (Path C)"
author: "Antigravity"
date: "2026-11-14"
tags: [proposal, operative, experiment, counterforces, lotteries, welfare, path-c]
project: "sciencemodels"
status: open
---

- **Status:** open
- **Class:** operative
- **Date:** 2026-11-14
- **Spark:** `CONT1-depletion-counterforces-refills.md` (Section 3, Item 1.2) and `AdmiralsLog/way_forward.md` (Path C: test Smaldino et al. 2019 interventions against the welfare collapse threshold).
- **Outcome target:** a ruling on whether to run the Path C institutional intervention sweep against the $\xi^* \in [0.6, 0.8]$ welfare kink.
- **Ruling:** pending
- **Spawned:** none

## The proposal

Extend the Path C welfare probe (`output/reports/welfare_probe_v1.md`) with two minimal institutional counterforce experiments scored against `ideal_merit_cum`, `Corr_prestige_ideal_merit`, and the $\xi^* \in [0.6, 0.8]$ collapse breakpoint:

1. **Modified Entry Lotteries / Prestige Damping (Smaldino et al. 2019 analogue):**
   Generalize the supervisor roulette weight in `model.py:315-326` to $P(\text{sup}_i) \propto \text{prestige}_i^\alpha$ for $\alpha \in \{1.0 \text{ (baseline linear roulette)}, 0.5 \text{ (concave damping)}, 0.0 \text{ (pure placement lottery)}\}$.
2. **Sabbatical / Maverick Relocation Shocks (Zollman 2010 transient diversity):**
   With probability $p_{\text{reloc}} \in \{0.0, 0.02\}$ per step, relocate an agent trapped in a stationary cluster to a random unexplored tile, breaking status-quo lock-in (`agents.py:166`).

## The stated reason

`output/reports/welfare_probe_v1.md` completed the *diagnostic* half of Path C (establishing the convex collapse threshold at $\xi^* \in [0.6, 0.8]$ and the $31\times$ weight-vs.-toggle asymmetry), but explicitly left the *intervention* half of Path C (`AdmiralsLog/way_forward.md`: *"test interventions in the Smaldino et al. (2019) spirit"*) unrun. Testing whether prestige damping ($\alpha < 1$) or maverick shocks shift $\xi^*$ rightward provides the normative policy contribution of the project.

## Experimental design & cost

- **Grid:** 3 $\alpha$ levels (`1.0`, `0.5`, `0.0`) $\times$ 2 relocation rates (`0.0`, `0.02`) $\times$ 3 high-pressure visibility weights (`1-c` $\in \{0.5, 0.7, 0.9\}$) $\times$ 2 landscapes (`mgstd5`, `noisypr5`) at $g = 0.1$ = 36 cells $\times$ 20 seeds = **720 runs** (~10 minutes at `n_jobs=3`).
- **Implementation:** Driver-side wrapper with baseline bit-identity check at $(\alpha=1.0, p_{\text{reloc}}=0.0)$.
- **Deliverable:** Intervention ranking table and welfare-recovery curves for Path C.

Do we proceed with the Path C institutional counterforce sweep?
