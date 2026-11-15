---
title: "P-0002 - Spatial spawn dispersal experiment: disentangle birth-tile depletion from visibility herding"
author: "Antigravity"
date: "2026-11-15"
tags: [proposal, operative, experiment, spatial-herding, depletion, entry, exercised]
project: "sciencemodels"
status: exercised
---

- **Status:** exercised
- **Class:** operative
- **Date:** 2026-11-14 (proposed); 2026-11-15 (exercised in `output/reports/interventions_v1.md`)
- **Spark:** `CONT1-depletion-counterforces-refills.md` (Section 2, Item 1.1); in `sciencemodels/model.py:389-420`, newcomers spawn at `coords = sup.pos`, fusing prestige-proportional supervision with birth-tile starvation.
- **Outcome target:** a ruling on whether to run the 18-cell spawn-dispersal decomposition sweep (`r_spawn` in `{0, 1, random}`).
- **Ruling:** exercised (2026-11-15, Arm A of `output/reports/interventions_v1.md`, 18 cells / 360 runs). Hypothesis falsified: dispersing birth tiles (`ring1`, `random`) makes `Corr_prestige_ideal_merit` *more negative* at all three curiosities (`starvation share = -5.7%, -21.5%, -53.5%`) and widens the newcomer-incumbent gap by `0.08%–12.0%`.
- **Spawned:** `output/reports/interventions_v1.md` (Result A), `AdmiralsLog/ideas/rejected/IE-0001-birth-tile-starvation-hypothesis.md`, `CONT1` (v2), `CONT4` (Section 3.1)

## The proposal

Run a clean 18-cell experimental probe (`scripts/interventions/`, 20 seeds per cell, $H = 200$) varying a single birth-placement parameter $r_{\text{spawn}} \in \{0 \text{ (exact supervisor tile)}, 1 \text{ (von Neumann ring around supervisor)}, \text{uniform random grid tile}\}$ across curiosity $c \in \{0.3, 0.5, 0.7\}$ and landscape $\in \{\text{mgstd5}, \text{noisypr5}\}$ at entry rate $g = 0.1$.

## The stated reason

In Waves 1–3 (`output/reports/sensitivity_map_v1.md`, `output/reports/career_flows_v1.md`), activating prestige-proportional entry flips `Corr_prestige_ideal_merit` negative and prevents newcomers from ever closing the prestige gap with incumbents. However, because `model.py` places every newcomer directly on `sup.pos` (which the prestigious supervisor has already multiplicatively depleted via `farm()`), two distinct mechanisms are confounded:

1. **Mechanical birth-tile starvation:** Newcomers harvest crumbs on an exhausted birth tile while their supervisor retains accumulated historical prestige.
2. **Social/attentional herding:** Visibility-weighted movement ($1-c$) pulls agents toward crowded prestige clusters even when they do not start on the same tile.

Varying $r_{\text{spawn}}$ separates the two mechanisms quantitatively without altering any other model rule.

## Exercise Summary (`output/reports/interventions_v1.md`, Result A)

- **Executed:** 18 cells $\times$ 20 seeds = 360 runs (`data/interventions_v1/cells/cellA_*`).
- **Result:** Removing birth-tile inheritance (`ring1`, `random`) worsens the quality drop (`-0.154 -> -0.171` at $c=0.3$, `-0.057 -> -0.145` at $c=0.5$, `+0.028 -> -0.095` at $c=0.7$) and widens the newcomer gap. Prestige-proportional recruitment and visibility herding account for $105.7\%$ to $153.5\%$ of the entry quality drop. No further spawn-dispersal runs are needed.
