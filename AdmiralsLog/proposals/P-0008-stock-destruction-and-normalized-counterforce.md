---
title: "P-0008 - Path D v2: stock-level epistemic destruction and normalized superstar counterforce"
author: "Antigravity"
date: "2026-11-15"
tags: [proposal, strategic, experiment, typology, path-d, hunting-proxy-patterns, destruction, counterforce]
project: "sciencemodels"
status: open
---

- **Status:** open
- **Class:** strategic
- **Date:** 2026-11-15
- **Spark:** `output/reports/typology_v1.md` and `CONT2-hunting-proxy-patterns-formalization.md` (v2): in `typology_v1`, modelling $AP_F \dashv G$ destruction as candidate-perception noise ($\epsilon_{\text{eff}} = \epsilon \cdot m^\gamma$) acted as **exploratory annealing** (`ideal_loss_cum` fell by $16.3\%$, `IE-0002`), while modelling $P \to G$ counterforce via raw supervisor prestige ($\epsilon_0 / (1 + \kappa P_{\text{sup}})$) was **scale-trapped** (`<= 0.0039`, `IE-0003`) because $\bar{P} \approx 0.0254$.
- **Outcome target:** a ruling on whether to run Path D v2 (`typology_v2`) with stock/harvest-level epistemic destruction and dimensionless normalized prestige counterforce.
- **Ruling:** pending
- **Spawned:** none

## The proposal

Re-run the 24-cell Path D typology grid (`scripts/typology/run_typology_v2.py`, 20 seeds per cell, $H = 200$) with two corrected channel specifications that honour the causal DAGs of `context/Hunting_Proxy_Patterns.md`:

1. **Stock/Harvest-Level Epistemic Destruction ($AP_F \dashv G$, replacing perception-annealing $\gamma$):**
   When $m_t(\text{pos}) > 1$ agents crowd onto the same visible tile, crowding directly degrades epistemic output $G$ via excess depletion ("over-trampling / p-hacking pollution"): the tile loses $k_{xy} \leftarrow (1 - h)^{m_t^\gamma} k_{xy}$ (or agents harvest local proxy credit while a fraction $1 - m_t^{-\gamma}$ of the extracted knowledge is wasted / false-positive and excluded from `ideal_Merit`). Unlike candidate-perception noise, this cannot help agents escape herds and strictly reduces $G$ when agents stack on visible tiles.
2. **Dimensionless Normalized Superstar Counterforce ($P \to G$, fixing the $\bar{P} \approx 0.025$ scale trap):**
   Normalize supervisor (or live co-located peer-max) prestige by the step-level population mean prestige, $\tilde{P}_{\text{sup}} = P_{\text{sup}} / \max(\bar{P}_t, 10^{-12})$ (so $\mathbb{E}[\tilde{P}] = 1$ and top decile $\tilde{P} \approx 2.5\text{–}4.0$), and set $\epsilon_i = \epsilon_0 / (1 + \kappa \tilde{P}_{\text{sup}})$ with $\kappa \in \{0.0, 1.0\}$ (plus clamping perceived novelty at $\max(0, 1 + \epsilon_{\text{eff}} \eta)$ to eliminate the fractional-power `NaN` regime).

## The stated reason

`typology_v1` provided a crucial diagnostic: in a spatial search model with strict-improvement inertia, adding perception noise to crowded neighbors scatters herds (*annealing*), whereas `Hunting_Proxy_Patterns.md` defines **destruction ($AP_F \dashv G$)** as proxy-seeking activity that *directly damages the goal $G$*. Simultaneously, `typology_v1` diagnosed why raw $P_{\text{sup}} \approx 0.025$ made $\kappa = 1$ a $< 2.5\%$ nudge. Testing stock-level destruction $\times$ normalized counterforce delivers the clean 4-quadrant phase diagram needed for the Theory-Companion Note.

## Experimental design & cost

- **Grid:** 2 stock-destruction levels ($\gamma \in \{0, 1\}$) $\times$ 2 normalized counterforce levels ($\kappa \in \{0, 1\}$) $\times$ 3 curiosities ($c \in \{0.3, 0.5, 0.7\}$) $\times$ 2 entry rates ($g \in \{0.0, 0.1\}$) on `mgstd5` (plus optional `noisypr5` arm) = 24 cells $\times$ 20 seeds = **480 runs** (~3.5 minutes at `n_jobs=3`).
- **Deliverable:** `output/reports/typology_v2.md` comparing perceptual annealing (`v1`) against stock-level epistemic destruction (`v2`).

Do we proceed with the Path D v2 typology sweep?
