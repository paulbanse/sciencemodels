---
title: "P-0005 - Path D typology experiment: formalize distraction, destruction, and P->G counterforces"
author: "Antigravity"
date: "2026-11-15"
tags: [proposal, strategic, experiment, typology, path-d, hunting-proxy-patterns, exercised]
project: "sciencemodels"
status: exercised
---

- **Status:** exercised
- **Class:** strategic
- **Date:** 2026-11-14 (proposed); 2026-11-15 (exercised in `output/reports/typology_v1.md`)
- **Spark:** `CONT2-hunting-proxy-patterns-formalization.md` (Item 1.4), `context/Hunting_Proxy_Patterns.md` (Kapeller, Paul, Braganza 2026), and `AdmiralsLog/way_forward.md` (Path D: instantiate the typology as parameterized variants of one model).
- **Outcome target:** a ruling on whether to run the $2 \times 2 \times 3$ Path D typology sweep to anchor the Theory-Companion Note / Discussion section.
- **Ruling:** exercised (2026-11-15, `output/reports/typology_v1.md`, 24 cells / 480 runs across $\gamma \in \{0,1\} \times \kappa \in \{0,1\} \times c \in \{0.3, 0.5, 0.7\} \times g \in \{0.0, 0.1\}$ on `mgstd5`). Revealed that candidate-perception noise ($\epsilon \cdot m^\gamma$) acts as simulated annealing (`wedge_fin` narrows by `+0.0316`, `ideal_loss_cum` drops by `16.3%`) rather than destructive crowding-out, and that unnormalized $1/(1 + \kappa P_{\text{sup}})$ is scale-trapped (`<= 0.0039`).
- **Spawned:** `output/reports/typology_v1.md`, `P-0008`, `AdmiralsLog/ideas/rejected/IE-0002-crowding-perception-noise-as-epistemic-destruction.md`, `IE-0003`, `CONT2` (v2), `CONT4` (Sections 2.2 & 3.2)

## The proposal

Instantiate the three causal channels of `context/Hunting_Proxy_Patterns.md` as parameterized regimes of the ABM (`CONT2-hunting-proxy-patterns-formalization.md`):
- **Distraction ($AP_F$ crowding out $A$):** Measured by the baseline visibility elasticity $1 - c \in \{0.3, 0.5, 0.7\}$.
- **Destruction ($AP_F \dashv G$ direct harm):** Controlled by a congestion-noise parameter $\gamma \in \{0.0 \text{ (off)}, 1.0 \text{ (on)}\}$ where $m$ agents stacked on the same visible tile experience amplified perception noise $\epsilon_{\text{eff}} = \epsilon \cdot m^\gamma$.
- **Counterforce ($P \to G$ functional spillover):** Controlled by a superstar-spillover parameter $\kappa \in \{0.0 \text{ (off)}, 1.0 \text{ (on)}\}$ where high-prestige supervision reduces perception noise $\epsilon_i = \epsilon / (1 + \kappa P_{\text{sup}})$.

## Exercise Summary (`output/reports/typology_v1.md`)

- **Executed:** 24 cells $\times$ 20 seeds = 480 runs (`data/typology_v1/`).
- **Result:**
  1. `Pure Distraction` ($\gamma=0, \kappa=0$) cleanly reproduces the entry-driven wedge (`wedge_fin = -0.0002` at $g=0.0$ vs. `-0.1444` at $g=0.1$).
  2. Candidate-perception crowding noise ($\gamma=1$) acts as **exploratory annealing** rather than epistemic destruction, narrowing `wedge_fin` by $+0.0316$ and cutting `ideal_loss_cum` by $16.3\%$ ($-0.0783$) because upward noise spikes on crowded neighbors break strict-improvement inertia (`IE-0002`).
  3. Superstar counterforce ($\kappa=1$) binds bit-wise in 114/120 runs at $g=0.1$ (and is bit-identical $0.0$ at $g=0.0$) but moves outcomes by $\le 0.0039$ because raw `prestige` averages $\sim 0.0254$.
  4. **Follow-up spawned (`P-0008`):** Re-specify destruction on the knowledge/merit stock directly and normalize $P_{\text{sup}}$ by mean prestige ($\tilde{P}_{\text{sup}} = P_{\text{sup}} / \bar{P}_t$).
