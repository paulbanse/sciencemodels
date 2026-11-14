---
title: "P-0005 - Path D typology experiment: formalize distraction, destruction, and P->G counterforces"
author: "Antigravity"
date: "2026-11-14"
tags: [proposal, strategic, experiment, typology, path-d, hunting-proxy-patterns]
project: "sciencemodels"
status: open
---

- **Status:** open
- **Class:** strategic
- **Date:** 2026-11-14
- **Spark:** `CONT2-hunting-proxy-patterns-formalization.md` (Item 1.4), `context/Hunting_Proxy_Patterns.md` (Kapeller, Paul, Braganza 2026), and `AdmiralsLog/way_forward.md` (Path D: instantiate the typology as parameterized variants of one model).
- **Outcome target:** a ruling on whether to run the $2 \times 2 \times 3$ Path D typology sweep to anchor the Theory-Companion Note / Discussion section.
- **Ruling:** pending
- **Spawned:** none

## The proposal

Instantiate the three causal channels of `context/Hunting_Proxy_Patterns.md` as parameterized regimes of the ABM (`CONT2-hunting-proxy-patterns-formalization.md`):
- **Distraction ($AP_F$ crowding out $A$):** Measured by the baseline visibility elasticity $1 - c \in \{0.3, 0.5, 0.7\}$.
- **Destruction ($AP_F \dashv G$ direct harm):** Controlled by a congestion-noise parameter $\gamma \in \{0.0 \text{ (off)}, 1.0 \text{ (on)}\}$ where $m$ agents stacked on the same visible tile experience amplified perception noise $\epsilon_{\text{eff}} = \epsilon \cdot m^\gamma$ ("hot fields lowers positive predictive value," Ioannidis 2005).
- **Counterforce ($P \to G$ functional spillover):** Controlled by a superstar-spillover parameter $\kappa \in \{0.0 \text{ (off)}, 1.0 \text{ (on)}\}$ where high-prestige supervision/co-location reduces perception noise $\epsilon_i = \epsilon / (1 + \kappa P_{\text{sup}})$ (Azoulay et al. 2010).

## The stated reason

`AdmiralsLog/way_forward.md` (Path D) identifies formalizing the March 2026 Kapeller–Paul–Braganza note (`Hunting_Proxy_Patterns.md`) as the fastest route to a theory-companion publication and as the fallback partner for Path A (`A + D`) if Gate 2 finds the dyadic layer adds only robustness. Running this 12-cell crossing generates the exact 4-quadrant phase diagram (`Distraction only`, `Distraction + Destruction`, `Distraction + Counterforce`, `Full Interaction`) needed to classify when prestige hierarchies stabilize honest signalling versus when they collapse into proxy failure.

## Experimental design & cost

- **Grid:** 2 destruction levels ($\gamma \in \{0, 1\}$) $\times$ 2 counterforce levels ($\kappa \in \{0, 1\}$) $\times$ 3 visibility weights ($1-c \in \{0.3, 0.5, 0.7\}$) $\times$ 2 landscapes (`mgstd5`, `noisypr5`) at $g = 0.1$ = 24 cells $\times$ 20 seeds = **480 runs** (~7 minutes at `n_jobs=3`).
- **Deliverable:** `output/reports/typology_path_d_v1.md` + 4-quadrant $P\text{–}G$ decay figure matching `Hunting_Proxy_Patterns.md` Fig. 1a/1b.

Do we proceed with the Path D typology experiment?
