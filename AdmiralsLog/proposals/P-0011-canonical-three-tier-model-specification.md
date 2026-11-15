---
title: "P-0011 - Canonical three-tier model specification (M0 Basic vs. M1 Entry vs. M2 Dyadic) and M0+M2 isolation probe"
author: "Antigravity"
date: "2026-11-15"
tags: [proposal, strategic, operative, model-specification, m0-m1-m2, basic-vs-collaborators, path-b]
project: "sciencemodels"
status: open
---

- **Status:** open
- **Class:** strategic / operative
- **Date:** 2026-11-15
- **Spark:** Owner review (2026-11-15, Item 2c) and `CONT6-canonical-model-specifications-basic-vs-collaborators.md`: across Waves 1–4, the boundary between the **Basic Model** and the **Collaborators Enter Model** blurred because vertical mentorship entry (`agent_generation_time > 0`, `M1`) and horizontal dyadic co-authorship (`dyadic_mode != "off"`, `M2`) were both referred to as "collaborators entering," and Path B (`path_b_probe_v1`, `path_b_probe_v2`) was run *only* on top of `M1` ($g = 0.1$), never on top of the closed-cohort Basic Model `M0` ($g = 0.0$).
- **Outcome target:** a ruling on (a) adopting the `M0` / `M1` / `M2` canonical three-tier specification in `sciencemodels/sweeps.py` and the paper drafts, and (b) running a 24-cell `M0 + M2` ($g = 0.0$ dyadic) isolation sweep to test horizontal collaboration without demographic entry.
- **Ruling:** pending
- **Spawned:** none

## The proposal

1. **Codify the Three Canonical Model Tiers (`M0`, `M1`, `M2`) in `sciencemodels/sweeps.py`:**
   - **`M0_BASIC` (Closed-Cohort Solo Search):** `n_agents = 20`, `agent_generation_time = 0.0`, `exit_mode = "none"`, `dyadic_mode = "off"`, `dyadic_move_pull = 0.0`.
   - **`M1_ENTRY` (Open-Cohort Vertical Mentorship / Lineage):** `M0_BASIC` + `agent_generation_time = 0.1` (and optional `exit_mode` $\in \{\text{none}, \text{fixed\_span}, \text{age\_hazard}\}$).
   - **`M2_DYADIC` (Horizontal Co-Authorship Choice & Co-Production):** `dyadic_mode` $\in \{\text{logit}, \text{threshold}\}$, `dyadic_lambda` $\in [0, 1]$, `dyadic_move_pull` $\in \{0.5, 1.0\}$, crossed explicitly with either `M0` ($g = 0.0$, pure horizontal collaboration) or `M1` ($g = 0.1$, lineage + co-authorship).
2. **Run the Missing 24-Cell `M0 + M2` Isolation Probe (`g = 0.0` with Dyadic Choice & Movement Pull):**
   Cross $\lambda \in \{0.0, 0.5, 1.0\} \times \text{dyadic\_move\_pull} \in \{0.0, 1.0\} \times \text{formulation} \in \{\text{logit}, \text{threshold}\} \times \text{landscape} \in \{\text{mgstd5}, \text{noisypr5}\}$ at **`agent_generation_time = 0.0`** ($N \equiv 20$ fixed cohort, `credit = structural`, 20 seeds per cell = **480 runs**).

## The stated reason

In `CONT6`, mapping every round onto `M0`, `M1`, and `M2` revealed that we have tested:
- `M0` alone (the 6 legacy notebooks / `figures_v1`, where $c_{\text{ideal}} > 0$ and `top10` $\approx 0.12$),
- `M1` alone (`sensitivity_map_v1`, `career_flows_v1`, `turnover_v1`, where $c_{\text{ideal}} < 0$ and `top10` $\approx 0.22\text{–}0.41$), and
- `M1 + M2` combined (`path_b_probe_v1` and `v2`, where $g = 0.1$ was pinned on).

Running the 24-cell `M0 + M2` ($g = 0.0$) grid answers a fundamental question for the Flagship paper: **can horizontal collaborator choice (`dyadic_mode != "off"` with `dyadic_move_pull = 1.0` at $\lambda = 0$) generate proxy failure and prestige concentration in a fixed cohort of peers (`M0`), or is intergenerational mentorship/entry (`M1`) a necessary condition for the negative wedge?**

## Experimental design & cost

- **Grid:** 3 $\lambda$ (`0.0`, `0.5`, `1.0`) $\times$ 2 `dyadic_move_pull` (`0.0`, `1.0`) $\times$ 2 formulations (`logit`, `threshold`) $\times$ 2 landscapes (`mgstd5`, `noisypr5`) at $g = 0.0, c = 0.5$ = 24 cells $\times$ 20 seeds = **480 runs** (~4 minutes at `n_jobs=3`).
- **Deliverable:** `M0_BASIC` / `M1_ENTRY` / `M2_DYADIC` constants in `sciencemodels/sweeps.py` + `M0 + M2` isolation table in `output/reports/path_b_m0_isolation_v1.md`.

Do we adopt the `M0` / `M1` / `M2` canonical specification and run the 480-run `M0 + M2` ($g=0$) isolation check?
