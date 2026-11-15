---
title: "CONT2 - Formalizing the Hunting_Proxy_Patterns Typology in the Epistemic ABM"
author: "Antigravity (Review & Synthesis Round)"
date: "2026-11-15"
tags: [contribution, typology, proxy-failure, distraction, destruction, counterforce, path-d, typology-v1, m0-m1]
project: "sciencemodels"
---

- **Source:** Antigravity inspection & evaluation round (Item 1.4), updated with empirical findings from `output/reports/typology_v1.md` (24 cells / 480 runs) and contextual/scale framing (`CONT6`)
- **Date:** 2026-11-14 (v1); 2026-11-15 (v3 — post-`typology_v1` synthesis + contextual & scale framing)
- **Model Tier:** Crosses **Tier `M0` (Basic Closed-Cohort, $g = 0.0$)** and **Tier `M1` (Open-Cohort Mentorship Entry, $g = 0.1$)** with `dyadic_mode = "off"` (see [CONT6-canonical-model-specifications-basic-vs-collaborators.md](file:///Users/joko/Git/sciencemodels/AdmiralsLog/contributions/CONT6-canonical-model-specifications-basic-vs-collaborators.md))
- **Related:** `context/Hunting_Proxy_Patterns.md`, `context/John_2023_Proxy_Failure.md`, `context/Braganza_2020_Irreproducibility.md`, `AdmiralsLog/way_forward.md`, `output/reports/typology_v1.md`
- **Spawned:** [executed/P-0005](file:///Users/joko/Git/sciencemodels/AdmiralsLog/proposals/executed/P-0005-typology-distraction-destruction-counterforce.md), [P-0008](file:///Users/joko/Git/sciencemodels/AdmiralsLog/proposals/P-0008-stock-destruction-and-normalized-counterforce.md), [rejected/IE-0002](file:///Users/joko/Git/sciencemodels/AdmiralsLog/ideas/rejected/IE-0002-crowding-perception-noise-as-epistemic-destruction.md), [IE-0003](file:///Users/joko/Git/sciencemodels/AdmiralsLog/ideas/IE-0003-dimensionless-prestige-normalization.md)

# 1. Purpose: Bridging the Qualitative DAGs to Simulation Primitives

`context/Hunting_Proxy_Patterns.md` (Kapeller, Paul, Braganza 2026) establishes a minimal causal grammar for proxy failure across biological, economic, and metascientific systems:
- **Proxy ($P$):** Observable metric used to regulate or reward agents (here: prestige and topic visibility).
- **Goal ($G$):** Underlying epistemic or social objective that $P$ is meant to track (here: true knowledge discovery / `ideal_Merit`).
- **Aligned Action ($A$):** Effort that advances both $P$ and $G$ ($M \to A \to \{P, G\}$: curiosity-driven search toward high-value problems).
- **Proxy-Oriented Action ($AP_F$):** Activity motivated by $M$ that boosts $P$ without advancing (or while harming) $G$ (clustering in crowded, high-visibility topics).
- **Three Failure & Stabilizing Channels:**
  1. **Distraction (indirect crowding out):** Pursuing $AP_F$ diverts time/placement away from $A$, reducing $G$ by neglect.
  2. **Destruction (direct crowding out, $AP_F \dashv G$):** Applying $AP_F$ directly degrades $G$ (the "blue path" in Fig. 1b).
  3. **Built-in Counterforce ($P \to G$):** When higher $P$ causally facilitates $G$ (e.g., resources or genuine superstar mentorship spillovers), it attenuates proxy failure unless overwhelmed by $AP_F$.

---

# 2. Exact Mapping from Causal Typology to ABM Mechanics (Revised After `typology_v1`)

| Typology Element (`Hunting_Proxy_Patterns.md`) | ABM Counterpart in `sciencemodels` | Code / Formula Anchor & `typology_v1` Lesson |
| --- | --- | --- |
| **Motivation ($M$)** | Step-level utility maximization $R_i = (k_i/\bar{k})^{c_i} \cdot V_i^{1-c_i}$ + prestige-weighted recruitment | `agents.py:125-127`, `model.py:389-420` |
| **Goal ($G$)** | Epistemic ground truth: greedy-optimal knowledge harvest (`ideal_Merit`, `ideal_merit_cum`, and aggregate `percentage_knowledge_harvested`) | `metrics.py:53-106` |
| **Proxy ($P$)** | Agent prestige stocks (`prestige`, `prestige_vanishing`, `prestige_visibility`) and spatial visibility $V_i$ | `agents.py:64-84` |
| **Aligned Action ($A$)** | Curiosity-driven search toward high-novelty tiles ($c_i \to 1$) | `agents.py:125-127` |
| **Distraction ($AP_F$ indirect)** | Visibility-weighted movement ($1 - c_i \to 1$): agents cluster near peers on depleted tiles instead of exploring high-$k$ tiles | Validated in `welfare_probe_v1` and `typology_v1` (`Pure Distraction` `wedge_fin = -0.0002` in `M0` vs. `-0.1444` in `M1`) |
| **Destruction ($AP_F \dashv G$ direct)** | **Lesson from `typology_v1`:** Modelling destruction as candidate-perception noise ($\epsilon_{\text{eff}} = \epsilon \cdot m^\gamma$) acts as *exploratory annealing* rather than destruction (`ideal_loss_cum` drops $16.3\%$). True $AP_F \dashv G$ destruction must degrade **the knowledge stock or harvested merit directly** under crowding. | Tested in `typology_v1.md` ([rejected/IE-0002](file:///Users/joko/Git/sciencemodels/AdmiralsLog/ideas/rejected/IE-0002-crowding-perception-noise-as-epistemic-destruction.md)); stock/harvest formulation in [P-0008](file:///Users/joko/Git/sciencemodels/AdmiralsLog/proposals/P-0008-stock-destruction-and-normalized-counterforce.md) |
| **Counterforce ($P \to G$)** | **Lesson from `typology_v1`:** Unnormalized $\epsilon_i = \epsilon_0 / (1 + \kappa P_{\text{sup}})$ binds bit-wise ($114/120$ pairs) but is scale-trapped ($\le 0.0039$) because raw $\bar{P} \approx 0.025$. Superstar spillovers require dimensionless prestige normalization $\tilde{P}_{\text{sup}} = P_{\text{sup}} / \bar{P}_t$ ([IE-0003](file:///Users/joko/Git/sciencemodels/AdmiralsLog/ideas/IE-0003-dimensionless-prestige-normalization.md)). | Tested in `typology_v1.md`; normalized counterforce in [P-0008](file:///Users/joko/Git/sciencemodels/AdmiralsLog/proposals/P-0008-stock-destruction-and-normalized-counterforce.md) |

---

# 3. Substantive & Contextual Findings from Path D v1 ([typology_v1.md](file:///Users/joko/Git/sciencemodels/output/reports/typology_v1.md), 24 cells / 480 runs)

## 3.1 The Scale and Tier Split of Pure Distraction (`M0` vs. `M1`)

Look at the `Pure Distraction` row of the entry gradient table in [typology_v1.md](file:///Users/joko/Git/sciencemodels/output/reports/typology_v1.md#L147-L153) across $c \in \{0.3, 0.5, 0.7\}$:
- **In the Basic Model (`Tier M0`, $g = 0.0$):** The terminal proxy wedge (`wedge_fin = c_ideal_fin - c_local_fin`) is **$-0.0002$** (virtually zero!), and cumulative 200-step ideal-merit shortfall `ideal_loss_cum` is $0.45$ units of knowledge (out of $\approx 2.0$ initial grid units, corresponding to the $\sim 67\%$ aggregate harvest plateau documented in `CONT6`).
- **In the Mentorship Entry Model (`Tier M1`, $g = 0.1$):** The terminal proxy wedge jumps **700-fold** to **$-0.1444$**, while cumulative ideal-merit shortfall rises only modestly from $0.45$ to $0.51$.

**Contextual interpretation:** This directly reinforces the two-stage scale insight of `CONT6`. At moderate visibility weights ($c \in \{0.3, 0.5, 0.7\}$), **Pure Distraction in a closed peer community (`M0`) does not create a terminal wedge between local proxy merit and ideal merit** (`-0.0002`). The terminal proxy wedge (`-0.1444`) is overwhelmingly an **intergenerational recruitment phenomenon (`M1`)**: new entrants are funnelled onto senior stars' depleted topics, creating a stark divergence between who holds accumulated prestige and who is making genuine marginal discoveries, even though cumulative harvest shortfall moves by only $\sim 13\%$ ($0.45 \to 0.51$).

## 3.2 Why "Noisy Hot Fields" ($\epsilon_{\text{eff}} = \epsilon \cdot m^\gamma$) Help Rather Than Hurt Discovery (`IE-0002`)

In `P-0005`, we operationalized Ioannidis's (2005) claim ("crowded, hot fields generate noisy evaluations") by setting candidate-tile perception noise to $\epsilon_{\text{eff}} = \epsilon \cdot m_t(\text{candidate})^\gamma$. Instead of worsening proxy failure, setting $\gamma = 1$ **narrowed the endpoint proxy wedge by $+0.0316$** (bootstrap 95% CI $[0.0101, 0.0532]$) and **reduced cumulative welfare loss by $16.3\%$** ($-0.0783$, from $0.48$ to $0.40$)!

**Contextual interpretation (Perceptual Noise vs. Epistemic Destruction):**
Why does making crowded subfields perceptually noisier *improve* collective discovery?
- In our model, scientists suffer from **status-quo orthodoxy lock-in** (`agents.py:166`: they only leave their current topic if a neighboring topic looks strictly better). When a crowded topic has high visibility $V_i$, deterministic evaluation keeps everyone glued in place.
- Injecting higher evaluation noise around crowded topics ($\epsilon \cdot m^\gamma$) acts like **intellectual ferment / disagreement inside hot fields**: occasional optimistic misperceptions of adjacent topics kick researchers out of the herd and onto neighbouring tiles, while unoccupied frontier tiles ($m = 0$) are evaluated without crowding noise!
- **Theoretical lesson for `Hunting_Proxy_Patterns.md`:** We must distinguish **perceptual turbulence in crowded fields** (which destabilizes herds and aids exploration) from **true epistemic destruction ($AP_F \dashv G$)**, where crowding directly destroys the underlying knowledge stock or floods the literature with irreproducible false-positive claims that carry negative social value ([P-0008](file:///Users/joko/Git/sciencemodels/AdmiralsLog/proposals/P-0008-stock-destruction-and-normalized-counterforce.md)).

![Figure 1: Four-quadrant prestige vs ideal merit decay with the wedge arrow](../../data/typology_v1/figures/fig1_quadrant_pg_decay.png)

![Figure 3: wedge_fin heatmap by curiosity and quadrant, per entry regime](../../data/typology_v1/figures/fig3_wedge_fin_heatmap.png)
