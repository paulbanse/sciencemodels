---
title: "CONT2 - Formalizing the Hunting_Proxy_Patterns Typology in the Epistemic ABM"
author: "Antigravity (Review & Synthesis Round)"
date: "2026-11-14"
tags: [contribution, typology, proxy-failure, distraction, destruction, counterforce, path-d]
project: "sciencemodels"
---

- **Source:** Antigravity inspection & evaluation round (Item 1.4)
- **Date:** 2026-11-14
- **Related:** `context/Hunting_Proxy_Patterns.md`, `context/John_2023_Proxy_Failure.md`, `context/Braganza_2020_Irreproducibility.md`, `AdmiralsLog/way_forward.md`
- **Spawned:** `P-0005`

# 1. Purpose: Bridging the Qualitative DAGs to Simulation Primitives

`context/Hunting_Proxy_Patterns.md` (Kapeller, Paul, Braganza 2026) establishes a minimal causal grammar for proxy failure across biological, economic, and metascientific systems:
- **Proxy ($P$):** Observable metric used to regulate or reward agents.
- **Goal ($G$):** Underlying epistemic or social objective that $P$ is meant to track.
- **Aligned Action ($A$):** Effort that advances both $P$ and $G$ ($M \to A \to \{P, G\}$).
- **Proxy-Oriented Action ($AP_F$):** Activity motivated by $M$ that boosts $P$ without advancing (or while harming) $G$.
- **Three Failure & Stabilizing Channels:**
  1. **Distraction (indirect crowding out):** Pursuing $AP_F$ diverts time/placement away from $A$, reducing $G$ by neglect.
  2. **Destruction (direct crowding out, $AP_F \dashv G$):** Applying $AP_F$ directly degrades $G$ (the "blue path" in Fig. 1b).
  3. **Built-in Counterforce ($P \to G$):** When higher $P$ causally facilitates $G$ (e.g., resources or genuine superstar spillovers), it attenuates proxy failure unless overwhelmed by $AP_F$.

The `sciencemodels` ABM already contains the exact primitives needed to turn this qualitative typology into a formal, parameterized $2 \times 2 \times 2$ phase diagram (Path D of `AdmiralsLog/way_forward.md`).

---

# 2. Exact Mapping from Causal Typology to ABM Mechanics

| Typology Element (`Hunting_Proxy_Patterns.md`) | ABM Counterpart in `sciencemodels` | Code / Formula Anchor |
| --- | --- | --- |
| **Motivation ($M$)** | Step-level utility maximization $R_i = (k_i/\bar{k})^{c_i} \cdot V_i^{1-c_i}$ + prestige-weighted recruitment | `agents.py:125-127`, `model.py:315-326` |
| **Goal ($G$)** | Epistemic ground truth: greedy-optimal knowledge harvest (`ideal_Merit` and cumulative `ideal_merit_cum`) | `metrics.py:53-106` |
| **Proxy ($P$)** | Agent prestige stocks (`prestige`, `prestige_vanishing`, `prestige_visibility`) and spatial visibility $V_i$ | `agents.py:64-84` |
| **Aligned Action ($A$)** | Curiosity-driven search toward high-novelty tiles ($c_i \to 1$) | `agents.py:125-127` |
| **Distraction ($AP_F$ indirect)** | Visibility-weighted movement ($1 - c_i \to 1$) and low-$\lambda$ partner choice: agents cluster near peers on depleted tiles instead of exploring high-$k$ tiles | Already active in baseline model (`welfare_probe_v1` kink $\xi^* \in [0.6, 0.8]$) |
| **Destruction ($AP_F \dashv G$ direct)** | Crowding-induced epistemic degradation: when $m > 1$ agents stack on a visible tile, either (a) perception noise $\epsilon_{\text{eff}} = \epsilon \cdot m^\gamma$ rises ("hot fields generate false positives," Ioannidis 2005) or (b) congestion destroys unharvested tile knowledge | 1-parameter extension ($\gamma \ge 0$, with $\gamma = 0$ = pure distraction) |
| **Counterforce ($P \to G$)** | Superstar epistemic spillovers (Azoulay et al. 2010): high-prestige supervisors or collaborators reduce an agent's perception noise $\epsilon_i = \epsilon_0 / (1 + \kappa P_{\text{sup}})$ or expand search radius from von Neumann ($r=1$) to Moore ($r=1.5$) | 1-parameter extension ($\kappa \ge 0$, with $\kappa = 0$ = no counterforce) |

---

# 3. Theoretical Contribution: Distinguishing Distractive, Destructive, and Functional Prestige Regimes

Formalizing the three causal channels in one spatial-epistemic framework clarifies a central ambiguity in the metascience and Goodhart literature (John et al. 2023; Braganza 2020; Ioannidis 2005; Azoulay et al. 2010):

1. **Pure Distraction (Baseline Model):**
   $P$ and $G$ are initially confounded by tile quality ($P \leftarrow A \to G$, $\text{Corr}(P, G) \approx 1.0$ at step 1). As visibility pressure ($1-c$) or prestige-proportional entry pulls scientists toward crowded regions ($AP_F$) rather than unexplored peaks ($A$), collective harvest $G$ stagnates by *spatial neglect* rather than active falsification.
2. **Destruction ($AP_F \dashv G$, the "Blue Path"):**
   When crowding on hot, visible topics directly degrades epistemic reliability (amplifying perception noise or destroying unharvested signal, as in Ioannidis 2005), proxy competition actively harms $G$ even in the absence of demographic entry.
3. **Conditional Counterforce ($P \to G$, Superstar Spillover):**
   When prestige reflects or confers genuine epistemic capability (reducing perception noise or expanding neighborhood awareness, as in Azoulay et al. 2010), moderate prestige weighting can improve collective discovery up to an interior threshold before the proxy treadmill overwhelms the counterforce.

*(The concrete experimental specification to sweep these three channels and generate the $2 \times 2 \times 3$ regime diagram is filed separately in [P-0005-typology-distraction-destruction-counterforce.md](file:///Users/joko/Git/sciencemodels/AdmiralsLog/proposals/P-0005-typology-distraction-destruction-counterforce.md).)*
