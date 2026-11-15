---
title: "IE-0002 - Candidate-perception crowding noise (epsilon * m^gamma) as epistemic destruction (AP_F -| G)"
author: "Antigravity"
date: "2026-11-15"
tags: [idea, rejected, typology, path-d, destruction, perception-noise, annealing, typology-v1]
project: "sciencemodels"
status: rejected
---

- **Status:** rejected
- **Date:** 2026-11-14 (captured in `CONT2` v1 / `P-0005`); 2026-11-15 (rejected as a destruction mechanism after `output/reports/typology_v1.md`)
- **Related:** `AdmiralsLog/proposals/P-0005-typology-distraction-destruction-counterforce.md`, `AdmiralsLog/proposals/P-0008-stock-destruction-and-normalized-counterforce.md`, `output/reports/typology_v1.md`, `AdmiralsLog/contributions/CONT2-hunting-proxy-patterns-formalization.md`, `AdmiralsLog/contributions/CONT4-inert-and-invariant-degrees-of-freedom.md` (Section 3.2)

## The idea

To operationalize the **Destruction ($AP_F \dashv G$)** channel of `context/Hunting_Proxy_Patterns.md` (where proxy-oriented crowding directly damages the epistemic goal $G$, cf. Ioannidis 2005 on hot fields), `P-0005` multiplied candidate-neighbor perception noise by tile occupancy: $\epsilon_{\text{eff}} = \epsilon \cdot m_t(\text{candidate})^\gamma$ with $\gamma = 1$.

## Why it was rejected as a model of destruction (`output/reports/typology_v1.md`)

Across 24 cells / 480 runs (`typology_v1`), setting $\gamma = 1$ worked **in the exact opposite direction of epistemic destruction**:
- It **narrowed the endpoint proxy wedge (`wedge_fin`) by $+0.0316$** (bootstrap 95% CI $[0.0101, 0.0532]$, excluding 0).
- It **reduced cumulative ideal-merit welfare loss (`ideal_loss_cum`) by $16.3\%$** ($-0.0783$, 95% CI $[-0.0977, -0.0598]$, from $0.48$ to $0.40$).

**Three mechanical reasons why candidate-perception noise acts as *simulated annealing* rather than *destruction*:**
1. **Breaking strict-improvement lock-in (`agents.py:166`):** In `Scientist.step`, the current tile's utility is evaluated noise-free, while neighbor candidates receive multiplicative noise $1 + \epsilon_{\text{eff}} \cdot \text{Uniform}(-1, 1)$. Amplifying $\epsilon_{\text{eff}}$ on crowded neighbor tiles produces upward noise spikes that help trapped agents clear the strict-improvement threshold and scatter off depleted proxy peaks.
2. **Noise-free frontier reads at $m = 0$:** Under the literal formula $m^\gamma$, an unoccupied frontier tile ($m = 0$) receives $\epsilon_{\text{eff}} = 0$, giving agents a noiseless read on virgin territory.
3. **Fractional-power `NaN` wall at $m \ge 11$:** When $m \ge 11$ at $\epsilon = 0.1$, $1 + \epsilon_{\text{eff}} \cdot \text{Uniform}(-1, 1)$ can go negative; raising negative perceived novelty to fractional curiosity power $c \in \{0.3, 0.5, 0.7\}$ yields `NaN`, which evaluates `False` in `>` comparisons and blocks extreme pile-ups.

**Lesson carried forward (`P-0008`):** To model true epistemic destruction ($AP_F \dashv G$), crowding must degrade **the knowledge stock or harvested merit directly** (e.g., over-trampling depletes tile knowledge without yielding ideal merit), not candidate perception noise.
