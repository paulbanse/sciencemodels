---
title: "IE-0003 - Dimensionless prestige normalization (P_i / mean_P) across all multiplicative couplings"
author: "Antigravity"
date: "2026-11-15"
tags: [idea, prestige-scale, normalization, visibility-reward, counterforce, setup]
project: "sciencemodels"
status: idea
---

- **Status:** idea
- **Date:** 2026-11-15
- **Related:** `AdmiralsLog/contributions/CONT4-inert-and-invariant-degrees-of-freedom.md` (Section 2.2), `AdmiralsLog/proposals/P-0008-stock-destruction-and-normalized-counterforce.md`, `output/reports/welfare_probe_v1.md`, `output/reports/typology_v1.md`

## The idea

In `sciencemodels`, an individual scientist's raw `prestige` is a small float averaging **$\bar{P} \approx 0.0254$** (canary anchor). Because of this numerical scale, two different mechanisms that coupled raw prestige multiplicatively via $(1 + P_i)$ or $1 / (1 + \kappa P_{\text{sup}})$ with $O(1)$ coefficients turned out to be near-inert ($< 2.6\%$ perturbation):
1. `use_visibility_reward = True` in `sciencemodels/agents.py:100-123` (`(1 + peer.prestige)` moves cumulative welfare loss by at most $6.1 \times 10^{-4}$ or $2.6\%$ in `welfare_probe_v1`).
2. Superstar counterforce $\epsilon_i = \epsilon_0 / (1 + \kappa P_{\text{sup}})$ at $\kappa = 1.0$ in `typology_v1` (moves all four outcome metrics by $\le 0.0039$, 95% CIs spanning 0).

**Fix:** Define a dimensionless step-normalized prestige helper on `MyModel`:

$$\tilde{P}_i(t) = \frac{P_i(t)}{\max(\bar{P}(t), 10^{-12})}$$

so that $\mathbb{E}[\tilde{P}_i(t)] \equiv 1.0$ at every step and top-decile scientists have $\tilde{P}_i \approx 2.5\text{–}4.0$, independent of harvest rate $h$, grid size, or run horizon $H$. Use $\tilde{P}_i$ in `P-0008` ($\kappa$ counterforce) and as an optional normalized visibility mode (`use_visibility_reward = "normalized"`).
