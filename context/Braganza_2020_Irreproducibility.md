---
title: "Braganza (2020) – A Simple Model Suggesting Economically Rational Sample-Size Choice Drives Irreproducibility"
version: 1.0
date: 2026-09-22
tags: [Braganza-2020, reproducibility, sample-size, proxyeconomics, MO-review]
project: niklasklann
---

# A Simple Model Suggesting Economically Rational Sample-Size Choice Drives Irproducibility

**Oliver Braganza**  
*PLoS ONE*, 15(3), e0229615 (2020). DOI: 10.1371/journal.pone.0229615

## Core Claim

Under **positive publication bias** and **competitive funding**, the **economically rational (profit-maximising) sample size** is frequently small, leading to a bimodal distribution of achieved power and low overall reproducibility. The paper formalises this intuition in a minimal model and derives empirically testable predictions.

## Model Setup

### Three exogenous parameters

| Parameter | Meaning |
|-----------|---------|
| $b$ | Pre-study probability that the hypothesis is true (base rate). |
| $d$ | True effect size. |
| $IF$ | Income from publication / funding per positive result (in units of "samples purchasable per publication"). |

### Key structure

- Scientists receive income **only** from positive findings (positive publication bias).
- Cost is linear in sample size: $C(n) = c \cdot n$.
- Power is a function of $(b, d, n)$ via standard normal formulas.
- Expected profit = $PPV(n) \cdot P(\text{significance} \mid n) \cdot IF - C(n)$.
- The **Equilibrium Sample Size (ESS)** maximises expected profit.

## Key Predictions

| Prediction | Empirical status (as assessed in paper) |
|------------|----------------------------------------|
| (i) $b$ ↔ sample size correlation | Confirmed: high-impact / novel-finding journals have **smaller** samples. |
| (ii) $d$ ↔ sample size correlation | Ambiguous: observed negative correlation likely driven by winner's curse. |
| (iii) $IF$ ↔ sample size correlation | Indirectly supported; larger sample sizes where cost per sample is lower. |
| (iv) **Bimodal power distribution** | Strongly confirmed — lack of a mode at 80% power in both model and data. |
| (v) Low overall reproducibility | Confirmed: model predictions match observed reproducibility rates (11–62% depending on field). |

## Relation to Proxyeconomics

- The model is an **instance of proxyeconomics**: the proxy is "publication output"; the goal is "scientific progress".
- Positive publication bias is an **informational asymmetry** that drives proxy divergence.
- The "ESS" is the analogue of "proxy equilibrium" — a stable point where agents cannot improve by changing sample size, but the system-level outcome (low reproducibility) is collectively undesirable.

## Policy Implications Explored

1. **Increase funding per publication (IF)** → shifts ESS toward larger $n$.
2. **Conditional Equivalence Testing (CET)** → allows publication of well-powered negative findings, removing the asymmetry.
3. **More confirmatory research** (higher $b$) → shifts the ESS toward larger $n$.
4. **Stringency proposals** ($\alpha = 0.005$) → can worsen the problem by making small-$b$ niches unprofitable, increasing waste.
5. **Independent confirmatory branch** → long-term structural fix (Utzerath & Fernandez proposal).

## Robustness & Caveats

- Predictions are **conservative**: if true effect sizes are smaller than published estimates (winner's curse), ESS would be even lower.
- Cost function is linear; strong non-linearities could shift ESS but require implausibly steep curvature.
- Scientific deliberation and ethical norms are acknowledged but not modelled; the model predicts the floor, not necessarily the realised sample size.
- **No dynamic equilibrium**: the ESS is a static optimum; cultural-evolution interpretation (selection across niches) is offered as an alternative.

## Relevance to the Wider Project

- Provides a **formal microfoundation** for why "irreproducibility" is a systemic feature, not a collection of individual failures.
- The proxy-failure framing (John et al. 2023) is explicitly referenced: publication count is the proxy, truth is the goal.
- Used in the proposal "Consequences of Collaboration" to argue that economic pressures shape scientific output beyond individual intentions.

## MO Assessment

| Dimension | Rating |
|-----------|--------|
| Formal clarity | High — minimal, transparent derivations; predictions cleanly mapped to empirical patterns. |
| Empirical validation | Medium — predictions are compared with existing studies, but no new data or formal model fitting. |
| Scope conditions | Well delineated — assumptions about positive publication bias, linear cost, and competitive funding are explicit. |
| Policy relevance | High — generates concrete, testable levers. |
| Integration with proxyeconomics | High — the paper is a bridge between formal metascience and the proxy-failure framework. |
