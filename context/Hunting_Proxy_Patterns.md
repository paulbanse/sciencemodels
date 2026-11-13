---
title: "Hunting the Causal Patterns Behind Proxy Failure (Kapeller, Paul, Braganza, 2026)"
version: 1.0
date: 2026-09-22
tags: [hunting-proxy, causal-patterns, proxy-failure, work-in-progress, MO-review]
project: niklasklann
---

# Hunting the Causal Patterns Behind Proxy Failure

**Jakob Kapeller, Paul (surname not given), Oliver Braganza**  
Work-in-progress note, March 2026.

## Purpose

An informal, collaborative note mapping **typical causal structures** that give rise to **proxy failure (PF)**, with the goal of establishing a shared conceptual vocabulary and identifying recurring motifs across domains.

## Core Framework

### Necessary ingredients (minimal consensus)

A proxy-failure situation minimally requires:

| Element | Role |
|---------|------|
| Proxy $P$ | Observable measure used to regulate agents. |
| Goal $G$ | The underlying objective $P$ is supposed to track. |
| Action $A$ | Activity that supports both $P$ and $G$ (or could). |
| Alternative action $AP_F$ | Activity that boosts $P$ **at the expense of** $G$. |
| Motivation $M$ | Why $AP_F$ is attractive (incentive, reward). |

### Three paths from $P$ to $G$ (or lack thereof)

1. **Direct**: $P \to G$ or $G \to P$.
2. **Confounding** via a third factor $A$: $P \leftarrow A \to G$.
3. **No link** (independent): $P$ and $G$ are unrelated — the proxy is uninformative.

### The failure mechanism

$AP_F$ always **decreases** the $P$–$G$ correlation via:
- **Distraction (indirect crowding out)**: less attention to $G$ because it is easier to raise $P$ alone.
- **Destruction (direct crowding out)**: applying $AP_F$ to raise $P$ directly harms $G$.

### A built-in counterforce

When $P \to G$ exists, there is a counterforce: increasing the proxy also benefits the goal. This can stabilise honest signalling — but is **not guaranteed** to dominate.

## Standard Mappings (Fig. 1a / 1b)

### Minimal consensus model (Fig. 1a)

$$M \to AP_F \to P, \quad M \to A \to P, \quad AP_F \dashv G, \quad A \to G, \quad P \leftrightarrow G$$

### Extended model (Fig. 1b)

Adds:
- Negative direct edge $AP_F \dashv G$ (crowding-out).
- Possible $P \leftrightarrow G$ interactions.
- Cross-action interactions between $A$ and $AP_F$.

## Documented Examples

| Domain | Proxy $P$ | Goal $G$ | $AP_F$ (PF driver) |
|--------|-----------|----------|---------------------|
| **Profit & Welfare** | Firm profit | Social welfare | Negative externalities, exploitation |
| **The Cobra Case** | Rat tails delivered | Reduced cobra population | Breeding cobras for tails |
| **Dead Rats / Science** | Publication count / impact | Scientific truth | Questionable research practices, salami slicing |
| **Incentives in Science** | Publications / KPIs | Reliable knowledge | P-hacking, HARKing, low-powered studies |
| **Peacocks** | Tail length | Offspring fitness | Fisherian "sexy sons" runaway |
| **Diet** | Sweet taste receptor activation | Nutritional status | Artificial sweeteners (aspartame) |
| **SEO** | Search-engine visibility | Information relevance | Keyword stuffing, clickbait |
| **Addiction** | Dopamine signal | Biological fitness / wellbeing | Substance-induced proxy hacking |
| **Collaborator Choice** | Visible partner | Deep research | Selecting prestigious but shallow collaborators |
| **Generics in Science** | Wide-ranging claims | Accurate information | Over-generalising for visibility |

## Cross-Cutting Insights

1. **Confounding is the rule, not the exception**: In almost all examples, $P$ is a confounder between $G$ and other factors, not a pure mediator.
2. **Motivation $M$ is always aligned with $A$**: The origin of $AP_F$ is the same as the origin of $A$ — agents can redirect effort from "good" to "gaming" without leaving their existing niche.
3. **Blue path matters**: The negative link $AP_F \dashv G$ is what distinguishes genuine proxy failure from mere noise in the proxy-goal mapping.
4. **Counterforce is conditional**: Where $P \to G$ is strong, PF is attenuated; where it is weak (e.g., purely symbolic proxies), PF dominates.
5. **Hierarchy and cascade are implicit**: When $AP_F$ at one level feeds into $P$ at a higher level, failures can cascade (anticipating John et al. 2023, Section 5).

## Position Relative to Published Work

- **Pre-formalisation** companion to John et al. (2023): the causal diagrams here ground the narratives in the BBS paper.
- **Bridge to Braganza (2020)**: the science-incentives examples directly reference the ESS model.
- **Methodological contribution**: the attempt to extract a **typology** of PF (distraction vs. destruction, direct vs. confounding) is a step toward formal classification.

## Relevance to MA Klann

- The **incentives-in-science** example maps directly onto the co-authorship-as-proxy idea: if high-rank co-authorship is a proxy for research quality, then choosing co-authors for rank rather than intellectual fit is a PF dynamic.
- The note documents the **conceptual genealogy** that underpins the proposal's use of proxy failure as a lens for analysing collaboration networks.

## MO Assessment

| Dimension | Rating |
|-----------|--------|
| Analytical clarity | Medium — the causal diagrams are transparent, but the text is informal and iterative. |
| Exemplary breadth | High — covers biological, social, and metascience cases. |
| Typological progress | Medium — the minimal consensus model is promising, but the extended version is still exploratory. |
| Formality | Low — no equations, no estimation; purely qualitative causal reasoning. |
| Integration potential | High — directly usable for designing formal models or coding schemes. |
