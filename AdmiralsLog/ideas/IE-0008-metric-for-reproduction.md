---
title: "IE-0008 - Metric for reproduction"
author: "Paul"
date: "2026-11-24"
tags: [idea]
project: "sciencemodels"
status: idea
---

- **Status:** idea
- **Date:** 2026-11-24
- **Related:** 

## The idea

1. **Reproduction in the model should probably not be based on prestige:**
   - *Mechanism:* Using prestige as the metric for reproduction means using the goal to reward agents, in the context of the work it probably makes more sense to use a proxy measure or a mixed proxy-goal measure
   - *Examples:* One could use current visibility of a scientist, visibility multiplied by (normalized) prestige, or a balance between prestige and visibility depending on the current mean curiosity of the agents, ...
   - *Pro:* It will allow to investigate if science is selecting for people who do not have the best curiosity value
   - *Cons:* It opens the door to criticism on which metric we use. 

## Code-verified assessment

The visibility-based supervisor channel already exists. `entry_prestige_stock` is validated with `"prestige_visibility"` as an allowed value (`sciencemodels/model.py:183-190`), and `__generate_new_agents` weights supervisor draws by `getattr(a, entry_prestige_stock)` (`sciencemodels/model.py:411-423`). Setting `entry_prestige_stock="prestige_visibility"` (default `"prestige"`, `sciencemodels/model.py:144`) delivers the visibility reproduction option with no new code; a multiplicative visibility $\times$ normalised-prestige metric would still need a new stock option. The heritability premise does not hold, though: newcomers receive `self.initial_curiosity` via `agent_generation_function` (`sciencemodels/model.py:425-427`; default identity at `sciencemodels/model.py:137-139`), not the supervisor's curiosity. Curiosity is currently non-heritable, so reproduction cannot select for or against it; the proposed test requires a transmission/mutation channel first.


## Explicit selection mechanism (Gate B expansion, 2026-11-26)

To make evolutionary selection on curiosity transparent and testable (without
changing the Phase-1 paper's reproduction rule), the mechanism decomposes into
**three explicit ingredients** that together form a Darwinian selection loop:

1. **Differential reproduction (fitness = supervisor weight $W_i$):**
   When a newcomer is spawned at step $t$, supervisor $i$ is drawn with
   probability $p_i = W_i / \sum_j W_j$, comparing three explicit regimes:
   - *Output-proxy baseline:* $W_i = P_i$ (`prestige`, current default),
   - *Hybrid proxy:* $W_i = P_{\text{vis-mix}, i}$ (`prestige_visibility`, already in code),
   - *Pure attention proxy:* $W_i = V_i$ (instantaneous spatial `visibility`, new option) or $W_i = V_i \cdot \tilde{P}_i$.
2. **Trait transmission with mutation (curiosity heritability):**
   Instead of drawing newborn curiosity independently from
   `agent_generation_function(initial_curiosity)`, the newborn inherits its
   supervisor's curiosity with small bounded Gaussian mutation:
   $$c_{\text{new}} = \text{clip}\!\left(c_{\text{sup}} + \eta,\; 0,\; 1\right), \qquad \eta \sim \mathcal{N}(0, \sigma_c^2), \quad \sigma_c \in \{0.02, 0.05\}.$$
   Starting from a uniform initial cohort $c_i(0) \sim U(0, 1)$ ($\bar{c}(0) = 0.5$),
   supervisors with higher $W_i$ pass their curiosity trait $c_{\text{sup}}$
   disproportionately into the next generation.
3. **Turnover closure (finite population via `ADR-0003`):**
   Under `fixed_span` ($N \equiv 20$, 80-step career, $g = 0.25$), older
   cohorts retire as newcomers enter, so traits that fail to reproduce die out
   rather than lingering forever as immortal founders.

**Testable hypothesis:** Under $W_i = P_i$ (output), high-$c$ pioneers harvest
richer peaks early and reproduce more, shifting population mean curiosity
$\bar{c}(t)$ upward (or stabilizing it near the cognitive-diversity optimum).
Under $W_i = V_i$ (pure attention proxy), low-$c$ visibility chasers sit in
dense clusters, enjoy higher $V_i$, and out-reproduce pioneers—driving
$\bar{c}(t)$ downward toward the Stage-2 herding collapse ($\xi \ge 0.8$).
