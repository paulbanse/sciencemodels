---
title: "Path B dyadic sketch: a decision menu for collaborator choice"
version: 1
date: 2026-11-13
revision-vehicle: strikethrough
revision-renders: "md, pdf, html"
tags: [path-b, dyadic, collaborator-choice, homophily, credit-sharing, design-menu, not-calibrated]
project: sciencemodels
---

> **ARCHIVED 2026-11-15.** The design menu this document specifies was executed
> and the probe result now lives in `output/reports/path_b_probe_v1.md`. The
> owner-only decisions Q1-Q11 were ruled and are recorded there: Q1-Q8 landed
> as the implemented pins and sweep axes, while Q9 (the identity of $AP_F$)
> stays deferred to Path D and Q10-Q11 are answered by that report's
> deferred-need and no-calibration statements.

**Version 1** (November 2026)

> **DESIGN MENU, NOT A PLAN.** No code was written and no run was made for
> this document. **NOT CALIBRATED**: per the owner ruling of 2026-11-13 the
> moments below are diagnostic targets to approach in a sweep, never fitting
> targets. Every value quoted is from `docs/connections_niklas.md` (pinned
> errata included).

# Standing constraints

Path B (`docs/way_forward.md`) requires *one decision rule with one mixing
parameter*, on pain of unfalsifiability by accumulation. This menu therefore
offers two formulations that share one state, one score and one parameter
$\lambda$, and differ only in how a proposal becomes a link. The owner picks
one; the other is kept as the robustness twin.

# The dyadic step (shared by both)

State: a link set $L$ of undirected pairs. Score of a candidate partner $j$
for proposer $i$:

$$U_{ij} \;=\; (1-\lambda)\,\tilde V_j \;+\; \lambda\,\sqrt{\tilde C_{ij}\,\tilde M_j}$$

$\tilde V$ = normalised visibility stock (a choice: see Q4), $\tilde C$ =
topical closeness, $\tilde M$ = merit (a choice: see Q3). All three are
scaled to $[0,1]$; the goal term is a geometric mean so a partner who is
close *or* good but not both scores low. $\lambda\in[0,1]$ is the single new
parameter: $\lambda=0$ is pure proxy matching, $\lambda=1$ pure goal
matching.

```{.python}
# once per step, agents in random order
for i in agents:
    if not proposes(i, regime):            # regime: see Q5
        continue
    U = {(1-lam)*V[j] + lam*sqrt(C[i][j]*M[j]) for j in candidates(i)}
    # Formulation A (logit):   j = softmax_sample(U / eps)   # eps = existing noise
    # Formulation B (threshold): j = argmax(U); keep only if U[j] >= bar(i)
    propose(i, j)

# link update (identical in A and B)
if accepted(i, j):                          # mutual best / incumbent consents
    L |= {(i, j)}
for (i, j) in L:                            # dissolution
    if score(i, j) < bar(i) or score(i, j) < bar(j):
        L -= {(i, j)}
```

`bar(i)` = the score of $i$'s current partner, or of working alone if
unlinked. `C[i][j]` needs a definition the model does not yet have (Q2).

# Formulation A vs formulation B, side by side

| | **A: logit choice** | **B: threshold rule** |
| --- | --- | --- |
| proposal | sample $j$ with $P\propto e^{U_{ij}/\epsilon}$ | take $\operatorname{argmax}_j U_{ij}$, keep if it clears `bar` |
| parameter spent | $\lambda$ mixes proxy vs goal; noise is the **existing** $\epsilon$ | $\lambda$ mixes proxy vs goal; the bar is **relative** (incumbent/solo), so no new continuous knob unless an absolute $\theta$ is wanted |
| link formation | frequent, probabilistic, many weak ties | rare, all-or-nothing, strong ties |
| dissolution | same probabilistic erosion | same relative bar, so ties break when a better partner appears |
| sparsity | low: nearly everyone links | high: many agents stay solo |
| expected signature | smooth assortativity, probabilistic entry into elite links | mass at 0 and 1 in repeated ties, small teams |
| risk | too dense to hit mean team size 2.65 without an extra cap | too brittle to reproduce O/E 3.1 / 3.0 / 2.3 |

# Co-production and credit sharing

Three options, escalating in cost. The menu deliberately includes *not at
all*, because Path B's own risk note is parsimony.

- **(i) Structural only (no credit rule).** Links affect who is where and who
  harvests next to whom; prestige accrues exactly as today
  (`prestige_accrual`). Joint harvest already happens implicitly through tile
  co-location. Zero new knobs; team size and repetition become *structural*
  facts only. Weak on explaining *why* teams matter.
- **(ii) Equal split.** When $\ge 2$ linked agents harvest the same tile in a
  step, the harvested flow and its prestige are split evenly among them. No
  new continuous parameter (the split fraction is pinned at $1/n$), but it
  changes `prestige_accrual` behaviour and therefore moves the Path A wedge
  and the top-10% share *directly*, confounding choice with credit.
- **(iii) Contribution-weighted.** Shares $\beta_k$ from a contribution
  measure (knowledge before harvest, or proximity to the tile). Adds $\beta$'s
  functional form **plus** a contribution definition: two new choices, and
  the second is a measurement problem the model does not currently have.

Recommendation-shaped fact, not a recommendation: only (ii) and (iii) make
team size an *outcome* variable rather than a label, and only (iii) lets the
leader-takes-all credit rule of `connections_niklas.md` be tested.

# Which moments each formulation targets, and which knobs it adds

| | **A: logit** | **B: threshold** |
| --- | --- | --- |
| calibration targets | team size mean 2.65 (median 2); `repeated_collab_share` 0.34 with mass at 0 and at 1 | same two; the mass at 0/1 is its natural fit |
| validation targets | O/E elite cells 3.1 / 3.0 / 2.3 (dyad-corrected); Markov C retention 0.3995, entry 0.0619 | same two; brittle ties may under-shoot entry 0.0619 |
| new continuous knobs | $\lambda$ (plus $p_{propose}$ only if the universal regime is chosen) | $\lambda$ (plus $\theta$ only if an absolute bar is wanted) |
| new categorical knobs | proposal regime, directedness, team aggregation | proposal regime, directedness, team aggregation |
| reuses existing | $\epsilon$, `entry_prestige_stock` (for $\tilde V$), ideal/local merit reporters, grid distance, `agent_generation_time` | same |

Secondary targets both formulations inherit: collaboration index 1.6 -> 2.4
and collaborative share 46% -> 79% over 1990-2018 (dynamic, low priority).

# Open questions: decisions only the owner can make

| # | Decision | Options |
| --- | --- | --- |
| Q1 | Formulation | A, B, or both as primary+twin |
| Q2 | Topical closeness in a model with no topics | grid distance / overlap of harvested-knowledge profiles / curiosity similarity |
| Q3 | Which merit enters $\tilde M$ | `ideal_Merit` (omniscient, the yardstick) vs `current_local_merit` (observable) |
| Q4 | Which visibility stock for $\tilde V$ | `prestige_visibility` / raw `prestige` / the swept `entry_prestige_stock` |
| Q5 | Who proposes | universal ($p_{propose}$) / newcomer-driven (mirrors the supervisor channel) / prestige-driven / mutual-best only |
| Q6 | Link and team objects | undirected vs directed; team = clique, connected component, or an explicit team object (decides whether 2.65 is reachable) |
| Q7 | Relation to the existing channel | does the dyadic layer **replace** the supervisor roulette (`model.py:315-326`) or run alongside it? |
| Q8 | Credit sharing | (i) structural only, (ii) equal split, (iii) contribution-weighted; and if (ii)/(iii), does `prestige_accrual` stay `harvested`? |
| Q9 | **What plays $AP_F$ (the gaming action) in Path D** | (a) an explicit effort split into visibility work that buys prestige at the cost of harvest; (b) *proposal for visibility*: a link whose score is dominated by $\tilde V$, i.e. $\lambda\to 0$ read as a gaming margin; (c) credit grab (leader-takes-all in Q8-iii); (d) reciprocal citation / visibility trading between linked agents; (e) topic-switching toward visible questions. Companion choice: does $AP_F$ **distract** (crowd out $A$) or **destroy** (harm $G$)? |
| Q10 | Time scale | how many steps are one paper and one link? Markov C retention 0.3995 is per what interval? |
| Q11 | Stance on moments | confirm they stay *diagnostic* under the no-calibration ruling, even though `way_forward.md` labels team size and repeated share as calibration moments |

Gate 2 (`way_forward.md`) still governs: if the dyadic layer does not
materially improve the team and closure moments, the honest paper is A+D and
this layer is reported as robustness.

# Revision Log

- **Version 1** (November 2026) --- Initial Path B design menu: shared dyadic
  skeleton with one mixing parameter, logit vs threshold formulations
  side by side, three credit-sharing options, a moments-and-knobs table, and
  eleven owner-only decisions including the identity of $AP_F$. No code, no
  runs.
