---
title: "CONT5 - Generational Turnover, Paradigm Lock-In, and Spatial Partner Coupling (Wave-4 Synthesis)"
author: "Antigravity (Wave-4 Synthesis Round)"
date: "2026-11-15"
tags: [contribution, turnover, steady-state, interventions, paradigm-lockin, path-b-v2, publishable-results, m1-m2]
project: "sciencemodels"
---

- **Source:** Synthesis of Wave-4 external execution rounds `F1`–`F4` (`path_b_probe_v2`, `interventions_v1`, `typology_v1`, `turnover_v1` — 198 cells / 3,960 runs), updated with real-world contextual interpretations and `M0`/`M1`/`M2` + scale framing (`CONT6`)
- **Date:** 2026-11-15 (v2 — expanded with contextual interpretation and scale analysis)
- **Model Tier:** **Tier `M1` (Open-Cohort Mentorship / Stationary Turnover)** in Sections 2–4; **Tier `M1 + M2` (Entry + Explicit Dyadic Collaboration with Movement Pull)** in Section 5 (see [CONT6-canonical-model-specifications-basic-vs-collaborators.md](file:///Users/joko/Git/sciencemodels/AdmiralsLog/contributions/CONT6-canonical-model-specifications-basic-vs-collaborators.md))
- **Related:** `output/reports/turnover_v1.md`, `output/reports/interventions_v1.md`, `output/reports/path_b_probe_v2.md`, `output/reports/typology_v1.md`, `AdmiralsLog/contributions/CONT1-depletion-counterforces-refills.md`, `AdmiralsLog/contributions/CONT3-publishable-results-synthesis.md`, `AdmiralsLog/contributions/CONT6-canonical-model-specifications-basic-vs-collaborators.md`
- **Spawned:** [P-0007](file:///Users/joko/Git/sciencemodels/AdmiralsLog/proposals/P-0007-sabbatical-shocks-x-frontier-refills.md), [P-0008](file:///Users/joko/Git/sciencemodels/AdmiralsLog/proposals/P-0008-stock-destruction-and-normalized-counterforce.md), [P-0009](file:///Users/joko/Git/sciencemodels/AdmiralsLog/proposals/P-0009-core-turnover-and-steady-state-interventions.md), [P-0010](file:///Users/joko/Git/sciencemodels/AdmiralsLog/proposals/P-0010-dyadic-link-dissolution-and-shift-gate.md), [P-0011](file:///Users/joko/Git/sciencemodels/AdmiralsLog/proposals/P-0011-canonical-three-tier-model-specification.md), `IE-0003`–`IE-0005`, `UE-0001`–`UE-0003`

# 1. Why Wave 4 (`F1`–`F4`) Changes the Paper's Substantive Arc

Waves 1–3 (`CONT3`) established the relational prestige-merit wedge, the newcomer-incumbent gap, and the $\xi^* \in [0.6, 0.8]$ welfare kink in a world where **founding professors never retired**, **new scientific questions never emerged**, **institutional counterforces were untested**, and **dyadic partner choice did not pull spatial movement**.

Wave 4 (`F1`–`F4`, 198 cells / 3,960 runs, 0 failures) removed each of those four brackets and produced **four new substantive findings about the modelled world** — while clarifying the exact scale of output vs. distributional distortion (`CONT6`).

---

# 2. Result 4: Immortal Founders Masked Proxy Failure — Generational Turnover Deepens It $2.8\times$ to $6.6\times$ (`turnover_v1`, Tier `M1`)

In [turnover_v1.md](file:///Users/joko/Git/sciencemodels/output/reports/turnover_v1.md) ([executed/P-0006](file:///Users/joko/Git/sciencemodels/AdmiralsLog/proposals/executed/P-0006-demographic-turnover-steady-state.md), 54 cells / 1,080 runs at $H = 400$), the unbounded-population `M1` model (`none`, $N_t = 20 + \lfloor gt \rfloor$) was tested against two retirement rules:
- **`fixed_span` ($N \equiv 20$ stationary population):** Every newcomer arrival retires the oldest active scientist.
- **`age_hazard` ($\tau_{\max} = 200$ stationary tenure cap):** Scientists reaching 200 steps of tenure retire and are replaced one-for-one via the supervisor roulette.

| Exit Mode | $N$ (last 100) | Mean Tenure | `gap_recent` (tenure $<20$ vs $\ge 20$) | $c_{\text{ideal}}$ Final | $c_{\text{ideal}} < 0$ (% of runs) | $c_{\text{ideal}}$ Time-Mean | Harvested Share (Final) | Top-10% Prestige Share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `none` (immortal founders) | 61.45 | 267.6 | $-0.01362$ | $-0.0860$ | $75.6\%$ | $+0.1141$ | **$78.7\%$** | $0.2605$ |
| `fixed_span` ($N \equiv 20$) | 20.00 | 110.8 | $-0.00612$ (**$-55.0\%$**) | **$-0.2401$** (**$2.79\times$**) | **$89.2\%$** | $+0.1042$ | **$61.1\%$** | $0.2327$ |
| `age_hazard` ($\tau_{\max}=200$) | 61.45 | 61.1 | $-0.00282$ (**$-79.3\%$**) | **$-0.5678$** (**$6.61\times$**) | **$100.0\%$** | $+0.1112$ | **$78.8\%$** | **$0.4086$** |

**Contextual & Scale Interpretation (What This Means for Real Academic Demographics):**
1. **Aggregate output is identical (`78.8%` vs. `78.7%`), yet reputional distortion is $6.6\times$ worse under `age_hazard`:**
   Look at the `age_hazard` row vs. `none`: both have the exact same headcount ($61.45$ scientists) and harvest the **exact same share of total landscape knowledge ($78.75\%$ vs. $78.69\%$)**! Yet under `age_hazard`, the top-10% prestige share surges from **$26.1\%$ to $40.9\%$** and terminal `Corr_prestige_ideal_merit` plunges from **$-0.0860$ to $-0.5678$ (negative in $100\%$ of runs)**!
   - *Why?* Because in the immortal-founder model (`none`), the 20 founding pioneers who discovered genuine peaks at $t=0$ stayed alive forever, anchoring a positive link between historical achievement and prestige. When those founders retire (`fixed_span` or `age_hazard`), they pass their departmental slots and topical locations (`sup.pos`) down to second- and third-generation advisees via the prestige roulette. Those successor cohorts inherit *exhausted paradigms* where prestige is concentrated in the biggest academic dynasties (`top10 = 40.9%`), while actual frontier discovery happens among isolated low-prestige outsiders (`c_ideal = -0.568`).
2. **Separating Pioneer Longevity ($55\%\text{–}79\%$) from Structural Cohort Handicap ($21\%\text{–}45\%$):**
   The prestige gap between recent arrivals (tenure $< 20$) and established scholars (`gap_recent`) remains negative in **all 1,080 of 1,080 runs**, proving that junior disadvantage is a permanent feature of prestige-mediated reproduction. At the same time, retiring founders shrinks the gap's size by $55\%\text{–}79\%$, showing that over half of the raw status gap in growing fields is simply pioneer longevity.
3. **The Life-Cycle of a Scientific Field — Discovery Alignment ($+0.11$) vs. Mature Orthodoxy ($-0.57$):**
   Across all three exit modes, the **whole-run time mean** of $c_{\text{ideal}}$ is **positive** ($+0.104$ to $+0.114$), whereas the **mature end-of-run state** is strongly **negative** ($-0.086$ to $-0.568$). In a young scientific field (steps 1–100), prestige works as intended: the pioneers who find the major peaks earn the highest prestige. In a mature, depleted field (steps 200–400), the very same reward rules turn pathological: prestige locks in around the heirs of old peaks and anti-tracks current epistemic value ([IE-0004](file:///Users/joko/Git/sciencemodels/AdmiralsLog/ideas/IE-0004-discovery-vs-depletion-window-reporters.md)).

---

# 3. Result 5: Mid-Career Mobility Fellowships Outperform Entry Lotteries by $40\times$ (`interventions_v1`, Tier `M1`)

In [interventions_v1.md](file:///Users/joko/Git/sciencemodels/output/reports/interventions_v1.md) ([executed/P-0003](file:///Users/joko/Git/sciencemodels/AdmiralsLog/proposals/executed/P-0003-institutional-counterforces-lotteries.md), 72 cells / 1,440 runs), crossing supervisor-roulette damping ($\alpha \in \{1.0, 0.5, 0.0\}$) with sabbatical relocation shocks ($p_{\text{reloc}} \in \{0.00, 0.02\}$) across visibility weights $1-c \in \{0.5, 0.7, 0.9\}$ yielded a decisive policy ranking:
- **Sabbatical/maverick relocation (`p_reloc = 0.02`)** cuts the high-visibility welfare loss penalty by **$40.0\%$** ($0.00929 \to 0.00557$) and convexity by **$26.8\%$**.
- **Pure entry lotteries ($\alpha = 0.0$)** cut the high-visibility welfare loss penalty by only **$0.9\%$** ($0.00929 \to 0.00921$).

**Contextual significance:** Randomizing who gets hired or funded at entry ($\alpha = 0$) cannot prevent conformity cascades if researchers still face intense visibility pressure ($1-c \ge 0.7$) throughout their careers. Only interventions that fund **mid-career topical exits** (`p_reloc = 0.02`, e.g., sabbatical fellowships to switch subfields once a topic is exhausted) break spatial herd lock-in.

---

# 4. Result 6: Normal-Science Bailouts vs. Neglected Kuhnian Frontiers (`interventions_v1`, Tiers `M0` & `M1`)

Testing `new_questions` regeneration ([interventions_v1.md](file:///Users/joko/Git/sciencemodels/output/reports/interventions_v1.md#L129-L165), [executed/P-0004](file:///Users/joko/Git/sciencemodels/AdmiralsLog/proposals/executed/P-0004-paradigm-refill-regimes.md)) uncovered a stark contrast between **incumbent hot-spot bailouts (`occupied`)** and **unoccupied paradigm emergence (`frontier`)** at $g = 0.1$:
- **Normal-Science Exploitation Lock-In (`occupied`):** When depleted topics regenerate under existing crowds, scientists cease exploring (`explored` collapses by **$84\%$**, from $37.7\%$ to $6.1\%$ of the grid) and the newcomer gap widens **$11\times$** to $-0.451$.
- **Unclaimed-Frontier Lock-In (`frontier`):** When new questions open on unoccupied tiles, $c_{\text{ideal}}$ flips positive ($+0.078$) for the few mavericks who reach them, but the visibility-bound majority stays behind in depleted clusters — leaving **unclaimed frontier knowledge $315\times$ higher than baseline** ($0.1895$ vs. $0.0006$).

---

# 5. Result 7: Spatial Partner Pull Couples Collaborator Choice to Co-Harvest Team Size and the Wedge (`path_b_probe_v2`, Tier `M1 + M2`)

In [path_b_probe_v2.md](file:///Users/joko/Git/sciencemodels/output/reports/path_b_probe_v2.md) (48 cells / 960 runs), introducing `dyadic_move_pull` $\in \{0.5, 1.0\}$ connected declared co-authorship links (`model.links`) to spatial movement and co-harvesting:
- On `mgstd5`, increasing partner-choice goal weight from $\lambda = 0$ (choosing co-authors for prestige) to $\lambda = 1$ (choosing co-authors for local merit and topical closeness) at `pull = 1.0` reduces co-harvest `mean_team_size` from **$7.32$ (un-pulled) / $5.33$ ($\lambda=0$) down to $3.42$** ($2.89$ in the logit-structural arm, against the $2.65$ empirical reference; **4/4 `mgstd5` contrasts pass**).
- **Contextual meaning of the $\lambda$ wedge flip:** When scientists choose co-authors for **prestige ($\lambda = 0$)** and pull toward their partners' topics (`pull = 1.0`), co-authorship acts as an **spatial magnet toward exhausted elite clusters**, reducing time-mean `Corr_prestige_ideal_merit` by **$-0.0664$** ($[-0.0774, -0.0558]$). Conversely, when scientists choose co-authors for **current epistemic merit and topical complementarity ($\lambda = 1$)**, partner pull draws collaborators toward active discovery sites, raising `Corr_prestige_ideal_merit` by **$+0.0522$** ($[+0.0425, +0.0622]$) and flipping the time-mean wedge from $+0.035$ to $-0.037$.
- **Two open follow-ups (`P-0010` and `P-0011`):**
  1. Adding a post-project link dissolution hazard ([P-0010](file:///Users/joko/Git/sciencemodels/AdmiralsLog/proposals/P-0010-dyadic-link-dissolution-and-shift-gate.md)) to bring `repeated_collab_share` down from $0.91$ toward the empirical $0.34$ reference.
  2. Running `M2` on top of the closed-cohort Basic Model `M0` ($g = 0$, [P-0011](file:///Users/joko/Git/sciencemodels/AdmiralsLog/proposals/P-0011-canonical-three-tier-model-specification.md)) to isolate pure horizontal co-authorship from vertical PhD entry (`M1`).
