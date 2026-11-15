---
title: "CONT5 - Generational Turnover, Paradigm Lock-In, and Spatial Partner Coupling (Wave-4 Synthesis)"
author: "Antigravity (Wave-4 Synthesis Round)"
date: "2026-11-15"
tags: [contribution, turnover, steady-state, interventions, paradigm-lockin, path-b-v2, publishable-results]
project: "sciencemodels"
---

- **Source:** Synthesis of Wave-4 external execution rounds `F1`–`F4` (`path_b_probe_v2`, `interventions_v1`, `typology_v1`, `turnover_v1` — 198 cells / 3,960 runs)
- **Date:** 2026-11-15
- **Related:** `output/reports/turnover_v1.md`, `output/reports/interventions_v1.md`, `output/reports/path_b_probe_v2.md`, `output/reports/typology_v1.md`, `AdmiralsLog/contributions/CONT1-depletion-counterforces-refills.md`, `AdmiralsLog/contributions/CONT3-publishable-results-synthesis.md`
- **Spawned:** `P-0007`, `P-0008`, `P-0009`, `P-0010`, `IE-0003`, `IE-0004`, `IE-0005`, `UE-0001`, `UE-0002`, `UE-0003`

# 1. Why Wave 4 (`F1`–`F4`) Changes the Paper's Substantive Arc

Waves 1–3 (`CONT3`) established the relational prestige-merit wedge, the newcomer-incumbent gap, and the $\xi^* \in [0.6, 0.8]$ welfare kink in a world where **founders never retired**, **new questions never emerged**, **institutional interventions were untested**, and **dyadic partner choice did not pull spatial movement**.

Wave 4 (`F1`–`F4`, 198 cells / 3,960 runs, 0 failures) removed each of those four brackets and produced **four new substantive findings about the modelled world** that directly strengthen both the Phase-1 paper (`P-0001`) and the Flagship paper.

---

# 2. Result 4: Immortal Founders Masked Proxy Failure — Generational Turnover Deepens It $2.8\times$ to $6.6\times$ (`turnover_v1`)

In [turnover_v1.md](file:///Users/joko/Git/sciencemodels/output/reports/turnover_v1.md) (54 cells / 1,080 runs at $H = 400$), the unbounded-population baseline (`none`, $N_t = 20 + \lfloor gt \rfloor$) was tested against two retirement rules:
- **`fixed_span` ($N \equiv 20$ stationary population):** Every newcomer arrival retires the oldest active scientist.
- **`age_hazard` ($\tau_{\max} = 200$ stationary tenure cap):** Scientists reaching 200 steps of tenure retire and are replaced one-for-one via the supervisor roulette.

| Exit Mode | $N$ (last 100) | Mean Tenure | `gap_recent` (tenure $<20$ vs $\ge 20$) | $c_{\text{ideal}}$ Final | $c_{\text{ideal}} < 0$ (% of runs) | $c_{\text{ideal}}$ Time-Mean | Top-10% Prestige Share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `none` (immortal founders) | 61.45 | 267.6 | $-0.01362$ | $-0.0860$ | $75.6\%$ | $+0.1141$ | $0.2605$ |
| `fixed_span` ($N \equiv 20$) | 20.00 | 110.8 | $-0.00612$ (**$-55.0\%$**) | **$-0.2401$** (**$2.79\times$**) | **$89.2\%$** | $+0.1042$ | $0.2327$ |
| `age_hazard` ($\tau_{\max}=200$) | 61.45 | 61.1 | $-0.00282$ (**$-79.3\%$**) | **$-0.5678$** (**$6.61\times$**) | **$100.0\%$** | $+0.1112$ | **$0.4086$** |

Three publishable insights follow immediately:
1. **Generational turnover amplifies proxy failure ($2.8\times\text{–}6.6\times$):**
   Far from being an artifact of immortal founders sitting on exhausted peaks, negative quality-prestige correlation was *diluted* by founders who had discovered genuine peaks at $t=0$. Once all founders retire, the active population consists exclusively of dynasty-recruited cohorts herding around depleted prestige clusters, driving terminal $c_{\text{ideal}}$ to $-0.2401$ (`fixed_span`, $p = 7.7 \times 10^{-32}$) and $-0.5678$ (`age_hazard`, negative in $100\%$ of runs, $p < 10^{-60}$), while `age_hazard` pushes top-10% prestige concentration from $26.1\%$ to **$40.9\%$**.
2. **Decomposition of the arrival gap into founder-tenure premium ($55\%\text{–}79\%$) and steady-state stratification ($21\%\text{–}45\%$):**
   The arrival gap (`gap_recent`) is negative in **all 1,080 of 1,080 runs** across all three modes, proving that cohort stratification survives complete generational replacement. At the same time, its magnitude shrinks by $55.0\%$ (`fixed_span`) to $79.3\%$ (`age_hazard`) in 360 of 360 seed-matched pairs — cleanly isolating the steady-state structural handicap from the immortal-founder stock premium.
3. **Discovery-Phase Alignment vs. Depletion-Phase Lock-In:**
   Across all three exit modes, the **whole-run time mean** of $c_{\text{ideal}}$ is **positive** ($+0.104$ to $+0.114$), whereas the **terminal steady state** is strongly **negative** ($-0.086$ to $-0.568$). Prestige genuinely tracks merit during the initial discovery phase (when high-prestige pioneers are actively harvesting virgin Gaussian peaks) and inverts into proxy failure once the initial peaks are depleted and dynastic herding takes over.

---

# 3. Result 5: Attention-Side Mobility Shocks Outperform Entry Lotteries by $40\times$ (`interventions_v1`)

In [interventions_v1.md](file:///Users/joko/Git/sciencemodels/output/reports/interventions_v1.md) (72 cells / 1,440 runs), crossing supervisor-roulette damping ($\alpha \in \{1.0, 0.5, 0.0\}$) with sabbatical relocation shocks ($p_{\text{reloc}} \in \{0.00, 0.02\}$) across visibility weights $1-c \in \{0.5, 0.7, 0.9\}$ yielded a decisive policy ranking:
- **Sabbatical/maverick relocation (`p_reloc = 0.02`)** cuts the high-visibility welfare loss penalty by **$40.0\%$** ($0.00929 \to 0.00557$) and convexity by **$26.8\%$**.
- **Pure entry lotteries ($\alpha = 0.0$)** cut the high-visibility welfare loss penalty by only **$0.9\%$** ($0.00929 \to 0.00921$).

**Literature significance (Smaldino et al. 2019; Gross & Bergstrom 2019; Zollman 2010):** In an epistemic community where post-entry topic choice is governed by visibility rewards ($1-c$), randomizing entry-stage patronage or grant allocation cannot prevent conformity cascades; preserving **mid-career spatial/topical mobility** (`p_reloc`) is the binding lever.

---

# 4. Result 6: Paradigm Refills Reveal Two Distinct Forms of Epistemic Lock-In (`interventions_v1`)

Testing `new_questions` regeneration ([interventions_v1.md](file:///Users/joko/Git/sciencemodels/output/reports/interventions_v1.md#L129-L165), Arm C) uncovered a sharp duality between **incumbent hot-spot bailouts (`occupied`)** and **unoccupied paradigm emergence (`frontier`)** at $g = 0.1$:
- **Exploitation Lock-In (`occupied`):** When depleted tiles regenerate under existing crowds, scientists cease exploring (`explored` collapses by **$84\%$**, from $0.377$ to $0.061$) and the newcomer gap widens **$11\times$** to $-0.451$.
- **Unclaimed-Frontier Lock-In (`frontier`):** When new questions open on unoccupied tiles, $c_{\text{ideal}}$ flips positive ($+0.078$) for the few pioneers who reach them, but the visibility-bound majority stays behind in depleted clusters — leaving **unclaimed frontier knowledge $315\times$ higher than baseline** ($0.1895$ vs. $0.0006$) while exploration ($0.330$) stays below the no-refill baseline ($0.377$).

---

# 5. Result 7: Spatial Partner Pull Couples Collaborator Choice to Co-Harvest Team Size and the Wedge (`path_b_probe_v2`)

In [path_b_probe_v2.md](file:///Users/joko/Git/sciencemodels/output/reports/path_b_probe_v2.md) (48 cells / 960 runs), introducing `dyadic_move_pull` $\in \{0.5, 1.0\}$ bridged the gap between `model.links` and spatial co-harvesting:
- On `mgstd5`, increasing partner-choice goal weight from $\lambda = 0$ (prestige-seeking) to $\lambda = 1$ (merit/closeness-seeking) at `pull = 1.0` reduces co-harvest `mean_team_size` from **$7.32$ (un-pulled) / $5.33$ ($\lambda=0$) down to $3.42$** (and $2.89$ in the logit-structural arm, against the $2.65$ empirical reference; **4/4 `mgstd5` contrasts pass**).
- Simultaneously, `dyadic_move_pull = 1.0` makes $\lambda$ govern epistemic alignment: at $\lambda = 0$ (partnering for prestige + pulling toward partners), time-mean `Corr_prestige_ideal_merit` drops by **$-0.0664$** (95% CI $[-0.0774, -0.0558]$), whereas at $\lambda = 1$ (partnering for merit/closeness + pulling toward partners), `Corr_prestige_ideal_merit` rises by **$+0.0522$** ($[+0.0425, +0.0622]$), flipping the time-mean wedge (`c_local - c_ideal`) from $+0.035$ to $-0.037$.
- **Remaining structural gap (`P-0010`):** Because active links only dissolve when utility drops below the reservation bar and co-located partners stay co-located, `repeated_collab_share` remains pinned near **$0.90\text{–}0.92$** (vs. $0.34$ reference), pointing directly to a **post-project link dissolution hazard** as the final Path B mechanism.
