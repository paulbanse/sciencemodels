---
title: "CONT3 - Publishable Results Synthesis & Visual Alignment Across Waves 1–4"
author: "Antigravity (Review & Synthesis Round)"
date: "2026-11-15"
tags: [contribution, synthesis, publishable-results, literature-niche, figures, phase-1, scale-of-distraction, m0-m1-m2]
project: "sciencemodels"
---

- **Source:** Antigravity inspection & evaluation round (Item 2: Publishable Results & Visual Alignment), updated with Wave-4 refinements, contextual interpretations, and the `M0`/`M1`/`M2` + scale framework (`CONT6`, `CONT7`)
- **Date:** 2026-11-14 (v1); 2026-11-15 (v3 — full contextual, scale, and model-tier integration)
- **Related:** `AdmiralsLog/proposals/P-0001-phase1-paper.md`, `AdmiralsLog/contributions/CONT6-canonical-model-specifications-basic-vs-collaborators.md`, `AdmiralsLog/contributions/CONT7-legacy-notebooks-preliminary-findings.md`, `docs/related_publications.md`, `docs/connections_niklas.md`, `output/reports/sensitivity_map_v1.md`, `output/reports/career_flows_v1.md`, `output/reports/career_flows_noisy_v1.md`, `output/reports/welfare_probe_v1.md`, `output/reports/prestige_stock_v1.md`, `output/reports/turnover_v1.md`, `output/reports/interventions_v1.md`, `output/reports/path_b_probe_v2.md`
- **Spawned:** Visual and analytical spine for `P-0001` (`output/reports/paper_phase1_v1.md`)

# 1. How `CONT1`–`CONT7` Fit Together as a Complete Paper Dossier

The seven contribution files in `AdmiralsLog/contributions/` now form a complete, self-contained analytical architecture:

| ID | Title & Role in the Project | Model Tier(s) Covered |
| --- | --- | --- |
| **[CONT6](file:///Users/joko/Git/sciencemodels/AdmiralsLog/contributions/CONT6-canonical-model-specifications-basic-vs-collaborators.md)** | **Canonical Model Specifications (`M0`, `M1`, `M2`) & The Scale of Distraction:** Defines the exact boundary between the Basic Model (`M0`), the Mentorship Entry Model (`M1`), and the Dyadic Collaboration Model (`M2`), and explains why moderate distraction ($1-c \le 0.5$) leaves aggregate output $99\%$ intact while inverting the prestige signal. | `M0`, `M1`, `M2` |
| **[CONT7](file:///Users/joko/Git/sciencemodels/AdmiralsLog/contributions/CONT7-legacy-notebooks-preliminary-findings.md)** | **Preliminary Findings from the Legacy Notebooks (`main` / `figures_v1`):** Recovers the 5 core `M0` findings from the 6 original notebooks (the curiosity plateau above $c=0.3$, unweighted vs. peak-weighted exploration, the `mixed_pop` cognitive-division-of-labor advantage, closed-cohort prestige equality, and longitudinal peak-then-decay). | `M0` |
| **[CONT3](file:///Users/joko/Git/sciencemodels/AdmiralsLog/contributions/CONT3-publishable-results-synthesis.md)** *(this file)* | **Core Publishable World Results (Waves 1–3, Refined by Wave 4):** Synthesizes the three headline findings driven by entry ($g$), visibility weight ($1-c$), and landscape legibility (`F1`, `F2`/`F2b`, `F3`) with full real-world contextual interpretation. | `M0` $\to$ `M1` |
| **[CONT5](file:///Users/joko/Git/sciencemodels/AdmiralsLog/contributions/CONT5-generational-turnover-and-paradigm-lockin.md)** | **Wave-4 World Results (Generational Turnover, Paradigm Lock-In & Dyadic Pull):** Synthesizes the 3,960 Wave-4 runs (`turnover_v1`, `interventions_v1`, `path_b_probe_v2`). | `M1` (stationary), `M2` |
| **[CONT1](file:///Users/joko/Git/sciencemodels/AdmiralsLog/contributions/CONT1-depletion-counterforces-refills.md)** | **Micro-Mechanism & Institutional Counterforces (Path C):** Recruitment-herding dominance over birth-tile starvation, sabbatical mobility vs. entry lotteries, and orthodox bailouts vs. unclaimed frontier paradigms. | `M0`, `M1` |
| **[CONT2](file:///Users/joko/Git/sciencemodels/AdmiralsLog/contributions/CONT2-hunting-proxy-patterns-formalization.md)** | **Formalizing the `Hunting_Proxy_Patterns` Typology (Path D):** Maps distraction, destruction ($AP_F \dashv G$), and $P \to G$ counterforces onto the ABM (`typology_v1`). | `M0`, `M1` |
| **[CONT4](file:///Users/joko/Git/sciencemodels/AdmiralsLog/contributions/CONT4-inert-and-invariant-degrees-of-freedom.md)** | **What Does Not Matter Much (Setup Invariances, Scale Traps & Rejected Mechanisms):** Consolidates the 11 setup-level invariances and falsified parameterizations (`F4`, `F5`, `IE-0001`, `IE-0002`). | All tiers |

---

# 2. The Three Foundational Results About the World (Contextualized & Scaled)

Against the 46 verified studies in `docs/related_publications.md`, no existing model combines (i) spatial/topical landscape search (Kitcher 1990; Weisberg & Muldoon 2009; Zollman 2007, 2010), (ii) visibility-proxy incentives and prestige recruitment (Smaldino & McElreath 2016; Braganza 2020; John et al. 2023), and (iii) an external omniscient-planner benchmark (`ideal_Merit`, answering Payette 2011's critique that ABMs of science lack reward-independent success criteria).

## Result 1: Endogenous Channel-Specific Proxy Failure ("The Relational Wedge" Between `M0` and `M1`)

- **Simulation Finding (`sensitivity_map_v1`, `remeasure_v1`, `turnover_v1`, `interventions_v1`):**
  In the **Basic Closed-Cohort Model (`Tier M0`, $g = 0$)**, prestige positively tracks both local quantity (`Corr_prestige_local_merit` $= +0.19$) and ideal quality (`Corr_prestige_ideal_merit` $= +0.220$), and prestige concentration remains low (`top10pct_prestige_share` $= 0.135$).
  Simply activating **prestige-proportional mentorship entry (`Tier M1`, $g = 0.1$)** — without any scientist consciously gaming a metric — flips the quality correlation `Corr_prestige_ideal_merit` negative to **$-0.144$** (17/18 entry cells negative across 720 runs), while the local proxy correlation `Corr_prestige_local_merit` stays positive ($+0.038$), and prestige concentration nearly doubles ($0.135 \to 0.228$). Under stationary generational turnover (`turnover_v1`), this negative correlation deepens **$2.8\times$ (`-0.240`, `fixed_span`) to $6.6\times$ (`-0.568`, `age_hazard`)**.
- **Contextual & Scale Interpretation (Why This Happens Without Aggregate Output Collapse):**
  As shown in `CONT6`, at moderate curiosity ($c = 0.5$), the scientific community still harvests **$67.0\%$ of all landscape knowledge** (vs. $67.8\%$ at $c = 1.0$). Science has *not* stopped discovering truth! Rather, what fails is the **allocative and reputational signal**:
  - In real academia, founding pioneers discover a major paradigm (a Gaussian peak) early in their careers and accumulate high prestige.
  - Because doctoral students and junior hires gravitate toward prestigious mentors (`M1`) and visible, crowded subfields ($1-c$), successive cohorts pile into the *already-exhausted core* of that paradigm.
  - By mid-to-late run, the scientists who hold the highest accumulated prestige are senior insiders (and their dynastic clusters) dividing dwindling incremental returns on exhausted topics, while the lower-prestige scientists on the periphery are the ones making higher-value marginal discoveries (`c_ideal < 0`).
  - Meanwhile, evaluated by **local neighborhood standards (`local_merit`)**, the crowded orthodoxy still looks respectable ($c_{\text{local}} > 0$) — reproducing the exact empirical wedge observed in the Klann economics co-authorship panel ($\text{corr}(\text{prestige}, \text{future5}) = 0.50$ vs. $\text{corr}(\text{prestige}, \text{citations}) = 0.07$).

![F1: Prestige no longer tracks ideal merit once entry is on](../../output/report_notebook_v1/F1_sensitivity_entry_curiosity.png)

## Result 2: Endogenous Cumulative Advantage & Intergenerational Stratification (`M1`)

- **Simulation Finding (`career_flows_v1`, `career_flows_noisy_v1`, refined by `turnover_v1`):**
  In `M1`, newcomers never close the prestige gap with incumbents at any entry rate $g \in \{0.05, 0.10, 0.20\}$ (final gap ratio $-0.713$ on clean terrain and $-0.588$ on noisy terrain; 100% of cells negative).
  - **Wave-4 Decomposition (`turnover_v1`, 1,080 runs at $H=400$):** Retiring all founders under `fixed_span` ($N \equiv 20$) and `age_hazard` ($\tau_{\max} = 200$) proves that the arrival gap (`gap_recent`, tenure $<20$ vs. $\ge 20$) remains negative in **1,080 of 1,080 runs**, while shrinking in magnitude by **$55.0\%$ (`fixed_span`) to $79.3\%$ (`age_hazard`)**.
  - **Qualification of the Entry-Rate Effect:** In the unbounded-population baseline (`none`), faster entry ($g = 0.20$) diluted proxy failure ($-0.139 \to -0.043$) only because fresh low-prestige entrants outnumbered the 20 immortal founders. When population size is held stationary (`fixed_span`, $N \equiv 20$), proxy failure is **invariant to the entry rate** ($-0.231, -0.247, -0.243$ at $g = 0.05, 0.10, 0.20$).
- **Contextual Interpretation:**
  In real scientific fields, how much of the status gap between senior and junior scholars is simply **mechanical career length** (having had more years to publish on early breakthroughs), and how much is **structural positional disadvantage** (arriving after the field's low-hanging fruit has been harvested and being pulled into crowded topics)? Our turnover experiment answers this quantitatively: **55%–79% of the raw gap in an expanding field is the pioneer-longevity premium, while the remaining 21%–45% is permanent steady-state cohort stratification** that persists even after every founding pioneer has retired. Furthermore, when landscape legibility is noisy (`noisypr5` — fields where true quality is hard to distinguish from background noise), early founders cannot monopolize the true peaks as cleanly, which **narrows the newcomer handicap (`-0.713 -> -0.588`) while widening the local-minus-ideal evaluation wedge**.

![F2: Newcomer prestige stays below incumbent prestige on both terrains](../../output/report_notebook_v1/F2_career_flows_gap_ratio.png)

![F2b: Slower entry decouples prestige from ideal merit more strongly](../../output/report_notebook_v1/F2b_career_flows_noisy_vs_clean.png)

## Result 3: The Two-Stage Welfare Curve and the $\xi^* \in [0.6, 0.8]$ Collapse Kink (`M0` & `M1`)

- **Simulation Finding (`welfare_probe_v1`, `remeasure_v1`, `interventions_v1`):**
  Across both `M0` ($g=0$) and `M1` ($g=0.1$), epistemic loss as a function of visibility weight ($1-c \in [0.1, 0.9]$) is **flat across the lower half of the axis ($1-c \le 0.5$) and kinks sharply upward at $\xi^* \in [0.6, 0.8]$** across all 24 outcome series (F-screen + interior bootstrap 95% CI passed in 24/24 series). Across the full axis, single-step terminal `ideal_Merit` loss rises $10\times\text{–}44\times$, while **macroscopic knowledge harvested (`percentage_knowledge_harvested`) stays flat near $66.5\%$ for $1-c \le 0.5$ and then drops by half to $33.0\%$ at $1-c = 0.9$ on `mg2` (and from $41.5\%$ to $15.1\%$ on `mg6`)**.
- **Contextual & Scale Interpretation:**
  Why is visibility pressure benign for aggregate discovery up to $\xi^* \approx 0.6$ and destructive above it?
  - **Below the threshold ($1-c \le 0.5$, i.e. $c \ge 0.5$):** Intrinsic curiosity is strong enough that pioneers locate the major paradigms (Gaussian peaks). Once pioneers are on a peak, moderate visibility attraction acts as a **coordination device** — drawing colleagues toward where the real breakthroughs are happening and harvesting $66\%\text{–}68\%$ of total landscape knowledge (even though, under `M1`, prestige within that cluster becomes distributionally distorted).
  - **Above the threshold ($1-c \ge 0.7$, i.e. $c \le 0.3$):** Visibility attraction overwhelms curiosity *before* the community has finished locating the peaks. Scientists lock into premature conformity cascades around whatever arbitrary topic had the highest initial density, leaving half to two-thirds of the discipline's potential discoveries permanently unharvested!
  - And as `interventions_v1` Arm B proved, **only mid-career mobility shocks (`p_reloc = 0.02`, cutting the kink penalty by $40\%$)** — not entry lotteries ($\alpha = 0$, $-0.9\%$) — can break this high-visibility conformity trap.

![F3: Visibility pressure raises loss; toggle gap at most 6.1e-04](../../output/report_notebook_v1/F3_welfare_loss_vs_visibility_weight.png)
