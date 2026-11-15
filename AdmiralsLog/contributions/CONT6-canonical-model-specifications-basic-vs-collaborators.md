---
title: "CONT6 - Canonical Model Specifications (M0 Basic vs. M1 Entry vs. M2 Dyadic) and the True Scale of Distraction"
author: "Antigravity (Review & Synthesis Round)"
date: "2026-11-15"
tags: [contribution, model-specification, basic-vs-collaborators, scale-of-distraction, ideal-merit, m0-m1-m2]
project: "sciencemodels"
---

- **Source:** Owner review (2026-11-15, Items 2b & 2c: clarifying the overall scale of distraction / `ideal_Merit` and codifying the "Basic Model vs. Collaborators Enter" specification principle)
- **Date:** 2026-11-15
- **Related:** `AdmiralsLog/way_forward.md`, `docs/conceptual_degrees_of_freedom.md`, `sciencemodels/sweeps.py`, `sciencemodels/metrics.py`, `AdmiralsLog/contributions/CONT1-depletion-counterforces-refills.md`, `AdmiralsLog/contributions/CONT3-publishable-results-synthesis.md`, `AdmiralsLog/contributions/CONT5-generational-turnover-and-paradigm-lockin.md`
- **Spawned:** `P-0011` (`AdmiralsLog/proposals/P-0011-canonical-three-tier-model-specification.md`)

# 1. Why This Document Exists: Two Structural Clarifications

As the project expanded from the original 6 notebooks (`main`) across nine simulation rounds (`sensitivity_map_v1` through `turnover_v1`), two core conceptual questions became blurred across reports:

1. **The Scale Question (Item 2b):** *How large is the epistemic loss from visibility distraction in absolute and relative terms? Why do `ideal_Merit` loss numbers often look tiny ($10^{-4}$ to $10^{-2}$), and is actual aggregate knowledge harvested actually very close to the ideal benchmark across most of the parameter space?*
2. **The Model-Tier Principle (Item 2c):** *We originally committed to separating a **Basic Model** (individual epistemic search under visibility incentives) from a model **where collaborators enter**. Has that boundary been held steadily across rounds, and what are the exact canonical specifications we use for simulation?*

This contribution resolves both questions with exact accounting from the codebase and data archives.

---

# 2. The Scale of the Argument: Why Moderate Distraction Barely Hurts Output While Inverting Prestige

## 2.1 Why Raw `ideal_Merit` Numbers Look Numerically Tiny ($10^{-4}$ to $10^{-2}$)

In `welfare_probe_v1.md` and `interventions_v1.md`, terminal `ideal_merit_sum` is reported as numbers like $0.00063$ (at $c = 0.9$) to $0.01033$ (at $c = 0.1$), and per-agent `ideal_merit_mean` is $1.5 \times 10^{-5}$ to $4.7 \times 10^{-4}$. Even in `typology_v1.md`, the 200-step cumulative sum `ideal_loss_cum` is $0.40$ to $0.48$. Three mechanical facts in `sciencemodels/metrics.py:53-107` explain why those numbers look small:

1. **Small unit scale of the grid:** On a $20 \times 20 = 400$-tile torus, a two-Gaussian landscape (`MultG2`, `std=5`) has total initial knowledge $\sum_{x,y} k_{xy}(0) \approx 2.0$ across all 400 tiles (mean tile $0.005$, peak tile $\approx 0.04$).
2. **Multiply by $h = 0.02$ per step:** Each step harvests only $2\%$ of a tile's remaining knowledge ($h = 0.02$), so a single scientist's harvest in a single step is of order $0.02 \times 0.005 = 10^{-4}$.
3. **Crucially, `update_ideal_merit` is a 1-step myopic residual on the *already-depleted* grid:** At step $t$, `update_ideal_merit` does **not** compare the community's cumulative harvest against an omniscient planner's trajectory from $t=0$. Instead, it takes the *actual* grid left over at step $t$ (`previous_knowledge`) and computes the 1-step difference between what the $N$ scientists harvested at step $t$ and what $N$ greedy agents placed on the $N$ highest remaining tiles of that *same* step-$t$ grid would have harvested in that single step. And in `welfare_probe_v1` / `interventions_v1`, `ideal_merit_sum` is recorded only at the **single final step $t = 200$**, when the peaks are already largely exhausted!

## 2.2 The True Intuitive Scale: Aggregate Knowledge Harvested (`percentage_knowledge_harvested`)

When we read the outcome on its natural macroscopic scale — **the percentage of total initial landscape knowledge harvested by step 200 (`percentage_knowledge_harvested`)** and **the knowledge-weighted share of tiles explored (`explored_weighted_by_initial_knowledge`)** — **your core intuition is completely confirmed**:

| Curiosity $c$ | Visibility Weight ($1-c$) | `MultG2` Harvested Share | `MultG2` Peak-Weighted Explored | `MultG6` Harvested Share | `Noisy2` Harvested Share | `MultG2` $c_{\text{ideal}}$ (`M0`, $g=0$) | `MultG2` $c_{\text{ideal}}$ (`M1`, $g=0.1$) | `MultG2` $c_{\text{ideal}}$ (`M1` Turnover) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **$1.0$** (pure curiosity) | **$0.0$** | **$67.8\%$** | $93.7\%$ | **$42.3\%$** | **$27.6\%$** | $+0.356$ | — | — |
| **$0.9$** | **$0.1$** | **$67.8\%$** | $93.5\%$ | **$42.3\%$** | **$27.8\%$** | $+0.341$ | $+0.060$ | — |
| **$0.7$** | **$0.3$** | **$67.5\%$** | $93.1\%$ | **$42.1\%$** | **$27.9\%$** | $+0.259$ | $+0.028$ | $-0.175$ |
| **$0.5$** (balanced) | **$0.5$** | **$67.0\%$** | $92.2\%$ | **$41.9\%$** | **$27.9\%$** | $+0.352$ | **$-0.057$** | **$-0.240\text{ to }-0.568$** |
| **$0.3$** (kink $\xi^*$) | **$0.7$** | **$60.1\%$** | $88.4\%$ | **$36.8\%$** | **$25.4\%$** | $+0.154$ | **$-0.154$** | **$-0.310\text{ to }-0.610$** |
| **$0.1$** (high distraction) | **$0.9$** | **$34.5\%$** | $75.7\%$ | **$15.5\%$** | **$13.1\%$** | $+0.041$ | **$-0.281$** | — |
| **$0.0$** (pure visibility) | **$1.0$** | **$4.8\%$** | $47.2\%$ | **$1.9\%$** | **$1.8\%$** | $+0.036$ | — | — |

*(Sources: `data/figures_v1/run_number{2,4,6}.csv`, `output/reports/welfare_probe_v1.md`, `output/reports/interventions_v1.md`, `output/reports/turnover_v1.md`.)*

Look at what this table says about the **scale of the argument**:

1. **Across the entire moderate range ($c \in [0.5, 1.0]$, visibility weight $1-c \in [0.0, 0.5]$), distraction causes almost ZERO aggregate output loss!**
   - Moving from pure curiosity ($c = 1.0$) to a 50/50 curiosity-visibility balance ($c = 0.5$) reduces total knowledge harvested on `MultG2` from **$67.8\%$ to $67.0\%$** — an aggregate output loss of **less than 1 percentage point ($0.8\%$)**! On `MultG6` ($42.3\% \to 41.9\%$) and `Noisy2` ($27.6\% \to 27.9\%$), aggregate output is likewise virtually identical.
   - *Why?* Because on a peaked epistemic landscape, early discoveries occur on the Gaussian peaks anyway. Once a few scientists find a peak, moderate visibility attraction ($1-c \le 0.5$) pulls nearby peers *toward that same productive peak*, helping harvest the $92\%+$ of knowledge located there!
2. **Yet inside that exact same moderate regime ($c = 0.5$, where $99\%$ of aggregate harvest is preserved), the prestige-merit correlation ALREADY collapses and turns negative under entry/turnover!**
   - At $c = 0.5$, aggregate scientific harvest is completely healthy ($67.0\%$), **but `Corr_prestige_ideal_merit` flips from $+0.352$ (`M0`) to $-0.057$ (`M1` entry) and $-0.240\text{ to }-0.568$ (`M1` stationary turnover)**!
   - **Contextual meaning:** This is the central insight of the model. You do **not** need an epistemic catastrophe where science stops producing knowledge in order to get severe **proxy failure**. At realistic, moderate levels of status-seeking ($1-c \approx 0.5$), **the scientific community still harvests nearly all available knowledge on the peaks, but its reward system becomes perverse**: prestige is captured by senior dynastic insiders camping on already-depleted orthodox peaks, while the scientists actually harvesting high-value marginal tiles at step $t$ hold low prestige!
3. **Aggregate output collapse ONLY occurs in the extreme visibility tail ($1-c > 0.7$, i.e. $c \le 0.2$, past the $\xi^* \in [0.6, 0.8]$ kink):**
   - Only when visibility weight exceeds $70\%\text{–}80\%$ do scientists prioritize crowding over novelty so strongly that they lock into wherever the initial herd formed before finding the Gaussian peaks. At $c = 0.1$ ($1-c = 0.9$), harvested knowledge drops by **half** on `MultG2` ($67.8\% \to 34.5\%$) and by **two-thirds** on `MultG6` ($42.3\% \to 15.5\%$); at $c = 0.0$, science freezes completely ($4.8\%$ harvested).

> **Summary of the Scale Argument for the Paper:**
> Our model reveals a **two-stage degradation of science under visibility incentives**:
> - **Stage 1 ($1-c \in [0.3, 0.6]$, Moderate Visibility + Demographic Entry/Turnover): *Distributional Proxy Failure Without Output Collapse.*** Aggregate knowledge production stays within $1\%\text{–}10\%$ of the curiosity benchmark, so science *as a whole* keeps working — but **prestige decouples from and anti-tracks current epistemic contribution ($c_{\text{ideal}} < 0$)** because dynastic recruitment piles newcomers onto exhausted ground while founders/incumbents coast on historical stock.
> - **Stage 2 ($1-c \in [0.7, 1.0]$, Extreme Visibility Past $\xi^*$): *Macroscopic Epistemic Collapse.*** Visibility herding overwhelms peak discovery, cutting aggregate scientific output by $50\%\text{–}95\%$.

---

# 3. Codifying the "Basic Model vs. Collaborators Enter" Principle (`M0` vs. `M1` vs. `M2`)

## 3.1 How the Principle Drifted Across Rounds

In `AdmiralsLog/way_forward.md`, we stated a foundational design principle: **first study the Basic Model (quality–prestige interplay without collaborator choice), and then introduce collaborator choice as a distinct step.**

However, because the codebase contains **two** distinct social/relational mechanisms, different rounds used the phrase "collaborators enter" in two different ways:
1. **Vertical Mentorship / Demographic Entry (`agent_generation_time` $g > 0$):** `way_forward.md` called `model.py:389-420` (*newcomers pick a supervisor proportional to prestige and spawn on `sup.pos`*) *"the literal 'choose the prestigious collaborator' mechanism — the cheapest explicit collaborator element, already present in code."* Consequently, Waves 1–4 (`sensitivity_map_v1`, `career_flows_v1`, `prestige_stock_v1`, `interventions_v1`, `turnover_v1`) treated **$g = 0$ as the Basic Model** and **$g > 0$ as the Collaborator/Entry Model**.
2. **Horizontal Dyadic Co-Authorship (`dyadic_mode` $\in \{\text{logit}, \text{threshold}\}$ + `dyadic_move_pull`):** Path B (`path_b_probe_v1.md`, `path_b_probe_v2.md`) introduced explicit bilateral collaboration links (`model.links`), partner-proximity movement pull, and joint credit sharing — while *also* pinning $g = 0.1$ ON in the background!

Because Path B turned on dyadic links ($M2$) while keeping demographic entry ($M1$) on, and because `welfare_probe_v1` / `sensitivity_map_v1` crossed $g \in \{0.0, 0.1\}$ in the same tables, the clean boundary between **"Basic Model"** and **"Collaborators Enter"** became hard to track.

## 3.2 The Canonical Three-Tier Model Specification (`M0`, `M1`, `M2`)

To restore complete clarity across the repository and paper drafts, every simulation cell in `sciencemodels` belongs to one of **three explicit canonical model tiers**:

| Specification Dimension | **Tier `M0`: Basic Model (Closed-Cohort Search)** | **Tier `M1`: Mentorship / Demographic Entry Model (Vertical Lineage)** | **Tier `M2`: Explicit Dyadic Collaboration Model (Horizontal Co-Authorship)** |
| --- | --- | --- | --- |
| **Substantive Meaning** | A fixed cohort of solo scientists searches an epistemic landscape; social influence operates *only* via spatial visibility ($V_i$) ("hot-topic crowd attraction"). | New generations of PhDs/juniors enter, select a mentor via prestige roulette ($P(\text{sup}_i) \propto P_i^\alpha$), and inherit their mentor's research topic (`sup.pos`). | Active scientists form bilateral co-authorship links (`model.links`) balancing partner visibility vs. merit/closeness ($\lambda$), move toward partners (`dyadic_move_pull`), and co-harvest. |
| **Population & Entry ($g$)** | $N \equiv 20$ fixed (`agent_generation_time = 0.0`, `exit_mode = "none"`) | Open cohort: $g \in \{0.05, 0.10, 0.20\}$; either growing ($N_t = 20 + \lfloor gt \rfloor$) or stationary ($N \equiv 20$, `exit_mode = "fixed_span"` / `"age_hazard"`) | Can be run on top of `M0` ($g=0$, pure horizontal collaboration) or `M1` ($g=0.1$, lineage + co-authorship) |
| **Dyadic Layer (`dyadic_mode`)** | `"off"` (`dyadic_move_pull = 0.0`, `credit_sharing = "structural"`) | `"off"` (`dyadic_move_pull = 0.0`, `credit_sharing = "structural"`) | `"logit"` or `"threshold"`, $\lambda \in [0,1]$, `dyadic_move_pull` $\in \{0.0, 0.5, 1.0\}$, `credit_sharing` $\in \{\text{structural}, \text{equal\_split}\}$ |
| **Shared Core Pins** | $20 \times 20$ torus, $n_0 = 20$, $\epsilon = 0.1$, $h = 0.02$, `use_distance = True`, `knowledge_normalization = "share"`, `prestige_accrual = "harvested"`, `vanishing = 0.9` (`0.5` in `W_v05`), `use_visibility_reward = True` | Same shared core pins | Same shared core pins |
| **Headline Behavior of `Corr(prestige, ideal_merit)`** | **Positive (`+0.15` to `+0.79`)** for all $c \ge 0.2$ (drops near $0$ only at extreme distraction $c \le 0.1$) | **Flips Negative (`-0.06` to `-0.28` with immortal founders; `-0.24` to `-0.57` under stationary turnover)** | **Governed by $\lambda$ under `dyadic_move_pull = 1.0`:** negative wedge at $\lambda = 0$ (prestige-seeking partners), positive alignment shift ($+0.052$) at $\lambda = 1$ (merit-seeking partners) |
| **Headline Behavior of `top10pct_prestige_share`** | **Low / Near-Equal (`0.112–0.133`)** for $c \ge 0.3$ (spikes to $0.165\text{–}0.189$ only at $c \le 0.1$) | **High Inequality (`0.21–0.26` with immortal founders; `0.409` under `age_hazard` turnover)** | **High Assortativity at $\lambda = 0$, De-concentrated at $\lambda = 1$** (`mean_team_size` drops $7.32 \to 2.89$ on `mgstd5`) |
| **Rounds & Reports Using This Tier** | • Legacy notebooks (`main`, `figures_v1`, `CONT7`)<br>• $g=0$ control arms of `sensitivity_map_v1`, `remeasure_v1`, `welfare_probe_v1`, `prestige_stock_v1`, `interventions_v1` (Arm C), `typology_v1` | • $g > 0$ arms of `sensitivity_map_v1`, `remeasure_v1`, `welfare_probe_v1`<br>• All of `career_flows_v1`, `career_flows_noisy_v1`, `prestige_stock_v1`, `interventions_v1` (Arms A–C), `turnover_v1`, `typology_v1` ($g=0.1$) | • `path_b_probe_v1` ($\text{pull}=0.0$, run on `M1` $g=0.1$)<br>• `path_b_probe_v2` ($\text{pull} \in \{0.5, 1.0\}$, run on `M1` $g=0.1$) |

## 3.3 Crucial Gap Surfaced by This Matrix (`P-0011`)

Look at the third column (`Tier M2`): **every single Path B run so far (`path_b_probe_v1` and `path_b_probe_v2`) was run with `agent_generation_time = 0.1` (`M1 + M2` combined), NEVER on the pure Basic Model `M0` (`agent_generation_time = 0.0` + `dyadic_mode != "off"`)!**

That means we have never isolated whether **horizontal dyadic collaborator choice alone (`M0 + M2`, $g = 0$)** opens the prestige-merit wedge on a closed cohort of $N = 20$ scientists, or whether the negative wedge requires **vertical demographic entry (`M1`, $g > 0$)**! We have filed [P-0011-canonical-three-tier-model-specification.md](file:///Users/joko/Git/sciencemodels/AdmiralsLog/proposals/P-0011-canonical-three-tier-model-specification.md) to codify the `M0` / `M1` / `M2` presets in `sciencemodels/sweeps.py` and run the missing `M0 + M2` ($g=0$ dyadic) isolation check.
