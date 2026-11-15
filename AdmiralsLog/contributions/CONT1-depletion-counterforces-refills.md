---
title: "CONT1 - Spatial Herding, Institutional Counterforces, and Paradigm Refills on the Epistemic Grid"
author: "Antigravity (Review & Synthesis Round)"
date: "2026-11-15"
tags: [contribution, mechanism, spatial-herding, counterforces, paradigm-shifts, path-c, interventions-v1, m1]
project: "sciencemodels"
---

- **Source:** Antigravity inspection & evaluation round (Items 1.1–1.3), updated with empirical findings from `output/reports/interventions_v1.md` (72 cells / 1,440 runs) and contextual/scale clarifications (`CONT6`)
- **Date:** 2026-11-14 (v1); 2026-11-15 (v3 — post-`interventions_v1` synthesis + contextual & scale framing)
- **Model Tier:** Primarily **Tier `M1` (Open-Cohort Mentorship / Entry, $g = 0.1$, `dyadic_mode = "off"`)**, with **Tier `M0` ($g = 0.0$)** as the closed-cohort control arm (see [CONT6-canonical-model-specifications-basic-vs-collaborators.md](file:///Users/joko/Git/sciencemodels/AdmiralsLog/contributions/CONT6-canonical-model-specifications-basic-vs-collaborators.md))
- **Related:** `AdmiralsLog/way_forward.md`, `output/reports/interventions_v1.md`, `output/reports/sensitivity_map_v1.md`, `output/reports/welfare_probe_v1.md`, `output/reports/career_flows_v1.md`
- **Spawned:** [executed/P-0002](file:///Users/joko/Git/sciencemodels/AdmiralsLog/proposals/executed/P-0002-spatial-spawn-dispersal.md), [executed/P-0003](file:///Users/joko/Git/sciencemodels/AdmiralsLog/proposals/executed/P-0003-institutional-counterforces-lotteries.md), [executed/P-0004](file:///Users/joko/Git/sciencemodels/AdmiralsLog/proposals/executed/P-0004-paradigm-refill-regimes.md), [P-0007](file:///Users/joko/Git/sciencemodels/AdmiralsLog/proposals/P-0007-sabbatical-shocks-x-frontier-refills.md), [rejected/IE-0001](file:///Users/joko/Git/sciencemodels/AdmiralsLog/ideas/rejected/IE-0001-birth-tile-starvation-hypothesis.md)

# 1. One Unified Mechanism Story (and Its Real-World Context)

Across our exploration waves (`sensitivity_map_v1`, `career_flows_v1`, `welfare_probe_v1`, and `interventions_v1`), moving from the **Basic Closed-Cohort Model (`Tier M0`, $g = 0$)** to the **Mentorship / Entry Model (`Tier M1`, $g = 0.1$)** produces three striking macro-phenomena:
1. **Distributional Proxy Failure at Moderate Distraction ($c \in [0.4, 0.7]$):** Even though the community still harvests **$\sim 67\%$ of all available landscape knowledge** on `MultG2` (virtually identical to the $67.8\%$ pure-curiosity benchmark; see `CONT6` Section 2), activating prestige-proportional PhD/junior entry flips `Corr_prestige_ideal_merit` from $+0.352$ (`M0`) to $-0.057$ (`M1`, and $-0.240\text{ to }-0.568$ under generational turnover).
2. **Permanent Cohort Stratification:** Junior entrants never close the prestige gap with founding incumbents.
3. **Macroscopic Output Collapse Above the $\xi^* \in [0.6, 0.8]$ Kink ($c \le 0.2$):** When visibility weight ($1-c$) exceeds $70\%\text{–}80\%$, distraction ceases to be merely a distributional distortion of who holds prestige and becomes an **aggregate epistemic collapse**, cutting total knowledge harvested in half ($67.8\% \to 34.5\%$ at $c=0.1$) as scientists huddle in crowded topics before locating the Gaussian peaks.

Items 1.1, 1.2, and 1.3 form a single causal chain — tested across 1,440 runs in [interventions_v1.md](file:///Users/joko/Git/sciencemodels/output/reports/interventions_v1.md) — explaining **why** the prestige signal inverts under mentorship entry, **which** institutional reforms actually mitigate visibility lock-in, and **what** happens when new scientific questions emerge.

---

# 2. Act I: Recruitment-Herding Dominance over Birth-Tile Starvation (1.1 — `P-0002` Executed)

In the `M1` demographic rule (`sciencemodels/model.py:389-420`), two real-world academic processes are fused at career entry:
- **Prestige-proportional mentorship matching:** A new PhD/junior scientist selects a supervisor $i$ with probability proportional to the supervisor's prestige ($P(\text{sup}_i) \propto \text{prestige}_i$).
- **Topical inheritance:** The newcomer starts their career directly on the supervisor's current research topic (`coords = sup.pos`).

Initially (`v1` / [executed/P-0002](file:///Users/joko/Git/sciencemodels/AdmiralsLog/proposals/executed/P-0002-spatial-spawn-dispersal.md)), we hypothesized that **mechanical birth-tile starvation** (juniors inheriting a topic their famous supervisor had already picked clean) drove a large share of the negative `Corr_prestige_ideal_merit` and the newcomer-incumbent gap.

**What `interventions_v1` Arm A proved ([interventions_v1.md](file:///Users/joko/Git/sciencemodels/output/reports/interventions_v1.md#L50-L92), 18 cells / 360 runs):**
- Forcing newcomers to start on an adjacent topic (`ring1`) or an independent random topic (`random`) while keeping prestige-proportional mentorship and visibility-driven movement **does not repair the quality correlation — it makes it even more negative** at every curiosity level ($c = 0.3$: $-0.154 \to -0.171$; $c = 0.5$: $-0.057 \to -0.145$; $c = 0.7$: $+0.028 \to -0.095$).
- In the formal decomposition $D_{\text{total}} = (\text{off} - \text{random}) + (\text{random} - \text{exact})$, the birth-tile starvation share is **negative everywhere** ($-5.7\%, -21.5\%, -53.5\%$), meaning **prestige-proportional recruitment plus visibility herding accounts for $105.7\%$ to $153.5\%$ of the quality drop**.
- Dispersing birth tiles also **widens the newcomer-incumbent prestige gap** by $0.08\%$ to $2.6\%$ on `mgstd5` and $5.0\%$ to $12.0\%$ on `noisypr5`.

![FA1: Exact spawn keeps the quality correlation highest](../../data/interventions_v1/figures/FA1_spawn_mode_quality.png)

![FA2: Dispersal removes little of the drop](../../data/interventions_v1/figures/FA2_drop_decomposition.png)

**Contextual interpretation in real-world science:**
Why does forcing doctoral students to pick dissertation topics far away from their famous advisor (`random`) actually *worsen* epistemic inequality and proxy failure? Because in a peaked scientific landscape (`mgstd5`), famous founding professors became famous by sitting near genuine high-potential paradigms (Gaussian peaks). Starting a PhD student on their advisor's topic (`exact`) at least places the student in a fertile neighborhood near the peak. If students are instead scattered onto random topics (`random`) while senior stars retain an uncontested monopoly on the core peak — and while visibility incentives ($1-c$) pull scattered juniors toward crowded mid-tier bandwagons — the founding elite's relative advantage grows even larger! Thus, **the Matthew effect in `M1` is not a trivial artifact of students being born on exhausted dissertation topics; it is driven by ongoing attentional herding toward crowded clusters.**

---

# 3. Act II: Attention-Side Mobility Shocks Beat Entry Lotteries by $40\times$ (1.2 — `P-0003` Executed)

To test institutional reforms from the metascience debate (Smaldino, Turner, & Contreras Kallens 2019; Gross & Bergstrom 2019; Osterloh & Frey 2015; Zollman 2010), `interventions_v1` Arm B ([interventions_v1.md](file:///Users/joko/Git/sciencemodels/output/reports/interventions_v1.md#L94-L127), 36 cells / 720 runs) crossed two policy instruments across high-pressure visibility weights $1-c \in \{0.5, 0.7, 0.9\}$:
1. **Entry-Stage Placement/Funding Lotteries ($\alpha \in \{1.0, 0.5, 0.0\}$):** Flattening the supervisor roulette $P(\text{sup}_i) \propto \text{prestige}_i^\alpha$ from linear Matthew patronage ($\alpha = 1.0$) to concave damping ($\alpha = 0.5$) or a pure egalitarian lottery ($\alpha = 0.0$).
2. **Mid-Career Sabbatical / Maverick Relocation Fellowships ($p_{\text{reloc}} \in \{0.00, 0.02\}$):** Relocating scientists trapped on depleted topics ($k < 0.025$) to a random unexplored topic with probability $2\%$ per step, breaking local status-quo inertia (`agents.py:166`).

**What `interventions_v1` Arm B proved:**
- **Sabbatical mobility shocks (`p_reloc = 0.02`) cut the high-visibility welfare loss penalty by ~40%** ($\text{loss}(0.9) - \text{loss}(0.5)$ drops from $0.00929$ to $0.00557$), reduce discrete convexity by **27%** ($0.00552 \to 0.00404$), cut terminal loss at weight $0.9$ by **41%** ($0.01033 \to 0.00605$), and improve `Corr_prestige_ideal_merit` at weight $0.9$ from $-0.2572$ to $-0.2391$.
- **Entry lotteries ($\alpha = 0.0$) do virtually nothing to welfare loss:** a pure placement lottery reduces the high-visibility penalty by only **$-0.9\%$** ($-0.000079$) and concave damping ($\alpha = 0.5$) by $-3.5\%$, only nudging top-10% prestige concentration slightly ($0.2091 \to 0.2039$).

![FB: Sabbatical shocks flatten the loss penalty](../../data/interventions_v1/figures/FB_welfare_loss_curves.png)

**Contextual interpretation in real-world science:**
Reform proposals in science policy often focus on **entry-stage egalitarianism** — e.g., randomizing initial grant/fellowship allocation or de-concentrating PhD placement across departments ($\alpha \to 0$). Our simulation shows why entry lotteries fail when the broader academic evaluation system still rewards **visibility and hot-topic conformity ($1-c \ge 0.7$)**: even if young researchers are initially placed at random across the discipline, career-long visibility incentives immediately pull them back into the same overcrowded bandwagon topics within a few steps! By contrast, **sabbatical/exploratory mobility grants (`p_reloc = 0.02`)** directly break mid-career status-quo lock-in inside exhausted subfields, seeding new pioneers into unexplored territory and cutting the high-visibility welfare penalty by $40\%$.

---

# 4. Act III: Orthodox Bailouts vs. Unclaimed Frontier Paradigms (1.3 — `P-0004` Executed)

`interventions_v1` Arm C ([interventions_v1.md](file:///Users/joko/Git/sciencemodels/output/reports/interventions_v1.md#L129-L165), 18 cells / 360 runs) activated the dormant `new_questions` regeneration channel (`sciencemodels/model.py:422-429`), contrasting two philosophies of how scientific questions replenish when a topic is exhausted below $0.025$:
1. **`occupied` ("Orthodox / Incumbent Hot-Field Bailout"):** New questions $U(0,1)$ appear *only on tiles where scientists are already standing* (the original rule coded in `model.py:424`).
2. **`frontier` ("Kuhnian Frontier Paradigm Emergence"):** When an occupied tile is exhausted, the new question $U(0,1)$ opens on a random *unoccupied* tile elsewhere on the grid.

**What `interventions_v1` Arm C proved:**
- **Orthodox bailout (`occupied`) induces catastrophic intellectual narrowing:** The share of the epistemic grid ever explored collapses by **84%** (from $37.7\%$ in the no-refill baseline to **$6.1\%$**), `Corr_prestige_ideal_merit` stays negative under entry ($-0.045$), and the newcomer-incumbent prestige gap widens **eleven-fold** (from $-0.019$ to **$-0.451$**).
- **Frontier paradigm emergence (`frontier`) restores positive quality alignment (`+0.078`), but new paradigms pile up unclaimed:** Under `frontier` refills with entry on, `Corr_prestige_ideal_merit` flips positive ($+0.078$) because the few mavericks who reach new frontier questions harvest massive ideal merit. Yet the majority of the community remains trapped in old visibility clusters — leaving **unclaimed frontier knowledge $315\times$ higher than baseline** (`mean_grid_knowledge = 0.1895` vs. $0.0006$ baseline, while exploration drops from $37.7\%$ to $33.0\%$).

![FC: Frontier refills raise prestige concentration](../../data/interventions_v1/figures/FC_refill_lockin.png)

**Contextual interpretation in real-world science (Dobusch & Kapeller 2009, 2012; Kuhn 1962):**
- The `occupied` regime captures **self-perpetuating normal-science orthodoxy** (or "manufactured novelty" in established mainstream schools, cf. Dobusch & Kapeller 2009): when journals and funders continuously manufacture fresh micro-puzzles inside whatever subfield the incumbent elite already occupies, scientists never need to leave their starting cluster (`94%` of the epistemic landscape is never even visited!), and senior insiders amass an insurmountable $11\times$ prestige lead over newcomers.
- The `frontier` regime captures **paradigm lock-in in the face of genuine scientific opportunities**: even when nature opens rich new questions on the intellectual frontier (`best_knowledge = 0.992`), a discipline whose reward function weights visibility ($V_i^{1-c}$) and local peer proximity cannot coordinate a migration to the frontier — leaving revolutionary questions unharvested while the crowd fights over crumbs on exhausted peaks. Testing whether **sabbatical mobility shocks (`p_reloc`) unlock these unclaimed frontier paradigms** is the subject of [P-0007-sabbatical-shocks-x-frontier-refills.md](file:///Users/joko/Git/sciencemodels/AdmiralsLog/proposals/P-0007-sabbatical-shocks-x-frontier-refills.md).
