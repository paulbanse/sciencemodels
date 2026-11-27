---
title: "IE-0007  - Draft review proposals"
author: "Paul"
date: "2026-11-24"
tags: [idea]
project: "sciencemodels"
status: embraced
---

- **Status:** embraced (2026-11-27 — all 8 items addressed and integrated into `drafts/paper_sketch_v1.md` v3.5 and `ADR-0003`)
- **Date:** 2026-11-24
- **Updated:** 2026-11-27
- **Related:** `AdmiralsLog/decisions/accepted/ADR-0003-time-scale-anchoring.md`, `drafts/paper_sketch_v1.md` (v3.5), `output/reports/adr0003_remeasure_v1.md`, `output/reports/m2_decomposition_v1.md`, `output/reports/attraction_heatmap_landscapes_v1.md`

## The idea

1. **Trying to make the framing closer to real academia:**
   - *Mechanism:* In the redaction of the article, try justifying design parameter values regarding using the general context. 
   - *Examples:* Let's say a step in the model corresponds to 6 month in an academic career. When agents are not immortals, one could say that then the generation parameter and generational replacement should allow for a full generation to be renewed within 80 steps, and people should retire for the population to stay relatively constant. Similarly, in science, an article looses importance in one's career as it gets older. The scale of the vanishing parameter can be estimated and justified in the draft based on real life situations, for example one could say that the prestige from an article published 5 years ago is half of its initial prestige. Justifications like these should be investigated for most of the parameters that are unjustified in the model.
   - *Pro:* It gives a new way to justify some parameters that were before unjustified and increases the likelihood to be relevant for a reader in academia
   - *Cons:* It requires a bit more explanation in the article

2. **The vanishing parameter tested is too big:**
   - *Mechanism:* testing vanishing values of 0.5, 0.9 and 1.0 creates extremely strong vanishment, lower values such as 0.1 for example would be more realistic. 
   - *Pros:* this will allow having some longer term signal on prestige which makes it more likely to correlate with ideal Merit
   - *Cons:* not yet identified
   
3. **The use of distance= 0 version is exposed confusingly in the article:**
   - *Mechanism:* In the draft, the simulations where distance = 0 appear are called simulations with "jump" but there is no mention of a jump mechanism. If distance = 0 is useful it has to be justified, if the only thing distance = 0 does is to de-activate visibility, then it is hard to justify using it with different level of curiosity.
   - *Pros:* Potentially delete the distance = 0 option, or make it usefull for the reader
   - *Cons:* difficult to forsee given that distance = 0 is not explained

4. **Standardize simulation procedure:**
   - *Mechanism:* a paragraph should be written that summarizes what we use as a standard simulation procedure. How many steps in a simulation, what is the stopping criterion. Values of curiosity can be picked to be representative of the stage1, kink, stage2 situations (for example 0.5, 0.7, 0.9) in order to allow for representation along a different x axis.
   - *Examples:* Figure 4 first panel shows the result of the simulation at 10% knowledge harvested. One could wonder why was 10% used in that situation, if there's a justification it is not present
   - *Pros:* helps the reader understand what is the normal situation and which parameters are used in a single section
   - *Cons:* uses space in the potential article.

5. **The use of the word "lab" is not justified:**
   - *Mechanism:* the article uses the word lab probably in reference to Smaldino, but this usage is unclear and needs to be justified or deleted
   - *Pros:* Deletion would make the article less confusing, and smaller
   - *Cons:* 
   
6. **Figure 9 method is interesting:**
   - *Mechanism:* the method used to produce figure 9 is very interesting, it questions how beneficial is the fact that the new researchers are created on the cell of their supervisor. And it shows that despite putting the researcher on a cell that's for sure already explored, it also puts it into a cell that's close to interesting research spots. This is relevant in real life and an interesting question, I would suggests other design questions of the model to be investigated this way. 
   - *Pros:* 
   - *Cons:* 

7. **Figure 1: Simulation grid examples**
   - *Mechanism:* Figure 1 identifies different phases (Stage 1,2 and kink) with very few data point, I would advise having at least 2 point per phase to identify a bit more precisely the transition. Once these stages are properly identified, I would advise adding in Figure 1 three panels with representative examples of research behaviors during simulations, potentially in a noisy landscape: In Stage 1, researchers would be scattered accross the board, in the kink phase, researcher would be grouping but still exploring, and in stage 3 researchers would be exploring much less.
   - *Pros:* More instutitive visualization of our work
   - *Cons:* It could look a bit basic.

8. **Missing Visualization**
   - *Mechanism:* There is only one temporal visualization of simulation and it's to look at the evolution of corr(prestige, ideal Merit). Investigate the different possible temporal visualizations, for example Figure 2 could use the standardize procedure (see point 4 of this file) and show the temporal dynamics of the different simulations instead of the end point. Or one could investigate the evolution of curiosity along time in populations that reproduce for example.
   - *Pros:* It will give a more instinctive understanding of the work
   - *Cons:* uses space

## Code-verified assessment

- **#2 is inverted by the code.** `vanishing_factor_for_prestige` is applied in `Scientist.increase_prestige`: the accumulated stock is multiplied by $x$ and the new accrual added (`sciencemodels/agents.py:84-91`, `temp = temp * x; temp += value`). $x$ is a retention factor (1.0 = no forgetting; 0.5 = halve the stock per accrual), not a decay rate. The tested values 0.5/0.9/1.0 already span strong-to-no forgetting; lowering to 0.1 would strengthen forgetting, not weaken it, and weakening it means raising $x$ toward 1.0. It also acts per accrual event (roughly per step), so a "half after five years" reading needs a step-to-time mapping.
- **#3 is confirmed, with a stronger consequence.** With `use_distance=0`, `compute_all_rewards` returns `curiosity * novelty / avg_current_agent_knowledge` and zero visibility (`sciencemodels/agents.py:123-126`), while `Scientist.step` still evaluates only the 4-neighbour von Neumann neighbourhood (`sciencemodels/agents.py:194-196`). This is local search with visibility off, not jump search. Because curiosity is a common multiplicative constant across all candidate rewards, it cancels in the comparison; under `use_distance=0` curiosity cannot alter movement (only `epsilon` noise can), so the draft's "different levels of curiosity" comparison is mechanically empty unless the reward is reformulated.


## Final Disposition Report (2026-11-27 — Embraced)

All eight review proposals from Paul Banse (`f3d53a3`) have been resolved and integrated into **Paper v3.5** (`drafts/paper_sketch_v1.md` / `.pdf`) and the core package (`sciencemodels/`):

| # | Proposal | Resolution | Executed Artifacts / Commits |
| ---: | :--- | :--- | :--- |
| **1** | **Real-academia time-scale framing** (1 step = 6 months, 80-step career, constant-population retirement, 5-year prestige half-life) | **Adopted in full as `ADR-0003`** (mechanical re-parameterization, non-calibration preserved): `age` $25 \to 65$ (`+0.5`/step), `tau_max = 80`, $g_{\text{canon}} = 0.25$, $\gamma = 0.5^{1/10} \approx 0.933$, `fixed_span` promoted to core; 6,380 + 600 + 5,400 runs re-executed on the new clock and all 11 core figures rebuilt. | `ADR-0003`, `8b36a72`, `4e3c301`, `08397f8`, `06ecf65` |
| **2** | **Vanishing parameter tested is too big** | **Code inversion clarified & disambiguated:** `vanishing_factor_for_prestige` is a *retention* factor in $P_{\text{van}}(t) = \gamma P_{\text{van}}(t-1) + \Delta P(t)$ ($\gamma = 0.9$ = 10% decay/step, $\sim 3.3$-yr half-life; $\gamma = 0.1$ would be 90% decay/step). Adopted Paul's 5-year half-life target ($\gamma = 0.933$) and replaced "vanishing pin" terminology with explicit retention-factor notation throughout the paper. | `ADR-0003`, `paper_sketch_v1.md` §§2.2–2.3 (v3.3–v3.5) |
| **3** | **`use_distance = 0` confusingly called "jump"** | **Adopted (removed):** Confirmed `use_distance = 0` disables visibility on the 4-neighbour grid rather than implementing jumps. Removed the inaccurate §3.2 and old Figure 3, and updated all figure numbers and provenance tables. | `paper_sketch_v1.md` v3.3 (`06ecf65` / `4c65bef`) |
| **4** | **Standardize simulation procedure & justify 10% harvest milestone** | **Adopted:** Added §2.2.1 *Standard Simulation Protocol* fixing canonical $H = 200/400$, $\gamma = 0.933$, $g = 0.25$, representative slices $c \in \{0.9, 0.5, 0.1\}$, and justifying `harvested_10_step` as a horizon-independent early-discovery speed milestone. | `paper_sketch_v1.md` §2.2.1 (v3.4–v3.5) |
| **5** | **Unjustified use of `"lab"`** | **Adopted:** Replaced both stray `"lab"` / `"laboratories"` occurrences with `"mentor-advisee cluster"` (§4.1) and `"research group"` (§5.1). | `paper_sketch_v1.md` v3.3 |
| **6** | **Extend Figure 9's counterfactual decomposition method** | **Adopted:** Extended Figure 9's additive ablation-share decomposition to `Tier M2` (`dyadic_move_pull` vs. nominal links vs. `equal_split`), independently reproduced to 0-digit mismatch, and re-verified at $n = 40$ on the `ADR-0003` clock (co-movement carries $+96.6\%$–$+117.9\%$). | `681497d`, `fb797fb`, `08397f8`, `output/reports/m2_decomposition_v1.md`, `paper_sketch_v1.md` §6 |
| **7** | **Densify Figure 1 + 3-panel spatial grid snapshots** | **Adopted:** Densified Figure 1 (`F6_v2`) to an **11-point $\xi$ grid** (`data/fig1_xi_adr0003/`, 1,100 runs) with $\ge 2$ points per phase, built the 3-panel `Noisy2` spatial snapshot figure (Stage 1 scattered / Kink clustering / Stage 2 single-tile collapse) as **Figure 3**, and added **Figure 1b** (`F_attraction_trias_heatmap`, 12/12 vertical panels across 5,400 runs on `mgstd5`, `MultG6`, `Noisy2`). | `06ecf65`, `c2b94eb`, `4049f28`, `4c65bef` |
| **8** | **Missing temporal visualizations** | **Adopted:** Built and integrated **Figure 4** (`D5_temporal_Noisy2_H200_v0p933_g0_n20`: 2-panel harvest & exploration trajectories over $t = 0\text{..}200$ with the 10% milestone line and 95% CI bands). Evolutionary curiosity-over-time visualization is scoped under `IE-0008` / future `P-0012`. | `scripts/fig_prototypes/build_snapshots_and_temporal.py`, `paper_sketch_v1.md` §3 (Fig. 4) |
