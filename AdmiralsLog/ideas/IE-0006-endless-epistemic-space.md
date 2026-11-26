---
title: "IE-0006 - Options for an endless epistemic space"
author: "Antigravity, Paul"
date: "2026-11-16"
updated: "2026-11-24"
tags: [idea, epistemic-landscape, open-endedness, refills, scaling, horizon]
project: "sciencemodels"
status: idea
---

- **Status:** idea
- **Date:** 2026-11-16
- **Updated:** 2026-11-24
- **Related:** `AdmiralsLog/proposals/P-0007-sabbatical-shocks-x-frontier-refills.md`, `output/reports/interventions_v1.md` (Arm C), `output/reports/turnover_v1.md` (`R6`), `drafts/paper_sketch_v1.md` (Sections 4.3 & 5.3)

## The idea

Currently, `sciencemodels` runs on a fixed, finite $20 \times 20$ torus (`MultG2`, `MultG6`, `Noisy2`) with a static initial knowledge endowment ($\sum k_{xy}(0) \approx 2.0$) that depletes monotonically over $T = 200\text{–}400$ steps. Once the initial peaks are mined out, the system enters a **Depletion Phase** ($t \gtrsim 150$) where global exhaustion drives signal inversion (`turnover_v1` `R6`). Substantively, however, real epistemic space is **endless**: science does not run out of questions globally, even when individual research programs exhaust.

Four design options can make epistemic space effectively endless, with distinct mechanical and analytical tradeoffs:

1. **Increasing the Grid Size ($L \times L \gg 20 \times 20$) and Peak Count ($K \propto L^2$):**
   - *Mechanism:* Scale the torus from $20 \times 20$ to e.g. $50 \times 50$ or $100 \times 100$ with proportionally many Gaussian peaks (or sparse frontier peaks) so total initial knowledge far exceeds what $N=20$ agents can harvest within $T$ steps.
   - *Pros:* Preserves local neighborhood search (`use_distance=1`), true topic-level depletion on visited peaks, and exact `ideal_Merit` computability over the grid.
   - *Cons:* `ideal_Merit` and global visibility/distance calculations scale with $L^2$; `percentage_knowledge_harvested` shrinks mechanically toward $0\%$ as $L$ grows; and without higher mobility (`p_reloc` or jump search), agents still remain trapped in a local patch of the larger torus.
2. **Decreasing the Number of Turns ($T \lesssim 80\text{–}100$):**
   - *Mechanism:* Truncate simulation horizons before the community exits the Discovery Phase ($t \lesssim 80$).
   - *Pros:* Zero code changes and faster runs.
   - *Cons:* **Does not make epistemic space endless—it merely censors the Depletion Phase.** Shortening $T$ precludes studying steady-state generational turnover, career-long trajectories, or long-run paradigm lock-in (`turnover_v1` `R5`–`R6`).
3. **Refilling / Regenerating Grids (Dynamic Paradigm Renewal):**
   - *Mechanism:* Keep a compact $20 \times 20$ grid and continuously or stochastically inject new knowledge—extending `interventions_v1` Arm C (`occupied`, `uniform`, `frontier`) to **stochastic new-peak birth** (periodically spawning fresh Gaussian peaks on unvisited or exhausted regions, or decaying/recycling depleted tiles into new questions).
   - *Pros:* Keeps `ideal_Merit` fast ($O(400)$), supports arbitrarily long horizons ($T \to \infty$) with stationary-ergodic dynamics, and models Kuhnian paradigm shifts directly.
   - *Cons:* As `interventions_v1` Arm C (`R8`) showed, `occupied` refill induces severe mainstream lock-in ($-83.8\%$ exploration) while `frontier` refill leaves new peaks unclaimed ($315\times$ residual knowledge) unless paired with mobility (`sabbatical` shocks, `P-0007`) or turnover. Also requires replacing `percentage_knowledge_harvested` (share of static initial stock) with a flow/rate metric (e.g., harvest per step relative to frontier potential).
4. **Open-Ended / Procedurally Generated $\mathbb{R}^2$ Frontier (Scrolling or Chunked Infinite Space):**
   - *Mechanism:* Replace the periodic torus with an unbounded continuous or chunked 2D space where new landscape chunks are generated procedurally on demand as pioneers move outward.
   - *Pros:* True open-endedness with no periodic wrap-around artifacts.
   - *Cons:* Breaks the current finite-grid `ideal_Merit` greedy-planner benchmark unless `ideal_Merit` is redefined over a bounded horizon-reachable ball around active agents, and requires rethinking spatial distance/visibility normalization.

## Assessment note

**Option 3 (dynamic peak birth / `frontier` refill paired with mobility shocks, `P-0007`)** is the most promising path for `sciencemodels`: it preserves the fast $20 \times 20$ `ideal_Merit` yardstick and isolates *why* endless frontiers still fail to prevent lock-in when visibility herding anchors scientists to old paradigms. **Option 1** ($L \times L$ scaling) is a clean secondary robustness check for finite-size sensitivity.

## New option to be assessed

5. **Refilling the Grid on neigbours:**
   - *Mechanism:* When knowledge is harvested on one cell, distribute the harvested knowledge in *another* neighbouring cell following a gaussian probability
   - *Pros:* Keeps `ideal_Merit` fast ($O(400)$), supports arbitrarily long horizons ($T \to \infty$) with stationary-ergodic dynamics, and models Kuhnian paradigm shifts directly.
   - *Cons:* Requires replacing `percentage_knowledge_harvested` (share of static initial stock) with a flow/rate metric (e.g., harvest per step relative to frontier potential). Estimation of the dynamics to be decided

## Code-verified assessment

The Gaussian-neighbour refill (Option 5) is new code with a $\sigma$-dependent failure mode at both ends. `MyModel.farm` only removes harvested knowledge from the harvested tile (`sciencemodels/model.py:393-404`); no redistribution kernel exists. With small $\sigma$, refilled stock lands inside the 4-neighbour search horizon (`sciencemodels/agents.py:194-196`), so agents re-harvest the same local peak and visibility herding reproduces the `occupied` regime's mainstream lock-in (Arm C `R8`, $-83.8\%$ exploration). With $\sigma$ approaching the torus scale the kernel tends to uniform injection, leaving peaks unclaimed (`frontier`-like, $315\times$ residual knowledge) unless mobility (`P-0007`) is added. $\sigma$ must therefore be reported relative to both the search radius and $L$, with explicit truncation and renormalisation on the $20\times20$ torus. Finally, `percentage_knowledge_harvested` is `(total_initial_knowledge - sum(knowledge)) / total_initial_knowledge` (`sciencemodels/model.py:542-545`), so injection can make it fall or go negative; the rate/flow metric named in the Cons is mandatory.
