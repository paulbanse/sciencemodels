---
title: "CONT1 - Spatial Herding, Institutional Counterforces, and Paradigm Refills on the Epistemic Grid"
author: "Antigravity (Review & Synthesis Round)"
date: "2026-11-14"
tags: [contribution, mechanism, spatial-herding, counterforces, paradigm-shifts, path-c]
project: "sciencemodels"
---

- **Source:** Antigravity inspection & evaluation round (Items 1.1–1.3)
- **Date:** 2026-11-14
- **Related:** `AdmiralsLog/way_forward.md`, `output/reports/sensitivity_map_v1.md`, `output/reports/welfare_probe_v1.md`, `output/reports/career_flows_v1.md`
- **Spawned:** `P-0002`, `P-0003`, `P-0004`

# 1. One Unified Mechanism Story

Across the first three exploration waves (`sensitivity_map_v1`, `career_flows_v1`, `welfare_probe_v1`), the model exhibits three striking macro-phenomena:
1. Activating prestige-proportional entry flips the quality correlation (`Corr_prestige_ideal_merit`) negative in 17 of 18 cells while local quantity correlation (`Corr_prestige_local_merit`) stays positive.
2. Newcomers never close the prestige gap with incumbents (holding roughly one-third of incumbent prestige on clean terrain and one-half on noisy terrain).
3. Epistemic welfare loss kinks sharply upward at visibility weight $\xi^* \in [0.6, 0.8]$, driven $31\times$ more strongly by the agent's internal visibility elasticity ($1-c$) than by the peer-prestige visibility toggle (`use_visibility_reward`, $2.6\%$).

Items 1.1, 1.2, and 1.3 form a single causal chain explaining **why** this happens on the $20 \times 20$ torus, **how** institutional rules can counter it, and **what** happens when the epistemic landscape itself regenerates.

---

# 2. Act I: Disentangling Spatial Herding from Birth-Tile Starvation (1.1)

In the current demographic rule (`sciencemodels/model.py:315-338`), two mechanisms are fused at birth:
- **Prestige-proportional matching:** A newcomer selects a supervisor $i$ with probability $P(\text{sup}_i) \propto \text{prestige}_i$.
- **Exact spatial co-location:** The newcomer is placed directly on the supervisor's current tile (`coords = sup.pos`).

Because harvesting depletes tile knowledge multiplicatively ($k_{xy} \leftarrow (1-h)k_{xy}$, `model.py:303-307`), an incumbent who has accrued high prestige has necessarily stood on and depleted their local peak. When newcomers spawn at `sup.pos`, they inherit **exhausted ground** while simultaneously increasing local agent density (which inflates the visibility reward $v = (\bar{d}_{\text{global}}+1)/(d_{\text{own}}+1)$ and traps neighbors via strict-improvement inertia).

Thus, the negative `Corr_prestige_ideal_merit` under entry mixes two distinct forces:
1. **Mechanical birth-tile starvation:** Advisees are born on depleted tiles and harvest crumbs while their supervisor retains historical prestige.
2. **Social/informational herding (attention cascades):** Even away from the birth tile, visibility-seeking ($1-c$) pulls agents toward crowded prestige clusters rather than unexplored peaks.

Separating spatial starvation at birth from attentional herding in motion is a clean conceptual contribution to the Matthew-effect literature (Merton 1968; Petersen et al. 2014): cumulative advantage in science operates simultaneously through **positional inheritance** (where advisees start on the epistemic landscape) and **evaluative attraction** (how visibility rewards pull searchers toward established clusters). *(The experimental design to decompose these two channels is filed separately in [P-0002-spatial-spawn-dispersal.md](file:///Users/joko/Git/sciencemodels/AdmiralsLog/proposals/P-0002-spatial-spawn-dispersal.md), and the companion test of stationary generational turnover is filed in [P-0006-demographic-turnover-steady-state.md](file:///Users/joko/Git/sciencemodels/AdmiralsLog/proposals/P-0006-demographic-turnover-steady-state.md).)*

---

# 3. Act II: Institutional Counterforces at the $\xi^* \in [0.6, 0.8]$ Kink (1.2)

Once the spatial and attentional mechanics of the $\xi^* \in [0.6, 0.8]$ collapse kink are understood, the model speaks directly to the institutional-reform debate (Smaldino, Turner, & Contreras Kallens 2019; Gross & Bergstrom 2019; Osterloh & Frey 2015; Avin 2019). Specifically, because proxy failure in this ABM arises from the interaction of **status-quo lock-in** (`agents.py:166`, strict utility improvement on a noise-free current tile) and **linear prestige-proportional recruitment** (`model.py:315-326`), three institutional mechanisms act as natural theoretical counterforces:

1. **Recruitment De-Targeting (Damped Entry / Funding Lotteries):**
   Weakening the link between accumulated prestige and advisee allocation prevents depleted peaks from acting as demographic sinks while preserving exploration incentives.
2. **Transient-Diversity Preservation (Zollman 2010):**
   Breaking local status-quo lock-in inside high-visibility clusters restores cognitive diversity without requiring individual scientists to become altruistic.
3. **Topical Complementarity in Co-Production (Smith et al. 2021):**
   Rewarding intermediate epistemic distance counteracts pure visibility assortativity.

*(The concrete experimental specification for testing these institutional counterforces against the $\xi^*$ collapse threshold is filed separately in [P-0003-institutional-counterforces-lotteries.md](file:///Users/joko/Git/sciencemodels/AdmiralsLog/proposals/P-0003-institutional-counterforces-lotteries.md).)*

---

# 4. Act III: Endogenous Paradigm Shifts via the Dormant `new_questions` Channel (1.3)

All explorations in Waves 1–3 pinned `new_questions = 0`, treating science as the exhaustion of a finite knowledge stock. Yet `sciencemodels/model.py:342-349` already contains a dormant regeneration rule:

```python
if self.new_questions:
    occupied_pos = list(set([agent.pos for agent in self.agents]))
    for pos_x, pos_y in occupied_pos:
        if self.grid.properties["knowledge"].data[pos_x, pos_y] < 0.025:
            self.grid.properties["knowledge"].data[pos_x, pos_y] = self.rng.random()
```

Notice the critical micro-assumption on line 344: **only currently occupied tiles are refilled** (`occupied_pos`). Conceptually, this models **incumbent hot-field bailout** (or "manufactured novelty" in crowded fields, cf. Ioannidis 2005; Dobusch & Kapeller 2009 neoclassical lock-in): when a crowded cluster exhausts its tile below $0.025$, a fresh question $U(0,1)$ appears right under the incumbents' feet, rewarding them for staying in the herd.

The conceptual distinction between **incumbent hot-field bailout** (new questions generated where the crowd already stands) and **frontier paradigm emergence** (new questions opening on unoccupied regions of the epistemic landscape) links the ABM directly to pluralist vs. monist theories of scientific paradigms (Dobusch & Kapeller 2009, 2012). Under incumbent bailout, visibility herding looks benign because orthodoxy self-replenishes; under frontier emergence, visibility-locked communities miss paradigm shifts entirely. *(The experimental proposal to test these refill regimes is filed separately in [P-0004-paradigm-refill-regimes.md](file:///Users/joko/Git/sciencemodels/AdmiralsLog/proposals/P-0004-paradigm-refill-regimes.md).)*
