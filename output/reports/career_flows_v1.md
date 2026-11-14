---
title: "Career flows and the entry oscillation: confirmation at horizon 400 (T2)"
author: "Hermes Agent (squad lead) for Jakob Kapeller"
version: 1
date: "2026-11-13"
tags: [career-flows, entry, oscillation, proxy-failure, sensitivity-confirmation, not-calibrated]
project: sciencemodels
revision-vehicle: strikethrough
revision-renders: "md, pdf, html"
---

**Version 1** (November 2026)

**Verdict: T1's headline oscillation claim SURVIVES at horizon 400.** All 6/6 confirmation cells oscillate (45-127 pooled sign changes; per-seed range 2-137, 238/240 runs with >= 3) and all 6/6 end with the pooled ideal correlation below zero (seed-level final sign rate 0.60-0.90 per cell, 181/240 runs overall); 0 cell mean-reverts. The noisy cells end clearly negative (pooled finals -0.133 to -0.102); the clean cells also end negative but their last-third pooled means are positive-leaning (+0.025 to +0.056) versus negative on noisy (-0.019 to -0.004) --- in a dense sawtooth the final step's sign is phase-sensitive, so the seed-level sign rate, not the single final read, carries the claim.

> **NOT CALIBRATED.** Model-internal exploration only (owner ruling 2026-11-13).
> No empirical moment was used as a target; the Klann panel appears below solely
> as reference context for the steps-to-years framing, and that framing is
> conditional arithmetic, not fitting.

# Design

Two sweeps, both driven through `MyModel` directly (no sweep-writer), pins as in
T1 (`scripts/bulk_runner.py` base): $n=20$ founders, 20x20 torus,
$\epsilon=0.1$, $h=0.02$, `use_distance=True`, `new_questions=0`,
`use_visibility_reward=True`, `knowledge_normalization='share'`,
`prestige_accrual='harvested'`, vanishing $x=0.9$ throughout (the tree's unified
value), identity curiosity function.

| Sweep | Cells | Grid | Seeds | Horizon | Runs |
| --- | --- | --- | --- | --- | --- |
| A (confirmation) | 6 | $c \in \{0.3, 0.5, 0.7\} \times$ {mgstd5, noisypr5}, $x=0.9$, $g=0.1$ | 40 | 400 | 240 |
| B (career flows) | 9 | $g \in \{0.05, 0.1, 0.2\} \times c \in \{0.3, 0.5, 0.7\}$, $x=0.9$, mgstd5 | 20 | 200 | 180 |

Declared pins and disclosed deviations:

- **(a) ran 6 cells, not 4.** The brief said "4 representative entry cells" but
  enumerated curiosity 0.3/0.5/0.7 x {mgstd5, noisy} = 6; the full crossing was
  run and is reported. No level was dropped.
- **(b) landscape pinned to mgstd5** (the canonical bulk bundle). The brief named
  no landscape for the career sweep; legibility was T1's axis, demographics are
  this task's question, and the within-$g$ contrasts are the deliverable. A noisy
  rerun is the natural extension.
- Population under entry follows $21 + \lfloor g \cdot t \rfloor$ (the +1 is T1's
  Q1 init-time accumulator quirk); sweep A's final population is 61, not 60.
- The agent-vars frame cannot distinguish the init-time extra spawn (entry step 0)
  from founders; the newcomer-incumbent gap therefore treats it as incumbent
  (1 agent out of 21-61; effect negligible, disclosed).
- `prestige_gap_ratio` is degenerate while incumbent prestige is ~0 (ratio pinned
  at -1); it is reported from step 50 onward (`gap ratio (late)` = mean over
  steps 50-200).
- Sweep A does not record per-run population (analytic rule above suffices);
  sweep B records it per step from the agent-vars frame.

# Canary and constraints

Canary gate (`MPLBACKEND=Agg /opt/hermes/.venv/bin/python tests/test_smoke.py`;
the driver now hard-gates on it at launch --- aborting before any cell on
failure --- added as a post-sweep source addition, see Reproducibility; the
embedded run is the document-time execution; exit code 0):

```{.bash}
$ MPLBACKEND=Agg /opt/hermes/.venv/bin/python tests/test_smoke.py
canary run OK: 40 report keys, harvested_10_step=37.5, harvested_25_step=37.5
same-seed rerun bit-identical OK
newcomer run OK: 6 scientists after 20 steps
4-arg Gaussian OK: np.float64(0.00550470835249002) (float64)
/tmp/smoke_append_lwaexkrt/append_header_check.csv w
now processing param set  1 out of 1
/tmp/smoke_append_lwaexkrt/append_header_check.csv a
now processing param set  1 out of 1
append-mode header check OK: 3 lines = 1 header + 2 data rows
ALL SMOKE TESTS PASSED
```

Concurrent-sweep hygiene (constraint compliance):

- `n_jobs=3` cap honored in the driver (a foreign sweep, `scripts/figures/run_fresh_sweep.py`
  writing `data/figures_v1/`, was observed live at launch, plus three Jupyter
  kernel runners; my paths are disjoint from theirs).
- No orphaned processes of this squad before launch (verified via ps; the kernel
  runners and the foreign sweep are not mine to touch).
- Restart-safe checkpoint per completed cell (`checkpoint.json`, atomic replace;
  the SKIP-on-restart path is implemented but was NOT exercised --- this run
  completed without interruption, as a single continuous pass). Crash policy:
  failing seeds recorded, never dropped.
- No git commands; writes confined to `data/career_flows_v1/`,
  `scripts/career_flows/`, `output/reports/career_flows_v1.md`. Frozen `context/` and
  read-only `AdmiralsLog/` untouched.

Gross vs net --- sweep A: 6 cells gross / 6 net,
240 runs gross / 240 net, 0 failed seeds,
0 crashed cells. Sweep B: 9 cells gross / 9 net,
180 runs gross / 180 net, 0 failed seeds,
0 crashed cells. Total 420 runs,
420 completed.

# Sweep A --- does the oscillation survive horizon 400?

T1 (horizon 200, 20 seeds) found the ideal channel under entry ending negative
in 17/18 cells, with oscillation asserted at the pooled landscape-x-entry level
(35 and 17 pooled sign changes). Confirmation rule (declared before running): a
cell *oscillates* if the pooled trajectory has >= 3 sign changes and a negative
share strictly between 5% and 95%; it *ends negative* if the pooled final value
is < 0; it *mean-reverts* if it oscillates but ends >= 0. Cells whose final
exact-zero (zero-guard) share exceeds 50% are classed degenerate and excluded
from the oscillation count. Sign changes count strictly opposite consecutive
signs; exact zeros break neither. No cell was classed degenerate in this round:
exact-0 final reads are 0.0 in all six cells (the entry-grown population
preserves variance; see the zero-guard finding below).

| cell | sign changes (pooled) | neg share | final pooled | final sign rate | last-third mean | exact-0 final share | harvested final | verdict |
|---|---|---|---|---|---|---|---|---|
| cur3_van9_mgstd5_gen0p1 | 83 | 0.113 | -0.105 | 0.875 | 0.025 | 0.000 | 0.906 | oscillates_ends_negative |
| cur3_van9_noisypr5_gen0p1 | 127 | 0.388 | -0.133 | 0.900 | -0.019 | 0.000 | 0.574 | oscillates_ends_negative |
| cur5_van9_mgstd5_gen0p1 | 77 | 0.098 | -0.031 | 0.625 | 0.040 | 0.000 | 0.929 | oscillates_ends_negative |
| cur5_van9_noisypr5_gen0p1 | 73 | 0.235 | -0.106 | 0.750 | -0.004 | 0.000 | 0.654 | oscillates_ends_negative |
| cur7_van9_mgstd5_gen0p1 | 45 | 0.058 | -0.015 | 0.600 | 0.056 | 0.000 | 0.929 | oscillates_ends_negative |
| cur7_van9_noisypr5_gen0p1 | 51 | 0.315 | -0.102 | 0.775 | -0.006 | 0.000 | 0.657 | oscillates_ends_negative |

Zero-guard / depletion context (the T1-F5 protocol at the longer horizon):

| cell | exact-0 final share | harvested final | zero share pooled | mean age final |
|---|---|---|---|---|
| cur3_van9_mgstd5_gen0p1 | 0.000 | 0.906 | 0.000 | 292.800 |
| cur3_van9_noisypr5_gen0p1 | 0.000 | 0.574 | 0.000 | 292.600 |
| cur5_van9_mgstd5_gen0p1 | 0.000 | 0.929 | 0.000 | 292.800 |
| cur5_van9_noisypr5_gen0p1 | 0.000 | 0.654 | 0.000 | 292.600 |
| cur7_van9_mgstd5_gen0p1 | 0.000 | 0.929 | 0.000 | 292.800 |
| cur7_van9_noisypr5_gen0p1 | 0.000 | 0.657 | 0.000 | 292.600 |

Contrast with T1 on the same six cells (T1: horizon 200, 20 seeds, vanishing 0.9,
same landscape bundles. T1-side columns are re-derived from
`data/sensitivity_v1/` raw CSVs under this round's declared rule --- T1 itself
published endstate counts and pooled crossings, not per-cell oscillation
verdicts; read-only):

| cell | T1 sign chg (h200) | T1 final (h200) | T1 sign rate (h200) | T2 sign chg (h400) | T2 final (h400) | T2 sign rate (h400) |
|---|---|---|---|---|---|---|
| cur3_van9_mgstd5_gen0p1 | 53 | -0.145 | 0.900 | 83 | -0.105 | 0.875 |
| cur3_van9_noisypr5_gen0p1 | 41 | -0.206 | 0.900 | 127 | -0.133 | 0.900 |
| cur5_van9_mgstd5_gen0p1 | 39 | -0.054 | 0.550 | 77 | -0.031 | 0.625 |
| cur5_van9_noisypr5_gen0p1 | 19 | -0.111 | 0.750 | 73 | -0.106 | 0.750 |
| cur7_van9_mgstd5_gen0p1 | 36 | 0.020 | 0.550 | 45 | -0.015 | 0.600 |
| cur7_van9_noisypr5_gen0p1 | 7 | -0.187 | 1.000 | 51 | -0.102 | 0.775 |

Reading (all numbers from the tables above):

- Oscillation confirmed and dense: 6/6 cells oscillate with 45-127 pooled sign changes (one per ~3 steps at the dense end, one per ~9 at the sparse end --- the
  10-step entry waves reshuffle the prestige roulette). Per-seed the count ranges
  2-137 with 238/240 runs at >= 3: a population-level
  phenomenon, not a per-run law and not a pooling artifact of out-of-phase cells.
- End-state: 6/6 cells end with the pooled ideal correlation below
  zero; seed-level final sign rates run 0.60-0.90 per cell
  (181/240 runs overall). Because a dense sawtooth's final step is
  phase-sensitive, the sign rate is the robust statistic; the clean cells'
  last-third pooled means are positive-leaning while the noisy cells' are
  negative-leaning --- the clean end-state is a negative-leaning oscillation
  around zero, the noisy one a clearly negative end-state.
- The feared zero-guard degeneracy did NOT materialize: exact-0 final reads are
  0.0/40 seeds in every cell and the pooled zero share is 0.0 (T1's Q3 worry,
  refuted here --- the entry-grown population, 61 agents at step 400, preserves
  cross-sectional variance even at 90%+ harvested). The h400 confirmation is not
  depletion-guard-contaminated; this inverts the caveat T1 carried forward.
- Mean reversion: 0 cell(s) oscillate but end non-negative; the mid-run
  pooled minima (-0.418 to -0.193) are not corrected into stable positivity
  by the longer horizon.
- Horizon/seed extension: T1 (h200, 20 seeds) saw 7-53 pooled sign changes and
  final sign rates 0.55-1.00 on these six cells, with one ambiguous clean cell
  (curiosity 0.7, mgstd5: T1 final +0.020); at h400/40 seeds the same
  cells show 45-127 sign changes, the ambiguous cell flips to -0.015,
  and every cell's majority of seeds ends negative. T1's endstate 17/18
  (16/18 under this round's conjunctive rule) becomes 6/6 at horizon 400 with
  40 seeds on this crossing.

# Sweep B --- career flows under entry

Per-cell means over 20 seeds (pooled over steps where noted):

| cell | pop final | entrants | pop growth x | top10 @10 | top10 delta | gap final | gap ratio (late) | percap harvest | harvest flow/step (last50) | ideal corr final | mean age final |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cur3_van9_mgstd5_gen0p05 | 31.000 | 10.000 | 1.476 | 0.274 | -0.083 | -0.020 | -0.664 | 0.022 | 0.002 | -0.235 | 196.300 |
| cur3_van9_mgstd5_gen0p1 | 41.000 | 20.000 | 1.952 | 0.274 | -0.065 | -0.019 | -0.643 | 0.018 | 0.002 | -0.145 | 178.400 |
| cur3_van9_mgstd5_gen0p2 | 61.000 | 40.000 | 2.905 | 0.262 | -0.022 | -0.017 | -0.641 | 0.013 | 0.002 | -0.086 | 160.300 |
| cur5_van9_mgstd5_gen0p05 | 31.000 | 10.000 | 1.476 | 0.275 | -0.080 | -0.022 | -0.676 | 0.023 | 0.002 | -0.126 | 196.300 |
| cur5_van9_mgstd5_gen0p1 | 41.000 | 20.000 | 1.952 | 0.275 | -0.061 | -0.020 | -0.651 | 0.019 | 0.002 | -0.054 | 178.400 |
| cur5_van9_mgstd5_gen0p2 | 61.000 | 40.000 | 2.905 | 0.263 | -0.018 | -0.018 | -0.648 | 0.014 | 0.002 | -0.030 | 160.300 |
| cur7_van9_mgstd5_gen0p05 | 31.000 | 10.000 | 1.476 | 0.277 | -0.081 | -0.022 | -0.677 | 0.023 | 0.002 | -0.038 | 196.300 |
| cur7_van9_mgstd5_gen0p1 | 41.000 | 20.000 | 1.952 | 0.277 | -0.062 | -0.020 | -0.654 | 0.019 | 0.002 | 0.020 | 178.400 |
| cur7_van9_mgstd5_gen0p2 | 61.000 | 40.000 | 2.905 | 0.265 | -0.019 | -0.019 | -0.652 | 0.014 | 0.002 | 0.041 | 160.300 |

Pooled trajectories by entry rate (mean over the three curiosity cells and 20
seeds; gap = newcomer minus incumbent mean prestige; shown from step 50 --- the
ratio is degenerate while incumbent prestige is ~0):

| gen | step | pop | gap (new-inc) | gap ratio | top10 | percap harvest |
|---|---|---|---|---|---|---|
| 0.050 | 50.000 | 23.000 | -0.009 | -0.656 | 0.167 | 0.014 |
| 0.050 | 100.000 | 26.000 | -0.015 | -0.698 | 0.156 | 0.019 |
| 0.050 | 150.000 | 28.000 | -0.018 | -0.679 | 0.150 | 0.022 |
| 0.050 | 200.000 | 31.000 | -0.021 | -0.718 | 0.194 | 0.023 |
| 0.100 | 50.000 | 26.000 | -0.009 | -0.638 | 0.158 | 0.012 |
| 0.100 | 100.000 | 31.000 | -0.014 | -0.648 | 0.189 | 0.017 |
| 0.100 | 150.000 | 36.000 | -0.017 | -0.677 | 0.179 | 0.018 |
| 0.100 | 200.000 | 41.000 | -0.020 | -0.705 | 0.213 | 0.018 |
| 0.200 | 50.000 | 31.000 | -0.008 | -0.584 | 0.189 | 0.011 |
| 0.200 | 100.000 | 41.000 | -0.013 | -0.634 | 0.203 | 0.014 |
| 0.200 | 150.000 | 51.000 | -0.016 | -0.677 | 0.223 | 0.014 |
| 0.200 | 200.000 | 61.000 | -0.018 | -0.716 | 0.244 | 0.013 |

Findings:

- **Population grows linearly in $g$ exactly as the fractional accumulator
  predicts**: 31 / 41 / 61 scientists at $g=0.05/0.1/0.2$ (20 founders + 1
  init-time spawn + 10/20/40 scheduled newcomers); no exit channel exists, so
  growth is unbounded in the horizon.
- **The newcomer-incumbent prestige gap is persistent and negative at every
  entry rate**: newcomers start at zero prestige, spawn on their supervisor's
  tile (already depleted by the supervisor), and never close the gap within the
  horizon --- the late gap ratio (mean over steps 50-200) runs -0.68 to -0.64
  across all nine cells, i.e. newcomers hold roughly a third of incumbent
  prestige at every $g$ (newcomer mean = 0.36-0.32 \times incumbent; absolute gap -0.0220 to -0.0174). This is the model's internal career-dynamics
  statement of proxy failure at the individual level: prestige-proportional
  matching hands newcomers to the prestigious, and the prestigious stay
  prestigious. (Per-seed nuance, disclosed: at $g=0.2$ the gap turns briefly
  positive for individual runs at steps 7-9 --- 6-7 of
  20 seeds per cell, max +0.0009 --- while the pooled trajectory is
  strictly negative at every step in all nine cells, min -0.02202; the
  claim above is the pooled one the tables tabulate.)
- **Concentration: cross-sectionally higher under entry, temporally decaying
  within entry cells.** The early top-10% spike (0.26-0.28 at step 10)
  decays to 0.19-0.25 by step 200 as the
  entry cohort diversifies the prestige stock (per-cell deltas -0.083 to -0.018; the decay
  is weakest at high $g$, where dilution from the first waves prevents the spike
  from forming). Cross-sectionally, entry cells still sit above T1's founder
  cells at matched horizons (T1 final top-10% share: 0.115-0.164 without entry vs 0.208-0.250 with) ---
  both contrasts are true; they measure different things (level vs within-cell
  trajectory).
- **The decoupling signal is strongest at SLOW entry**: ideal-channel finals run
  -0.235 to -0.038 at $g=0.05$ versus -0.086 to
  +0.041 at $g=0.2$ --- with fast entry the prestige stock is diluted
  from the first waves and the supervisor roulette stays near-uniform longer.
  Descriptive pattern; the mechanism reading above is the only causal claim made.
- **Per-capita harvest falls with $g$**: cumulative harvested share rises (0.67-0.83)
  while per-capita cumulative harvest declines monotonically in $g$ (0.0217-0.0234 at $g=0.05$, 0.0177-0.0187 at $g=0.1$, 0.0131-0.0136 at $g=0.2$) --- entry dilutes
  individual productivity; the harvest flow per step per capita decays over the
  horizon everywhere (depletion).
- **Driver equivalence (unplanned integrity check):** sweep B's $g=0.1$ mgstd5
  cells re-run T1's entry cells with identical seeds and reproduce them
  bit-exactly at the raw-trajectory level (max abs diff 0.0e+00 over the
  five reporters x 4020 rows x 3 cells); the analysis-level aggregates are
  bit-identical at full precision too (same bit-exact inputs, same seed-mean
  operation). Diffs up to ~4e-05 appear only against T1's
  full-precision cell_summary, because this round's analysis archive is
  4dp-rounded --- display rounding, not summation order. The T1 grid and
  this round's grid are the same model.
- **Age is a recorded-but-dead lever** (noted, not fixed): `mean_age` reaches
  178 by step 200 (sweep B) and 293 at horizon 400 (sweep A). Individual
  founders pass 400 years of model age, but the population mean cannot: entry
  continuously injects 23-year-olds. Age has no lifecycle effect anywhere in
  the code (no death, retirement, or age-dependent behavior; newcomers reset
  to 23). Any career framing must use steps, not the age column.

# Steps-to-years framing (framing only, no fitting)

The model has no exit and no calendar, so the steps-to-years ratio is constrained
from the inside: demographic plausibility bounds it, the panel cannot discipline
it (its in-core coverage is ~2 author-years per author --- 50,830 author-years
over 25,498 authors, `docs/calibration_moments.yaml` --- so it observes short
windows, not careers). Conditional arithmetic, for assumed career spans:

| assumed career span (years) | steps/year at h200 | entrants/year, $g=0.05$ | entrants/year, $g=0.1$ | entrants/year, $g=0.2$ | years covered at h400 |
|---|---|---|---|---|---|
| 2.0 | 100.0 | 5.0 | 10.0 | 20.0 | 4.0 |
| 5.0 | 40.0 | 2.0 | 4.0 | 8.0 | 10.0 |
| 10.0 | 20.0 | 1.0 | 2.0 | 4.0 | 20.0 |
| 20.0 | 10.0 | 0.5 | 1.0 | 2.0 | 40.0 |
| 40.0 | 5.0 | 0.2 | 0.5 | 1.0 | 80.0 |

Reading (conditional arithmetic, no empirical quantity fitted): the table's
entrants/year column is $g \times$ steps/year. A field adding a few to tens of
percent of its active population per year --- the only demographic plausibility
window the model itself can speak to --- corresponds to roughly 20-100
steps per career-year (one step ~ 4-18 days): at 50 steps/year,
$g=0.05/0.1/0.2$ give 2.5/5/10 entrants per year into a founding community of
20 (12-50% of the founders per year, diluted as the population grows). At that
mapping the 200-step career sweep spans ~4 field-years and the 400-step
confirmation horizon ~8. Two structural consequences, stated as properties of
the model: (i) the ABM captures early-career / short-window dynamics, not full
careers --- full-career claims require an exit channel (death/retirement) the
model does not have; (ii) the population grows 1.48x-2.90x per 200-step window
with no attrition, so any multi-decade mapping is demographically implausible
without exit. The panel cannot discipline this ratio from its own coverage
(~2 in-core author-years per author); the mapping remains a declared
assumption, not a measurement.

# Model quirks this round surfaced

- **Q1 (carried from T1, replicated):** the init-time accumulator spawn makes
  step-0 population 21, not 20; all entry cells end at $21+\lfloor g \cdot H
  \rfloor$ scientists.
- **Q2 (carried, aggravated):** the ideal-merit benchmark is founder-sized
  (`range(number_agents)` = 20 ideal slots) while 31-61 agents actually harvest
  under entry; the ideal channel's degradation under entry is partly benchmark
  strain. The local channel (benchmark-free) is the cleaner career-flow
  companion and is recorded alongside.
- **Q5 (new, an a-priori caveat REFUTED by this round):** T1 carried forward a
  fear that long-horizon clean-landscape runs would be depletion-degenerate
  (zero-guard reads). At h400 this did NOT bind: exact-0 final reads are 0.0/40
  seeds in every cell and the pooled zero share is 0.0, despite 0.91-0.93
  harvested on the clean landscape --- the entry-grown population (61 agents)
  preserves cross-sectional variance. The h400 confirmation needs no zero-guard
  disclaimer; T1's h200 founder-cell caveat does not extend to entry cells at
  longer horizons.
- **Q6 (new, standing property):** the zero-guard still returns exactly 0 for
  degenerate stocks, so `neg share` and sign-change counts are conservative in
  principle --- true decoupling reads as zero, not negative, wherever variance
  collapses. It did not collapse here; the property is recorded for future
  founder-only or longer runs.

# Honest caveats

- Not calibrated (banner above); the steps-to-years section is conditional
  arithmetic, explicitly not a mapping claim.
- 40 seeds per confirmation cell is modest for sign-rate differences of ~0.1;
  the verdict rule is declared a priori and applied mechanically.
- The clean-landscape cells are heavily depleted by step 400 (0.91-0.93
  harvested) but NOT zero-guard-degenerate (Q5: cross-sectional variance
  survives via the 61-agent population), so their negative finals are real
  reads; the noisy cells (0.57-0.66 harvested) remain the more legible
  evidence and carry the clearest end-state signal.
- The newcomer-incumbent gap treats the init-time spawn as incumbent and the
  early ratio is degenerate; both disclosed above.
- No exit/death in the model bounds every career interpretation to short
  windows (framing section).
- The foreign live sweep observed at launch wrote to its own paths
  (`scripts/figures/`, `data/figures_v1/`); this task's paths are disjoint.

# Machine-readable outputs

All under `data/career_flows_v1/` (gitignored, run-tagged):

| File | Content |
| --- | --- |
| `confirmA_<tag>.csv` x 6 | Sweep A seed-level model-var trajectories, h400: params + seed + Step + 7 reporters (5 T1 metrics + mean_age + mean_prestige) |
| `careerB_<tag>.csv` x 9 | Sweep B seed-level model-var trajectories, h200 |
| `careerB_<tag>_career.csv` x 9 | Sweep B seed-level career measures per step: n_agents, incumbent/newcomer mean prestige, gap, gap ratio, harvest flows, per-capita harvest |
| `confirmA_pooled.csv`, `careerB_pooled.csv` | Per cell x step seed-mean trajectories |
| `confirmA_analysis.csv` | Per-cell oscillation statistics and verdicts |
| `confirmA_t1_contrast.csv` | T1 (h200/20 seeds) vs T2 (h400/40 seeds) on the same cells |
| `careerB_analysis.csv` | Per-cell career-flow scalars |
| `checkpoint.json`, `run_log.txt` | Restart-safe manifest and append-only progress log |
| `manifest.json` | Grid, pins, counts, canary, source hashes, environment, deviations |

Cell tag legend: `confirmA_cur{3,5,7}_van9_{mgstd5,noisypr5}_gen0p1_h400`;
`careerB_gen0p05|0p1|0p2_cur{3,5,7}_mgstd5_h200`. Tags carry integer-scaled
curiosity/vanishing and decimal-point-free entry rates (`0p1` = 0.1).

# Reproducibility

- Environment: python 3.13.5, mesa 3.2.0, numpy 2.5.4, pandas 3.1.0
  (`/opt/hermes/.venv/bin/python`).
- Driver sha256 (first 16): `5cc5b8562dd65438`; consolidator: `123c5c24a4bce4dc`; both in
  `scripts/career_flows/` (ADR-0002: analysis logic in scripts, notebooks stay
  thin).
- Regenerate: `career_flows_driver.py --sweep both` (checkpointed; delete
  `data/career_flows_v1/` for a fresh sweep), then `consolidate.py`, then this
  document via `make_doc.py`.
- Post-sweep source additions (disclosed; no dynamics code touched): the driver
  gained a hard canary launch gate and the consolidator drops a spurious seed
  column from `careerB_pooled.csv` (a groupby artifact, constant 9.5; no doc
  number derived from it). Both were added after the sweep completed and before
  delivery; the sweep data itself was produced by the pre-gate driver with the
  canary verified in-session pre-launch. Hashes above are of the archived
  (current) sources.
- Boundaries honoured: writes confined to `data/career_flows_v1/`,
  `scripts/career_flows/`, `output/reports/career_flows_v1.md`; no git; no code outside
  `scripts/career_flows/` touched; `context/` frozen; `AdmiralsLog/` read-only
  (the session-log entry is deferred to the owner's next records-permitted
  round).

# Revision Log

- **Version 1** (November 2026) --- initial document: oscillation confirmation at
  horizon 400 (sweep A, 6 cells x 40 seeds) and career-flow sweep (sweep B, 9
  cells x 20 seeds); verdict composed mechanically from the declared rule;
  deviations (6-vs-4 cells, mgstd5 pin for sweep B) disclosed in Design.
  Squad-verified before delivery: every table independently re-derived from
  the raw CSVs; five prose defects and one artifact blemish found by that
  check (a misquoted T1 context claim, a wrong oscillation-density figure, an
  age claim contradicted by the doc's own table, a wrong equivalence
  mechanism, a double-rounding artifact) were corrected in this same version
  pre-delivery.
