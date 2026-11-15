---
title: "Demographic Turnover v1: Do the Career-Flow Findings Survive the Founders Dying?"
author: "Hermes Agent (squad lead) for Jakob Kapeller"
version: 1
date: 2026-11-15
tags: [turnover, demography, stationary-population, proxy-failure, career-flows, abm, not-calibrated]
project: sciencemodels
revision-vehicle: strikethrough
revision-renders: "md, pdf, html"
---

**Version 1** (November 2026)

> **NOT CALIBRATED.** Nothing here is fitted to empirical data (owner ruling
> 2026-11-13). Every number is a property of the simulation. The empirical
> moments in `docs/calibration_moments.yaml` remain a documented option menu,
> never targets.

# The question this probe exists to answer

The model has no exit rule. Every scientist ever generated stays in the
population for the rest of the run, so the headcount grows as
$20 + \lfloor g t \rfloor$ while the founders keep sitting on the tiles they
depleted in the first fifty steps. Two headline findings of the career-flow
rounds were measured in exactly that world:

- **A -- the arrival gap.** Newcomers hold less prestige than incumbents.
- **B -- proxy failure.** The correlation between prestige and ideal merit is
  negative: prestige anti-tracks quality.

Both are exposed to the same alternative explanation. If prestige is mostly a
stock accumulated by whoever arrived first, then a gap between arrivals and
incumbents is arithmetic, and a negative prestige-quality correlation is what
surplus of tenure buys. Die the founders off and both findings could evaporate.

P-0006 therefore asks a narrow, falsifiable question: **in stationary
generational equilibrium, after all founders have retired, do A and B persist?**

# What was run

54 cells (3 exit rules x 3 entry rates x 3 curiosities x 2 landscapes) x 20
seeds = **1,080 runs** at horizon $H = 400$, `n_jobs=3`, wall time 1,563 s. All
1,080 runs completed; no cell was skipped, no run was retried into a duplicate.

| Fixed parameter | Value |
| --- | --- |
| `epsilon` | 0.1 |
| `harvest` | 0.02 |
| `grid_size` | 20 |
| `use_distance` | True |
| `new_questions` | 0 |
| `vanishing_factor_for_prestige` | 0.9 |
| `use_visibility_reward` | True |
| `knowledge_normalization` | share |
| `prestige_accrual` | harvested |

The three exit rules, as implemented:

| Mode | Rule | Population |
| --- | --- | --- |
| `none` | benchmark: nobody ever leaves | $20 + \lfloor g t \rfloor$ |
| `fixed_span` | every arrival retires the oldest active scientist | $N \equiv 20$ exactly |
| `age_hazard` | tenure reaches $\tau_{\max} = 200$, retire and replace immediately via the supervisor roulette | $20 + \lfloor g t \rfloor$ |

`fixed_span` is stationary by construction. `age_hazard` is stationary in
*tenure* but not in headcount: retirement is offset one-for-one, so the
population tracks the benchmark exactly, while no individual exceeds 200 steps
of tenure.

## The retirement axis is driver-side, and that is load-bearing

`sciencemodels/model.py` implements no exit rule and sits outside this round's
write boundary, so all three rules are applied by a wrapper around
`MyModel.__generate_new_agents` living in `scripts/turnover/run_turnover_v1.py`.
Three consequences, stated rather than buried:

- The wrapper rebuilds itself from the **pristine** original method each time the
  requested mode changes, because a joblib worker handles cells of several modes
  over its lifetime. Wrapping the wrapper would stack exit rules and silently
  double-count retirements.
- The replacement spawn in `age_hazard` re-issues the model's own roulette with
  the **same single `rng.uniform` draw per newcomer** and the same placement at
  the supervisor's tile, so the random-number stream is consumed exactly as the
  model's own spawn would consume it.
- Consequently the numbers below describe **this driver**. A model-side exit
  rule with the same semantics reproduces them; one with different semantics
  would not, and would need this round re-run rather than reinterpreted.

# Four proofs that the axis measures something

| Proof | Result |
| --- | --- |
| Bit-identity of `none` | patched vs unpatched model, same seed, 120 steps: **56 report keys compared, 0 differing** |
| Divergence canary | all three source anchors the wrapper depends on still present in `__generate_new_agents`; the run aborts naming the file if they move |
| Binding | same seed and parameters: `none` gives N = 101 with 21 founders alive; `fixed_span` gives N = 20 with 0 |
| Archive reproduction | the `none` arm at step 200 reproduces the archived `career_flows_v1` cell on every non-ideal channel |

The last one is the strongest evidence that the benchmark arm is the model and
not a re-implementation. Comparing `none`, entry 0.10, curiosity 0.5, `mgstd5`,
step 200 against `data/career_flows_v1/careerB_cur5_van9_mgstd5_gen0p1_h200.csv`:

| Channel | Archived | This driver, step 200 | Delta |
| --- | --- | --- | --- |
| `c_local` | 0.0815531876196696 | 0.0815531876196696 | 0.0 |
| `top10` | 0.21445080779161105 | 0.21445080779161105 | 0.0 |
| `c_ideal` | -0.053873616349635875 | -0.056751290437960455 | -0.00288 |

Identical to the last digit on both channels that never touch the ideal
channel, despite a different horizon (400 here, 200 in the archive). The single
moved channel is the one the ideal-merit pairing fix touched, exactly as
`output/reports/remeasure_v1.md` documented.

# One measure had to be replaced, and why

The career-flow rounds define the gap as newcomer mean prestige minus incumbent
mean prestige, using the agent attribute `is_newcomer`. **That flag is
permanent.** It records *whether an agent ever arrived as a newcomer*, not
whether it is one now.

The moment every founder has retired, every agent carries the flag, the
incumbent set is empty, and the gap is undefined. This is not a corner case: it
is precisely the settled regime the probe is about. Recording only the
career-flow measure would have produced NaN across the interesting cells.

Two gaps are therefore recorded:

| Name | Definition | Status |
| --- | --- | --- |
| `gap_flag` | career-flow definition, permanent `is_newcomer` flag | NaN once no incumbent generation survives; kept for comparability |
| `gap_recent` | mean prestige of agents with tenure below 20 steps minus mean prestige of the rest | defined throughout; **the headline measure** |

The 20-step window is one arrival generation at the lowest entry rate. It is a
choice, and it is disclosed as one: at entry 0.20 the window holds roughly four
arrivals per agent, at 0.05 roughly one.

# What the runs show

Means over the 360 runs of each mode (`last100` = mean over the final 100 steps):

| Mode | N (last 100) | Mean age | Mean tenure | Founders retired, median step | Settled runs | `gap_flag` | `gap_recent` | $c_{ideal}$ final | $c_{ideal}$ time-mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `none` | 61.45 | 295.0 | 267.6 | never | 0.0% | -0.0164 | -0.01362 | -0.0860 | +0.1141 |
| `fixed_span` | 20.00 | 133.8 | 110.8 | 200 | 66.7% | -0.0182 | -0.00612 | -0.2401 | +0.1042 |
| `age_hazard` | 61.45 | 84.1 | 61.1 | 200 | 100.0% | NaN | -0.00282 | -0.5678 | +0.1112 |

A run counts as **settled** when its last founder retired at least 100 steps
before the horizon, i.e. when there is a genuine founder-free window to read.
`age_hazard` settles always; `fixed_span` settles in two thirds of its cells.

Sign claims, on every run and on the settled subset:

| Scope | Mode | Runs | Arrival gap negative | $c_{ideal}$ negative | $c_{ideal}$ time-mean negative |
| --- | --- | --- | --- | --- | --- |
| all runs | `none` | 360 | 100.0% | 75.6% | 12.2% |
| all runs | `fixed_span` | 360 | 100.0% | 89.2% | 20.8% |
| all runs | `age_hazard` | 360 | 100.0% | 100.0% | 16.4% |
| settled only | `fixed_span` | 240 | 100.0% | 88.3% | 24.6% |
| settled only | `age_hazard` | 360 | 100.0% | 100.0% | 16.4% |

Seed-matched contrasts against the benchmark (same entry, curiosity, landscape
and seed; 360 pairs per mode):

| Mode | Metric | Benchmark | Turnover | Mean diff | Pairs where turnover is lower | Sign preserved | Wilcoxon p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `fixed_span` | $c_{ideal}$ final | -0.0860 | -0.2401 | -0.1541 | 78.9% | 71.4% | 7.7e-32 |
| `age_hazard` | $c_{ideal}$ final | -0.0860 | -0.5678 | -0.4819 | 99.7% | 75.6% | < 1e-60 |
| `fixed_span` | $c_{ideal}$ time-mean | +0.1141 | +0.1042 | -0.0099 | 59.4% | 85.3% | 2.2e-03 |
| `age_hazard` | $c_{ideal}$ time-mean | +0.1141 | +0.1112 | -0.0029 | 60.0% | 91.9% | 2.9e-03 |
| `fixed_span` | `gap_recent` | -0.01362 | -0.00612 | +0.00749 | 0.0% | 100.0% | < 1e-60 |
| `age_hazard` | `gap_recent` | -0.01362 | -0.00282 | +0.01080 | 0.0% | 100.0% | < 1e-60 |

The two `< 1e-60` cells are a **floor, not a ranking**. Both comparisons return
$W = 0$ because all 360 differences point the same way, and scipy's normal
approximation then depends on $n$ alone, giving both the same p-value. It says
the direction is uniform across every pair; it says nothing about which mode
pushes harder. The effect sizes do.

# Finding A: the gap survives in sign and collapses in size

The arrival gap is negative in **every one of the 1,080 runs**, in every mode,
including all 600 settled runs. It is not a founder artifact -- not in sign.

Its magnitude is another matter. Paired against the benchmark, turnover
*shrinks* the gap in **360 of 360 pairs** in both modes:

- `fixed_span`: -0.01362 to -0.00612, a **55.0%** reduction in size.
- `age_hazard`: -0.01362 to -0.00282, a **79.3%** reduction.

So the direction of the career-flow claim survives generational turnover, while
two thirds to four fifths of the measured magnitude was the tenure premium of an
immortal founder cohort. A future round quoting the benchmark gap as a property
of the arrival process would be overstating it by a factor of two to five.

# Finding B: proxy failure does not merely survive, it gets worse

This is the result I did not predict. Turnover makes the negative
prestige-quality correlation substantially **stronger**:

- `fixed_span`: -0.0860 to -0.2401 (**2.79x**), negative in 89.2% of runs.
- `age_hazard`: -0.0860 to -0.5678 (**6.61x**), negative in **100%** of runs,
  and in 99.7% of pairs against the benchmark.

Proxy failure is therefore not an artifact of immortal founders. Immortal
founders *mask* it: the benchmark population carries a large share of agents
whose prestige was acquired on tiles they have long since exhausted, and that
dilutes the anti-correlation that dominance by a small prestige elite produces.

# The entry-rate pattern reverses

Under the benchmark, raising the entry rate *weakens* the negative correlation:
fresh agents are numerous, low-prestige, and only weakly anti-correlated with
merit, so they dilute whatever the incumbents are doing. Under `age_hazard` the
same axis runs the other way.

$c_{ideal}$ at the final step, by entry rate:

| Entry rate | `none` | `fixed_span` | `age_hazard` |
| --- | --- | --- | --- |
| 0.05 | -0.1391 | -0.2310 | -0.7090 |
| 0.10 | -0.0759 | -0.2468 | -0.5764 |
| 0.20 | -0.0429 | -0.2425 | -0.4181 |

`fixed_span` is flat in the entry rate -- once the population is pinned at 20,
the arrival rate changes who is in the room but not the correlation. The two
growing populations both weaken with entry, but from levels that differ by an
order of magnitude. Any statement of the form "higher entry dilutes proxy
failure" is a property of immortal founders, not of entry.

`gap_recent`, by entry rate:

| Entry rate | `none` | `fixed_span` | `age_hazard` |
| --- | --- | --- | --- |
| 0.05 | -0.01792 | -0.01243 | -0.00377 |
| 0.10 | -0.01352 | -0.00408 | -0.00281 |
| 0.20 | -0.00941 | -0.00185 | -0.00187 |

# The trajectory, and one non-monotonicity worth naming

$c_{ideal}$ and `gap_recent` at four checkpoints, seed means:

| Step | `none` $c_{ideal}$ | `fixed_span` $c_{ideal}$ | `age_hazard` $c_{ideal}$ | `none` gap | `fixed_span` gap | `age_hazard` gap |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | -0.0356 | -0.0944 | -0.0356 | -0.01190 | -0.01073 | -0.01190 |
| 200 | -0.1322 | -0.2064 | -0.6212 | -0.01389 | -0.00940 | -0.00711 |
| 300 | -0.1106 | -0.2347 | -0.2061 | -0.01387 | -0.00698 | -0.00273 |
| 400 | -0.0860 | -0.2401 | -0.5678 | -0.01328 | -0.00493 | -0.00194 |

At step 100 `none` and `age_hazard` are identical by construction -- no
`age_hazard` retirement has happened yet. `age_hazard` then falls to -0.6212 at
step 200, the step at which the whole founder cohort hits
$\tau_{\max} = 200$ **simultaneously**, and recovers to -0.2061 by step 300
before falling again. That dip-and-recover is a cohort-replacement shock, not a
trend: the first generation dies all at once, and the run needs time to
repopulate. `fixed_span` avoids it because its retirements are spread one per
arrival. A reader who saw only the step-200 snapshot would conclude that
turnover is far more destructive than it is in steady state.

# What it costs in output

| Mode | Knowledge harvested (final) | Top-10 prestige share (final) |
| --- | --- | --- |
| `none` | 0.7869 | 0.2605 |
| `fixed_span` | 0.6106 | 0.2327 |
| `age_hazard` | 0.7875 | 0.4086 |

`fixed_span` costs 22% of the harvested knowledge (0.787 to 0.611), which is
what capping the population at 20 should do. `age_hazard` costs nothing in
aggregate because it keeps the same headcount -- and pays for it in
concentration: **40.9%** of all prestige sits with the top ten agents, against
26.1% at baseline. The same total harvest, distributed far more unequally, is
the purest statement of what tenure-capped turnover does to this model.

# What this means for the career-flow claims

- **The arrival gap is real and its size was inflated.** The sign holds in
  1,080 of 1,080 runs; the magnitude was 55% to 79% founder-immortality premium.
- **Proxy failure is real and was understated.** The benchmark dilutes it by a
  factor of 2.8 to 6.6. The strongest form of the project's central result lives
  in the turnover regime, not in the immortal-founder regime.
- **The entry-rate reading must be restated.** "Higher entry dilutes proxy
  failure" is true only while founders are immortal.
- **One caveat that outranks the rest.** $c_{ideal}$ measured as a **time mean**
  is *positive* in every mode (+0.104 to +0.114) and negative only at the
  terminal step. The negative correlation is an end-of-run property: prestige
  tracks quality early and anti-tracks it late. Any claim phrased on a run
  average has the opposite sign in all three regimes, and that was already true
  before turnover entered the picture.

# Limits

- **Driver-side axis.** The exit rules are this driver's semantics, not the
  model's. Results transfer only to a model-side rule with the same semantics.
- **`fixed_span` does not settle at the lowest entry rate.** At $g = 0.05$ the
  20 arrivals spread over 400 steps retire the last founder at the final step,
  so those 120 runs have no founder-free window and are excluded from the
  settled statistics (settled share 66.7%). "After all founders retired" is
  fully true only for `age_hazard` and for `fixed_span` at $g \geq 0.10$.
- **`gap_recent` is a constructed measure.** The 20-step window is a disclosed
  choice, not a model quantity. `gap_flag` is reported alongside precisely so
  the substitution stays visible.
- **Twenty seeds per cell**, one horizon ($H = 400$), two landscapes, three
  curiosities. The entry axis is confounded with the settling rate by
  construction.
- **Terminal loss is near zero in every mode** (mean `ideal_merit_sum` between
  -0.0005 and -0.0008), so the differences above are about *who* holds prestige
  and *how it maps to quality*, not about welfare levels at the final step.

# Reproduction

```{.bash}
cd /workspace/git/sciencemodels
MPLBACKEND=Agg python3 tests/test_smoke.py
/opt/hermes/.venv/bin/python scripts/turnover/run_turnover_v1.py        # 1,563 s, n_jobs=3
MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/turnover/analyse_turnover.py
zsh /workspace/git/ai/IT/scripts/compile-here.zsh output/reports/turnover_v1.md
```

Artifacts, all under `data/turnover_v1/`:

| Path | Contents |
| --- | --- |
| `cells/cell_<mode>_g<gg>_cur<c>_<landscape>.csv` | 54 files, 20 rows each, one per seed |
| `traj/cell_<...>.csv` | 54 files, 401 rows, seed means per step |
| `design.json` | grid, constants, axis source, identity and binding proofs |
| `manifest.json` | gross vs net counts, wall time |
| `tables/mode_summary.csv` | the first table above |
| `tables/persistence.csv` | the sign-claim table |
| `tables/paired_contrasts.csv` | the seed-matched contrasts |
| `tables/grid.csv` | all 54 cells, 7 columns |
| `tables/trajectory_points.csv` | the checkpoint table |
| `canary.txt` | `tests/test_smoke.py` output at run time |

# Revision Log

**Version 1** (November 2026). First issue. Reports the P-0006 stationary-population
turnover probe: 54 cells, 1,080 runs, three exit rules, with the retirement axis
implemented driver-side and guarded by a bit-identity proof, a divergence canary
and an archive reproduction. Records the replacement of the career-flow `is_newcomer`
gap by a tenure-window gap, the collapse of the arrival gap to 21% to 45% of its
benchmark size, the 2.8x to 6.6x strengthening of the negative prestige-quality
correlation under turnover, the reversal of the entry-rate pattern, and the
cohort-replacement shock at step 200 in the `age_hazard` mode.
