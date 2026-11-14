---
title: "Career flows on the noisy landscape: which findings are landscape-dependent"
author: "Hermes Agent (squad lead) for Jakob Kapeller"
version: 1
date: 2026-11-13
tags: [career-flows, entry, proxy-failure, landscape-legibility, sensitivity, not-calibrated]
project: sciencemodels
revision-vehicle: strikethrough
revision-renders: "md, pdf, html"
---

**Version 1** (November 2026)

**Verdict: four of the five headline career-flow findings hold on the noisy
landscape, one qualifies, none fails.** Population arithmetic, the per-capita
harvest ordering in entry rate, the concentration decay and the
decoupling-by-entry-rate ordering all reproduce. The finding that moves is the
*size* of the newcomer-incumbent prestige gap: the sign and persistence are
robust (1359/1359 late cell-steps negative) but newcomers hold about half of
incumbent prestige on the noisy landscape instead of about a third, and the
absolute gap shrinks to 36% of its clean-arm value. Separately, the level of
decoupling is stronger on the noisy landscape at every entry rate (the ideal
correlation is 0.087-0.109 more negative), while its ordering in $g$ is
preserved --- the landscape moves levels, not orderings. One caveat governs the
ideal-channel numbers: the pairing audit landed mid-sweep, so the delta against
the current code is measured rather than assumed (see the code-movement
section), and it leaves the orderings intact while shifting levels by about
0.016.

> **NOT CALIBRATED.** Model-internal exploration only (owner ruling 2026-11-13).
> No empirical moment was used as a target, and nothing here is fitted.

# Design

The brief is the declared extension of `output/reports/career_flows_v1.md` (Design,
disclosed deviation (b): "a noisy rerun is the natural extension"). The clean
sweep B is rerun with the noisy landscape bundle and everything else identical.

| Arm | Cells | Grid | Seeds | Horizon | Runs |
| --- | --- | --- | --- | --- | --- |
| noisy (this round) | 9 | $g \in \{0.05, 0.1, 0.2\} \times c \in \{0.3, 0.5, 0.7\}$, `noisyGaussian` {prop_random 5, std 5, std_curiosity 0.1} | 20 | 200 | 180 |
| control (comparability) | 1 | $c = 0.5$, $g = 0.1$, `multipleGaussians` {std 5, 2 gaussians} | 20 | 200 | 20 |

Pins, identical to sweep B of the clean round: $n=20$ founders, 20x20 torus,
$\epsilon=0.1$, $h=0.02$, `use_distance=True`, `new_questions=0`,
`use_visibility_reward=True`, `knowledge_normalization='share'`,
`prestige_accrual='harvested'`, vanishing factor 0.9, identity curiosity
function. Seed semantics unchanged: one `agent_seed` per run (0-19), which
`MyModel` also uses as the landscape seed, so landscape and behaviour seeds
coincide exactly as in the clean round.

Two design decisions deserve naming:

- **A control arm was added, not requested.** The brief compares against the
  clean original, and a comparison is only as good as the bridge. The control
  cell re-runs one clean cell through this round's driver and diffs it against
  `data/career_flows_v1/careerB_cur5_van9_mgstd5_gen0p1_h200.csv`; it is the
  proof that the only difference between the arms is the landscape (section
  below). Twenty extra runs, and no clean-arm number is taken on trust.
- **The measurement definitions are copied verbatim** from
  `scripts/career_flows/career_flows_driver.py` (`run_one`,
  `career_measures_from_agent_vars`) so the two arms are comparable by
  construction. The duplication is reported as a need; ADR-0002 wants one home
  per measure.

Declared verdict rules, fixed before the noisy numbers were read (tolerances
included):

| Test | Rule | Tolerance |
| --- | --- | --- |
| F1 population formula | `pop_final == 21 + floor(g*H)` in every cell | exact |
| F2 newcomer gap | pooled `prestige_gap < 0` at every step >= 50 in all 9 cells, and mean late gap ratio < 0 | 0.10 on the mean late gap ratio vs the clean arm |
| F3 per-capita harvest | `percap_harvest_final` strictly decreasing in $g$ within each curiosity cell (3/3) | 25% on the $g=0.05 / g=0.2$ ratio vs the clean arm |
| F4 concentration decay | `top10_delta < 0` in all 9 cells, and the decay weakest at $g=0.2$ in >= 2 of 3 curiosity cells | 2 of 3 high-$g$ cells |
| F5 decoupling ordering | mean `ideal_corr_final` increasing in $g$ | 0.10 on the spread over $g$ vs the clean arm |

A finding **holds** when the sign/order test passes and the tolerance is met,
**qualifies** when the sign/order passes and the tolerance is missed, and
**fails** when the sign/order test itself breaks.

# Canary and constraints

Canary before the first edit (exit 0):

```{.text}
canary run OK: 40 report keys, harvested_10_step=37.5, harvested_25_step=37.5
same-seed rerun bit-identical OK
newcomer run OK: 6 scientists after 20 steps
4-arg Gaussian OK: np.float64(0.00550470835249002) (float64)
/tmp/smoke_append_7hqadn4v/append_header_check.csv w
now processing param set  1 out of 1
/tmp/smoke_append_7hqadn4v/append_header_check.csv a
now processing param set  1 out of 1
append-mode header check OK: 3 lines = 1 header + 2 data rows
ALL SMOKE TESTS PASSED
```

Canary after the last edit (exit 0) --- **the canary itself changed during this
round**, which is a finding in its own right: a sibling round landed an
ideal-merit pairing audit, an entry-prestige-stock alternative and an upstream
binding probe, taking the report keys from 40 to 44 and adding four checks.

```{.text}
canary run OK: 44 report keys, harvested_10_step=37.5, harvested_25_step=37.5
same-seed rerun bit-identical OK
newcomer run OK: 6 scientists after 20 steps
4-arg Gaussian OK: np.float64(0.00550470835249002) (float64)
/tmp/smoke_append_xak82qd5/append_header_check.csv w
now processing param set  1 out of 1
/tmp/smoke_append_xak82qd5/append_header_check.csv a
now processing param set  1 out of 1
append-mode header check OK: 3 lines = 1 header + 2 data rows
ideal-merit pairing audit OK: A=0.0, C=0.0, B=-0.04999999999999999 (hand-computed 0.0, 0.0, -0.05); cum after 2 calls: A=0.0, B=-0.09999999999999998
seed columns OK: agent_seed=1, landscape_seed=1
WARNING: agent_generation_function '<lambda>' returned IDENTICAL draws for mu=0.1 and mu=0.9 on the same rng - the swept axis (initial_curiosity) is INERT.
binding probe OK: constant generator flagged inert, identity binds
entry_prestige_stock OK: validated, both alternate stocks run
ALL SMOKE TESTS PASSED
```

Both gates passed, but they are not the same gate, and the sweep below ran
between them. The next section measures what that means for the numbers.

Constraints honoured: interpreter `/opt/hermes/.venv/bin/python`; no installs;
`n_jobs=3` in the driver; a process snapshot and the design record written into
the run folder before the launch (`process_snapshot.txt`, `design.json`); one
append-only log line per completed cell (`run_log.txt`); atomic checkpoint and
atomic per-cell CSVs gated on the row count; no git commands; writes confined
to `data/career_flows_noisy_v1/`, `scripts/career_flows_noisy/` and this
document; `context/` and `AdmiralsLog/` untouched.

Gross vs net: 200 runs gross, 200 net, 10 cells gross, 10 net, 0 failed seeds,
0 crashed cells, 40,200 rows written (200 seeds x 201 steps), total driver wall
time 1188.0 s for the run window 2026-11-13T23:54:08 to 2026-11-14T00:13:52 UTC.
Cell wall time runs from 23.4 s at $g=0.05$ to 240.8 s at $g=0.2$ (244.9 s for
the control cell); the spread is shared-tree load --- other sessions ran a
collective-behaviour demo, a shim verification and an entry-prestige-stock sweep
in this window --- not a model property.

The critical-path probe (`scripts/career_flows_noisy/probe.py`) ran first, two
seeds per arm: schema present in a CSV written to disk and read back, timing
recorded (~1 s per 200-step run at $g=0.1$), and the binding proof --- at a
fixed seed the two landscapes differ on every headline outcome (ideal
correlation 0.308, local 0.063, top-10% share 0.024, harvested share 0.355,
prestige gap 0.012), so the swept axis is not inert.

# Control arm: the bridge to the clean round

The control cell reproduces the clean original bit-exactly:

| Comparison | Rows | Max absolute difference |
| --- | --- | --- |
| 7 model reporters (ideal/local corr, top-10% share, explored share, harvested share, mean age, mean prestige) | 4020 vs 4020 | 0.0 |
| 6 career measures (n_agents, incumbent/newcomer prestige, gap, gap ratio, per-capita harvest) | 4020 vs 4020 | 0.0 |

Every one of the thirteen comparisons is exactly zero
(`control_equivalence.csv`). The model code has not moved since the clean
round, so **every clean-vs-noisy difference below is the landscape**, not a
code change and not the ideal-merit pairing question the brief anticipated: no
delta needs to be attributed to a moved ideal-merit implementation in this
round.

# Code movement during the round

The sweep ran 2026-11-13T23:54:08 to 2026-11-14T00:13:52 UTC. In that window a
sibling round landed an ideal-merit pairing audit, an entry-prestige-stock
parameter and an upstream binding probe, so the tree that produced these numbers
is not the tree that exists now:

| File | sha256 (first 16) before the sweep | after | mtime after |
| --- | --- | --- | --- |
| `sciencemodels/metrics.py` | 270cff27592c34d2 | d752f08b4c9e3653 | 2026-11-14 00:17 |
| `sciencemodels/model.py` | 927afdbf23732afa | 3c3c87f7e6869a82 | 2026-11-14 00:26 |
| `sciencemodels/io.py` | 4cfa8edbfa25f8a6 | 0aa0bdb4758861c1 | 2026-11-14 00:17 |
| `tests/test_smoke.py` | 40-key canary | 44-key canary plus four checks | 2026-11-14 00:29 |

The change is the pairing fix in `update_ideal_merit` (the k-th largest actual
harvest is now matched against the k-th largest greedy-ideal harvest instead of
the static head of the ideal list) plus a benchmark that follows the live
population rather than the founder count. Both touch the ideal channel only.
Rather than force the old values, the delta was measured on re-runs of the same
cells and seeds under the current code (`audit_code_delta.py`):

| Reporter | Control cell, 5 seeds, 1005 rows | Noisy cell, 3 seeds, 603 rows |
| --- | --- | --- |
| `Corr_prestige_ideal_merit` | max abs delta 0.7297 (all-step mean -0.0036) | max abs delta 0.5480 (all-step mean -0.0001) |
| `Corr_prestige_local_merit` | 0.0 | 0.0 |
| `top10pct_prestige_share`, `explored_percentage`, `percentage_knowledge_harvested`, `mean_age`, `mean_prestige` | 0.0 | 0.0 |
| `n_agents`, incumbent/newcomer prestige, `prestige_gap`, `prestige_gap_ratio`, `percap_harvest_cum` | 0.0 | 0.0 |

**Only the ideal channel moved, and only along the path, not in level.** At the
step F5 actually uses, the paired difference is small and sign-unstable: four
paired runs (c=0.5, g=0.1, seeds 0-1 per arm) give step-200 ideal deltas of
-0.0239, +0.0014, -0.0149 and +0.0227 (mean absolute 0.0157, mean signed -0.0113
on the clean arm and +0.0039 on the noisy one), while the local correlation is
identical to the last digit in all four.

Two consequences, stated rather than glossed:

- **F1-F4 and every local-channel statement are unaffected by construction** ---
  their inputs are bit-identical under both codes. The verdicts for population,
  the prestige gap, per-capita harvest and concentration stand as measured.
- **F5's ordering survives a magnitude argument, not a re-run.** The
  implementation change perturbs the step-200 ideal correlation by about 0.016
  with mixed sign, while the $g$-contrast F5 rests on is 0.06-0.09 on the clean
  arm and 0.06-0.12 on the noisy one; an unsystematic perturbation of a fifth to
  a quarter of the contrast cannot produce the ordering. The *levels* quoted for
  the ideal channel are pre-audit numbers and should be re-derived before
  publication.
- **A powered re-measurement was attempted and is reported as inconclusive.**
  `remeasure_f5.py` ran 2 seeds x 3 entry rates x 2 arms on the current code; the
  ordering test fails on the noisy arm (-0.135 / -0.224 / -0.188) *and on the
  clean arm* (-0.051 / -0.113 / +0.103), whose 20-seed ordering is unambiguous.
  A test that fails on the arm with the known answer is measuring seed noise
  (per-seed step-200 ideal finals span -0.30 to +0.24), not the ordering. It is
  recorded as a power calibration, not as evidence against F5; the powered
  version (20 seeds x 3 rates x 2 arms = 120 runs) is filed as a need.

# Results: the findings against the landscape

One row per headline finding, with the clean arm as reference. Numbers are
means over the three curiosity cells and 20 seeds; "late" means steps 50-200.

| Finding | Clean arm (`multipleGaussians`, reference) | Noisy arm (`noisyGaussian` pr5, this round) | Verdict |
| --- | --- | --- | --- |
| F1 population formula $21 + \lfloor g t \rfloor$ | 31 / 41 / 61 scientists at $g=0.05/0.1/0.2$ | 31 / 41 / 61, exact in 9/9 cells | holds |
| F2 newcomer-incumbent gap persistent and negative | late ratio -0.656 (newcomers hold ~a third of incumbent prestige); gap -0.0197 | late ratio -0.516; 1359/1359 late cell-steps negative; gap -0.0070 (36% of clean) | qualifies |
| F3 per-capita harvest falls with entry rate | ratio $g=.05 / g=.2$ = 1.694; monotone in 3/3 | ratio 1.468; monotone in 3/3 | holds |
| F4 concentration decays within entry cells, weakest at high $g$ | top-10% delta < 0 in 9/9; weakest at $g=0.2$ in 3/3; delta at $g=0.2$ = -0.0195 | delta < 0 in 9/9; weakest at $g=0.2$ in 3/3; delta at $g=0.2$ = -0.0761 | holds |
| F5 decoupling strongest at slow entry | ideal corr final -0.133 / -0.060 / -0.025; spread 0.1081 | ideal corr final -0.227 / -0.168 / -0.112; spread 0.1156 | holds (ordering robust; ideal levels are pre-audit, see code movement) |

Per-cell results, noisy arm:

| Cell | pop final | top-10% @10 | top-10% delta | gap final | gap ratio (late) | per-capita harvest | ideal corr final | local corr final |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c3, g=.05 | 31 | 0.3341 | -0.1046 | -0.00751 | -0.577 | 0.00926 | -0.3060 | -0.0032 |
| c5, g=.05 | 31 | 0.3338 | -0.0959 | -0.00767 | -0.514 | 0.01061 | -0.2330 | 0.0362 |
| c7, g=.05 | 31 | 0.3364 | -0.0898 | -0.00751 | -0.505 | 0.01071 | -0.1429 | 0.1231 |
| c3, g=.1 | 41 | 0.3341 | -0.0950 | -0.00684 | -0.529 | 0.00780 | -0.2064 | 0.0223 |
| c5, g=.1 | 41 | 0.3338 | -0.0910 | -0.00700 | -0.482 | 0.00898 | -0.1110 | 0.0883 |
| c7, g=.1 | 41 | 0.3364 | -0.0860 | -0.00676 | -0.463 | 0.00913 | -0.1874 | 0.0472 |
| c3, g=.2 | 61 | 0.3270 | -0.0757 | -0.00655 | -0.557 | 0.00626 | -0.0885 | 0.0406 |
| c5, g=.2 | 61 | 0.3281 | -0.0777 | -0.00682 | -0.514 | 0.00723 | -0.0902 | -0.0295 |
| c7, g=.2 | 61 | 0.3305 | -0.0749 | -0.00666 | -0.501 | 0.00734 | -0.1563 | -0.0158 |

Arm-level comparison by entry rate (means over the three curiosity cells):

| Quantity | clean $g=.05$ | clean $g=.1$ | clean $g=.2$ | noisy $g=.05$ | noisy $g=.1$ | noisy $g=.2$ |
| --- | --- | --- | --- | --- | --- | --- |
| top-10% share at step 10 | 0.2753 | 0.2753 | 0.2635 | 0.3348 | 0.3348 | 0.3285 |
| top-10% share at step 200 | 0.1940 | 0.2127 | 0.2441 | 0.2380 | 0.2440 | 0.2525 |
| top-10% delta | -0.0813 | -0.0626 | -0.0195 | -0.0968 | -0.0907 | -0.0761 |
| prestige gap at step 200 | -0.0213 | -0.0198 | -0.0182 | -0.0076 | -0.0069 | -0.0067 |
| gap ratio (late) | -0.672 | -0.649 | -0.647 | -0.532 | -0.491 | -0.524 |
| incumbent mean prestige | 0.0296 | 0.0280 | 0.0254 | 0.0126 | 0.0161 | 0.0113 |
| per-capita harvest (cumulative) | 0.0227 | 0.0184 | 0.0134 | 0.0102 | 0.0086 | 0.0069 |
| ideal corr final | -0.1331 | -0.0596 | -0.0250 | -0.2273 | -0.1683 | -0.1117 |
| local corr final | 0.0803 | 0.0836 | 0.0562 | 0.0520 | 0.0526 | -0.0016 |
| harvested share final | 0.705 | 0.753 | 0.819 | 0.316 | 0.354 | 0.424 |
| mean age final | 196.3 | 178.4 | 160.3 | 195.7 | 178.0 | 160.0 |

Figure: `data/career_flows_noisy_v1/figures/career_noisy_vs_clean.png` ---
panel A the gap ratio (steps 50-200), panel B per-capita cumulative harvest,
panel C the top-10% share, panel D the final ideal and local correlations by
entry rate; solid = noisy, dashed = clean. The rendered PNG was inspected after
generation; one defect was found and fixed before delivery (panel D initially
averaged over all steps while its title claimed the final step, which showed
all-step means of 0.06-0.45 instead of the negative finals; the panel now uses
step 200 and reads -0.23 to +0.08).

# What is landscape-dependent

- **Levels are landscape-dependent; orderings are not.** Every quantity that
  measures *how much* prestige or knowledge exists moves by roughly a factor
  two on the noisy landscape (incumbent prestige 0.011-0.016 vs 0.025-0.030,
  per-capita harvest 0.007-0.010 vs 0.013-0.023, harvested share 0.32-0.42 vs
  0.71-0.82), while every quantity that measures *how the population orders
  itself* in entry rate keeps its direction. The noisy landscape has a uniform
  floor, so less total knowledge is harvestable and every prestige level is
  compressed; the demographic structure is untouched.
- **The individual-level proxy-failure statement is the one that weakens.** The
  clean round's sentence was "newcomers hold roughly a third of incumbent
  prestige at every entry rate". On the noisy landscape they hold roughly half
  (late ratio -0.52 to -0.46 vs -0.68 to -0.64), and in absolute terms the gap
  is a third smaller than the ratio suggests (36% of the clean gap) because the
  incumbent level itself halved. The mechanism reading survives --- newcomers
  start at zero prestige, spawn on a depleted supervisor tile, and never close
  the gap --- but its quantitative summary is landscape-conditional and must be
  quoted per landscape.
- **Decoupling is stronger where merit is harder to see, as the theory
  predicts, but the entry-rate contrast is invariant.** The ideal correlation
  is 0.087-0.109 more negative on the noisy arm at every $g$, and its spread
  over $g$ is 0.1156 against 0.1081 on the clean arm (difference 0.0075, far
  inside the 0.10 tolerance). This is the cleanest form of the answer: the
  noisy landscape adds a level shift to the decoupling, not a new ordering.
- **The concentration finding gains a qualification the verdict rule does not
  catch.** F4 passes as declared (delta < 0 in 9/9, weakest at $g=0.2$ in
  3/3), but the clean arm's near-flat high-$g$ case (delta -0.0195) does not
  generalise: on the noisy arm the decay at $g=0.2$ is -0.0761,
  nearly as steep as at $g=0.05$ (-0.0968). The ordering "weakest at high $g$"
  holds in both arms; the *flatness* at high $g$ does not generalise. Read the
  clean round's "the decay is weakest at high $g$, where dilution prevents the
  spike from forming" as a statement about the clean landscape only.
- **The benchmark-free channel does not confirm the ordering as cleanly.**
  T2's Q2 warns that the ideal-merit benchmark is founder-sized (20 slots) while
  31-61 agents harvest under entry, so part of the ideal-channel degradation is
  benchmark strain. On the local (benchmark-free) channel the noisy arm reads
  0.0520 / 0.0526 / -0.0016 across $g$: flat between $g=0.05$ and $g=0.1$, then
  negative at $g=0.2$. So "decoupling strongest at slow entry" is robust on the
  ideal channel and only partly visible on the local one --- which is what
  benchmark strain would look like, and is worth knowing before the ordering is
  quoted as a mechanism result.

# Model quirks carried and observed

- **Q1 (carried, replicated):** the init-time accumulator spawn makes step-0
  population 21, not 20; final populations are $21 + \lfloor g H \rfloor$ =
  31/41/61, exactly as on the clean arm. F1 is a property of the accumulator,
  not of the landscape, and its exactness in both arms is the point.
- **Q2 (carried, load-bearing for these numbers, since fixed in the tree):** the
  founder-sized ideal benchmark strained as entry grew the population; the local
  channel is the benchmark-free companion and is reported alongside every ideal
  number. The current tree makes the benchmark follow the live population, so
  this caveat applies to this round's ideal-channel levels, not to the code as
  it now stands (see the code-movement section).
- **Q6 (carried, not triggered):** the correlation reporter's zero guard returns
  exactly 0 for degenerate stocks, which would read as "no decoupling". No cell
  here is degenerate --- the noisy arm harvests only 0.32-0.42 of the landscape,
  so variance survives comfortably.
- **New observation, disclosed not fixed:** the per-seed gap turns briefly
  positive in individual runs at $g=0.1$ (5-8 of 20 seeds per cell, max +0.00082),
  at $g=0.05$ (2-3 of 20, max +0.00065) and at $g=0.2$ (2 of 20, max +0.00031),
  always at small magnitude;
  the pooled trajectory is negative at every late step in all nine cells. This
  is the noisy counterpart of the clean round's per-seed nuance, and it is
  larger there (up to 8/20 seeds) than on the clean arm (6-7/20 at $g=0.2$).

# Honest caveats

- Not calibrated; no empirical moment was used as a target anywhere in this
  round.
- 20 seeds per cell and a single horizon (200 steps). The verdict rules were
  declared before the run, but the tolerances (0.10 gap ratio, 25% per-capita
  ratio, 0.10 spread) are round numbers chosen to make "materially different"
  testable, not sampling-error bounds.
- One noisy bundle only: `noisyGaussian` with prop_random 5 changes two things
  at once relative to the clean bundle (a uniform noise floor and a different
  peak structure). The round therefore speaks to *this* noisy landscape, not to
  "legibility" as a continuous axis; the sensitivity map's legibility leg is
  the place where the axis itself is swept.
- No exit channel and no lifecycle in the model, so the career framing stays
  bounded to short windows exactly as the clean round documented; mean age is a
  recorded-but-dead lever in both arms.
- The gap ratio is degenerate while incumbent prestige is ~0 and is reported
  from step 50 onward, as in the clean round; the init-time extra spawn is
  classified as incumbent.
- The figure's panels pool over curiosity cells; per-cell values are in the
  tables above, and the arm-level means in the comparison table are what the
  panels show.

# Needs (reported, not fixed)

- **One home for the measurement definitions.** `run_one` and
  `career_measures_from_agent_vars` are now duplicated in
  `scripts/career_flows/career_flows_driver.py` and
  `scripts/career_flows_noisy/driver.py` (verbatim, deliberately). ADR-0002
  wants one home; the fix is to lift both into the package or a shared module
  and let each driver import them.
- **The clean round's pooled artifact carries the ideal channel only.** The
  document states that "the local channel ... is recorded alongside", but
  `data/career_flows_v1/careerB_pooled.csv` has no local-correlation column;
  this round recovered it from the raw per-cell CSVs (read-only). Either add it
  to that consolidator or reword the sentence in `output/reports/career_flows_v1.md` ---
  both are outside this round's write boundary.
- **The clean arm's analysis table lacks the local-channel scalar** for the same
  reason; the clean local numbers quoted above were re-derived from the raw
  CSVs, and a later round should carry them in the archive rather than in a
  session's recomputation.
- **A powered F5 re-run on the current code.** The 2-seed re-measurement cannot
  resolve an ordering (it fails on the control arm too); 20 seeds x 3 entry
  rates x 2 landscapes = 120 runs would settle the ideal-channel levels and the
  ordering under the post-audit model, and should be run before any ideal-channel
  number from this round is published.

# Machine-readable outputs

All under `data/career_flows_noisy_v1/` (gitignored, run-tagged):

| File | Content |
| --- | --- |
| `noisyB_cur{3,5,7}_van9_noisypr5_gen0p{05,1,2}_h200.csv` x 9 | seed-level model-variable trajectories, h200 |
| `..._career.csv` x 9 | seed-level career measures per step (population, incumbent/newcomer prestige, gap, gap ratio, harvest flows, per-capita harvest) |
| `control_cur5_van9_mgstd5_gen0p1_h200.csv` and `_career.csv` | the comparability control cell |
| `noisyB_pooled.csv` | per-cell x step seed-mean trajectories, noisy arm only |
| `pooled_all_cells.csv` | the same including the control cell |
| `noisyB_analysis.csv` | per-cell career-flow scalars, noisy arm |
| `finding_tests.csv` | the five tests with clean value, noisy value, tolerance and verdict |
| `control_equivalence.csv` | bit-exactness of the control cell against the clean original |
| `code_delta.csv`, `code_delta.json` | per-reporter delta between the sweep's code and the tree's current code |
| `f5_remeasure.csv`, `f5_remeasure.json` | F5 re-measured on fresh runs under the current code |
| `figures/career_noisy_vs_clean.png` | the four-panel comparison figure |
| `design.json`, `checkpoint.json`, `run_log.txt`, `process_snapshot.txt`, `manifest.json` | design record, restart-safe manifest, append-only log, pre-launch process snapshot, counts and source hashes |

# Changes (file:line)

New files only; no existing file was edited.

| Path | Lines | Purpose |
| --- | --- | --- |
| `scripts/career_flows_noisy/driver.py` | 1-333 | grid, checkpointed execution, canary gate, design record |
| `scripts/career_flows_noisy/probe.py` | 1-101 | schema, binding and timing proofs before the launch |
| `scripts/career_flows_noisy/consolidate.py` | 1-277 | per-cell scalars, declared tests, control equivalence |
| `scripts/career_flows_noisy/make_figures.py` | 1-133 | four-panel figure |
| `scripts/career_flows_noisy/audit_code_delta.py` | 1-124 | per-reporter delta against the current tree |
| `scripts/career_flows_noisy/remeasure_f5.py` | 1-95 | F5 re-measured on the current code |
| `output/reports/career_flows_noisy_v1.md` | 1-end | this document |

# Reproducibility

- Environment: python 3.13.5, mesa 3.2.0, numpy 2.5.4, pandas 3.1.0
  (`/opt/hermes/.venv/bin/python`).
- Driver sha256 (first 16) and consolidator hash are in
  `data/career_flows_noisy_v1/manifest.json`.
- Regenerate: `driver.py --sweeps both` (checkpointed; delete the run folder for
  a fresh sweep), then `consolidate.py`, then `make_figures.py`.
- The control arm is part of the driver, so a regeneration re-proves
  comparability with the clean round instead of assuming it.

# Revision Log

- **Version 1** (November 2026) --- noisy rerun of career-flow sweep B
  (9 cells x 20 seeds) plus a bit-exact control cell; five headline findings
  tested against declared rules with fixed tolerances: four hold, one (the
  newcomer gap magnitude) qualifies, none fails. One figure defect found by the
  post-generation inspection and fixed before delivery; the local-channel
  qualification and the duplicated measurement definitions are reported as
  needs, not fixed. Same-version addition: the code-movement section, after a
  sibling round replaced the ideal-merit pairing mid-sweep --- the delta is
  measured per reporter (only the ideal channel moves, ~0.016 at step 200 with
  mixed sign), the 2-seed F5 re-measurement is disclosed as underpowered, and a
  powered re-run is filed as a need.