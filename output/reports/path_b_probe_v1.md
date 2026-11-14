---
title: "Path B probe v1: the dyadic collaborator-choice layer and the Gate 2 sweep"
version: 1
date: 2026-11-14
revision-vehicle: strikethrough
revision-renders: "md, pdf, html"
tags: [path-b, dyadic, collaborator-choice, gate-2, team-size, repeated-collab, proxy-failure, abm, not-calibrated]
project: sciencemodels
---

**Version 1** (November 2026)

**NOT CALIBRATED.** Model-internal exploration only (owner ruling,
2026-11-13). The Klann moments (team size mean 2.65 / median 2,
repeated-collaboration share 0.34, O/E elite homophily 3.1/3.0/2.3, Markov C
0.3995/0.0619) are *diagnostic reference values* for the Gate 2 evaluation,
never fitting targets. Every number below was computed by
`scripts/path_b/run_path_b_sweep.py`, `scripts/path_b/run_baseline_off.py`
and `scripts/path_b/analyse_path_b.py` from the run-tagged CSVs under
`data/path_b_v1/`, and is reproducible with the commands quoted at the end.

# Purpose and scope

`docs/way_forward.md` Path B puts the paper's title noun --- the collaborator
--- into the model as an actor: agents form and dissolve dyadic ties, choosing
partners by the **proxy** (normalized visibility stock) versus the **goal**
(topical closeness $\times$ observable merit), with one mixing parameter
$\lambda$. This round implements the ruled W3-C menu (owner rulings Q1-Q8 in
the Path B brief: both formulations, grid-distance closeness, observable
local merit, visibility stock, newcomer proposals, undirected links plus an
explicit team object, alongside the supervisor roulette, structural or equal
credit) and runs the **Gate 2 sweep**: 40 cells, 800 seeded runs, asking
whether the layer materially improves the team and closure moments.

The bracket is defined operationally, as `way_forward.md` demands: with
`dyadic_mode="off"` (the default) there is **no link formation, no team
production and no credit sharing** --- the off-mode dynamics are bit-identical
to the pre-Path-B model (Check 10 of `tests/test_smoke.py` proves this at the
bit level, and the closing canary below carries the anchor
`mean_prestige = 0.025399760810316363`). Everything between the two modes is
the dyadic layer's own effect.

# Design

## Grid

$\lambda \in \{0, 0.25, 0.5, 0.75, 1.0\}$ $\times$ formulation
$\in \{\text{logit}, \text{threshold}\}$ (Q1: both, primary + robustness
twin) $\times$ credit $\in \{\text{structural}, \text{equal\_split}\}$
(Q8) $\times$ landscape $\in \{\text{mgstd5}, \text{noisypr5}\}$ =
**40 cells, 20 seeds, 800 runs**, `step_limit=200`, `agent_generation_time=0.1`,
`initial_curiosity=0.5`. Fixed pins (manifest `design_pins`):
`n_agents=20`, `grid_size=20`, `epsilon=0.1`, `harvest=0.02`,
`use_distance=True`, `new_questions=0`, `use_visibility_reward=True`,
`knowledge_normalization="share"`, `prestige_accrual="harvested"`,
`vanishing_factor_for_prestige=0.9`, identity `agent_generation_function`
(curiosity pinned at 0.5), `entry_prestige_stock="prestige"`,
`dyadic_proposal_regime="newcomer"`, `closeness_mode="grid"`,
`dyadic_visibility_stock="prestige_visibility"`. Seed semantics: `agent_seed`
0-19 drives agent and landscape RNG jointly (the model injects the seed into
`generation_params`).

## Declared micro-pins (the brief's rulings leave these mechanics to the implementer)

| # | Pin | Definition |
| --- | --- | --- |
| D1 | Acceptance | bilateral relative-bar clearance: link $(i,j)$ forms or upgrades iff $U_{ij} \geq bar(i)$ **and** $U_{ji} \geq bar(j)$ (the sketch's "link update identical in A and B") |
| D2 | $bar(k)$ | best current link's score if $k$ is linked; else the mean candidate score (the knob-free relative solo bar) |
| D3 | Dissolution | each step, $(i,j)$ breaks if $U_{ij} < bar(i)$ or $U_{ji} < bar(j)$; strict, ties persist |
| D4 | Candidates | all other scientists; one proposal per proposer per step, random order |
| D5 | Two layers | *choice layer* `model.links` (declared undirected ties; `n_active_links`); *production layer* = per-step co-harvest teams (agents sharing a tile, size $\geq 2$; `mean_team_size`, `repeated_collab_share`, `dyadic_assortativity_prestige`) |
| D6 | equal_split mechanics | farm-time accrual buffered per agent; per tile, linked co-located agents pool and split equally; credited in `end_loop_update` **before** the dyadic scores. Accrual is therefore deferred to end-of-step under this rule only --- a mechanics change inherent to pooling, disclosed |
| D7 | Newcomer regime | permanent `is_newcomer` flag (born via entry); newcomers propose from their birth step; with entry off the newcomer regime proposes nothing (proven inert in Check 10) |
| D8 | Reporter naming | the four brief-named reporters are lowercase (last-value semantics per the `io.py` convention); the driver adds time-means (`_tm`) from the datacollector, plus two choice-layer reporters (`link_repeated_share`, `link_assortativity_prestige`) so the choice layer's own response is measurable |
| D9 | Horizon | `step_limit=200` (the house entry-round standard; the brief pins $g$, $c$, seeds and cells but not the horizon) |

Newborns land on their supervisor's tile and count in that step's team
(D5's team object is positional); their birth-step `ideal_Merit` stays at the
`Scientist.__init__` preset 0 (pre-existing, disclosed in N1).

## The one ruled feedback channel

Under `credit_sharing="structural"` links change nothing in the production
layer --- they are declarations. The single ruled channel from choice to
production is `equal_split`: split prestige changes the prestige stocks,
which feed the visibility reward's prestige-weighted distance term, which
moves agents, which changes co-harvest teams. The sweep's credit axis is
therefore also the test of whether the choice layer can reach production at
all within the ruled menu.

# Measurement

| Quantity | Definition | Layer |
| --- | --- | --- |
| `mean_team_size` | mean size of co-harvest groups $\geq 2$ per step (reporter: last value; driver `_tm`: time-mean) | production |
| `repeated_collab_share` | per step, active co-harvest pairs seen active in any earlier step / active pairs (0.0 when none) | production |
| `dyadic_assortativity_prestige` | Pearson corr of endpoint prestige over active co-harvest pairs (guarded, 0.0 when degenerate) | production |
| `n_active_links` | `len(model.links)` | choice |
| `link_repeated_share`, `link_assortativity_prestige` | the two repetition/assortativity statistics over the declared link set | choice |
| wedge pair | `Mean_of_Corr_prestige_local_merit` minus `Mean_of_Corr_prestige_ideal_merit` (time-mean), plus `ideal_loss_cum` $=-\sum_i \mathrm{ideal\_merit\_cum}_i$ | welfare |

Gate 2 rule (declared in `analyse_path_b.py` before results were read):
contrast $\lambda=1.0$ vs $\lambda=0.0$, seed-matched, 20 pairs, per
formulation $\times$ credit $\times$ landscape (8 contrasts); a production
moment *improves* when the paired distance $|\text{sim} - \text{target}|$
shrinks with a bootstrap 95% CI over seed-pairs excluding 0; **PASS** needs
$\geq 6$ of 8 contrasts improving for team size *and* $\geq 6$ of 8 for
repeated share; **PARTIAL** means the choice layer responds in $\geq 6$ of 8
but production does not; else **FAIL**.

# Results

## The choice layer binds

$\lambda$ moves the declared ties in both formulations (seed-matched paired
$\Delta$ = mean$(\lambda{=}1) -$ mean$(\lambda{=}0)$, bootstrap 95% CI over
20 pairs; `links_tm` is the time-mean active link count; a CI excluding 0
marks a significant response):

| formulation | credit | landscape | $\Delta$ links_tm | CI lo | CI hi | $\Delta$ link_rep | $\Delta$ link_assort |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logit | structural | mgstd5 | 6.607 | 5.456 | 7.794 | -0.159 | -0.301 |
| logit | structural | noisypr5 | 3.289 | 3.047 | 3.526 | -0.178 | -0.053 |
| logit | equal_split | mgstd5 | 6.175 | 5.221 | 7.145 | -0.167 | -0.268 |
| logit | equal_split | noisypr5 | 3.341 | 3.072 | 3.588 | -0.174 | -0.033 |
| threshold | structural | mgstd5 | 5.388 | 4.785 | 6.006 | -0.184 | -0.401 |
| threshold | structural | noisypr5 | 3.630 | 3.365 | 3.889 | -0.209 | -0.318 |
| threshold | equal_split | mgstd5 | 5.248 | 4.623 | 5.906 | -0.196 | -0.406 |
| threshold | equal_split | noisypr5 | 3.585 | 3.369 | 3.808 | -0.211 | -0.329 |

Choice-layer response significant in 8 of 8 contrasts (at least one of the three choice diagnostics).

## Production moments against the diagnostic targets

Paired distance-to-target reduction from $\lambda=0$ to $\lambda=1$
(positive = moved toward the target; the CI column is the bootstrap 95% CI
of the reduction; sig = CI excludes 0):

| formulation | credit | landscape | team d($\lambda$0) | team d($\lambda$1) | team reduction [CI] | rep reduction [CI] |
| --- | --- | --- | --- | --- | --- | --- |
| logit | structural | mgstd5 | 5.472 | 5.472 | 0.000 [0.000, 0.000] ns | 0.000 [0.000, 0.000] ns |
| logit | structural | noisypr5 | 0.564 | 0.564 | 0.000 [0.000, 0.000] ns | 0.000 [0.000, 0.000] ns |
| logit | equal_split | mgstd5 | 5.542 | 5.361 | 0.181 [-0.151, 0.524] ns | 0.000 [-0.001, 0.001] ns |
| logit | equal_split | noisypr5 | 0.442 | 0.443 | -0.001 [-0.101, 0.084] ns | -0.000 [-0.003, 0.003] ns |
| threshold | structural | mgstd5 | 5.736 | 5.736 | 0.000 [0.000, 0.000] ns | 0.000 [0.000, 0.000] ns |
| threshold | structural | noisypr5 | 0.552 | 0.552 | 0.000 [0.000, 0.000] ns | 0.000 [0.000, 0.000] ns |
| threshold | equal_split | mgstd5 | 5.147 | 5.351 | -0.203 [-0.874, 0.423] ns | -0.000 [-0.001, 0.001] ns |
| threshold | equal_split | noisypr5 | 0.538 | 0.516 | 0.022 [-0.075, 0.139] ns | 0.001 [-0.001, 0.002] ns |

## The brief's central question: $\lambda = 0 \to 1$

Paired mean change from pure proxy matching to pure goal matching (wedge
change negative = the wedge narrows; ideal-loss change negative = less ideal
merit destroyed):

| formulation | credit | landscape | $\Delta$ wedge_tm | $\Delta$ wedge [CI] | $\Delta$ c_ideal_tm | $\Delta$ ideal_loss_cum | $\Delta$ team_tm | $\Delta$ rep_tm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logit | structural | mgstd5 | 0.0000 | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.00e+00 | 0.000 | 0.000 |
| logit | structural | noisypr5 | 0.0000 | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.00e+00 | 0.000 | 0.000 |
| logit | equal_split | mgstd5 | 0.0018 | 0.0018 [-0.0013, 0.0055] | 0.0032 | 6.15e-03 | 0.296 | -0.011 |
| logit | equal_split | noisypr5 | -0.0010 | -0.0010 [-0.0097, 0.0070] | 0.0029 | -2.27e-04 | 0.235 | -0.012 |
| threshold | structural | mgstd5 | 0.0000 | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.00e+00 | 0.000 | 0.000 |
| threshold | structural | noisypr5 | 0.0000 | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.00e+00 | 0.000 | 0.000 |
| threshold | equal_split | mgstd5 | -0.0048 | -0.0048 [-0.0098, -0.0003] | -0.0004 | -1.28e-03 | -0.900 | 0.004 |
| threshold | equal_split | noisypr5 | -0.0012 | -0.0012 [-0.0058, 0.0029] | 0.0060 | -1.55e-04 | 0.177 | 0.006 |

## Five-level $\lambda$ trends

Cell means per $\lambda$ level (20 seeds each), one row per
formulation $\times$ credit $\times$ landscape:

**mean_team_size (diagnostic target 2.65)**
| formulation | credit | landscape | $\lambda$=0 | 0.25 | 0.5 | 0.75 | 1.0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logit | structural | mgstd5 | 5.414 | 5.414 | 5.414 | 5.414 | 5.414 |
| logit | structural | noisypr5 | 2.863 | 2.863 | 2.863 | 2.863 | 2.863 |
| logit | equal_split | mgstd5 | 7.425 | 7.100 | 6.964 | 10.260 | 7.721 |
| logit | equal_split | noisypr5 | 2.913 | 3.141 | 3.040 | 3.124 | 3.149 |
| threshold | structural | mgstd5 | 6.069 | 6.069 | 6.069 | 6.069 | 6.069 |
| threshold | structural | noisypr5 | 3.230 | 3.230 | 3.230 | 3.230 | 3.230 |
| threshold | equal_split | mgstd5 | 6.981 | 6.679 | 5.300 | 7.058 | 6.081 |
| threshold | equal_split | noisypr5 | 3.084 | 3.068 | 3.299 | 2.968 | 3.261 |

**repeated_collab_share (diagnostic target 0.34)**
| formulation | credit | landscape | $\lambda$=0 | 0.25 | 0.5 | 0.75 | 1.0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logit | structural | mgstd5 | 0.918 | 0.918 | 0.918 | 0.918 | 0.918 |
| logit | structural | noisypr5 | 0.831 | 0.831 | 0.831 | 0.831 | 0.831 |
| logit | equal_split | mgstd5 | 0.919 | 0.936 | 0.945 | 0.927 | 0.907 |
| logit | equal_split | noisypr5 | 0.824 | 0.810 | 0.839 | 0.841 | 0.812 |
| threshold | structural | mgstd5 | 0.923 | 0.923 | 0.923 | 0.923 | 0.923 |
| threshold | structural | noisypr5 | 0.879 | 0.879 | 0.879 | 0.879 | 0.879 |
| threshold | equal_split | mgstd5 | 0.899 | 0.904 | 0.905 | 0.886 | 0.903 |
| threshold | equal_split | noisypr5 | 0.844 | 0.866 | 0.861 | 0.840 | 0.850 |

**n_active_links (last step)**
| formulation | credit | landscape | $\lambda$=0 | 0.25 | 0.5 | 0.75 | 1.0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logit | structural | mgstd5 | 4.350 | 5.150 | 6.800 | 9.050 | 12.600 |
| logit | structural | noisypr5 | 5.550 | 5.450 | 7.000 | 9.750 | 10.850 |
| logit | equal_split | mgstd5 | 4.300 | 5.350 | 6.750 | 9.100 | 14.150 |
| logit | equal_split | noisypr5 | 5.200 | 5.200 | 6.950 | 9.400 | 10.300 |
| threshold | structural | mgstd5 | 1.000 | 2.400 | 3.500 | 4.650 | 11.050 |
| threshold | structural | noisypr5 | 1.150 | 2.100 | 3.750 | 5.600 | 7.750 |
| threshold | equal_split | mgstd5 | 1.000 | 2.400 | 3.050 | 5.250 | 8.450 |
| threshold | equal_split | noisypr5 | 1.150 | 1.950 | 3.800 | 6.100 | 8.900 |

**Mean_of_Corr_prestige_ideal_merit**
| formulation | credit | landscape | $\lambda$=0 | 0.25 | 0.5 | 0.75 | 1.0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logit | structural | mgstd5 | 0.104 | 0.104 | 0.104 | 0.104 | 0.104 |
| logit | structural | noisypr5 | 0.377 | 0.377 | 0.377 | 0.377 | 0.377 |
| logit | equal_split | mgstd5 | 0.087 | 0.086 | 0.085 | 0.085 | 0.090 |
| logit | equal_split | noisypr5 | 0.365 | 0.361 | 0.364 | 0.367 | 0.368 |
| threshold | structural | mgstd5 | 0.080 | 0.080 | 0.080 | 0.080 | 0.080 |
| threshold | structural | noisypr5 | 0.366 | 0.366 | 0.366 | 0.366 | 0.366 |
| threshold | equal_split | mgstd5 | 0.063 | 0.070 | 0.064 | 0.064 | 0.063 |
| threshold | equal_split | noisypr5 | 0.369 | 0.375 | 0.378 | 0.374 | 0.375 |

## The structural floor (off-mode baseline)

The same skeleton with `dyadic_mode="off"` (40 runs; co-harvest teams exist
under the visibility coupling alone, with no choice layer; `rep_mass` =
fraction of steps with the repeated share at exactly 0 / 1):

| landscape | team_tm | rep_tm | assort_tm | rep mass@0 | rep mass@1 | c_ideal_tm | c_local_tm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mgstd5 | 8.121 | 0.931 | 0.001 | 0.006 | 0.426 | 0.106 | 0.129 |
| noisypr5 | 3.193 | 0.876 | 0.116 | 0.013 | 0.301 | 0.348 | 0.373 |

Repeated-share distribution shape (fraction of steps at exactly 0 and at exactly 1, $\lambda=0$ then $\lambda=1$ --- the empirical moment is mass at 0 and at 1):
| formulation | credit | landscape | mass@0 ($\lambda$0) | mass@1 ($\lambda$0) | mass@0 ($\lambda$1) | mass@1 ($\lambda$1) |
| --- | --- | --- | --- | --- | --- | --- |
| logit | structural | mgstd5 | 0.005 | 0.437 | 0.005 | 0.437 |
| logit | structural | noisypr5 | 0.014 | 0.310 | 0.014 | 0.310 |
| logit | equal_split | mgstd5 | 0.005 | 0.439 | 0.005 | 0.426 |
| logit | equal_split | noisypr5 | 0.014 | 0.299 | 0.014 | 0.301 |
| threshold | structural | mgstd5 | 0.006 | 0.443 | 0.006 | 0.443 |
| threshold | structural | noisypr5 | 0.012 | 0.304 | 0.012 | 0.304 |
| threshold | equal_split | mgstd5 | 0.006 | 0.421 | 0.006 | 0.431 |
| threshold | equal_split | noisypr5 | 0.015 | 0.307 | 0.015 | 0.306 |

# Gate 2 verdict

**PARTIAL** (mechanical rule, counts from `gate2_verdict.json`): team-size
distance reduction significant in 0 of 8 contrasts; repeated-share
reduction significant in 0 of 8; choice-layer response significant in 8
of 8.


Reading: the dyadic layer works as a *choice* mechanism --- partner
selection responds to $\lambda$ in both formulations --- but under the ruled
menu it does not reach *joint production*: co-harvest teams are set by
co-location dynamics (the visibility coupling and entry clustering), which
no ruled channel connects to the link set. The equal_split credit rule is
the one ruled feedback channel (split prestige $\to$ prestige-weighted
visibility reward $\to$ movement $\to$ co-location); its paired effects are
in the tables above and in `gate2_verdict.json` (`feedback_descriptive`).
Per `way_forward.md`'s own Gate 2 rule, the honest paper is A+D and the team
layer is reported as robustness; the movement coupling is filed as a Need,
not silently added.

# Gross versus net

The gate ran before the first edit and after the last. Opening (pre-Path-B
code = the N1-closing tree, 44 report keys; text copied from this round's
session log into `data/path_b_v1/canary_opening.txt`, the random `/tmp`
append-check stem normalized):

```{.text}
canary run OK: 44 report keys, harvested_10_step=37.5, harvested_25_step=37.5
same-seed rerun bit-identical OK
newcomer run OK: 6 scientists after 20 steps
4-arg Gaussian OK: np.float64(0.00550470835249002) (float64)
/tmp/smoke_append_<tmpdir>/append_header_check.csv w
now processing param set  1 out of  1
/tmp/smoke_append_<tmpdir>/append_header_check.csv a
now processing param set  1 out of  1
append-mode header check OK: 3 lines = 1 header + 2 data rows
ideal-merit pairing audit OK: A=0.0, C=0.0, B=-0.04999999999999999 (hand-computed 0.0, 0.0, -0.05); cum after 2 calls: A=0.0, B=-0.09999999999999998
seed columns OK: agent_seed=1, landscape_seed=1
WARNING: agent_generation_function '<lambda>' returned IDENTICAL draws for mu=0.1 and mu=0.9 on the same rng - the swept axis (initial_curiosity) is INERT.
binding probe OK: constant generator flagged inert, identity binds
entry_prestige_stock OK: validated, both alternate stocks run
ALL SMOKE TESTS PASSED
```

Closing (post-Path-B code, 56 report keys --- this round's six dyadic columns among them;
this is the sweep driver's own in-run gate, `data/path_b_v1/canary.txt`):

```{.text}
canary run OK: 56 report keys, harvested_10_step=37.5, harvested_25_step=37.5
same-seed rerun bit-identical OK
newcomer run OK: 6 scientists after 20 steps
4-arg Gaussian OK: np.float64(0.00550470835249002) (float64)
/tmp/smoke_append_dgz7eyc8/append_header_check.csv w
now processing param set  1 out of 1
/tmp/smoke_append_dgz7eyc8/append_header_check.csv a
now processing param set  1 out of 1
append-mode header check OK: 3 lines = 1 header + 2 data rows
ideal-merit pairing audit OK: A=0.0, C=0.0, B=-0.04999999999999999 (hand-computed 0.0, 0.0, -0.05); cum after 2 calls: A=0.0, B=-0.09999999999999998
seed columns OK: agent_seed=1, landscape_seed=1
WARNING: agent_generation_function '<lambda>' returned IDENTICAL draws for mu=0.1 and mu=0.9 on the same rng - the swept axis (initial_curiosity) is INERT.
binding probe OK: constant generator flagged inert, identity binds
entry_prestige_stock OK: validated, both alternate stocks run
Path B off-mode bit-identity OK (anchor mean_prestige=np.float64(0.025399760810316363)); 0 dynamics keys differ
Path B binding OK: logit lambda0=(np.int64(2), 1.024390243902439) lambda1=(np.int64(5), 2.1951219512195124)
Path B binding OK: threshold lambda0=(np.int64(1), 1.0) lambda1=(np.int64(3), 2.048780487804878)
Path B knob validation OK: all six knobs reject invalid values
ALL SMOKE TESTS PASSED
```

The key-count delta 44 $\to$ 56 is +12: this round's six dyadic reporters
(`mean_team_size`, `repeated_collab_share`, `dyadic_assortativity_prestige`,
`n_active_links`, `link_repeated_share`, `link_assortativity_prestige`),
additive only, plus 6 further keys that appeared in the closing canary
because the owner's concurrent rounds landed in the live tree while this
round ran (the same live coordination as the CDF re-anchor below;
`sciencemodels/sweeps.py` at 22:29, the CDF doc at 22:56 and 23:10). The
anchor `mean_prestige` and the off-mode dynamics are bit-identical across
the gate regardless: the opening canary was copied from the pre-Path-B
tree, the closing one is the sweep driver's own in-run gate.

|| Unit | Attempted | Verified |
|| --- | --- | --- | --- |
|| sweep cells ($\lambda$ $\times$ formulation $\times$ credit $\times$ landscape) | 40 | 40 (40 ok, 0 skipped, 0 failed) |
|| sweep runs | 800 | 800 (800 rows net, 0 failed seeds) |
|| off-mode baseline runs (2 landscapes $\times$ 20 seeds) | 40 | 40 |
|| probe runs (critical path, schema + timing) | 2 | 2 |
|| smoke-test invocations | 2 (opening, closing) | 2 pass |

The driver's closing line, reconstructed from the run log (40 ok cell
lines):

```{.text}
GROSS cells=40 runs=800 | NET ok=40 skipped=0 failed_cells=0 failed_seeds=0 rows_net=800/800
```

Wall time: 2221 s (manifest wall_s=2221.1). Independent recount: 40 cell CSVs under
`data/path_b_v1/cells/`, 800 data rows (800 expected); the off-mode baseline
CSV carries 40 rows.

Defects found and fixed inside this round, all mine, all re-verified: the
analyser's first choice-contrast used a distance-to-zero reduction (wrong
semantics for a sign-free response; replaced by the raw paired difference);
the doc generator's first draft interpolated empty placeholders for the
driver's closing line and the recount (rebuilt to compute before
formatting); the timing probe's schema list misnamed the two correlation
report keys (uppercase `Mean_of_` convention). Nothing left unresolved.


# Reproduction

```{.bash}
MPLBACKEND=Agg /opt/hermes/.venv/bin/python tests/test_smoke.py
MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/path_b/run_path_b_sweep.py
MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/path_b/run_baseline_off.py
MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/path_b/analyse_path_b.py
MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/path_b/make_doc.py
```

Outputs: `data/path_b_v1/cells/cell_lam*_*.csv` (40), `baseline_off.csv`,
`gate2_table.csv`, `gate2_verdict.json`, `lambda_trend.csv`,
`cell_summary.csv`, `baseline_off_summary.csv`, `manifest.json`,
`run_log.txt`, `canary.txt`, `canary_opening.txt`. `data/` is gitignored;
the CSVs plus the four scripts are the source of truth.

# Needs reported rather than changed

- **Movement coupling.** The ruled menu contains no channel from the link
  set to agent movement; under `structural` credit the production moments
  are $\lambda$-insensitive by construction (the choice layer binds, the
  production layer cannot). A partner-proximity pull in the reward (linked
  partners enter the distance/visibility term) is the natural candidate ---
  a new mechanism, not a parameter, and therefore an owner decision, filed
  here rather than added silently.
- **Closure moment.** The Markov C club-closure moment (retention 0.3995,
  entry 0.0619) is not measurable from a 200-step single-layer simulation;
  it needs the panel reconstruction (author $\times$ step club membership)
  the Klann pipeline uses. Filed as a measurement need for a Gate 2 re-run,
  not approximated here.
- **Team-size definition.** The diagnostic 2.65 is a per-paper author count
  (whether solos are included is not pinned in the memo); the model's
  `mean_team_size` counts co-harvest groups $\geq 2$ only. The solos-
  inclusive variant is one line in the analyser if the owner wants it.
- **Newborn team accounting.** Newcomers spawn on the supervisor's tile and
  count in that step's team (D5); their birth-step `ideal_Merit` is the
  pre-existing 0 preset (N1 disclosure).
- **equal_split deferral.** Under `equal_split`, prestige accrual is
  deferred to end-of-step (D6) --- inherent to pooling, disclosed; the
  structural path is untouched and bit-identical.
- **CDF re-anchor.** The Path B code shifted `model.py`/`agents.py`/
  `metrics.py` line numbers non-uniformly. The catalogue's Reference cells
  were re-anchored against the live tree and the reanchor mapping regenerated
  (106 rows: the re-measure round's two deferred rows - `entry_prestige_stock`
  and `ideal_merit_cum` - landed with their mapping entries in one change,
  as that round supplied ready-to-paste); `reanchor_cdf_refs.py` now
  **PASSES** on all 106 rows and 8 inline anchors. Two coordination notes:
  the doc was edited concurrently by the owner's round mid-re-anchor (the
  deferral revert at 22:56, plus three hand-edited cells at 23:10 that had
  copied neighbouring rows' values - corrected content-keyed, gate re-run
  green); and the gate's callout counter prints 111 gross / 109 resolvable
  because the two supplied mapping entries carry the owner's no-predecessor
  marker string, which holds no file refs (the re-measure round's 114/112
  prediction assumed the drafted cells in the old slot). See the CDF doc's
  Version 5 revision-log entry for the anchor-level detail.

# Revision Log

- **Version 1** (November 2026) --- Initial Path B probe: dyadic layer
  implemented per the ruled W3-C menu (Q1-Q8), Gate 2 sweep (40 cells, 800
  runs), off-mode structural baseline, pre-declared mechanical Gate 2
  verdict, declared micro-pins D1-D9, needs filed. Model-internal, not
  calibrated.

