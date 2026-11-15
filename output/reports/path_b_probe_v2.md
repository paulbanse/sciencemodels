---
title: "Path B probe v2: the movement coupling and Gate 2 re-run"
version: 1
date: 2026-11-15
revision-vehicle: strikethrough
revision-renders: "md, pdf, html"
tags: [path-b, dyadic, movement-coupling, gate-2, team-size, repeated-collab, proxy-failure, abm, not-calibrated]
project: sciencemodels
---

**Version 1** (November 2026)

**NOT CALIBRATED.** Model-internal exploration only (owner ruling 2026-11-13).
The Klann moments (team size mean 2.65 / median 2, repeated-collaboration share
0.34) are *diagnostic reference values* for the Gate 2 evaluation, never fitting
targets. Every number below was written by `scripts/path_b/run_path_b_v2_sweep.py`,
`scripts/path_b/probe_move_pull.py` and `scripts/path_b/analyse_path_b_v2.py`
from `data/path_b_v2/`.

# Purpose and scope

`output/reports/path_b_probe_v1.md` closed with Gate 2 **PARTIAL**: partner
choice responded to $\lambda$ in 8 of 8 contrasts, but the production moments
did not move in any, because the ruled menu contained no channel from the link
set to agent movement --- co-harvest teams are set by spatial co-location, and
nothing connected them to `model.links`. That probe filed the gap as a Need
("Movement coupling ... a new mechanism, not a parameter, and therefore an
owner decision").

This round implements that mechanism --- `dyadic_move_pull` --- and re-runs
Gate 2 on a targeted grid. It is the same model, the same landscape bundles and
the same diagnostic moments; what changed is that a declared link now exerts a
pull on where its holders move.

# The mechanism

`dyadic_move_pull` multiplies a candidate tile's reward by the agent's
proximity to its declared partners:

$$R_i(pos) \;\leftarrow\; R_i(pos)\cdot\left(1 + \text{pull} \cdot \overline{\frac{1}{d_{torus}(pos, partner_j) + 1}}\right)$$

over the agents $j$ this agent holds an active link with
(`Scientist.partner_proximity`, `sciencemodels/agents.py`;
`MyModel.linked_partners`, `sciencemodels/model.py`). The pull is applied to the
current tile and to every candidate neighbour alike, so the comparison between
staying and moving stays like-for-like. Three implementation choices matter for
reading the results:

- **Default 0.0 is bit-identical to Path B v1, not merely close.** The whole
  block is skipped when the pull is 0 or the agent has no link, so no
  multiplication by 1.0 is ever performed. The canary asserts eight bit-level
  anchors and the driver repeats the check against four stored v1 runs on
  60 numeric columns (max absolute difference **0.0**).
- **A positive pull with the layer off is refused** (`ValueError`), because the
  pull reads `model.links` and would otherwise be silently inert --- the failure
  mode the project's canary discipline exists to catch.
- **One-step lag.** Links are formed at the end of a step
  (`metrics.dyadic_step`), so the pull reads the previous step's link set. This
  is disclosed rather than engineered away: closing the lag would mean
  re-running the link update inside the movement loop, changing the rng order.

# Design

$\lambda \in \{0.0, 0.5, 1.0\}$ $\times$ `dyadic_move_pull` $\in \{0.5, 1.0\}$
$\times$ formulation $\in \{$logit$, $threshold$\}$ $\times$ credit
$\in \{$structural$, $equal_split$\}$ $\times$ landscape
$\in \{$mgstd5$, $noisypr5$\}$ = **48 cells, 20 seeds, 960 runs**,
`step_limit=200`, `agent_generation_time=0.1`, `initial_curiosity=0.5`,
`n_jobs=3`. Constants and the seed semantics are in
`data/path_b_v2/manifest.json` (`design_pins`); the pull levels are the two
values the owner's order named.

**Control arm inside the driver.** Before the grid, two Path B v1 cells x two
seeds are re-run at `dyadic_move_pull=0.0` and every numeric column is diffed
against the stored v1 CSV row with the same `agent_seed`. This is what licenses
using the v1 cells as the pull=0 reference arm; a regeneration re-proves it
instead of assuming it.

# Verification

Canary before the round (pre-edit tree, detached worktree at `HEAD`): 56 report
keys, all checks pass. Closing canary (post-edit tree, run inside the driver
before the first cell --- the tail of `data/path_b_v2/canary.txt`):

```{.text}
entry_prestige_stock OK: validated, both alternate stocks run
Path B off-mode bit-identity OK (anchor mean_prestige=np.float64(0.025399760810316363)); 0 dynamics keys differ
Path B binding OK: logit lambda0=(np.int64(2), 1.024390243902439) lambda1=(np.int64(5), 2.1951219512195124)
Path B binding OK: threshold lambda0=(np.int64(1), 1.0) lambda1=(np.int64(3), 2.048780487804878)
Path B knob validation OK: all six knobs reject invalid values
ALL SMOKE TESTS PASSED
```

```{.text}
Path B binding OK: logit lambda0=(np.int64(2), 1.024390243902439) lambda1=(np.int64(5), 2.1951219512195124)
Path B binding OK: threshold lambda0=(np.int64(1), 1.0) lambda1=(np.int64(3), 2.048780487804878)
Path B v2 identity OK: dyadic_move_pull=0.0 reproduces Path B v1 on 8 anchors
Path B v2 coupling OK: team/rep/links pull0_lam1=(4.239837398373984, 0.8680933654246947, 1.7073170731707317) lam0_pull1=(3.9837398373983737, 0.8711882388625078, 1.024390243902439) lam1_pull1=(4.128048780487805, 0.8650409606774322, 2.707317073170732)
Path B knob validation OK: seven knobs reject invalid values, and a positive pull with the layer off is refused
ALL SMOKE TESTS PASSED
```

The round added three canary checks: `dyadic_move_pull=0.0` reproduces Path B
v1 on 8 bit-level anchors; the pull and $\lambda$ both bind the production
layer; seven knobs reject invalid values and a positive pull with the layer off
is refused.

**Probe (critical path, before the grid).** Identity: the pull=0 reruns match
the v1 artifacts on 60 numeric columns, max absolute difference
0.0. Binding, at (logit, structural, mgstd5, seeds 0-1):

| dyadic_move_pull | dyadic_lambda | team_tm | rep_tm | links_tm | c_ideal_tm |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 0.0 | 7.319 | 0.934 | 2.147 | 0.0689 |
| 0.0 | 1.0 | 7.319 | 0.934 | 8.047 | 0.0689 |
| 1.0 | 0.0 | 4.634 | 0.925 | 2.348 | -0.0005 |
| 1.0 | 1.0 | 2.888 | 0.921 | 9.443 | 0.1438 |

The production layer moves in 4 of 4 comparisons, in the direction of the
diagnostic: at $\lambda=1$ the pull takes mean team size from 7.32 to 2.89
against a reference of 2.65. Schema: the analysis keys were read back from a
worker-written file. Timing: **0.93 s per run**
on an idle box, projecting 297 s for
960 runs at `n_jobs=3`.

# Gross versus net

```{.text}
3 lambda x 2 dyadic_move_pull x 2 formulation x 2 credit x 2 landscape = 48 cells, 20 seeds, 960 runs
GROSS cells=48 runs=960 | NET cells_new=48 skipped=0 failed_cells=0 failed_seeds=0 rows_net=960/960
wall=705.2s | sources moved during run: none
```

The recount is from disk (`data/path_b_v2/cells/cell_*.csv`, 48 files x 21 lines).
Source snapshot: sha256 of `agents.py`, `model.py`, `metrics.py` and the round's
common module recorded in `manifest.json` before the launch and re-checked
after it; the recorded result is **no change**.

# Results

## Gate 2 v2

Within each cell quadruple, $\lambda=1.0$ against $\lambda=0.0$, seed-matched;
`team_reduction` is the paired reduction in distance to 2.65 (positive = toward
the reference), with a bootstrap 95% CI over 20 seed-pairs (10,000 resamples,
seed 20261115). `delta_links_tm` is the choice layer's own
response.

| pull | formulation | credit | landscape | team_d_lam0 | team_d_lam1 | team_reduction | team_ci_lo | team_ci_hi | rep_reduction | rep_ci_lo | rep_ci_hi | delta_links_tm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | logit | equal_split | mgstd5 | 3.844 | 2.546 | +1.298 | +0.427 | +2.234 | +0.0048 | -0.0020 | +0.0121 | +6.724 |
| 0.5 | logit | equal_split | noisypr5 | 0.200 | 0.281 | -0.080 | -0.139 | -0.021 | -0.0317 | -0.0386 | -0.0243 | +4.957 |
| 0.5 | logit | structural | mgstd5 | 3.279 | 2.276 | +1.003 | +0.059 | +1.957 | +0.0050 | -0.0026 | +0.0124 | +6.784 |
| 0.5 | logit | structural | noisypr5 | 0.271 | 0.235 | +0.036 | -0.069 | +0.154 | -0.0264 | -0.0332 | -0.0188 | +4.872 |
| 0.5 | threshold | equal_split | mgstd5 | 4.601 | 2.045 | +2.557 | +1.520 | +3.634 | +0.0078 | +0.0002 | +0.0151 | +6.427 |
| 0.5 | threshold | equal_split | noisypr5 | 0.339 | 0.259 | +0.080 | -0.076 | +0.249 | -0.0130 | -0.0238 | -0.0023 | +5.428 |
| 0.5 | threshold | structural | mgstd5 | 4.542 | 2.478 | +2.064 | +0.932 | +3.218 | +0.0065 | +0.0004 | +0.0125 | +6.636 |
| 0.5 | threshold | structural | noisypr5 | 0.255 | 0.269 | -0.014 | -0.139 | +0.143 | -0.0099 | -0.0169 | -0.0031 | +5.468 |
| 1.0 | logit | equal_split | mgstd5 | 2.297 | 0.825 | +1.473 | +0.632 | +2.372 | +0.0021 | -0.0055 | +0.0096 | +6.824 |
| 1.0 | logit | equal_split | noisypr5 | 0.198 | 0.336 | -0.139 | -0.186 | -0.086 | -0.0345 | -0.0423 | -0.0268 | +5.198 |
| 1.0 | logit | structural | mgstd5 | 2.176 | 0.930 | +1.247 | +0.549 | +1.950 | +0.0039 | -0.0037 | +0.0112 | +6.574 |
| 1.0 | logit | structural | noisypr5 | 0.164 | 0.327 | -0.163 | -0.207 | -0.122 | -0.0333 | -0.0426 | -0.0254 | +5.326 |
| 1.0 | threshold | equal_split | mgstd5 | 3.186 | 1.219 | +1.967 | +1.055 | +2.939 | +0.0068 | -0.0002 | +0.0139 | +6.511 |
| 1.0 | threshold | equal_split | noisypr5 | 0.337 | 0.338 | -0.000 | -0.177 | +0.212 | -0.0108 | -0.0209 | -0.0004 | +5.264 |
| 1.0 | threshold | structural | mgstd5 | 3.187 | 1.187 | +2.000 | +1.129 | +2.885 | +0.0069 | +0.0002 | +0.0137 | +6.715 |
| 1.0 | threshold | structural | noisypr5 | 0.196 | 0.338 | -0.142 | -0.232 | -0.028 | -0.0175 | -0.0271 | -0.0075 | +5.527 |

**Verdict: PARTIAL.** Team-size reduction significant and positive in
4/8 contrasts at
`pull=0.5` and 4/8 at
`pull=1`; repeated-share reduction in
2/8 and
1/8; the choice layer responds in
8/8 and
8/8. The declared rule needs
$\geq 6$ of 8 in **both** pull levels for both production moments, so the
verdict is computed by `scripts/path_b/analyse_path_b_v2.py` from the table
above and recorded in `data/path_b_v2/gate2_v2_verdict.json`.

## $\lambda$ trends

Averaged over formulation $\times$ credit, 20 seeds per cell:

| dyadic_move_pull | landscape | links_tm_0.0 | links_tm_0.5 | links_tm_1.0 | rep_tm_0.0 | rep_tm_0.5 | rep_tm_1.0 | team_tm_0.0 | team_tm_0.5 | team_tm_1.0 | wedge_tm_0.0 | wedge_tm_0.5 | wedge_tm_1.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | mgstd5 | 1.644 | 4.000 | 8.286 | 0.928 | 0.919 | 0.922 | 6.681 | 5.224 | 4.757 | 0.038 | 0.054 | -0.025 |
| 0.5 | noisypr5 | 1.978 | 3.880 | 7.159 | 0.866 | 0.863 | 0.886 | 2.715 | 2.428 | 2.396 | 0.080 | 0.102 | 0.014 |
| 1.0 | mgstd5 | 1.704 | 4.159 | 8.360 | 0.925 | 0.915 | 0.920 | 5.326 | 3.842 | 3.424 | 0.035 | 0.064 | -0.037 |
| 1.0 | noisypr5 | 2.054 | 4.012 | 7.383 | 0.869 | 0.866 | 0.893 | 2.674 | 2.347 | 2.315 | 0.085 | 0.123 | 0.021 |

At `pull=1` the mean co-harvest team size is 4.000 at
$\lambda=0$ and 2.870 at $\lambda=1$ (reference 2.65); the repeated
share is 0.8971 and 0.9066 (reference 0.34).

## Why the verdict is PARTIAL: the gate rule meets an arm that is already at the reference

The eight contrasts per pull level split cleanly by landscape, and the split is
the whole result:

| pull | landscape | n_contrasts | baseline_team_distance | team_sig_pos | team_sig_neg | team_reduction_mean | rep_sig_pos | rep_reduction_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | mgstd5 | 4 | 4.067 | 4 | 0 | +1.7308 | 2 | +0.0060 |
| 0.5 | noisypr5 | 4 | 0.266 | 0 | 1 | +0.0054 | 0 | -0.0203 |
| 1.0 | mgstd5 | 4 | 2.712 | 4 | 0 | +1.6715 | 1 | +0.0049 |
| 1.0 | noisypr5 | 4 | 0.224 | 0 | 3 | -0.1111 | 0 | -0.0240 |

On `mgstd5` the $\lambda=0$ baseline sits 3.39 away from the team-size
reference and **all four** contrasts improve significantly at both pull levels
(mean paired reduction +1.672
at `pull=1`). On `noisypr5` the baseline is already 0.25 away ---
the arm *starts* at the diagnostic value --- so a reduction toward it cannot be
large, and at `pull=1` the pull overshoots: three of four contrasts are
significantly *negative*, i.e. team size is pushed below the reference
(2.32 against 2.65).

The pre-declared rule requires $\geq 6$ of 8 in both landscapes, which is
unsatisfiable for an arm whose baseline distance is already 0.2 wide. The rule
is reported as declared, not re-cut after the fact; the honest reading is that
the coupling reaches production wherever there is distance to close, and that a
future gate should express the criterion as a paired *shift* with a stated
tolerance rather than a distance-to-target that a well-placed baseline cannot
improve on.

The prestige--ideal wedge moves too: on `mgstd5` at `pull=1` it goes from
+0.035 at $\lambda=0$ to -0.037 at $\lambda=1$ --- goal
matching under the coupling narrows the wedge and takes it slightly negative.

## The pull's own effect against the Path B v1 arm

v2 cells minus the matched v1 cell (same $\lambda$, formulation, credit,
landscape and seed), bootstrap 95% CI:

| metric | pull | lambda | mean_delta_vs_pull0 | ci_lo | ci_hi | significant |
| --- | --- | --- | --- | --- | --- | --- |
| team_tm | 0.5 | 0.0 | -0.9327 | -1.1500 | -0.7355 | True |
| team_tm | 0.5 | 1.0 | -2.0540 | -2.4349 | -1.6783 | True |
| team_tm | 1.0 | 0.0 | -1.6309 | -1.9843 | -1.2995 | True |
| team_tm | 1.0 | 1.0 | -2.7606 | -3.2894 | -2.2525 | True |
| rep_tm | 0.5 | 0.0 | -0.0064 | -0.0083 | -0.0044 | True |
| rep_tm | 0.5 | 1.0 | +0.0008 | -0.0026 | +0.0041 | False |
| rep_tm | 1.0 | 0.0 | -0.0065 | -0.0084 | -0.0045 | True |
| rep_tm | 1.0 | 1.0 | +0.0032 | -0.0009 | +0.0072 | False |
| c_ideal_tm | 0.5 | 0.0 | -0.0552 | -0.0639 | -0.0467 | True |
| c_ideal_tm | 0.5 | 1.0 | +0.0430 | +0.0342 | +0.0517 | True |
| c_ideal_tm | 1.0 | 0.0 | -0.0664 | -0.0774 | -0.0558 | True |
| c_ideal_tm | 1.0 | 1.0 | +0.0522 | +0.0425 | +0.0622 | True |
| c_local_tm | 0.5 | 0.0 | -0.0208 | -0.0283 | -0.0134 | True |
| c_local_tm | 0.5 | 1.0 | +0.0142 | +0.0053 | +0.0231 | True |
| c_local_tm | 1.0 | 0.0 | -0.0305 | -0.0389 | -0.0222 | True |
| c_local_tm | 1.0 | 1.0 | +0.0209 | +0.0110 | +0.0306 | True |
| ideal_loss_cum | 0.5 | 0.0 | +0.0052 | -0.0005 | +0.0110 | False |
| ideal_loss_cum | 0.5 | 1.0 | +0.0198 | +0.0132 | +0.0262 | True |
| ideal_loss_cum | 1.0 | 0.0 | +0.0002 | -0.0053 | +0.0053 | False |
| ideal_loss_cum | 1.0 | 1.0 | +0.0355 | +0.0284 | +0.0423 | True |

## Figures

- `data/path_b_v2/path_b_v2_gate2.png` --- team size (left) and repeated share
  (right) against $\lambda$, one line per pull level, dashed = diagnostic
  reference. Each point is the mean over formulation $\times$ credit $\times$
  landscape (20 seeds per cell), so the left panel starts between the two
  landscape baselines rather than at either.

# Needs reported rather than changed

- **Repeated-collaboration share is structurally far above its reference.** It
  sits near 0.9 in every cell against a diagnostic 0.34, and the mass of steps
  at exactly 1 is large. The model's co-harvest pairs repeat because 20 agents
  on 400 tiles co-locate heavily; matching 0.34 would need a partner-turnover
  mechanism (links or teams dissolving faster), not a stronger pull. Reported,
  not added.
- **Team-size definition.** The diagnostic 2.65 is a per-paper author count;
  `mean_team_size` counts co-harvest groups $\geq 2$. The solos-inclusive
  variant is one line in the analyser if the owner wants it.
- **Closure moment.** Markov C club closure (retention 0.3995, entry 0.0619)
  still needs the author $\times$ step panel reconstruction; unchanged from
  probe v1.
- **One-step lag** between link formation and the pull (above), if the owner
  wants it closed.

# Reproduction

```{.bash}
MPLBACKEND=Agg /opt/hermes/.venv/bin/python tests/test_smoke.py
MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/path_b/probe_move_pull.py
MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/path_b/run_path_b_v2_sweep.py
MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/path_b/analyse_path_b_v2.py
MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/path_b/make_doc_v2.py
```

# Revision Log

- **Version 1** (November 2026) --- Movement coupling `dyadic_move_pull`
  implemented (default 0.0, bit-identical to Path B v1, positive-pull-with-layer-off
  refused); canary extended with identity and coupling checks; 48-cell, 960-run
  Gate 2 v2 grid with an in-driver control arm; gate verdict PARTIAL;
  model-internal, not calibrated.
