---
title: "Sensitivity map v1: the quality-prestige relation across the model's own degrees of freedom (Path A, model-internal)"
author: "Hermes Agent (squad lead) for Jakob Kapeller"
date: "2026-11-13"
tags: [sensitivity, proxy-failure, wedge, prestige-merit, abm, path-a, not-calibrated]
project: sciencemodels
revision-vehicle: strikethrough
revision-renders: "md, pdf, html"
---

**Version 1** (November 2026)

> **NOT CALIBRATED.** This is a model-internal sensitivity exploration (Path A of
> `docs/way_forward.md`). Per the owner ruling of 2026-11-13 the model is NOT
> calibrated or fitted to empirical data at this stage; the moments in
> `docs/calibration_moments.yaml` were used as reference context only, never as
> targets. Every number below is a property of the simulation, not of the
> Klann panel. No empirical claim is made or implied.

# Scope and design

Question (Path A): which parameter regions of the model's own degrees of freedom
produce a wedge-like decoupling --- prestige tracking a quantity-side merit
(`Corr_prestige_local_merit`) while decoupling from the quality-side ideal merit
(`Corr_prestige_ideal_merit`)?

Full factorial grid, 36 cells, 20 seeds per cell, 720 runs, 200 steps each:

| Axis | Parameter | Levels |
| --- | --- | --- |
| (a) visibility weight | `initial_curiosity` | 0.3, 0.5, 0.7 |
| (b) prestige memory | `vanishing_factor_for_prestige` | 0.5, 0.9, 1.0 |
| (c) landscape legibility | `cell_init_function` bundle | `multipleGaussians` {std:5, number_gaussians:2} vs `noisyGaussian` {prop_random:5, std:5, std_curiosity:0.1} |
| (d) entry rate | `agent_generation_time` | 0 (off), 0.1 (supervisor-proportional entry on) |

Fixed pins (identical in every cell; the base configuration follows
`scripts/bulk_runner.py` with two task-mandated exceptions --- `step_limit=200`
(bulk_runner pins 400) and scalar `initial_curiosity` per cell with the identity
curiosity function (bulk_runner's curiosity axis sweeps lists / uses `uniform`)):
`n_agents=20`, `grid_size=20` (20x20 torus), `step_limit=200`, `epsilon=0.1`,
`harvest=0.02`, `use_distance=True`, `new_questions=0`, `use_visibility_reward=True`,
`knowledge_normalization='share'`, `prestige_accrual='harvested'`,
`agent_generation_function` = model default (identity).

Seed semantics: one knob (`agent_seed` = 0..19) drives the agent RNG; the model
injects the same seed into `generation_params` (`sciencemodels/model.py:139-141`),
so landscape and agent RNG vary jointly --- 20 seeds are 20 joint draws, the
`bulk_runner.py` convention. Landscape bundles mirror bulk setups 2/3 (clean) and
6/7 (noisy) exactly. Note on axis (c): the two levels are two different landscape
*generators* (a clean 2-peak sum of Gaussians vs a uniform-noise-floor Gaussian),
so legibility is contrasted jointly with family, not as one generator at two noise levels.

Gross vs net: 36 cells gross, 36 net completed; 720 runs gross,
720 net completed; 0 failed seeds; 0 crashed cells; 144,720 trajectory
rows (720 runs x 201 steps, step 0-200). Sweep wall time 16.8 min
(joblib n_jobs=8). Final population: founders-only cells 20 agents; entry cells
41-41 agents (see quirk Q1).

# Canary and integrity

Canary (`tests/test_smoke.py`; the pre-sweep run used the then-current 3-check
version, the embedded run is the extended 5-check version a concurrent session
had landed by document time; exit code 0):

```{.bash}
$ /opt/hermes/.venv/bin/python tests/test_smoke.py
canary run OK: 40 report keys, harvested_10_step=37.5, harvested_25_step=37.5
same-seed rerun bit-identical OK
newcomer run OK: 6 scientists after 20 steps
4-arg Gaussian OK: np.float64(0.00550470835249002) (float64)
/tmp/smoke_append_yx_w5f3c/append_header_check.csv w
now processing param set  1 out of 1
/tmp/smoke_append_yx_w5f3c/append_header_check.csv a
now processing param set  1 out of 1
append-mode header check OK: 3 lines = 1 header + 2 data rows
ALL SMOKE TESTS PASSED
```

Integrity chain:

- Restart-safe checkpointing: `checkpoint.json` updated atomically after every
  completed cell; a restart skips checkpointed cells (the driver's SKIP path).
  The mechanism was in place but NOT exercised: the run log shows one continuous
  run (single 'sweep start' line, zero SKIP lines, 36 sequential completions
  15:50:10-16:05:55Z). What actually happened mid-run: this session's foreground
  terminal wait timed out (~420 s); the driver process itself kept running
  detached and finished uninterrupted. Restart-safety is a verified design
  property of the driver, not a claim about this run.
- Crash policy: a failing seed is recorded in the checkpoint and run log, never
  dropped silently; a cell with >=1 successful seed still yields a mean trajectory
  over its successful seeds (`n_seeds` discloses). Net result: zero failures occurred.
- Consolidation canary: at step 1 both prestige-merit correlations must be >= 0.95
  (ideal) and >= 0.80 (local) across all cells --- the first harvest is proportional
  to tile knowledge by construction, so step-1 correlation near 1 is the expected
  mechanical signature. Measured: ideal min 0.9999, local min
  0.8874 pooled over cells (the noisy landscape's uniform floor
  legitimately decorrelates local merit at step 1); seed-level minima at step 1:
  ideal 0.999, local 0.556 --- single seeds can start
  decorrelated, dispersion the pooled canary does not capture.
- Label canary: consolidated `trajectories_mean.csv` re-checked bit-exactly against
  raw per-cell CSVs (v1 of the consolidator mislabeled interleaved mean/sd columns;
  caught by this check, fixed, re-run --- the numbers in this document come from
  the corrected consolidation).
- Independent verification (squad): a read-only verifier re-derived every tabled
  number from the 36 raw cell CSVs --- all 36 cell-mean rows, the 24 pooled
  snapshot rows, the crossings, the 10 axis contrasts, all 8 final-step matrices,
  the zero-guard tables and every prose number matched (cell_summary.csv and
  trajectories_mean.csv re-derive bit-exactly, max abs diff 0.0); the same check
  flagged five prose overstatements and two stale code attributions in the first
  draft, all corrected in this version before delivery.
- Concurrent-edit disclosure: a concurrent session edited the shared tree DURING
  the sweep window, in stages --- `sciencemodels/model.py` 15:52:15Z (the BUG-0003
  header-path fix), `tests/test_smoke.py` 15:53:35Z (regression checks for both
  bugs), `sciencemodels/landscapes.py` 15:53:47Z (the BUG-0002 `rng=None` default).
  Cells 1-7 (finished 15:50:10-15:51:55Z) ran fully pre-edit; cells 8+ ran under a
  tree patched in stages (the model.py edit landed 15:52:15Z, three seconds before
  cell 8 finished at 15:52:18Z --- mid-cell; cell 10 finished 15:53:16Z, before the
  landscapes edit). Dynamics-neutrality of all three edits is established, not
  assumed: (a) the
  `model.py` change is +9 bytes inside
  `generate_data_parametric_exploration` only (this driver calls `MyModel` directly,
  never the sweep writer); (b) the `landscapes.py` change adds a default to a
  parameter `Gaussian` never reads; (c) empirically,
  `cell_cur03_van05_mgstd5_gen00` (written 15:50:10Z, fully pre-edit) was re-run on
  the post-edit code and is BIT-EXACT (max abs diff 0.0 over 4020 rows x 5 metrics).
  The mixed-code window therefore does not contaminate the grid. The embedded
  canary above is the extended 5-check smoke test (the pre-sweep canary was the
  3-check version; both passed).

# Cell means

Time-mean over steps 0-200 and over 20 seeds (`_tm`), and seed-mean at the final
step 200 (`_fin`). `wedge_fin` = `c_local_fin - c_ideal_fin` (local-minus-ideal
correlation gap at the horizon). All 36 cells, 20/20 seeds each:

| cell | cur | van | land | gen | c_ideal_tm | c_local_tm | top10_tm | explored_tm | harvested_tm | c_ideal_fin | c_local_fin | wedge_fin |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cell_cur03_van05_mgstd5_gen00 | 0.300 | 0.500 | mgstd5 | 0.000 | 0.211 | 0.213 | 0.134 | 0.312 | 0.395 | 0.343 | 0.343 | 0.000 |
| cell_cur05_van05_mgstd5_gen00 | 0.500 | 0.500 | mgstd5 | 0.000 | 0.291 | 0.279 | 0.134 | 0.313 | 0.424 | 0.280 | 0.284 | 0.005 |
| cell_cur07_van05_mgstd5_gen00 | 0.700 | 0.500 | mgstd5 | 0.000 | 0.333 | 0.306 | 0.134 | 0.314 | 0.431 | 0.348 | 0.325 | -0.023 |
| cell_cur03_van09_mgstd5_gen00 | 0.300 | 0.900 | mgstd5 | 0.000 | 0.195 | 0.199 | 0.133 | 0.312 | 0.389 | 0.160 | 0.157 | -0.002 |
| cell_cur05_van09_mgstd5_gen00 | 0.500 | 0.900 | mgstd5 | 0.000 | 0.284 | 0.280 | 0.134 | 0.313 | 0.424 | 0.343 | 0.327 | -0.016 |
| cell_cur07_van09_mgstd5_gen00 | 0.700 | 0.900 | mgstd5 | 0.000 | 0.339 | 0.313 | 0.134 | 0.314 | 0.431 | 0.265 | 0.281 | 0.017 |
| cell_cur03_van10_mgstd5_gen00 | 0.300 | 1.000 | mgstd5 | 0.000 | 0.196 | 0.202 | 0.132 | 0.311 | 0.393 | 0.164 | 0.134 | -0.030 |
| cell_cur05_van10_mgstd5_gen00 | 0.500 | 1.000 | mgstd5 | 0.000 | 0.280 | 0.273 | 0.134 | 0.313 | 0.424 | 0.275 | 0.354 | 0.079 |
| cell_cur07_van10_mgstd5_gen00 | 0.700 | 1.000 | mgstd5 | 0.000 | 0.327 | 0.299 | 0.134 | 0.314 | 0.431 | 0.372 | 0.378 | 0.006 |
| cell_cur03_van05_mgstd5_gen01 | 0.300 | 0.500 | mgstd5 | 0.100 | 0.072 | 0.107 | 0.185 | 0.321 | 0.457 | -0.168 | 0.062 | 0.230 |
| cell_cur05_van05_mgstd5_gen01 | 0.500 | 0.500 | mgstd5 | 0.100 | 0.110 | 0.132 | 0.188 | 0.327 | 0.484 | -0.049 | 0.074 | 0.123 |
| cell_cur07_van05_mgstd5_gen01 | 0.700 | 0.500 | mgstd5 | 0.100 | 0.119 | 0.127 | 0.188 | 0.328 | 0.490 | -0.044 | 0.049 | 0.093 |
| cell_cur03_van09_mgstd5_gen01 | 0.300 | 0.900 | mgstd5 | 0.100 | 0.061 | 0.099 | 0.185 | 0.322 | 0.455 | -0.145 | 0.054 | 0.199 |
| cell_cur05_van09_mgstd5_gen01 | 0.500 | 0.900 | mgstd5 | 0.100 | 0.107 | 0.129 | 0.187 | 0.328 | 0.484 | -0.054 | 0.082 | 0.135 |
| cell_cur07_van09_mgstd5_gen01 | 0.700 | 0.900 | mgstd5 | 0.100 | 0.120 | 0.128 | 0.188 | 0.329 | 0.490 | 0.020 | 0.115 | 0.095 |
| cell_cur03_van10_mgstd5_gen01 | 0.300 | 1.000 | mgstd5 | 0.100 | 0.058 | 0.097 | 0.185 | 0.320 | 0.451 | -0.175 | 0.066 | 0.241 |
| cell_cur05_van10_mgstd5_gen01 | 0.500 | 1.000 | mgstd5 | 0.100 | 0.094 | 0.115 | 0.187 | 0.327 | 0.484 | -0.078 | 0.003 | 0.081 |
| cell_cur07_van10_mgstd5_gen01 | 0.700 | 1.000 | mgstd5 | 0.100 | 0.097 | 0.110 | 0.187 | 0.329 | 0.490 | -0.048 | -0.001 | 0.047 |
| cell_cur03_van05_noisypr5_gen00 | 0.300 | 0.500 | noisypr5 | 0.000 | 0.335 | 0.378 | 0.179 | 0.320 | 0.145 | -0.031 | -0.033 | -0.003 |
| cell_cur05_van05_noisypr5_gen00 | 0.500 | 0.500 | noisypr5 | 0.000 | 0.575 | 0.593 | 0.192 | 0.338 | 0.154 | 0.096 | 0.258 | 0.162 |
| cell_cur07_van05_noisypr5_gen00 | 0.700 | 0.500 | noisypr5 | 0.000 | 0.726 | 0.735 | 0.200 | 0.344 | 0.152 | 0.366 | 0.461 | 0.095 |
| cell_cur03_van09_noisypr5_gen00 | 0.300 | 0.900 | noisypr5 | 0.000 | 0.332 | 0.384 | 0.179 | 0.321 | 0.147 | 0.023 | 0.077 | 0.054 |
| cell_cur05_van09_noisypr5_gen00 | 0.500 | 0.900 | noisypr5 | 0.000 | 0.592 | 0.606 | 0.193 | 0.339 | 0.154 | 0.164 | 0.140 | -0.024 |
| cell_cur07_van09_noisypr5_gen00 | 0.700 | 0.900 | noisypr5 | 0.000 | 0.720 | 0.722 | 0.200 | 0.345 | 0.152 | 0.382 | 0.423 | 0.041 |
| cell_cur03_van10_noisypr5_gen00 | 0.300 | 1.000 | noisypr5 | 0.000 | 0.335 | 0.391 | 0.180 | 0.322 | 0.147 | -0.120 | -0.059 | 0.061 |
| cell_cur05_van10_noisypr5_gen00 | 0.500 | 1.000 | noisypr5 | 0.000 | 0.594 | 0.615 | 0.193 | 0.339 | 0.154 | 0.158 | 0.206 | 0.048 |
| cell_cur07_van10_noisypr5_gen00 | 0.700 | 1.000 | noisypr5 | 0.000 | 0.721 | 0.727 | 0.200 | 0.344 | 0.152 | 0.348 | 0.442 | 0.094 |
| cell_cur03_van05_noisypr5_gen01 | 0.300 | 0.500 | noisypr5 | 0.100 | 0.239 | 0.279 | 0.237 | 0.351 | 0.179 | -0.217 | -0.021 | 0.196 |
| cell_cur05_van05_noisypr5_gen01 | 0.500 | 0.500 | noisypr5 | 0.100 | 0.350 | 0.373 | 0.246 | 0.378 | 0.194 | -0.143 | -0.007 | 0.136 |
| cell_cur07_van05_noisypr5_gen01 | 0.700 | 0.500 | noisypr5 | 0.100 | 0.430 | 0.455 | 0.256 | 0.382 | 0.193 | -0.197 | -0.001 | 0.196 |
| cell_cur03_van09_noisypr5_gen01 | 0.300 | 0.900 | noisypr5 | 0.100 | 0.230 | 0.272 | 0.236 | 0.354 | 0.180 | -0.206 | 0.022 | 0.229 |
| cell_cur05_van09_noisypr5_gen01 | 0.500 | 0.900 | noisypr5 | 0.100 | 0.346 | 0.373 | 0.247 | 0.375 | 0.193 | -0.111 | 0.088 | 0.199 |
| cell_cur07_van09_noisypr5_gen01 | 0.700 | 0.900 | noisypr5 | 0.100 | 0.419 | 0.445 | 0.256 | 0.381 | 0.193 | -0.187 | 0.047 | 0.235 |
| cell_cur03_van10_noisypr5_gen01 | 0.300 | 1.000 | noisypr5 | 0.100 | 0.226 | 0.269 | 0.237 | 0.350 | 0.181 | -0.253 | -0.056 | 0.197 |
| cell_cur05_van10_noisypr5_gen01 | 0.500 | 1.000 | noisypr5 | 0.100 | 0.339 | 0.367 | 0.247 | 0.378 | 0.193 | -0.148 | 0.036 | 0.184 |
| cell_cur07_van10_noisypr5_gen01 | 0.700 | 1.000 | noisypr5 | 0.100 | 0.424 | 0.446 | 0.256 | 0.381 | 0.193 | -0.160 | 0.075 | 0.234 |

# Where the prestige-merit correlation degrades (the wedge behaviour)

Two channels are tracked throughout: the **quantity-side** channel
(`Corr_prestige_local_merit`, prestige vs the 5-tile local knowledge mean) and the
**quality-side** channel (`Corr_prestige_ideal_merit`, prestige vs the loss against
a greedy ideal-harvest benchmark). The **wedge** is the local-minus-ideal gap:
prestige still tracking where the knowledge *is* while ceasing to track how well
the community's placement matches the ideal allocation.

## The degradation is a horizon-and-channel effect, not a level effect

Pooled mean trajectories (mean over the 9 cells sharing each landscape x entry
combination, i.e. over curiosity, vanishing factor and seeds):

| landscape | gen | step | c_ideal | c_local | wedge(L-I) | top10 | explored | harvested |
|---|---|---|---|---|---|---|---|---|
| mgstd5 | 0 | 1 | 1.000 | 0.998 | -0.002 | 0.653 | 0.095 | 0.001 |
| mgstd5 | 0 | 25 | 0.134 | 0.118 | -0.016 | 0.134 | 0.292 | 0.165 |
| mgstd5 | 0 | 50 | 0.228 | 0.225 | -0.003 | 0.122 | 0.302 | 0.292 |
| mgstd5 | 0 | 100 | 0.241 | 0.227 | -0.014 | 0.118 | 0.319 | 0.461 |
| mgstd5 | 0 | 150 | 0.238 | 0.241 | 0.003 | 0.116 | 0.333 | 0.569 |
| mgstd5 | 0 | 200 | 0.283 | 0.287 | 0.004 | 0.116 | 0.347 | 0.646 |
| mgstd5 | 0.1 | 1 | 1.000 | 0.998 | -0.002 | 0.765 | 0.096 | 0.001 |
| mgstd5 | 0.1 | 25 | 0.073 | 0.091 | 0.018 | 0.180 | 0.296 | 0.178 |
| mgstd5 | 0.1 | 50 | -0.246 | 0.100 | 0.346 | 0.158 | 0.307 | 0.323 |
| mgstd5 | 0.1 | 100 | -0.177 | 0.063 | 0.240 | 0.189 | 0.330 | 0.526 |
| mgstd5 | 0.1 | 150 | -0.108 | 0.089 | 0.198 | 0.179 | 0.353 | 0.660 |
| mgstd5 | 0.1 | 200 | -0.082 | 0.056 | 0.138 | 0.212 | 0.376 | 0.755 |
| noisypr5 | 0 | 1 | 1.000 | 0.902 | -0.098 | 0.221 | 0.085 | 0.002 |
| noisypr5 | 0 | 25 | 0.922 | 0.918 | -0.004 | 0.239 | 0.199 | 0.044 |
| noisypr5 | 0 | 50 | 0.803 | 0.810 | 0.006 | 0.216 | 0.271 | 0.085 |
| noisypr5 | 0 | 100 | 0.524 | 0.560 | 0.037 | 0.183 | 0.360 | 0.161 |
| noisypr5 | 0 | 150 | 0.312 | 0.354 | 0.042 | 0.165 | 0.416 | 0.220 |
| noisypr5 | 0 | 200 | 0.154 | 0.213 | 0.059 | 0.155 | 0.458 | 0.268 |
| noisypr5 | 0.1 | 1 | 1.000 | 0.895 | -0.105 | 0.278 | 0.086 | 0.002 |
| noisypr5 | 0.1 | 25 | 0.773 | 0.771 | -0.002 | 0.309 | 0.206 | 0.048 |
| noisypr5 | 0.1 | 50 | 0.484 | 0.607 | 0.123 | 0.264 | 0.282 | 0.099 |
| noisypr5 | 0.1 | 100 | 0.086 | 0.273 | 0.187 | 0.266 | 0.392 | 0.197 |
| noisypr5 | 0.1 | 150 | -0.128 | 0.088 | 0.216 | 0.225 | 0.469 | 0.280 |
| noisypr5 | 0.1 | 200 | -0.180 | 0.020 | 0.201 | 0.244 | 0.544 | 0.354 |

The ideal channel starts at 1.0 in every cell at step 1 --- mechanically, because
the first harvest is proportional to tile knowledge and both merit measures are
functions of tile knowledge. The local channel starts at ~1.0 on the clean
landscape but only ~0.90 pooled on the noisy one (the uniform noise
floor decorrelates a tile from its cross-neighborhood mean; seed-level minima at
step 1: ideal 0.999, local 0.556 --- single seeds can
start decorrelated, which the pooled canary does not see). What matters is the
decay:

- On the **clean** landscape the mechanical component washes out within ~25 steps
  (pooled ideal corr already 0.13 at step 25) and the rest of the horizon is
  depletion-contaminated (see F5).
- On the **noisy** landscape the correlations stay high until ~step 100
  (0.52 at step 100 without entry) and then decay smoothly --- the legible late-run
  evidence in this sweep.
- **Entry** (gen=0.1) does not produce a clean monotone collapse: the ideal
  channel oscillates (newcomers arrive in 10-step waves and the prestige roulette
  reshuffles), the pooled trajectory hovering near zero with negative dips (clean:
  min -0.309; noisy: first crossing only at step
  120), while the per-cell ENDSTATE is consistently
  negative --- 17 of 18 entry cells finish below zero. The local channel stays
  positive (clean) or decays to ~0 (noisy) throughout.

Crossings and oscillation of the pooled mean trajectory (step 0 excluded; the
'>=26' columns skip the mechanical transient; 'neg share' = fraction of steps
>=26 with pooled ideal correlation below zero; sign changes count zero crossings):

| landscape | gen | ideal<0 first | ideal<0 first (>=26) | ideal sign changes (>=26) | ideal neg share (>=26) | ideal max (>=26) | ideal min (>=26) | local<0.5 first | local<0 first (>=26) |
|---|---|---|---|---|---|---|---|---|---|
| mgstd5 | 0.0 | 14 | never | 0 | 0.0 | 0.327 | 0.13 | 10 | never |
| mgstd5 | 0.1 | 13 | 30 | 35 | 0.1 | 0.146 | -0.309 | 10 | never |
| noisypr5 | 0.0 | never | never | 0 | 0.0 | 0.922 | 0.148 | 111 | never |
| noisypr5 | 0.1 | 120 | 120 | 17 | 0.12 | 0.805 | -0.194 | 70 | never |

## Axis contrasts

Cell-level means pooled over the other three axes (time-mean and final-step):

| axis | level | c_ideal_tm | c_local_tm | c_ideal_fin | c_local_fin | wedge_fin | top10_fin | explored_tm | harvested_tm |
|---|---|---|---|---|---|---|---|---|---|
| curiosity (visibility weight 1-c) | 0.3 | 0.208 | 0.241 | -0.052 | 0.062 | 0.114 | 0.177 | 0.326 | 0.293 |
| curiosity (visibility weight 1-c) | 0.5 | 0.330 | 0.345 | 0.061 | 0.154 | 0.093 | 0.182 | 0.339 | 0.314 |
| curiosity (visibility weight 1-c) | 0.7 | 0.398 | 0.401 | 0.122 | 0.216 | 0.094 | 0.186 | 0.342 | 0.317 |
| vanishing_factor_for_prestige | 0.5 | 0.316 | 0.331 | 0.049 | 0.150 | 0.101 | 0.182 | 0.336 | 0.308 |
| vanishing_factor_for_prestige | 0.9 | 0.312 | 0.329 | 0.054 | 0.151 | 0.097 | 0.182 | 0.336 | 0.308 |
| vanishing_factor_for_prestige | 1 | 0.308 | 0.326 | 0.028 | 0.131 | 0.104 | 0.182 | 0.336 | 0.308 |
| landscape | mgstd5 | 0.183 | 0.189 | 0.101 | 0.172 | 0.071 | 0.164 | 0.319 | 0.446 |
| landscape | noisypr5 | 0.441 | 0.468 | -0.013 | 0.117 | 0.130 | 0.199 | 0.352 | 0.170 |
| agent_generation_time | 0 | 0.410 | 0.417 | 0.219 | 0.250 | 0.031 | 0.135 | 0.324 | 0.283 |
| agent_generation_time | 0.1 | 0.213 | 0.240 | -0.131 | 0.038 | 0.170 | 0.228 | 0.348 | 0.332 |

## Final-step correlation matrices (curiosity x vanishing, per landscape x entry)

### **ideal, mgstd5, gen=0**

| curiosity \ vanishing | 0.5 | 0.9 | 1 |
|---|---|---|---|
| 0.3 | 0.343 | 0.160 | 0.164 |
| 0.5 | 0.280 | 0.343 | 0.275 |
| 0.7 | 0.348 | 0.265 | 0.372 |

### **local, mgstd5, gen=0**

| curiosity \ vanishing | 0.5 | 0.9 | 1 |
|---|---|---|---|
| 0.3 | 0.343 | 0.157 | 0.134 |
| 0.5 | 0.284 | 0.327 | 0.354 |
| 0.7 | 0.325 | 0.281 | 0.378 |

### **ideal, mgstd5, gen=0.1**

| curiosity \ vanishing | 0.5 | 0.9 | 1 |
|---|---|---|---|
| 0.3 | -0.168 | -0.145 | -0.175 |
| 0.5 | -0.049 | -0.054 | -0.078 |
| 0.7 | -0.044 | 0.020 | -0.048 |

### **local, mgstd5, gen=0.1**

| curiosity \ vanishing | 0.5 | 0.9 | 1 |
|---|---|---|---|
| 0.3 | 0.062 | 0.054 | 0.066 |
| 0.5 | 0.074 | 0.082 | 0.003 |
| 0.7 | 0.049 | 0.115 | -0.001 |

### **ideal, noisypr5, gen=0**

| curiosity \ vanishing | 0.5 | 0.9 | 1 |
|---|---|---|---|
| 0.3 | -0.031 | 0.023 | -0.120 |
| 0.5 | 0.096 | 0.164 | 0.158 |
| 0.7 | 0.366 | 0.382 | 0.348 |

### **local, noisypr5, gen=0**

| curiosity \ vanishing | 0.5 | 0.9 | 1 |
|---|---|---|---|
| 0.3 | -0.033 | 0.077 | -0.059 |
| 0.5 | 0.258 | 0.140 | 0.206 |
| 0.7 | 0.461 | 0.423 | 0.442 |

### **ideal, noisypr5, gen=0.1**

| curiosity \ vanishing | 0.5 | 0.9 | 1 |
|---|---|---|---|
| 0.3 | -0.217 | -0.206 | -0.253 |
| 0.5 | -0.143 | -0.111 | -0.148 |
| 0.7 | -0.197 | -0.187 | -0.160 |

### **local, noisypr5, gen=0.1**

| curiosity \ vanishing | 0.5 | 0.9 | 1 |
|---|---|---|---|
| 0.3 | -0.021 | 0.022 | -0.056 |
| 0.5 | -0.007 | 0.088 | 0.036 |
| 0.7 | -0.001 | 0.047 | 0.075 |

## Findings

**F1 --- Entry is the strongest degrader, and it concentrates prestige.** With
`agent_generation_time=0.1` the final ideal-merit correlation collapses from
+0.219 to -0.131 (pooled); 17 of 18 entry
cells end with a *negative* ideal correlation (vs 2 of 18 without entry).
The quantity-side channel also degrades (+0.250 -> +0.038) but
stays near zero --- only 5 of 18 entry cells end even marginally negative
(worst -0.056) --- nothing like the ideal channel's collapse
(17 of 18 negative, worst -0.253). Prestige concentration roughly
doubles: final top-10% prestige share 0.135 -> 0.228. Mechanism
(code-grounded): newcomers spawn at their supervisor's tile and prestige accrues
to tile occupants, and supervisor selection is prestige-proportional --- the
prestigious accumulate both attention and successors. The wedge (local-minus-ideal)
at the horizon is +0.031 without entry vs +0.170 with entry.

**F2 --- Harder-to-read merit does not lower the average correlation; it
specifically rots the quality channel.** The noisy landscape *retains* high
correlations longer (slow depletion: 0.27
vs 0.65 harvested by step 200 without entry), but the
local-minus-ideal wedge is wider on the noisy landscape at every horizon (final
wedge +0.130 vs +0.071; time-mean wedge
+0.028 vs +0.006). This is the way_forward a priori
prediction (weak proxy -> worse proxy failure) confirmed in the ideal channel,
expressed as channel-specific decay rather than a lower average correlation.

**F3 --- Curiosity is protective.** Final ideal correlation rises monotonically in
curiosity: -0.052 / +0.061 / +0.122 for curiosity
0.3 / 0.5 / 0.7. On the noisy landscape without entry, curiosity 0.3 ends at or
near zero (two of three cells negative, the third at +0.023) while curiosity 0.7
ends at +0.35 to +0.38 (matrices above). Curiosity
is the novelty elasticity of the Cobb-Douglas reward; low curiosity means
visibility-weighted movement dominates, and prestige decouples from where merit
actually is.

**F4 --- Prestige memory is inert at this horizon (a useful negative result).**
Vanishing factor 0.5 / 0.9 / 1.0 moves the final ideal correlation only between
+0.028 and +0.054 (final local:
+0.131 to +0.151; every time-mean contrast is <= 0.02).
Memory length is therefore immaterial for 200-step outcomes; it matters for
comparability and for longer horizons, not for this horizon's magnitudes. Related
documentation-debt finding (checked first-hand against the tree): the Path-0 item
'unify `vanishing_factor_for_prestige` across constructor and runners' is ALREADY
satisfied in code --- constructor default 0.9 (`sciencemodels/model.py:131`),
`scripts/runner.py:59` 0.9, `scripts/bulk_runner.py:70` 0.9. The 'runners disagree
(0.5 vs 0.9)' premise survives only in the pre-restructure catalogue
(`docs/conceptual_degrees_of_freedom.md:132-134,191`) and in `way_forward.md`'s
Path-0 list; the W2 restructure unified the code. Path 0's remaining action is the
docs, not the code.

**F5 --- Part of the late-run 'decay' on clean landscapes is measurement
degeneracy, not dynamics.** By step 200 the clean landscape is
0.65-0.76
harvested; tile knowledge collapses toward uniformity, merit variance vanishes, and
the correlation reporter's zero-guard returns exactly 0 (indistinguishable from a
true zero). Exact-0 reads at step 200 (share of seeds):

| landscape | gen | ideal0 | local0 | harv |
|---|---|---|---|---|
| mgstd5 | 0.000 | 0.094 | 0.089 | 0.646 |
| mgstd5 | 0.100 | 0.000 | 0.078 | 0.755 |
| noisypr5 | 0.000 | 0.067 | 0.072 | 0.268 |
| noisypr5 | 0.100 | 0.000 | 0.072 | 0.354 |

Worst cells:

| cell_tag | ideal0 | local0 | harv |
|---|---|---|---|
| cell_cur03_van10_mgstd5_gen00 | 0.300 | 0.150 | 0.616 |
| cell_cur03_van05_mgstd5_gen00 | 0.250 | 0.200 | 0.618 |
| cell_cur03_van10_noisypr5_gen00 | 0.250 | 0.100 | 0.247 |

The noisy landscape (27-35% harvested at step 200) is the trustworthy late-run
evidence; clean-landscape values after ~step 100 should be read with this
contamination in mind.

## Where, in one paragraph

The wedge opens in the **ideal-merit channel**, **late** (under entry it oscillates
and ends negative in 17 of 18 cells; the step-1 value of 1.0 is mechanical), under
**entry** (strongest single
lever), on the **noisy** landscape (widest local-minus-ideal gap), at **low
curiosity** (visibility-dominated movement), and is **independent of prestige
memory** at this horizon. The worst region --- entry x noisy x curiosity 0.3 ---
ends the run with ideal correlation near -0.2 while the local channel sits near
zero: prestige tracks neither merit measure, and the small positive local values
that remain are quantity-side. The best region (no entry, clean landscape, any
curiosity) keeps both channels positive but modest (+0.13 to +0.38 final) with a
wedge near zero.

## Model quirks this sweep surfaced (findings about the model, not results)

- **Q1 --- Entry starts at 21 agents, not 20.** The init-time `end_loop_update()`
  runs the fractional accumulator once at step 0, where
  `floor(0*g) - floor(-1*g) = 1`; all 18 entry cells end with 41 scientists
  (20 founders + 20 scheduled newcomers + 1 init-time newcomer), not 40.
- **Q2 --- The ideal-merit benchmark is founder-sized.** `update_ideal_merit` builds
  the greedy ideal harvest for `range(model.number_agents)` = 20 slots while 41
  agents actually harvest under entry. Ideal-merit values in entry cells are
  measured against a founder-sized optimum, inflating the ideal channel's
  degradation. The benchmark-free local channel also degrades under entry
  (+0.250 -> +0.038), so the entry effect is real; its
  ideal-channel magnitude is partly benchmark strain. Both readings are reported,
  neither is corrected (no code edits allowed this round).
- **Q3 --- Zero-guard reads as zero.** `correlation_reporter` returns 0 when either
  stock has zero variance; degenerate late-run reads are indistinguishable from
  true decoupling downstream (F5).
- **Q4 --- One seed knob, two RNGs.** Landscape and agent RNG are coupled through
  `agent_seed` (the model injects `self._seed` into `generation_params`); per-seed
  landscape difficulty varies because the Gaussian centres are redrawn per seed.
  This is the `bulk_runner.py` convention, kept for comparability.

## Gate-1 relevance (model-internal reading)

Gate 1 asks whether any plausible region produces the wedge *and* concentration
simultaneously. Within this grid: yes --- the entry regions combine a negative
final ideal correlation (-0.131 pooled) with elevated concentration
(final top-10% share 0.228 vs 0.135) and a widened wedge
(+0.170 vs +0.031), with no explicit gaming margin in the model --- the
visibility-coupled reward plus prestige-proportional recruitment suffice to
decouple prestige from ideal merit. Three honesty clauses attach: (i) this is
model-internal, not calibrated --- whether the region is *empirically plausible*
is exactly the question the no-calibration ruling defers; (ii) the bracket is
leaky by design (`use_visibility_reward=True` throughout), so prestige-mediated
attention is active even in founder cells; (iii) the ideal channel under entry is
benchmark-strained (Q2).

# Honest caveats

- **Not calibrated** (see banner). Empirical moments were never used as targets;
  comparisons to them would be a category error at this stage.
- Synthetic landscape: absolute knowledge levels are meaningless; only ranks and
  correlations carry interpretation. Keep claims at the level of correlations and
  shares, never levels.
- Horizon: 200 steps; one model step maps to no empirical time unit yet.
- Axis (c) contrasts two landscape generators, not one generator at two noise
  levels (family and legibility move together).
- Late-run clean-landscape correlations are depletion-contaminated (F5).
- The seed knob couples agent and landscape RNG (Q4); 20 seeds is a modest
  replication count for correlation differences of ~0.05.
- The wedge measured here is the model's *internal* ideal-vs-local contrast; the
  empirical wedge (b quantity-side, quality-side) maps onto these channels only
  loosely (`docs/calibration_moments.yaml` documents the mapping caveats).

# Machine-readable outputs

All under `data/sensitivity_v1/` (gitignored; run-tagged):

| File | Content |
| --- | --- |
| `cell_<tag>.csv` x 36 | Per-cell seed-level trajectories, long format: `cell_tag, curiosity, vanishing_factor_for_prestige, landscape, agent_generation_time, seed, Step` + the 5 metrics; 4020 rows (20 seeds x 201 steps) each |
| `trajectories_mean.csv` | 7236 rows: per cell x step seed-mean and seed-sd of the 5 metrics + `n_seeds` |
| `cell_summary.csv` | 36 rows: per-cell time-mean and final values, wedge contrasts, seed counts, status |
| `checkpoint.json` | Restart-safe manifest: per-cell status, seeds ok/failed, errors, final population size, wall time |
| `run_log.txt` | Append-only progress log with per-cell completion lines |
| `driver_source.py`, `consolidator_source.py`, `doc_generator_source.py` | Exact source of the sweep driver, the consolidator and this document's generator (self-describing runs) |
| `manifest.json` | Grid spec, fixed pins, seed semantics, canary output, sha256 of sources, environment versions, counts |

Cell tag legend: `cell_cur{03,05,07}_van{05,09,10}_{mgstd5,noisypr5}_gen{00,01}`;
`mgstd5` = `multipleGaussians` {std:5, number_gaussians:2}, `noisypr5` =
`noisyGaussian` {prop_random:5, std:5, std_curiosity:0.1}; tags carry integer-scaled
parameter values, the CSVs carry the exact floats.

# Reproducibility

- Environment: python 3.13.5, mesa 3.2.0, numpy 2.5.4, pandas 3.1.0
  (`/opt/hermes/.venv/bin/python`).
- Driver sha256 (first 16): `159eade38e914022`; consolidator sha256 (first 16): `6cfd107039862f20`.
- Regenerate: `/opt/hermes/.venv/bin/python data/sensitivity_v1/driver_source.py`
  (checkpointed; delete `data/sensitivity_v1/` for a fresh sweep), then
  `consolidator_source.py`. This document: `doc_generator_source.py` (copied into
  `data/sensitivity_v1/`; every number interpolated from the CSVs).
- Boundaries honoured: no repository code touched; no `AdmiralsLog/` write (the
  session log entry and any records are deferred --- the task's write boundary
  permits only this document and `data/sensitivity_v1/*`); no git commands; no
  parameter values outside the four axes; frozen `context/` untouched.

