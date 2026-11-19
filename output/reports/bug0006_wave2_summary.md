# BUG-0006 wave 2 — re-measurement of the remaining g > 0 archives

**Model-internal; NOT calibrated** (owner ruling 2026-11-13). Generated programmatically by `scripts/bug0006_wave2/analyse_wave2.py`; every number below is derived from the archives at analysis time.

## Runs and invariance ledger

| archive | runs (net) |
| --- | --- |
| interventions_v2 (A 18 + B 36 + C 18 cells) | 1440 |
| typology_v2 | 480 |
| path_b_postbug (baseline 40 + v1 800 + v2 960) | 1800 |
| m0_m2_probe | 80 |

Invariance tolerance: `<= 1e-11` on every paired run. **Failures: none.**

| archive | n_pairs | worst non-ideal | column |
| --- | --- | --- | --- |
| interventions_non_ideal | 1440 | 0.0e+00 | None |
| interventions_g0_full_identity | 180 | 0.0e+00 | None |
| typology_non_ideal | 480 | 0.0e+00 | None |
| typology_g0_full_identity | 240 | 0.0e+00 | None |
| path_b_v1_non_ideal | 800 | 0.0e+00 | None |
| path_b_v2_non_ideal | 960 | 0.0e+00 | None |
| baseline_off_non_ideal | 40 | 0.0e+00 | None |

### What moves: ideal-channel census

| archive | metric | pairs | frac changed | pre mean | post mean | mean delta |
| --- | --- | --- | --- | --- | --- | --- |
| interventions | corr_prestige_ideal_merit_final | 1440 | 0.88 | -0.0708 | 0.0875 | +0.158 |
| interventions | Mean_of_Corr_prestige_ideal_merit | 1440 | 0.88 | 0.2043 | 0.2259 | +0.022 |
| typology | c_ideal_fin | 480 | 0.50 | 0.0995 | 0.1647 | +0.065 |
| typology | Mean_of_Corr_prestige_ideal_merit | 480 | 0.50 | 0.1797 | 0.1916 | +0.012 |
| typology | c_ideal_tm | 480 | 0.50 | 0.1797 | 0.1916 | +0.012 |
| typology | wedge_fin | 480 | 0.50 | -0.0571 | 0.0081 | +0.065 |
| path_b_v1 | Mean_of_Corr_prestige_ideal_merit | 800 | 1.00 | 0.2273 | 0.2491 | +0.022 |
| path_b_v1 | c_ideal_tm | 800 | 1.00 | 0.2273 | 0.2491 | +0.022 |
| path_b_v2 | Mean_of_Corr_prestige_ideal_merit | 960 | 1.00 | 0.1968 | 0.2177 | +0.021 |
| path_b_v2 | c_ideal_tm | 960 | 1.00 | 0.1968 | 0.2177 | +0.021 |

**Disclosed artifact.** The read-only archived driver (`run_interventions_v1.py:287-290`) sums `a.ideal_Merit` over ALL agents, so its `ideal_merit_sum_final` column is NaN post-fix whenever an unscored newborn exists; the finite-part sum reproduces v1 bit-exactly (probe: `finite_part_sum_diff_vs_v1 = 0.0`). The column is excluded from the numeric assertions and must not be used post-fix without the finite-part filter.

## Arm A — birth-tile decomposition (`r_spawn`)

| landscape | c | r_spawn | c_ideal_fin post [95% CI] | pre | c_ideal_tm post | gap_ratio post |
| --- | --- | --- | --- | --- | --- | --- |
| mgstd5 | 0.30 | exact | +0.062 [-0.001, +0.125] | -0.154 | +0.089 | -0.665 |
| mgstd5 | 0.30 | random | -0.009 [-0.095, +0.077] | -0.171 | +0.085 | -0.655 |
| mgstd5 | 0.30 | ring1 | +0.000 [-0.075, +0.076] | -0.187 | +0.080 | -0.659 |
| mgstd5 | 0.50 | exact | +0.077 [-0.017, +0.170] | -0.057 | +0.132 | -0.672 |
| mgstd5 | 0.50 | random | -0.023 [-0.108, +0.061] | -0.145 | +0.162 | -0.676 |
| mgstd5 | 0.50 | ring1 | +0.064 [-0.010, +0.138] | -0.075 | +0.119 | -0.670 |
| mgstd5 | 0.70 | exact | +0.119 [+0.033, +0.206] | +0.028 | +0.139 | -0.675 |
| mgstd5 | 0.70 | random | -0.004 [-0.087, +0.079] | -0.095 | +0.213 | -0.684 |
| mgstd5 | 0.70 | ring1 | +0.081 [+0.006, +0.157] | -0.003 | +0.131 | -0.673 |
| noisypr5 | 0.30 | exact | -0.011 [-0.076, +0.054] | -0.222 | +0.250 | -0.589 |
| noisypr5 | 0.30 | random | -0.064 [-0.173, +0.045] | -0.235 | +0.237 | -0.577 |
| noisypr5 | 0.30 | ring1 | +0.003 [-0.074, +0.079] | -0.207 | +0.254 | -0.593 |
| noisypr5 | 0.50 | exact | +0.082 [+0.017, +0.146] | -0.124 | +0.367 | -0.538 |
| noisypr5 | 0.50 | random | -0.060 [-0.155, +0.035] | -0.233 | +0.431 | -0.572 |
| noisypr5 | 0.50 | ring1 | -0.042 [-0.123, +0.039] | -0.224 | +0.371 | -0.537 |
| noisypr5 | 0.70 | exact | -0.002 [-0.061, +0.057] | -0.205 | +0.439 | -0.523 |
| noisypr5 | 0.70 | random | +0.066 [-0.038, +0.171] | -0.162 | +0.530 | -0.566 |
| noisypr5 | 0.70 | ring1 | +0.019 [-0.071, +0.110] | -0.220 | +0.441 | -0.527 |

Ordering verdicts (final-step ideal correlation, post-fix means):

| landscape | c | post order | pre order | exact top? |
| --- | --- | --- | --- | --- |
| mgstd5 | 0.30 | exact > ring1 > random | exact > random > ring1 | yes |
| mgstd5 | 0.50 | exact > ring1 > random | exact > ring1 > random | yes |
| mgstd5 | 0.70 | exact > ring1 > random | exact > ring1 > random | yes |
| noisypr5 | 0.30 | ring1 > exact > random | ring1 > exact > random | no |
| noisypr5 | 0.50 | exact > ring1 > random | exact > ring1 > random | yes |
| noisypr5 | 0.70 | random > ring1 > exact | random > exact > ring1 | no |

## Arm B — sabbaticals vs lotteries (Panel D post-fix)

| alpha | p_reloc | xi | landscape | Panel D post [95% CI] | pre | A/B invariance | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0.5 | mgstd5 | +0.135 [+0.066, +0.205] | +0.111 | yes |  |
| 0 | 0 | 0.5 | noisypr5 | +0.423 [+0.385, +0.462] | +0.405 | yes |  |
| 0 | 0 | 0.7 | mgstd5 | +0.116 [+0.053, +0.178] | +0.081 | yes |  |
| 0 | 0 | 0.7 | noisypr5 | +0.277 [+0.247, +0.308] | +0.251 | yes |  |
| 0 | 0 | 0.9 | mgstd5 | +0.040 [+0.037, +0.042] | +0.001 | yes |  |
| 0 | 0 | 0.9 | noisypr5 | +0.073 [+0.060, +0.086] | +0.037 | yes |  |
| 0 | 0.02 | 0.5 | mgstd5 | +0.080 [+0.064, +0.095] | +0.068 | yes |  |
| 0 | 0.02 | 0.5 | noisypr5 | +0.338 [+0.303, +0.372] | +0.322 | yes |  |
| 0 | 0.02 | 0.7 | mgstd5 | +0.076 [+0.062, +0.091] | +0.058 | yes |  |
| 0 | 0.02 | 0.7 | noisypr5 | +0.274 [+0.246, +0.302] | +0.254 | yes |  |
| 0 | 0.02 | 0.9 | mgstd5 | +0.085 [+0.075, +0.095] | +0.054 | yes |  |
| 0 | 0.02 | 0.9 | noisypr5 | +0.112 [+0.096, +0.129] | +0.077 | yes |  |
| 0.5 | 0 | 0.5 | mgstd5 | +0.152 [+0.078, +0.226] | +0.127 | yes |  |
| 0.5 | 0 | 0.5 | noisypr5 | +0.410 [+0.370, +0.450] | +0.391 | yes |  |
| 0.5 | 0 | 0.7 | mgstd5 | +0.112 [+0.054, +0.170] | +0.077 | yes |  |
| 0.5 | 0 | 0.7 | noisypr5 | +0.273 [+0.239, +0.306] | +0.247 | yes |  |
| 0.5 | 0 | 0.9 | mgstd5 | +0.041 [+0.038, +0.044] | +0.004 | yes |  |
| 0.5 | 0 | 0.9 | noisypr5 | +0.080 [+0.064, +0.095] | +0.044 | yes |  |
| 0.5 | 0.02 | 0.5 | mgstd5 | +0.082 [+0.064, +0.100] | +0.071 | yes |  |
| 0.5 | 0.02 | 0.5 | noisypr5 | +0.327 [+0.307, +0.346] | +0.311 | yes |  |
| 0.5 | 0.02 | 0.7 | mgstd5 | +0.068 [+0.055, +0.081] | +0.049 | yes |  |
| 0.5 | 0.02 | 0.7 | noisypr5 | +0.264 [+0.240, +0.287] | +0.244 | yes |  |
| 0.5 | 0.02 | 0.9 | mgstd5 | +0.078 [+0.069, +0.087] | +0.047 | yes |  |
| 0.5 | 0.02 | 0.9 | noisypr5 | +0.119 [+0.101, +0.138] | +0.083 | yes |  |
| 1 | 0 | 0.5 | mgstd5 | +0.132 [+0.074, +0.190] | +0.106 | yes |  |
| 1 | 0 | 0.5 | noisypr5 | +0.367 [+0.340, +0.394] | +0.348 | yes |  |
| 1 | 0 | 0.7 | mgstd5 | +0.089 [+0.054, +0.124] | +0.056 | yes |  |
| 1 | 0 | 0.7 | noisypr5 | +0.250 [+0.225, +0.275] | +0.224 | yes |  |
| 1 | 0 | 0.9 | mgstd5 | +0.041 [+0.038, +0.043] | +0.003 | yes |  |
| 1 | 0 | 0.9 | noisypr5 | +0.077 [+0.062, +0.093] | +0.041 | yes |  |
| 1 | 0.02 | 0.5 | mgstd5 | +0.077 [+0.056, +0.099] | +0.066 | yes |  |
| 1 | 0.02 | 0.5 | noisypr5 | +0.315 [+0.291, +0.339] | +0.300 | yes |  |
| 1 | 0.02 | 0.7 | mgstd5 | +0.078 [+0.064, +0.092] | +0.058 | yes |  |
| 1 | 0.02 | 0.7 | noisypr5 | +0.271 [+0.251, +0.291] | +0.251 | yes |  |
| 1 | 0.02 | 0.9 | mgstd5 | +0.074 [+0.063, +0.084] | +0.042 | yes | rescue_pp=+20.22 |
| 1 | 0.02 | 0.9 | noisypr5 | +0.112 [+0.095, +0.130] | +0.077 | yes |  |

## Arm C — paradigm refill regimes (M1, mgstd5, g = 0.1)

| regime | terminal post [CI] | terminal pre | time-mean post [CI] | time-mean pre |
| --- | --- | --- | --- | --- |
| frontier | +0.280 [+0.236, +0.325] | +0.078 | +0.432 [+0.417, +0.447] | +0.401 |
| occupied | +0.080 [+0.041, +0.120] | -0.045 | +0.256 [+0.247, +0.265] | +0.239 |
| off | +0.086 [+0.039, +0.133] | -0.061 | +0.120 [+0.090, +0.150] | +0.094 |

Post-fix, the regimes with a positive terminal correlation are: frontier, occupied, off. The highest time-mean correlation belongs to **frontier**.

## Typology — destruction invariance and the post-fix wedge

`ideal_loss_cum` destruction main effect post-fix: -0.0783 [-0.0977, -0.0598] over 240 pairs (pre-fix `-0.0783`, i.e. -16.3% of the Pure mean — invariance is proven by the bit-identical `ideal_loss_cum` rows in the ledger above).

| regime | n pairs | mean Δr(γ=0) | mean Δr(γ=1) | narrowing | 95% CI |
| --- | --- | --- | --- | --- | --- |
| M0 | 120 | +0.000 | -0.014 | -0.014 | [-0.049, +0.020] |
| M1 | 120 | +0.001 | -0.019 | -0.020 | [-0.046, +0.005] |

Wedge sign note: `Δr = c_local − c_ideal = −wedge_fin`; positive Δr means the local proxy outranks global merit (proxy failure).

## Path B — post-fix alignment by movement pull

| pull | lambda | landscape | c_ideal_tm [CI] | wedge_tm [CI] |
| --- | --- | --- | --- | --- |
| 0 | 0 | mgstd5 | +0.108 [+0.085, +0.130] | -0.003 [-0.006, -0.001] |
| 0 | 0 | noisypr5 | +0.389 [+0.372, +0.406] | +0.008 [+0.004, +0.012] |
| 0 | 0.25 | mgstd5 | +0.109 [+0.087, +0.132] | -0.005 [-0.008, -0.001] |
| 0 | 0.25 | noisypr5 | +0.389 [+0.373, +0.406] | +0.007 [+0.003, +0.012] |
| 0 | 0.5 | mgstd5 | +0.108 [+0.085, +0.130] | -0.005 [-0.008, -0.002] |
| 0 | 0.5 | noisypr5 | +0.391 [+0.375, +0.407] | +0.006 [+0.002, +0.011] |
| 0 | 0.75 | mgstd5 | +0.108 [+0.085, +0.130] | -0.004 [-0.007, -0.002] |
| 0 | 0.75 | noisypr5 | +0.391 [+0.375, +0.407] | +0.006 [+0.002, +0.010] |
| 0 | 1 | mgstd5 | +0.108 [+0.086, +0.131] | -0.004 [-0.007, -0.001] |
| 0 | 1 | noisypr5 | +0.391 [+0.374, +0.407] | +0.007 [+0.003, +0.012] |
| 0.5 | 0 | mgstd5 | +0.067 [+0.042, +0.092] | +0.015 [+0.011, +0.018] |
| 0.5 | 0 | noisypr5 | +0.316 [+0.295, +0.336] | +0.062 [+0.054, +0.069] |
| 0.5 | 0.5 | mgstd5 | +0.067 [+0.042, +0.091] | +0.031 [+0.026, +0.037] |
| 0.5 | 0.5 | noisypr5 | +0.300 [+0.284, +0.317] | +0.083 [+0.078, +0.089] |
| 0.5 | 1 | mgstd5 | +0.176 [+0.156, +0.195] | -0.050 [-0.056, -0.043] |
| 0.5 | 1 | noisypr5 | +0.411 [+0.394, +0.428] | -0.006 [-0.011, -0.001] |
| 1 | 0 | mgstd5 | +0.060 [+0.034, +0.086] | +0.013 [+0.009, +0.017] |
| 1 | 0 | noisypr5 | +0.299 [+0.276, +0.322] | +0.068 [+0.060, +0.076] |
| 1 | 0.5 | mgstd5 | +0.047 [+0.022, +0.071] | +0.042 [+0.036, +0.049] |
| 1 | 0.5 | noisypr5 | +0.265 [+0.248, +0.282] | +0.105 [+0.099, +0.111] |
| 1 | 1 | mgstd5 | +0.192 [+0.174, +0.211] | -0.061 [-0.069, -0.053] |
| 1 | 1 | noisypr5 | +0.412 [+0.396, +0.429] | +0.001 [-0.005, +0.007] |

## P-0011 — M0 + M2 without demographic entry (g = 0)

| arm | landscape | team_tm [CI] | links_tm [CI] | c_ideal_tm [CI] |
| --- | --- | --- | --- | --- |
| M0+M2 lam=0 | mgstd5 | 8.24 [6.04, 10.44] | 0.00 [0.00, 0.00] | +0.284 [+0.151, +0.416] |
| M0+M2 lam=0 | noisypr5 | 3.80 [3.30, 4.29] | 0.00 [0.00, 0.00] | +0.587 [+0.537, +0.637] |
| M0+M2 lam=1 | mgstd5 | 8.24 [6.04, 10.44] | 0.00 [0.00, 0.00] | +0.284 [+0.151, +0.416] |
| M0+M2 lam=1 | noisypr5 | 3.80 [3.30, 4.29] | 0.00 [0.00, 0.00] | +0.587 [+0.537, +0.637] |
| M0 dyadic=off | mgstd5 | 8.40 [6.15, 10.64] | 0.00 [0.00, 0.00] | +0.283 [+0.151, +0.415] |
| M0 dyadic=off | noisypr5 | 3.81 [3.24, 4.38] | 0.00 [0.00, 0.00] | +0.595 [+0.541, +0.648] |

## Provenance

- Wave-2 cells call the ARCHIVED round `run_one` entry points unchanged (`scripts/interventions`, `scripts/typology`, `scripts/path_b`), so grids, seeds and RNG streams are inherited; the only difference is the model code.
- Namespaces: `data/interventions_v2/`, `data/typology_v2/`, `data/path_b_postbug/`; manifests carry `git_head`, `sources_moved_during_sweep`, gross/net counts and seed lists.
- Paired CIs: 95% bootstrap (B = 20,000, seed 20261115) for deltas; run-level mean ± 1.96·SE for levels.
