---
title: "ADR-0003 gap arms: PhD lotteries, the career-noisy leg, and the M2 n=40 lift check (v1)"
version: 1
date: 2026-11-26
revision-vehicle: strikethrough
revision-renders: "md, pdf"
tags: [adr-0003, gaps, lottery, career-flows, path-b, m2, not-calibrated]
project: sciencemodels
---

**Version 1** (26 November 2026)

> **STATUS: MODEL-INTERNAL, NOT CALIBRATED** (owner ruling 2026-11-13).
> Prompt F2 (task/adr0003-missing-arms), worktree `wt-adr0003-gaps`, branch
> `task/adr0003-gaps`: the three cells that D6/D3 left out or underpowered.
> 21 cells, 600 runs, 20 seeds (40 for the M2 arm), 0 failed seeds and 0
> crashed cells; new archives only, no pre-ADR or D6 archive overwritten.
> **No model file was edited**: the one model-side addition is the driver
> subclass `GapModel`, whose roulette copies are bit-identical to `MyModel`
> at `alpha = 1` (probed, 41 columns, max diff 0.0).

**Verdict.** Three answers.

**(i) PhD lotteries are inert at the central cell.** At `c = 0.5` the
`alpha` grid moves neither concentration nor alignment significantly: pooled
over relocation regimes, `alpha = 0` versus `1.0` gives `top10_fin`
`+0.0034` `[-0.0047, +0.0115]` and `c_ideal_tm` `+0.0217`
`[-0.0056, +0.0491]`. The only marginal paired effects are a small
cumulative-loss increase at `alpha = 0.5` without relocation (`+0.0227`
`[+0.0010, +0.0443]`, `p = 0.041`) and a small harvest decrease for
`alpha = 0.5` pooled over regimes (`-0.0096` `[-0.0189, -0.0003]`). The
paper's high-visibility lottery claim lives at `c = 0.1` and is not
testable in this compact panel.

**(ii) The career-noisy leg keeps the D6 ordering, not the D6 levels.**
Per-capita harvest falls with entry under staggered turnover (`-0.0033`
`[-0.0034, -0.0032]`, high minus low `g`) and weakly under `fixed_span`
(`-0.0007` `[-0.0009, -0.0004]`). Under `fixed_span` the gradient is thus
not flat (the D6 `mgstd5` reading) but 13 times smaller than the pre-ADR
value (`-0.0093`); on noisy terrain the leg sides with the staggered,
still-negative branch of the D6 flip.

**(iii) The M2 `lambda = 1` lift does not exist on the ADR-0003 clock; it
clears 95% as a drop.** Paired over 40 seeds, `mgstd5` `lambda = 1`,
`pull = 1`, `logit` sits `-0.0504` `[-0.0868, -0.0140]` (structural) and
`-0.0404` `[-0.0777, -0.0030]` (`equal_split`) below M1 on time-mean
alignment. The seeds-0-19 subset already reads `-0.0433` `[-0.0909,
+0.0044]` and `-0.0336` `[-0.0834, +0.0163]`, so the sign reversal against
D3's old-clock `+0.0652` `[+0.0023, +0.1282]` is the clock/exit change;
`n = 40` supplies the precision that clears zero. The `lambda` ordering
survives (`lambda = 1` 0.086 versus `lambda = 0` `-0.069` at `pull = 1`);
the level sign does not. On the terminal metric the `lambda = 1` arms are
flat versus M1 (`+0.0052` and `+0.0214`, both ns): the drop is a time-mean
channel effect.

# Design

The ADR-0003 clock throughout: 1 step = 0.5 year, gamma `0.5**(1/10)` =
0.933, `tau_max = 80`, entry age 25, `H = 200`, `g = 0.25` unless swept.
Row schema, parameter block and summariser are imported from the D6 runner
(`scripts/adr0003_remeasure/run_adr0003_wave.py`), so the new M2 cells are
bit-comparable with the D6 archives.

| Arm | Grid | Seeds | Cells | Runs |
| --- | --- | ---: | ---: | ---: |
| `lottery` | `alpha` {1.0, 0.5, 0.0} x `p_reloc` {0.0, 0.02}, `mgstd5`, `c = 0.5`, `fixed_span` | 20 | 6 | 120 |
| `career_noisy` | `g` {0.125, 0.25, 0.5} x {`fixed_span`, `age_hazard_staggered`}, `noisypr5`, `c = 0.5` | 20 | 6 | 120 |
| `m2_n40` | M1 baseline + `lambda` {0, 1} x `pull` {0, 1} x credit {`structural`, `equal_split`}, `logit`, `mgstd5`, `fixed_span` | 40 | 9 | 360 |

Scope decisions, disclosed: the lottery and career arms keep the D6 central
curiosity (`c = 0.5`) because the prompt fixes the grids to `alpha x
p_reloc` and `g x exit`; the M2 arm keeps the D6 flagship formulation
(`logit`) and adds the credit axis D6 dropped. The sabbatical relocation
rule is the D6 `AdrInterventionModel` rule (one draw per trapped agent per
step, relocation to a random unexplored tile); the lottery replaces the
supervisor weight `entry_prestige_stock` with `entry_prestige_stock **
alpha`, the P-0003.1 mechanic.

# Results

## (a) PhD lottery arm

Cell means over 20 seeds (paired contrasts below):

| `alpha`, `p_reloc` | `top10_fin` | `ideal_loss_cum` | `c_ideal_tm` | `harvested_fin` |
| --- | ---: | ---: | ---: | ---: |
| 1.0, 0.00 | 0.2164 | 0.0368 | 0.1002 | 0.6580 |
| 0.5, 0.00 | 0.2413 | 0.0594 | 0.1402 | 0.6383 |
| 0.0, 0.00 | 0.2281 | 0.0471 | 0.1411 | 0.6478 |
| 1.0, 0.02 | 0.2174 | 0.0254 | 0.0974 | 0.6451 |
| 0.5, 0.02 | 0.2155 | 0.0241 | 0.0837 | 0.6456 |
| 0.0, 0.02 | 0.2125 | 0.0244 | 0.0999 | 0.6456 |

Paired differences against `alpha = 1.0` on the common seed grid (95% t
intervals). Without relocation the lottery shows a weakly *positive*
cumulative-loss difference, i.e. slightly worse epistemic outcomes, not
better; with relocation (`p_reloc = 0.02`) everything collapses to zero:

| Contrast | `top10_fin` | `ideal_loss_cum` | `c_ideal_tm` |
| --- | --- | --- | --- |
| `alpha = 0.5`, `p_reloc = 0` | +0.0250 [-0.005, +0.055] | +0.0227 [+0.001, +0.044] | +0.0400 [-0.015, +0.095] |
| `alpha = 0.0`, `p_reloc = 0` | +0.0118 [-0.003, +0.026] | +0.0103 [-0.006, +0.026] | +0.0409 [-0.002, +0.084] |
| `alpha = 0.5`, `p_reloc = 0.02` | -0.0019 [-0.008, +0.004] | -0.0014 [-0.004, +0.001] | -0.0137 [-0.041, +0.013] |
| `alpha = 0.0`, `p_reloc = 0.02` | -0.0049 [-0.012, +0.002] | -0.0010 [-0.0045, +0.0024] | +0.0025 [-0.033, +0.038] |
| `alpha = 0.5`, pooled | +0.0115 [-0.004, +0.027] | +0.0106 [-0.0005, +0.0218] | +0.0131 [-0.017, +0.044] |
| `alpha = 0.0`, pooled | +0.0034 [-0.005, +0.012] | +0.0046 [-0.003, +0.013] | +0.0217 [-0.006, +0.049] |

## (b) career_flows noisypr5 leg

Cell means over 20 seeds; `career_gap_final` is structurally NaN (D6
disclosed: the permanent newcomer flag voids the `is_newcomer` gap once
founders retire) and `career_gap_ratio_mean_late` collapses to 0.0 in five
of six cells, so neither is usable here; per-capita harvest and alignment
are the readable outcomes:

| Exit | `g` | per-cap. harvest | `c_ideal_tm` |
| --- | ---: | ---: | ---: |
| `fixed_span` | 0.125 | 0.0138 | 0.3393 |
| `fixed_span` | 0.25 | 0.0135 | 0.2598 |
| `fixed_span` | 0.5 | 0.0131 | 0.1777 |
| `age_hazard_staggered` | 0.125 | 0.0083 | 0.2426 |
| `age_hazard_staggered` | 0.25 | 0.0067 | 0.2244 |
| `age_hazard_staggered` | 0.5 | 0.0050 | 0.1894 |

Paired high-minus-low entry gradients (g = 0.5 minus g = 0.125, 20 seeds):

| Exit | per-cap. harvest | `c_ideal_tm` |
| --- | --- | --- |
| `fixed_span` | -0.0007 [-0.0009, -0.0004] | -0.1617 [-0.1995, -0.1238] |
| `age_hazard_staggered` | -0.0033 [-0.0034, -0.0032] | -0.0531 [-0.0760, -0.0303] |

## (c) M2 panel at n = 40

Cell means (40 seeds). `links_tm` shows the declared-link gradient
(`lambda`: 2.5 to 13.6 at pull = 0); `team_tm` shows the co-movement
collapse (7.4 at M1 and pull = 0 down to 3.5-4.7 at pull = 1):

| Cell | `c_ideal_tm` | `c_ideal_fin` | `links_tm` | `team_tm` |
| --- | ---: | ---: | ---: | ---: |
| M1 (dyadic off) | 0.1264 | 0.0818 | 0.00 | 7.43 |
| `lambda=0`, pull=0, structural | 0.1236 | 0.1040 | 2.53 | 7.25 |
| `lambda=0`, pull=0, equal | 0.1254 | 0.1350 | 2.53 | 7.00 |
| `lambda=0`, pull=1, structural | -0.0649 | -0.1851 | 2.58 | 4.64 |
| `lambda=0`, pull=1, equal | -0.0687 | -0.1535 | 2.57 | 4.70 |
| `lambda=1`, pull=0, structural | 0.1236 | 0.1040 | 13.57 | 7.25 |
| `lambda=1`, pull=0, equal | 0.1243 | 0.1190 | 13.15 | 7.04 |
| `lambda=1`, pull=1, structural | 0.0760 | 0.0870 | 10.54 | 3.71 |
| `lambda=1`, pull=1, equal | 0.0860 | 0.1032 | 10.07 | 3.46 |

**The lift test.** Lift = cell minus M1 on the common 40-seed grid,
paired t interval (primary) and the unpaired Welch twin:

| Cell (vs M1) | `c_ideal_tm` lift | 95% paired CI | Welch CI | seeds 0-19 |
| --- | ---: | --- | --- | --- |
| `lambda=1`, pull=1, structural | -0.0504 | [-0.0868, -0.0140] | [-0.0992, -0.0016] | -0.0433 [-0.0909, +0.0044] |
| `lambda=1`, pull=1, equal | -0.0404 | [-0.0777, -0.0030] | [-0.0892, +0.0085] | -0.0336 [-0.0834, +0.0163] |
| `lambda=0`, pull=1, structural | -0.1913 | [-0.2390, -0.1436] | [-0.2575, -0.1251] | -0.2043 [-0.2757, -0.1330] |
| `lambda=0`, pull=1, equal | -0.1951 | [-0.2427, -0.1474] | [-0.2612, -0.1290] | -0.2053 [-0.2770, -0.1335] |
| `lambda=1`, pull=0, structural | -0.0028 | [-0.0396, +0.0341] | [-0.0533, +0.0478] | +0.0061 [-0.0425, +0.0547] |

On the terminal metric (`c_ideal_fin`) the picture is flat at the top:
`lambda=1`, pull=1 gives `+0.0052 [-0.1028, +0.1133]` (structural) and
`+0.0214 [-0.0821, +0.1249]` (equal), both ns; the `lambda=0` drop is
larger there too (`-0.2669` and `-0.2353`, both clearing zero).

**The chain at n = 40** (D3 definitions: gap = M1 minus full M2; shares of
that gap, seed-paired bootstrap intervals, 10,000 draws):

| `lambda` | gap [95% CI] | link share | move share | credit share |
| --- | --- | --- | --- | --- |
| 0 | +0.1951 [+0.1483, +0.2405] | +1.4 [-20.4, +16.7] | +96.6 [+79.8, +121.1] | +1.9 [-8.4, +11.4] |
| 1 | +0.0404 [+0.0061, +0.0765] | +6.8 [-304.1, +60.2] | +117.9 [+53.6, +557.9] | -24.8 [-180.9, +12.9] |

The structure is D3's: co-movement carries the change, nominal links are
inert (the pull = 0 leg sits at `-0.0028` with a zero-crossing interval),
credit is second-order. What changed with the clock is only the sign of the
`lambda = 1` gap: positive (a drop) here, negative (a lift) at D3's old
clock, where the paired interval already touched zero.

# Verification

- Counts: 21/21 cells, 600/600 runs, 0 failed seeds, sources unchanged
  during the run (manifest). Checkpointed per cell (CSV + state JSON);
  a restart skips verified cells.
- Probe gate before the grid: `tests/test_smoke.py` PASS;
  `GapModel(alpha = 1, p_reloc = 0)` bit-identical to `MyModel` on two
  seeds (41 numeric columns, max diff 0.0); two seeds per arm schema-checked.
- Cross-archive canaries (bit-identity on shared seeds, 41 numeric
  columns each): the four `equal_split` M2 cells at seeds 0-19 reproduce
  the D6 `path_b_adr0003` cells exactly; `lottery alpha = 1` reproduces the
  D6 `interventions_adr0003` `intB` cells exactly (both relocation regimes).
  Max abs difference 0.0 in all six checks.
- The seeds-0-19 subset lifts are computed from the same archive and
  reported above to separate the clock change from the precision gain.
- Ledger numbers quoted from `data/adr0003_gaps/*.csv` (written by
  `analyse_adr0003_gaps.py`); the old-clock reference
  recomputed from `data/path_b_postbug/` (logit + equal_split cell).

# Caveats

- Compact panel: the lottery and career arms are central-curiosity slices
  (`c = 0.5`); no claim about the visibility axis.
- The `n = 40` M2 arm is `mgstd5` + `logit` only; the credit axis is
  complete but the formulation twin is not re-run.
- Clock effects and exit effects are entangled by design (the ADR-0003
  clock is the package: gamma, `tau_max`, entry age, `g = 0.25`,
  `fixed_span`); the report attributes the sign reversal to the package,
  not to a single component.
- `career_gap_final` is NaN and `career_gap_ratio_mean_late` degenerate
  post-retirement (D6's disclosed measurement boundary); career readings
  rest on per-capita harvest, flow and alignment.
- Paired contrasts are primary because the design shares the agent-seed
  grid; the Welch intervals are reported as the unpaired twin, and for
  `equal_split` at `pull = 1` the `lambda = 1` verdict is paired-only
  (Welch `p = 0.104`).
- All runs are model-internal and not calibrated.

# Provenance

- Runner: `scripts/adr0003_gaps/run_adr0003_gaps.py` (`--probe` gate;
  per-cell `cells/` + `state/` under
  `data/interventions_lottery_adr0003/`,
  `data/career_flows_noisy_adr0003/`, `data/path_b_m2_n40_adr0003/`;
  central `data/adr0003_gaps/` with `design.json`, `manifest.json`,
  `run_log.txt`, `probe_results.json`).
- Analysis: `analyse_adr0003_gaps.py` (in `scripts/adr0003_gaps/`)
  writes nine canonical tables under data/adr0003_gaps/: summary_lottery,
  contrasts_lottery, summary_career_noisy, gradients_career_noisy,
  summary_m2_n40, lift_tests, chain_m2_n40, crosscheck (all .csv) and
  analysis (.json).
- Worktree `wt-adr0003-gaps` on branch `task/adr0003-gaps`; `data/`
  reached the shared archive tree through per-directory links (the
  literal `ln -s` instruction lands at `data/data` because the checkout
  materialises tracked figure files under `data/`; disclosed in the log
  summary).
- No model file edited; no archive overwritten; nothing pushed.