---
title: "ADR-0003 re-run wave: directional robustness against the pre-ADR archives"
version: 1
date: 2026-11-26
revision-vehicle: strikethrough
revision-renders: "md, pdf, html"
tags: [adr-0003, time-scale, robustness, re-run, turnover, proxy-failure, not-calibrated]
project: sciencemodels
---

**Version 1** (November 2026)

> **STATUS: MODEL-INTERNAL, NOT CALIBRATED** (owner ruling, 2026-11-13).
> Compact re-run of the core Phase-1 sweeps on the ADR-0003 6-month clock:
> 319 cells, 6,380 runs, 20 seeds per cell, new `data/*_adr0003/` archives.
> No pre-ADR archive is overwritten. **No model default is changed in this
> round**; the ADR-0003 core landed in `8b36a72` (plus the retirement-link
> guard used when Path B and turnover meet).

**Verdict: the directional spine survives, with three disclosed qualifications.**
Sixteen of nineteen directional contrasts keep their pre-ADR sign or ordering.
The exceptions are (i) the welfare entry gradient of cumulative ideal loss,
which flips sign under fixed-span replacement; (ii) the career-flow per-capita
harvest gradient under `fixed_span`, which is flat rather than negative (it
remains negative under `age_hazard_staggered`); and (iii) the prestige-stock
wedge, whose *ordering* is preserved but whose level moves from slightly
positive to slightly negative for all three stocks. These are clock/exit-rule
effects, not measurement noise: the run is fully seeded and the new cells are
checkpointed and schema-checked.

# Design and gates

The wave keeps each archive's signature axis and applies the ADR-0003 clock:
1 step = 0.5 year; entry age 25; `tau_max = 80`; gamma = `0.5**(1/10)` =
0.933; `g in {0.125, 0.25, 0.5}`; `exit_mode in {fixed_span,
age_hazard_staggered}`. `H = 200` for the core arms and `H = 400` for turnover
and the Figure 1 M1 tiers. Seeded smoke gate and a two-seed-per-arm probe run
before the grid; every cell is written atomically (CSV + state JSON) and
verified on restart. The full run had **0 failed seeds and 0 crashed cells**
(wall 1,356.5 s, `n_jobs = 10`, sources unchanged during the run).

| Arm | Cells | Runs | Horizon | Retained axis |
| --- | ---: | ---: | --- | --- |
| `sensitivity` (T1) | 18 | 360 | 200 | c x landscape x g |
| `welfare` (W_v05) | 60 | 1,200 | 200 | c x visibility reward x landscape leg x g |
| `career_flows` | 18 | 360 | 200 | c x g x exit mode, mgstd5, career measures |
| `prestige` | 54 | 1,080 | 200 | stock x c x landscape x g |
| `turnover` | 36 | 720 | 400 | c x landscape x g x exit mode |
| `interventions` | 60 | 1,200 | 200 | spawn / sabbatical / refill compact sub-arms |
| `path_b` | 18 | 360 | 200 | lambda x pull x g, logit + equal_split, mgstd5 |
| `fig1_xi` | 55 | 1,100 | 200/400 | dense xi x five tiers |
| **Total** | **319** | **6,380** | --- | --- |

**Scope reductions, disclosed.** This is a compact core panel, not a full
Cartesian replay. The interventions lottery (`alpha`) arm and the full 48-cell
Path B formulation/credit cross are not re-run; Path B uses the flagship
`logit + equal_split` configuration. The career arm keeps `mgstd5`; the
welfare arm keeps both visibility levels and both landscape legs. Every
reduction is recorded in `data/adr0003_remeasure/design.json`.

# Before/after ledger

Pre-ADR reference values are computed from the archived CSVs named in the
table; post values are computed from `data/*_adr0003/cells/`. The comparison is
**directional** (sign or ordering), not a level test: changing gamma, g and the
exit rule deliberately moves levels.

| Archive | Contrast | Pre | Post | Verdict |
| --- | --- | ---: | ---: | --- |
| T1 | `c_ideal_tm`: high g minus low g | -0.197 | -0.076 | robust |
| W_v05 | corr(xi, `ideal_loss_cum`) | +0.740 | +0.684 | robust |
| W_v05 | corr(xi, `harvested_fin`) | -0.519 | -0.691 | robust |
| W_v05 | `ideal_loss_cum`: high g minus low g | +0.166 | -0.176 | **flip** |
| career | per-capita harvest: high g minus low g (`fixed_span`) | -0.0093 | +0.0003 | **flip** |
| career | per-capita harvest: high g minus low g (`staggered`) | -0.0093 | -0.0092 | robust |
| prestige | time-mean wedge ordering | prestige > visibility > vanishing | prestige > visibility > vanishing | ordering robust |
| prestige | time-mean wedge level: prestige minus visibility | +0.00064 | +0.00023 | robust |
| turnover | `c_ideal_late`: fixed_span minus staggered | +0.0279 | +0.0187 | robust |
| interventions | `explored_fin`: occupied minus off | -0.319 | -0.297 | robust |
| interventions | `explored_fin`: frontier minus off | -0.045 | -0.094 | robust |
| interventions | `explored_fin`: random spawn minus exact | +0.082 | +0.254 | robust |
| interventions | `explored_fin`: relocation 0.02 minus 0 | +0.448 | +0.528 | robust |
| path_b | team size: pull=1 minus pull=0 (lambda=1) | -0.707 | -4.332 | robust |
| fig1_xi | corr(xi, `harvested_fin`) [M0] | -0.539 | -0.666 | robust |
| fig1_xi | corr(xi, `harvested_fin`) [M1 fixed] | -0.539 | -0.622 | robust |
| fig1_xi | corr(xi, `harvested_fin`) [M1 staggered] | -0.539 | -0.455 | robust |
| fig1_xi | corr(xi, `harvested_fin`) [M2 lambda=1] | -0.539 | -0.696 | robust |
| fig1_xi | corr(xi, `harvested_fin`) [M2 lambda=0] | -0.539 | -0.657 | robust |

Sources: pre values from `data/remeasure_v1/{runs,w_v05_runs}.csv`,
`data/career_flows_v1/careerB_analysis.csv`,
`data/prestige_stock_v2/summary_cell.csv`,
`data/turnover_v2/runs.csv`, `data/interventions_v2/cells/` and
`data/path_b_v2/cells/`; post values from the new
`data/<arm>_adr0003/cells/` archives. The machine-readable ledger is
`data/adr0003_remeasure/ledger.csv`.

# What held, and what moved

**T1 and the visibility gradient.** Entry still lowers time-mean ideal
alignment: pre `c_ideal_tm` falls from 0.410 (g=0) to 0.213 (g=0.1); on the new
clock it falls from 0.223 (g=0.125) to 0.147 (g=0.5). The visibility gradient
is stronger, not weaker: corr(xi, harvest) moves from -0.519 pre to -0.691
post, and corr(xi, loss) stays positive (+0.740 to +0.684). The first
qualification is the **entry gradient of cumulative loss**: pre-ADR entry
*raised* mean `ideal_loss_cum` (+0.166 from g=0 to g=0.1); under fixed-span
replacement it *falls* (-0.176 from g=0.125 to g=0.5). The likely mechanism is
demographic: fixed-span replacement removes old agents instead of accumulating
them, so higher entry no longer means a larger stock of long-lived depleted
careers. The directional claim "entry worsens alignment" survives; the claim
"entry raises cumulative loss" does not transfer mechanically.

**Career flows.** Under staggered turnover the pre-ADR entry gradient in
per-capita harvest survives: 0.0227 (g=0.05) to 0.0134 (g=0.20) pre;
0.0168 (g=0.125) to 0.0076 (g=0.5) post. Under `fixed_span` the gradient is
flat (0.0321 to 0.0324) because the population is pinned at N=20 and the
oldest-out rule removes exactly one agent per arrival. The `is_newcomer`
prestige gap is structurally NaN once founders retire (all survivors carry the
permanent newcomer flag); the core `gap_recent` reporter is the replacement
measure and is finite in the new archives.

**Prestige stocks.** The time-mean wedge ordering is exactly preserved:
`prestige` > `prestige_visibility` > `prestige_vanishing`. But all three
levels move from slightly positive to slightly negative (pre +0.0063, +0.0056,
+0.0053; post -0.0067, -0.0069, -0.0071). The stock choice still does not
reorder the wedge, but the ADR clock/exit combination shifts its sign. This is
a level finding, not a rank reversal.

**Turnover.** Fixed-span replacement still produces better late alignment than
staggered age-hazard replacement: pre +0.0279, post +0.0187 (fixed minus
staggered). The staggered arm's population grows because scheduled entry and
one-for-one retirement replacement both operate; the comparison is therefore
regime-level, not same-N. `top10_late` is 0.205 (fixed) versus 0.235
(staggered), so the concentration ordering is also stable.

**Interventions.** The paradigm-refill result is unchanged in direction:
in-place `occupied` refill collapses exploration (-0.297 versus off),
`frontier` refill also reduces exploration relative to off (-0.094) but much
less than `occupied`; random spawn still expands exploration (+0.254 versus
exact), and sabbatical relocation still expands it (+0.528). The refill arms
inject stock, so their `harvested_fin` is not a conservation measure and is not
used in the ledger.

**Path B.** The movement-pull result is robust: at lambda=1, team size is 8.50
at pull=0 and 4.17 at pull=1 (difference -4.33); the pre-ADR pull gradient
(pull=0.5 to 1.0) was -0.71. The archived grid has no pull=0 cell, so the
pre-ADR reference uses the nearest available pull. Link counts still rise with
lambda (2.44 to 15.34 at pull=0), and the pull still reduces alignment at
every lambda, though least at lambda=1.

**Figure 1 dense xi.** All five tiers keep the negative xi-to-harvest gradient.
At xi=0.1/0.5/0.9, M1 fixed harvest is 0.829/0.825/0.493 and M1 staggered is
0.980/0.979/0.868; M0 is 0.665/0.657/0.327, M2 lambda=1 is
0.661/0.648/0.296, and M2 lambda=0 is 0.663/0.651/0.329. The dense grid thus
sustains the two-stage shape: flat to xi=0.5, then a collapse at the high-
visibility end. One exception is visible in the ideal channel: the M2
lambda=0 tier has a *positive* xi-to-`c_ideal_tm` correlation (+0.304) while
the other tiers are negative; that tier's ideal alignment is already negative
at low xi and rises toward zero at high xi. The harvest gradient, which is the
Figure 1 panel (a) claim, is unaffected.

# Caveats

- This is a compact core panel, not a complete replay of every historical
  cell; see the scope reductions above.
- Pre-ADR references are at the old clock (gamma=0.9/0.5, g=0/0.1,
  `tau_max=200`); the ledger tests direction, not levels or exact magnitudes.
- `fixed_span` pins N=20, while `age_hazard_staggered` can grow; turnover
  comparisons between those modes are regime-level.
- The 6,380 runs are model-internal and not calibrated (owner ruling
  2026-11-13). ADR-0003 values are narrative scale anchors, not empirical
  fits.
- The dense Figure 1 panel is `mgstd5` only; it is a robustness check on the
  shape, not a landscape-sensitivity claim.

# Provenance

- Runner: `scripts/adr0003_remeasure/run_adr0003_wave.py`
  (`--probe` gate; per-cell `cells/` + `state/`; central `design.json`,
  `manifest.json`, `run_log.txt`, `probe_results.json`).
- Analysis: `scripts/adr0003_remeasure/analyse_adr0003_wave.py`.
- New archives: `data/sensitivity_adr0003/`, `data/welfare_adr0003/`,
  `data/career_flows_adr0003/`, `data/prestige_adr0003/`,
  `data/turnover_adr0003/`, `data/interventions_adr0003/`,
  `data/path_b_adr0003/`, `data/fig1_xi_adr0003/`.
- Ledger: `data/adr0003_remeasure/ledger.csv`; summaries:
  `summary_by_cell.csv`, `summary_by_arm.csv`; headline JSON:
  `analysis.json`.
- Gates: `tests/test_smoke.py` PASS; probe two seeds per arm; 319/319 cells
  and 6,380/6,380 runs completed; source hashes unchanged during the run.
