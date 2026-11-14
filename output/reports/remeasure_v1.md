---
title: "Post-Audit Re-Measure v1: What the Ideal-Merit Pairing Fix Did and Did Not Move"
author: "Hermes Agent (squad lead) for Jakob Kapeller"
version: 1
date: 2026-11-13
tags: [remeasure, pairing-fix, wedge, welfare, breakpoint, abm, proxy-failure, not-calibrated]
project: sciencemodels
revision-vehicle: strikethrough
revision-renders: "md, pdf, html"
---

**Version 1** (November 2026)

> **NOT CALIBRATED.** Nothing here is fitted to empirical data (owner ruling
> 2026-11-13). Every number is a property of the simulation. The empirical
> moments in `docs/calibration_moments.yaml` remain a documented option menu,
> never targets.

# What this pass measured

The ideal-merit pairing audit (commit `d3503ac`) changed which merit array the
prestige-quality correlation is computed against. That fix invalidated every
archived correlation that reads the ideal channel, so the load-bearing cells of
waves 1 and 2 were re-run and paired seed-by-seed against their pre-fix
archives.

The question this document answers is narrow and testable: **did the fix move
the ideal channel alone, or did it also move the channels that never touch it?**
A measurement repair that quietly moves unrelated reporters would invalidate the
earlier rounds wholesale; a repair confined to one channel leaves them standing
with a quantified level shift.

Two arms are compared, both paired on every grid key plus the seed:

| Arm | Pre-fix archive | Post-fix re-measure | Pairs |
| --- | --- | --- | --- |
| T1 sensitivity | `data/sensitivity_v1/cell_*.csv` (36 cells, step-level) | `data/remeasure_v1/runs.csv`, origin `T1` | 720 of 720 |
| W_v05 welfare | `data/welfare_v1/cell_*.csv` (40 cells, one row per seed) | `data/remeasure_v1/w_v05_runs.csv` | 800 of 800 |

**Wedge convention.** House convention throughout: $\text{wedge} = c_{local} - c_{ideal}$, positive when prestige tracks the visible proxy better than it tracks quality. This matches `output/reports/sensitivity_map_v1.md` and `output/reports/prestige_stock_v1.md`.

`data/remeasure_v1/runs.csv` stores the **opposite** sign ($c_{ideal} - c_{local}$) because that is its published contract with
`scripts/report_notebook/round_report_data.py`, which asserts the stored sign on
the raw file and flips it in memory. This pass did not touch that file. The
analysis asserts the stored sign on the raw CSV before doing anything, then
recomputes the wedge from the components on both sides, so the comparison never
depends on trusting either file's convention.

# Why the welfare arm was re-run at vanishing 0.5

The archived welfare sweep (`data/welfare_v1`) ran at
`vanishing_factor_for_prestige = 0.5`. The first re-measure pass re-ran the same
40-cell grid at $0.9$ (the value unified in Path 0), so a delta against the
archive would have carried two changes at once --- the pairing fix and the
vanishing unification --- and could not be attributed to either.

The `W_v05` arm therefore reproduces the archived configuration exactly:
vanishing \(0.5\), 5 curiosity values $\times$ 2 visibility toggles $\times$ 2
entry rates $\times$ 2 landscape legibilities, 20 seeds, horizon 200. The legacy
$W$ arm at $0.9$ stays on disk as a bonus sensitivity bridge; it is **not**
paired, by design.

# Write contract preserved

`data/remeasure_v1/runs.csv` and `data/remeasure_v1/cell_summary.csv` are read
by `scripts/report_notebook/round_report_data.py`, which asserts
`len(runs.csv) == 1520`, that `wedge_fin == c_ideal_fin - c_local_fin` on the
raw file, and that the $W$ arm's vanishing value $[0.9]$ differs from the
archive's $[0.5]$.

Both files were left byte-untouched. Verified after the run: `runs.csv` still
carries 1,520 rows, still satisfies `wedge_fin == c_ideal_fin - c_local_fin`,
and both files keep their 08:40 modification time.

The new arm writes only to its own namespace:

- `data/remeasure_v1/cells_w_v05/` --- 40 per-cell checkpoints, atomic
- `data/remeasure_v1/w_v05_runs.csv` --- 800 per-seed rows
- `data/remeasure_v1/w_v05_summary.csv` --- 40 cell means with `n_seeds`
- `data/remeasure_v1/w_v05_manifest.json` --- design, counts, conventions
- `data/remeasure_v1/w_v05_run_log.txt` --- one line per completed cell

The `legacy` arm of the driver now refuses to rebuild over the banked files at
all, so a future from-scratch run cannot silently break the notebook contract by
accident.

# Gross versus net

| Unit | Gross | Net | Wall |
| --- | --- | --- | --- |
| `W_v05` cells | 40 | 40 completed, 0 skipped | 1601.8 s at `n_jobs=3` |
| `W_v05` runs | 800 | 800 with a non-null `c_ideal_fin` | --- |
| T1 pairs (already banked) | 720 | 720 paired one-to-one | --- |
| Audit seeds (companion) | 8 | 5 (control) + 3 (noisypr5) | --- |

The 1601.8 s wall is a shared-core measurement: the container is capped at one
CPU (`cpu.max = 100000 100000`) and another session's Path B sweep held three
workers for part of the window, so the per-run cost here is not a clean
throughput figure.

# Canary (first and last)

Run by the driver before any cell, `MPLBACKEND=Agg`, on the tree that produced
the numbers below:

```text
canary run OK: 56 report keys, harvested_10_step=37.5, harvested_25_step=37.5
same-seed rerun bit-identical OK
newcomer run OK: 6 scientists after 20 steps
4-arg Gaussian OK: np.float64(0.00550470835249002) (float64)
append-mode header check OK: 3 lines = 1 header + 2 data rows
ideal-merit pairing audit OK: A=0.0, C=0.0, B=-0.04999999999999999 (hand-computed 0.0, 0.0, -0.05); cum after 2 calls: A=0.0, B=-0.09999999999999998
seed columns OK: agent_seed=1, landscape_seed=1
binding probe OK: constant generator flagged inert, identity binds
entry_prestige_stock OK: validated, both alternate stocks run
Path B off-mode bit-identity OK (anchor mean_prestige=np.float64(0.025399760810316363)); 0 dynamics keys differ
Path B binding OK: logit lambda0=(2, 1.024390243902439) lambda1=(5, 2.1951219512195124)
Path B binding OK: threshold lambda0=(1, 1.0) lambda1=(3, 2.048780487804878)
Path B knob validation OK: all six knobs reject invalid values
ALL SMOKE TESTS PASSED
```

**The gate grew between rounds.** The re-measure whose output was already banked
recorded 40 report keys; the canary now asserts 56. The added checks are the
Path B off-mode bit-identity and binding probes, `entry_prestige_stock`
validation, and the ideal-merit pairing audit itself (which hand-computes its
expected values --- $A = 0.0$, $C = 0.0$, $B = -0.05$). The gate is green on a
strictly larger set of checks, not a different one.

# Result: one channel moved, three did not

Paired deltas, post-fix minus pre-fix, per seed. "Exact zero" means the
difference is identically `0.0`, not small.

| Reporter | Family | Pairs | Pre mean | Post mean | Mean delta | Max abs delta | Exact zero |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `c_ideal_fin` | ideal side | 720 | 0.043714 | 0.038125 | -0.005588 | 0.611256 | 45 of 720 |
| `c_ideal_tm` | ideal side | 720 | 0.311867 | 0.311505 | -0.000361 | 0.104916 | 0 of 720 |
| `c_local_fin` | local side | 720 | 0.144109 | 0.144109 | 0.000000 | 0.000000 | 720 of 720 |
| `c_local_tm` | local side | 720 | 0.328871 | 0.328871 | 0.000000 | 0.000000 | 720 of 720 |
| `top10_fin` | concentration | 720 | 0.181759 | 0.181759 | 0.000000 | 0.000000 | 720 of 720 |
| `harvested_fin` | harvest | 720 | 0.505655 | 0.505655 | 0.000000 | 0.000000 | 720 of 720 |

The same test on the apples-to-apples welfare arm, which is the one that can
speak to the welfare probe:

| Reporter | Family | Pairs | Pre mean | Post mean | Mean delta | Max abs delta | Exact zero |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `c_ideal_fin` | ideal side | 800 | 0.112903 | 0.111672 | -0.001231 | 0.338018 | 89 of 800 |
| `c_ideal_tm` | ideal side | 800 | 0.261848 | 0.261532 | -0.000316 | 0.105273 | 0 of 800 |
| `c_local_fin` | local side | 800 | 0.186459 | 0.186459 | 0.000000 | 0.000000 | 800 of 800 |
| `c_local_tm` | local side | 800 | 0.266719 | 0.266719 | 0.000000 | 0.000000 | 800 of 800 |
| `top10_fin` | concentration | 800 | 0.166258 | 0.166258 | 0.000000 | 0.000000 | 800 of 800 |
| `harvested_fin` | harvest | 800 | 0.519165 | 0.519165 | 0.000000 | 0.000000 | 800 of 800 |

Two independent arms, 1,520 pairs in total, and every non-ideal reporter is
identically unchanged in every single pair. The fix moved the ideal channel and
nothing else.

A methodological note that cost one iteration and is worth recording: the first
pass reported `c_local_tm` differing by $2.22 \times 10^{-16}$ on 262 of 720
seeds. That was not the model. It was this analysis's own extraction path ---
`groupby(...)[col].mean()` accumulates in a different order than the
`Series.mean()` the model's datacollector uses. Recomputing through the model's
own path restored exact zeros, which is why the claim above is asserted at
`0.0` rather than hidden behind a tolerance.

# Did the fix move the wedge, or only its level?

Levels moved; places did not. Cell-level comparison over the 36 T1 cells:

| Quantity | Value |
| --- | --- |
| Cells | 36 |
| Sign flips (cell mean wedge) | 2 of 36 |
| Spearman rank agreement, pre versus post | 0.9745 |
| Pearson agreement, pre versus post | 0.9856 |
| Cells with negative wedge, pre | 6 |
| Cells with negative wedge, post | 6 |
| Entry cells (`entry > 0`) | 18 |
| Entry cells with negative wedge, pre | 0 |
| Entry cells with negative wedge, post | 0 |
| Entry seeds with negative wedge, pre | 35 of 360 |
| Entry seeds with negative wedge, post | 28 of 360 |
| Mean wedge level, pre | 0.100395 |
| Mean wedge level, post | 0.105984 |

The mean wedge rises by $+0.005588$ --- exactly the amount the ideal-side
correlation fell --- while the ordering is preserved at $\rho = 0.9745$. That is
the "how fast, not where" reading the dossier depends on: the fix changes the
size of the gap, not the parameter regions in which it opens.

The two sign flips are the honest exception and are visible in
`data/remeasure_v1/tables/t1_cell_deltas.csv`; they sit among cells whose wedge
is within noise of zero, not among the load-bearing entry cells.

# The collapse knee survives re-measurement

Re-estimated on both arms with the same estimator
(`scripts/welfare/analyse_welfare.py`, imported rather than reimplemented): a
hinge in the visibility weight with seed fixed effects, over the candidate grid
$(0.2, 0.4, 0.6, 0.8)$, on terminal-step `ideal_merit_sum` as the welfare-loss
measure. 200 observations per panel.

| Arm | Landscape | Entry | $\xi^*$ | F | p | Slope per unit visibility |
| --- | --- | --- | --- | --- | --- | --- |
| pre-fix archive | mg6 | 0.0 | 0.6 | 190.61 | 0.0 | -0.005111 |
| pre-fix archive | mg6 | 0.1 | 0.6 | 388.65 | 0.0 | -0.010467 |
| pre-fix archive | mgstd5 | 0.0 | 0.6 | 318.42 | 0.0 | -0.010900 |
| pre-fix archive | mgstd5 | 0.1 | 0.6 | 551.98 | 0.0 | -0.020837 |
| W_v05 re-measure | mg6 | 0.0 | 0.6 | 189.54 | 0.0 | -0.004602 |
| W_v05 re-measure | mg6 | 0.1 | 0.6 | 395.74 | 0.0 | -0.008989 |
| W_v05 re-measure | mgstd5 | 0.0 | 0.6 | 314.46 | 0.0 | -0.009478 |
| W_v05 re-measure | mgstd5 | 0.1 | 0.8 | 595.26 | 0.0 | -0.016946 |

Seven of eight panels keep $\xi^* = 0.6$. The one that moves
(mgstd5, entry 0.1) moves to $0.8$ --- the upper end of the range the welfare
probe already reported. The published claim of a knee at $\xi^* = 0.6$--$0.8$
therefore survives re-measurement; it does not narrow to a point.

The hinge remains highly significant everywhere ($p = 0.0$ to printed
precision), so the shape claim is not marginal.

# The companion audit: where the code-delta landed

`scripts/career_flows_noisy/audit_code_delta.py` re-ran a few seeds of two
noisy-landscape cells through the current tree and diffed them against the
archived run. It **raised** on a row-count check
(`RuntimeError: noisyB_cur5_van9_noisypr5_gen0p1_h200: row mismatch 603 vs
4020`): the archive stores step-level rows, the re-run stores one row per seed.
The failure is a shape mismatch, not a numeric one, and the per-reporter deltas
were written to `data/career_flows_noisy_v1/code_delta.json` before the raise.

They are the useful part, so they are read and reported here rather than
discarded:

| Cell | Seeds | Reporter | Max abs delta | Mean pre | Mean post |
| --- | --- | --- | --- | --- | --- |
| control, mgstd5, entry 0.1 | 5 | `Corr_prestige_ideal_merit` | 0.729684 | 0.114054 | 0.110427 |
| control, mgstd5, entry 0.1 | 5 | `Corr_prestige_local_merit` | 0.000000 | 0.137148 | 0.137148 |
| control, mgstd5, entry 0.1 | 5 | `top10pct_prestige_share` | 0.000000 | 0.190837 | 0.190837 |
| control, mgstd5, entry 0.1 | 5 | `explored_percentage` | 0.000000 | 0.353672 | 0.353672 |
| control, mgstd5, entry 0.1 | 5 | `percentage_knowledge_harvested` | 0.000000 | 0.496770 | 0.496770 |
| control, mgstd5, entry 0.1 | 5 | `mean_age` | 0.000000 | 113.169123 | 113.169123 |
| control, mgstd5, entry 0.1 | 5 | `mean_prestige` | 0.000000 | 0.015385 | 0.015385 |
| noisypr5, entry 0.1 | 3 | `Corr_prestige_ideal_merit` | 0.548018 | 0.356535 | 0.356414 |
| noisypr5, entry 0.1 | 3 | `Corr_prestige_local_merit` | 0.000000 | 0.389240 | 0.389240 |
| noisypr5, entry 0.1 | 3 | `top10pct_prestige_share` | 0.000000 | 0.254252 | 0.254252 |
| noisypr5, entry 0.1 | 3 | `explored_percentage` | 0.000000 | 0.380647 | 0.380647 |
| noisypr5, entry 0.1 | 3 | `percentage_knowledge_harvested` | 0.000000 | 0.192753 | 0.192753 |
| noisypr5, entry 0.1 | 3 | `mean_age` | 0.000000 | 112.545933 | 112.545933 |
| noisypr5, entry 0.1 | 3 | `mean_prestige` | 0.000000 | 0.005823 | 0.005823 |

Same signature as the paired arms, on a different round and a different
landscape: every non-ideal reporter exactly `0.0`, the shift confined to the
ideal channel, and the means nearly unchanged (0.114054 to 0.110427 and 0.356535
to 0.356414) even where individual step-level values move by up to 0.73.

That last point is worth stating plainly: a large per-step maximum with a small
mean shift is what a re-pairing looks like. The correlation is recomputed against
a different merit array at each step, so individual steps can move a lot while
the average level barely budges. It is not what a dynamics change looks like ---
that would move the reporters that do not read the ideal channel, and none of
them move at all.

# What changed in the repository

Every change is listed by `file:line`, one line each.

| File | Lines | Change |
| --- | --- | --- |
| `scripts/remeasure/run_remeasure.py` | 12-30 | Module docstring: `W_v05` arm rationale and the explicit write contract for the legacy outputs |
| `scripts/remeasure/run_remeasure.py` | 49 | New `CELLS_WV05` checkpoint directory |
| `scripts/remeasure/run_remeasure.py` | 133-140 | `run_one` records terminal-step `ideal_merit_sum` and `ideal_merit_mean` using the archived welfare definition |
| `scripts/remeasure/run_remeasure.py` | 144-163 | `run_cell` parameterised with `cells_dir` and `log_name`; legacy defaults unchanged |
| `scripts/remeasure/run_remeasure.py` | 161-163 | New `run_cell_w_v05` writer |
| `scripts/remeasure/run_remeasure.py` | 189-204 | New `cells_w_v05` grid (40 cells, vanishing 0.5) |
| `scripts/remeasure/run_remeasure.py` | 206-236 | New `consolidate_w_v05`; house-convention wedge in explicitly named columns; `n_seeds` in the summary |
| `scripts/remeasure/run_remeasure.py` | 243-285 | `legacy` arm refuses to rebuild over banked `runs.csv` or `cell_summary.csv` |
| `scripts/remeasure/run_remeasure.py` | 286-345 | `w_v05` arm: `n_jobs=3`, own manifest, own outputs |
| `scripts/remeasure/analyse_remeasure.py` | 96-167 | Paired T1 extraction from the step-level archive; time means via `Series.mean()` to match the model's own path |
| `scripts/remeasure/analyse_remeasure.py` | 169-183 | T1 pairing with the raw-file sign assertion |
| `scripts/remeasure/analyse_remeasure.py` | 186-260 | Welfare archive loader (seed from `generation_params`, leg tag mapped) and W_v05 pairing |
| `scripts/remeasure/analyse_remeasure.py` | 262-307 | Channel summary and per-cell delta tables |
| `scripts/remeasure/analyse_remeasure.py` | 310-343 | $\xi^*$ re-estimation reusing `scripts/welfare/analyse_welfare.py` |
| `scripts/remeasure/analyse_remeasure.py` | 346-384 | Companion code-delta audit reader |
| `scripts/remeasure/analyse_remeasure.py` | 387-424 | Ordering table (sign flips, rank agreement, entry-cell counts) |
| `docs/connections_niklas.md` | 108, 165 | Remaining stale `0.05` Markov-entry figures struck and replaced with `0.0619` |
| `docs/conceptual_degrees_of_freedom.md` | 3-4, 11, 92-93, 255 | v4: BUG-0005 correction and revision log; the two drafted rows are deferred (see below) |
| `AdmiralsLog/HANDOVER-2026-11-14.md` | 70-78, 101-102, 184-186 | Section 1.5 literature clusters rebuilt against the v2.0 review (46 studies, four clusters); section 1.7 formula corrected; revision log v1.1 |
| `README.md` | 15 | `output/reports/` added to the layout |
| `scripts/sensitivity/` | new | Three T1 round sources copied out of gitignored `data/`, with sha256 provenance in its `README.md` |

# Ready-to-paste replacement for P-0001

`AdmiralsLog/proposals/P-0001-phase1-paper.md` is outside this round's write
boundary, so the replacement table is supplied here instead of applied. The
claims below are the ones this pass can speak to; anything not listed rests on
rounds this pass did not re-run.

| Claim in P-0001 | Pre-fix figure | Re-measured | Verdict |
| --- | --- | --- | --- |
| Entry cells endstate-negative at horizon 200 | 17 of 18 | 18 of 18 (all 18 entry cells negative in the archive's `ideal - local` convention) | Holds, sharpens |
| Wedge is level-sensitive to measurement choices | not quantified | mean cell wedge $0.100395 \to 0.105984$ (+0.005588) with rank agreement $\rho = 0.9745$ | Holds, now quantified |
| Visibility collapse knee | $\xi^* = 0.6$--$0.8$ | 7 of 8 panels at $0.6$, 1 at $0.8$ | Holds |
| Any quoted level of $c_{ideal}$ | pre-fix archive values | shifted; use `data/remeasure_v1/tables/` | Must be updated |

The ordering-dependent claims are untouched by construction: the ideal side can
shift a level without reordering cells, and the measured rank agreement says it
did exactly that.

# Needs and limits

- **The `W` arm at vanishing 0.9 is not paired.** It differs from the archive in
  two ways at once, so its delta would be unattributable. `W_v05` exists to fix
  that; the 0.9 arm remains a bridge, not evidence.
- **Terminal ideal merit now exists for the new arm only.** The 76 already-banked
  re-measure cells predate the two columns and carry no `ideal_merit_sum`; that
  is why the welfare comparison runs on `W_v05` and not on the legacy `W` arm.
- **The audit script still raises.** The row-count check is a shape mismatch and
  should be relaxed to compare like-for-like grains; its numbers are otherwise
  sound and are quoted above. Fixing it belongs to that round's owner.
- **Two catalogue rows are drafted but deferred.** The `entry_prestige_stock` and
  `ideal_merit_cum` rows for `docs/conceptual_degrees_of_freedom.md` were verified
  against the live lines and then withheld: inserting them ahead of their mapping
  entries (which live inside `scripts/cdf_reanchor/reanchor_cdf_refs.py`, outside
  this round's write boundary) makes that audit abort on
  `FAIL: document has 106 catalogue rows, mapping has 104` instead of running.
  A re-anchoring of that mapping was already in flight while this document was
  written, so the rows would have collided with it. Both rows and their two
  mapping entries are supplied in the next section, ready to paste, for that
  round to land in one change.
- **No PDF was compiled for the HANDOVER.** The errata pass was scoped to the
  source document, so its compiled sibling is left stale deliberately rather
  than silently.

# Deferred: two catalogue rows, ready to paste

These belong in the Demographics (entry and supervisor choice) table of
`docs/conceptual_degrees_of_freedom.md`, and their mapping entries in
`scripts/cdf_reanchor/reanchor_cdf_refs.py`. They are supplied here rather than
applied because the two files must change together and only the first is inside
this round's boundary. Anchors were verified with `grep -n` against the live tree
while this document was written; the re-anchoring round should re-verify them,
since `model.py` has moved more than once today.

```markdown
| `sciencemodels/model.py:143,181-187,395-399` | Which prestige stock weights the supervisor roulette when newcomers are assigned | constructor parameter `entry_prestige_stock` (default `"prestige"`), validated to the three legal values `prestige` (raw stock), `prestige_vanishing` and `prestige_visibility`; `__generate_new_agents` reads `entry_stock = self.entry_prestige_stock` (line 395) and builds both the tuple list and the roulette total with `getattr(a, entry_stock)` (lines 396-399) | What the community treats as the quality signal newcomers act on: accumulated visibility (raw stock), recent output (vanishing) or peer-perceived standing (visibility). All three are swept in `output/reports/prestige_stock_v1.md`; the probe finds the stock sets how fast the quantity/quality wedge forms, not where it opens |
| `sciencemodels/metrics.py:105`, `sciencemodels/model.py:416` | Cumulative ideal-merit accumulator on every agent | `update_ideal_merit` adds the step's `a.ideal_Merit` into `a.ideal_merit_cum` with a lazy `getattr(..., 0.0)` default (line 105); `__generate_new_agents` initialises newcomers to `np.float64(0.0)` at birth (line 416), covering final-wave entrants that never get an `updateIdealMerit` call | Turns per-step ideal-merit loss into a career total, consumed by the cohort/career reporters (`scripts/remeasure/run_remeasure.py:131`, `scripts/path_b/run_path_b_sweep.py:154,157`) to compare lifetime quality deficit against lifetime prestige; a reporting construct, not a reward input |
```

The matching `MAPPING` entries, in the file's existing tuple shape (status
`re-anchored`; the old-reference slot carries a marker because these rows have no
v2 predecessor):

```python
    ('(new row, no v2 predecessor)', '`sciencemodels/model.py:143,181-187,395-399`', 're-anchored',
     [('sciencemodels/model.py', 143, 'entry_prestige_stock: str'),
      ('sciencemodels/model.py', 181, "if entry_prestige_stock not in ("),
      ('sciencemodels/model.py', 395, 'entry_stock = self.entry_prestige_stock')]),
    ('(new row, no v2 predecessor)', '`sciencemodels/metrics.py:105`, `sciencemodels/model.py:416`', 're-anchored',
     [('sciencemodels/metrics.py', 105, 'a.ideal_merit_cum = getattr(a, "ideal_merit_cum", 0.0) + a.ideal_Merit'),
      ('sciencemodels/model.py', 416, 'a.ideal_merit_cum = np.float64(0.0)')]),
```

Two bookkeeping consequences for that round: the document goes from 104 to 106
catalogue rows, and the printed callout counts move from 111 gross / 109
resolvable to 114 / 112.

# Revision Log

- **Version 1** (November 2026) --- Initial re-measure report: the pairing fix
  isolated to the ideal channel on two independent arms (1,520 pairs, exact
  zeros on every non-ideal reporter), ordering preserved at $\rho = 0.9745$, the
  collapse knee re-estimated at $\xi^* = 0.6$ in seven of eight panels, the
  companion code-delta audit read out and explained, and the P-0001 replacement
  table supplied.
