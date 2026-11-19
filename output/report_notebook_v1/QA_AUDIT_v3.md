---
title: "QA_AUDIT_v3 — rendered-canvas audit of the repaired and post-BUG-0006 figures"
date: "2026-11-19"
status: "complete; per-panel confirmations below are the coordinator's own"
---

# What this audit is

The v2 verifiers audited `_data.csv` sidecars and text-vs-text boxes only. A
first-hand review of commit `71239c1` found what they could not see: one
catastrophic overlay (`F12`), several text-vs-title / text-vs-data collisions
(`F13`, `F9_v2`, `F10a`) and two polish items (`F8_v2`, `F10b`). This round
adds a rendered-canvas contract (`scripts/report_notebook/visual_qa.py`) and
re-audits every figure against it.

# The contract

`visual_qa.assert_figure_clean(fig, panels)` runs before every `savefig`:

1. **Rendered-artist data contract** — per panel: series/bar counts match the
   panel's single assigned metric; plotted min/max *including error-bar
   endpoints* match the sidecar rows for that metric at `tol = 1e-9`; every
   value sits inside `ylim` with >= 3% headroom (log-panel headroom is
   delegated to the builder).
2. **Collision-free layout contract** — every visible text artist (titles,
   suptitle, axis labels, tick labels, `ax.texts`, legend boxes **and** legend
   texts, `fig.texts`, figure legends) vs canvas, vs each other, and vs
   plotted geometry (lines sampled in **display coordinates**, bar rectangles,
   error-bar segments and caps, spines). Duplicate legend labels fail.
3. **Bounded layout relief** — colliding labels are nudged by the smallest
   clean offset (<= 28 pt, vertical first; `Annotation.xyann` for offset-point
   labels) and the report records every move; unrecoverable positions stay
   hard failures.

Two defects in the first version of the contract were found by the lanes
themselves and fixed here: (i) line/error-bar segments were compared in DATA
units against DISPLAY-space text boxes, so line-over-text never fired (proved
by a minimal repro; now transformed through `ax.transData`); (ii) the suptitle
is registered in both `fig.texts` and `fig._suptitle` (double-counted) and
figure legends were missing from the text inventory — both included now.

Proof that the contract catches the original bug: replaying v2's panel loop
raises 156 failures, first line
`panel[harvested knowledge share]: 24 plotted series, expected 6` and
`series 0 plots 15 points, expected 3`.

# Defects found and fixed

| figure | defect (v2) | fix |
| --- | --- | --- |
| `F12` | **catastrophic**: every panel plotted all four metrics per arm (no `series[metric==...]` filter) | filter inside the panel loop; bands/labels re-placed data-aware |
| `F12` | band labels inside the axes collided with curves reaching the top | placed last, after `subplots_adjust`, into the first data-free band |
| `F9_v2` | panel A callout over the Pure marker; panel B clipped `c=0.7` cap; identical M0/M1 swatches; panel C values on the spine | ylim from sidecar caps, boxed callout, hollow/hatched vs solid swatches, padded spines |
| `F13` | panel B rotated labels under the title; `g=0.1` label at x-offset 0; panel C callout vs title, diamonds inside hatched bars; D1 equality tag on a bar; D2 note inside bars, `-67.0%` on the spine | decade padding, +0.18 label offset, top/bottom pads, labels right of the pairs, equality tag in the white band, D2 note above the zero line |
| `F10a` | panel B duplicate `pull = x` labels + legend on the `pull=0` whiskers; panel C legend inside curves; panel D numbers on whiskers | label only the mgstd5 arm, 5-handle legend, legend outside, labels above `ci_hi` |
| `F8_v2`, `F10b` (pre-fix) | minor polish | superseded by `F8_v3` / `F10b_v2`; old files byte-identical |

# Coordinator's per-panel confirmations (own visual pass)

`F12_sabbaticals_vs_lotteries.png`
- A: single metric (harvested share), 6 arms, curves monotone in xi, Stage
  labels in a data-free band at the panel top, callout lower-left clear of
  curves and whiskers.
- B: explored share rising 0.30 -> 0.98 for the sabbatical arm; `kink` label
  sits between the boundary rules, above the data.
- C: terminal loss rising with xi for all six arms; callout clear.
- D: alignment rising +0.001 -> +0.319; 3-line callout wrapped, clear.
- Shared 6-entry legend below the panels, clear of x-labels and footnote.

`F9_v2_typology_annealing.png`
- A: Pure hollow marker at 0, destruction and full arms at -0.0784/-0.0788,
  counterforce at -0.0007; all inside the axes; boxed gamma/kappa/P-bar
  callout clear of the Pure marker.
- B: `c=0.7, gamma=1, M0` hollow green bar and its -0.1308 cap fully visible;
  arrow labels above the caps; M0 hollow+hatched vs M1 solid.
- C: the -0.191 bar and its -0.2643 cap clear of the spine; zoomed check:
  value labels sit BELOW the lower caps, clear of the stems.

`F13_paradigm_refill_regimes.png`
- A: exploration collapse 37.7% -> 6.1% (occupied) legible; note in white air.
- B (log): rotated 0.186078 / 0.189537 labels below the title, above their own
  bars, M1 column right-offset; frontier residual 0.1895 vs baseline 0.000617
  unmistakable.
- C: frontier terminal +0.28 stays the only positive terminal regime; M1
  time-mean diamonds right of the pairs with labels; legend upper-right.
- D1: top-10% 0.213/0.221/0.248, equality rule in the legend band.
- D2: gaps -67.0/-55.5/-51.6%; the 31.6x/43.9x inflation note sits above the
  zero line, clear of bars; nothing touches the spine.

`F10a_m2_choice_to_production.png`
- A: links vs lambda by pull, legend top-left clear, assortativity note clear.
- B: unique legend (pull = 0/0.5/1 plus a solid/dashed landscape key), clear
  of the `pull=0` whiskers and the M1/2.65 reference; `pull=0` flat at ~8.1.
- C: legend outside the axes below the x-label; top-right note clear of the
  curves and of the y ticks.
- D + D': zoomed confirmation of the vertical order `number -> upper cap ->
  whisker -> bar` for all eight bars; rep_tm flat 0.919-0.929 with the Klann
  0.34 rule labelled `Empirical benchmark (unfitted)`.

`F6_v2_two_stage_proxy_failure.png`
- A: five tiers, staggered arm top-flat at 0.938, entry arms flat 76.4-77.0%
  to xi=0.5 then collapsing to 38.9-43.2% (M0 33.0%); band labels above the
  data; callout lower-left clear.
- B: like-for-like `c_ideal_tm` hierarchy M0 > M2 lambda=1 > M1 > M1+staggered
  > M2 lambda=0; the 7-line block inside the axes above every curve; zero rule
  visible; shared 5-entry legend between x-labels and footnote.

`F8_v3_turnover_diagnostics.png`
- (a): four post-fix trajectories, Discovery/Depletion bands, dotted cohort
  lines at 200/400; inset ghost (pre-fix -0.568 -> post-fix +0.017) fully
  framed with its own legend; pooled note in white space, uncrossed.
- (b): settled alignment (301-400) vs g for four modes; legend above the caps.
- (c): gap_recent -0.01362 / -0.00612 [-55.0%] / -0.00281 [-79.3%] / -0.00233
  [-82.9%], labels inside the axes, diamonds vs triangles distinct, note
  `negative in 1,440/1,440 runs` (common support) top-left.
- (d): twelve bars; step-400 top-10% sync 0.409 -> staggered 0.270; labels
  above caps; equality rule in the legend.

`F10b_v2_m2_visibility_like_for_like.png`
- A: entry arms flat to xi=0.5 (76.4-77.0%), collapse 38.9-43.2% at xi=0.9,
  M0 33.0%; callout clear.
- B: no stray "0" text (only the tick and the zero rule), no vertical
  band-boundary lines across the series, hierarchy block inside the axes above
  the curves.
- C: mob growth 2.2 -> 27.0 (lambda=0) / 25.1 (lambda=1); the 2.65 Klann rule
  labelled `Empirical benchmark (unfitted)`, clear of the curves.

`F1_v2_t1_postbug.png`
- (a): twelve bars (M0 hatched vs M1 solid, mgstd5 vs noisypr5) with CIs;
  attenuation box `pooled 1.74x / mgstd5 2.30x / noisypr5 1.55x` clear; legend
  clear.
- (b): four bars with nine hollow cell dots each; labels `+0.281 0/9 < 0`,
  `+0.064 0/9 < 0`, `+0.159 2/9 < 0`, `+0.002 5/9 < 0` in the empty band below
  zero; no stray ticks.

# Verification evidence

- `section_figures_v2.py` — ten families build, zero warnings.
- `verify_section_figures_v2.py` — `ALL CHECKS PASSED` (exit 0): every family
  recomputed from raw CSVs at 1e-9, builders re-run in temp dirs so the QA
  contract re-executes, counts asserted (1,080 / 1,520 / 720 / 480 / 360 /
  1,800 / 200 / 720 / 800 runs).
- Canary `tests/test_smoke.py`: `ALL SMOKE TESTS PASSED` after the round.
- Old figures byte-identical: `F8_v2_*`, `F10b_*`, `F6_*`, `F11_*`,
  `F9_*` (pre-v2), `F10_*` — sha256 unchanged.

# Deviations and pins (disclosed)

- `F8_v3` uses the common-support curiosity {0.3, 0.5, 0.7} (1,440 of 1,520
  runs) because the hazard arms carry extra curiosity cells 0.1/0.9; the 80
  excluded runs are named in the footnote and sidecar.
- Band labels are placed dynamically (first data-free band) after the final
  layout — fixed-point boxes change size relative to a resized axes.
- `F13` panel B is a log axis: 3%-of-span headroom is meaningless there, so the
  builder passes `headroom=0` and pads by decades; panel C carries one measure
  at two aggregations (bars final, diamonds time mean), labelled as such.
- `F6_v2` pins: w_v05 vanishing 0.5, m2_xi vanishing 0.9; the staggered
  turnover arm runs 400 steps and is not level-comparable — stated in the
  footnote.
- Short-lived residual flags for the paper text: post-BUG-0006 the M1
  annealing narrowing is -0.0204 [-0.0460, +0.0045] (pre-fix -0.0489
  [-0.0751, -0.0225], pooled -0.0316), and `F13` panel C shows all three
  regimes positive at the terminal step (frontier still dominant). Both
  supersede pre-fix claims in `drafts/paper_sketch_v1.md`.
