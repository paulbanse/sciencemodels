---
title: "M2 counterfactual ablation-share decomposition (v1)"
version: 1
date: 2026-11-26
revision-vehicle: strikethrough
revision-renders: "md, pdf"
tags: [m2, path-b, dyadic, decomposition, ablation, not-calibrated]
project: sciencemodels
---

**Version 1** (26 November 2026)

> **STATUS: MODEL-INTERNAL, NOT CALIBRATED** (owner ruling, 2026-11-13).
> Workplan section 2.4 item 14 (former H4, IE-0007 #6): the Figure 8
> counterfactual ablation-share decomposition applied to Tier M2 on the
> existing `data/path_b_postbug/` archive. Read-only on the archive: no
> file under `sciencemodels/` or `drafts/` was edited, no run was
> re-run; every number below is recomputed from the archived CSVs by
> `scripts/m2_decomposition/m2_decomposition.py`.

**Verdict.** The dyadic layer's effect on prestige-merit alignment is
carried by **spatial co-movement**, not by nominal link formation.
Decomposing the signed M1 -> M2 change in time-mean
`Corr(prestige, ideal merit)` along the chain M1 -> link formation only
(pull = 0) -> + spatial co-movement (pull = 1) -> + equal-split credit
sharing, the co-movement leg carries @@V_MOV_MG_L0@@% of the drop at
$\lambda = 0$ on clean terrain and @@V_MOV_NP_L0@@% on noisy terrain
(more than the whole drop, because link formation alone runs the other
way), and @@V_MOV_MG_L1@@% of the alignment gain at $\lambda = 1$ on
clean terrain. Equal-split credit sharing is a second-order trim: it
runs counter to the $\lambda = 1$ gain on clean terrain
(@@V_CRD_MG_L1@@%) and is negligible at $\lambda = 0$. The link-only
leg is inert by
construction: at pull = 0 under structural credit the archived
trajectories are exactly $\lambda$-blind (max per-seed spread
@@V_LAM_BLIND@@), so its share is an arm-activation footprint, not a
partner-choice mechanism. The one cell where the picture splits is
noisy terrain at $\lambda = 1$, where link-only and co-movement shares
divide the gain @@V_LNK_NP_L1@@% / @@V_MOV_NP_L1@@%.

# Design

The method is Figure 8's (paper sketch section 4.2; builder
`fig_FA2_v2_arm_a_decomposition.py`) with the tier and the chain changed.
Figure 8 splits the entry-induced M0 -> M1 drop into a starvation and a
herding share via one counterfactual spawn arm; here the object of
decomposition is the dyadic-layer-induced M1 -> M2 alignment change, and
the ablation arms are the three M2 mechanisms named in the workplan
item: link formation alone, spatial co-movement, and equal-split credit
sharing. The chain and its shares are

- L0: M1 baseline, dyadic layer off;
- L1: link formation only, dyadic on, `dyadic_move_pull = 0`, structural
  credit;
- L2: + spatial co-movement, `dyadic_move_pull = 1.0`, structural
  credit;
- L3: + equal-split credit sharing, `dyadic_move_pull = 1.0`,
  `equal_split`, the full M2 arm.

The signed gap is $gap = L0 - L3$ (positive means M2 sits below M1) and
the three mechanism shares are $(L0 - L1)/gap$, $(L1 - L2)/gap$ and
$(L2 - L3)/gap$, which sum to exactly 100% by construction (asserted in
code). Sequential decompositions are order-dependent, so a second order
(credit leg moved to pull = 0, movement leg last) is computed as a
robustness twin; the first leg is shared by both orders and therefore
order-invariant. The primary rows pool over the two link formulations
(`logit`, `threshold`); per-formulation rows are the robustness twin.

The metric is `c_ideal_tm`, the time-mean of the model's
`Corr(prestige, ideal merit)` reporter, the same quantity as Figure 15c
of the paper sketch. The M2 archive carries no terminal alignment
column, so the time-mean is the archive-native quantity; this differs
from Figure 8's terminal metric and is disclosed again under caveats.

Sources (all read-only; `agent_seed` 0-19 drives agent and landscape RNG
jointly, per `manifest.json`): `baseline_off.csv` with @@V_N_BASE@@ runs
(M1 reference, dyadic off), `cells_v1/` with @@V_N_V1@@ runs (pull = 0),
`cells_v2/` with @@V_N_V2@@ runs (pull 0.5 and 1.0), and
`m0_m2_probe/` with @@V_N_PROBE@@ runs (the $g = 0$ canary). All arms:
curiosity $c = 0.5$, entry rate $g = 0.1$, horizon $H = 200$, 20 seeds
per cell; each chain node is 2 cells x 20 seeds = 40 runs when pooled
over formulation, 20 runs per formulation. Uncertainty is a seed-paired
bootstrap over the 20 common seeds (10,000 draws, RNG seed 20261126),
reported as percentile 95% intervals; the paired design is legitimate
because every arm shares the same agent-seed grid.

# Results

The chain levels and the signed gaps (pooled over formulation):

@@T_LEVELS@@

The mechanism shares, with seed-paired bootstrap intervals in brackets
(shares in every row sum to exactly 100%; the largest deviation across
all twelve chains is reported under Verification):

@@T_SHARES@@

![Figure 1: M2 ablation-share decomposition of the M1 to M2 alignment change by mechanism share, per landscape and partner-choice rule](../../data/m2_decomposition/figures/m2_decomposition_v1.png)

Clean terrain, $\lambda = 0$ (prestige matching): the dyadic layer dropped
time-mean alignment by a gap of @@V_GAP_MG_L0@@
(95% CI @@V_GAP_MG_L0_CI@@). Co-movement explains @@V_MOV_MG_L0@@%
(@@V_MOV_MG_L0_CI@@), the only share whose interval excludes zero;
link formation (@@V_LNK_MG_L0@@%) and equal-split credit
(@@V_CRD_MG_L0@@%) are statistically indistinguishable from zero weight.

Clean terrain, $\lambda = 1$ (closeness x local merit): the layer lifts
alignment (gap @@V_GAP_MG_L1@@, CI @@V_GAP_MG_L1_CI@@), and co-movement
more than explains the lift (@@V_MOV_MG_L1@@%): link formation alone
moves the metric the wrong way for this sign (@@V_LNK_MG_L1@@%) and
equal-split credit trims the gain (@@V_CRD_MG_L1@@%). Because the gap
interval touches zero here, this share triple is indicative only (see
caveats).

Noisy terrain, $\lambda = 0$: the drop (gap @@V_GAP_NP_L0@@, CI
@@V_GAP_NP_L0_CI@@) is again a co-movement effect
(@@V_MOV_NP_L0@@%, CI @@V_MOV_NP_L0_CI@@) that exceeds the whole gap,
because the link-only arm runs counter to it (@@V_LNK_NP_L0@@%, CI
@@V_LNK_NP_L0_CI@@); equal-split credit is negligible
(@@V_CRD_NP_L0@@%).

Noisy terrain, $\lambda = 1$: the only split verdict (gap
@@V_GAP_NP_L1@@, CI @@V_GAP_NP_L1_CI@@): link-only and co-movement
shares are @@V_LNK_NP_L1@@% (CI @@V_LNK_NP_L1_CI@@) and
@@V_MOV_NP_L1@@% (CI @@V_MOV_NP_L1_CI@@), with credit at
@@V_CRD_NP_L1@@%.

**What the link-only leg can and cannot mean.** In this arm the link
layer is provably inert: `sciencemodels/agents.py` skips the move pull
entirely at pull = 0 (lines 193-211), and structural credit accrues
without consulting links (lines 255-271). Consistently, the archived
trajectories are bit-identical across all five archived $\lambda$ levels
at pull = 0 under structural credit (max per-seed spread
@@V_LAM_BLIND@@) even though declared links vary from
@@V_LINKS_LO@@ to @@V_LINKS_HI@@ mean active links across $\lambda$
(logit, `mgstd5`). The M1 and link-only levels are @@V_A0_MG@@ and
@@V_A1_MG@@ on clean terrain, @@V_A0_NP@@ and @@V_A1_NP@@ on noisy
terrain, so the leg-1 level difference ($L0 - L1$) reads
@@V_LNKLEV_MG@@
(95% CI @@V_LNKLEV_MG_CI@@) on clean terrain and @@V_LNKLEV_NP@@
(@@V_LNKLEV_NP_CI@@) on noisy terrain, both intervals straddling zero,
consistent with the inert-footprint reading. The correct reading of the share
is "dyadic layer switched on but behaviorally inert", and the residual
contrast (layer on vs. off with inert links) is an arm-activation
footprint of the proposal machinery that this archive cannot attribute
further; a targeted proposal-layer invariance probe is the named open
question.

**What the credit leg is.** Under `equal_split` the model pools each
step's accrual and splits it among linked co-located agents
(`sciencemodels/agents.py`, lines 255-266), which is the only channel
from link identity into payoffs when the pull is zero. At pull = 1 it
operates on top of co-movement, and its share is a modest trim of the
$\lambda = 1$ gain on clean terrain.

Per-formulation robustness (point shares; the identity again sums to
100% in every row):

@@T_BY_FORM@@

The formulation twin moves the weight between the legs (clean terrain at
$\lambda = 0$: the co-movement share runs @@V_MOV_MG_L0_FORMRANGE@@
across formulations, against the pooled share above) but never the
verdict, except that one cell becomes
near-degenerate: @@V_DEGEN_CELL@@ has gap @@V_DEGEN_GAP@@, where ratio
shares explode to @@V_DEGEN_SHARES@@ and carry no information. The
alternative chain order confirms the movement leg as the carrier while
showing the expected order sensitivity of the credit leg:

@@T_ORDER_B@@

Under order B the clean-terrain $\lambda = 1$ movement share is
@@V_MOV_MG_L1_B@@% (vs. @@V_MOV_MG_L1@@% under order A) and the credit
share @@V_CRD_MG_L1_B@@%. Two internal consistencies close the
mechanism reading. First, the movement leg is smooth in the pull:
the pull = 0.5 level sits between pull = 0 and pull = 1 in
@@V_SMOOTH_N@@ of 8 formulation x landscape x $\lambda$ cells (the
exceptions are small noisy-terrain threshold deviations, listed with
their margins in the checks table). Second,
without demographic entry the layer degenerates: in the $g = 0$ probe
(`m0_m2_probe/`) declared links are @@V_PROBE_LINKS@@ and the two
$\lambda$ arms are identical per seed (max difference
@@V_PROBE_DIFF@@), so the decomposition is strictly entry-mediated,
consistent with paper section 6.

# Verification

@@T_CHECKS@@

The decomposition identity holds to a maximum absolute deviation of
@@V_SUM_DEV@@ across all twelve chains (pooled and per-formulation,
both orders). The figure passes the house visual QA with
@@V_QA_FAILS@@ failures (rendered-canvas data contract and collision
check, `scripts/report_notebook/visual_qa.py`). Reproduction is a
single command from the repository root:

```{.bash}
MPLBACKEND=Agg /opt/hermes/.venv/bin/python \
    scripts/m2_decomposition/m2_decomposition.py
```

The script re-derives every number in this report from
`data/path_b_postbug/` and rewrites the report from
`scripts/m2_decomposition/report_template.md` by counted token
replacement, so no computed number is hand-typed here. A same-script
rerun reproduces the CSV outputs exactly (bit-level; floats recomputed
from the same parsed inputs); the figure PNG and PDF bytes carry
matplotlib metadata and are not compared byte-wise.

# Scope and caveats

- **Chain order.** Shares are sequential-decomposition quantities and
  order-dependent; the credit and movement legs trade weight across the
  two orders (table above), while the total gap and the link leg do not.
- **Ratio instability.** When the gap is small or its interval touches
  zero (clean terrain at $\lambda = 1$; the near-degenerate
  per-formulation cell above), ratio shares explode and their bootstrap
  intervals are unbounded in practice; read the leg level differences
  and gap intervals as the primary inference, the shares as the
  decomposition of the point estimate.
- **Metric.** Time-mean alignment, not the terminal correlation used in
  Figure 8; the archive has no terminal alignment column.
- **Design slice.** One cell of the parameter space: $c = 0.5$,
  $g = 0.1$, $H = 200$, 20 seeds, two landscapes; no curvature claims
  beyond this slice.
- **Pooling.** Primary rows pool over the two formulations (n = 40 per
  node); the paired bootstrap treats the common agent-seed grid as the
  pairing key.
- **Leg-1 interpretation.** As argued above, the link-only contrast
  cannot be read causally in this archive; it is reported for
  completeness of the mandated chain.
- **Code base.** The archive was produced at git head `117f632`
  (manifest `sources_snapshot_before/after`, 2026-11-19), i.e. the
  pre-ADR-0003 model; this analysis reads that archive as-is and makes
  no claim about post-ADR-0003 dynamics.

# Provenance

- Analysis: `scripts/m2_decomposition/m2_decomposition.py` (loader,
  chains, bootstrap, checks, figure, report fill).
- Report template: `scripts/m2_decomposition/report_template.md`; the
  filled document is this file.
- Sources (read-only): `data/path_b_postbug/baseline_off.csv`,
  `cells_v1/`, `cells_v2/`, `m0_m2_probe/`, `manifest.json`; counts and
  schema asserted against `manifest.json` `gross_net`.
- Canonical numbers: `data/m2_decomposition/verification_table.csv`
  (all twelve chains with levels, gaps, shares, bootstrap intervals),
  `checks.csv`; generated table fragments under
  `data/m2_decomposition/tables/`; figure data sidecar
  `data/m2_decomposition/figures/m2_decomposition_v1_data.csv`.
- Figure: `data/m2_decomposition/figures/m2_decomposition_v1.png`
  (and vector PDF), built on the house figure contract
  (`diagram_templates`, `visual_qa`) with zero hardcoded model numbers.
- No model file, no `sciencemodels/` and no `drafts/` file was edited;
  nothing was pushed.
