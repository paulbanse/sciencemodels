---
title: "Refill-gaussian v1: Gaussian neighbour refill across dispersion"
version: 1
date: 2026-11-26
revision-vehicle: strikethrough
revision-renders: "md, pdf, html"
tags: [ie-0006, refill, gaussian, open-endedness, lock-in, not-calibrated]
project: sciencemodels
---

**Version 1** (November 2026)

> **STATUS: MODEL-INTERNAL, NOT CALIBRATED** (owner ruling, 2026-11-13).
> Prototype of `IE-0006` Option 5 / workplan item 13: 36 cells, 720 runs,
> $H = 400$, 20 seeds per cell. **No model file was edited**;
> `GaussianRefillModel` subclasses `MyModel` in
> `scripts/refill_gaussian/run_refill_gaussian.py`. Every number below is
> recomputed from `data/refill_gaussian_v1/` (see *Provenance*).

**Verdict.** Harvest-conserving Gaussian neighbour refill does **not** replicate
the `occupied` arm's lock-in at small $\sigma$. At $\sigma = 1$ the final
explored share is only $0.03$--$0.07$ below `off`, against $0.32$--$0.42$ for
`occupied`, and late flow is $0.003$--$0.009$ per step against
$0.073$--$0.209$. Large $\sigma = 6$ does sustain much broader exploration
(final explored share $0.83$ at $c = 0.5$/$0.7$, $g = 0.1$; $+0.35$ over `off`),
but it disperses stock so thinly that harvest throughput falls: late flow is
$0.003$ per step. The axis trades local retention against breadth, not lock-in
against open-endedness in the strong sense. `frontier` remains the only arm
with both high breadth and high flow/correlation, but it injects new stock
rather than conserving it.

# Design

| Element | Setting |
| --- | --- |
| Landscape | `mgstd5` = `multipleGaussians` (`std = 5`, `number_gaussians = 2`) |
| Population / horizon | $N = 20$, $H = 400$ |
| Axes | refill regime $\times c \in \{0.3, 0.5, 0.7\} \times g \in \{0.0, 0.1\}$ |
| Seeds | 20 per cell (`agent_seed` drives agent and landscape RNG) |
| Fixed | $\epsilon = 0.1$, harvest $= 0.02$, `use_distance=True`, retention $= 0.9$, visibility reward on, share normalization, harvested prestige accrual, `entry_prestige_stock="prestige"`, dyadic off |
| Grid | 6 regimes $\times$ 3 curiosity levels $\times$ 2 entry rates $= 36$ cells; 720 runs |

Refill regimes:

- `off` --- no refill; the baseline.
- `occupied` --- the model's own `new_questions=True` rule: a depleted occupied
  tile is replaced in place by a uniform draw.
- `frontier` --- a depleted occupied tile spawns its new question on a random
  unoccupied tile (the interventions-v1 frontier arm).
- `gaussian_s1` / `s3` / `s6` --- each harvest $\Delta k$ at $(x,y)$ is removed
  from the source and added to another tile drawn from a discretised 2D torus
  Gaussian centred at $(x,y)$, excluding the source. The kernel uses the
  shortest torus displacement, renormalises after zeroing the source, and is
  sampled by inverse CDF from the seeded model RNG. Total stock is conserved.

Because Gaussian refill conserves stock, `percentage_knowledge_harvested` is
not a usable flow measure there. The driver records the removed harvest
directly as `cumulative_harvest_flow` (sum of $\Delta k$ over 400 steps) and as
`late_window_flow_rate` (mean removed flow per step over the last 50 steps),
alongside the final/mean explored share and Corr(prestige, ideal_merit).

# Results

Central cell ($c = 0.5$, $g = 0.1$), 20 seeds; mean (SD in parentheses for
explored share). "Late flow" is per step over the last 50 steps; "cum. flow" is
the 400-step sum; "corr mean" is the 400-step mean of
Corr(prestige, ideal_merit); "final stock" is total knowledge (initial $= 1$).

| Regime | Explored final | Late flow | Cum. flow | Corr mean | Final stock |
| --- | ---: | ---: | ---: | ---: | ---: |
| `off` | 0.479 (0.038) | 0.0005 | 0.937 | 0.085 | 0.063 |
| `occupied` | 0.063 (0.013) | 0.2053 | 59.411 | 0.145 | 4.521 |
| `frontier` | 0.537 (0.059) | 0.2709 | 84.678 | 0.305 | 57.519 |
| `gaussian_s1` | 0.409 (0.039) | 0.0092 | 3.567 | 0.046 | 1.000 |
| `gaussian_s3` | 0.577 (0.050) | 0.0041 | 1.949 | 0.095 | 1.000 |
| `gaussian_s6` | 0.832 (0.050) | 0.0030 | 1.498 | 0.150 | 1.000 |

Across all six $(c,g)$ cells (120 runs per regime):

- **Small $\sigma$ is local retention, not occupied lock-in.** `gaussian_s1`
  explored-share change versus `off` ranges from $-0.028$ to $-0.070$; the
  `occupied` change ranges from $-0.322$ to $-0.416$. Small-$\sigma$ late flow
  is $0.003$--$0.009$ per step, far below `occupied` ($0.073$--$0.209$) and
  `frontier` ($0.116$--$0.271$). It extends harvesting relative to `off`
  (cumulative flow $1.80$--$3.65$ versus $0.78$--$0.94$), but does not collapse
  the searched area.
- **Large $\sigma$ buys breadth, not throughput.** `gaussian_s6`
  explored-share change versus `off` ranges from $+0.032$ to $+0.359$, with the
  largest gains at $g = 0.1$ and $c \in \{0.5, 0.7\}$. Yet its late flow
  ($0.0011$--$0.0031$ per step) is the lowest of the non-`off` regimes, and
  its cumulative flow ($0.91$--$1.52$) is only modestly above `off`. The stock
  is spread widely enough that agents keep finding low-density tiles and keep
  exploring, but each step removes little.
- **$\sigma = 3$ is intermediate on breadth** ($+0.005$ to $+0.108$ explored
  versus `off`) and on flow: cumulative flow $1.11$--$1.98$, above `off` but
  below `s1`. On the 400-step average,
  Corr(prestige, ideal_merit) is $0.111$ for `s1`, $0.136$ for `s3`, $0.189$
  for `s6`, and $0.165$ for `off`; the `s6` average is slightly above `off`,
  while `s1` is below. Final-step correlation is noisier (e.g. `s3` central
  cell: mean $0.095$, final $-0.017$), so the run-mean is the safer headline.
- **Non-conserving arms inflate flow.** `occupied` and `frontier` end with
  total stock $4.52$ and $57.52$ at the central cell, against $1.00$ for the
  Gaussian arms. Their large cumulative flows therefore measure generated
  harvest opportunities, not circulation of a conserved stock. The Gaussian
  arms are the only refill arms where `cumulative_harvest_flow` is a pure
  redistribution measure.

**Interpretation.** The $\sigma$ knob controls the spatial scale of the
recycled harvest, and therefore whether refill is visible to the model's
4-neighbour search. At small $\sigma$ the target is usually inside or near the
agent's local patch, which keeps local tiles alive and raises late flow, but
because only the harvested fraction is moved --- unlike `occupied`'s in-place
replacement --- it does not pin the population to a tiny set of tiles. At large
$\sigma$ most targets lie outside the search horizon, so the effect is closer
to a diffuse frontier: agents roam over a broadly stocked grid and breadth
rises, while local concentration and per-step harvest fall. Whether large
$\sigma$ produces true *traveling* frontiers cannot be settled from these
scalar summaries; it needs spatial tracking of peak positions.

# Scope and caveats

- One landscape (`mgstd5`), one $20 \times 20$ torus, $H = 400$, 20 seeds, and
  no sabbatical/mobility shocks. The conclusions are conditional on that
  design.
- The Gaussian kernel is an exact discrete torus Gaussian in tile units with
  the source excluded and the rest renormalised. $\sigma$ is therefore
  directly comparable to the grid length and to the 4-neighbour search radius.
- Total-stock conservation is verified per run (maximum error
  $1.8 \times 10^{-15}$ over all Gaussian runs); `off` is bit-identical to
  plain `MyModel` (3 seeds, worst difference $0.0$). All 720 runs completed,
  with 0 seed failures and 0 crashed cells (wall $316.9$ s, `n_jobs = 6`).
- No $T \to \infty$ test: "late" means the final 50 of 400 steps. A longer
  horizon is the natural next check if the diffuse large-$\sigma$ regime is to
  be called stationary or open-ended in a strong sense.

# Provenance

- Runner: `scripts/refill_gaussian/run_refill_gaussian.py`
  (`--probe` canaries + checkpointed grid; per-cell state under `state/`).
- Analysis: `scripts/refill_gaussian/analyse_refill_gaussian.py`.
- Outputs: `data/refill_gaussian_v1/cells/` (36 cell CSVs, 20 rows each),
  `summary_by_cell.csv`, `summary_by_mode.csv`, `summary_by_mode_entry.csv`,
  `headline_c05_g01.csv`, `analysis.json`, `probe/probe_results.json`,
  `design.json`, `run_log.txt`.
- Canaries: baseline bit-identity, Gaussian conservation, regime binding,
  worker-written schema (74 columns), and per-regime timing.
- Smoke test: `MPLBACKEND=Agg /opt/hermes/.venv/bin/python tests/test_smoke.py`
  passes.
