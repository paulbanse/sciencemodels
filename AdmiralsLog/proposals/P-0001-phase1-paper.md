---
title: "P-0001 - Phase-1 paper on the strength of waves 1-3: consolidate and proceed to drafting?"
author: "Hermes Agent"
date: "2026-11-14"
tags: [proposal, strategic, paper, proxy-failure, phase-1]
project: "sciencemodels"
status: exercised
---

- **Status:** exercised
- **Class:** strategic
- **Date:** 2026-11-14 (proposed); 2026-11-15 (exercised in `output/reports/paper_phase1_v1.md`)
- **Spark:** Five verified exploration rounds (waves 1-3) now support the
  Phase-1 claim of `AdmiralsLog/way_forward.md`; the owner ruled the dossier should
  live here as a proposal (2026-11-13).
- **Outcome target:** a ruling on whether to proceed to paper drafting along
  the spine below, and on venue/authorship framing.
- **Ruling:** exercised (initial Phase-1 draft delivered at `output/reports/paper_phase1_v1.md`; ready for Wave-4 integration from `CONT5` and venue selection)
- **Spawned:** `output/reports/paper_phase1_v1.md` (initial Phase-1 draft landed 2026-11-14)

## The proposal

Consolidate waves 1-3 into the Phase-1 paper spine --- *Does Prestige Track
Quality? Modelling the Merit-Prestige Nexus under Alternative Modelling
Choices* (the skeleton in `AdmiralsLog/way_forward.md`, Paper skeletons) --- and
proceed to drafting. This record is the dossier: the evidence index is
linked, not restated (each claim below names its source document).

## The stated reason

The model-internal evidence (nothing calibrated; owner ruling 2026-11-13)
now forms a coherent mechanism story across independent probes:

- The wedge opens under entry, in the ideal-merit channel: 17/18 entry cells
  endstate-negative at horizon 200 (`output/reports/sensitivity_map_v1.md` and
  post-audit `output/reports/remeasure_v1.md`: $+0.220 \to -0.144$, 0/36 cell
  sign flips on $c_{\text{ideal}}$, wedge rank agreement $\rho = 0.9745$), and the
  claim survives at horizon 400 in 6/6 cells, 16/18 conjunctive
  (`output/reports/career_flows_v1.md`). Curiosity is protective; noisy legibility
  widens the local-ideal gap without lowering average correlation.
- Visibility weight is the dominant lever: welfare loss rises 10x-44x across
  the weight axis with a collapse knee at $\xi^* = 0.6$-$0.8$
  (`output/reports/welfare_probe_v1.md`; post-audit `W_v05` re-measure keeps
  $\xi^* = 0.6$ in 7/8 panels and $0.8$ in 1/8, `output/reports/remeasure_v1.md`);
  the visibility toggle is nearly inert (2.6%),
  and entry doubles total loss through population, not per-agent pressure.
- Cumulative advantage is endogenous: the newcomer-incumbent prestige gap
  never closes (`output/reports/career_flows_v1.md`), robust in sign and persistence on
  the noisy landscape though smaller in size (4/5 findings hold, 1 qualifies;
  `output/reports/career_flows_noisy_v1.md`).
- Which prestige stock weights the supervisor channel changes how fast the
  wedge opens, not where ($\rho = 0.965$-$0.993$ rank agreement across
  stocks; `output/reports/prestige_stock_v1.md`) --- the Goodhart target is the timing,
  not the location.
- Measurement discipline is in place and honest: the dedicated 1,520-pair
  re-measure pass (`output/reports/remeasure_v1.md`, 720 T1 pairs + 800 `W_v05`
  pairs) confirmed `0.0000` delta on every non-ideal reporter and isolated the
  pairing shift strictly to the ideal channel ($\Delta c_{\text{ideal,fin}} = -0.0056$
  on T1, $-0.0012$ on `W_v05`). The empirical moments remain a
  documented option menu (`docs/calibration_moments.yaml`), not targets.
- The niche is real and unoccupied: no published model combines landscape
  search, visibility-proxy incentives and partner selection scored against an
  ideal allocation (`docs/related_publications.md` v2.0, 46 verified studies).

## What this proposal does not rule out

The flagship Path B (explicit dyadic collaborator layer; implemented and Gate 2
probed in `output/reports/path_b_probe_v1.md`) remains open as the second paper or the extended
version; Path C (welfare/interventions, `P-0002`–`P-0004`, `P-0006`) and Path D (typology formalization, `P-0005`)
remain available as sections or companions; empirical calibration stays a
documented option for a later stage, not a commitment.

## Consequences if accepted

A workplan for drafting is spawned (figures from `output/report_notebook_v1/`
plus the re-measure results in `output/reports/paper_phase1_v1.md`; venue shortlist JEI / JASSS / Scientometrics per
`AdmiralsLog/way_forward.md`), and any behavioural commitment it makes spawns an
ADR citing this proposal under `Spawned:`. This record binds nothing until
ruled.

Do we proceed to draft the Phase-1 paper along this spine --- and if so,
which venue and authorship framing should the drafting target?