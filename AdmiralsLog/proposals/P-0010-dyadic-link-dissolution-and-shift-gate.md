---
title: "P-0010 - Path B v3: post-project link dissolution hazard and shift-based Gate 2 evaluation"
author: "Antigravity"
date: "2026-11-15"
tags: [proposal, operative, experiment, path-b, dyadic, link-dissolution, repeated-collaboration, gate-2]
project: "sciencemodels"
status: open
---

- **Status:** open
- **Class:** operative
- **Date:** 2026-11-15
- **Spark:** `output/reports/path_b_probe_v2.md` (Lines 195–279) and `CONT5-generational-turnover-and-paradigm-lockin.md` (Section 5): adding `dyadic_move_pull` solved `mgstd5` co-harvest team size ($7.32 \to 2.89$ vs. $2.65$ reference; 4/4 `mgstd5` contrasts pass) and coupled $\lambda$ to the prestige-merit wedge ($+0.035 \to -0.037$), but left two structural items open: (1) `repeated_collab_share` stayed pinned at $0.90\text{–}0.92$ (vs. $0.34$ empirical reference) because links never dissolve upon project completion, and (2) the mechanical distance-to-$2.65$ Gate 2 rule failed on `noisypr5` only because `noisypr5` already started at $2.67$ (distance $0.22$) and overshot to $2.32$.
- **Outcome target:** a ruling on whether to add an optional post-coharvest link dissolution rate (`dyadic_link_decay` / partner turnover) and evaluate Gate 2 via a paired-shift + tolerance band criterion.
- **Ruling:** pending
- **Spawned:** none

## The proposal

1. **Post-Coharvest Link Dissolution Hazard (`dyadic_dissolve_prob` $\in \{0.0, 0.3, 0.6\}$, default `0.0` bit-identical to Path B v2):**
   When two linked scientists co-harvest on the same tile at step $t$ ("complete a joint paper") or at the end of `dyadic_step`, dissolve their active link with probability `dyadic_dissolve_prob` (or require partner cooldown for $\tau_{\text{cool}}$ steps) so they must re-search or match with new collaborators rather than co-harvesting indefinitely on the same tile.
2. **Dual Team-Size & Gate 2 Reporting in `analyse_path_b`:**
   - Report both `mean_team_size` (co-harvest groups $\ge 2$, current definition) and `mean_paper_authors_incl_solo` (co-harvest groups $\ge 1$, matching the Klann per-paper author count where solo papers have size $1$).
   - Evaluate Gate 2 on both (a) the original distance-to-reference rule and (b) a **paired-shift + target-band criterion** (whether `dyadic_move_pull > 0` and $\lambda \to 1$ move team size and repetition monotonically in 8/8 cells and place the outcome within a stated tolerance band around $2.65$ and $0.34$).

## The stated reason

`path_b_probe_v2` proved that `dyadic_move_pull` successfully connects collaborator choice (`model.links`) to spatial co-production (`mean_team_size` and the prestige-merit wedge). The only reason `repeated_collab_share` remains stuck near $0.91$ is that `dyadic_step` retains links indefinitely as long as $U_{ij} \ge \text{bar}(i)$, and co-located partners pulling toward each other stay on the same tile step after step. A single post-coharvest link dissolution parameter tests whether finite project lifetimes bring `repeated_collab_share` into the empirical $0.30\text{–}0.45$ range without adding complex network rules.

## Experimental design & cost

- **Grid:** 3 `dyadic_dissolve_prob` (`0.0`, `0.3`, `0.6`) $\times$ 2 $\lambda$ (`0.0`, `1.0`) $\times$ 2 formulations (`logit`, `threshold`) $\times$ 2 landscapes (`mgstd5`, `noisypr5`) at `dyadic_move_pull = 1.0`, `credit = structural` = 24 cells $\times$ 20 seeds = **480 runs** at $H = 200$ (~4 minutes at `n_jobs=3`).
- **Deliverable:** `output/reports/path_b_probe_v3.md`.

Do we proceed with the 24-cell Path B v3 partner-turnover probe?
