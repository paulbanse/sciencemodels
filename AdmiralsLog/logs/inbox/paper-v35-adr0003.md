# F1 — Paper v3.5: Figures & Prose Re-base on ADR-0003 (inbox summary)

**Branch:** `task/paper-v35-adr0003` (worktree `../wt-paper-v35`, base `jokoexplore` @ `fb797fb`).
**Note on the base:** the v3.4 paper edits (Standard Simulation Protocol, threefold
prose) were *uncommitted* in the main tree when this task started; their diff was
ported into the worktree explicitly (`git apply` of the main-tree diff), so v3.5
sits on v3.4 content rather than silently on v3.3. The concurrent session's
model-code and paper edits were never touched; no edits outside the worktree.

## Delivered

- **`scripts/runner.py`**: `vanishing_factor_for_prestige` now imports and uses
  `CANONICAL_VANISHING_FACTOR` (0.933) — the last entry point that still pinned 0.9.
- **Figure rebuilds (11, in place, from `data/*_adr0003/`)**, each with a
  parameterized `data_root` (ADR-0003 default; pre-ADR selectable) and an
  independent verifier recomputing every plotted value from the raw cells:
  - Fig 1 `F6_v2_two_stage_proxy_failure` ← `data/fig1_xi_adr0003/` (dense 11-point
    ξ, five tiers, 1,100 runs; 15/15 ledger anchors reproduce at 3 dp).
  - Fig 2 `F3_welfare_loss_vs_visibility_weight` ← `data/welfare_adr0003/`
    (two panels: terminal + cumulative; corr(ξ, loss) +0.684, corr(ξ, harvest)
    −0.691; entry-gradient flip −0.176 vs pre-ADR +0.166, qualification (i)).
  - Figs 5–6 `F1_v2_t1_postbug`, `F11_v2_t1_cell_matrix` ← `data/sensitivity_adr0003/`
    (18 cells: c × landscape × g; γ pinned 0.933; no vanishing tier, no g=0 cell —
    M0 control is Fig 1's tier; gradient −0.076 pooled / −0.006 mgstd5 / −0.146
    noisypr5; negatives 0/9 vs 2/9).
  - Fig 8 `FA2_v2_drop_decomposition` ← `data/interventions_adr0003/` (`intA`;
    time-mean decomposition; M0 reference reconstructed from Fig 1's archive).
  - Fig 9 `F2b_v2_career_flows_noisy_vs_clean` ← `data/career_flows_adr0003/`
    (mgstd5 only → exit-mode split replaces clean/noisy; per-capita harvest flat
    under `fixed_span` 0.0321→0.0324, falling under staggered 0.0168→0.0076,
    qualification (ii)).
  - Fig 10 `F8_v3_turnover_diagnostics` ← `data/turnover_adr0003/` (+0.0187 paired
    fixed−staggered, CI [+0.0090, +0.0284]; top10 0.205/0.235).
  - Fig 11 `F4_v2_prestige_stock_wedge_matrix` ← `data/prestige_adr0003/`
    (54 cells; Δr levels −0.0067/−0.0069/−0.0071; ordering intact at the mean and
    in 3/18 cells, reversed terminally, qualification (iii)).
  - Fig 14 `F13_paradigm_refill_regimes` ← `data/interventions_adr0003/` (`intC`;
    occupied −0.296, frontier −0.095; residual knowledge 199× baseline).
  - Fig 15 `F10a_m2_choice_to_production` ← `data/path_b_adr0003/` (18 cells,
    `logit + equal_split` only; links 2.44→15.34; teams −4.33; M1 reference from
    `welfare_adr0003`; formulation×credit cross retired).
  - Fig 16 `F10b_v2_m2_visibility_like_for_like` ← `data/fig1_xi_adr0003/`
    (five tiers, dense axis; staggered tier excluded from the team panel, N=121).
- **D5 prototypes into §3** as Figures 3–4 (`D5_spatial_snapshots`, `D5_temporal`),
  copied with their builder `scripts/fig_prototypes/build_snapshots_and_temporal.py`;
  all later figure numbers shift +2 (M2 figures are now 17–18).
- **Paper v3.5**: (a) §2.1–§2.3 define the two movement-attractiveness rules —
  unweighted `d_own` (`use_visibility_reward = False`) vs. `prestige_vanishing`-
  weighted `d_own,P` (`use_visibility_reward = True`, the model default behind every
  main-text figure; M2 multiplies in `dyadic_move_pull`) — plus a "Movement
  attractiveness rule" row in the §2.2 Tier Matrix and a matching column in the §8
  Figure-to-artifact map; (b) §§3–6 numbers re-based on `data/*_adr0003/` with the
  three qualifications folded in as computed annotations/counts; per-figure archive
  provenance replaces the blanket pre-ADR caveat; abstract re-stated; Revision Log
  v3.5 entry; 18-row map with updated builder/archive columns.
- **PDF recompiled** (`compile-here.zsh`: PASS, 32 pp) and **smoke test PASS**
  (`tests/test_smoke.py`, ADR-0003 canary green).

## Discrepancies and judgement calls (for the owner)

1. **The starvation verdict reverses on the ADR-0003 clock.** On the time-mean
   alignment, random dispersal now beats exact spawn at every (c,g) — starvation
   shares +39.7%…+235.4%, herding negative at high entry; on the terminal step the
   shares span −24.1%…+129.2%. The pre-ADR terminal rejection of `IE-0001` is
   retained as the pre-clock record; the text flags the entry-rate dependence as an
   open item rather than a settled mechanism. §4.2, the Fig 8 caption/table and the
   abstract were rewritten accordingly.
2. **Fig 15 alignment ordering corrected.** The lane's draft caption said λ=1 at
   pull=1 "lifts above the M1 reference"; its own numbers (λ=1 +0.072 vs M1
   reference +0.113) say λ=1 sits *below* M1 while beating the λ=0 arm (−0.083).
   The caption and §6 text now say exactly that.
3. **Fig 11 ordering is convention-sensitive.** Δr = c_local − c_ideal gives
   −0.0067/−0.0069/−0.0071; mean ordering `prestige > visibility > vanishing`
   holds at the pooled mean but in only 3/18 cells (Spearman 0.74–0.91) and
   reverses on the terminal measure. The paper states the counts, not a claim.
4. **Archive reductions disclosed in place:** T1 36→18 cells (no γ tier, no g=0),
   turnover 4→2 exit rules (none/sync pre-ADR only), career mgstd5-only, path_b
   logit+equal_split only, refill arm without per-agent prestige levels (native
   arrival gap instead of the normalized gap).
5. **Stale out-of-lane verifiers** (disclosed, not edited): `verify_figD.py`,
   `verify_figE.py` and the older `section_figures_v2.py` / `verify_section_figures_v2.py`
   aggregator still encode the pre-ADR inventories; they need a follow-up re-point.
6. **Cosmetic**: the PDF caption renderer doubles the label ("Figure 18: Figure 18:
   …") — inherited from alt-text plus caption prefix, pre-existing pipeline behaviour.
7. **Two stray `**` markers** (pre-existing link-then-bold artifacts, §5.2 and §6)
   broke the LaTeX build ("File ended while scanning use of \textbf") and were
   removed; no content lost.

## Verification trail

Verifiers run first-hand by the coordinator (exit 0): `verify_F6_v2`, `verify_F1_v2`,
`verify_F8_v3`, `verify_F10b_v2`, `verify_F2b_v2`, `verify_F4_v2`, `verify_FA2_v2`,
`make_f11_t1_matrix.verify()`, `round_report_figures.verify_welfare()`; smoke test
PASS; all 11 PNGs inspected visually; PDF markers checked (v3.5, Figures 3–4
prototypes, Figure-to-artifact map, Movement attractiveness rule).
