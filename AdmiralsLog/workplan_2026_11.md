---
title: "Workplan: Post-Review Consolidation, Paper v3.3, and Next Exploration Branches"
author: "Jakob Kapeller & Antigravity"
date: "2026-11-26"
version: 1.2
revision-vehicle: strikethrough
revision-renders: "md, pdf, html"
tags: [workplan, paper-v3.5, adr-0003, review, proxy-failure, sciencemodels]
project: "sciencemodels"
---

**Version 1.2** (27 November 2026)
**Version 1.1** (26 November 2026)
**Version 1.0** (26 November 2026)

This workplan consolidates the current project state after Paul's review commit
`f3d53a3` (`IE-0006` Option 5, `IE-0007`, `IE-0008`), incorporates the owner's
2026-11-26 rulings on **Gate A (`ADR-0003`: full 6-month step clock)** and
**Gate B (threefold goal/proxy distinction in Phase-1 paper; explicit selection
mechanism in `IE-0008`)**, integrates the archived
`AdmiralsLog/archive/way_forward.md` (v5.0) and
`AdmiralsLog/archive/HANDOVER-2026-11-14.md` (v1.2), and binds all open
proposals (`P-0001`, `P-0007`--`P-0011`) into one ordered register.

---

# 1. Resolved Design Gates (Ruled 2026-11-26)

- **Gate A --- Time-Scale Mapping (`ADR-0003`, RESOLVED):** Owner ruled full
  mechanical re-parameterization (Option A2) to a **6-month step clock**
  (`AdmiralsLog/decisions/accepted/ADR-0003-time-scale-anchoring.md`):
  1 step = 0.5 years, 40-year career = 80 steps (`TAU_MAX = 80`), canonical
  entry $g = 0.25$ (full $N=20$ renewal per 80-step generation), canonical
  retention $\gamma = 0.5^{1/10} \approx 0.933$ (5-year half-life), and
  `fixed_span` ($N \equiv 20$) as canonical constant-population `M1` turnover.
  Non-calibration ruling (2026-11-13) remains intact.
- **Gate B --- Goal vs. Proxy & Reproduction Metric (RESOLVED):** Phase-1 paper
  retains the current reproduction rule and makes the **threefold distinction**
  ($M^*$ global goal vs. `prestige` historical output proxy vs. `visibility`
  pure attention proxy) explicit in §2.1--§2.3. `IE-0008` is expanded with an
  explicit three-part Darwinian selection mechanism (fitness $W_i$, heritable
  curiosity + mutation, `fixed_span` turnover closure) before scoping `P-0012`.

---

# 2. High-Priority Items

## 2.1 Governance & Record Hygiene (`f3d53a3` Cleanup) --- *[COMPLETED: H1, `2d32e49`]*

1. ~~**Rename `IE-0007` without spaces**~~ *(Done in `2d32e49`.)*
2. ~~**Fix `IE-0008` frontmatter title**~~ *(Done in `2d32e49`.)*
3. ~~**Repair `IE-0006` provenance metadata**~~ *(Done in `2d32e49`.)*
4. ~~**Update register counters (`IE-0009`, `ADR-0004`)**~~ *(Done: `2d32e49` + 2026-11-26.)*
5. ~~**Append code-verified assessment notes to `IE-0006`, `IE-0007`, `IE-0008`**~~ *(Done in `2d32e49`; `IE-0008` selection mechanism expanded 2026-11-26.)*

## 2.2 Immediate Manuscript Credibility Fixes (`paper_sketch_v1.md` v3.3 $\to$ v3.4)

6. ~~**Audit and resolve §3.2 / Figure 3 (`use_distance = 0` "jump search" bug)**~~ *(Done by H2 in v3.3: §3.2 / old Figure 3 removed, figures renumbered 1--16.)*
7. ~~**Disambiguate `vanishing_factor_for_prestige` everywhere & fix $g$ definition**~~ *(Done by H2 in v3.3: $P_{\text{van}}(t)=\gamma P_{\text{van}}(t-1)+\Delta P(t)$ retention wording and global $g$ formula.)*
8. ~~**Scrub unjustified `"lab"` / `"laboratories"` wording**~~ *(Done by H2 in v3.3.)*
9. ~~**Add "Standard Simulation Protocol" (`ADR-0003` clock) + explicit threefold
   Goal/Proxy distinction (§2.1--§2.3)**~~ *(Done by D2 in v3.4; §6/§7 updated with D3 shares & `IE-0006` pointer; $d_{\text{own},P}$ definition queued in F1.)*

## 2.3 ADR-0003 Mechanical Re-Parameterization & Re-Run Wave

10. ~~**Implement `ADR-0003` core clock & `P-0009` turnover in `sciencemodels/`**~~
    *(Done by D1 in `8b36a72`; `prestige_weight_in_visibility` added by F3 in `c2b94eb` on `task/attraction-trias-heatmap`.)*
11. **Re-run Phase-1 archives under `ADR-0003` clock & rebuild Paper v3.5** ---
    *Core re-run DONE (D6 `4e3c301`, 6,380 runs, 16/19 robust; F2 `08397f8`, 600 runs; F3 `c2b94eb`, 1,800 runs). Paper v3.5 figure/prose rebuild (`F1`, `task/paper-v35-adr0003`) in flight; must incorporate F2's `M2` $\lambda=1$ sign flip ($-0.0404$ below `M1` on `ADR-0003`) and F3's vertical-contour heatmap ($<0.7^\circ$ tilt).*

## 2.4 High-Leverage Figure & Methodological Upgrades (`paper_sketch_v1.md` v3.5)

12. ~~**Densify Figure 1 along $\xi$ + add 3-panel spatial grid snapshots**~~ ---
    *(Dense 11-point $\xi$ data banked in `data/fig1_xi_adr0003/`; D5 3-panel `Noisy2` snapshot + temporal curves built; integration into v3.5 in F1.)*
13. ~~**Add missing temporal visualizations**~~ --- *(D5 `D5_temporal_Noisy2_H200_v0p933_g0_n20.{png,pdf}` built; integration into v3.5 in F1.)*
14. ~~**Extend Figure 9's counterfactual ablation-share decomposition to `M2`**~~
    *(Done by H4 `681497d` + independently reproduced by D3 `fb797fb` + re-checked at $n=40$ on `ADR-0003` by F2 `08397f8`.)*
15. ~~**2D Attraction-Trias Heatmap ($\xi \times w$)**~~ *(Done by F3 `c2b94eb`, 1,800 runs: iso-lines vertical $<0.7^\circ$ in all 4 panels; ready to merge & embed in v3.5.)*

## 2.5 Scoped New Research Branches (Separate from Phase-1 Paper Closure)

16. ~~**Prototype `IE-0006` Option 5 (Gaussian neighbour refill) + peak tracking + `P-0007` sabbaticals**~~
    *(Done by H3 `f459ce7` [720 runs] + D4 `4d9da1a` [360 runs @ $H=800$]: all 6 arms flat diffusion [directedness 0.023--0.032], sabbaticals saturate breadth but leave throughput flat at $0.0049$/step vs `frontier` $0.343$/step. Full findings recorded in `IE-0006`, status **`deferred`**.)*
17. **Draft `P-0012` (Evolutionary curiosity selection from `IE-0008`)** ---
    *Selection mechanism expanded in `IE-0008` (2026-11-26); ready to scope as `P-0012` outside Phase-1 paper.*
18. **Governance: Draft Two-Admirals ADR (`ADR-0004`)** --- *Awaiting owner ruling to formalize co-command (Jakob Kapeller & Paul Banse).*

---

# 3. Integrated Carry-Over from Prior Plans & Open Decisions

This section integrates all remaining open items from
`AdmiralsLog/archive/way_forward.md` (v5.0),
`AdmiralsLog/archive/HANDOVER-2026-11-14.md` (v1.2), `IDEA.md`, and
`AdmiralsLog/proposals/README.md`.

## 3.1 Open Proposals Awaiting Owner Ruling (`AdmiralsLog/proposals/`)

| ID | Topic | Class | Status | Relation to This Plan |
| --- | --- | --- | --- | --- |
| **`P-0001`** | Phase-1 paper synthesis (`drafts/paper_sketch_v1.md`): venue & authorship framing (JEI / JASSS / Scientometrics) | strategic | exercised (awaiting ruling) | Determines how much of §2.2--§2.4 enters v3.4 before submission |
| **`P-0007`** | Sabbatical shocks $\times$ frontier paradigm refills (Path C v2) | operative | open | Natural home to pair `IE-0006` Option 5 (Gaussian neighbour refill) with sabbaticals |
| **`P-0008`** | Path D v2: stock-level epistemic destruction & normalized superstar counterforce (`typology_v2`) | strategic | open | Builds on Gate B proxy-goal taxonomy |
| **`P-0009`** | Core demographic turnover promotion (`exit_mode`, `gap_recent`) into `sciencemodels/model.py` | operative | **adopted via `ADR-0003`** | Implemented directly in Workplan §2.3 item #10 |
| **`P-0010`** | Path B v3: post-project link dissolution hazard & shift-based Gate 2 evaluation (`path_b_probe_v3`) | operative | open | Independent of Phase-1 closure; tests whether `M2` clears Gate 2 |
| **`P-0011`** | Canonical three-tier specification (`M0` / `M1` / `M2`) & 24-cell `M0 + M2` ($g=0$) isolation probe | strategic / operative | open | Clarifies horizontal vs. vertical channels |

## 3.2 Carried-Over Items from Archived Plans and `IDEA.md`

- **Decision Gates (`archive/way_forward.md`):**
  - *Gate 1* (wedge + concentration): evaluated model-internally in Waves 1--4;
    empirical calibration remains deferred by owner ruling (2026-11-13).
  - *Gate 2* (Path B dyadic layer): awaits `P-0010` to decide flagship `B+A` vs.
    `A+D` with `M2` as robustness.
  - *Gate 3* (held-out dynamic falsification against Klann event study):
    deferred until/unless empirical calibration is lifted.
- **Deferred Path B Menu Items (`archive/HANDOVER-2026-11-14.md` §3.1):**
  - `Q9` (what plays $AP_F$ in Path D) $\to$ folded into `P-0008`.
  - `Q10` (time scale: 1 paper/link = ? steps) $\to$ **resolved by `ADR-0003`** (1 step = 6 months).
  - `Q11` (confirm moments remain diagnostic, not calibration targets) $\to$ reconfirmed.
- **Repository & Environment Gaps (`IDEA.md`):**
  - Branch and PR policy for merging `jokoexplore` into `main` (owner action; agents never push).
  - Regenerate exact `requirements.txt` pins for `joblib` and `pandas` from the host `.venv`.
  - Untracked scratch log `scripts/bug0006_wave2/run_log.txt`: confirm whether to delete or ignore.

---

# Revision Log

- **Version 1.2** (27 November 2026) --- Marked D1 (`8b36a72`), D2 (v3.4), D3/H4 (`681497d`/`fb797fb`), D4 (`4d9da1a`), D5, D6 (`4e3c301`), F2 (`08397f8`), and F3 (`c2b94eb`) completed; updated §2.3/§2.4 with F2's `M2` $\lambda=1$ sign flip and F3's vertical-contour heatmap; added proposed `ADR-0004` (Two Admirals co-command).
- **Version 1.1** (26 November 2026) --- Recorded owner rulings on Gate A
  (`ADR-0003`: full 6-month step clock) and Gate B (threefold goal/proxy
  distinction in Phase-1 paper; explicit selection mechanism added to
  `IE-0008`); archived superseded `way_forward.{md,pdf}` and
  `HANDOVER-2026-11-14.{md,pdf}`; marked H1 (`2d32e49`), H2 (`paper_sketch_v1`
  v3.3), and H3 (`refill_gaussian_v1`, 720 runs) completed; added §2.3
  (`ADR-0003` mechanical re-parameterization & re-run wave).
- **Version 1.0** (26 November 2026) --- Initial consolidated workplan after
  commit `f3d53a3` (`IE-0006` Option 5, `IE-0007`, `IE-0008`), integrating open
  proposals `P-0001` and `P-0007`--`P-0011` plus carried-over items from
  `way_forward.md` (v5.0), `HANDOVER-2026-11-14.md` (v1.2), and `IDEA.md`.
