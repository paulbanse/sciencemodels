---
title: "ADR-0003 - Canonical time-scale anchoring (6-month step clock)"
author: "Jakob Kapeller"
date: "2026-11-26"
tags: [adr, time-scale, demography, retention, turnover, parameters]
project: "sciencemodels"
---

- **Status:** accepted (ruled 2026-11-26 from Gate A discussion of `IE-0007` #1 and #2)
- **Date:** 2026-11-26
- **Supersedes:** -
- **Related:** `IDEA.md`, `AdmiralsLog/workplan_2026_11.md`, `AdmiralsLog/ideas/IE-0007-draft-review-proposals.md`, `AdmiralsLog/proposals/P-0009-core-turnover-and-steady-state-interventions.md`, `drafts/paper_sketch_v1.md`
- **Spawned:** ADR-0003 model re-parameterization & sweep re-run wave (Workplan v1.1 §2.5)

## Context

Prior simulation rounds used an unanchored step clock: `age` incremented by
`+1` per step from $23$ (implicitly 1 year), `vanishing_factor_for_prestige`
defaulted to $\gamma = 0.9$ (half-life $6.6$ steps), `agent_generation_time`
swept $g \in \{0.05, 0.1, 0.2\}$ (global newcomers/step, renewing $N=20$ in
$400/200/100$ steps), and `age_hazard` used `TAU_MAX = 200` steps. In
`IE-0007` (#1--#2), Paul proposed anchoring the model clock to real academic
career scales (**1 step = 6 months**) so that career span, generational
turnover, and citation-prestige decay rest on a coherent, transparent narrative
without empirically calibrating the model (preserving the 2026-11-13
no-calibration ruling).

On 2026-11-26, the owner ruled to implement Paul's time-scale anchoring
**completely** (mechanical re-parameterization, Option A2) rather than as a
post-hoc interpretive gloss.

## Decision

Every new canonical simulation and the re-measured Phase-1 paper archives adopt
the **6-month step clock**:

1. **Clock unit:** $1\text{ step} = 0.5\text{ academic years}$ (6 months).
   Agent age increments by $0.5$ years per step (or is tracked in half-year
   steps from entry age $25$ to retirement at $65$).
2. **Career span & retirement (`exit_mode`):** A full academic career spans
   $40\text{ years} = \mathbf{80\text{ steps}}$ (`TAU_MAX = 80`). Constant-population
   replacement (`fixed_span`, $N \equiv 20$, oldest-out) is promoted to the
   canonical `M1` turnover baseline alongside `none` (immortal-founder control)
   and `age_hazard_staggered` (`TAU_MAX = 80`).
3. **Generational entry rate ($g$):** Full renewal of $N = 20$ scientists over
   one 80-step generation sets the canonical entry rate to
   $$g_{\text{canon}} = \frac{20}{80} = \mathbf{0.25\text{ newcomers/step}}$$
   (with sensitivity sweep $g \in \{0.125, 0.25, 0.5\}$, i.e., $0.5\times$,
   $1\times$, $2\times$ generational renewal).
4. **Prestige retention factor ($\gamma$ = `vanishing_factor_for_prestige`):**
   Defined by $P_{\text{van}}(t) = \gamma P_{\text{van}}(t-1) + \Delta P(t)$.
   A 5-year prestige half-life ($10\text{ steps}$) fixes the canonical retention
   factor at
   $$\gamma_{\text{canon}} = 0.5^{1/10} \approx \mathbf{0.933}$$
   (sensitivity sweep $\gamma \in \{0.871\text{ (2.5 yr)}, 0.933\text{ (5 yr)}, 1.0\text{ (permanent)}\}$).
5. **Simulation horizons ($H$):** Standard horizon $H = 200$ steps
   ($100\text{ years}$, $2.5$ academic generations); long-run steady-state
   turnover horizon $H = 400$ steps ($200\text{ years}$, $5$ generations).
6. **Non-calibration boundary preserved:** These values are conceptual
   scale anchors for narrative coherence, **not** empirical fits to the Klann
   panel (owner ruling 2026-11-13 remains in force).

## Consequences

- Core defaults in `sciencemodels/model.py` and `sciencemodels/sweeps.py`
  (plus `P-0009` turnover promotion) must be updated under a dedicated
  code-change PR/commit with a refreshed `tests/test_smoke.py` canary anchor
  (recording the pre-ADR-0003 canary in the log).
- Affected Phase-1 archives (`T1`, `W_v05`, `career_flows`, `prestige_stock`,
  `turnover`, `interventions`, `path_b`) must be re-run under the ADR-0003
  clock into new `data/*_adr0003/` directories (never overwriting pre-ADR-0003
  archives) and `drafts/paper_sketch_v1.md` updated with a "Standard Simulation
  Protocol" section and refreshed figures.
- All pre-ADR-0003 archives remain intact as the historical baseline for
  before/after sensitivity verification.

## Enforcement

No new sweep or manuscript figure after 2026-11-26 may mix pre-ADR-0003
($\gamma=0.9$, $g=0.1$, `TAU_MAX=200`) and post-ADR-0003 ($\gamma=0.933$,
$g=0.25$, `TAU_MAX=80`) runs in the same figure or table without explicit
disclosure.
