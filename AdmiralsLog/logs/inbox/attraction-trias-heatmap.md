# Log inbox - task/attraction-trias-heatmap

Single-file entry for the coordinator to fold into
`AdmiralsLog/logs/2026-11.md` (ADR-0001 parallel-helper isolation, amended
2026-11-26). Suggested numbering: next free entry.

---

## 2026-11-26 --- F3 attraction-trias heatmap: `prestige_weight_in_visibility` knob + 90-cell ADR-0003 sweep (`attraction_heatmap_v1`)

**Summary.** Executed task `attraction-trias-heatmap` (Prompt F3) in worktree
`../wt-attraction-heatmap` (branch `task/attraction-trias-heatmap`), following
ADR-0001 (worktree isolation + inbox handoff + checkpointing/canary
discipline), ADR-0002 and ADR-0003. (i) Backward-compatible code extension:
`prestige_weight_in_visibility: float | None = None` on `MyModel.__init__`
(validated in $[0,1]$), resolved in `Scientist.compute_all_rewards` as
$d_{\text{eff}}(w) = (1-w) d_{\text{own}} + w d_{\text{own},P}$,
$V_{\text{eff}} = (\bar d_{\text{global}}+1)/(d_{\text{eff}}+1)$; `None` keeps
the pre-existing code path byte-for-byte. (ii) Checkpointed 2D sweep on the
ADR-0003 clock ($\gamma = 0.933$, $H = 200$, $N = 20$, mgstd5, 20 seeds/cell):
$\xi = 1-c$ (9 levels) $\times$ $w$ (5 levels) $\times$ tiers
$\{M0\ (g{=}0,\ \texttt{none}),\ M1\ (g{=}0.25,\ \texttt{fixed\_span},\
\tau_{\max}{=}80)\}$ = 90 cells $\times$ 20 seeds = 1,800 runs into
`data/attraction_heatmap_adr0003/` (gross 1,800 / net 1,800, wall 301.7 s).
(iii) 2x2 heatmap `output/report_notebook_v1/F_attraction_trias_heatmap.{png,pdf}`
with contour lines and computed iso-tilt annotations; report + compiled PDF
`output/reports/attraction_heatmap_v1.{md,pdf}`.

**Results.** (i) *Contours vertical in 4/4 panels* (iso-line angle
$0.33^\circ$–$0.63^\circ$ from vertical; $|b| \le 0.004$ vs $|a| = 0.08$–$0.46$
on normalized axes): the attraction trias is **proximity-dominated** — past-
prestige weighting barely moves either aggregate. (ii) The $\xi$ axis carries
the story: harvest $0.66$–$0.67$ flat to $\xi = 0.5$, kink at $0.7$–$0.8$,
collapse to $0.01$–$0.04$ at $\xi = 1$; alignment gradient $-0.297$
$[-0.443, -0.158]$ paired. (iii) *The $w$ lever binds at the herding corner*:
at $\xi = 1$, $w = 1$ vs $w = 0$ cushions the harvest collapse ($0.010 \to
0.042$, $\Delta = +0.032$ $[+0.021, +0.043]$) and flips $M0$ alignment from
$-0.023$ to $+0.031$ ($\Delta = +0.055$ $[+0.020, +0.108]$; $M1$ $+0.028$
$[+0.016, +0.039]$) — past prestige is a corner-corrective, not a contour-
tilter. (iv) Canonical turnover compresses alignment ($M1 - M0 = -0.232$
$[-0.363, -0.107]$ at $\xi = 0$) without changing the geometry ($\theta =
0.33^\circ$).

**Verification.** `tests/test_smoke.py` ALL PASS incl. new canary #14:
`None` bit-identity vs pre-edit digests (measured at branch point
`fb797fb`), $w{=}0.0 \equiv$ `use_visibility_reward=False`, $w{=}1.0 \equiv$
`True` (reports + full trajectories), $w{=}0.5$ binds cleanly, out-of-range
and NaN refused. Sweep in-run canaries all OK (determinism; drift probe vs
the stored anchor cell; $w$-extreme identity); one timeout kill resumed from
24/90 checkpoints with zero loss; atomic cell writes. Figure passes
`visual_qa.assert_figure_clean` **0 failures** (disclosed scope: layout
contract enforced on the heatmap canvas; the QA data contract does not see
mesh/contour collections, so data verification lives in the builder +
sidecar CSV). Report PDF via `compile-here.zsh` PASS. Commits by pathspec on
the task branch; nothing pushed.

**Notes.** The drift canary's first implementation false-positived twice
(int `80` vs float `80.0` after CSV round-trip of `weighted_90_step`; NaN
`gap_recent` on $M0$) — comparison signature fixed to numeric-aware exact
equality with NaN == NaN; no model-code change was needed or made.
`data/attraction_heatmap_adr0003/` stays untracked (run outputs).
