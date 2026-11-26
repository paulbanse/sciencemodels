# Log inbox - task/attraction-trias-landscapes

Single-file entry for the coordinator to fold into
`AdmiralsLog/logs/2026-11.md` (ADR-0001 parallel-helper isolation, amended
2026-11-26). Suggested numbering: next free entry.

---

## 2026-11-26 --- F3b attraction-trias landscape check: iso-lines vertical on `mgstd5`, `MultG6` and `Noisy2` (`attraction_heatmap_landscapes_v1`)

**Summary.** Executed task `attraction-trias-landscapes` (Prompt F3b) in
worktree `../wt-attraction-landscapes` (branch
`task/attraction-trias-landscapes`, stacked on `task/attraction-trias-heatmap`),
following ADR-0001/-0002/-0003. (i) `run_attraction_heatmap.py` extended with
`--landscape {mgstd5,mg6,noisypr5}` (house bundles: `mg6` = `MultG6` =
multipleGaussians std 5 / 6 gaussians; `noisypr5` = `Noisy2` =
noisyGaussian `prop_random` 5, `std_curiosity` 0.1 — the modern bundle,
legacy 0.2 disclosed); `plot_attraction_heatmap.py` extended with the same
flag and landscape-named main titles. (ii) Two replicated 90-cell sweeps on
the ADR-0003 clock ($\xi \times w \times \{M0, M1\}$, 20 seeds):
1,800/1,800 runs into `data/attraction_heatmap_mg6_adr0003/` (wall 463.0 s)
and `data/attraction_heatmap_noisypr5_adr0003/` (wall 400.5 s), in-run
canaries OK on both. (iii) Three figures
(`F_attraction_trias_heatmap{,_mg6,_noisypr5}.{png,pdf}`; mgstd5 regenerated
with named title, reproducing F3's tilt values exactly) and the
cross-landscape report `output/reports/attraction_heatmap_landscapes_v1.{md,pdf}`.

**Results.** (i) *Proximity-dominated verdict replicates universally:
iso-lines vertical in 12/12 panels*, $\theta \in [0.22^\circ, 1.33^\circ]$
across all three landscapes and both tiers (mgstd5 $0.33^\circ$–$0.63^\circ$,
MultG6 $0.22^\circ$–$0.63^\circ$, Noisy2 $0.75^\circ$–$1.33^\circ$), every
one far inside the $10^\circ$ threshold. (ii) The $\xi$ gradient dominates
everywhere ($-0.30$ / $-0.71$ / $-0.68$ for $M0$ alignment, 3–13x the
corner $w$ effect); terrain composition sets levels, not geometry (harvest
plateau $0.666$ / $0.415$ / $0.277$; $M0$ alignment at $\xi{=}0$ $0.340$ /
$0.759$ / $0.775$). (iii) *The corner correction replicates and is
terrain-dependent*: at $\xi = 1$, $w{=}1$ vs $w{=}0$ rescues $M0$ alignment
on every landscape ($+0.055$ / $+0.038$ / $+0.209$, paired CIs excluding
zero) — most strongly on Noisy2, where pure proximity herding drives
alignment to $-0.174$ and $w{=}1$ brings it back to $+0.034$; under $M1$
turnover the rescue is significant only on mgstd5. (iv) Turnover
attenuation replicates ($M1{-}M0 = -0.232$ / $-0.504$ / $-0.458$ at
$\xi = 0$) on the same vertical geometry.

**Verification.** Both sweeps: gross = net 1,800/1,800 runs each, 0 failed;
determinism + $w$-extreme canaries OK; checkpointed per-cell (atomic
writes). All three figures pass `visual_qa.assert_figure_clean` **0
failures** (disclosed scope as in F3: layout contract enforced on heatmap
canvases; data verified in the builders against checkpoints, exact values in
`_data.csv` sidecars). Report PDF via `compile-here.zsh` PASS. The mgstd5
figure regeneration reproduced the F3 $\theta$ values exactly (regression
consistency). Commits by pathspec on the task branch; nothing pushed.

**Notes.** The F3 `data/attraction_heatmap_adr0003/` archive and its v1
manifest were read-only inputs (the runner keeps their directory name and
skips complete grids without rewriting manifests). "Noisy2" is a
bundle-level replication of the modern `noisypr5` landscape, not a bit-level
replay of the legacy 0.2-`std_curiosity` files.
