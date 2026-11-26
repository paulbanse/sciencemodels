---
title: "adr0003-gaps: D6/D3 missing arms (lottery, career-noisy leg, M2 n=40 check)"
date: 2026-11-26
branch: task/adr0003-gaps
worktree: wt-adr0003-gaps
status: done
---

# Prompt F2 summary (task/adr0003-missing-arms)

**Summary.** Filled the three cells D6/D3 omitted or underpowered, on the
ADR-0003 clock (gamma 0.933, g = 0.25, fixed_span, tau_max 80, H = 200):
(a) the interventions PhD lottery arm `alpha` {1.0, 0.5, 0.0} x `p_reloc`
{0.0, 0.02} (mgstd5, c = 0.5, 20 seeds) -> `data/interventions_lottery_adr0003/`;
(b) the career_flows noisypr5 leg `g` {0.125, 0.25, 0.5} x {fixed_span,
age_hazard_staggered} (c = 0.5, 20 seeds) -> `data/career_flows_noisy_adr0003/`;
(c) the M2 panel at n = 40 on mgstd5 (lambda {0, 1} x pull {0, 1} x credit
{structural, equal_split}, logit, plus the M1 baseline) ->
`data/path_b_m2_n40_adr0003/`. 21 cells, 600 runs, 0 failed seeds,
0 crashed cells, sources unchanged. No model file edited; the driver
subclass `GapModel` copies the two roulette bodies and is bit-identical to
`MyModel` at alpha = 1 (probe: 41 numeric columns, max diff 0.0).

**Results.** (i) Lottery: inert at c = 0.5 (alpha = 0 vs 1.0 pooled:
top10 +0.0034 [-0.0047, +0.0115], c_ideal_tm +0.0217 [-0.0056, +0.0491]);
only marginal effects are a small cumulative-loss increase at alpha = 0.5
without relocation (+0.0227 [+0.0010, +0.0443], p = 0.041) and a small
harvest decrease pooled (-0.0096 [-0.0189, -0.0003]). (ii) Career noisy:
per-capita harvest gradient negative under staggered turnover (-0.0033
[-0.0034, -0.0032]) and weakly negative under fixed_span (-0.0007
[-0.0009, -0.0004]); the noisy leg sides with the staggered branch of the
D6 flip, not with the mgstd5 fixed_span "flat" reading. (iii) M2 n = 40:
the lambda = 1 lift does NOT survive the ADR-0003 clock; it clears 95% as a
DROP: lambda = 1 pull = 1 sits -0.0504 [-0.0868, -0.0140] (structural) and
-0.0404 [-0.0777, -0.0030] (equal_split) below M1 on time-mean alignment
(paired over 40 seeds; Welch equal_split p = 0.104). The seeds-0-19 subset
already reads -0.0433 [-0.0909, +0.0044] / -0.0336 [-0.0834, +0.0163], so
the reversal against D3's old-clock +0.0652 [+0.0023, +0.1282] is the
clock/exit package, and n = 40 supplies the precision. The lambda ordering
survives (lambda = 1 0.086 vs lambda = 0 -0.069 at pull = 1); the terminal
metric stays flat (+0.0052 / +0.0214, ns). Chain at n = 40: co-movement
carries (96.6% / 117.9%), links inert (pull = 0 leg -0.0028).

**Verification.** Probe gate: smoke canary PASS, GapModel bit-identity PASS,
two seeds per arm schema-checked. Cross-archive canaries: the four
equal_split M2 cells at seeds 0-19 reproduce D6 `path_b_adr0003` exactly;
lottery alpha = 1 reproduces D6 `interventions_adr0003` intB exactly; max
abs difference 0.0 in all six checks (41 numeric columns each). Report
`output/reports/adr0003_gaps_v1.md` + PDF: md-style PASS, 4 pages, 0 words
past the right margin; every table number audited against the canonical
CSVs.

**Paths.** Committed on branch `task/adr0003-gaps`:
`scripts/adr0003_gaps/run_adr0003_gaps.py`,
`scripts/adr0003_gaps/analyse_adr0003_gaps.py`,
`output/reports/adr0003_gaps_v1.md` (+ PDF), this summary. Run outputs
(gitignored) under `data/interventions_lottery_adr0003/`,
`data/career_flows_noisy_adr0003/`, `data/path_b_m2_n40_adr0003/`,
`data/adr0003_gaps/` in the shared data tree.

**Deviation, disclosed.** The prompt's literal `ln -s <main>/data data/`
lands at `data/data` because the worktree checkout materialises 143 tracked
figure files under `data/`; the shared archive view was therefore built
with per-directory links from the worktree's `data/` to the main tree, and
the new archives were written to the main data tree. Merge:
`git merge task/adr0003-gaps` from `jokoexplore`. Nothing pushed.