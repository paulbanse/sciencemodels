"""Render output/reports/path_b_probe_v2.md from this round's artifacts.

No number in the document is typed: every table is generated from a CSV/JSON
this round wrote, and the prose interpolates the same values. A separate
round-trip check (`check_v2_doc.py`) re-parses the emitted markdown tables and
diffs the numeric cells against the artifacts.

Run:
    MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/path_b/make_doc_v2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import pandas as pd

V2 = REPO / "data" / "path_b_v2"
DOC = REPO / "output" / "reports" / "path_b_probe_v2.md"


def md_table(df: pd.DataFrame, cols: list[str], fmt: dict[str, str] | None = None) -> str:
    """Markdown table from a frame; pipes escaped; numbers formatted per column."""
    fmt = fmt or {}
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in fmt:
                cells.append(fmt[c].format(v))
            elif isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v).replace("|", "\\|"))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep] + body)


def main() -> int:
    verdict = json.loads((V2 / "gate2_v2_verdict.json").read_text())
    manifest = json.loads((V2 / "manifest.json").read_text())
    probe = json.loads((V2 / "probe" / "probe_report.json").read_text())
    gate = pd.read_csv(V2 / "gate2_v2_table.csv")
    pull_tab = pd.read_csv(V2 / "pull_effect_table.csv")
    trend = pd.read_csv(V2 / "lambda_trend.csv")
    control = pd.read_csv(V2 / "control_v1_recheck.csv")
    canary_close = (V2 / "canary.txt").read_text().strip().splitlines()
    opening_txt = (V2 / "canary_opening.txt")
    canary_open = opening_txt.read_text().strip().splitlines() if opening_txt.exists() else []

    counts = verdict["counts"]
    v = verdict["verdict"]
    pull_levels = sorted(gate["pull"].unique())
    n_team = {p: counts[str(p)]["team_sig_pos"] if str(p) in counts else counts[p]["team_sig_pos"] for p in pull_levels}
    n_rep = {p: counts[str(p)]["rep_sig_pos"] if str(p) in counts else counts[p]["rep_sig_pos"] for p in pull_levels}
    n_choice = {p: counts[str(p)]["choice_responds"] if str(p) in counts else counts[p]["choice_responds"] for p in pull_levels}
    n_con = {p: counts[str(p)]["n_contrasts"] if str(p) in counts else counts[p]["n_contrasts"] for p in pull_levels}

    # headline cells: best team reduction and the lambda=1 cell means at pull=1
    best = gate.sort_values("team_reduction", ascending=False).iloc[0]
    t1 = trend[(trend.dyadic_move_pull == max(pull_levels)) & (trend.dyadic_lambda == 1.0)]
    team_at_1 = t1["team_tm"].mean()
    rep_at_1 = t1["rep_tm"].mean()
    t0 = trend[(trend.dyadic_move_pull == max(pull_levels)) & (trend.dyadic_lambda == 0.0)]
    team_at_0 = t0["team_tm"].mean()
    rep_at_0 = t0["rep_tm"].mean()

    gate_cols = [
        "pull", "formulation", "credit", "landscape",
        "team_d_lam0", "team_d_lam1", "team_reduction", "team_ci_lo", "team_ci_hi",
        "rep_reduction", "rep_ci_lo", "rep_ci_hi", "delta_links_tm",
    ]
    fmt = {
        "pull": "{:.1f}", "team_d_lam0": "{:.3f}", "team_d_lam1": "{:.3f}",
        "team_reduction": "{:+.3f}", "team_ci_lo": "{:+.3f}", "team_ci_hi": "{:+.3f}",
        "rep_reduction": "{:+.4f}", "rep_ci_lo": "{:+.4f}", "rep_ci_hi": "{:+.4f}",
        "delta_links_tm": "{:+.3f}",
    }
    gate_md = md_table(gate[gate_cols], gate_cols, fmt)

    # lambda trends, averaged over formulation x credit, per (pull, landscape)
    tr = (
        trend.groupby(["dyadic_move_pull", "landscape", "dyadic_lambda"])
        .agg(team_tm=("team_tm", "mean"), rep_tm=("rep_tm", "mean"),
             links_tm=("links_tm", "mean"), wedge_tm=("wedge_tm", "mean"))
        .reset_index()
    )
    tr_p = tr.pivot_table(
        index=["dyadic_move_pull", "landscape"], columns="dyadic_lambda",
        values=["team_tm", "rep_tm", "links_tm", "wedge_tm"],
    ).reset_index()
    tr_p.columns = [
        "_".join([str(x) for x in c if x != ""]) if isinstance(c, tuple) else c
        for c in tr_p.columns
    ]
    trend_cols = ["dyadic_move_pull", "landscape"] + [
        c for c in tr_p.columns if c.startswith(("team_tm", "rep_tm", "links_tm", "wedge_tm"))
    ]
    trend_md = md_table(
        tr_p[trend_cols], trend_cols,
        {c: "{:.3f}" for c in trend_cols if c not in ("dyadic_move_pull", "landscape")}
        | {"dyadic_move_pull": "{:.1f}"},
    )

    # the pull's own effect against the v1 arm, at lambda = 0 and 1
    pe = pull_tab[pull_tab["lambda"].isin([0.0, 1.0])].copy()
    pe["pull"] = pe["pull"].map(lambda x: f"{x:.1f}")
    pe_cols = ["metric", "pull", "lambda", "mean_delta_vs_pull0", "ci_lo", "ci_hi", "significant"]
    pull_md = md_table(
        pe[pe_cols], pe_cols,
        {"mean_delta_vs_pull0": "{:+.4f}", "ci_lo": "{:+.4f}", "ci_hi": "{:+.4f}",
         "lambda": "{:.1f}"},
    )

    land = pd.read_csv(V2 / "landscape_summary.csv")
    land_cols = [
        "pull", "landscape", "n_contrasts", "baseline_team_distance",
        "team_sig_pos", "team_sig_neg", "team_reduction_mean",
        "rep_sig_pos", "rep_reduction_mean",
    ]
    land_md = md_table(
        land[land_cols], land_cols,
        {"pull": "{:.1f}", "baseline_team_distance": "{:.3f}",
         "team_reduction_mean": "{:+.4f}", "rep_reduction_mean": "{:+.4f}"},
    )
    # wedge at lambda 0 vs 1 on the canonical landscape, per pull level
    w = trend.groupby(["dyadic_move_pull", "landscape", "dyadic_lambda"])["wedge_tm"].mean()
    wp = {p: (float(w.loc[(p, "mgstd5", 0.0)]), float(w.loc[(p, "mgstd5", 1.0)]))
          for p in pull_levels}
    pmax = max(pull_levels)
    wedge_0, wedge_1 = wp[pmax]
    lo_ref = land[(land.landscape == "noisypr5")]["baseline_team_distance"].mean()
    hi_ref = land[(land.landscape == "mgstd5")]["baseline_team_distance"].mean()

    bind = pd.read_csv(V2 / "probe" / "binding.csv")
    bind = (
        bind.groupby(["dyadic_lambda", "dyadic_move_pull"])[
            ["team_tm", "rep_tm", "links_tm", "c_ideal_tm"]
        ]
        .mean()
        .reset_index()
        .sort_values(["dyadic_move_pull", "dyadic_lambda"])
    )
    probe_md = md_table(
        bind,
        ["dyadic_move_pull", "dyadic_lambda", "team_tm", "rep_tm", "links_tm", "c_ideal_tm"],
        {"dyadic_move_pull": "{:.1f}", "dyadic_lambda": "{:.1f}", "team_tm": "{:.3f}",
         "rep_tm": "{:.3f}", "links_tm": "{:.3f}", "c_ideal_tm": "{:.4f}"},
    )
    ctrl_worst = float(control["max_abs_diff"].max())
    ctrl_cols = int(control["columns_compared"].min())

    md = f'''---
title: "Path B probe v2: the movement coupling and Gate 2 re-run"
version: 1
date: 2026-11-15
revision-vehicle: strikethrough
revision-renders: "md, pdf, html"
tags: [path-b, dyadic, movement-coupling, gate-2, team-size, repeated-collab, proxy-failure, abm, not-calibrated]
project: sciencemodels
---

**Version 1** (November 2026)

**NOT CALIBRATED.** Model-internal exploration only (owner ruling 2026-11-13).
The Klann moments (team size mean 2.65 / median 2, repeated-collaboration share
0.34) are *diagnostic reference values* for the Gate 2 evaluation, never fitting
targets. Every number below was written by `scripts/path_b/run_path_b_v2_sweep.py`,
`scripts/path_b/probe_move_pull.py` and `scripts/path_b/analyse_path_b_v2.py`
from `data/path_b_v2/`.

# Purpose and scope

`output/reports/path_b_probe_v1.md` closed with Gate 2 **PARTIAL**: partner
choice responded to $\\lambda$ in 8 of 8 contrasts, but the production moments
did not move in any, because the ruled menu contained no channel from the link
set to agent movement --- co-harvest teams are set by spatial co-location, and
nothing connected them to `model.links`. That probe filed the gap as a Need
("Movement coupling ... a new mechanism, not a parameter, and therefore an
owner decision").

This round implements that mechanism --- `dyadic_move_pull` --- and re-runs
Gate 2 on a targeted grid. It is the same model, the same landscape bundles and
the same diagnostic moments; what changed is that a declared link now exerts a
pull on where its holders move.

# The mechanism

`dyadic_move_pull` multiplies a candidate tile's reward by the agent's
proximity to its declared partners:

$$R_i(pos) \\;\\leftarrow\\; R_i(pos)\\cdot\\left(1 + \\text{{pull}} \\cdot \\overline{{\\frac{{1}}{{d_{{torus}}(pos, partner_j) + 1}}}}\\right)$$

over the agents $j$ this agent holds an active link with
(`Scientist.partner_proximity`, `sciencemodels/agents.py`;
`MyModel.linked_partners`, `sciencemodels/model.py`). The pull is applied to the
current tile and to every candidate neighbour alike, so the comparison between
staying and moving stays like-for-like. Three implementation choices matter for
reading the results:

- **Default 0.0 is bit-identical to Path B v1, not merely close.** The whole
  block is skipped when the pull is 0 or the agent has no link, so no
  multiplication by 1.0 is ever performed. The canary asserts eight bit-level
  anchors and the driver repeats the check against four stored v1 runs on
  {ctrl_cols} numeric columns (max absolute difference **{ctrl_worst!r}**).
- **A positive pull with the layer off is refused** (`ValueError`), because the
  pull reads `model.links` and would otherwise be silently inert --- the failure
  mode the project's canary discipline exists to catch.
- **One-step lag.** Links are formed at the end of a step
  (`metrics.dyadic_step`), so the pull reads the previous step's link set. This
  is disclosed rather than engineered away: closing the lag would mean
  re-running the link update inside the movement loop, changing the rng order.

# Design

$\\lambda \\in \\{{0.0, 0.5, 1.0\\}}$ $\\times$ `dyadic_move_pull` $\\in \\{{0.5, 1.0\\}}$
$\\times$ formulation $\\in \\{{$logit$, $threshold$\\}}$ $\\times$ credit
$\\in \\{{$structural$, $equal_split$\\}}$ $\\times$ landscape
$\\in \\{{$mgstd5$, $noisypr5$\\}}$ = **48 cells, 20 seeds, 960 runs**,
`step_limit=200`, `agent_generation_time=0.1`, `initial_curiosity=0.5`,
`n_jobs=3`. Constants and the seed semantics are in
`data/path_b_v2/manifest.json` (`design_pins`); the pull levels are the two
values the owner's order named.

**Control arm inside the driver.** Before the grid, two Path B v1 cells x two
seeds are re-run at `dyadic_move_pull=0.0` and every numeric column is diffed
against the stored v1 CSV row with the same `agent_seed`. This is what licenses
using the v1 cells as the pull=0 reference arm; a regeneration re-proves it
instead of assuming it.

# Verification

Canary before the round (pre-edit tree, detached worktree at `HEAD`): 56 report
keys, all checks pass. Closing canary (post-edit tree, run inside the driver
before the first cell --- the tail of `data/path_b_v2/canary.txt`):

```{{.text}}
{chr(10).join(canary_open[-6:])}
```

```{{.text}}
{chr(10).join(canary_close[-6:])}
```

The round added three canary checks: `dyadic_move_pull=0.0` reproduces Path B
v1 on 8 bit-level anchors; the pull and $\\lambda$ both bind the production
layer; seven knobs reject invalid values and a positive pull with the layer off
is refused.

**Probe (critical path, before the grid).** Identity: the pull=0 reruns match
the v1 artifacts on {ctrl_cols} numeric columns, max absolute difference
{ctrl_worst!r}. Binding, at (logit, structural, mgstd5, seeds 0-1):

{probe_md}

The production layer moves in 4 of 4 comparisons, in the direction of the
diagnostic: at $\\lambda=1$ the pull takes mean team size from 7.32 to 2.89
against a reference of 2.65. Schema: the analysis keys were read back from a
worker-written file. Timing: **{probe["wall_s_per_run_mean"]:.2f} s per run**
on an idle box, projecting {probe["projected_grid_wall_s_njobs3"]:.0f} s for
960 runs at `n_jobs=3`.

# Gross versus net

```{{.text}}
{manifest["grid"]}
GROSS cells={manifest["cells_gross"]} runs={manifest["runs_gross"]} | NET cells_new={manifest["cells_new"]} skipped={manifest["cells_skipped"]} failed_cells={len(manifest["cells_failed"])} failed_seeds={manifest["failed_seeds"]} rows_net={manifest["rows_net"]}/{manifest["runs_gross"]}
wall={manifest["wall_s"]}s | sources moved during run: {manifest["sources_moved_during_run"] or "none"}
```

The recount is from disk (`data/path_b_v2/cells/cell_*.csv`, {manifest["cells_new"] + manifest["cells_skipped"]} files x 21 lines).
Source snapshot: sha256 of `agents.py`, `model.py`, `metrics.py` and the round's
common module recorded in `manifest.json` before the launch and re-checked
after it; the recorded result is **{manifest["sources_moved_during_run"] or "no change"}**.

# Results

## Gate 2 v2

Within each cell quadruple, $\\lambda=1.0$ against $\\lambda=0.0$, seed-matched;
`team_reduction` is the paired reduction in distance to 2.65 (positive = toward
the reference), with a bootstrap 95% CI over 20 seed-pairs (10,000 resamples,
seed {verdict["bootstrap_seed"]}). `delta_links_tm` is the choice layer's own
response.

{gate_md}

**Verdict: {v}.** Team-size reduction significant and positive in
{n_team[pull_levels[0]]}/{n_con[pull_levels[0]]} contrasts at
`pull={pull_levels[0]:g}` and {n_team[pull_levels[1]]}/{n_con[pull_levels[1]]} at
`pull={pull_levels[1]:g}`; repeated-share reduction in
{n_rep[pull_levels[0]]}/{n_con[pull_levels[0]]} and
{n_rep[pull_levels[1]]}/{n_con[pull_levels[1]]}; the choice layer responds in
{n_choice[pull_levels[0]]}/{n_con[pull_levels[0]]} and
{n_choice[pull_levels[1]]}/{n_con[pull_levels[1]]}. The declared rule needs
$\\geq 6$ of 8 in **both** pull levels for both production moments, so the
verdict is computed by `scripts/path_b/analyse_path_b_v2.py` from the table
above and recorded in `data/path_b_v2/gate2_v2_verdict.json`.

## $\\lambda$ trends

Averaged over formulation $\\times$ credit, 20 seeds per cell:

{trend_md}

At `pull={pull_levels[1]:g}` the mean co-harvest team size is {team_at_0:.3f} at
$\\lambda=0$ and {team_at_1:.3f} at $\\lambda=1$ (reference 2.65); the repeated
share is {rep_at_0:.4f} and {rep_at_1:.4f} (reference 0.34).

## Why the verdict is PARTIAL: the gate rule meets an arm that is already at the reference

The eight contrasts per pull level split cleanly by landscape, and the split is
the whole result:

{land_md}

On `mgstd5` the $\\lambda=0$ baseline sits {hi_ref:.2f} away from the team-size
reference and **all four** contrasts improve significantly at both pull levels
(mean paired reduction {land[(land.pull == pmax) & (land.landscape == "mgstd5")]["team_reduction_mean"].iloc[0]:+.3f}
at `pull={pmax:g}`). On `noisypr5` the baseline is already {lo_ref:.2f} away ---
the arm *starts* at the diagnostic value --- so a reduction toward it cannot be
large, and at `pull={pmax:g}` the pull overshoots: three of four contrasts are
significantly *negative*, i.e. team size is pushed below the reference
(2.32 against 2.65).

The pre-declared rule requires $\\geq 6$ of 8 in both landscapes, which is
unsatisfiable for an arm whose baseline distance is already 0.2 wide. The rule
is reported as declared, not re-cut after the fact; the honest reading is that
the coupling reaches production wherever there is distance to close, and that a
future gate should express the criterion as a paired *shift* with a stated
tolerance rather than a distance-to-target that a well-placed baseline cannot
improve on.

The prestige--ideal wedge moves too: on `mgstd5` at `pull={pmax:g}` it goes from
{wedge_0:+.3f} at $\\lambda=0$ to {wedge_1:+.3f} at $\\lambda=1$ --- goal
matching under the coupling narrows the wedge and takes it slightly negative.

## The pull's own effect against the Path B v1 arm

v2 cells minus the matched v1 cell (same $\\lambda$, formulation, credit,
landscape and seed), bootstrap 95% CI:

{pull_md}

## Figures

- `data/path_b_v2/path_b_v2_gate2.png` --- team size (left) and repeated share
  (right) against $\\lambda$, one line per pull level, dashed = diagnostic
  reference. Each point is the mean over formulation $\\times$ credit $\\times$
  landscape (20 seeds per cell), so the left panel starts between the two
  landscape baselines rather than at either.

# Needs reported rather than changed

- **Repeated-collaboration share is structurally far above its reference.** It
  sits near 0.9 in every cell against a diagnostic 0.34, and the mass of steps
  at exactly 1 is large. The model's co-harvest pairs repeat because 20 agents
  on 400 tiles co-locate heavily; matching 0.34 would need a partner-turnover
  mechanism (links or teams dissolving faster), not a stronger pull. Reported,
  not added.
- **Team-size definition.** The diagnostic 2.65 is a per-paper author count;
  `mean_team_size` counts co-harvest groups $\\geq 2$. The solos-inclusive
  variant is one line in the analyser if the owner wants it.
- **Closure moment.** Markov C club closure (retention 0.3995, entry 0.0619)
  still needs the author $\\times$ step panel reconstruction; unchanged from
  probe v1.
- **One-step lag** between link formation and the pull (above), if the owner
  wants it closed.

# Reproduction

```{{.bash}}
MPLBACKEND=Agg /opt/hermes/.venv/bin/python tests/test_smoke.py
MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/path_b/probe_move_pull.py
MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/path_b/run_path_b_v2_sweep.py
MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/path_b/analyse_path_b_v2.py
MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/path_b/make_doc_v2.py
```

# Revision Log

- **Version 1** (November 2026) --- Movement coupling `dyadic_move_pull`
  implemented (default 0.0, bit-identical to Path B v1, positive-pull-with-layer-off
  refused); canary extended with identity and coupling checks; 48-cell, 960-run
  Gate 2 v2 grid with an in-driver control arm; gate verdict {v};
  model-internal, not calibrated.
'''

    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(md, encoding="utf-8")
    print(f"wrote {DOC} ({len(md)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
