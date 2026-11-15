"""Path B Gate 2 v2 analysis: does the movement coupling reach production?

Reads only this round's folder (`data/path_b_v2/`) plus the Path B v1 cells as
the pull=0 reference arm. Writes every number the report quotes.

# Gate 2 v2 rule -- DECLARED BEFORE THE GRID WAS READ (2026-11-15, pre-launch)

Contrast within each cell quadruple `(dyadic_move_pull, formulation, credit,
landscape)`: lambda = 1.0 against lambda = 0.0, seed-matched (20 pairs).

* `team_reduction` = mean over seeds of
  `|team_tm(lambda=0) - 2.65| - |team_tm(lambda=1) - 2.65|`
  (positive = the production moment moved TOWARD the diagnostic reference
  value 2.65). `rep_reduction` is the same with `rep_tm` and the reference
  0.34. These references are diagnostics, never fitting targets.
* A reduction is **significant** when its bootstrap 95% CI (10,000 resamples
  over the 20 seed-pairs, percentile method) excludes zero.
* **PASS**: in BOTH pull levels, >= 6 of 8 contrasts show a significant
  positive team reduction AND >= 6 of 8 a significant positive rep reduction.
* **PARTIAL**: some production contrast is significant but PASS is not met
  (this includes "choice responds, production does not").
* **FAIL**: no production contrast is significant in either pull level.
* Secondary, reported and not gated: the pull's own effect -- v2 cells against
  the Path B v1 arm at the same (lambda, formulation, credit, landscape, seed)
  -- and the prestige-ideal wedge (`c_local_tm - c_ideal_tm`) across lambda.

Guards: a cell must carry exactly 20 rows; a cell still being written is
reported INCOMPLETE, never counted as a difference. Link formation must be
non-zero in every cell (otherwise the pull has nothing to act on).

Run:
    MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/path_b/analyse_path_b_v2.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

V2 = REPO / "data" / "path_b_v2"
V1_CELLS = REPO / "data" / "path_b_v1" / "cells"
CELLS = V2 / "cells"
TEAM_TARGET = 2.65
REP_TARGET = 0.34
SEEDS = 20
N_BOOT = 10_000
RNG = np.random.default_rng(20261115)


def load_cells() -> tuple[pd.DataFrame, list[str]]:
    """All v2 cells, with the cell identity parsed from the tag."""
    frames, incomplete = [], []
    for path in sorted(CELLS.glob("cell_lam*_pull*_*.csv")):
        rows = sum(1 for _ in open(path)) - 1
        tag = path.stem
        if rows != SEEDS:
            incomplete.append(f"{tag} ({rows} rows)")
            continue
        df = pd.read_csv(path)
        df["cell_tag"] = tag
        frames.append(df)
    if not frames:
        raise SystemExit("no complete v2 cells found")
    return pd.concat(frames, ignore_index=True), incomplete


def load_v1_reference() -> pd.DataFrame:
    """Path B v1 cells (the pull=0 arm); lambda lives only in the filename."""
    frames = []
    for path in sorted(V1_CELLS.glob("cell_lam*_*.csv")):
        tag = path.stem
        m = re.match(r"cell_lam(\d{3})_(logit|threshold)_(struct|equal)_(\w+)$", tag)
        if not m:
            continue
        df = pd.read_csv(path)
        df["dyadic_lambda"] = int(m.group(1)) / 100.0
        df["formulation"] = m.group(2)
        df["credit_sharing_axis"] = "structural" if m.group(3) == "struct" else "equal_split"
        df["landscape"] = m.group(4)
        df["cell_tag"] = tag
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["dyadic_move_pull"] = 0.0
    return out


def boot_ci(values: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float]:
    idx = RNG.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def paired(df: pd.DataFrame, axis: str, level, ref, metric: str) -> pd.DataFrame:
    """Seed-matched wide frame: one row per (cell quadruple, seed).

    The swept axis is deliberately NOT in the merge key: the two arms sit at
    different levels of it, so including it would join nothing (an all-NaN
    contrast that looks like a null result).
    """
    key = ["formulation", "credit_sharing_axis", "landscape", "agent_seed"]
    a = df[df[axis] == level][key + [metric]].rename(columns={metric: "hi"})
    b = df[df[axis] == ref][key + [metric]].rename(columns={metric: "lo"})
    return a.merge(b, on=key, how="inner")


def distance_reduction(wide: pd.DataFrame, target: float, col_hi: str, col_lo: str) -> dict:
    """Paired |d(lo)-target| - |d(hi)-target|: positive = toward the target."""
    red = (wide[col_lo] - target).abs() - (wide[col_hi] - target).abs()
    lo, hi = boot_ci(red.to_numpy())
    return {
        "n_pairs": int(len(red)),
        "reduction": float(red.mean()),
        "ci_lo": lo,
        "ci_hi": hi,
        "significant": bool(lo > 0.0 or hi < 0.0),
        "significant_positive": bool(lo > 0.0),
    }


def main() -> int:
    df, incomplete = load_cells()
    v1 = load_v1_reference()
    print(f"v2 rows: {len(df)} from {df['cell_tag'].nunique()} cells")
    if incomplete:
        print(f"INCOMPLETE cells (excluded, never counted as differences): {incomplete}")

    # ---- guard: the pull needs links to act on ----------------------------
    zero_link_cells = [
        t for t, g in df.groupby("cell_tag") if (g["links_tm"] == 0).all()
    ]
    if zero_link_cells:
        print(f"WARNING: cells with no links at all: {zero_link_cells}")

    # ---- Gate 2 v2: lambda 1 vs 0 within each (pull, form, credit, land) --
    # A quadruple is CONTRAASTABLE only when both lambda levels are present (20
    # rows each); a half-written quadruple is reported, never differenced.
    rows = []
    skipped_quadruples = []
    for (pull, form, credit, land), g in df.groupby(
        ["dyadic_move_pull", "formulation", "credit_sharing_axis", "landscape"]
    ):
        levels = set(g["dyadic_lambda"].round(6))
        if not {0.0, 1.0} <= levels:
            skipped_quadruples.append(
                f"pull={pull} {form}/{credit}/{land}: lambda levels {sorted(levels)}"
            )
            continue
        key = ["formulation", "credit_sharing_axis", "landscape", "agent_seed"]
        team_wide = paired(g, "dyadic_lambda", 1.0, 0.0, "team_tm").rename(
            columns={"lo": "team_lo", "hi": "team_hi"}
        )
        rep_wide = paired(g, "dyadic_lambda", 1.0, 0.0, "rep_tm").rename(
            columns={"lo": "rep_lo", "hi": "rep_hi"}
        )
        wide = team_wide.merge(rep_wide, on=key, how="inner")
        team = distance_reduction(wide, TEAM_TARGET, "team_hi", "team_lo")
        rep = distance_reduction(wide, REP_TARGET, "rep_hi", "rep_lo")
        choice = paired(g, "dyadic_lambda", 1.0, 0.0, "links_tm")
        delta_links = (choice["hi"] - choice["lo"])
        clo, chi = boot_ci(delta_links.to_numpy())
        rows.append(
            {
                "pull": pull,
                "formulation": form,
                "credit": credit,
                "landscape": land,
                "team_d_lam0": float((wide["team_lo"] - TEAM_TARGET).abs().mean()),
                "team_d_lam1": float((wide["team_hi"] - TEAM_TARGET).abs().mean()),
                "team_reduction": team["reduction"],
                "team_ci_lo": team["ci_lo"],
                "team_ci_hi": team["ci_hi"],
                "team_sig_pos": team["significant_positive"],
                "rep_d_lam0": float((wide["rep_lo"] - REP_TARGET).abs().mean()),
                "rep_d_lam1": float((wide["rep_hi"] - REP_TARGET).abs().mean()),
                "rep_reduction": rep["reduction"],
                "rep_ci_lo": rep["ci_lo"],
                "rep_ci_hi": rep["ci_hi"],
                "rep_sig_pos": rep["significant_positive"],
                "delta_links_tm": float(delta_links.mean()),
                "delta_links_ci_lo": clo,
                "delta_links_ci_hi": chi,
                "choice_responds": bool(clo > 0 or chi < 0),
            }
        )
    if not rows:
        # The grid is ordered lambda-major, so no quadruple carries both
        # lambda levels until the last third of the run. That is a partial
        # grid, not a null result.
        out = {
            "verdict": "PENDING (no contrastable quadruple yet)",
            "skipped_quadruples": skipped_quadruples,
            "incomplete_cells": incomplete,
            "v2_cells": int(df["cell_tag"].nunique()),
            "v2_rows": int(len(df)),
        }
        with open(V2 / "gate2_v2_verdict.json", "w") as f:
            json.dump(out, f, indent=1)
        print(
            f"PENDING: {df['cell_tag'].nunique()} cells, no quadruple with both "
            f"lambda levels yet ({len(skipped_quadruples)} skipped)"
        )
        return 0
    gate = pd.DataFrame(rows).sort_values(
        ["pull", "formulation", "credit", "landscape"]
    )
    gate.to_csv(V2 / "gate2_v2_table.csv", index=False)

    # Landscape decomposition: the gate rule compares every arm to a fixed
    # reference, so an arm whose lambda=0 baseline already sits at the
    # reference cannot "improve". Report the split explicitly; the verdict
    # itself stays the pre-declared rule's.
    land_rows = []
    for (pull, land), sel in gate.groupby(["pull", "landscape"]):
        land_rows.append(
            {
                "pull": float(pull),
                "landscape": land,
                "n_contrasts": int(len(sel)),
                "team_sig_pos": int(sel["team_sig_pos"].sum()),
                "team_sig_neg": int((sel["team_ci_hi"] < 0).sum()),
                "team_reduction_mean": float(sel["team_reduction"].mean()),
                "baseline_team_distance": float(sel["team_d_lam0"].mean()),
                "rep_sig_pos": int(sel["rep_sig_pos"].sum()),
                "rep_reduction_mean": float(sel["rep_reduction"].mean()),
            }
        )
    land_tab = pd.DataFrame(land_rows).sort_values(["pull", "landscape"])
    land_tab.to_csv(V2 / "landscape_summary.csv", index=False)

    counts = {}
    for pull in sorted(gate["pull"].unique()):
        sel = gate[gate["pull"] == pull]
        counts[float(pull)] = {
            "team_sig_pos": int(sel["team_sig_pos"].sum()),
            "rep_sig_pos": int(sel["rep_sig_pos"].sum()),
            "choice_responds": int(sel["choice_responds"].sum()),
            "n_contrasts": int(len(sel)),
        }
    expected_cells = 48
    grid_complete = df["cell_tag"].nunique() == expected_cells
    verdict = "PENDING (grid incomplete)" if not grid_complete else "FAIL"
    if not grid_complete:
        pass
    elif any(
        c["team_sig_pos"] >= 6 and c["rep_sig_pos"] >= 6 for c in counts.values()
    ):
        verdict = (
            "PASS"
            if all(c["team_sig_pos"] >= 6 and c["rep_sig_pos"] >= 6 for c in counts.values())
            else "PARTIAL"
        )
    elif any(
        c["team_sig_pos"] > 0 or c["rep_sig_pos"] > 0 for c in counts.values()
    ):
        verdict = "PARTIAL"
    elif any(c["choice_responds"] >= 6 for c in counts.values()):
        verdict = "PARTIAL"

    # ---- secondary: the pull's own effect vs the v1 (pull=0) arm ---------
    pull_rows = []
    common = ["formulation", "credit_sharing_axis", "landscape", "dyadic_lambda", "agent_seed"]
    for metric in ("team_tm", "rep_tm", "c_ideal_tm", "c_local_tm", "ideal_loss_cum"):
        v2m = df[common + ["dyadic_move_pull", metric]].rename(columns={metric: "v2"})
        v1m = v1[common + [metric]].rename(columns={metric: "v1"})
        j = v2m.merge(v1m, on=common, how="inner")
        for pull in sorted(j["dyadic_move_pull"].unique()):
            sel = j[j["dyadic_move_pull"] == pull]
            for lam in sorted(sel["dyadic_lambda"].unique()):
                s = sel[sel["dyadic_lambda"] == lam]
                d = s["v2"] - s["v1"]
                lo, hi = boot_ci(d.to_numpy())
                pull_rows.append(
                    {
                        "metric": metric,
                        "pull": float(pull),
                        "lambda": float(lam),
                        "n_pairs": int(len(d)),
                        "mean_delta_vs_pull0": float(d.mean()),
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "significant": bool(lo > 0 or hi < 0),
                    }
                )
    pull_tab = pd.DataFrame(pull_rows)
    pull_tab.to_csv(V2 / "pull_effect_table.csv", index=False)

    # ---- lambda trends (cell means) --------------------------------------
    trend = (
        df.groupby(
            ["dyadic_move_pull", "formulation", "credit_sharing_axis", "landscape", "dyadic_lambda"]
        )
        .agg(
            team_tm=("team_tm", "mean"),
            rep_tm=("rep_tm", "mean"),
            links_tm=("links_tm", "mean"),
            c_ideal_tm=("c_ideal_tm", "mean"),
            c_local_tm=("c_local_tm", "mean"),
            ideal_loss_cum=("ideal_loss_cum", "mean"),
            rep_mass0=("rep_mass0", "mean"),
            rep_mass1=("rep_mass1", "mean"),
        )
        .reset_index()
    )
    trend["wedge_tm"] = trend["c_local_tm"] - trend["c_ideal_tm"]
    trend.to_csv(V2 / "lambda_trend.csv", index=False)

    # ---- figure ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, target, title in (
        (axes[0], "team_tm", TEAM_TARGET, "co-harvest team size"),
        (axes[1], "rep_tm", REP_TARGET, "repeated-collaboration share"),
    ):
        sub = trend.groupby(["dyadic_move_pull", "dyadic_lambda"])[metric].mean().reset_index()
        for pull in sorted(sub["dyadic_move_pull"].unique()):
            s = sub[sub["dyadic_move_pull"] == pull]
            ax.plot(s["dyadic_lambda"], s[metric], marker="o", label=f"move_pull={pull:g}")
        ax.axhline(target, ls="--", lw=1, color="grey")
        ax.set_xlabel("lambda (0 = proxy matching, 1 = goal matching)")
        ax.set_ylabel(metric)
        ax.set_title(f"{title} (dashed = diagnostic {target:g})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(V2 / "path_b_v2_gate2.png", dpi=150)
    plt.close(fig)

    out = {
        "verdict": verdict,
        "rule": "PASS needs >=6/8 significant positive reductions for team AND rep in BOTH pull levels",
        "counts": counts,
        "targets": {"team_tm": TEAM_TARGET, "rep_tm": REP_TARGET},
        "v2_rows": int(len(df)),
        "v2_cells": int(df["cell_tag"].nunique()),
        "incomplete_cells": incomplete,
        "skipped_quadruples": skipped_quadruples,
        "grid_complete": bool(grid_complete),
        "expected_cells": expected_cells,
        "zero_link_cells": zero_link_cells,
        "n_boot": N_BOOT,
        "bootstrap_seed": 20261115,
    }
    with open(V2 / "gate2_v2_verdict.json", "w") as f:
        json.dump(out, f, indent=1)

    print(f"\nGATE 2 v2 verdict: {verdict}")
    for pull, c in counts.items():
        print(
            f"  pull={pull}: team sig+ {c['team_sig_pos']}/{c['n_contrasts']}, "
            f"rep sig+ {c['rep_sig_pos']}/{c['n_contrasts']}, "
            f"choice responds {c['choice_responds']}/{c['n_contrasts']}"
        )
    print(f"wrote: gate2_v2_table.csv, pull_effect_table.csv, lambda_trend.csv, "
          f"gate2_v2_verdict.json, path_b_v2_gate2.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
