"""Gate 2 evaluation for the Path B dyadic layer (2026-11-14).

Question (docs/way_forward.md Gate 2): does the dyadic layer materially
improve the moment fit, especially the team and closure moments?

VERDICT RULE (declared before the sweep results were read; applied
mechanically):

* Contrasts: lambda=1.0 vs lambda=0.0, seed-matched (20 pairs), within each
  of the 8 formulation x credit x landscape cells-pairs.
* Production moments (the Gate 2 targets): mean_team_size vs the Klann
  diagnostic 2.65, repeated_collab_share vs 0.34. A moment IMPROVES when
  the paired distance |sim - target| shrinks from lambda 0 to 1 with a
  bootstrap 95% CI over the 20 seed-pairs excluding 0.
* GATE 2 PASS: team-size distance reduction significant in >= 6 of 8
  contrasts AND repeated-share distance reduction significant in >= 6 of 8.
* PARTIAL (choice binds, production does not): the choice-layer diagnostics
  (links_tm, link_rep_tm, link_assort_tm, n_active_links) respond to lambda
  significantly in >= 6 of 8 contrasts, but the production rule fails.
  Reading: the choice layer works as a choice mechanism but does not reach
  joint production; the honest paper is A+D and the layer is robustness.
* FAIL: neither rule met.

The equal_split x lambda interaction is evaluated descriptively (the one
ruled feedback channel: split prestige -> visibility reward's
prestige weighting -> movement -> co-location).

NOT CALIBRATED: the Klann values are diagnostic reference points per the
owner ruling of 2026-11-13, never fitting targets.

Run: MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/path_b/analyse_path_b.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT_DIR = REPO / "data" / "path_b_v1"
CELLS_DIR = OUT_DIR / "cells"

LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
FORMULATIONS = ("logit", "threshold")
CREDITS = ("structural", "equal_split")
LANDSCAPES = ("mgstd5", "noisypr5")

TEAM_TARGET = 2.65  # Klann diagnostic: mean team size (median 2)
REP_TARGET = 0.34  # Klann diagnostic: repeated-collaboration share
BOOTSTRAP = 5000
PASS_THRESHOLD = 6  # of 8 contrasts

TAG_RE = re.compile(
    r"^cell_lam(?P<lam>\d+)_(?P<form>logit|threshold)_?"
    r"(?P<credit>struct|equal)_(?P<land>mgstd5|noisypr5)$"
)


def load_runs():
    frames, files = [], []
    for path in sorted(CELLS_DIR.glob("cell_*.csv")):
        m = TAG_RE.match(path.stem)
        if not m:
            continue
        df = pd.read_csv(path)
        df["lam"] = int(m.group("lam")) / 100.0
        df["form"] = m.group("form")
        df["credit"] = "structural" if m.group("credit") == "struct" else "equal_split"
        df["land"] = m.group("land")
        frames.append(df)
        files.append(path.name)
    runs = pd.concat(frames, ignore_index=True)
    return runs, files


def boot_ci_paired(reduction, n_boot=BOOTSTRAP, seed=0):
    """Bootstrap 95% CI for the mean of paired per-seed reductions."""
    r = np.asarray(reduction, dtype=float)
    n = len(r)
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = r[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def contrast(runs, form, credit, land, metric, target):
    """Seed-matched lambda 1.0 vs 0.0 paired distance-to-target reduction."""
    r0 = runs[(runs.form == form) & (runs.credit == credit)
              & (runs.land == land) & (runs.lam == 0.0)].set_index("agent_seed")
    r1 = runs[(runs.form == form) & (runs.credit == credit)
              & (runs.land == land) & (runs.lam == 1.0)].set_index("agent_seed")
    common = sorted(set(r0.index) & set(r1.index))
    d0 = np.abs(r0.loc[common, metric].to_numpy(float) - target)
    d1 = np.abs(r1.loc[common, metric].to_numpy(float) - target)
    reduction = d0 - d1  # positive = moved toward the target
    lo, hi = boot_ci_paired(reduction)
    return {
        "n_pairs": len(common),
        "mean_d0": float(d0.mean()),
        "mean_d1": float(d1.mean()),
        "mean_reduction": float(reduction.mean()),
        "ci_lo": lo,
        "ci_hi": hi,
        "significant": bool(lo > 0.0),
    }


def choice_contrast(runs, form, credit, land, metric):
    """Seed-matched lambda 1.0 vs 0.0 raw paired difference (choice-layer
    response: any significant movement, either direction)."""
    r0 = runs[(runs.form == form) & (runs.credit == credit)
              & (runs.land == land) & (runs.lam == 0.0)].set_index("agent_seed")
    r1 = runs[(runs.form == form) & (runs.credit == credit)
              & (runs.land == land) & (runs.lam == 1.0)].set_index("agent_seed")
    common = sorted(set(r0.index) & set(r1.index))
    delta = (r1.loc[common, metric] - r0.loc[common, metric]).to_numpy(float)
    lo, hi = boot_ci_paired(delta)
    return {
        "n_pairs": len(common),
        "mean_delta": float(delta.mean()),
        "ci_lo": lo,
        "ci_hi": hi,
        "significant": bool(lo > 0.0 or hi < 0.0),
    }


def main():
    runs, files = load_runs()
    print("loaded %d cell files, %d runs" % (len(files), len(runs)))
    assert len(files) == 40, "expected 40 cell files, got %d" % len(files)
    assert len(runs) == 800, "expected 800 runs, got %d" % len(runs)
    runs["wedge_tm"] = runs["c_local_tm"] - runs["c_ideal_tm"]

    PROD = (("mean_team_size", "team_tm", TEAM_TARGET, "team size"),
            ("repeated_collab_share", "rep_tm", REP_TARGET, "repeated share"))
    CHOICE = ("links_tm", "link_rep_tm", "link_assort_tm")

    rows = []
    for form in FORMULATIONS:
        for credit in CREDITS:
            for land in LANDSCAPES:
                row = {"form": form, "credit": credit, "land": land}
                for _p, metric, target, name in PROD:
                    c = contrast(runs, form, credit, land, metric, target)
                    row["%s_d0" % metric] = c["mean_d0"]
                    row["%s_d1" % metric] = c["mean_d1"]
                    row["%s_red" % metric] = c["mean_reduction"]
                    row["%s_lo" % metric] = c["ci_lo"]
                    row["%s_hi" % metric] = c["ci_hi"]
                    row["%s_sig" % metric] = c["significant"]
                for metric in CHOICE:
                    c = choice_contrast(runs, form, credit, land, metric)
                    row["%s_delta" % metric] = c["mean_delta"]
                    row["%s_lo" % metric] = c["ci_lo"]
                    row["%s_hi" % metric] = c["ci_hi"]
                    row["%s_sig" % metric] = c["significant"]
                # descriptive paired lambda 1.0 - 0.0 deltas (the brief's
                # central question: wedge, ideal-merit loss, and the raw
                # production-moment movements)
                for metric in ("wedge_tm", "c_ideal_tm", "ideal_loss_cum",
                               "mean_team_size", "repeated_collab_share"):
                    c = choice_contrast(runs, form, credit, land, metric)
                    row["%s_delta" % metric] = c["mean_delta"]
                    row["%s_dlo" % metric] = c["ci_lo"]
                    row["%s_dhi" % metric] = c["ci_hi"]
                rows.append(row)
    gate = pd.DataFrame(rows)
    gate.to_csv(OUT_DIR / "gate2_table.csv", index=False)

    n_team = int(gate["team_tm_sig"].sum())
    n_rep = int(gate["rep_tm_sig"].sum())
    n_choice = int(
        (gate["links_tm_sig"] | gate["link_rep_tm_sig"] | gate["link_assort_tm_sig"]).sum()
    )

    # the one ruled feedback channel: does credit=equal_split make the
    # production moments lambda-sensitive where structural does not?
    feedback = {}
    for form in FORMULATIONS:
        for land in LANDSCAPES:
            for metric, _tm, target, _n in PROD:
                for credit in CREDITS:
                    c = contrast(runs, form, credit, land, metric, target)
                    feedback["%s|%s|%s|%s" % (form, land, metric, credit)] = {
                        "reduction": c["mean_reduction"],
                        "significant": c["significant"],
                    }

    verdict = {
        "rule": "PASS: team-size and repeated-share distance reductions significant (bootstrap 95% CI excludes 0) in >= 6 of 8 formulation x credit x landscape contrasts each; PARTIAL: choice layer responds in >= 6 of 8 but production does not; FAIL otherwise.",
        "n_contrasts": 8,
        "team_size_sig": n_team,
        "repeated_share_sig": n_rep,
        "choice_sig": n_choice,
        "pass": bool(n_team >= PASS_THRESHOLD and n_rep >= PASS_THRESHOLD),
        "partial": bool(
            (not (n_team >= PASS_THRESHOLD and n_rep >= PASS_THRESHOLD))
            and n_choice >= PASS_THRESHOLD
        ),
        "feedback_descriptive": feedback,
    }
    if verdict["pass"]:
        verdict["verdict"] = "PASS"
    elif verdict["partial"]:
        verdict["verdict"] = "PARTIAL"
    else:
        verdict["verdict"] = "FAIL"

    with open(OUT_DIR / "gate2_verdict.json", "w") as f:
        json.dump(verdict, f, indent=1, default=str)

    # lambda trends for the report tables (all 40 cells, per metric)
    trend_metrics = [
        "mean_team_size", "repeated_collab_share", "n_active_links",
        "dyadic_assortativity_prestige", "link_repeated_share",
        "link_assortativity_prestige", "Mean_of_Corr_prestige_ideal_merit",
        "Mean_of_Corr_prestige_local_merit", "ideal_loss_cum",
        "ideal_loss_mean", "top10pct_prestige_share",
        "percentage_knowledge_harvested", "mean_prestige",
    ]
    trend_rows = []
    for form in FORMULATIONS:
        for credit in CREDITS:
            for land in LANDSCAPES:
                sub = runs[(runs.form == form) & (runs.credit == credit)
                           & (runs.land == land)]
                for metric in trend_metrics:
                    if metric not in sub.columns:
                        continue
                    t = sub.groupby("lam")[metric].mean().to_dict()
                    trend_rows.append(
                        {"form": form, "credit": credit, "land": land,
                         "metric": metric, **{"lam%d" % int(round(k * 100)): v
                                               for k, v in t.items()}}
                    )
    pd.DataFrame(trend_rows).to_csv(OUT_DIR / "lambda_trend.csv", index=False)

    # per-cell summary (the doc's interpolation source for cell-level tables)
    cell_cols = ["team_tm", "rep_tm", "assort_tm", "links_tm",
                 "link_rep_tm", "link_assort_tm", "c_ideal_tm", "c_local_tm",
                 "rep_mass0", "rep_mass1", "ideal_loss_cum", "ideal_loss_mean",
                 "n_active_links", "mean_team_size", "repeated_collab_share",
                 "dyadic_assortativity_prestige", "agents_final",
                 "newcomers_final", "top10pct_prestige_share",
                 "percentage_knowledge_harvested", "mean_prestige"]
    cell_cols = [c for c in cell_cols if c in runs.columns]
    runs.groupby(["lam", "form", "credit", "land"])[cell_cols].mean().reset_index()\
        .to_csv(OUT_DIR / "cell_summary.csv", index=False)
    # off-mode structural baseline (written by run_baseline_off.py)
    base_path = OUT_DIR / "baseline_off.csv"
    if base_path.exists():
        base = pd.read_csv(base_path)
        base.groupby("land")[cell_cols].mean().reset_index()\
            .to_csv(OUT_DIR / "baseline_off_summary.csv", index=False)

    print("gate2_table.csv + lambda_trend.csv + gate2_verdict.json written")
    print(json.dumps({k: v for k, v in verdict.items()
                      if k not in ("feedback_descriptive", "rule")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
