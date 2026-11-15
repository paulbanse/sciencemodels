"""Analysis for the P-0006 turnover probe (turnover_v1).

The question the round exists to answer: do the career-flow findings survive
once generations actually turn over, or are they artifacts of founder
immortality?

Two findings are tested, both stated as sign claims so they can fail:

* **A -- the arrival gap.** Recent arrivals hold less prestige than established
  agents. Measured by tenure (`gap_recent`: tenure < 20 steps versus >= 20),
  because the model's `is_newcomer` flag is PERMANENT: once every founder has
  retired it is true for the whole population and contrasts nothing. The
  flag-based gap (`gap_flag`) is carried alongside for comparability with
  `output/reports/career_flows_v1.md` and is NaN exactly where it degenerates.
* **B -- the ideal-merit correlation.** `Corr_prestige_ideal_merit` is negative
  at the end of the run, i.e. prestige anti-tracks quality under entry.

Both are re-read on the founder-free window, which is the part of the run the
proposal actually cares about: a cell counts as *settled* when its last founder
retired at least `SETTLE` steps before the horizon.

Everything is model-internal; nothing is calibrated (owner ruling 2026-11-13).

Run from the repo root:

    MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/turnover/analyse_turnover.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "data", "turnover_v1")
CELLS = os.path.join(OUT, "cells")
TRAJ = os.path.join(OUT, "traj")
TABLES = os.path.join(OUT, "tables")

HORIZON = 400
SETTLE = 100          # steps of founder-free equilibrium required to call it settled
MODES = ("none", "fixed_span", "age_hazard")
KEYS = ["entry", "curiosity", "landscape", "seed"]


def _require(cond, msg):
    if not cond:
        raise AssertionError(msg)


def load_cells():
    files = sorted(f for f in os.listdir(CELLS) if f.endswith(".csv"))
    _require(len(files) == 54, "gross cells=54, found %d" % len(files))
    df = pd.concat(
        [pd.read_csv(os.path.join(CELLS, f)) for f in files], ignore_index=True
    )
    _require(len(df) == 1080, "gross runs=1080, got %d" % len(df))
    _require(df["seed"].nunique() == 20, "seeds must be 0..19")
    _require(
        set(df["exit_mode"].unique()) == set(MODES),
        "unexpected modes: %r" % (sorted(df["exit_mode"].unique()),),
    )
    # a settled cell is one whose last founder left early enough to leave a
    # founder-free window; 'none' cells are never settled by construction
    df["settled"] = df["founders_retired_step"].notna() & (
        df["founders_retired_step"] <= HORIZON - SETTLE
    )
    return df


def load_traj():
    files = sorted(f for f in os.listdir(TRAJ) if f.endswith(".csv"))
    _require(len(files) == 54, "trajectories=54, found %d" % len(files))
    frames = [pd.read_csv(os.path.join(TRAJ, f)) for f in files]
    tj = pd.concat(frames, ignore_index=True)
    tj["exit_mode"] = tj["cell_tag"].str.extract(r"^cell_(none|fixed_span|age_hazard)_")
    return tj


def mode_summary(df):
    rows = []
    for mode in MODES:
        sub = df[df["exit_mode"] == mode]
        rows.append(
            {
                "exit_mode": mode,
                "runs": len(sub),
                "n_mean_last100": sub["n_mean_last100"].mean(),
                "mean_age_final": sub["mean_age_final"].mean(),
                "mean_tenure_final": sub["mean_tenure_final"].mean(),
                "founders_retired_step_median": sub["founders_retired_step"].median(),
                "settled_share": sub["settled"].mean(),
                "gap_flag_last100": sub["gap_flag_mean_last100"].mean(),
                "gap_flag_defined_share": sub["gap_flag_mean_last100"].notna().mean(),
                "gap_recent_last100": sub["gap_recent_mean_last100"].mean(),
                "c_ideal_fin": sub["c_ideal_fin"].mean(),
                "c_ideal_tm": sub["c_ideal_tm"].mean(),
                "c_local_fin": sub["c_local_fin"].mean(),
                "top10_fin": sub["top10_fin"].mean(),
                "harvested_fin": sub["harvested_fin"].mean(),
                "ideal_merit_sum": sub["ideal_merit_sum"].mean(),
            }
        )
    return pd.DataFrame(rows)


def grid_table(df):
    g = df.groupby(
        ["exit_mode", "entry", "curiosity", "landscape"], as_index=False
    ).agg(
        n_runs=("seed", "size"),
        n_mean_last100=("n_mean_last100", "mean"),
        mean_age_final=("mean_age_final", "mean"),
        gap_recent=("gap_recent_mean_last100", "mean"),
        c_ideal_fin=("c_ideal_fin", "mean"),
        c_ideal_tm=("c_ideal_tm", "mean"),
        settled=("settled", "mean"),
    )
    return g


def persistence(df):
    """Sign claims, on the full grid and on the settled subset."""
    rows = []
    for scope, sub in (("all runs", df), ("settled cells only", df[df["settled"]])):
        for mode in MODES:
            s = sub[sub["exit_mode"] == mode]
            if not len(s):
                continue
            rows.append(
                {
                    "scope": scope,
                    "exit_mode": mode,
                    "runs": len(s),
                    "gap_negative_share": float(
                        (s["gap_recent_mean_last100"] < 0).mean()
                    ),
                    "gap_mean": float(s["gap_recent_mean_last100"].mean()),
                    "c_ideal_negative_share": float((s["c_ideal_fin"] < 0).mean()),
                    "c_ideal_mean": float(s["c_ideal_fin"].mean()),
                    "c_ideal_tm_negative_share": float((s["c_ideal_tm"] < 0).mean()),
                    "c_ideal_tm_mean": float(s["c_ideal_tm"].mean()),
                }
            )
    return pd.DataFrame(rows)


def paired_contrasts(df):
    """Same seed, same (entry, curiosity, landscape): turnover versus baseline."""
    base = df[df["exit_mode"] == "none"].set_index(KEYS)
    rows = []
    for mode in ("fixed_span", "age_hazard"):
        cur = df[df["exit_mode"] == mode].set_index(KEYS)
        j = base.join(cur, lsuffix="_none", rsuffix="_" + mode, how="inner")
        _require(len(j) == 360, "%s pairs=360, got %d" % (mode, len(j)))
        for metric in ("c_ideal_fin", "c_ideal_tm", "gap_recent_mean_last100"):
            a = j[metric + "_none"].to_numpy()
            b = j[metric + "_" + mode].to_numpy()
            d = b - a
            w = stats.wilcoxon(d, zero_method="wilcox") if np.any(d != 0) else None
            rows.append(
                {
                    "mode": mode,
                    "metric": metric,
                    "n_pairs": len(d),
                    "baseline_mean": float(np.mean(a)),
                    "turnover_mean": float(np.mean(b)),
                    "mean_diff": float(np.mean(d)),
                    "share_turnover_lower": float((d < 0).mean()),
                    "sign_preserved_share": float((np.sign(a) == np.sign(b)).mean()),
                    "wilcoxon_p": float(w.pvalue) if w is not None else np.nan,
                }
            )
    return pd.DataFrame(rows)


def trajectory_points(tj):
    rows = []
    for step in (100, 200, 300, 400):
        s = tj[tj["step"] == step]
        for mode in MODES:
            m = s[s["exit_mode"] == mode]
            if not len(m):
                continue
            rows.append(
                {
                    "step": step,
                    "exit_mode": mode,
                    "gap_recent": m["gap_recent"].mean(),
                    "c_ideal": m["c_ideal"].mean(),
                    "n_active": m["n_active"].mean(),
                    "n_founders": m["n_founders"].mean(),
                    "mean_age": m["mean_age"].mean(),
                }
            )
    return pd.DataFrame(rows)


def main():
    os.makedirs(TABLES, exist_ok=True)
    df = load_cells()
    tj = load_traj()

    ms = mode_summary(df)
    grid = grid_table(df)
    pers = persistence(df)
    pairs = paired_contrasts(df)
    traj = trajectory_points(tj)

    for name, frame in (
        ("mode_summary", ms),
        ("grid", grid),
        ("persistence", pers),
        ("paired_contrasts", pairs),
        ("trajectory_points", traj),
    ):
        frame.to_csv(os.path.join(TABLES, name + ".csv"), index=False)

    # ---- verdicts, computed rather than asserted -------------------------
    settled = pers[pers["scope"] == "settled cells only"].set_index("exit_mode")
    verdict = {}
    for mode in MODES:
        if mode not in settled.index:
            continue
        r = settled.loc[mode]
        verdict[mode] = {
            "runs": int(r["runs"]),
            "gap_persists": bool(r["gap_negative_share"] > 0.5),
            "gap_negative_share": float(r["gap_negative_share"]),
            "c_ideal_negative_persists": bool(r["c_ideal_negative_share"] > 0.5),
            "c_ideal_negative_share": float(r["c_ideal_negative_share"]),
        }

    summary = {
        "runs": int(len(df)),
        "cells": 54,
        "settle_window": SETTLE,
        "settled_cells": int(df[df["settled"]]["seed"].count()),
        "verdict_settled": verdict,
        "mode_summary": ms.to_dict(orient="records"),
        "paired_contrasts": pairs.to_dict(orient="records"),
        "trajectory_points": traj.to_dict(orient="records"),
    }
    with open(os.path.join(TABLES, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)

    pd.set_option("display.width", 250)
    print("== mode summary ==")
    print(ms[[
        "exit_mode", "n_mean_last100", "mean_age_final", "mean_tenure_final",
        "founders_retired_step_median", "settled_share", "gap_flag_last100",
        "gap_recent_last100", "c_ideal_fin", "c_ideal_tm",
    ]].to_string(index=False))
    print("\n== persistence (sign claims) ==")
    print(pers.to_string(index=False))
    print("\n== paired contrasts vs baseline ==")
    print(pairs.to_string(index=False))
    print("\n== trajectory points ==")
    print(traj.to_string(index=False))
    print("\ntables -> %s" % TABLES)


if __name__ == "__main__":
    main()
