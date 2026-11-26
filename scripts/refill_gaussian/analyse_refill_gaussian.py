"""Analysis for refill-gaussian v1 (IE-0006 Option 5 / workplan item 13).

Reads ONLY ``data/refill_gaussian_v1/cells/``, asserts the gross shape, and
writes tidy summaries under ``data/refill_gaussian_v1/``.  The report memo
``output/reports/refill_gaussian_v1.md`` quotes these files by name; no number
is transcribed by hand from a live run.

Headline comparisons
--------------------
- ``cumulative_harvest_flow``: total removed harvest Delta k over H=400.  It is
  the flow analogue required because Gaussian refill conserves total stock.
- ``late_window_flow_rate``: mean removed flow per step over the last 50 steps,
  the open-endedness indicator.
- ``explored_percentage_final`` / ``_mean``: breadth of search.
- ``corr_prestige_ideal_merit_final`` / ``_mean`` / ``_late_mean``: how well
  prestige tracks the ideal-merit yardstick.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

OUT = REPO / "data" / "refill_gaussian_v1"
CELLS = OUT / "cells"

N_CELLS, N_SEEDS = 36, 20
MODE_ORDER = ("off", "occupied", "frontier",
              "gaussian_s1", "gaussian_s3", "gaussian_s6")

METRICS = {
    "cumulative_harvest_flow": "cum_flow",
    "late_window_harvest_flow": "late_flow",
    "late_window_flow_rate": "late_flow_rate",
    "cumulative_refill_added": "refill_added",
    "final_total_knowledge": "final_total",
    "explored_percentage_final": "explored_final",
    "explored_percentage_mean": "explored_mean",
    "corr_prestige_ideal_merit_final": "corr_final",
    "corr_prestige_ideal_merit_mean": "corr_mean",
    "corr_prestige_ideal_merit_late_mean": "corr_late",
    "corr_prestige_local_merit_final": "corr_local_final",
    "top10pct_prestige_share_final": "top10_final",
    "mean_prestige_final": "mean_prestige_final",
    "n_agents_final": "n_agents_final",
}


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def load() -> pd.DataFrame:
    files = sorted(CELLS.glob("cell*.csv"))
    _require(len(files) == N_CELLS,
             f"gross cells={N_CELLS}, found {len(files)}")
    frames = []
    for path in files:
        d = pd.read_csv(path)
        _require(len(d) == N_SEEDS,
                 f"{path.name}: expected {N_SEEDS} rows, got {len(d)}")
        _require(d["agent_seed"].nunique() == N_SEEDS,
                 f"{path.name}: seed identity not unique")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    _require(len(df) == N_CELLS * N_SEEDS,
             f"net runs={N_CELLS * N_SEEDS}, got {len(df)}")
    counts = df.groupby("refill_mode").size().to_dict()
    _require(set(counts) == set(MODE_ORDER),
             f"modes wrong: {sorted(counts)}")
    _require(all(v == 120 for v in counts.values()),
             f"each mode must have 6 cells x 20 seeds = 120 rows: {counts}")
    return df


def _agg() -> dict:
    out = {"n": ("agent_seed", "size")}
    for src, dst in METRICS.items():
        out[f"{dst}_mean"] = (src, "mean")
        out[f"{dst}_sd"] = (src, "std")
    return out


def by_cell(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(
            ["refill_mode", "sigma", "initial_curiosity",
             "agent_generation_time"],
            dropna=False,
        )
        .agg(**_agg())
        .reset_index()
    )
    out["refill_mode"] = pd.Categorical(
        out["refill_mode"], categories=MODE_ORDER, ordered=True
    )
    out = out.sort_values(
        ["refill_mode", "initial_curiosity", "agent_generation_time"]
    ).reset_index(drop=True)

    # Deltas against the off baseline in the same (c, g) cell.
    ref = out[out.refill_mode == "off"][
        ["initial_curiosity", "agent_generation_time",
         "cum_flow_mean", "late_flow_rate_mean", "explored_final_mean",
         "corr_final_mean", "corr_mean_mean"]
    ].rename(columns={
        "cum_flow_mean": "off_cum_flow",
        "late_flow_rate_mean": "off_late_flow_rate",
        "explored_final_mean": "off_explored_final",
        "corr_final_mean": "off_corr_final",
        "corr_mean_mean": "off_corr_mean",
    })
    out = out.merge(ref, on=["initial_curiosity", "agent_generation_time"],
                    how="left")
    out["explored_delta_vs_off"] = (
        out["explored_final_mean"] - out["off_explored_final"]
    )
    out["corr_delta_vs_off"] = out["corr_final_mean"] - out["off_corr_final"]
    out["late_flow_ratio_vs_off"] = np.where(
        out["off_late_flow_rate"] > 0,
        out["late_flow_rate_mean"] / out["off_late_flow_rate"],
        np.nan,
    )
    return out


def by_mode(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["refill_mode", "sigma"], dropna=False)
        .agg(**_agg())
        .reset_index()
    )
    out["refill_mode"] = pd.Categorical(
        out["refill_mode"], categories=MODE_ORDER, ordered=True
    )
    return out.sort_values("refill_mode").reset_index(drop=True)


def by_entry(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(
            ["refill_mode", "sigma", "agent_generation_time"], dropna=False
        )
        .agg(**_agg())
        .reset_index()
    )
    out["refill_mode"] = pd.Categorical(
        out["refill_mode"], categories=MODE_ORDER, ordered=True
    )
    return out.sort_values(
        ["refill_mode", "agent_generation_time"]
    ).reset_index(drop=True)


def headline(cell: pd.DataFrame) -> pd.DataFrame:
    """Central-cell table (c=0.5, g=0.1) for the memo."""
    cols = [
        "refill_mode", "sigma", "cum_flow_mean", "late_flow_rate_mean",
        "explored_final_mean", "explored_delta_vs_off",
        "corr_final_mean", "corr_mean_mean", "corr_delta_vs_off",
        "final_total_mean", "refill_added_mean", "n_agents_final_mean",
    ]
    h = cell[
        (cell.initial_curiosity == 0.5)
        & (cell.agent_generation_time == 0.1)
    ][cols].copy()
    return h.reset_index(drop=True)


def main() -> int:
    df = load()
    cell = by_cell(df)
    mode = by_mode(df)
    entry = by_entry(df)
    head = headline(cell)

    cell.to_csv(OUT / "summary_by_cell.csv", index=False)
    mode.to_csv(OUT / "summary_by_mode.csv", index=False)
    entry.to_csv(OUT / "summary_by_mode_entry.csv", index=False)
    head.to_csv(OUT / "headline_c05_g01.csv", index=False)

    sanity = {
        "n_cells": int(df.groupby("cell_tag").ngroups),
        "n_runs": int(len(df)),
        "modes": {k: int(v) for k, v in
                  df.groupby("refill_mode").size().items()},
        "all_gaussian_conserved": bool(
            np.allclose(
                df.loc[df.refill_mode.str.startswith("gaussian"),
                       "final_total_knowledge"],
                1.0, atol=1e-9,
            )
        ),
        "max_gaussian_total_error": float(
            np.max(np.abs(
                df.loc[df.refill_mode.str.startswith("gaussian"),
                       "final_total_knowledge"] - 1.0
            ))
        ),
    }
    (OUT / "analysis.json").write_text(json.dumps(sanity, indent=1))
    print(json.dumps(sanity, indent=1))
    print(head.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
