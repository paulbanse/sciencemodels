"""Analysis for refill-gaussian v2 (peak tracking + P-0007 sabbatical pairing).

Reads ONLY ``data/refill_gaussian_v2/`` (cells + traj), requires the exact
panel shape (18 cells x 20 seeds), and writes:

- ``summary_by_cell.csv`` / ``summary_by_mode_reloc.csv``: means and SEs.
- ``paired_sabbatical.csv``: per (mode, c) seed-paired deltas (p_reloc 0.02
  minus 0.0) with 95% t-intervals (critical value computed, never typed).
- ``verdict_table.csv``: the pre-declared traveling-vs-diffusion rule applied
  per (mode, p_reloc) -- computed labels, not asserted prose.
- ``traj_summary.csv``: per (mode, p_reloc) mean trajectories of net centroid
  displacement, dispersion and agent-to-peak distance (from traj/).
- ``report_tables.md``: the ONE markdown renderer for the report's tables
  (``verify_refill_gaussian_v2.py`` re-renders these from the same CSVs and
  diffs against the report text).
- ``analysis.json``: headline numbers the report quotes.

Model-internal only (no calibration; owner ruling 2026-11-13).
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "refill_gaussian_v2"
CELLS = sorted(glob.glob(str(OUT / "cells" / "cell_*.csv")))
TRAJ = sorted(glob.glob(str(OUT / "traj" / "cell_*.csv")))

N_CELLS_EXPECTED = 18
N_SEEDS = 20

MODES = ("gaussian_s3", "gaussian_s6", "frontier")
CURIOSITIES = (0.3, 0.5, 0.7)
RELOCS = (0.0, 0.02)

MODE_LABEL = {"gaussian_s3": "gaussian_s3", "gaussian_s6": "gaussian_s6",
              "frontier": "frontier"}

# pre-declared verdict rule (must match the driver's design.json)
RULE = {
    "dir_at_least": 0.5, "net_at_least": 2.0,
    "disp_growth_at_most": 1.25, "agent_growth_at_most": 0.5,
    "dir_at_most": 0.2, "diff_net_below": 2.0, "diff_disp_growth_at_least": 1.5,
}

HEADLINE_METRICS = [
    "explored_percentage_final", "late_window_flow_rate",
    "corr_prestige_ideal_merit_final", "cumulative_harvest_flow",
    "centroid_net_disp_800", "mean_agent_peak_dist_800",
    "sabbatical_relocations",
]


def load_panel() -> pd.DataFrame:
    assert len(CELLS) == N_CELLS_EXPECTED, f"expected {N_CELLS_EXPECTED} cells, got {len(CELLS)}"
    frames = [pd.read_csv(p) for p in CELLS]
    panel = pd.concat(frames, ignore_index=True)
    assert len(panel) == N_CELLS_EXPECTED * N_SEEDS, f"panel rows {len(panel)}"
    counts = panel.groupby("cell_tag")["agent_seed"].count()
    assert (counts == N_SEEDS).all(), f"ragged cells: {counts[counts != N_SEEDS]}"
    # no probe leftovers: every cell_tag carries the reloc axis
    assert panel["cell_tag"].str.contains("reloc").all()
    return panel


def t_ci(values) -> tuple[float, float, float]:
    """Mean and 95% t-interval; critical value computed from n."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    n = len(v)
    mean = float(v.mean())
    if n < 2:
        return mean, float("nan"), float("nan")
    se = float(v.std(ddof=1) / math.sqrt(n))
    tcrit = float(stats.t.ppf(0.975, n - 1))
    return mean, mean - tcrit * se, mean + tcrit * se


def fmt(x: float, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def verdict_label(dirness: float, net: float, disp_growth: float,
                  agent_growth: float) -> str:
    if (dirness >= RULE["dir_at_least"] and net >= RULE["net_at_least"]
            and disp_growth <= RULE["disp_growth_at_most"]
            and agent_growth <= RULE["agent_growth_at_most"]):
        return "traveling frontier"
    if dirness <= RULE["dir_at_most"]:
        return "flat diffusion"
    if net < RULE["diff_net_below"] and disp_growth >= RULE["diff_disp_growth_at_least"]:
        return "flat diffusion"
    return "mixed / wandering"


def main() -> int:
    panel = load_panel()

    # ---------------------------------------------------------- summaries
    group = panel.groupby(["refill_mode", "p_reloc", "initial_curiosity"])
    summary = group[HEADLINE_METRICS].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]
    summary.to_csv(OUT / "summary_by_cell.csv", index=False)

    mode_reloc = panel.groupby(["refill_mode", "p_reloc"])[HEADLINE_METRICS].mean()
    mode_reloc.to_csv(OUT / "summary_by_mode_reloc.csv")

    # ------------------------------------------- paired sabbatical deltas
    pair_rows = []
    for mode in MODES:
        for c in CURIOSITIES:
            a = panel[(panel.refill_mode == mode) & (panel.initial_curiosity == c)
                      & (panel.p_reloc == 0.0)].set_index("agent_seed")
            b = panel[(panel.refill_mode == mode) & (panel.initial_curiosity == c)
                      & (panel.p_reloc == 0.02)].set_index("agent_seed")
            assert len(a) == len(b) == N_SEEDS
            for metric in HEADLINE_METRICS:
                delta = (b[metric] - a[metric]).to_numpy(dtype=float)
                mean, lo, hi = t_ci(delta)
                pair_rows.append(dict(mode=mode, curiosity=c, metric=metric,
                                      mean_delta=mean, ci_lo=lo, ci_hi=hi,
                                      n=int(np.isfinite(delta).sum())))
    paired = pd.DataFrame(pair_rows)
    paired.to_csv(OUT / "paired_sabbatical.csv", index=False)

    # -------------------------------------------------- verdict per arm
    verdict_rows = []
    for mode in MODES:
        for reloc in RELOCS:
            sub = panel[(panel.refill_mode == mode) & (panel.p_reloc == reloc)]
            dirness = float((sub["centroid_net_disp_800"]
                             / sub["centroid_path_len_800"]).mean())
            net = float(sub["centroid_net_disp_800"].mean())
            net400 = float(sub["centroid_net_disp_400"].mean())
            path = float(sub["centroid_path_len_800"].mean())
            disp_growth = float((sub["top5_dispersion_last_800"]
                                 / sub["top5_dispersion_last_400"]).mean())
            agent_growth = float((sub["mean_agent_peak_dist_800"]
                                  - sub["mean_agent_peak_dist_400"]).mean())
            label = verdict_label(dirness, net, disp_growth, agent_growth)
            verdict_rows.append(dict(
                refill_mode=mode, p_reloc=reloc,
                net_disp_400=net400, net_disp_800=net, path_len_800=path,
                directedness_800=dirness,
                agent_peak_400=float(sub["mean_agent_peak_dist_400"].mean()),
                agent_peak_800=float(sub["mean_agent_peak_dist_800"].mean()),
                agent_peak_growth=agent_growth,
                disp_last_400=float(sub["top5_dispersion_last_400"].mean()),
                disp_last_800=float(sub["top5_dispersion_last_800"].mean()),
                disp_growth=disp_growth,
                verdict=label))
    verdict = pd.DataFrame(verdict_rows)
    verdict.to_csv(OUT / "verdict_table.csv", index=False)

    # ------------------------------------------------------ traj summary
    # map cell files back to (mode, reloc) via the file name
    recs = []
    for p in TRAJ:
        name = Path(p).stem
        m = re.match(r"cell_refill-(gaussian_s\d+|frontier)_c(\d+)_gen(\d+)"
                     r"_reloc(\d+)_mgstd5", name)
        assert m, f"unparseable traj cell name: {name}"
        mode = m.group(1)
        reloc = int(m.group(4)) / 100.0
        df = pd.read_csv(p)
        df["refill_mode"] = mode
        df["p_reloc"] = reloc
        recs.append(df)
    traj = pd.concat(recs, ignore_index=True)
    traj_step = traj.groupby(["refill_mode", "p_reloc", "step"]).agg(
        agent_peak_dist=("agent_peak_dist", "mean"),
        dispersion=("dispersion", "mean"),
        cx=("cx", "mean"), cy=("cy", "mean"),
    ).reset_index()
    traj_step.to_csv(OUT / "traj_summary.csv", index=False)

    # ------------------------------------------------- report tables (ONE renderer)
    blocks = {}
    t1_rows = []
    for _, r in verdict.iterrows():
        t1_rows.append([
            r["refill_mode"], f"{r['p_reloc']:.2f}",
            fmt(r["net_disp_400"], 2), fmt(r["net_disp_800"], 2),
            fmt(r["path_len_800"], 1), fmt(r["directedness_800"]),
            fmt(r["agent_peak_400"], 2), fmt(r["agent_peak_800"], 2),
            fmt(r["disp_growth"], 2), r["verdict"]])
    blocks["table_tracking"] = md_table(
        ["mode", "p_reloc", "net disp H400", "net disp H800", "path H800",
         "directedness", "agent-peak H400", "agent-peak H800",
         "disp growth 800/400", "verdict (pre-declared rule)"], t1_rows)

    t2_rows = []
    for mode in MODES:
        for c in CURIOSITIES:
            for metric, nd in (("explored_percentage_final", 3),
                               ("late_window_flow_rate", 4),
                               ("corr_prestige_ideal_merit_final", 3)):
                m0 = panel[(panel.refill_mode == mode) & (panel.initial_curiosity == c)
                           & (panel.p_reloc == 0.0)][metric].mean()
                m1 = panel[(panel.refill_mode == mode) & (panel.initial_curiosity == c)
                           & (panel.p_reloc == 0.02)][metric].mean()
                pr = paired[(paired["mode"] == mode) & (paired.curiosity == c)
                            & (paired.metric == metric)].iloc[0]
                t2_rows.append([mode, f"{c:.1f}", metric,
                                fmt(m0, nd), fmt(m1, nd),
                                fmt(pr["mean_delta"], nd),
                                f"[{fmt(pr['ci_lo'], nd)}, {fmt(pr['ci_hi'], nd)}]"])
    blocks["table_pairing"] = md_table(
        ["mode", "c", "metric", "mean (p=0)", "mean (p=0.02)",
         "paired delta", "95% CI (t, n=20)"], t2_rows)

    (OUT / "report_tables.md").write_text(
        "<!-- rendered by analyse_refill_gaussian_v2.py; regenerate, never edit -->\n"
        + "\n\n".join(f"<!-- {k} -->\n{v}" for k, v in blocks.items()) + "\n")

    # ------------------------------------------------------- headline json
    def cell_mean(mode, reloc, c, metric):
        return float(panel[(panel.refill_mode == mode) & (panel.p_reloc == reloc)
                           & (panel.initial_curiosity == c)][metric].mean())

    s6_v = verdict[(verdict.refill_mode == "gaussian_s6")
                   & (verdict.p_reloc == 0.0)].iloc[0]
    s3_v = verdict[(verdict.refill_mode == "gaussian_s3")
                   & (verdict.p_reloc == 0.0)].iloc[0]
    f_v = verdict[(verdict.refill_mode == "frontier")
                  & (verdict.p_reloc == 0.0)].iloc[0]
    headline = {
        "panel_rows": int(len(panel)),
        "cells": int(panel["cell_tag"].nunique()),
        "s6_p0_verdict": s6_v["verdict"],
        "s3_p0_verdict": s3_v["verdict"],
        "frontier_p0_verdict": f_v["verdict"],
        "s6_net_disp_800": float(s6_v["net_disp_800"]),
        "s6_directedness": float(s6_v["directedness_800"]),
        "s3_net_disp_800": float(s3_v["net_disp_800"]),
        "frontier_net_disp_800": float(f_v["net_disp_800"]),
        "s6_agent_peak_growth": float(s6_v["agent_peak_growth"]),
        "s6_disp_growth": float(s6_v["disp_growth"]),
        "explore_pairing": {
            mode: {str(c): {
                "p0": cell_mean(mode, 0.0, c, "explored_percentage_final"),
                "p002": cell_mean(mode, 0.02, c, "explored_percentage_final"),
                "delta": float(paired[(paired["mode"] == mode)
                                      & (paired.curiosity == c)
                                      & (paired.metric == "explored_percentage_final")]
                               .iloc[0]["mean_delta"]),
            } for c in CURIOSITIES} for mode in MODES},
        "sabbatical_relocations_mean": {
            mode: float(panel[(panel.refill_mode == mode)
                              & (panel.p_reloc == 0.02)]
                        ["sabbatical_relocations"].mean()) for mode in MODES},
    }
    (OUT / "analysis.json").write_text(json.dumps(headline, indent=1))

    print(f"panel ok: {len(panel)} rows / {panel['cell_tag'].nunique()} cells")
    print(verdict.to_string(index=False))
    print(json.dumps(headline, indent=1)[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
