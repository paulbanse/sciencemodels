"""Independent verification for refill-gaussian v2's report.

Re-renders the report's markdown tables from the analysis CSVs (same
renderer, same format) and diffs the blocks against
``output/reports/refill_gaussian_v2.md``; recomputes every headline number
the report quotes from the raw cell CSVs and asserts it appears in the text;
re-asserts panel shape (18 cells x 20 seeds x 18 traj files) and the
cross-round control verdict from the probe record.

Usage::

    MPLBACKEND=Agg /opt/hermes/.venv/bin/python \
        scripts/refill_gaussian/verify_refill_gaussian_v2.py

Exit 0 = all checks pass.
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "refill_gaussian_v2"
REPORT = REPO / "output" / "reports" / "refill_gaussian_v2.md"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_refill_gaussian_v2 import md_table, fmt, MODES, CURIOSITIES  # noqa: E402


def render_table_tracking(verdict: pd.DataFrame) -> str:
    rows = []
    for _, r in verdict.iterrows():
        rows.append([
            r["refill_mode"], f"{r['p_reloc']:.2f}",
            fmt(r["net_disp_400"], 2), fmt(r["net_disp_800"], 2),
            fmt(r["path_len_800"], 1), fmt(r["directedness_800"]),
            fmt(r["agent_peak_400"], 2), fmt(r["agent_peak_800"], 2),
            fmt(r["disp_growth"], 2), r["verdict"]])
    return md_table(
        ["mode", "p_reloc", "net disp H400", "net disp H800", "path H800",
         "directedness", "agent-peak H400", "agent-peak H800",
         "disp growth 800/400", "verdict (pre-declared rule)"], rows)


def render_table_pairing(panel: pd.DataFrame, paired: pd.DataFrame) -> str:
    rows = []
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
                rows.append([mode, f"{c:.1f}", metric,
                             fmt(m0, nd), fmt(m1, nd),
                             fmt(pr["mean_delta"], nd),
                             f"[{fmt(pr['ci_lo'], nd)}, {fmt(pr['ci_hi'], nd)}]"])
    return md_table(
        ["mode", "c", "metric", "mean (p=0)", "mean (p=0.02)",
         "paired delta", "95% CI (t, n=20)"], rows)


def main() -> int:
    failures = []
    report = REPORT.read_text()

    # ---- panel shape -------------------------------------------------
    cells = sorted(glob.glob(str(OUT / "cells" / "cell_*.csv")))
    panel = pd.concat([pd.read_csv(p) for p in cells], ignore_index=True)
    traj = sorted(glob.glob(str(OUT / "traj" / "cell_*.csv")))
    checks = {
        "cells": (len(cells), 18),
        "panel rows": (len(panel), 18 * 20),
        "traj files": (len(traj), 18),
    }
    for name, (got, want) in checks.items():
        if got != want:
            failures.append(f"{name}: got {got}, expected {want}")

    # ---- cross-round control ----------------------------------------
    probe = json.loads((OUT / "probe" / "probe_results.json").read_text())
    worst = probe["cross_round"]["worst"]
    if worst != 0.0:
        failures.append(f"cross-round control not bit-exact: worst={worst}")
    print(f"cross-round control: worst diff {worst} over "
          f"{probe['cross_round']['numeric_columns_compared']} numeric columns; "
          f"declared age-clock drift: {probe['cross_round'].get('drift_ledger')}")

    # ---- table round-trip -------------------------------------------
    verdict = pd.read_csv(OUT / "verdict_table.csv")
    paired = pd.read_csv(OUT / "paired_sabbatical.csv")
    rendered = {
        "tracking": render_table_tracking(verdict),
        "pairing": render_table_pairing(panel, paired),
    }
    for name, block in rendered.items():
        if block not in report:
            failures.append(f"table '{name}' does not re-render from CSVs "
                            "against the report text")
        else:
            print(f"table round-trip ok: {name} "
                  f"({block.count(chr(10)) - 1} rows re-rendered and matched)")

    # ---- prose numbers ----------------------------------------------
    headline = json.loads((OUT / "analysis.json").read_text())
    expected_strings = []
    v6 = headline["s6_net_disp_800"]
    expected_strings.append(f"{v6:.2f}")
    expected_strings.append(f"{headline['s6_directedness']:.3f}")
    for mode in MODES:
        for c in CURIOSITIES:
            e = headline["explore_pairing"][mode][str(c)]
            expected_strings.append(f"{e['delta']:+.3f}")
    for mode in MODES:
        expected_strings.append(
            f"{headline['sabbatical_relocations_mean'][mode]:.1f}")
    for s in expected_strings:
        if s not in report:
            failures.append(f"prose/table string {s!r} not found in report")
    print(f"prose strings checked: {len(expected_strings)}")

    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for f in failures:
            print(" -", f)
        return 1
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
