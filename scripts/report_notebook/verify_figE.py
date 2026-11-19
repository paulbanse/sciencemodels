"""Independent verifier for F10a (figE_m2_choice_to_production).

Exposes ``check(outdir="output/report_notebook_v1", tol=1e-9) -> list[str]``:
empty list = all checks pass.  Every plotted quantity is recomputed from
the RAW source CSVs (never from the sidecar) and compared against
``F10a_m2_choice_to_production_data.csv``; the verifier also asserts the
expected input run/cell counts (40 off + 800 Path B v1 + 960 Path B v2 =
1,800 runs), the baseline_off vs baseline_off_summary aggregation
identity, and re-runs the figure build (which asserts zero clipped /
overlapping text artists on the canvas).
"""

from __future__ import annotations

import importlib.util
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parent.parent
DATA = REPO / "data"

KLANN_TEAM_SIZE = 2.65
KLANN_REPEATED_SHARE = 0.34
BENCH_LABEL = "Empirical benchmark (unfitted)"
Z95 = 1.96
LAM_LABEL = {0.0: "λ = 0 (prestige matching)",
             1.0: "λ = 1 (closeness × local merit)"}

V1_NAME = re.compile(
    r"cell_lam(\d+)_(logit|threshold)_(equal|struct)_(mgstd5|noisypr5)\.csv")
V2_NAME = re.compile(
    r"cell_lam(\d+)_pull(\d+)_(logit|threshold)_(equal|struct)_(mgstd5|noisypr5)\.csv")


def _mean_ci(vals) -> tuple[float, float, float, int]:
    v = np.asarray(vals, dtype=float)
    n = int(v.size)
    m = float(v.mean())
    half = Z95 * float(v.std(ddof=1)) / np.sqrt(n) if n > 1 else float("nan")
    return m, m - half, m + half, n


def _recompute() -> pd.DataFrame:
    """Rebuild the expected sidecar table from the raw CSVs (independent)."""
    failures: list[str] = []

    # -- raw loads with count assertions --
    off = pd.read_csv(DATA / "path_b_v1" / "baseline_off.csv")
    if len(off) != 40:
        failures.append(f"baseline_off.csv: expected 40 runs, got {len(off)}")
    per_land = off.groupby("land").size().to_dict()
    if per_land != {"mgstd5": 20, "noisypr5": 20}:
        failures.append(f"baseline_off.csv: 20 runs per land expected, got {per_land}")

    v1_rows = []
    v1_files = sorted((DATA / "path_b_v1" / "cells").glob("cell_lam*.csv"))
    if len(v1_files) != 40:
        failures.append(f"v1 cells: expected 40 files, got {len(v1_files)}")
    for f in v1_files:
        m = V1_NAME.match(f.name)
        d = pd.read_csv(f).copy()
        if len(d) != 20:
            failures.append(f"{f.name}: expected 20 runs, got {len(d)}")
        d["dyadic_lambda"] = int(m.group(1)) / 100.0
        d["dyadic_move_pull"] = 0.0
        d["formulation"] = m.group(2)
        d["credit_sharing_axis"] = ("equal_split" if m.group(3) == "equal"
                                    else "structural")
        d["landscape"] = m.group(4)
        v1_rows.append(d)
    v1 = pd.concat(v1_rows, ignore_index=True)

    v2_rows = []
    v2_files = sorted((DATA / "path_b_v2" / "cells").glob("cell_lam*_pull*.csv"))
    if len(v2_files) != 48:
        failures.append(f"v2 cells: expected 48 files, got {len(v2_files)}")
    for f in v2_files:
        m = V2_NAME.match(f.name)
        d = pd.read_csv(f).copy()
        if len(d) != 20:
            failures.append(f"{f.name}: expected 20 runs, got {len(d)}")
        d["dyadic_lambda"] = int(m.group(1)) / 100.0
        d["dyadic_move_pull"] = int(m.group(2)) / 100.0
        d["formulation"] = m.group(3)
        d["credit_sharing_axis"] = ("equal_split" if m.group(4) == "equal"
                                    else "structural")
        d["landscape"] = m.group(5)
        v2_rows.append(d)
    v2 = pd.concat(v2_rows, ignore_index=True)

    if len(v1) != 800:
        failures.append(f"v1 runs: expected 800, got {len(v1)}")
    if len(v2) != 960:
        failures.append(f"v2 runs: expected 960, got {len(v2)}")
    if len(off) + len(v1) + len(v2) != 1800:
        failures.append("Path B v1+v2+off must total 1,800 runs")

    # -- aggregation identity: baseline_off vs its archived summary --
    summ = pd.read_csv(DATA / "path_b_v1" / "baseline_off_summary.csv")
    for land, g in off.groupby("land"):
        row = summ[summ["land"] == land]
        for col in ("team_tm", "rep_tm", "links_tm", "c_ideal_tm",
                    "link_assort_tm"):
            if abs(float(g[col].mean()) - float(row[col].iloc[0])) > 1e-9:
                failures.append(f"baseline_off vs summary drift {land}.{col}")

    runs = pd.concat([v1, v2], ignore_index=True)

    # -- expected sidecar rows --
    exp: dict[tuple, tuple] = {}
    rows_meta: list[tuple] = []

    def add(panel, role, series, metric, x, *, vals=None, value=None):
        if vals is not None:
            v, lo, hi, n = _mean_ci(vals)
        else:
            v, lo, hi, n = float(value), float("nan"), float("nan"), 0
        key = (panel, role, series, metric,
               "NA" if x is None or (isinstance(x, float) and np.isnan(x))
               else f"{float(x):.3f}")
        if key in exp:
            failures.append(f"duplicate expected key {key}")
        exp[key] = (v, lo, hi, n)
        rows_meta.append(key)

    lands = ["mgstd5", "noisypr5"]
    pulls = sorted(float(p) for p in runs["dyadic_move_pull"].unique())

    for pull in pulls:
        sub_p = runs[runs["dyadic_move_pull"] == pull]
        lams = sorted(float(l) for l in sub_p["dyadic_lambda"].unique())
        for land in lands:
            for lam in lams:
                sel = sub_p[(sub_p["landscape"] == land)
                            & (sub_p["dyadic_lambda"] == lam)]
                if len(sel) != 80:
                    failures.append(
                        f"pull={pull} lam={lam} {land}: expected 80 runs, "
                        f"got {len(sel)}")
                series = f"pull = {pull:g} | {land}"
                add("A", "point", series, "links_tm", lam, vals=sel["links_tm"])
                add("B", "point", series, "team_tm", lam, vals=sel["team_tm"])

    for land in lands:
        sel = off[off["land"] == land]
        series = f"M1 (dyadic = off) | {land}"
        add("A", "reference", series, "links_tm", float("nan"),
            vals=sel["links_tm"])
        add("B", "reference", series, "team_tm", float("nan"),
            vals=sel["team_tm"])
    add("B", "reference", BENCH_LABEL, "team_tm", float("nan"),
        value=KLANN_TEAM_SIZE)

    for pull in (0.0, 1.0):
        for lam in (0.0, 1.0):
            sel = runs[(runs["dyadic_move_pull"] == pull)
                       & (runs["landscape"] == "mgstd5")
                       & (runs["dyadic_lambda"] == lam)]
            add("A", "annotation", f"pull = {pull:g} | mgstd5",
                "link_assort_tm", lam, vals=sel["link_assort_tm"])

    for land in lands:
        lams = sorted(float(l) for l in
                      runs[runs["dyadic_move_pull"] == 0.0]["dyadic_lambda"].unique())
        means = []
        for lam in lams:
            sel = runs[(runs["dyadic_move_pull"] == 0.0)
                       & (runs["landscape"] == land)
                       & (runs["dyadic_lambda"] == lam)]
            means.append(float(sel["team_tm"].mean()))
        add("B", "annotation", f"pull = 0 | {land}", "team_tm_pull0_min",
            float("nan"), value=min(means))
        add("B", "annotation", f"pull = 0 | {land}", "team_tm_pull0_max",
            float("nan"), value=max(means))

    for pull in pulls:
        for lam in (0.0, 1.0):
            for land in lands:
                sel = runs[(runs["dyadic_move_pull"] == pull)
                           & (runs["dyadic_lambda"] == lam)
                           & (runs["landscape"] == land)]
                series = f"{LAM_LABEL[lam]} | {land}"
                add("C", "point", series, "c_ideal_tm", pull,
                    vals=sel["c_ideal_tm"])
                add("C", "annotation", series, "wedge_tm", pull,
                    vals=sel["c_local_tm"] - sel["c_ideal_tm"])
    for land in lands:
        sel = off[off["land"] == land]
        add("C", "reference", f"M1 (dyadic = off) | {land}", "c_ideal_tm",
            float("nan"), vals=sel["c_ideal_tm"])

    sub_d = v2[(v2["dyadic_move_pull"] == 1.0) & (v2["landscape"] == "mgstd5")]
    for formulation in sorted(sub_d["formulation"].unique()):
        for credit in sorted(sub_d["credit_sharing_axis"].unique()):
            for lam in (0.0, 1.0):
                sel = sub_d[(sub_d["formulation"] == formulation)
                            & (sub_d["credit_sharing_axis"] == credit)
                            & (sub_d["dyadic_lambda"] == lam)]
                series = f"{formulation} + {credit} | {LAM_LABEL[lam]}"
                add("D", "point", series, "team_tm", float("nan"),
                    vals=sel["team_tm"])
                add("D", "point", series, "rep_tm", float("nan"),
                    vals=sel["rep_tm"])
    add("D", "reference", BENCH_LABEL, "rep_tm", float("nan"),
        value=KLANN_REPEATED_SHARE)

    if failures:
        return pd.DataFrame(), failures, exp
    return (pd.DataFrame(rows_meta, columns=["panel", "role", "series",
                                             "metric", "x"]),
            failures, exp)


def check(outdir: str = "output/report_notebook_v1", tol: float = 1e-9) -> list[str]:
    out = Path(outdir)
    if not out.is_absolute():
        out = REPO / out
    failures: list[str] = []

    _, load_failures, exp = _recompute()
    failures.extend(load_failures)
    if load_failures:
        return failures

    sidecar_path = out / "F10a_m2_choice_to_production_data.csv"
    if not sidecar_path.exists():
        return [f"missing sidecar {sidecar_path}"]
    got = pd.read_csv(sidecar_path)

    seen = set()
    for _, row in got.iterrows():
        key = (row["panel"], row["role"], row["series"], row["metric"],
               "NA" if pd.isna(row["x"]) else f"{float(row['x']):.3f}")
        if key in seen:
            failures.append(f"duplicate sidecar key {key}")
            continue
        seen.add(key)
        if key not in exp:
            failures.append(f"unexpected sidecar row {key}")
            continue
        v, lo, hi, n = exp[key]
        pairs = [("value", row["value"], v), ("ci_lo", row["ci_lo"], lo),
                 ("ci_hi", row["ci_hi"], hi)]
        for name, a, b in pairs:
            a, b = float(a), float(b)
            if np.isnan(a) and np.isnan(b):
                continue
            if np.isnan(a) != np.isnan(b) or abs(a - b) > tol:
                failures.append(
                    f"{key} {name}: sidecar {a!r} vs recomputed {b!r} "
                    f"(|diff| = {abs(a - b):.3g})")
        if int(row["n"]) != int(n):
            failures.append(f"{key} n: sidecar {int(row['n'])} vs {n}")
    for key in exp:
        if key not in seen:
            failures.append(f"missing sidecar row {key}")

    # text-fit: re-run the build (its assert_text_fits audits the canvas)
    try:
        spec = importlib.util.spec_from_file_location(
            "figE_m2_choice_to_production",
            _HERE / "figE_m2_choice_to_production.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as tmp:
            mod.build(outdir=tmp)
    except Exception as exc:  # noqa: BLE001
        failures.append(f"figure rebuild / text-fit assertion failed: {exc}")

    return failures


if __name__ == "__main__":
    fails = check()
    print("PASS" if not fails else "FAIL")
    for f in fails:
        print(" -", f)
    sys.exit(1 if fails else 0)
