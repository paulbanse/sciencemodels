"""Independent verifier for F10b (figF_m2_visibility_like_for_like).

Exposes ``check(outdir="output/report_notebook_v1", tol=1e-9) -> list[str]``:
empty list = all checks pass.  Every plotted quantity is recomputed from
the RAW source CSVs (never from the sidecar) and compared against
``F10b_m2_visibility_like_for_like_data.csv``; the verifier also asserts
the expected input run/cell counts (w_v05: 10 summary rows / 200 runs;
M2 xi-extension: 8 x 20 + 2 x 20 = 200 runs), the w_v05 runs-vs-summary
aggregation identity, and re-runs the figure build (which asserts zero
clipped / overlapping text artists on the canvas).
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
BENCH_LABEL = "Empirical benchmark (unfitted)"
Z95 = 1.96

M0_SERIES = "M0 (g = 0)"
M1_SERIES = "M1 (g = 0.1)"
M2_SERIES = {0.0: "M2 λ = 0 (prestige matching)",
             1.0: "M2 λ = 1 (closeness × local merit)"}
M2_NAME = re.compile(
    r"m2_cell_lam(\d+)_pull(\d+)_(logit|threshold)_(equal|struct)_(\w+)_cur(\d+)\.csv")


def _mean_ci(vals) -> tuple[float, float, float, int]:
    v = np.asarray(vals, dtype=float)
    n = int(v.size)
    m = float(v.mean())
    half = Z95 * float(v.std(ddof=1)) / np.sqrt(n) if n > 1 else float("nan")
    return m, m - half, m + half, n


def _recompute():
    failures: list[str] = []

    # -- M0/M1 raw loads with count assertions --
    summ = pd.read_csv(DATA / "remeasure_v1" / "w_v05_summary.csv")
    runs = pd.read_csv(DATA / "remeasure_v1" / "w_v05_runs.csv")
    sel_s = summ[(summ["landscape"] == "mgstd5")
                 & (summ["use_visibility_reward"] == True)]  # noqa: E712
    sel_r = runs[(runs["landscape"] == "mgstd5")
                 & (runs["use_visibility_reward"] == True)]  # noqa: E712
    if len(sel_s) != 10:
        failures.append(f"w_v05_summary mgstd5+vis: expected 10 rows, got {len(sel_s)}")
    if len(sel_r) != 200:
        failures.append(f"w_v05_runs mgstd5+vis: expected 200 runs, got {len(sel_r)}")
    if not (sel_s["n_seeds"] == 20).all():
        failures.append("w_v05_summary: n_seeds != 20")
    sizes = sel_r.groupby(["agent_generation_time", "curiosity"]).size()
    if not (sizes == 20).all():
        failures.append(f"w_v05_runs: expected 20 seeds per cell, got {sizes.to_dict()}")

    # aggregation identity: run means == summary means
    agg = sel_r.groupby(["agent_generation_time", "curiosity"]).mean(numeric_only=True)
    for _, row in sel_s.iterrows():
        a = agg.loc[(row["agent_generation_time"], row["curiosity"])]
        for col in ("harvested_fin", "c_ideal_tm", "wedge_local_minus_ideal_tm"):
            if abs(float(a[col]) - float(row[col])) > 1e-9:
                failures.append(f"w_v05 runs-vs-summary drift: {col}")

    # -- M2 raw loads --
    m2_rows = []
    ext_files = sorted((DATA / "fig1_extension_v1" / "cells").glob("m2_*.csv"))
    if len(ext_files) != 8:
        failures.append(f"fig1_extension m2 cells: expected 8 files, got {len(ext_files)}")
    for f in ext_files:
        m = M2_NAME.match(f.name)
        d = pd.read_csv(f).copy()
        if len(d) != 20:
            failures.append(f"{f.name}: expected 20 runs, got {len(d)}")
        if m is None or not (d["dyadic_lambda"] == int(m.group(1)) / 100.0).all():
            failures.append(f"{f.name}: dyadic_lambda mismatch")
        if not (d["dyadic_move_pull"] == 1.0).all():
            failures.append(f"{f.name}: expected pull = 1.0")
        if not ((d["formulation"] == "logit")
                & (d["credit_sharing_axis"] == "equal_split")
                & (d["landscape"] == "mgstd5")).all():
            failures.append(f"{f.name}: expected logit + equal_split, mgstd5")
        d["xi"] = np.round(1.0 - d["initial_curiosity"], 1)
        m2_rows.append(d)
    anchor_rows = []
    for lam in (0.0, 1.0):
        tag = "000" if lam == 0.0 else "100"
        f = (DATA / "path_b_v2" / "cells"
             / f"cell_lam{tag}_pull100_logit_equal_mgstd5.csv")
        d = pd.read_csv(f).copy()
        if len(d) != 20:
            failures.append(f"{f.name}: expected 20 runs, got {len(d)}")
        if not ((d["dyadic_lambda"] == lam) & (d["dyadic_move_pull"] == 1.0)
                & (d["initial_curiosity"] == 0.5)).all():
            failures.append(f"{f.name}: config mismatch")
        d["xi"] = 0.5
        anchor_rows.append(d)
    ext = pd.concat(m2_rows, ignore_index=True)
    m2 = pd.concat(m2_rows + anchor_rows, ignore_index=True)
    if len(ext) != 160:
        failures.append(f"m2 extension runs: expected 160, got {len(ext)}")
    if len(m2) != 200:
        failures.append(f"M2 xi-extension total: expected 200 runs, got {len(m2)}")

    # -- expected sidecar rows --
    exp: dict[tuple, tuple] = {}
    keys: list[tuple] = []

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
        keys.append(key)

    xis = [0.1, 0.3, 0.5, 0.7, 0.9]

    for g, series in ((0.0, M0_SERIES), (0.1, M1_SERIES)):
        for xi in xis:
            cur = round(1.0 - xi, 1)
            row = sel_s[(np.round(sel_s["agent_generation_time"], 2) == g)
                        & (np.round(sel_s["curiosity"], 1) == cur)]
            if len(row) != 1:
                failures.append(f"w_v05_summary row g={g} cur={cur} missing")
                continue
            sel = sel_r[(np.round(sel_r["agent_generation_time"], 2) == g)
                        & (np.round(sel_r["curiosity"], 1) == cur)]
            for panel, metric, col in (("A", "harvested_share", "harvested_fin"),
                                       ("B", "c_ideal_tm", "c_ideal_tm")):
                _, lo, hi, n = _mean_ci(sel[col])
                key = (panel, "point", series, metric, f"{xi:.3f}")
                exp[key] = (float(row[col].iloc[0]), lo, hi, n)
                keys.append(key)

    for lam, series in M2_SERIES.items():
        for xi in xis:
            sel = m2[(m2["dyadic_lambda"] == lam) & (m2["xi"] == xi)]
            if len(sel) != 20:
                failures.append(f"M2 lam={lam} xi={xi}: expected 20 runs, "
                                f"got {len(sel)}")
            for panel, metric, col in (("A", "harvested_share",
                                        "percentage_knowledge_harvested"),
                                       ("B", "c_ideal_tm", "c_ideal_tm"),
                                       ("C", "team_tm", "team_tm")):
                add(panel, "point", series, metric, xi, vals=sel[col])

    add("C", "reference", BENCH_LABEL, "team_tm", float("nan"),
        value=KLANN_TEAM_SIZE)

    entry_arms = (M1_SERIES, M2_SERIES[0.0], M2_SERIES[1.0])

    def arm_values(panel, metric, series_list, xi_list):
        return [v for (p, role, s, m_, x), (v, _, _, _) in exp.items()
                if p == panel and role == "point" and m_ == metric
                and s in series_list and x in {f"{q:.3f}" for q in xi_list}]

    plateau = arm_values("A", "harvested_share", entry_arms, [0.1, 0.3, 0.5])
    collapse = arm_values("A", "harvested_share", entry_arms, [0.9])
    add("A", "annotation", "entry arms (M1 + M2)", "plateau_harvested_min",
        float("nan"), value=min(plateau))
    add("A", "annotation", "entry arms (M1 + M2)", "plateau_harvested_max",
        float("nan"), value=max(plateau))
    add("A", "annotation", "entry arms (M1 + M2)", "stage2_collapse_min",
        float("nan"), value=min(collapse))
    add("A", "annotation", "entry arms (M1 + M2)", "stage2_collapse_max",
        float("nan"), value=max(collapse))

    small = arm_values("C", "team_tm", list(M2_SERIES.values()),
                       [0.1, 0.3, 0.5])
    mob = arm_values("C", "team_tm", list(M2_SERIES.values()), [0.7, 0.9])
    add("C", "annotation", "M2 (both λ)", "team_small_min", float("nan"),
        value=min(small))
    add("C", "annotation", "M2 (both λ)", "team_small_max", float("nan"),
        value=max(small))
    add("C", "annotation", "M2 (both λ)", "team_mob_min", float("nan"),
        value=min(mob))
    add("C", "annotation", "M2 (both λ)", "team_mob_max", float("nan"),
        value=max(mob))

    # cross-check rows: M1 c_ideal_tm at xi = 0.5 in both archives
    m1_xi05 = float(sel_s[(np.round(sel_s["agent_generation_time"], 2) == 0.1)
                          & (np.round(sel_s["curiosity"], 1) == 0.5)]
                    ["c_ideal_tm"].iloc[0])
    add("B", "annotation", M1_SERIES, "crosscheck_xi05_remeasure_vanishing05",
        0.5, value=m1_xi05)
    pb = pd.read_csv(DATA / "path_b_v1" / "baseline_off_summary.csv")
    add("B", "annotation", M1_SERIES, "crosscheck_xi05_pathb_vanishing09",
        0.5, value=float(pb[pb["land"] == "mgstd5"]["c_ideal_tm"].iloc[0]))

    return keys, failures, exp


def check(outdir: str = "output/report_notebook_v1", tol: float = 1e-9) -> list[str]:
    out = Path(outdir)
    if not out.is_absolute():
        out = REPO / out
    failures: list[str] = []

    _, load_failures, exp = _recompute()
    failures.extend(load_failures)
    if load_failures:
        return failures

    sidecar_path = out / "F10b_m2_visibility_like_for_like_data.csv"
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
        for name, a, b in (("value", row["value"], v),
                           ("ci_lo", row["ci_lo"], lo),
                           ("ci_hi", row["ci_hi"], hi)):
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
            "figF_m2_visibility_like_for_like",
            _HERE / "figF_m2_visibility_like_for_like.py")
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
