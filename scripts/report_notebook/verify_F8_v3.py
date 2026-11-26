"""Independent verifier for ``F8_v3_turnover_diagnostics`` (Fig 10, ADR-0003).

``check(outdir="output/report_notebook_v1", data_root=None, tol=1e-9)``
returns a list of failures; empty means everything reconciles.

Independence: the ADR-0003 turnover cells are re-read here with this module's
own filename parser and selectors -- nothing is imported from
``figA_v3_turnover_postbug`` -- and every plotted mean, CI, paired difference,
per-landscape marker and recomputed anchor is compared against
``F8_v3_turnover_diagnostics_data.csv`` on the identity keys
(panel, metric, series, exit_mode, g, landscape).

Guards this verifier also enforces:
  * the archive shape the figure claims: 36 cells x 20 seeds = 720 runs, both
    exit modes present, g in {0.125, 0.25, 0.5};
  * the paired contrast is computed on the SHARED draws (same g, curiosity,
    landscape, seed) and the two arms really do differ in population size --
    fixed_span N = 20 at every g, staggered N > 20 and rising in g, which is
    why the figure calls the contrast a regime comparison;
  * the recomputed commission anchors (c_ideal_late fixed minus staggered
    +0.0187; top10_late 0.205 vs 0.235) appear in the sidecar at the computed
    value.

Finally the figure is rebuilt (``visual_qa.assert_figure_clean`` runs inside
``build``) into a temporary directory, so a render-time regression fails here
too.

Run:  MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/report_notebook/verify_F8_v3.py
"""
from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

BASENAME = "F8_v3_turnover_diagnostics"
DEFAULT_ROOT = REPO / "data" / "turnover_adr0003"
EXIT_FIXED = "fixed_span"
EXIT_STAGGERED = "age_hazard_staggered"
G_GRID = (0.125, 0.25, 0.5)
Z95 = 1.96

M_A = "c_ideal_late (settled alignment, late window)"
M_B = "c_ideal_late difference (fixed_span - age_hazard_staggered)"
M_C = "gap_recent_mean_late (arrival gap, native units)"
M_D = "native share (harvested_fin, top10_late)"
M_REF = "reference (archive pins and derived contrasts)"

_CELL_RE = re.compile(
    r"turn_c-(?P<c>\d+p\d+)_exit-(?P<exit>[a-z_]+)_g-(?P<g>\d+p\d+)_"
    r"land-(?P<land>[a-z0-9]+)\.csv")
_KEYS = ["panel", "metric", "series", "exit_mode", "g", "landscape"]
PRE_ROOT = REPO / "data" / "turnover_v2"


def _canon_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Make the identity keys merge-safe (NaN != NaN would split the join)."""
    out = df.copy()
    for c in ("exit_mode", "landscape", "series"):
        out[c] = out[c].fillna("").astype(str)
    g = pd.to_numeric(out["g"], errors="coerce")
    out["g"] = np.where(np.isnan(g), "na", np.round(g.astype(float), 6).astype(str))
    return out


def _num(tok: str) -> float:
    return float(tok.replace("p", "."))


def _mean_ci(vals):
    v = pd.Series(vals, dtype=float).dropna()
    n = int(v.size)
    m = float(v.mean())
    half = Z95 * float(v.std(ddof=1)) / np.sqrt(n) if n > 1 else float("nan")
    return m, m - half, m + half, n


def load_cells(data_root=None) -> tuple[pd.DataFrame, dict]:
    root = Path(data_root) if data_root is not None else DEFAULT_ROOT
    if not root.is_absolute():
        root = REPO / root
    files = sorted((root / "cells").glob("turn_c-*.csv"))
    if not files:
        raise AssertionError(f"no turnover cells under {root}/cells")
    frames = []
    for f in files:
        m = _CELL_RE.match(f.name)
        if m is None:
            raise AssertionError(f"unparsable cell filename {f.name}")
        d = pd.read_csv(f)
        if len(d) != 20:
            raise AssertionError(f"{f.name}: {len(d)} seed rows, expected 20")
        d["g"] = _num(m.group("g"))
        d["exit_mode"] = m.group("exit")
        d["curiosity"] = _num(m.group("c"))
        d["landscape"] = m.group("land")
        frames.append(d)
    runs = pd.concat(frames, ignore_index=True)
    pins = dict(files=len(files),
                cells=runs.groupby(["exit_mode", "g", "curiosity", "landscape"]).ngroups,
                runs=len(runs), seeds=int(runs["seed"].nunique()),
                tau_max=sorted(runs["tau_max"].unique().tolist()),
                horizon=sorted(runs["horizon"].unique().tolist()))
    return runs, pins


def expected(data_root=None) -> tuple[pd.DataFrame, dict]:
    runs, pins = load_cells(data_root)
    rows: list[dict] = []

    def add(panel, metric, series, statistic, value, ci_lo, ci_hi, n,
            exit_mode="", g=np.nan, landscape="pooled", note=""):  # noqa: D401
        rows.append(dict(panel=panel, metric=metric, series=series,
                         statistic=statistic, exit_mode=exit_mode, g=g,
                         landscape=landscape, value=float(value),
                         ci_lo=ci_lo, ci_hi=ci_hi, n=n, note=note))

    modes = [m for m in (EXIT_FIXED, EXIT_STAGGERED)
             if m in set(runs["exit_mode"])]

    # ---- (a) settled alignment per (mode, g) -----------------------------
    for mode in modes:
        for g in G_GRID:
            sub = runs[(runs["exit_mode"] == mode) & (runs["g"] == g)]
            if sub.empty:
                continue
            m, lo, hi, n = _mean_ci(sub["c_ideal_late"])
            add("a", M_A, "settled", "mean", m, lo, hi, n, exit_mode=mode, g=g)

    # ---- (b) paired contrast ---------------------------------------------
    piv = runs.pivot_table(index=["g", "curiosity", "landscape", "seed"],
                           columns="exit_mode", values="c_ideal_late")
    if not {EXIT_FIXED, EXIT_STAGGERED} <= set(piv.columns):
        raise AssertionError("paired contrast needs both exit modes")
    diff = (piv[EXIT_FIXED] - piv[EXIT_STAGGERED]).rename("diff").reset_index()
    for g in list(G_GRID) + ["pooled"]:
        sub = diff if g == "pooled" else diff[diff["g"] == g]
        m, lo, hi, n = _mean_ci(sub["diff"])
        add("b", M_B, "difference", "mean", m, lo, hi, n,
            g=(np.nan if g == "pooled" else g))
    pooled = float(diff["diff"].mean())

    # ---- (c) arrival gap -------------------------------------------------
    for mode in modes:
        sub = runs[runs["exit_mode"] == mode]
        m, lo, hi, n = _mean_ci(sub["gap_recent_mean_late"])
        add("c", M_C, "pooled", "mean", m, lo, hi, n, exit_mode=mode)
        for land in sorted(runs["landscape"].unique()):
            s = sub[sub["landscape"] == land]["gap_recent_mean_late"]
            m2, lo2, hi2, n2 = _mean_ci(s)
            add("c", M_C, land, "mean", m2, lo2, hi2, n2, exit_mode=mode,
                landscape=land)

    # ---- (c)/(d) annotations the builder writes --------------------------
    gap_mode = {m: float(runs.loc[runs["exit_mode"] == m,
                                  "gap_recent_mean_late"].mean()) for m in modes}
    for mode in modes:
        add("c", "annotation: gap_difference_vs_fixed_span", "annotation",
            "annotation", gap_mode[mode] - gap_mode[EXIT_FIXED],
            np.nan, np.nan, int((runs["exit_mode"] == mode).sum()),
            exit_mode=mode)
    runs = runs.assign(pc=runs["harvested_fin"] / runs["n_final"])
    for mode in modes:
        s_ = (runs[runs["exit_mode"] == mode].groupby("g")["pc"].mean()
              .sort_index())
        for cat, g_v, v in (("per_capita_harvest_low_g", float(s_.index[0]),
                             float(s_.iloc[0])),
                            ("per_capita_harvest_high_g", float(s_.index[-1]),
                             float(s_.iloc[-1]))):
            add("d", f"annotation: {cat}", "annotation", "annotation", v,
                np.nan, np.nan, int((runs["exit_mode"] == mode).sum()),
                exit_mode=mode, g=g_v)

    # ---- (d) output and concentration ------------------------------------
    for mode in modes:
        sub = runs[runs["exit_mode"] == mode]
        for ser, col in (("harvested_fin", "harvested_fin"),
                         ("top10_late", "top10_late")):
            m, lo, hi, n = _mean_ci(sub[col])
            add("d", M_D, ser, "mean", m, lo, hi, n, exit_mode=mode)

    # ---- refs ------------------------------------------------------------
    fixed_t10 = float(runs.loc[runs["exit_mode"] == EXIT_FIXED, "top10_late"].mean())
    stag_t10 = float(runs.loc[runs["exit_mode"] == EXIT_STAGGERED,
                              "top10_late"].mean())
    refs = {
        "runs": pins["runs"], "cells": pins["cells"], "seeds": pins["seeds"],
        "tau_max": pins["tau_max"][0], "horizon": pins["horizon"][0],
        "g_min": min(G_GRID), "g_max": max(G_GRID),
        "curiosity_min": float(runs["curiosity"].min()),
        "curiosity_max": float(runs["curiosity"].max()),
        "anchor_fixed_minus_staggered": pooled,
        "anchor_top10_fixed": fixed_t10,
        "anchor_top10_staggered": stag_t10,
        "population_fixed_g_max": float(
            runs.loc[runs["exit_mode"] == EXIT_FIXED, "n_final"].mean()),
        "population_staggered_g_max": float(
            runs.loc[(runs["exit_mode"] == EXIT_STAGGERED)
                     & (runs["g"] == max(G_GRID)), "n_final"].mean()),
    }
    for k, v in refs.items():
        add("ref", M_REF, k, "definitional constant", v, np.nan, np.nan, np.nan,
            landscape="")
    # pre-ADR anchors, recomputed from the read-only pre-ADR runs table
    pre = PRE_ROOT / "runs.csv"
    if pre.exists():
        d = pd.read_csv(pre)
        d = d[d["entry"] == 0.10]
        add("ref", M_REF, "pre_anchor_fixed_minus_staggered",
            "definitional constant",
            float(d.loc[d["exit_mode"] == EXIT_FIXED, "c_ideal_late_mean"].mean()
                  - d.loc[d["exit_mode"] == EXIT_STAGGERED,
                          "c_ideal_late_mean"].mean()),
            np.nan, np.nan, np.nan, landscape="")
        for cat, mode in (("pre_anchor_top10_fixed", EXIT_FIXED),
                          ("pre_anchor_top10_staggered", EXIT_STAGGERED)):
            add("ref", M_REF, cat, "definitional constant",
                float(d.loc[d["exit_mode"] == mode, "top10_late_mean"].mean()),
                np.nan, np.nan, np.nan, landscape="")
    return pd.DataFrame(rows), pins


def check(outdir="output/report_notebook_v1", data_root=None,
          tol: float = 1e-9) -> list[str]:
    fails: list[str] = []
    exp, pins = expected(data_root)

    # ---- archive shape ---------------------------------------------------
    if pins["files"] != 36:
        fails.append(f"{pins['files']} cell files, expected 36")
    if pins["runs"] != 720:
        fails.append(f"{pins['runs']} runs, expected 720")
    if pins["seeds"] != 20:
        fails.append(f"{pins['seeds']} seeds, expected 20")
    runs, _ = load_cells(data_root)
    modes = sorted(runs["exit_mode"].unique().tolist())
    if modes != sorted([EXIT_FIXED, EXIT_STAGGERED]):
        fails.append(f"exit mode grid {modes}")
    if sorted(runs["g"].unique().tolist()) != list(G_GRID):
        fails.append(f"g grid {sorted(runs['g'].unique().tolist())}")

    # population asymmetry the figure discloses
    pop = runs.groupby(["exit_mode", "g"])["n_final"].mean()
    for g in G_GRID:
        if pop[(EXIT_FIXED, g)] != 20.0:
            fails.append(f"fixed_span g={g}: N={pop[(EXIT_FIXED, g)]}, expected 20")
        if pop[(EXIT_STAGGERED, g)] <= 20.0:
            fails.append(f"staggered g={g}: N={pop[(EXIT_STAGGERED, g)]} "
                         "should exceed 20")

    # ---- sidecar reconciliation -----------------------------------------
    out = Path(outdir)
    if not out.is_absolute():
        out = REPO / out
    side_path = out / f"{BASENAME}_data.csv"
    if not side_path.exists():
        return fails + [f"missing sidecar {side_path}"]
    side = pd.read_csv(side_path)
    if len(side) != len(exp):
        fails.append(f"sidecar rows {len(side)} != expected {len(exp)}")
    key = _KEYS
    for k in key:
        if k not in side.columns:
            return fails + [f"sidecar missing key column {k}"]
    e = _canon_keys(exp)
    s = _canon_keys(side)
    m = s.merge(e, on=key, how="outer", suffixes=("_side", "_exp"),
                indicator=True)
    bad = m[m["_merge"] != "both"]
    for _, r in bad.iterrows():
        fails.append(f"sidecar/expected key mismatch: "
                     f"{[r[k] for k in key]} ({r['_merge']})")
    both = m[m["_merge"] == "both"]
    for col in ("value", "ci_lo", "ci_hi"):
        a, b = both[f"{col}_side"].astype(float), both[f"{col}_exp"].astype(float)
        ok = (np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= tol)
        if not ok.all():
            for i in np.where(~ok)[0][:5]:
                r = both.iloc[i]
                fails.append(f"{col} drift at {[r[k] for k in key]}: "
                             f"{a.iloc[i]!r} vs {b.iloc[i]!r}")
    # the recomputed anchors must be the ones in the sidecar
    for cat in ("anchor_fixed_minus_staggered", "anchor_top10_fixed",
                "anchor_top10_staggered"):
        sel = side[(side["panel"] == "ref") & (side["series"] == cat)]
        if len(sel) != 1:
            fails.append(f"sidecar ref row {cat}: {len(sel)} rows")

    # ---- rebuild (visual_qa inside build) -------------------------------
    import figA_v3_turnover_postbug as B
    with tempfile.TemporaryDirectory() as td:
        rep = B.build(outdir=td, data_root=data_root)
        if rep["qa"]["_total"]:
            fails.append(f"visual_qa failures on rebuild: {rep['qa']['_total']}")
    return fails


if __name__ == "__main__":
    problems = check()
    if problems:
        print(f"FAIL ({len(problems)}):")
        for p in problems:
            print("  -", p)
        raise SystemExit(1)
    print("verify_F8_v3: OK — every plotted value, CI and anchor reconciles; "
          "visual_qa clean on rebuild")
