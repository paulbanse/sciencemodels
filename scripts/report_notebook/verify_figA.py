"""Independent verifier for Figure A / ``F8_v2_turnover_diagnostics``.

``check()`` recomputes EVERY plotted quantity and rendered annotation from
the RAW source CSVs in ``data/turnover_v1/`` (never from the sidecar), and
compares the result against ``F8_v2_turnover_diagnostics_data.csv`` at a
configurable tolerance (default 1e-9).  It also asserts the expected input
counts for this figure: 54 trajectory cells x 400 steps (18 cells per exit
mode) and 54 cells x 20 seeds = 1,080 runs (360 per mode).

The recomputation below is deliberately self-contained (no import of
``figA_turnover_diagnostics``); the annotation format strings are part of
the figure's specification and are re-rendered here from the raw numbers.

Run:  MPLBACKEND=Agg python3 scripts/report_notebook/verify_figA.py
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = REPO / "data" / "turnover_v1"
BASENAME = "F8_v2_turnover_diagnostics"

MODES = ["none", "fixed_span", "age_hazard"]
LANDSCAPES = ["mgstd5", "noisypr5"]
G_LEVELS = [0.05, 0.10, 0.20]

DISCOVERY_END = 80
DEPLETION_START = 150
BIRTH_SHOCK_STEPS = (200, 400)
LATE_START, LATE_END = 301, 399
EQUALITY_TOP10 = 0.10
Z95 = 1.96

_TRAJ_RE = re.compile(r"cell_([a-z_]+?)_g(\d+)_cur(\d+)_(mgstd5|noisypr5)\.csv")
SRC_TRAJ = "data/turnover_v1/traj/cell_*.csv"
SRC_CELLS = "data/turnover_v1/cells/cell_*.csv"

S_NB = ("mean of c_ideal across cells whose step t is a non-birth step "
        "(floor(t*g) = floor((t-1)*g)) and t not in {200, 400}; 95% CI = "
        "mean +/- 1.96*SE across contributing cells")
S_BIRTH = ("mean of c_ideal across cells whose step t is a birth step "
           "(floor(t*g) > floor((t-1)*g))")
S_B_SETTLED = ("mean of c_ideal over non-birth steps 301-399 pooled over "
               "the 6 (curiosity x landscape) cells of the (mode, g) group")
S_B_SNAP = ("mean of c_ideal at step 400 (birth step) over the 6 "
            "(curiosity x landscape) cells of the (mode, g) group")
S_C = ("mean of gap_recent_mean_last100 over runs; 95% CI = mean +/- "
       "1.96*SE over runs")
S_D = {
    "harvested_fin": "mean of harvested_fin over runs; 95% CI = mean +/- 1.96*SE over runs",
    "top10_step400": "mean of top10_fin (step 400) over runs; 95% CI = mean +/- 1.96*SE over runs",
    "top10_step399": ("mean of traj top10 at step 399 over the 18 cell "
                      "trajectories; 95% CI = mean +/- 1.96*SE across cells"),
}

_REF_ROWS = [
    ("discovery_end", DISCOVERY_END, DISCOVERY_END,
     f"Discovery (t ≤ {DISCOVERY_END})"),
    ("depletion_start", DEPLETION_START, DEPLETION_START,
     f"Depletion (t ≥ {DEPLETION_START})"),
    ("birth_shock_step", BIRTH_SHOCK_STEPS[0], BIRTH_SHOCK_STEPS[0],
     "dotted drop-line; tau_max = 200 synchronized cohort replacement"),
    ("birth_shock_step", BIRTH_SHOCK_STEPS[1], BIRTH_SHOCK_STEPS[1],
     "dotted drop-line; tau_max = 200 synchronized cohort replacement"),
    ("late_window_start", LATE_START, LATE_START, ""),
    ("late_window_end", LATE_END, LATE_END, ""),
    ("equality_top10", EQUALITY_TOP10, np.nan,
     f"{EQUALITY_TOP10:.2f} equality"),
    ("z95", Z95, np.nan, "95% CI multiplier (house convention)"),
]


# --- annotation rendering (spec; mirrored from the build module) ----------

def _frag_a(mode: str, b400: float, nb399: float) -> str:
    return f"{mode} {b400:+.3f} vs {nb399:+.3f}"


def _line_a(frags: list[str]) -> str:
    return "step 400 (birth) vs step 399 (settled): " + " · ".join(frags)


def _frag_b(mode: str, v05: float, v20: float) -> str:
    return f"{mode} {v05:+.3f}→{v20:+.3f}"


def _line_b_settled(frags: list[str]) -> str:
    return ("settled non-birth mean 301–399 (g 0.05→0.20): "
            + " · ".join(frags))


def _line_b_snap(frags: list[str]) -> str:
    return "step-400 birth snapshot (g 0.05→0.20): " + " · ".join(frags)


def _label_c(v: float, rel: float, mode: str) -> str:
    return f"{v:.5f}" if mode == "none" else f"{v:.5f}\n[{rel:.1f}%]"


def _note_c(neg: int, tot: int) -> str:
    return f"negative in {neg:,}/{tot:,} runs"


def _label_d(series: str, v: float) -> str:
    return f"{100 * v:.1f}%" if series == "harvested_fin" else f"{v:.3f}"


# --- raw inputs ------------------------------------------------------------

def _read_raw():
    traj_files = sorted((DATA / "traj").glob("cell_*.csv"))
    cells_files = sorted((DATA / "cells").glob("cell_*.csv"))
    tframes, cframes = [], []
    for f in traj_files:
        m = _TRAJ_RE.match(f.name)
        d = pd.read_csv(f)
        d["exit_mode"] = m.group(1) if m else ""
        d["g"] = int(m.group(2)) / 100 if m else np.nan
        tframes.append(d)
    for f in cells_files:
        d = pd.read_csv(f)
        d["_file"] = f.name
        cframes.append(d)
    traj = pd.concat(tframes, ignore_index=True)
    cells = pd.concat(cframes, ignore_index=True)
    t = traj["step"].to_numpy(float)
    g = traj["g"].to_numpy(float)
    traj["is_birth"] = np.floor(t * g) > np.floor((t - 1) * g)
    traj["non_birth"] = (~traj["is_birth"]) & (
        ~traj["step"].isin(BIRTH_SHOCK_STEPS))
    return traj_files, cells_files, traj, cells


def _mean_ci(vals) -> tuple[float, float, float, int]:
    v = pd.Series(vals, dtype=float).dropna()
    n = int(v.size)
    mean = float(v.mean())
    ci = Z95 * float(v.std(ddof=1)) / np.sqrt(n) if n > 1 else float("nan")
    return mean, mean - ci, mean + ci, n


# --- expected sidecar (recomputed from the raw CSVs) -----------------------

def _expected() -> pd.DataFrame:
    _tf, _cf, traj, cells = _read_raw()
    rows: list[dict] = []

    def row(**kw):
        rows.append(kw)

    # (a) non-birth trajectory
    for (mode, step), grp in traj[traj["non_birth"]].groupby(
            ["exit_mode", "step"]):
        v, lo, hi, n = _mean_ci(grp["c_ideal"])
        row(panel="a", series="c_ideal_nonbirth_mean", exit_mode=mode,
            g=np.nan, landscape="pooled", step=int(step), category="",
            value=v, ci_lo=lo, ci_hi=hi, n=n, stat=S_NB, annotation="",
            source=SRC_TRAJ)
    # (a) birth-step means
    for (mode, step), grp in traj[traj["is_birth"]].groupby(
            ["exit_mode", "step"]):
        row(panel="a", series="c_ideal_birth_mean", exit_mode=mode,
            g=np.nan, landscape="pooled", step=int(step), category="",
            value=float(grp["c_ideal"].mean()), ci_lo=np.nan, ci_hi=np.nan,
            n=int(grp["c_ideal"].size), stat=S_BIRTH, annotation="",
            source=SRC_TRAJ)

    b400 = traj[traj["step"] == 400].groupby("exit_mode")["c_ideal"].mean()
    nb399 = (traj[(traj["step"] == 399) & traj["non_birth"]]
             .groupby("exit_mode")["c_ideal"].mean())
    frags_a = [_frag_a(m, float(b400[m]), float(nb399[m])) for m in MODES]
    line_a = _line_a(frags_a)
    for m, frag in zip(MODES, frags_a):
        row(panel="a", series="annotation", exit_mode=m, g=np.nan,
            landscape="pooled", step=400, category="callout_fragment",
            value=float(b400[m]), ci_lo=np.nan, ci_hi=np.nan, n=np.nan,
            stat="", annotation=frag, source=SRC_TRAJ)
    row(panel="a", series="annotation", exit_mode="", g=np.nan,
        landscape="", step=np.nan, category="callout_line", value=np.nan,
        ci_lo=np.nan, ci_hi=np.nan, n=np.nan, stat="", annotation=line_a,
        source=SRC_TRAJ)

    # (b) entry-rate gradient
    late_rows, snap_rows = {}, {}
    for (mode, g), grp in traj[(traj["step"] >= LATE_START)
                               & (traj["step"] <= LATE_END)
                               & traj["non_birth"]].groupby(
                                   ["exit_mode", "g"]):
        late_rows[(mode, float(g))] = (float(grp["c_ideal"].mean()),
                                       int(grp["c_ideal"].size))
    for (mode, g), grp in traj[traj["step"] == BIRTH_SHOCK_STEPS[-1]] \
            .groupby(["exit_mode", "g"]):
        snap_rows[(mode, float(g))] = (float(grp["c_ideal"].mean()),
                                       int(grp["c_ideal"].size))
    for mode in MODES:
        for g in G_LEVELS:
            v, n = late_rows[(mode, g)]
            row(panel="b", series="settled_late_nonbirth_mean",
                exit_mode=mode, g=g, landscape="pooled", step=np.nan,
                category="", value=v, ci_lo=np.nan, ci_hi=np.nan, n=n,
                stat=S_B_SETTLED, annotation="", source=SRC_TRAJ)
            v, n = snap_rows[(mode, g)]
            row(panel="b", series="step400_birth_snapshot", exit_mode=mode,
                g=g, landscape="pooled", step=BIRTH_SHOCK_STEPS[-1],
                category="", value=v, ci_lo=np.nan, ci_hi=np.nan, n=n,
                stat=S_B_SNAP, annotation="", source=SRC_TRAJ)

    settled_ends = [(late_rows[(m, G_LEVELS[0])][0],
                     late_rows[(m, G_LEVELS[-1])][0]) for m in MODES]
    snap_ends = [(snap_rows[(m, G_LEVELS[0])][0],
                  snap_rows[(m, G_LEVELS[-1])][0]) for m in MODES]
    frags_bs = [_frag_b(m, *e) for m, e in zip(MODES, settled_ends)]
    frags_bn = [_frag_b(m, *e) for m, e in zip(MODES, snap_ends)]
    line_bs, line_bn = _line_b_settled(frags_bs), _line_b_snap(frags_bn)
    for m, frag, (v05, v20) in zip(MODES, frags_bs, settled_ends):
        row(panel="b", series="annotation", exit_mode=m, g=np.nan,
            landscape="pooled", step=np.nan, category="box_fragment_settled",
            value=float(v20 - v05), ci_lo=np.nan, ci_hi=np.nan, n=np.nan,
            stat="", annotation=frag, source=SRC_TRAJ)
    for m, frag, (v05, v20) in zip(MODES, frags_bn, snap_ends):
        row(panel="b", series="annotation", exit_mode=m, g=np.nan,
            landscape="pooled", step=np.nan, category="box_fragment_step400",
            value=float(v20 - v05), ci_lo=np.nan, ci_hi=np.nan, n=np.nan,
            stat="", annotation=frag, source=SRC_TRAJ)
    row(panel="b", series="annotation", exit_mode="", g=np.nan,
        landscape="", step=np.nan, category="box_line_settled", value=np.nan,
        ci_lo=np.nan, ci_hi=np.nan, n=np.nan, stat="", annotation=line_bs,
        source=SRC_TRAJ)
    row(panel="b", series="annotation", exit_mode="", g=np.nan,
        landscape="", step=np.nan, category="box_line_step400", value=np.nan,
        ci_lo=np.nan, ci_hi=np.nan, n=np.nan, stat="", annotation=line_bn,
        source=SRC_TRAJ)

    # (c) arrival-gap shrinkage
    pooled_vals = {}
    for m in MODES:
        for land in ["pooled"] + LANDSCAPES:
            vals = (cells.loc[cells["exit_mode"] == m,
                              "gap_recent_mean_last100"] if land == "pooled"
                    else cells.loc[(cells["exit_mode"] == m)
                                   & (cells["landscape"] == land),
                                   "gap_recent_mean_last100"])
            v, lo, hi, n = _mean_ci(vals)
            if land == "pooled":
                pooled_vals[m] = v
            row(panel="c", series="gap_recent_mean_last100", exit_mode=m,
                g=np.nan, landscape=land, step=np.nan, category="",
                value=v, ci_lo=lo, ci_hi=hi, n=n, stat=S_C, annotation="",
                source=SRC_CELLS)
    base = pooled_vals["none"]
    for m in MODES:
        v = pooled_vals[m]
        rel = 100.0 * (v / base - 1.0)
        row(panel="c", series="annotation", exit_mode=m, g=np.nan,
            landscape="pooled", step=np.nan, category="shrinkage_pct",
            value=rel, ci_lo=np.nan, ci_hi=np.nan, n=np.nan, stat="",
            annotation=(f"[{rel:.1f}%]" if m != "none" else ""),
            source=SRC_CELLS)
        row(panel="c", series="annotation", exit_mode=m, g=np.nan,
            landscape="pooled", step=np.nan, category="bar_label", value=v,
            ci_lo=np.nan, ci_hi=np.nan, n=np.nan, stat="",
            annotation=_label_c(v, rel, m), source=SRC_CELLS)
    neg = int((cells["gap_recent_mean_last100"] < 0).sum())
    tot = int(len(cells))
    row(panel="c", series="annotation", exit_mode="", g=np.nan,
        landscape="pooled", step=np.nan, category="negative_runs_note",
        value=float(neg), ci_lo=np.nan, ci_hi=np.nan, n=tot, stat="",
        annotation=_note_c(neg, tot), source=SRC_CELLS)

    # (d) output & concentration
    for m in MODES:
        sub = cells[cells["exit_mode"] == m]
        v, lo, hi, n = _mean_ci(sub["harvested_fin"])
        row(panel="d", series="harvested_fin", exit_mode=m, g=np.nan,
            landscape="pooled", step=np.nan, category="", value=v,
            ci_lo=lo, ci_hi=hi, n=n, stat=S_D["harvested_fin"],
            annotation="", source=SRC_CELLS)
        row(panel="d", series="annotation", exit_mode=m, g=np.nan,
            landscape="pooled", step=np.nan, category="bar_label",
            value=v, ci_lo=np.nan, ci_hi=np.nan, n=np.nan, stat="",
            annotation=_label_d("harvested_fin", v), source=SRC_CELLS)
        v, lo, hi, n = _mean_ci(sub["top10_fin"])
        row(panel="d", series="top10_step400", exit_mode=m, g=np.nan,
            landscape="pooled", step=BIRTH_SHOCK_STEPS[-1], category="",
            value=v, ci_lo=lo, ci_hi=hi, n=n, stat=S_D["top10_step400"],
            annotation="", source=SRC_CELLS)
        row(panel="d", series="annotation", exit_mode=m, g=np.nan,
            landscape="pooled", step=BIRTH_SHOCK_STEPS[-1],
            category="bar_label", value=v, ci_lo=np.nan, ci_hi=np.nan,
            n=np.nan, stat="", annotation=_label_d("top10_step400", v),
            source=SRC_CELLS)
        t399 = traj[(traj["exit_mode"] == m) & (traj["step"] == 399)]["top10"]
        v, lo, hi, n = _mean_ci(t399)
        row(panel="d", series="top10_step399", exit_mode=m, g=np.nan,
            landscape="pooled", step=399, category="", value=v,
            ci_lo=lo, ci_hi=hi, n=n, stat=S_D["top10_step399"],
            annotation="", source=SRC_TRAJ)
        row(panel="d", series="annotation", exit_mode=m, g=np.nan,
            landscape="pooled", step=399, category="bar_label", value=v,
            ci_lo=np.nan, ci_hi=np.nan, n=np.nan, stat="",
            annotation=_label_d("top10_step399", v), source=SRC_TRAJ)

    # reference constants
    for cat, val, step, ann in _REF_ROWS:
        row(panel="ref", series="reference", exit_mode="", g=np.nan,
            landscape="", step=step, category=cat, value=float(val),
            ci_lo=np.nan, ci_hi=np.nan, n=np.nan,
            stat="definitional constant (commission)", annotation=ann,
            source="commission_section_figures_v2.md")

    return pd.DataFrame(rows, columns=[
        "panel", "series", "exit_mode", "g", "landscape", "step", "category",
        "value", "ci_lo", "ci_hi", "n", "stat", "annotation", "source"])


# --- comparison ------------------------------------------------------------

_KEY = ["panel", "series", "exit_mode", "g", "landscape", "step", "category"]


def _key_str(df: pd.DataFrame) -> pd.Series:
    def one(r):
        def f(v):
            if isinstance(v, str):
                return v if v.strip() != "" else "-"
            if pd.isna(v):
                return "-"
            return f"{float(v):.12g}"
        return "|".join(f(r[c]) for c in _KEY)
    return df.apply(one, axis=1)


def check(outdir: str = "output/report_notebook_v1",
          tol: float = 1e-9) -> list[str]:
    fails: list[str] = []
    out = Path(outdir)
    if not out.is_absolute():
        out = REPO / out
    csv_path = out / f"{BASENAME}_data.csv"

    # --- expected input counts for this figure --------------------------
    traj_files = sorted((DATA / "traj").glob("cell_*.csv"))
    cells_files = sorted((DATA / "cells").glob("cell_*.csv"))
    if len(traj_files) != 54:
        fails.append(f"traj files: expected 54, found {len(traj_files)}")
    if len(cells_files) != 54:
        fails.append(f"cells files: expected 54, found {len(cells_files)}")
    _, _, traj, cells = _read_raw()
    if traj["cell_tag"].nunique() != 54:
        fails.append(f"traj cell tags: expected 54 unique, "
                     f"found {traj['cell_tag'].nunique()}")
    per_mode_cells = traj.groupby("exit_mode")["cell_tag"].nunique().to_dict()
    if per_mode_cells != {m: 18 for m in MODES}:
        fails.append(f"traj cells per mode: expected 18 each, got "
                     f"{per_mode_cells}")
    for f in traj_files:
        d = pd.read_csv(f)
        if len(d) != 400 or int(d["step"].min()) != 1 or int(d["step"].max()) != 400:
            fails.append(f"{f.name}: expected steps 1..400")
    if len(cells) != 1080:
        fails.append(f"runs: expected 1,080, found {len(cells)}")
    per_mode_runs = cells.groupby("exit_mode").size().to_dict()
    if per_mode_runs != {m: 360 for m in MODES}:
        fails.append(f"runs per mode: expected 360 each, got {per_mode_runs}")
    seeds_ok = all(pd.read_csv(f)["seed"].nunique() == 20 for f in cells_files)
    if not seeds_ok:
        fails.append("cells: some cell file lacks 20 unique seeds")

    # --- sidecar vs independent recomputation --------------------------
    if not csv_path.exists():
        fails.append(f"sidecar missing: {csv_path}")
        return fails
    got = pd.read_csv(csv_path)
    exp = _expected()

    got = got.copy()
    exp = exp.copy()
    got["_k"] = _key_str(got)
    exp["_k"] = _key_str(exp)
    if got["_k"].duplicated().any():
        fails.append(f"sidecar has duplicate keys: "
                     f"{got.loc[got['_k'].duplicated(), '_k'].head(3).tolist()}")
    if exp["_k"].duplicated().any():
        fails.append(f"internal: expected keys duplicated: "
                     f"{exp.loc[exp['_k'].duplicated(), '_k'].head(3).tolist()}")

    k_got, k_exp = set(got["_k"]), set(exp["_k"])
    missing = sorted(k_exp - k_got)[:5]
    extra = sorted(k_got - k_exp)[:5]
    if missing:
        fails.append(f"sidecar missing {len(k_exp - k_got)} recomputed row(s), "
                     f"e.g. {missing}")
    if extra:
        fails.append(f"sidecar has {len(k_got - k_exp)} unexpected row(s), "
                     f"e.g. {extra}")
    if fails:
        return fails

    merged = exp.merge(got, on="_k", suffixes=("_e", "_g"), validate="1:1")
    if len(merged) != len(exp):
        fails.append(f"row count mismatch: recomputed {len(exp)} vs sidecar "
                     f"{len(got)}")
        return fails

    for col in ["value", "ci_lo", "ci_hi", "n"]:
        e = merged[f"{col}_e"].to_numpy(float)
        g = merged[f"{col}_g"].to_numpy(float)
        both_nan = np.isnan(e) & np.isnan(g)
        one_nan = np.isnan(e) ^ np.isnan(g)
        if one_nan.any():
            i = int(np.where(one_nan)[0][0])
            fails.append(f"{col}: NaN mismatch at key "
                         f"{merged['_k'].iloc[i]}: expected {e[i]}, got {g[i]}")
            continue
        diff = np.abs(e - g)
        diff[both_nan] = 0.0
        if not (diff <= tol).all():
            i = int(np.argmax(diff))
            fails.append(f"{col}: max |diff| = {diff[i]:.3e} > tol {tol} at "
                         f"key {merged['_k'].iloc[i]}: expected {e[i]}, "
                         f"got {g[i]}")
    for col in ["stat", "annotation"]:
        e = merged[f"{col}_e"].fillna("").astype(str)
        g = merged[f"{col}_g"].fillna("").astype(str)
        bad = e != g
        if bad.any():
            i = int(np.where(bad.to_numpy())[0][0])
            fails.append(f"{col}: mismatch at key {merged['_k'].iloc[i]}: "
                         f"expected {e.iloc[i]!r}, got {g.iloc[i]!r}")
    return fails


def main() -> int:
    fails = check()
    if fails:
        for f in fails:
            print(f"FAIL {f}")
        return 1
    print(f"verify_figA: all checks passed "
          f"({len(_expected())} sidecar rows reproduced from raw CSVs "
          f"within 1e-9; 1,080 runs / 54 traj cells / 54 cells asserted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
