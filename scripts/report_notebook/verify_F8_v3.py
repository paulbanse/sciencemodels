"""Independent verifier for ``F8_v3_turnover_diagnostics``.

``check()`` recomputes EVERY plotted quantity and rendered annotation from the
RAW source CSVs -- ``data/turnover_v2/{runs.csv, traj/cell_*.csv}`` plus the
read-only pre-fix ``data/turnover_v1/traj/cell_*.csv`` for the ghost inset --
and compares the result against ``F8_v3_turnover_diagnostics_data.csv`` at a
configurable tolerance (default 1e-9).  It also asserts the expected input
counts:

  * 76 post-fix trajectory files (400 steps each) and 1,520 runs, restricted
    to the common-support curiosity {0.3, 0.5, 0.7} => 72 cells / 1,440 runs
    (18 cells and 360 runs per exit mode);
  * 54 pre-fix trajectory files (18 cells per legacy exit mode);
  * the structural claims the figure is built on: the pre-fix birth-step spike
    at t = 400 is negative and vanishes post-fix, ``gap_recent_mean_last100``
    is negative in every common-support run, and the synchronized step-400
    top-10% share exceeds the staggered one.

The recomputation below is deliberately self-contained (no import of
``figA_v3_turnover_postbug`` for the numbers); the annotation format strings
are part of the figure's specification and are re-rendered here from the raw
numbers.  The hardened visual QA is re-run by building the figure into a
temporary directory (``visual_qa.assert_figure_clean`` runs inside ``build``).

Run:  MPLBACKEND=Agg python3 scripts/report_notebook/verify_F8_v3.py
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

POST_ROOT = REPO / "data" / "turnover_v2"
PRE_ROOT = REPO / "data" / "turnover_v1"
BASENAME = "F8_v3_turnover_diagnostics"
FNAME = BASENAME

MODES = ["none", "fixed_span", "age_hazard_sync", "age_hazard_staggered"]
LANDSCAPES = ["mgstd5", "noisypr5"]
G_LEVELS = [0.05, 0.10, 0.20]
COMMON_CURIOSITY = (0.3, 0.5, 0.7)
PRE_MODE = "age_hazard"
GHOST_MODE = "age_hazard_sync"

DISCOVERY_END = 80
DEPLETION_START = 150
BIRTH_SHOCK_STEPS = (200, 400)
LATE_START, LATE_END = 301, 400
GHOST_START = DEPLETION_START - 5
EQUALITY_TOP10 = 0.10
Z95 = 1.96

_TRAJ_RE = re.compile(r"cell_([a-z_]+?)_g(\d+)_cur(\d+)_(mgstd5|noisypr5)\.csv")
POST_TRAJ_GLOB = "data/turnover_v2/traj/cell_*.csv"
POST_RUNS = "data/turnover_v2/runs.csv"
PRE_TRAJ_GLOB = "data/turnover_v1/traj/cell_*.csv"

M_A = "c_ideal (per-step mean across the 18 cells of the mode)"
M_GHOST = "c_ideal (age_hazard sync: early-draft ghost vs corrected)"
M_B = "c_ideal (settled mean over steps 301-400)"
M_C = "gap_recent_mean_last100 (pooled bar and per-landscape markers)"
M_D = "native share (harvested_fin, top10 at step 399, top10_fin at step 400)"
M_REF = "reference (definitional constants)"

_KEYS = ["panel", "metric", "statistic", "series", "exit_mode", "g",
         "landscape", "step"]
_COLS = ["figure", "panel", "metric", "statistic", "series", "exit_mode", "g",
         "landscape", "step", "value", "ci_lo", "ci_hi", "n", "source_file",
         "note"]


# --- raw inputs ------------------------------------------------------------

def _read_traj(root: Path) -> pd.DataFrame:
    frames = []
    for f in sorted((root / "traj").glob("cell_*.csv")):
        m = _TRAJ_RE.match(f.name)
        if m is None:
            raise AssertionError(f"unparsable trajectory filename {f.name}")
        d = pd.read_csv(f)
        d["exit_mode"] = m.group(1)
        d["g"] = int(m.group(2)) / 100
        d["curiosity"] = int(m.group(3)) / 10
        d["landscape"] = m.group(4)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def _post():
    traj_all = _read_traj(POST_ROOT)
    runs_all = pd.read_csv(POST_ROOT / "runs.csv")
    traj = traj_all[traj_all["curiosity"].isin(COMMON_CURIOSITY)].copy()
    runs = runs_all[runs_all["curiosity"].isin(COMMON_CURIOSITY)].copy()
    return traj_all, runs_all, traj, runs


def _pre() -> pd.DataFrame:
    return _read_traj(PRE_ROOT)


def _mean_ci(vals) -> tuple[float, float, float, int]:
    v = pd.Series(vals, dtype=float).dropna()
    n = int(v.size)
    m = float(v.mean())
    ci = Z95 * float(v.std(ddof=1)) / np.sqrt(n) if n > 1 else float("nan")
    return m, m - ci, m + ci, n


# --- annotation spec (mirrored from the builder) ---------------------------

def fmt_pct(rel: float) -> str:
    return f"{rel:+.1f}%"


def label_gap(v: float, rel: float) -> str:
    return f"{v:.5f}\n[{fmt_pct(rel)}]"


def label_share(series: str, v: float) -> str:
    return f"{v:.3f}"


def note_regimes(disc: float, depl: float, late: float) -> str:
    return (f"pooled c_ideal:Discovery {disc:+.4f} → Depletion {depl:+.4f}"
            f" (301–400 {late:+.4f})")


def note_negative(neg: int, tot: int) -> str:
    return f"negative in {neg:,}/{tot:,} runs"


def note_spike(spike: float, staggered: float) -> str:
    return (f"step-400 top-10%: sync {spike:.3f} → staggered {staggered:.3f}")


def label_ghost(pre400: float, post400: float) -> str:
    return f"t=400: early-draft {pre400:+.3f} → corrected {post400:+.3f}"


# --- expected sidecar (recomputed from the raw CSVs) -----------------------

def _expected() -> pd.DataFrame:
    _, _, traj, runs = _post()
    pre = _pre()
    rows: list[dict] = []

    def add(panel, metric, series, statistic, value, ci_lo, ci_hi, n,
            exit_mode="", g=np.nan, landscape="pooled", step=np.nan,
            source_file="", note=""):
        rows.append(dict(figure=FNAME, panel=panel, metric=metric,
                         statistic=statistic, series=series,
                         exit_mode=exit_mode, g=g, landscape=landscape,
                         step=step, value=float(value), ci_lo=float(ci_lo),
                         ci_hi=float(ci_hi), n=n, source_file=source_file,
                         note=note))

    # (a) per-step trajectory
    for (mode, step), grp in traj.groupby(["exit_mode", "step"]):
        v, lo, hi, n = _mean_ci(grp["c_ideal"])
        add("a", M_A, "trajectory", "mean", v, lo, hi, n, exit_mode=mode,
            step=int(step), source_file=POST_TRAJ_GLOB,
            note="per-step mean of c_ideal across the 18 cells of the mode")

    disc = float(traj.loc[traj["step"] <= DISCOVERY_END, "c_ideal"].mean())
    depl = float(traj.loc[traj["step"] >= DEPLETION_START, "c_ideal"].mean())
    late = float(traj.loc[traj["step"] >= LATE_START, "c_ideal"].mean())
    text_a = note_regimes(disc, depl, late)
    for key, val in (("discovery_mean", disc), ("depletion_mean", depl),
                     ("late_mean", late)):
        add("a", f"annotation: {key}", "annotation", "annotation", val,
            np.nan, np.nan, int(len(traj)), source_file=POST_TRAJ_GLOB,
            note=text_a if key == "discovery_mean" else "")

    # (a) inset ghost
    pre_ghost = pre[pre["exit_mode"] == PRE_MODE].groupby("step")["c_ideal"].mean()
    post_ghost = (traj[traj["exit_mode"] == GHOST_MODE]
                  .groupby("step")["c_ideal"].mean())
    for step in range(GHOST_START, 401):
        add("a_inset", M_GHOST, "pre_fix_ghost", "mean", pre_ghost.loc[step],
            np.nan, np.nan, 18, exit_mode=GHOST_MODE, step=step,
            source_file=PRE_TRAJ_GLOB,
            note="mean across the 18 cells; early-draft = pre-correction measurement")
        add("a_inset", M_GHOST, "post_fix", "mean", post_ghost.loc[step],
            np.nan, np.nan, 18, exit_mode=GHOST_MODE, step=step,
            source_file=POST_TRAJ_GLOB,
            note="mean across the 18 cells; early-draft = pre-correction measurement")
    pre400 = float(pre_ghost.loc[400])
    post400 = float(post_ghost.loc[400])
    add("a_inset", "annotation: pre_fix_step400", "annotation", "annotation",
        pre400, np.nan, np.nan, 18, exit_mode=GHOST_MODE, step=400,
        source_file=PRE_TRAJ_GLOB, note=label_ghost(pre400, post400))
    add("a_inset", "annotation: post_fix_step400", "annotation", "annotation",
        post400, np.nan, np.nan, 18, exit_mode=GHOST_MODE, step=400,
        source_file=POST_TRAJ_GLOB, note=label_ghost(pre400, post400))

    # (b) settled alignment
    late_traj = traj[(traj["step"] >= LATE_START) & (traj["step"] <= LATE_END)]
    cell_means = (late_traj.groupby(["exit_mode", "g", "cell_tag"])["c_ideal"]
                  .mean().reset_index())
    b = {}
    for (mode, g), grp in cell_means.groupby(["exit_mode", "g"]):
        v, lo, hi, n = _mean_ci(grp["c_ideal"])
        b[(mode, float(g))] = (v, lo, hi, n)
        add("b", M_B, "settled", "mean", v, lo, hi, n, exit_mode=mode,
            g=float(g), source_file=POST_TRAJ_GLOB,
            note=("mean over the 6 cells of their steps-301-400 means; "
                  "95% CI across the 6 cell means"))
    for m in MODES:
        delta = b[(m, G_LEVELS[-1])][0] - b[(m, G_LEVELS[0])][0]
        add("b", "annotation: settled_change_g005_to_g020", "annotation",
            "annotation", delta, np.nan, np.nan, 6, exit_mode=m,
            source_file=POST_TRAJ_GLOB,
            note=f"{m}: settled change g 0.05 -> 0.20")

    # (c) gap shrinkage
    pooled = {}
    for m in MODES:
        sub = runs[runs["exit_mode"] == m]
        v, lo, hi, n = _mean_ci(sub["gap_recent_mean_last100"])
        pooled[m] = v
        add("c", M_C, "pooled", "mean", v, lo, hi, n, exit_mode=m,
            source_file=POST_RUNS,
            note="mean of gap_recent_mean_last100 over the 360 runs; 95% CI = "
                 "mean +/- 1.96*SE over runs")
        for land in LANDSCAPES:
            vv, llo, hhi, nn = _mean_ci(
                sub.loc[sub["landscape"] == land, "gap_recent_mean_last100"])
            add("c", M_C, land, "mean", vv, llo, hhi, nn, exit_mode=m,
                landscape=land, source_file=POST_RUNS,
                note=f"mean over the {nn} {land} runs")
    base = pooled["none"]
    for m in MODES:
        rel = 100.0 * (pooled[m] / base - 1.0)
        add("c", "annotation: shrinkage_pct_vs_none", "annotation",
            "annotation", rel, np.nan, np.nan, 360, exit_mode=m,
            source_file=POST_RUNS,
            note=f"{m}: gap_recent change vs none = {fmt_pct(rel)}")
    neg = int((runs["gap_recent_mean_last100"] < 0).sum())
    tot = int(len(runs))
    add("c", "annotation: negative_runs_note", "annotation", "annotation",
        float(neg), np.nan, np.nan, tot, source_file=POST_RUNS,
        note=note_negative(neg, tot))

    # (d) output & concentration
    d_vals = {}
    for m in MODES:
        sub = runs[runs["exit_mode"] == m]
        v, lo, hi, n = _mean_ci(sub["harvested_fin"])
        d_vals[(m, "harvested_fin")] = v
        add("d", M_D, "harvested_fin", "mean", v, lo, hi, n, exit_mode=m,
            source_file=POST_RUNS,
            note="mean of harvested_fin over the 360 runs")
        v, lo, hi, n = _mean_ci(sub["top10_fin"])
        d_vals[(m, "top10_step400")] = v
        add("d", M_D, "top10_step400", "mean", v, lo, hi, n, exit_mode=m,
            step=BIRTH_SHOCK_STEPS[-1], source_file=POST_RUNS,
            note="mean of top10_fin (terminal step 400) over the 360 runs")
        t399 = traj[(traj["exit_mode"] == m) & (traj["step"] == 399)]["top10"]
        v, lo, hi, n = _mean_ci(t399)
        add("d", M_D, "top10_step399", "mean", v, lo, hi, n, exit_mode=m,
            step=399, source_file=POST_TRAJ_GLOB,
            note="mean of the trajectory top10 share at step 399 over the 18 "
                 "cells; 95% CI across cells")
    spike = d_vals[("age_hazard_sync", "top10_step400")]
    stag = d_vals[("age_hazard_staggered", "top10_step400")]
    text_d = note_spike(spike, stag)
    for m in MODES:
        add("d", "annotation: top10_step400", "annotation", "annotation",
            d_vals[(m, "top10_step400")], np.nan, np.nan, 360, exit_mode=m,
            step=BIRTH_SHOCK_STEPS[-1], source_file=POST_RUNS,
            note=text_d if m == "age_hazard_sync" else "")

    # reference rows
    for cat, val, step, ann in [
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
        ("equality_top10", EQUALITY_TOP10, np.nan, f"{EQUALITY_TOP10:.2f} equality"),
        ("z95", Z95, np.nan, "95% CI multiplier (house convention)"),
        ("common_support_curiosity_min", COMMON_CURIOSITY[0], np.nan,
         "common-support curiosity floor (all four modes)"),
        ("common_support_curiosity_max", COMMON_CURIOSITY[-1], np.nan,
         "common-support curiosity ceiling (all four modes)"),
    ]:
        rows.append(dict(figure=FNAME, panel="ref", metric=M_REF,
                         statistic="definitional constant", series=cat,
                         exit_mode="", g=np.nan, landscape="", step=step,
                         value=float(val), ci_lo=np.nan, ci_hi=np.nan, n=np.nan,
                         source_file="commission (team2_figure_repair_v3.md)",
                         note=ann))

    return pd.DataFrame(rows, columns=_COLS)


# ---------------------------------------------------------------------------
# extra rendered-artist guard: lines vs text, in DISPLAY coordinates
# ---------------------------------------------------------------------------
def _line_text_violations(fig, tol: float = 0.75) -> list[str]:
    """No visible text may be crossed by a drawn line.

    ``visual_qa`` compares ``Line2D`` vertices (data units) against text
    bounding boxes (display units), so a line can run straight through a
    label without being reported.  This guard transforms the vertices to
    display coordinates first and additionally covers the inset axes.
    """
    out: list[str] = []
    fig.canvas.draw()
    try:
        renderer = fig.canvas.get_renderer()
    except AttributeError:  # a closed figure falls back to the base canvas
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        FigureCanvasAgg(fig)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    axes = list(fig.axes) + [c for a in fig.axes for c in a.child_axes]
    for ax in axes:
        ab = ax.bbox
        texts = [ax.title, ax.xaxis.label, ax.yaxis.label]
        texts += [t for t in ax.texts if t.get_visible()]
        leg = ax.get_legend()
        if leg is not None:
            texts += list(leg.get_texts())
        for t in texts:
            if not t.get_text().strip():
                continue
            patch = t.get_bbox_patch()
            if patch is not None:  # an opaque backing box occludes the line
                fc = patch.get_facecolor()
                eff = (fc[3] if len(fc) == 4 else 1.0) * (
                    patch.get_alpha() if patch.get_alpha() is not None else 1.0)
                if eff >= 0.8:
                    continue
            bb = t.get_window_extent(renderer=renderer)
            if bb.width <= 0 or bb.height <= 0:
                continue
            if bb.x1 < ab.x0 or bb.x0 > ab.x1 or bb.y1 < ab.y0 or bb.y0 > ab.y1:
                continue
            for line in ax.get_lines():
                xy = line.get_xydata()
                if len(xy) == 0:
                    continue
                disp = ax.transData.transform(np.asarray(xy, dtype=float))
                pts = disp if len(disp) > 1 else np.vstack([disp, disp])
                for i in range(len(pts) - 1):
                    (x0, y0), (x1, y1) = pts[i], pts[i + 1]
                    if (min(x0, x1) > bb.x1 or max(x0, x1) < bb.x0
                            or min(y0, y1) > bb.y1 or max(y0, y1) < bb.y0):
                        continue
                    for f in np.linspace(0.0, 1.0, 24):
                        x, y = x0 + f * (x1 - x0), y0 + f * (y1 - y0)
                        if (bb.x0 + tol < x < bb.x1 - tol
                                and bb.y0 + tol < y < bb.y1 - tol):
                            msg = (f"text {t.get_text()[:34]!r} crossed by line "
                                   f"{line.get_label()[:28]!r}")
                            if msg not in out:
                                out.append(msg)
                            break
    return out


# ---------------------------------------------------------------------------
# extra rendered-artist guard: lines vs text, in DISPLAY coordinates
# ---------------------------------------------------------------------------
def _line_text_violations(fig, tol: float = 0.75) -> list[str]:
    """No visible text may be crossed by a drawn line.

    ``visual_qa`` compares ``Line2D`` vertices (data units) against text
    bounding boxes (display units), so a line can run straight through a
    label without being reported.  This guard transforms the vertices to
    display coordinates first and additionally covers the inset axes.
    """
    out: list[str] = []
    fig.canvas.draw()
    try:
        renderer = fig.canvas.get_renderer()
    except AttributeError:  # a closed figure falls back to the base canvas
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        FigureCanvasAgg(fig)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    axes = list(fig.axes) + [c for a in fig.axes for c in a.child_axes]
    for ax in axes:
        ab = ax.bbox
        texts = [ax.title, ax.xaxis.label, ax.yaxis.label]
        texts += [t for t in ax.texts if t.get_visible()]
        leg = ax.get_legend()
        if leg is not None:
            texts += list(leg.get_texts())
        for t in texts:
            if not t.get_text().strip():
                continue
            patch = t.get_bbox_patch()
            if patch is not None:  # an opaque backing box occludes the line
                fc = patch.get_facecolor()
                eff = (fc[3] if len(fc) == 4 else 1.0) * (
                    patch.get_alpha() if patch.get_alpha() is not None else 1.0)
                if eff >= 0.8:
                    continue
            bb = t.get_window_extent(renderer=renderer)
            if bb.width <= 0 or bb.height <= 0:
                continue
            if bb.x1 < ab.x0 or bb.x0 > ab.x1 or bb.y1 < ab.y0 or bb.y0 > ab.y1:
                continue
            for line in ax.get_lines():
                xy = line.get_xydata()
                if len(xy) == 0:
                    continue
                disp = ax.transData.transform(np.asarray(xy, dtype=float))
                pts = disp if len(disp) > 1 else np.vstack([disp, disp])
                for i in range(len(pts) - 1):
                    (x0, y0), (x1, y1) = pts[i], pts[i + 1]
                    if (min(x0, x1) > bb.x1 or max(x0, x1) < bb.x0
                            or min(y0, y1) > bb.y1 or max(y0, y1) < bb.y0):
                        continue
                    for f in np.linspace(0.0, 1.0, 24):
                        x, y = x0 + f * (x1 - x0), y0 + f * (y1 - y0)
                        if (bb.x0 + tol < x < bb.x1 - tol
                                and bb.y0 + tol < y < bb.y1 - tol):
                            msg = (f"text {t.get_text()[:34]!r} crossed by line "
                                   f"{line.get_label()[:28]!r}")
                            if msg not in out:
                                out.append(msg)
                            break
    return out


# --- comparison ------------------------------------------------------------

def _key_str(df: pd.DataFrame) -> pd.Series:
    def one(r):
        def f(v):
            if isinstance(v, str):
                return v if v.strip() != "" else "-"
            if pd.isna(v):
                return "-"
            return f"{float(v):.12g}"
        return "|".join(f(r[c]) for c in _KEYS)
    return df.apply(one, axis=1)


def check(outdir="output/report_notebook_v1",
          data_root=None, tol: float = 1e-9) -> list[str]:
    fails: list[str] = []
    out = Path(outdir)
    if not out.is_absolute():
        out = REPO / out
    csv_path = out / f"{BASENAME}_data.csv"

    root = Path(data_root) if data_root is not None else POST_ROOT

    # --- input counts ---------------------------------------------------
    post_files = sorted((root / "traj").glob("cell_*.csv"))
    if len(post_files) != 76:
        fails.append(f"post traj files: expected 76, found {len(post_files)}")
    if not (root / "runs.csv").exists():
        fails.append(f"missing {root / 'runs.csv'}")
        return fails
    traj_all, runs_all, traj, runs = _post()
    if len(runs_all) != 1520:
        fails.append(f"runs.csv: expected 1,520 rows, found {len(runs_all)}")
    if len(runs) != 1440:
        fails.append(f"common-support runs: expected 1,440, found {len(runs)}")
    per_mode_runs = runs.groupby("exit_mode").size().to_dict()
    if per_mode_runs != {m: 360 for m in MODES}:
        fails.append(f"runs per mode: expected 360 each, got {per_mode_runs}")
    per_mode_cells = traj.groupby("exit_mode")["cell_tag"].nunique().to_dict()
    if per_mode_cells != {m: 18 for m in MODES}:
        fails.append(f"cells per mode: expected 18 each, got {per_mode_cells}")
    for f in post_files:
        d = pd.read_csv(f)
        if len(d) != 400 or int(d["step"].min()) != 1 or int(d["step"].max()) != 400:
            fails.append(f"{f.name}: expected steps 1..400")
            break
    pre_files = sorted((PRE_ROOT / "traj").glob("cell_*.csv"))
    if len(pre_files) != 54:
        fails.append(f"pre-fix traj files: expected 54, found {len(pre_files)}")

    # --- structural claims the figure asserts ---------------------------
    pre = _pre()
    pre_ghost = pre[pre["exit_mode"] == PRE_MODE].groupby("step")["c_ideal"].mean()
    post_ghost = (traj[traj["exit_mode"] == GHOST_MODE]
                  .groupby("step")["c_ideal"].mean())
    if not (pre_ghost.loc[400] < 0.0 < post_ghost.loc[400]):
        fails.append(f"inset claim broken: pre-fix t=400 {pre_ghost.loc[400]:.6f} "
                     f"must be negative and post-fix {post_ghost.loc[400]:.6f} "
                     f"positive")
    if int((runs["gap_recent_mean_last100"] < 0).sum()) != len(runs):
        fails.append("gap_recent claim broken: not every common-support run is "
                     "negative")
    sync400 = float(runs.loc[(runs["exit_mode"] == "age_hazard_sync"),
                             "top10_fin"].mean())
    stag400 = float(runs.loc[(runs["exit_mode"] == "age_hazard_staggered"),
                             "top10_fin"].mean())
    if not sync400 > stag400:
        fails.append(f"panel (d) claim broken: sync step-400 top-10% "
                     f"{sync400:.6f} <= staggered {stag400:.6f}")

    # --- sidecar vs independent recomputation ---------------------------
    if not csv_path.exists():
        fails.append(f"sidecar missing: {csv_path}")
        return fails
    got = pd.read_csv(csv_path)
    exp = _expected()
    if list(got.columns) != _COLS:
        fails.append(f"sidecar columns {list(got.columns)} != {_COLS}")

    got = got.copy()
    exp = exp.copy()
    got["_k"] = _key_str(got)
    exp["_k"] = _key_str(exp)
    if got["_k"].duplicated().any():
        fails.append(f"sidecar duplicate keys: "
                     f"{got.loc[got['_k'].duplicated(), '_k'].head(3).tolist()}")
    if exp["_k"].duplicated().any():
        fails.append(f"internal: expected keys duplicated: "
                     f"{exp.loc[exp['_k'].duplicated(), '_k'].head(3).tolist()}")
    missing = sorted(set(exp["_k"]) - set(got["_k"]))[:5]
    extra = sorted(set(got["_k"]) - set(exp["_k"]))[:5]
    if missing:
        fails.append(f"sidecar missing {len(set(exp['_k']) - set(got['_k']))} "
                     f"recomputed row(s), e.g. {missing}")
    if extra:
        fails.append(f"sidecar has {len(set(got['_k']) - set(exp['_k']))} "
                     f"unexpected row(s), e.g. {extra}")
    if fails:
        return fails

    merged = exp.merge(got, on="_k", suffixes=("_e", "_g"), validate="1:1")
    for col in ["value", "ci_lo", "ci_hi", "n"]:
        e = merged[f"{col}_e"].to_numpy(float)
        g = merged[f"{col}_g"].to_numpy(float)
        both_nan = np.isnan(e) & np.isnan(g)
        one_nan = np.isnan(e) ^ np.isnan(g)
        if one_nan.any():
            i = int(np.where(one_nan)[0][0])
            fails.append(f"{col}: NaN mismatch at {merged['_k'].iloc[i]}: "
                         f"expected {e[i]}, got {g[i]}")
            continue
        diff = np.abs(e - g)
        diff[both_nan] = 0.0
        if not (diff <= tol).all():
            i = int(np.argmax(diff))
            fails.append(f"{col}: max |diff| {diff[i]:.3e} > tol {tol} at "
                         f"{merged['_k'].iloc[i]}: expected {e[i]}, got {g[i]}")
    for col in ["statistic", "note", "source_file"]:
        e = merged[f"{col}_e"].fillna("").astype(str)
        g = merged[f"{col}_g"].fillna("").astype(str)
        bad = e != g
        if bad.any():
            i = int(np.where(bad.to_numpy())[0][0])
            fails.append(f"{col}: mismatch at {merged['_k'].iloc[i]}: "
                         f"expected {e.iloc[i]!r}, got {g.iloc[i]!r}")

    # --- hardened visual QA re-run (build into a temp directory) --------
    import figA_v3_turnover_postbug as F  # builder, for the QA contract only
    tmp = Path(tempfile.mkdtemp(prefix="f8v3_"))
    col: dict = {}
    res = F.build(tmp, data_root=data_root, collector=col)
    if res["qa"]["_total"] != 0:
        fails.append(f"visual_qa re-run reported {res['qa']['_total']} failure(s)")
    fails.extend(_line_text_violations(col["fig"]))
    regot = pd.read_csv(Path(res["data_csv"]))
    if len(regot) != len(exp):
        fails.append(f"regenerated sidecar has {len(regot)} rows, expected {len(exp)}")
    else:
        regot = regot.copy()
        regot["_k"] = _key_str(regot)
        m2 = exp.merge(regot, on="_k", suffixes=("_e", "_r"), validate="1:1")
        d = np.abs(m2["value_e"].to_numpy(float) - m2["value_r"].to_numpy(float))
        d[np.isnan(d)] = 0.0
        if not (d <= tol).all():
            fails.append("regenerated sidecar values differ from the recomputation")
    return fails


def main() -> int:
    fails = check()
    if fails:
        for f in fails:
            print(f"FAIL {f}")
        return 1
    exp = _expected()
    print(f"verify_F8_v3: all checks passed ({len(exp)} sidecar rows reproduced "
          f"from raw CSVs within 1e-9; 1,440 runs / 72 cells / 54 pre-fix cells "
          f"asserted; visual_qa re-run clean)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
