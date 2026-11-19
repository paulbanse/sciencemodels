"""Independent verifier for ``F1_v2_t1_postbug``.

``check()`` recomputes every plotted quantity and rendered annotation from the
RAW ``data/bug0006_remeasure_v2/t1_runs.csv`` (720 post-fix runs) and compares
the result against ``F1_v2_t1_postbug_data.csv`` at a configurable tolerance
(default 1e-9).  Input contracts asserted here: 720 runs, entry grid
{0.0, 0.1}, curiosity x vanishing x landscape grids, 20 seeds per cell, 36
cells; and the two structural claims of the figure -- the M0/M1 time-mean
alignment attenuation exceeds 1 (entry collapses alignment) and the noisy
landscape carries more negative terminal cells than the clean one.

The recomputation is deliberately self-contained (the numbers do not come from
``fig_F1_v2_t1_postbug``); the hardened visual QA is re-run by building the
figure into a temporary directory, where
``visual_qa.assert_figure_clean`` runs inside ``build``.

Run:  MPLBACKEND=Agg python3 scripts/report_notebook/verify_F1_v2.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

DATA_ROOT = REPO / "data" / "bug0006_remeasure_v2"
BASENAME = "F1_v2_t1_postbug"
FNAME = BASENAME

G0, G1 = 0.0, 0.1
LANDSCAPES = ["mgstd5", "noisypr5"]
CURIOSITY = [0.3, 0.5, 0.7]
VANISHING = [0.5, 0.9, 1.0]
Z95 = 1.96

M_A = "c_ideal_tm (seed mean per landscape x entry rate x curiosity)"
M_B = "c_ideal_fin (pooled bar over runs, per-cell dots)"
M_REF = "reference (definitional constants)"

_KEYS = ["panel", "metric", "statistic", "series", "landscape", "curiosity",
         "g", "vanishing"]
_COLS = ["figure", "panel", "metric", "statistic", "series", "landscape",
         "curiosity", "g", "vanishing", "value", "ci_lo", "ci_hi", "n",
         "source_file", "note"]
SRC = "data/bug0006_remeasure_v2/t1_runs.csv"


def _mean_ci(vals) -> tuple[float, float, float, int]:
    v = pd.Series(vals, dtype=float).dropna()
    n = int(v.size)
    m = float(v.mean())
    ci = Z95 * float(v.std(ddof=1)) / np.sqrt(n) if n > 1 else float("nan")
    return m, m - ci, m + ci, n


def label_attenuation(pooled: float, mgstd5: float, noisypr5: float) -> str:
    return (f"M0/M1 c_ideal_tm attenuation: pooled {pooled:.2f}× · "
            f"mgstd5 {mgstd5:.2f}× · noisypr5 {noisypr5:.2f}×")


def label_cell(v: float, neg: int, cells: int) -> str:
    return f"{v:+.3f}\n{neg}/{cells} < 0"


def _runs(root: Path = DATA_ROOT) -> pd.DataFrame:
    return pd.read_csv(root / "t1_runs.csv")


def _expected(root: Path = DATA_ROOT) -> pd.DataFrame:
    t = _runs(root)
    rows: list[dict] = []

    def add(panel, metric, series, statistic, value, ci_lo, ci_hi, n,
            landscape="pooled", curiosity=np.nan, g=np.nan, vanishing=np.nan,
            source_file=SRC, note=""):
        rows.append(dict(figure=FNAME, panel=panel, metric=metric,
                         statistic=statistic, series=series,
                         landscape=landscape, curiosity=curiosity, g=g,
                         vanishing=vanishing, value=float(value),
                         ci_lo=float(ci_lo), ci_hi=float(ci_hi), n=int(n),
                         source_file=source_file, note=note))

    # (a) c_ideal_tm per (landscape, g, curiosity)
    for (land, g, cur), grp in t.groupby(["landscape",
                                          "agent_generation_time",
                                          "curiosity"]):
        v, lo, hi, n = _mean_ci(grp["c_ideal_tm"])
        add("a", M_A, f"{land}_{'M1' if g else 'M0'}", "mean", v, lo, hi, n,
            landscape=land, curiosity=float(cur), g=float(g),
            note="seed mean of c_ideal_tm; 95% CI = mean +/- 1.96*SE over runs")

    for land in ["pooled"] + LANDSCAPES:
        sub = t if land == "pooled" else t[t["landscape"] == land]
        m0 = float(sub.loc[sub["agent_generation_time"] == G0, "c_ideal_tm"].mean())
        m1 = float(sub.loc[sub["agent_generation_time"] == G1, "c_ideal_tm"].mean())
        add("a", f"annotation: attenuation_{land}", "annotation", "annotation",
            m0 / m1, np.nan, np.nan,
            int(sub.loc[sub["agent_generation_time"] == G0].shape[0]),
            landscape=land,
            note=f"{land}: M0 c_ideal_tm {m0:.6f} / M1 {m1:.6f}")
    ratios = {}
    for land in ["pooled"] + LANDSCAPES:
        sub = t if land == "pooled" else t[t["landscape"] == land]
        ratios[land] = (float(sub.loc[sub["agent_generation_time"] == G0,
                                    "c_ideal_tm"].mean())
                        / float(sub.loc[sub["agent_generation_time"] == G1,
                                      "c_ideal_tm"].mean()))
    add("a", "annotation: attenuation_line", "annotation", "annotation",
        ratios["pooled"], np.nan, np.nan, 360,
        note=label_attenuation(ratios["pooled"], ratios["mgstd5"],
                               ratios["noisypr5"]))

    # (b) c_ideal_fin per cell, pooled bars + cell dots
    for land in LANDSCAPES:
        for g in (G0, G1):
            sub = t[(t["landscape"] == land)
                    & (np.isclose(t["agent_generation_time"], g))]
            cell = (sub.groupby(["curiosity", "vanishing"])["c_ideal_fin"]
                    .agg(mean="mean", n="size").reset_index())
            pooled, lo, hi, n_cells = _mean_ci(cell["mean"])
            neg = int((cell["mean"] < 0).sum())
            series = f"{land}_{'M1' if g else 'M0'}"
            add("b", M_B, series, "mean", pooled, lo, hi, n_cells,
                landscape=land, g=g,
                note="pooled mean of the 9 cell means; 95% CI = mean +/- "
                     "1.96*SE across the 9 cell means")
            for r in cell.itertuples():
                add("b", M_B, series, "cell_mean", r.mean, r.mean, r.mean,
                    int(r.n), landscape=land, g=g,
                    curiosity=float(r.curiosity),
                    vanishing=float(r.vanishing),
                    note="mean of c_ideal_fin over the 20 seeds of the cell")
            add("b", f"annotation: negative_cells_{series}", "annotation",
                "annotation", float(neg), np.nan, np.nan, n_cells,
                landscape=land, g=g, note=label_cell(pooled, neg, n_cells))

    for cat, val, ann in [
        ("entry_off", G0, "M0: no entry (g = 0)"),
        ("entry_on", G1, "M1: entry on (g = 0.1)"),
        ("cells_per_group", 9, "cells = curiosity x vanishing"),
        ("z95", Z95, "95% CI multiplier (house convention)"),
    ]:
        rows.append(dict(figure=FNAME, panel="ref", metric=M_REF,
                         statistic="definitional constant", series=cat,
                         landscape="", curiosity=np.nan, g=np.nan,
                         vanishing=np.nan, value=float(val), ci_lo=np.nan,
                         ci_hi=np.nan, n=np.nan,
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
    root = Path(data_root) if data_root is not None else DATA_ROOT

    t = _runs(root)
    # --- input contracts -------------------------------------------------
    if len(t) != 720:
        fails.append(f"t1_runs: expected 720 rows, found {len(t)}")
        return fails
    if sorted(t["agent_generation_time"].unique().tolist()) != [G0, G1]:
        fails.append("t1_runs: entry grid must be {0.0, 0.1}")
    for col, grid in (("curiosity", CURIOSITY), ("vanishing", VANISHING),
                      ("landscape", LANDSCAPES)):
        if sorted(t[col].unique().tolist()) != grid:
            fails.append(f"t1_runs: {col} grid must be {grid}")
    per_cell = t.groupby(["landscape", "agent_generation_time", "curiosity",
                          "vanishing"]).size()
    if set(per_cell.unique()) != {20}:
        fails.append(f"t1_runs: 20 seeds per cell expected, got "
                     f"{sorted(set(per_cell.unique()))}")
    if len(per_cell) != 36:
        fails.append(f"t1_runs: 36 cells expected, found {len(per_cell)}")

    # --- structural claims ----------------------------------------------
    g0 = float(t.loc[t["agent_generation_time"] == G0, "c_ideal_tm"].mean())
    g1 = float(t.loc[t["agent_generation_time"] == G1, "c_ideal_tm"].mean())
    if not g0 > g1 > 0:
        fails.append(f"attenuation claim broken: M0 c_ideal_tm {g0:.6f} must "
                     f"exceed M1 {g1:.6f} (both positive)")
    neg_counts = {}
    for land in LANDSCAPES:
        sub = t[(t["landscape"] == land)
                & np.isclose(t["agent_generation_time"], G1)]
        cell = sub.groupby(["curiosity", "vanishing"])["c_ideal_fin"].mean()
        neg_counts[land] = int((cell < 0).sum())
        if neg_counts[land] != int((cell < 0).sum()):
            fails.append("internal: negative-cell recount mismatch")
    if not neg_counts["noisypr5"] > neg_counts["mgstd5"]:
        fails.append(f"negative-cell claim broken: noisypr5 {neg_counts['noisypr5']} "
                     f"<= mgstd5 {neg_counts['mgstd5']}")

    # --- sidecar vs recomputation ---------------------------------------
    if not csv_path.exists():
        fails.append(f"sidecar missing: {csv_path}")
        return fails
    got = pd.read_csv(csv_path)
    exp = _expected(root)
    if list(got.columns) != _COLS:
        fails.append(f"sidecar columns {list(got.columns)} != {_COLS}")
        return fails
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

    # --- hardened visual QA re-run --------------------------------------
    import fig_F1_v2_t1_postbug as F  # builder, for the QA contract only
    tmp = Path(tempfile.mkdtemp(prefix="f1v2_"))
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
    print(f"verify_F1_v2: all checks passed ({len(exp)} sidecar rows reproduced "
          f"from t1_runs.csv within 1e-9; 720 runs / 36 cells asserted; "
          f"visual_qa re-run clean)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
