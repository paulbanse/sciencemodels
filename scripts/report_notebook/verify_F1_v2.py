"""Independent verifier for ``F1_v2_t1_postbug`` (ADR-0003 re-base).

``check()`` recomputes every plotted quantity and rendered annotation from the
RAW cells of ``data/sensitivity_adr0003/cells`` (18 cells x 20 seeds = 360
runs) and compares the result against ``F1_v2_t1_postbug_data.csv`` at a
configurable tolerance (default 1e-9).  Input contracts asserted here: 360
runs, curiosity {0.3, 0.5, 0.7}, landscape {mgstd5, noisypr5}, g {0.125, 0.25,
0.5}, 20 seeds per cell, 18 cells, ``exit_mode = fixed_span``,
``tau_max = 80``, ``H = 200``; and the structural claims of the figure -- the
time-mean alignment gradient is negative pooled and on the noisy landscape,
the clean landscape is much flatter, and the negative terminal cells sit on
the noisy landscape.

It also re-derives the two ``ADR-0003`` ledger contrasts that this figure
carries (pre ``-0.197``, post ``-0.076``) from the raw archives and compares
them with ``data/adr0003_remeasure/ledger.csv``, so the quoted drift is
checked rather than remembered.

The recomputation is deliberately self-contained (no numbers come from
``fig_F1_v2_t1_postbug``); the hardened visual QA is re-run by building the
figure into a temporary directory.

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

DATA_ROOT = REPO / "data" / "sensitivity_adr0003"
PRE_RUNS = REPO / "data" / "remeasure_v1" / "runs.csv"
LEDGER = REPO / "data" / "adr0003_remeasure" / "ledger.csv"
BASENAME = "F1_v2_t1_postbug"
FNAME = BASENAME

GS = [0.125, 0.25, 0.5]
G_LO, G_HI = min(GS), max(GS)
LANDSCAPES = ["mgstd5", "noisypr5"]
CURIOSITY = [0.3, 0.5, 0.7]
Z95 = 1.96
TAU_MAX, HORIZON = 80, 200

M_A = "c_ideal_tm (seed mean per landscape x g x curiosity)"
M_B = "c_ideal_fin (pooled bar over cells)"
M_B_CELL = "c_ideal_fin (per-cell means behind the bars)"
M_REF = "reference (definitional constants and archive pins)"

_KEYS = ["panel", "metric", "statistic", "series", "landscape", "curiosity",
         "g"]
_COLS = ["figure", "panel", "metric", "statistic", "series", "landscape",
         "curiosity", "g", "value", "ci_lo", "ci_hi", "n", "source_file",
         "note"]
SRC = "data/sensitivity_adr0003/cells/*.csv"


def _mean_ci(vals) -> tuple[float, float, float, int]:
    v = pd.Series(vals, dtype=float).dropna()
    n = int(v.size)
    m = float(v.mean())
    ci = Z95 * float(v.std(ddof=1)) / np.sqrt(n) if n > 1 else float("nan")
    return m, m - ci, m + ci, n


def label_gradient(pooled: float, mg: float, nz: float) -> str:
    return (f"high-g minus low-g c_ideal_tm: pooled {pooled:+.3f} · "
            f"mgstd5 {mg:+.3f} · noisypr5 {nz:+.3f}")


def label_cell(v: float, neg: int, cells: int) -> str:
    return f"{v:+.3f}\n{neg}/{cells} < 0"


def note_cells(cells: int) -> str:
    return f"each bar pools {cells} cells; dots are the {cells} cell means"


def _runs(root: Path = DATA_ROOT) -> pd.DataFrame:
    files = sorted((root / "cells").glob("*.csv"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def _expected(root: Path = DATA_ROOT) -> pd.DataFrame:
    t = _runs(root)
    rows: list[dict] = []

    def add(panel, metric, series, statistic, value, ci_lo, ci_hi, n,
            landscape="pooled", curiosity=np.nan, g=np.nan,
            source_file=SRC, note=""):
        rows.append(dict(figure=FNAME, panel=panel, metric=metric,
                         statistic=statistic, series=series,
                         landscape=landscape, curiosity=curiosity, g=g,
                         value=float(value), ci_lo=float(ci_lo),
                         ci_hi=float(ci_hi), n=int(n),
                         source_file=source_file, note=note))

    # (a) c_ideal_tm per (landscape, g, curiosity)
    for (land, g, cur), grp in t.groupby(["landscape", "g", "curiosity"]):
        v, lo, hi, n = _mean_ci(grp["c_ideal_tm"])
        add("a", M_A, f"{land} c={cur:.1f}", "mean", v, lo, hi, n,
            landscape=land, curiosity=float(cur), g=float(g),
            note="seed mean of c_ideal_tm; 95% CI = mean +/- 1.96*SE over "
                 "the 20 seeds of the cell")

    grads = {}
    for land in ["pooled"] + LANDSCAPES:
        sub = t if land == "pooled" else t[t["landscape"] == land]
        m = sub.groupby("g")["c_ideal_tm"].mean()
        lo, hi = float(m.loc[G_LO]), float(m.loc[G_HI])
        grads[land] = (hi, lo, hi - lo)
        add("a", f"annotation: gradient_{land}", "annotation", "annotation",
            hi - lo, np.nan, np.nan, len(sub), landscape=land,
            note=f"{land}: c_ideal_tm {hi:.6f} at g={G_HI:g} minus "
                 f"{lo:.6f} at g={G_LO:g}")
    add("a", "annotation: gradient_line", "annotation", "annotation",
        grads["pooled"][2], np.nan, np.nan, len(t),
        note=label_gradient(grads["pooled"][2], grads["mgstd5"][2],
                            grads["noisypr5"][2]))

    # (b) c_ideal_fin per cell, pooled bars + per-cell means
    cell = (t.groupby(["landscape", "g", "curiosity"])["c_ideal_fin"]
            .agg(mean="mean", n="size").reset_index())
    for land in LANDSCAPES:
        for gv in GS:
            sub = cell[(cell["landscape"] == land) & (cell["g"] == gv)]
            pooled, lo, hi, n_cells = _mean_ci(sub["mean"])
            neg = int((sub["mean"] < 0).sum())
            series = f"{land} g={gv:g}"
            add("b", M_B, series, "mean", pooled, lo, hi, n_cells,
                landscape=land, g=gv,
                note=f"pooled mean of the {n_cells} cell means; 95% CI = mean "
                     f"+/- 1.96*SE across the cell means")
            for r in sub.itertuples():
                add("b", M_B_CELL, series, "cell_mean", r.mean, r.mean,
                    r.mean, int(r.n), landscape=land, g=gv,
                    curiosity=float(r.curiosity),
                    note="mean of c_ideal_fin over the 20 seeds of the cell")
            add("b", f"annotation: negative_cells_{series}", "annotation",
                "annotation", float(neg), np.nan, np.nan, n_cells,
                landscape=land, g=gv, note=label_cell(pooled, neg, n_cells))

    for cat, val, ann in [
        ("g_low", G_LO, "lowest entry rate of the ADR-0003 grid"),
        ("g_high", G_HI, "highest entry rate of the ADR-0003 grid"),
        ("g_levels", len(GS), "entry rates 0.125/0.25/0.5"),
        ("cells_per_group", len(CURIOSITY), "cells per bar = curiosity levels"),
        ("tau_max", TAU_MAX, "ADR-0003 career horizon in steps"),
        ("horizon", HORIZON, "run horizon in steps"),
        ("gamma", 0.5 ** (1.0 / 10.0), "5-yr retention half-life"),
        ("z95", Z95, "95% CI multiplier (house convention)"),
    ]:
        rows.append(dict(figure=FNAME, panel="ref", metric=M_REF,
                         statistic="definitional constant", series=cat,
                         landscape="", curiosity=np.nan, g=np.nan,
                         value=float(val), ci_lo=np.nan, ci_hi=np.nan,
                         n=np.nan, source_file="ADR-0003 pins", note=ann))

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


def ledger_checks() -> list[str]:
    """Re-derive the T1 ledger contrasts and compare with the published CSV."""
    fails: list[str] = []
    if not LEDGER.exists() or not PRE_RUNS.exists():
        return fails  # optional cross-check, never a hardcoded fallback
    led = pd.read_csv(LEDGER)
    row = led[led["contrast"].astype(str).str.contains("c_ideal_tm")
              & led["archive"].astype(str).str.startswith("T1")]
    if len(row) != 1:
        return [f"ledger: expected exactly one T1 c_ideal_tm row, "
                f"found {len(row)}"]
    row = row.iloc[0]
    pre = pd.read_csv(PRE_RUNS)
    origin = pre["origin"].astype(str).str.upper()
    pre = pre[origin.str.startswith("T")]
    g = pre.groupby("agent_generation_time")["c_ideal_tm"].mean()
    pre_grad = float(g.loc[g.index.max()] - g.loc[g.index.min()])
    t = _runs()
    gm = t.groupby("g")["c_ideal_tm"].mean()
    post_grad = float(gm.loc[G_HI] - gm.loc[G_LO])
    if abs(pre_grad - float(row["pre_value"])) > 1e-3:
        fails.append(f"ledger pre value {float(row['pre_value']):+.4f} != "
                     f"recomputed {pre_grad:+.4f}")
    if abs(post_grad - float(row["post_value"])) > 1e-3:
        fails.append(f"ledger post value {float(row['post_value']):+.4f} != "
                     f"recomputed {post_grad:+.4f}")
    return fails


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
    if len(t) != 360:
        return [f"T1 runs: expected 360, found {len(t)}"]
    if sorted(t["curiosity"].unique().tolist()) != CURIOSITY:
        fails.append(f"T1 curiosity grid must be {CURIOSITY}")
    if sorted(t["g"].unique().tolist()) != GS:
        fails.append(f"T1 g grid must be {GS}")
    if sorted(t["landscape"].unique().tolist()) != LANDSCAPES:
        fails.append(f"T1 landscape grid must be {LANDSCAPES}")
    if t["exit_mode"].unique().tolist() != ["fixed_span"]:
        fails.append("T1 exit_mode must be pinned to fixed_span")
    if t["tau_max"].unique().tolist() != [TAU_MAX]:
        fails.append(f"T1 tau_max must be {TAU_MAX}")
    if t["horizon"].unique().tolist() != [HORIZON]:
        fails.append(f"T1 horizon must be {HORIZON}")
    per_cell = t.groupby(["landscape", "g", "curiosity"]).size()
    if set(per_cell.unique()) != {20}:
        fails.append(f"T1 20 seeds per cell expected, got "
                     f"{sorted(set(per_cell.unique()))}")
    if len(per_cell) != 18:
        fails.append(f"T1 18 cells expected, found {len(per_cell)}")

    # --- structural claims ----------------------------------------------
    pooled = t.groupby("g")["c_ideal_tm"].mean()
    if not float(pooled.loc[G_HI]) < float(pooled.loc[G_LO]):
        fails.append("panel (a) claim broken: pooled c_ideal_tm does not fall "
                     "with g")
    grads = {}
    for land in LANDSCAPES:
        m = t[t["landscape"] == land].groupby("g")["c_ideal_tm"].mean()
        grads[land] = float(m.loc[G_HI] - m.loc[G_LO])
    if not grads["noisypr5"] < 0:
        fails.append("panel (a) claim broken: noisypr5 c_ideal_tm does not "
                     "fall with g")
    if not abs(grads["mgstd5"]) < abs(grads["noisypr5"]) / 3.0:
        fails.append(f"panel (a) contrast broken: |mgstd5 gradient| "
                     f"{abs(grads['mgstd5']):.4f} is not far below noisypr5 "
                     f"{abs(grads['noisypr5']):.4f}")
    neg = {}
    for land in LANDSCAPES:
        cell = t[t["landscape"] == land].groupby(["g", "curiosity"])[
            "c_ideal_fin"].mean()
        neg[land] = int((cell < 0).sum())
    if not neg["noisypr5"] > neg["mgstd5"]:
        fails.append(f"panel (b) claim broken: noisypr5 negative cells "
                     f"{neg['noisypr5']} <= mgstd5 {neg['mgstd5']}")
    fails.extend(ledger_checks())

    # --- sidecar vs recomputation ---------------------------------------
    if not csv_path.exists():
        return fails + [f"sidecar missing: {csv_path}"]
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
    for col in ["metric", "statistic", "note", "source_file"]:
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
        fails.append(f"visual_qa re-run reported {res['qa']['_total']} "
                     f"failure(s)")
    fails.extend(_line_text_violations(col["fig"]))
    regot = pd.read_csv(Path(res["data_csv"]))
    if len(regot) != len(exp):
        fails.append(f"regenerated sidecar has {len(regot)} rows, expected "
                     f"{len(exp)}")
    else:
        regot = regot.copy()
        regot["_k"] = _key_str(regot)
        m2 = exp.merge(regot, on="_k", suffixes=("_e", "_r"), validate="1:1")
        d = np.abs(m2["value_e"].to_numpy(float) - m2["value_r"].to_numpy(float))
        d[np.isnan(d)] = 0.0
        if not (d <= tol).all():
            fails.append("regenerated sidecar values differ from the "
                         "recomputation")
    return fails


def main() -> int:
    fails = check()
    if fails:
        for f in fails:
            print(f"FAIL {f}")
        return 1
    exp = _expected()
    t = _runs()
    grad = t.groupby("g")["c_ideal_tm"].mean()
    print(f"verify_F1_v2: all checks passed ({len(exp)} sidecar rows "
          f"reproduced from the cells within 1e-9; 360 runs / 18 cells "
          f"asserted; pooled c_ideal_tm {float(grad.loc[G_LO]):.4f} -> "
          f"{float(grad.loc[G_HI]):.4f}; visual_qa re-run clean)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
