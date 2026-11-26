"""F1_v2 -- ``T1`` entry sensitivity on the ``ADR-0003`` clock (Figure 5).

Re-based (paper v3.5) on ``data/sensitivity_adr0003/cells`` (18 cells x 20
seeds = 360 runs: curiosity 0.3/0.5/0.7 x landscape {mgstd5, noisypr5} x
g {0.125, 0.25, 0.5}, ``mgstd5``/``noisypr5`` landscapes, ``fixed_span``
replacement, ``H = 200``).

The previous version of this file read ``data/bug0006_remeasure_v2/t1_runs
.csv`` (36 cells: 3 curiosity x 3 vanishing x 2 landscapes x 2 entry rates
{0.0, 0.1}). The ADR-0003 T1 arm keeps no ``g = 0`` control and no vanishing
tier, so the old panel structure cannot be rebuilt honestly:

  (a) was the M0/M1 attenuation ratio (g = 0 vs g = 0.1). There is no
      ``g = 0`` cell in the archive, so panel (a) now shows ``c_ideal_tm``
      across the three entry rates, one line per (landscape, curiosity), and
      the annotation reports the computed high-g minus low-g gradient
      (pooled, mgstd5, noisypr5) against the pre-ADR value recomputed from
      ``data/remeasure_v1/runs.csv`` (origin = T1).
  (b) keeps the pooled-bar + per-cell-dot design, but a bar now pools the
      three curiosity cells of one (landscape, g) pair instead of nine
      (curiosity x vanishing) cells.

Vocabulary: hue = curiosity, line style = landscape (solid mgstd5, dashed
noisypr5); in panel (b) hue = landscape. Nothing here is calibrated: every
number is a model output of the ``T1`` arm.

Provenance rules: ZERO hardcoded numbers -- every plotted value, CI and text
annotation is computed here from the cells and written into
``F1_v2_t1_postbug_data.csv``.

Run:  MPLBACKEND=Agg python3 scripts/report_notebook/fig_F1_v2_t1_postbug.py
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from diagram_templates import (  # noqa: E402
    HOUSE_STYLE,
    PALETTE,
    _cosmetics,
    write_figure_data,
)
import visual_qa  # noqa: E402

FNAME = "F1_v2_t1_postbug"
BASENAME = FNAME

#: default archive = the ADR-0003 T1 re-run; any root with ``cells/`` works
DATA_ROOT = REPO / "data" / "sensitivity_adr0003"
#: pre-ADR reference for the before/after ledger (read-only, optional)
PRE_RUNS = REPO / "data" / "remeasure_v1" / "runs.csv"
#: the archive the PREVIOUS version of this figure plotted (drift report only)
PREV_RUNS = REPO / "data" / "bug0006_remeasure_v2" / "t1_runs.csv"

GS = [0.125, 0.25, 0.5]
G_LO, G_HI = min(GS), max(GS)
LANDSCAPES = ["mgstd5", "noisypr5"]
CURIOSITY = [0.3, 0.5, 0.7]
Z95 = 1.96

#: house vocabulary
CUR_COLOR = {0.3: PALETTE["purple"], 0.5: PALETTE["orange"],
             0.7: PALETTE["green"]}
LAND_STYLE = {"mgstd5": "-", "noisypr5": (0, (4, 2))}
LAND_COLOR = {"mgstd5": PALETTE["blue"], "noisypr5": PALETTE["orange"]}

M_A = "c_ideal_tm (seed mean per landscape x g x curiosity)"
M_B = "c_ideal_fin (pooled bar over cells)"
M_B_CELL = "c_ideal_fin (per-cell means behind the bars)"
M_REF = "reference (definitional constants and archive pins)"

ADR_PINS = ("ADR-0003 clock: 1 step = 0.5 yr, entry age 25, tau_max 80 "
            "(40-yr careers), gamma = 0.5**(1/10) = 0.933, g in "
            "{0.125, 0.25, 0.5}, exit_mode fixed_span, H = 200")


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------

def data_root_path(data_root=None) -> Path:
    return Path(data_root) if data_root is not None else DATA_ROOT


def load_runs(data_root=None) -> pd.DataFrame:
    root = data_root_path(data_root)
    files = sorted((root / "cells").glob("*.csv"))
    _require(len(files) == 18, f"T1 gross cells=18 expected, found {len(files)}")
    frames = []
    for f in files:
        d = pd.read_csv(f)
        _require(len(d) == 20, f"{f.name}: 20 seed rows expected, "
                               f"found {len(d)}")
        frames.append(d)
    t = pd.concat(frames, ignore_index=True)
    _require(len(t) == 360, f"T1 runs: 360 expected, found {len(t)}")
    _require(sorted(t["curiosity"].unique().tolist()) == CURIOSITY,
             "T1 curiosity grid must be {0.3, 0.5, 0.7}")
    _require(sorted(t["g"].unique().tolist()) == GS,
             "T1 g grid must be the ADR-0003 entry grid")
    _require(sorted(t["landscape"].unique().tolist()) == LANDSCAPES,
             "T1 landscape grid")
    _require(t["exit_mode"].unique().tolist() == ["fixed_span"],
             "T1 exit_mode is pinned to fixed_span")
    _require(t["tau_max"].unique().tolist() == [80], "T1 tau_max=80")
    _require(t["horizon"].unique().tolist() == [200], "T1 H=200")
    per_cell = t.groupby(["landscape", "g", "curiosity"]).size()
    _require(set(per_cell.unique()) == {20},
             f"T1 20 seeds per cell expected, got {sorted(set(per_cell.unique()))}")
    _require(len(per_cell) == 18, f"T1 {len(per_cell)} cells, expected 18")
    return t


def _mean_ci(vals) -> tuple[float, float, float, int]:
    v = pd.Series(vals, dtype=float).dropna()
    n = int(v.size)
    _require(n > 1, "CI needs at least two observations")
    m = float(v.mean())
    ci = Z95 * float(v.std(ddof=1)) / np.sqrt(n)
    return m, m - ci, m + ci, n


def pre_reference() -> dict:
    """Pre-ADR ``c_ideal_tm`` gradient, recomputed where the archive exists."""
    out: dict = {}
    for key, path, gcol, floor in (
            ("ledger", PRE_RUNS, "agent_generation_time", None),
            ("previous", PREV_RUNS, "agent_generation_time", None)):
        if not path.exists():
            continue
        d = pd.read_csv(path)
        if "origin" in d.columns:
            origin = d["origin"].astype(str).str.upper()
            d = d[origin.str.startswith("T1") | origin.str.startswith("T")]
        g = d.groupby(gcol)["c_ideal_tm"].mean()
        if len(g) < 2:
            continue
        hi, lo = float(g.index.max()), float(g.index.min())
        out[key] = {
            "path": str(path.relative_to(REPO)),
            "n_runs": int(len(d)),
            "gradient": float(g.loc[hi] - g.loc[lo]),
            "g_pair": (lo, hi),
            "hi_value": float(g.loc[hi]),
            "lo_value": float(g.loc[lo]),
        }
    return out


# ---------------------------------------------------------------------------
# annotation strings (spec; mirrored verbatim by verify_F1_v2.py)
# ---------------------------------------------------------------------------

def label_gradient(pooled: float, mg: float, nz: float) -> str:
    return (f"high-g minus low-g c_ideal_tm: pooled {pooled:+.3f} · "
            f"mgstd5 {mg:+.3f} · noisypr5 {nz:+.3f}")


def label_cell(v: float, neg: int, cells: int) -> str:
    return f"{v:+.3f}\n{neg}/{cells} < 0"


def note_cells(cells: int) -> str:
    return f"each bar pools {cells} cells; dots are the {cells} cell means"


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------

def compute(data_root=None):
    t = load_runs(data_root)
    rows: list[dict] = []

    def add(panel, metric, series, *, statistic, value, ci_lo, ci_hi, n,
            landscape="pooled", curiosity=np.nan, g=np.nan,
            source_file, note=""):
        rows.append(dict(figure=FNAME, panel=panel, metric=metric,
                         statistic=statistic, series=series,
                         landscape=landscape, curiosity=curiosity, g=g,
                         value=float(value), ci_lo=float(ci_lo),
                         ci_hi=float(ci_hi), n=int(n),
                         source_file=source_file, note=note))

    src = "data/sensitivity_adr0003/cells/*.csv"

    # ---------------- panel (a): time-mean alignment across g -------------
    a = (t.groupby(["landscape", "g", "curiosity"])["c_ideal_tm"]
         .agg(mean="mean", sd="std", n="size").reset_index())
    a["ci"] = Z95 * a["sd"] / np.sqrt(a["n"])
    _require(len(a) == len(LANDSCAPES) * len(GS) * len(CURIOSITY),
             f"panel a: {len(a)} points, expected 18")
    for r in a.itertuples():
        add("a", M_A, f"{r.landscape} c={r.curiosity:.1f}", statistic="mean",
            value=r.mean, ci_lo=r.mean - r.ci, ci_hi=r.mean + r.ci,
            n=int(r.n), landscape=r.landscape, curiosity=float(r.curiosity),
            g=float(r.g), source_file=src,
            note="seed mean of c_ideal_tm; 95% CI = mean +/- 1.96*SE over "
                 "the 20 seeds of the cell")

    grads = {}
    for land in ["pooled"] + LANDSCAPES:
        sub = t if land == "pooled" else t[t["landscape"] == land]
        m = sub.groupby("g")["c_ideal_tm"].mean()
        lo, hi = float(m.loc[G_LO]), float(m.loc[G_HI])
        grads[land] = (hi, lo, hi - lo)
        add("a", f"annotation: gradient_{land}", "annotation",
            statistic="annotation", value=hi - lo, ci_lo=np.nan, ci_hi=np.nan,
            n=int(len(sub)), landscape=land, source_file=src,
            note=f"{land}: c_ideal_tm {hi:.6f} at g={G_HI:g} minus "
                 f"{lo:.6f} at g={G_LO:g}")
    pre = pre_reference()
    text_a = label_gradient(grads["pooled"][2], grads["mgstd5"][2],
                            grads["noisypr5"][2])
    add("a", "annotation: gradient_line", "annotation", statistic="annotation",
        value=grads["pooled"][2], ci_lo=np.nan, ci_hi=np.nan, n=int(len(t)),
        source_file=src, note=text_a)

    # ---------------- panel (b): terminal alignment cell spread -----------
    cell = (t.groupby(["landscape", "g", "curiosity"])["c_ideal_fin"]
            .agg(mean="mean", n="size").reset_index())
    _require(len(cell) == 18, f"panel b: {len(cell)} cells, expected 18")
    b_frames = []
    for land in LANDSCAPES:
        for gv in GS:
            sub = cell[(cell["landscape"] == land) & (cell["g"] == gv)]
            _require(len(sub) == len(CURIOSITY),
                     f"panel b: {len(sub)} cells for {land}/{gv}")
            pooled, lo, hi, n_cells = _mean_ci(sub["mean"])
            neg = int((sub["mean"] < 0).sum())
            series = f"{land} g={gv:g}"
            add("b", M_B, series, statistic="mean", value=pooled, ci_lo=lo,
                ci_hi=hi, n=n_cells, landscape=land, g=gv, source_file=src,
                note=f"pooled mean of the {n_cells} cell means; 95% CI = mean "
                     f"+/- 1.96*SE across the cell means")
            for r in sub.itertuples():
                add("b", M_B_CELL, series, statistic="cell_mean", value=r.mean,
                    ci_lo=r.mean, ci_hi=r.mean, n=int(r.n), landscape=land,
                    g=gv, curiosity=float(r.curiosity), source_file=src,
                    note="mean of c_ideal_fin over the 20 seeds of the cell")
            add("b", f"annotation: negative_cells_{series}", "annotation",
                statistic="annotation", value=float(neg), ci_lo=np.nan,
                ci_hi=np.nan, n=n_cells, landscape=land, g=gv,
                source_file=src, note=label_cell(pooled, neg, n_cells))
            b_frames.append(dict(landscape=land, g=gv, series=series,
                                 value=pooled, ci_lo=lo, ci_hi=hi,
                                 n=n_cells, neg=neg,
                                 label=label_cell(pooled, neg, n_cells)))

    for cat, val, ann in [
        ("g_low", G_LO, "lowest entry rate of the ADR-0003 grid"),
        ("g_high", G_HI, "highest entry rate of the ADR-0003 grid"),
        ("g_levels", len(GS), "entry rates 0.125/0.25/0.5"),
        ("cells_per_group", len(CURIOSITY),
         "cells per bar = curiosity levels"),
        ("tau_max", 80, "ADR-0003 career horizon in steps"),
        ("horizon", 200, "run horizon in steps"),
        ("gamma", 0.5 ** (1.0 / 10.0), "5-yr retention half-life"),
        ("z95", Z95, "95% CI multiplier (house convention)"),
    ]:
        rows.append(dict(figure=FNAME, panel="ref", metric=M_REF,
                         statistic="definitional constant", series=cat,
                         landscape="", curiosity=np.nan, g=np.nan,
                         value=float(val), ci_lo=np.nan, ci_hi=np.nan,
                         n=np.nan, source_file="ADR-0003 pins", note=ann))

    sidecar = pd.DataFrame(rows, columns=[
        "figure", "panel", "metric", "statistic", "series", "landscape",
        "curiosity", "g", "value", "ci_lo", "ci_hi", "n", "source_file",
        "note"])
    n_cells = int(cell.groupby(["landscape", "g"]).size().max())
    texts = dict(gradient=text_a, cells=note_cells(n_cells), grads=grads,
                 pre=pre, labels={f["series"]: f["label"] for f in b_frames},
                 neg={f["series"]: f["neg"] for f in b_frames})
    return dict(a=a, b=pd.DataFrame(b_frames), cell=cell), texts, sidecar


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------

def _clip_ticks(ax, *, prune_x: bool = False, eps: float = 1e-12) -> None:
    """Drop locator ticks outside the view limits (they render outside the
    axes and collide with the title / neighbouring panel)."""
    lo, hi = ax.get_ylim()
    ax.set_yticks([t for t in ax.get_yticks() if lo - eps <= t <= hi + eps])
    if prune_x:
        lo, hi = ax.get_xlim()
        ax.set_xticks([t for t in ax.get_xticks() if lo - eps <= t <= hi + eps])


def _draw(frames, texts, sidecar, png: Path, pdf: Path,
          collector: dict | None = None) -> dict:
    s = HOUSE_STYLE
    a, b, cell = frames["a"], frames["b"], frames["cell"]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.4, 5.8), dpi=s["dpi"])

    # ---------------- (a) c_ideal_tm across the entry-rate grid -----------
    for land in LANDSCAPES:
        for cur in CURIOSITY:
            sub = (a[(a["landscape"] == land)
                     & np.isclose(a["curiosity"], cur)]
                   .sort_values("g"))
            axA.plot(sub["g"], sub["mean"], color=CUR_COLOR[cur],
                     linestyle=LAND_STYLE[land], marker="o", markersize=4.6,
                     linewidth=1.7, label=f"{land} · c = {cur:.1f}")
    # structural claim behind the panel (a) title
    _require(texts["grads"]["pooled"][2] < 0.0,
             "panel (a) claim broken: pooled c_ideal_tm does not fall with g")
    _require(texts["grads"]["noisypr5"][2] < 0.0,
             "panel (a) claim broken: noisypr5 c_ideal_tm does not fall with g")
    _cosmetics(axA, s,
               f"(a) Entry erodes time-mean alignment "
               f"(pooled {texts['grads']['pooled'][2]:+.3f})",
               "entry rate g (agents per step)",
               "c_ideal_tm\nCorr(prestige, ideal merit), time mean")
    axA.set_xticks(GS, [f"{v:g}" for v in GS], fontsize=s["tick_size"])
    axA.set_xlim(G_LO - 0.05 * (G_HI - G_LO), G_HI + 0.05 * (G_HI - G_LO))
    vmax = float(a["mean"].max())
    axA.set_ylim(-0.10 * vmax, 1.26 * vmax)
    axA.legend(fontsize=s["tick_size"] - 1.6, frameon=False, loc="upper right",
               labelspacing=0.28, handlelength=2.4, borderaxespad=0.3,
               title="landscape · curiosity", title_fontsize=s["tick_size"] - 1.6)

    grad_txt = textwrap.fill(texts["gradient"], width=42)
    pre = texts["pre"].get("ledger")
    if pre is not None:
        grad_txt += (f"\npre-ADR {pre['gradient']:+.3f} "
                     f"(g {pre['g_pair'][1]:g} minus {pre['g_pair'][0]:g})")
    axA.text(0.02, 0.035, grad_txt, transform=axA.transAxes, ha="left",
             va="bottom", fontsize=s["note_size"] - 1.0, zorder=6,
             bbox=dict(boxstyle="round,pad=0.3", fc="white",
                       ec=PALETTE["gray_light"], alpha=0.95))

    # ---------------- (b) terminal alignment cell spread -----------------
    width = 0.36
    for j, land in enumerate(LANDSCAPES):
        for k, gv in enumerate(GS):
            row = b[(b["landscape"] == land) & np.isclose(b["g"], gv)].iloc[0]
            pos = k + (j - 0.5) * width
            axB.bar(pos, row["value"], width=width, zorder=3,
                    facecolor=LAND_COLOR[land] if j == 0 else "white",
                    edgecolor=LAND_COLOR[land], linewidth=1.1,
                    hatch=None if j == 0 else "///",
                    label=land if k == 0 else None)
            axB.errorbar([pos], [row["value"]],
                         yerr=[[row["value"] - row["ci_lo"]],
                               [row["ci_hi"] - row["value"]]],
                         fmt="none", ecolor=LAND_COLOR[land],
                         elinewidth=1.0, capsize=2.4, zorder=4)
            sub = cell[(cell["landscape"] == land) & np.isclose(cell["g"], gv)]
            jitter = np.linspace(-0.10, 0.10, len(sub))
            axB.scatter(pos + jitter, sub["mean"], s=17, marker="o",
                        facecolor="white", edgecolor=LAND_COLOR[land],
                        linewidth=0.9, zorder=5)
    vlo = float(min(cell["mean"].min(), b["ci_lo"].min()))
    vhi = float(max(cell["mean"].max(), b["ci_hi"].max()))
    span = vhi - vlo
    axB.set_ylim(vlo - 0.42 * span, vhi + 0.30 * span)
    axB.axhline(0.0, color="black", linewidth=0.9, zorder=2)
    _cosmetics(axB, s,
               "(b) Terminal alignment: the negatives sit on noisy terrain",
               "entry rate g",
               "c_ideal_fin\nCorr(prestige, ideal merit), final step")
    axB.set_xticks(np.arange(len(GS)), [f"{v:g}" for v in GS],
                   fontsize=s["tick_size"])
    axB.set_xlim(-0.6, len(GS) - 0.4)
    lo, hi = axB.get_ylim()
    for j, land in enumerate(LANDSCAPES):
        for k, gv in enumerate(GS):
            series = f"{land} g={gv:g}"
            row = b[(b["landscape"] == land) & np.isclose(b["g"], gv)].iloc[0]
            pos = k + (j - 0.5) * width
            y = row["ci_hi"] + 0.03 * (hi - lo)
            if row["value"] < 0:
                y = row["ci_lo"] - 0.04 * (hi - lo)
                va = "top"
            else:
                va = "bottom"
            axB.text(pos, y, texts["labels"][series], ha="center", va=va,
                     fontsize=s["note_size"] - 1.6, zorder=6)
    axB.legend(fontsize=s["tick_size"] - 1.5, frameon=False, ncol=2,
               loc="upper left", handlelength=1.8, labelspacing=0.3,
               borderaxespad=0.25)
    axB.text(0.02, 0.02, texts["cells"], transform=axB.transAxes, ha="left",
             va="bottom", fontsize=s["note_size"] - 1.2, zorder=6)

    # ---------------- footnotes ------------------------------------------
    n_neg_noisy = sum(v for k, v in texts["neg"].items() if "noisypr5" in k)
    n_neg_clean = sum(v for k, v in texts["neg"].items() if "mgstd5" in k)
    n_cells = int(cell.groupby(["landscape", "g"]).size().max())
    foot = [
        textwrap.fill(
            f"source: data/sensitivity_adr0003/cells (18 cells x 20 seeds = "
            f"360 runs; curiosity 0.3/0.5/0.7 x landscape x g). {ADR_PINS}. "
            f"Panel (a) hue = curiosity, line style = landscape (solid "
            f"mgstd5, dashed noisypr5); panel (b) hue = landscape "
            f"(filled mgstd5, hatched noisypr5). Model-internal, not "
            f"calibrated.", width=150),
        textwrap.fill(
            f"The ADR-0003 T1 arm has no g = 0 control and no vanishing tier: "
            f"the previous M0/M1 attenuation ratio (and its 36-cell "
            f"curiosity x vanishing factorial) cannot be rebuilt from this "
            f"archive. Panel (a) therefore reports the entry-rate gradient "
            f"instead; panel (b) pools {n_cells} curiosity cells per bar "
            f"(previously 9 curiosity x vanishing cells). Negative terminal "
            f"cells: {n_neg_clean}/{n_cells * len(GS)} mgstd5 vs "
            f"{n_neg_noisy}/{n_cells * len(GS)} noisypr5.", width=150),
        textwrap.fill(
            "Measurement note (unchanged from BUG-0006): newborns no longer "
            "enter the data collector with prestige = 0, ideal_Merit = 0 and "
            "correlation_reporter drops the non-finite pairs, so a negative "
            "cell is a genuine misalignment of the surviving population, not "
            "a birth-step artifact. Every plotted value, CI and annotation "
            "is computed from the cells and stored in "
            f"{BASENAME}_data.csv.", width=150),
    ]
    lines: list[str] = []
    for block in foot:
        lines.extend(block.split("\n"))
    top, floor = 0.175, 0.012
    step = min(0.0195, (top - floor) / max(len(lines), 1))
    y = top
    for line in lines:
        fig.text(0.010, y, line, fontsize=s["note_size"] - 1.6,
                 color=PALETTE["gray"])
        y -= step

    fig.subplots_adjust(left=0.072, right=0.988, top=0.915, bottom=0.300,
                        wspace=0.22)

    _clip_ticks(axA, prune_x=True)
    _clip_ticks(axB)
    specs = [
        {"ax": axA, "metric": M_A,
         "expect_lines": len(LANDSCAPES) * len(CURIOSITY),
         "points_per_line": len(GS), "sidecar": sidecar,
         "filters": {"panel": "a"}, "ci_cols": ("value", "value")},
        {"ax": axB, "metric": M_B, "expect_bars": len(LANDSCAPES) * len(GS),
         "sidecar": sidecar, "filters": {"panel": "b"}},
    ]
    report = visual_qa.assert_figure_clean(fig, specs)

    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=s["dpi"])
    fig.savefig(pdf)
    if collector is not None:
        collector["fig"] = fig
    plt.close(fig)
    return report


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def build(outdir="output/report_notebook_v1", data_root=None,
          collector: dict | None = None) -> dict:
    out = Path(outdir)
    if not out.is_absolute():
        out = REPO / out
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{BASENAME}.png"
    pdf = out / f"{BASENAME}.pdf"
    frames, texts, sidecar = compute(data_root=data_root)
    report = _draw(frames, texts, sidecar, png, pdf, collector=collector)
    csv_path = write_figure_data(sidecar, png)
    print(f"F1_v2: {png} | {pdf} | {csv_path}")
    print(f"sidecar rows: {len(sidecar)} | visual_qa failures: {report['_total']}")
    return {"png": str(png), "pdf": str(pdf), "data_csv": str(csv_path),
            "qa": report, "sidecar_rows": len(sidecar)}


def main() -> int:
    build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
