"""F1_v2 -- ``T1`` entry sensitivity after ``BUG-0006`` (Section 2 refresh).

Refresh of the old ``F1_sensitivity_entry_curiosity`` line chart on the
post-fix re-measure ``data/bug0006_remeasure_v2/t1_runs.csv`` (720 runs =
2 entry rates x 3 curiosity x 3 vanishing x 2 landscapes x 20 seeds).

Two panels, both computed from the raw CSV:

  (a) time-mean alignment ``c_ideal_tm``: grouped bars of the seed mean for
      every (landscape, entry rate, curiosity) cell, M0 (``g = 0``, no entry:
      hatched) vs M1 (``g = 0.1``: solid).  The annotation box carries the
      M0/M1 attenuation ratio recomputed three ways -- pooled over both
      landscapes, mgstd5 only and noisypr5 only -- against the commission's
      quoted 1.74x (pooled) and 2.3x (mgstd5).
  (b) terminal alignment ``c_ideal_fin``: one bar per (landscape, entry rate)
      = pooled run mean over the 9 (curiosity x vanishing) cells, 95% CI
      across the 9 cell means, with one dot per cell mean.  The bar labels
      report the pooled value and the recomputed count of NEGATIVE cells
      (commission: 0/9 on mgstd5 vs 5/9 on noisypr5 at ``g = 0.1``).

Vocabulary: hue = landscape (mgstd5 blue, noisypr5 orange); fill = entry
regime (M0 hatched/hollow, M1 solid).  Nothing here is calibrated: the
numbers are model outputs of the ``T1`` arm.

Provenance rules (commission rule 3): ZERO hardcoded numbers -- every plotted
value, CI and text annotation is computed here from ``t1_runs.csv`` and
written into ``F1_v2_t1_postbug_data.csv``.  Commission-quoted values are
recomputed and any drift is reported to the coordinator.

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

DATA_ROOT = REPO / "data" / "bug0006_remeasure_v2"

#: entry regimes of the ``T1`` arm
G0, G1 = 0.0, 0.1
#: house vocabulary: hue = landscape, fill = entry regime
LAND_COLOR = {"mgstd5": PALETTE["blue"], "noisypr5": PALETTE["orange"]}
LANDSCAPES = ["mgstd5", "noisypr5"]
CURIOSITY = [0.3, 0.5, 0.7]
VANISHING = [0.5, 0.9, 1.0]
Z95 = 1.96

M_A = "c_ideal_tm (seed mean per landscape x entry rate x curiosity)"
M_B = "c_ideal_fin (pooled bar over runs, per-cell dots)"
M_REF = "reference (definitional constants)"


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------

def data_root_path(data_root=None) -> Path:
    return Path(data_root) if data_root is not None else DATA_ROOT


def load_runs(data_root=None) -> pd.DataFrame:
    path = data_root_path(data_root) / "t1_runs.csv"
    t = pd.read_csv(path)
    _require(len(t) == 720, f"t1_runs: 720 rows expected, found {len(t)}")
    _require(sorted(t["agent_generation_time"].unique().tolist()) == [G0, G1],
             "t1_runs: entry-rate grid must be {0.0, 0.1}")
    _require(sorted(t["curiosity"].unique().tolist()) == CURIOSITY,
             "t1_runs: curiosity grid")
    _require(sorted(t["vanishing"].unique().tolist()) == VANISHING,
             "t1_runs: vanishing grid")
    _require(sorted(t["landscape"].unique().tolist()) == LANDSCAPES,
             "t1_runs: landscape grid")
    per_cell = (t.groupby(["landscape", "agent_generation_time", "curiosity",
                           "vanishing"]).size().to_dict())
    _require(set(per_cell.values()) == {20},
             f"t1_runs: 20 seeds per cell expected, got {sorted(set(per_cell.values()))}")
    _require(len(per_cell) == 2 * 2 * 3 * 3, f"t1_runs: {len(per_cell)} cells")
    return t


def _mean_ci(vals) -> tuple[float, float, float, int]:
    v = pd.Series(vals, dtype=float).dropna()
    n = int(v.size)
    _require(n > 1, "CI needs at least two observations")
    m = float(v.mean())
    ci = Z95 * float(v.std(ddof=1)) / np.sqrt(n)
    return m, m - ci, m + ci, n


# ---------------------------------------------------------------------------
# annotation strings (spec; mirrored verbatim by verify_F1_v2.py)
# ---------------------------------------------------------------------------

def label_attenuation(pooled: float, mgstd5: float, noisypr5: float) -> str:
    return (f"M0/M1 c_ideal_tm attenuation: pooled {pooled:.2f}× · "
            f"mgstd5 {mgstd5:.2f}× · noisypr5 {noisypr5:.2f}×")


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
            landscape="pooled", curiosity=np.nan, g=np.nan, vanishing=np.nan,
            source_file, note=""):
        rows.append(dict(figure=FNAME, panel=panel, metric=metric,
                         statistic=statistic, series=series,
                         landscape=landscape, curiosity=curiosity, g=g,
                         vanishing=vanishing, value=float(value),
                         ci_lo=float(ci_lo), ci_hi=float(ci_hi), n=int(n),
                         source_file=source_file, note=note))

    src = "data/bug0006_remeasure_v2/t1_runs.csv"

    # ---------------- panel (a): time-mean alignment attenuation ---------
    a = (t.groupby(["landscape", "agent_generation_time", "curiosity"])
         ["c_ideal_tm"].agg(mean="mean", sd="std", n="size").reset_index())
    a["ci"] = Z95 * a["sd"] / np.sqrt(a["n"])
    a["ci_lo"] = a["mean"] - a["ci"]
    a["ci_hi"] = a["mean"] + a["ci"]
    _require(len(a) == 2 * 2 * 3, f"panel a: {len(a)} bars, expected 12")
    for r in a.itertuples():
        add("a", M_A, f"{r.landscape}_{'M1' if r.agent_generation_time else 'M0'}",
            statistic="mean", value=r.mean, ci_lo=r.ci_lo, ci_hi=r.ci_hi,
            n=int(r.n), landscape=r.landscape, curiosity=float(r.curiosity),
            g=float(r.agent_generation_time), source_file=src,
            note="seed mean of c_ideal_tm; 95% CI = mean +/- 1.96*SE over runs")

    ratios = {}
    for land in ["pooled"] + LANDSCAPES:
        sub = t if land == "pooled" else t[t["landscape"] == land]
        m0 = float(sub.loc[sub["agent_generation_time"] == G0, "c_ideal_tm"].mean())
        m1 = float(sub.loc[sub["agent_generation_time"] == G1, "c_ideal_tm"].mean())
        ratios[land] = (m0, m1, m0 / m1)
        add("a", f"annotation: attenuation_{land}", "annotation",
            statistic="annotation", value=m0 / m1, ci_lo=np.nan, ci_hi=np.nan,
            n=int(sub.loc[sub["agent_generation_time"] == G0].shape[0]),
            landscape=land, source_file=src,
            note=f"{land}: M0 c_ideal_tm {m0:.6f} / M1 {m1:.6f}")
    text_a = label_attenuation(ratios["pooled"][2], ratios["mgstd5"][2],
                               ratios["noisypr5"][2])
    add("a", "annotation: attenuation_line", "annotation", statistic="annotation",
        value=ratios["pooled"][2], ci_lo=np.nan, ci_hi=np.nan, n=360,
        source_file=src, note=text_a)

    # ---------------- panel (b): terminal alignment cell spread ----------
    cell = (t.groupby(["landscape", "agent_generation_time", "curiosity",
                       "vanishing"])["c_ideal_fin"]
            .agg(mean="mean", n="size").reset_index())
    _require(len(cell) == 4 * 9, f"panel b: {len(cell)} cells, expected 36")
    b_frames = []
    for land in LANDSCAPES:
        for g in (G0, G1):
            sub = cell[(cell["landscape"] == land)
                       & (cell["agent_generation_time"] == g)]
            _require(len(sub) == 9, f"panel b: {len(sub)} cells for {land}/{g}")
            pooled, lo, hi, n_cells = _mean_ci(sub["mean"])
            neg = int((sub["mean"] < 0).sum())
            series = f"{land}_{'M1' if g else 'M0'}"
            add("b", M_B, series, statistic="mean", value=pooled, ci_lo=lo,
                ci_hi=hi, n=n_cells, landscape=land, g=g, source_file=src,
                note="pooled mean of the 9 cell means; 95% CI = mean +/- "
                     "1.96*SE across the 9 cell means")
            for r in sub.itertuples():
                add("b", M_B, series, statistic="cell_mean", value=r.mean,
                    ci_lo=r.mean, ci_hi=r.mean, n=int(r.n), landscape=land,
                    g=g, curiosity=float(r.curiosity),
                    vanishing=float(r.vanishing), source_file=src,
                    note="mean of c_ideal_fin over the 20 seeds of the cell")
            add("b", f"annotation: negative_cells_{series}", "annotation",
                statistic="annotation", value=float(neg), ci_lo=np.nan,
                ci_hi=np.nan, n=n_cells, landscape=land, g=g, source_file=src,
                note=label_cell(pooled, neg, n_cells))
            b_frames.append(dict(landscape=land, g=g, series=series,
                                 value=pooled, ci_lo=lo, ci_hi=hi, n=n_cells,
                                 neg=neg, label=label_cell(pooled, neg, n_cells)))

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

    sidecar = pd.DataFrame(rows, columns=[
        "figure", "panel", "metric", "statistic", "series", "landscape",
        "curiosity", "g", "vanishing", "value", "ci_lo", "ci_hi", "n",
        "source_file", "note"])
    n_cells = int(cell.groupby(["landscape", "agent_generation_time"])
                  .size().max())
    texts = dict(attenuation=text_a, cells=note_cells(n_cells), ratios=ratios,
                 labels={f["series"]: f["label"] for f in b_frames})
    return dict(a=a, b=pd.DataFrame(b_frames), cell=cell), texts, sidecar


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------

def _bar_style(land: str, g: float) -> dict:
    """M0 = hatched/hollow, M1 = solid; hue = landscape."""
    col = LAND_COLOR[land]
    if g == G0:
        return dict(facecolor="white", edgecolor=col, hatch="///",
                    linewidth=1.1)
    return dict(facecolor=col, edgecolor=col, linewidth=1.1)


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
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.4, 5.6), dpi=s["dpi"])

    # ---------------- (a) c_ideal_tm attenuation -------------------------
    width = 0.19
    offsets = {-1.5: ("mgstd5", G0), -0.5: ("mgstd5", G1),
               0.5: ("noisypr5", G0), 1.5: ("noisypr5", G1)}
    xs = np.arange(len(CURIOSITY))
    for (dx, (land, g)) in offsets.items():
        sub = a[(a["landscape"] == land)
                & (np.isclose(a["agent_generation_time"], g))]
        sub = sub.set_index("curiosity").loc[CURIOSITY]
        pos = xs + dx * width
        axA.bar(pos, sub["mean"], width=width, zorder=3, **_bar_style(land, g))
        axA.errorbar(pos, sub["mean"],
                     yerr=[sub["mean"] - sub["ci_lo"], sub["ci_hi"] - sub["mean"]],
                     fmt="none", ecolor=LAND_COLOR[land], elinewidth=1.0,
                     capsize=2.2, zorder=4)
    _cosmetics(axA, s,
               "(a) Time-mean alignment falls when entry is switched on",
               "curiosity (1 - visibility weight ξ)",
               "c_ideal_tm\nCorr(prestige, ideal merit), time mean")
    axA.set_xticks(xs, [f"{c:.1f}" for c in CURIOSITY])
    axA.set_xlim(-0.55, len(CURIOSITY) - 0.45)
    vmax = float(max(a["ci_hi"].max(), a["mean"].max()))
    axA.set_ylim(0.0, vmax * 1.42)
    axA.text(0.015, 0.975, textwrap.fill(texts["attenuation"], width=44),
             transform=axA.transAxes, ha="left", va="top",
             fontsize=s["note_size"] - 0.8, zorder=6,
             bbox=dict(boxstyle="round,pad=0.32", fc="white",
                       ec=PALETTE["gray_light"], alpha=0.95))
    axA.legend(handles=[
        Patch(**_bar_style("mgstd5", G0), label="mgstd5 · M0 (g = 0)"),
        Patch(**_bar_style("mgstd5", G1), label="mgstd5 · M1 (g = 0.1)"),
        Patch(**_bar_style("noisypr5", G0), label="noisypr5 · M0 (g = 0)"),
        Patch(**_bar_style("noisypr5", G1), label="noisypr5 · M1 (g = 0.1)"),
    ], fontsize=s["tick_size"] - 1.5, frameon=False, loc="upper right",
        bbox_to_anchor=(1.0, 1.0), handlelength=1.8, labelspacing=0.3,
        borderaxespad=0.2)

    # ---------------- (b) terminal alignment cell spread -----------------
    groups = [(land, g) for land in LANDSCAPES for g in (G0, G1)]
    gx = np.arange(len(groups))
    for i, (land, g) in enumerate(groups):
        series = f"{land}_{'M1' if g else 'M0'}"
        row = b[(b["landscape"] == land) & (np.isclose(b["g"], g))].iloc[0]
        axB.bar(i, row["value"], width=0.52, zorder=3,
                **_bar_style(land, g))
        axB.errorbar([i], [row["value"]],
                     yerr=[[row["value"] - row["ci_lo"]],
                           [row["ci_hi"] - row["value"]]],
                     fmt="none", ecolor=LAND_COLOR[land], elinewidth=1.1,
                     capsize=2.6, zorder=4)
        sub = cell[(cell["landscape"] == land)
                   & (np.isclose(cell["agent_generation_time"], g))]
        jitter = np.linspace(-0.16, 0.16, len(sub))
        axB.plot(i + jitter, sub["mean"], linestyle="none", marker="o",
                 markersize=4.2, markerfacecolor="white",
                 markeredgecolor=LAND_COLOR[land], markeredgewidth=0.9,
                 zorder=5, label=f"{series} cell means")
    vlo = float(min(cell["mean"].min(), b["value"].min(), b["ci_lo"].min()))
    vhi = float(max(cell["mean"].max(), b["value"].max(), b["ci_hi"].max()))
    span = vhi - vlo
    axB.set_ylim(vlo - 0.55 * span, vhi + 0.30 * span)
    axB.axhline(0.0, color="black", linewidth=0.9, zorder=2)
    _cosmetics(axB, s,
               "(b) Terminal alignment: negative cells concentrate on noise",
               "landscape · entry regime",
               "c_ideal_fin\nCorr(prestige, ideal merit), final step")
    axB.set_xticks(gx, [f"{land}\n{'M0' if g == G0 else 'M1'}" for land, g in groups])
    axB.set_xlim(-0.6, len(groups) - 0.4)
    for i, (land, g) in enumerate(groups):
        series = f"{land}_{'M1' if g else 'M0'}"
        axB.text(i, axB.get_ylim()[0] + 0.06 * (axB.get_ylim()[1] - axB.get_ylim()[0]),
                 texts["labels"][series], ha="center", va="bottom",
                 fontsize=s["note_size"] - 1.0, zorder=6)
    axB.text(0.015, 0.975, texts["cells"], transform=axB.transAxes,
             ha="left", va="top", fontsize=s["note_size"] - 1.0, zorder=6)

    # ---------------- footnotes ------------------------------------------
    foot = [
        textwrap.fill(
            "Source: data/bug0006_remeasure_v2/t1_runs.csv (720 runs = "
            "2 entry rates x 3 curiosity x 3 vanishing x 2 landscapes x 20 "
            "seeds). M0 = g 0 (no entry), M1 = g 0.1. Hue = landscape, fill = "
            "entry regime (hatched/hollow = M0, solid = M1). Every plotted "
            "value, CI and annotation is computed from the raw CSV and stored in "
            f"{BASENAME}_data.csv.", width=150),
        textwrap.fill(
            "Measurement note: newborns no longer enter the data collector with "
            "prestige = 0, ideal_Merit = 0; correlation_reporter drops the "
            "non-finite pairs. Panel (b) therefore reads the terminal step as "
            "measured, and a negative cell is a genuine misalignment of the "
            "surviving population, not a birth-step artifact. MODEL-INTERNAL, "
            "NOT CALIBRATED: no empirical benchmark is plotted.", width=150),
    ]
    y = 0.158
    for block in foot:
        for line in block.split("\n"):
            fig.text(0.012, y, line, fontsize=s["note_size"] - 1.5,
                     color=PALETTE["gray"])
            y -= 0.021

    fig.subplots_adjust(left=0.075, right=0.988, top=0.905, bottom=0.245,
                        wspace=0.20)

    _clip_ticks(axA, prune_x=True)
    _clip_ticks(axB)
    specs = [
        {"ax": axA, "metric": M_A, "expect_bars": 12, "sidecar": sidecar,
         "filters": {"panel": "a"}},
        {"ax": axB, "metric": M_B, "expect_bars": 4, "expect_lines": 4,
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
