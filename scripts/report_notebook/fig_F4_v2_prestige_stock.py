"""F4_v2 - Post-fix prestige-stock wedge matrix (entry-on cells only).

Section 5.1 of the paper sketch. Terminal relational wedge
dr = c_local - c_ideal across the entry-on cells (g = 0.1, 3 curiosities
x 2 landscapes = 6 cells) for the three prestige stocks, from the
post-fix re-run:

    data/prestige_stock_v2/summary_cell.csv   (36 cells x 20 seeds = 720
    runs, produced by analyse_prestige_stock.py --out data/prestige_stock_v2)

VERIFICATIONS (brief), all computed here:
  * levels reflect post-fix c_ideal: entry-on wedge levels collapse
    relative to v1 (computed ratio in the note);
  * ordering: cross-stock cell-ranking rho read from both archives'
    wedge_rank_agreement.csv, plus the v1-vs-v2 entry-on rho (scipy);
    the honest computed range goes into the note -- the pre-fix
    0.965-0.993 does NOT reproduce post-fix, and the figure says so;
  * identity: g = 0 rows bit-identical across stocks (label columns
    excluded) -- asserted against the raw cell CSVs.

Replaces the pre-fix F4_prestige_stock_wedge_matrix.png. Model-internal,
NOT calibrated.

Run:
    MPLBACKEND=Agg /opt/hermes/.venv/bin/python \
        scripts/report_notebook/fig_F4_v2_prestige_stock.py
"""
from __future__ import annotations

import csv
import glob
import re
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import diagram_templates as T
import visual_qa

OUT = REPO / "output/report_notebook_v1/F4_v2_prestige_stock_wedge_matrix.png"

STOCKS = ("prestige", "prestige_vanishing", "prestige_visibility")
STOCK_LABEL = {"prestige": "prestige (raw)",
               "prestige_vanishing": "prestige_vanishing",
               "prestige_visibility": "prestige_visibility"}


def _identity_check() -> int:
    """g = 0 rows must be bit-identical across stocks (label cols excluded)."""
    tag = re.compile(r"cell_(?P<stock>pre|pvan|pvis)_gen(?P<gen>\d+)"
                     r"_cur(?P<cur>\d+)_(?P<land>\w+)")
    exclude = {"newcomer_incumbent_gap_active_stock", "entry_prestige_stock"}
    by_cell: dict = {}
    for path in sorted(glob.glob(str(REPO / "data/prestige_stock_v2/cell_*.csv"))):
        m = tag.match(Path(path).stem)
        if m is None or int(m["gen"]) != 0:
            continue
        with open(path, newline="") as fh:
            for rec in csv.DictReader(fh):
                key = (m["cur"], m["land"], rec["agent_seed"])
                row = tuple(sorted((k, v) for k, v in rec.items()
                                   if k not in exclude))
                by_cell.setdefault(key, {})[m["stock"]] = row
    bad = 0
    for key, per_stock in by_cell.items():
        if len(per_stock) == 3 and len(set(per_stock.values())) > 1:
            bad += 1
    return bad


def _rho_range(archive: str) -> tuple[float, float]:
    d = pd.read_csv(REPO / f"data/prestige_stock_{archive}/wedge_rank_agreement.csv")
    return float(d.spearman_rho.min()), float(d.spearman_rho.max())


def build():
    v2 = pd.read_csv(REPO / "data/prestige_stock_v2/summary_cell.csv")
    v1 = pd.read_csv(REPO / "data/prestige_stock_v1/summary_cell.csv")
    e2 = v2[v2.agent_generation_time > 0].copy()
    e1 = v1[v1.agent_generation_time > 0].copy()
    assert len(e2) == 18, f"expected 18 entry-on stock-cells, got {len(e2)}"

    # levels: v1 -> v2 (computed)
    lvl1, lvl2 = float(e1.wedge_fin_mean.mean()), float(e2.wedge_fin_mean.mean())
    ratio = lvl1 / lvl2 if lvl2 else float("nan")

    # ordering: cross-stock rank ranges (both archives) + v1-vs-v2 entry-on
    r2_lo, r2_hi = _rho_range("v2")
    r1_lo, r1_hi = _rho_range("v1")
    key = ["stock", "agent_generation_time", "initial_curiosity", "landscape"]
    m = e1.merge(e2, on=key, suffixes=("_v1", "_v2"))
    rho_v1v2, _ = stats.spearmanr(m.wedge_fin_mean_v1, m.wedge_fin_mean_v2)

    bad = _identity_check()
    assert bad == 0, f"g=0 identity violated in {bad} cells"

    s = {**T.HOUSE_STYLE, "figsize": (9.6, 4.9)}
    fig, ax = plt.subplots(figsize=s["figsize"])

    e2 = e2.copy()
    e2["cell"] = ("cur" + e2.initial_curiosity.map(lambda v: f"{v:.1f}")
                  + " " + e2.landscape.map(lambda s_: "mg" if s_ == "mgstd5"
                                           else "noisy"))
    cells = sorted(e2.cell.unique())
    x = np.arange(len(cells))
    w = 0.26
    colors = [T.PALETTE["purple"], T.PALETTE["orange"], T.PALETTE["green"]]
    for i, stock in enumerate(STOCKS):
        sub = e2.set_index(["cell", "stock"]).loc[
            [(c, stock) for c in cells]].reset_index()
        ax.bar(x + (i - 1) * w, sub.wedge_fin_mean, width=w,
               color=colors[i], label=STOCK_LABEL[stock])
        for xi, r in zip(x + (i - 1) * w, sub.itertuples()):
            v = r.wedge_fin_mean
            ax.text(xi, v + (0.0012 if v >= 0 else -0.0012),
                    f"{v:.3f}", ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=s["tick_size"] - 2)
    ax.axhline(0, color=T.PALETTE["gray"], linewidth=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(cells, fontsize=s["tick_size"] - 1)
    vmin, vmax = float(e2.wedge_fin_mean.min()), float(e2.wedge_fin_mean.max())
    ax.set_ylim(vmin - 0.013, vmax + 0.015)
    # pinned ticks: AutoLocator ghost labels collided with the ylabel
    # (same pattern as F4/F5)
    ax.set_yticks([-0.02, 0.0, 0.02, 0.04, 0.06, 0.08, 0.10])
    T._cosmetics(ax, s,
                 f"Post-fix wedge: same stock axis, {ratio:.0f}x lower levels",
                 "entry 0.1 cell (curiosity x landscape legibility)",
                 "relational wedge (c_local - c_ideal), final step")
    ax.legend(loc="upper left", fontsize=s["tick_size"] - 1, frameon=False)

    note = textwrap.fill(
        "source: data/prestige_stock_v2/summary_cell.csv (36 cells x 20 "
        "seeds = 720 runs, H = 200; analyse_prestige_stock.py --out "
        "data/prestige_stock_v2), entry-on cells only; wedge = c_local - "
        f"c_ideal. Levels: v1 entry-on mean {lvl1:.3f} -> v2 {lvl2:.3f} "
        f"({ratio:.0f}x lower): the pre-fix level carried the birth-step "
        "artifact in c_ideal. Cross-stock cell ordering (12 cells, "
        f"wedge_rank_agreement): v1 rho {r1_lo:.3f}-{r1_hi:.3f} -> v2 rho "
        f"{r2_lo:.3f}-{r2_hi:.3f}; v1-vs-v2 entry-on rho {rho_v1v2:.3f} "
        "-- ordering loosens post-fix, it is NOT preserved at >0.95. "
        f"g=0 identity: {bad} mismatches. Model-internal, NOT calibrated.",
        width=150)
    fig.text(0.006, 0.012, note, fontsize=s["note_size"],
             color=T.PALETTE["gray"], va="bottom", linespacing=1.45)
    fig.tight_layout(rect=(0, 0.23, 1, 0.90))

    export = e2[["stock", "initial_curiosity", "landscape", "cell",
                 "wedge_fin_mean", "wedge_fin_se"]].copy()
    export = export.rename(columns={"wedge_fin_mean": "value"})
    # no CI is drawn on a bar canvas, so the QA envelope must equal the
    # plotted values; the SE rides along as an extra column
    export["ci_lo"] = export["value"]
    export["ci_hi"] = export["value"]
    export.insert(0, "metric", "f4v2")

    report = visual_qa.assert_figure_clean(fig, [
        {"ax": ax, "metric": "f4v2", "expect_bars": 18,
         "sidecar": export, "ci_cols": ("ci_lo", "ci_hi")},
    ])

    outdir = OUT.parent
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=s["dpi"])
    pdf = OUT.with_suffix(".pdf")
    fig.savefig(pdf)
    plt.close(fig)
    csv_path = T.write_figure_data(export, OUT)
    print(f"QA: {report.get('_total', 0)} failure(s)")
    print(f"Wrote {OUT} / {pdf} / {csv_path}")
    print(f"levels v1={lvl1:.4f} v2={lvl2:.4f} ratio={ratio:.1f} | "
          f"rho v2 {r2_lo:.3f}-{r2_hi:.3f} | v1v2 {rho_v1v2:.3f} | bad={bad}")
    return {"png": str(OUT), "pdf": str(pdf), "data_csv": str(csv_path)}


if __name__ == "__main__":
    build()
