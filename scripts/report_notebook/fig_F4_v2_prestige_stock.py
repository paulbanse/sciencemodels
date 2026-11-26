"""F4_v2 - Prestige-stock wedge matrix (ADR-0003 clock, stock x curiosity x landscape x g).

Section 5.1 of the paper sketch.  The relational wedge on the ADR-0003 clock,
``wedge_tm = c_local_tm - c_ideal_tm`` (time-mean Corr(prestige, local) minus
time-mean Corr(prestige, ideal merit)) for the three entry prestige stocks:

    prestige | prestige_vanishing | prestige_visibility

Left panel: the full matrix -- one row per (stock, g) pair, one column per
(curiosity, landscape) cell -- annotated with the cell mean over 20 seeds.
Right panel: the stock means (mean of the 18 cell means, 95% CI over those
cell means) with the ordering and the near-zero level of all three stocks.

ARCHIVE NAMING.  Cells are ``prest_c-<c>_g-<g>_land-<land>_stock-<stock>``
with the ADR-0003 exit rule ``fixed_span`` and g in {0.125, 0.25, 0.5}
(pre-ADR archive: per-cell means in ``summary_cell.csv`` with the old entry
grid); one step is 0.5 yr, tau_max = 80, gamma = 0.933.

Sources (read-only): ``data/prestige_adr0003/cells/prest_*.csv`` (54 cells =
3 stocks x 3 c x 2 landscapes x 3 g, 20 seeds each = 1,080 runs).

QUALIFICATION (iii) carried on the figure, as computed annotations: the pooled
time-mean wedge ordering prestige > prestige_visibility > prestige_vanishing
holds, but ALL THREE levels are slightly negative and their CIs include zero;
per-cell the ordering is not resolved.  This is a level finding, not a rank
reversal.

``build(outdir, data_root)``: ``data_root`` defaults to the ADR-0003 archive;
pass ``data_root="data/prestige_stock_v2"`` to rebuild the pre-ADR figure from
``summary_cell.csv`` (that archive stores per-cell means only, so the g = 0
rows are dropped and no seed-level CI exists).

Model-internal, NOT calibrated.

Run:
    MPLBACKEND=Agg /opt/hermes/.venv/bin/python \
        scripts/report_notebook/fig_F4_v2_prestige_stock.py
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import diagram_templates as T
import visual_qa

BASENAME = "F4_v2_prestige_stock_wedge_matrix"
DATA_ROOT = REPO / "data" / "prestige_adr0003"

STOCKS = ("prestige", "prestige_visibility", "prestige_vanishing")
STOCK_LABEL = {"prestige": "prestige (raw)",
               "prestige_visibility": "prestige_visibility",
               "prestige_vanishing": "prestige_vanishing"}
CURIOSITIES = (0.3, 0.5, 0.7)
LANDSCAPES = ("mgstd5", "noisypr5")
G_LEVELS = (0.125, 0.25, 0.5)
Z95 = 1.96


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def data_root_path(data_root=None) -> Path:
    """Prestige-stock archive root (default: ``data/prestige_adr0003``)."""
    root = Path(data_root) if data_root is not None else DATA_ROOT
    return root if root.is_absolute() else (REPO / root)


def load(data_root=None) -> tuple[pd.DataFrame, str, str]:
    """Per-cell wedge means: stock, g, curiosity, landscape, value, n."""
    root = data_root_path(data_root)
    files = sorted((root / "cells").glob("prest_*.csv"))
    if files:
        _require(len(files) == 54,
                 f"{root}/cells: 54 stock cells expected, found {len(files)}")
        d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        _require(len(d) == 54 * 20, f"{len(d)} run rows, expected 1,080")
        _require(sorted(d.stock.unique().tolist()) == sorted(STOCKS),
                 f"stock grid {sorted(d.stock.unique())}")
        _require(sorted(d.g.unique().tolist()) == list(G_LEVELS),
                 f"entry grid {sorted(d.g.unique())}")
        _require(sorted(d.curiosity.unique().tolist()) == list(CURIOSITIES),
                 f"curiosity grid {sorted(d.curiosity.unique())}")
        _require(sorted(d.landscape.unique().tolist()) == sorted(LANDSCAPES),
                 f"landscape grid {sorted(d.landscape.unique())}")
        rel = (d.wedge_tm - (d.c_local_tm - d.c_ideal_tm)).abs().max()
        _require(rel < 1e-12, f"wedge identity violated (max {rel:g})")
        cell = (d.groupby(["stock", "g", "curiosity", "landscape"])
                .agg(value=("wedge_tm", "mean"),
                     value_fin=("wedge_fin", "mean"),
                     n=("wedge_tm", "size"),
                     se=("wedge_tm", lambda v: v.std(ddof=1) / np.sqrt(len(v))))
                .reset_index())
        layout, src = "adr0003", f"{root.relative_to(REPO)}/cells/prest_*.csv"
    else:
        tbl = root / "summary_cell.csv"
        _require(tbl.is_file(),
                 f"{root}: neither a cells/ archive nor a pre-ADR "
                 f"summary_cell.csv (the pre-ADR prestige archive is not "
                 f"available in this worktree)")
        d = pd.read_csv(tbl)
        d = d[d.agent_generation_time > 0]
        _require(sorted(d.stock.unique().tolist()) == sorted(STOCKS),
                 f"stock grid {sorted(d.stock.unique())}")
        cell = (d.rename(columns={"agent_generation_time": "g",
                                  "initial_curiosity": "curiosity",
                                  "wedge_tm_mean": "value"})
                .assign(value_fin=lambda x: x.get("wedge_fin_mean"),
                        n=lambda x: 20,
                        se=lambda x: x.get("wedge_tm_se", np.nan))
                [["stock", "g", "curiosity", "landscape", "value",
                  "value_fin", "n", "se"]])
        layout, src = "pre-adr", f"{root.relative_to(REPO)}/summary_cell.csv"
    _require(len(cell) == 54 or (layout == "pre-adr"),
             f"cell table has {len(cell)} rows, expected 54")
    return cell, layout, src


def stock_summary(cell: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for st in STOCKS:
        v = cell[cell.stock == st].value.to_numpy(float)
        n = int(v.size)
        m = float(v.mean())
        ci = Z95 * float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        rows.append({"stock": st, "mean": m, "ci_lo": m - ci, "ci_hi": m + ci,
                     "sd_cells": float(v.std(ddof=1)) if n > 1 else float("nan"),
                     "n_cells": n})
    out = pd.DataFrame(rows)
    out.attrs["sd_max"] = float(out.sd_cells.max())
    return out


def ordering_facts(cell: pd.DataFrame) -> dict:
    """Ordering and rank-correlation facts across the (g, c, landscape) cells."""
    wide = cell.pivot_table(index=["g", "curiosity", "landscape"],
                            columns="stock", values="value")
    ok = ((wide["prestige"] > wide["prestige_visibility"])
          & (wide["prestige_visibility"] > wide["prestige_vanishing"]))
    pairs = {}
    for a, b in (("prestige", "prestige_visibility"),
                 ("prestige", "prestige_vanishing"),
                 ("prestige_visibility", "prestige_vanishing")):
        rho, p = stats.spearmanr(wide[a], wide[b])
        pairs[(a, b)] = (float(rho), float(p))
    return {"n_pos": int(len(wide)), "n_ordered": int(ok.sum()),
            "rho": pairs, "wide": wide}


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def build(outdir="output/report_notebook_v1", data_root=None) -> dict:
    cell, layout, src = load(data_root)
    sm = stock_summary(cell)
    facts = ordering_facts(cell)
    # terminal-step variant (not plotted): the mean ordering is a time-mean
    # fact only, so the figure states where it does not transfer
    fin = cell.groupby("stock").value_fin.agg(["mean", "std"]).reindex(list(STOCKS))
    fin_order = [st for st in fin["mean"].sort_values(ascending=False).index]
    fin_txt = (" > ".join(fin_order)
               + " (" + "/".join(f"{fin.loc[st, 'mean']:+.4f}"
                                 for st in fin_order) + ")")
    g_levels = sorted(cell.g.unique().tolist())

    s = {**T.HOUSE_STYLE, "figsize": (9.9, 6.4)}
    fig = plt.figure(figsize=s["figsize"])
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.40], left=0.175,
                          right=0.985, top=0.855, bottom=0.300, wspace=0.30)

    # ---- left: the matrix ------------------------------------------------
    cols = [(c, land) for c in CURIOSITIES for land in LANDSCAPES]
    rows = [(st, g) for st in STOCKS for g in g_levels]
    mat = np.full((len(rows), len(cols)), np.nan)
    for i, (st, g) in enumerate(rows):
        for j, (c, land) in enumerate(cols):
            sel = cell[(cell.stock == st) & (cell.g == g)
                       & (cell.curiosity == c) & (cell.landscape == land)]
            _require(len(sel) == 1, f"missing cell {st}/{g}/{c}/{land}")
            mat[i, j] = float(sel.value.iloc[0])
    lim = float(np.nanmax(np.abs(mat))) * 1.02
    norm = mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(mat, cmap="RdBu_r", norm=norm, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            lum = abs(v) / lim if lim > 0 else 0.0
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                    fontsize=s["tick_size"] - 2.5,
                    color="white" if lum > 0.55 else "#222222")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([f"c = {c:.1f}\n{land}" for c, land in cols],
                       fontsize=s["tick_size"] - 1.5)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([f"{STOCK_LABEL[st]}  |  g = {g:g}" for st, g in rows],
                       fontsize=s["tick_size"] - 1.5)
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)
    for k in (1, 3, 5):
        ax.axvline(k - 0.5, color="#333333", linewidth=1.2)
    ax.set_title("Time-mean wedge by stock, entry rate and cell",
                 fontsize=s["title_size"])
    ax.set_xlabel("curiosity c and landscape", fontsize=s["label_size"])
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.015)
    cb.ax.tick_params(labelsize=s["tick_size"] - 2)
    cb.set_label("wedge = c_local - c_ideal (time mean)",
                 fontsize=s["tick_size"] - 1)

    # ---- right: stock means ----------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(STOCKS), dtype=float)
    cols_ = [T.PALETTE["purple"], T.PALETTE["green"], T.PALETTE["orange"]]
    for i, st in enumerate(STOCKS):
        r = sm[sm.stock == st].iloc[0]
        ax2.errorbar([x[i]], [r["mean"]],
                     yerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]],
                     fmt="o", color=cols_[i], markersize=7, capsize=4,
                     elinewidth=1.4)
        ax2.text(x[i], r["ci_hi"], f"{r['mean']:+.4f}", ha="center",
                 va="bottom", fontsize=s["tick_size"] - 1.5, color=cols_[i])
    ax2.axhline(0, color=T.PALETTE["gray"], linewidth=1.2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["prestige", "visibility", "vanishing"],
                        fontsize=s["tick_size"] - 1)
    ax2.set_xlim(-0.6, len(STOCKS) - 0.4)
    lo = float(sm.ci_lo.min())
    hi = float(sm.ci_hi.max())
    span = (hi - lo) if hi > lo else 1e-3
    ax2.set_ylim(lo - 0.12 * span, hi + 0.22 * span)
    ax2.set_title("Stock means (18 cells)", fontsize=s["title_size"] - 1)
    ax2.set_ylabel("time-mean wedge", fontsize=s["label_size"])
    ax2.tick_params(labelsize=s["tick_size"])
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    ax2.grid(axis="y", alpha=0.3, linewidth=0.6)
    ax2.set_axisbelow(True)

    order = ", ".join(sm.sort_values("mean", ascending=False).stock.tolist())
    fig.suptitle(
        "Ordering holds at the mean but every stock level is slightly "
        f"negative ({sm[sm.stock == 'prestige']['mean'].iloc[0]:+.4f} / "
        f"{sm[sm.stock == 'prestige_visibility']['mean'].iloc[0]:+.4f} / "
        f"{sm[sm.stock == 'prestige_vanishing']['mean'].iloc[0]:+.4f})",
        fontsize=s["title_size"] - 0.5, y=0.975)

    rho = facts["rho"]
    rho_txt = ", ".join(f"{a.replace('prestige', 'P')}/{b.replace('prestige', 'P')}"
                        f" {r[0]:.2f}" for (a, b), r in rho.items())
    note = textwrap.fill(
        f"source: {src} ({layout}; 3 stocks x 3 c x 2 landscapes x "
        f"{len(g_levels)} g, 20 seeds/cell, "
        f"{int(cell.n.sum())} runs). ADR-0003 pins: 1 step = 0.5 yr, entry age "
        "25, tau_max = 80, gamma = 0.933 (vanishing stock), g in "
        "{" + ",".join(f"{g:g}" for g in g_levels) + "}, H = 200. wedge = "
        "c_local - c_ideal, time mean; right panel = mean of the 18 cell means "
        "with 95% CI over cell means (a design-dispersion interval, not a "
        "sampling CI). Qualification (iii): the mean ordering "
        f"{order} holds, but all three levels are slightly negative (largest "
        f"|mean| {abs(sm['mean']).max():.4f}, cell-mean SD up to "
        f"{sm.attrs['sd_max']:.4f}) and the interval over cells stops short of "
        f"zero only at its upper edge ({sm.ci_hi.max():+.4f}); the ordering "
        f"holds in only {facts['n_ordered']}/{facts['n_pos']} cells, cross-"
        f"stock cell Spearman rho {rho_txt}; on the terminal measure the "
        f"ordering is {fin_txt}, i.e. NOT preserved. A level finding, not a "
        "rank reversal. "
        "Model-internal, NOT calibrated.", width=155)
    fig.text(0.006, 0.012, note, fontsize=s["note_size"],
             color=T.PALETTE["gray"], va="bottom", linespacing=1.45)

    fin_rows = [{
        "figure": BASENAME, "panel": "stock_mean", "metric": "f4_stock_mean_fin",
        "statistic": "terminal_step_mean", "series": st, "stock": st,
        "value": float(fin.loc[st, "mean"]),
        "ci_lo": float(fin.loc[st, "mean"]) - 1.96 * float(fin.loc[st, "std"])
        / np.sqrt(len(cell[cell.stock == st])),
        "ci_hi": float(fin.loc[st, "mean"]) + 1.96 * float(fin.loc[st, "std"])
        / np.sqrt(len(cell[cell.stock == st])),
        "n": int(len(cell[cell.stock == st])), "source_file": src,
        "note": "terminal-step wedge (not plotted)"} for st in STOCKS]
    cell_fin = cell.assign(
        figure=BASENAME, panel="matrix_fin", metric="f4_wedge_cell_fin",
        statistic="mean_terminal_wedge", series=lambda x: x.stock,
        value=lambda x: x.value_fin, source_file=src)[
        ["figure", "panel", "metric", "statistic", "series", "stock", "g",
         "curiosity", "landscape", "value", "n", "source_file"]]
    sidecar = pd.concat([
        cell.assign(figure=BASENAME, panel="matrix", metric="f4_wedge_cell",
                    statistic="mean_time_mean_wedge", series=lambda x: x.stock,
                    source_file=src)[
            ["figure", "panel", "metric", "statistic", "series", "stock", "g",
             "curiosity", "landscape", "value", "n", "source_file"]],
        cell_fin,
        sm.assign(figure=BASENAME, panel="stock_mean", metric="f4_stock_mean",
                  statistic="mean_of_cell_means", series=lambda x: x.stock,
                  source_file=src, value=lambda x: x["mean"],
                  n=lambda x: x.n_cells)[
            ["figure", "panel", "metric", "statistic", "series", "stock",
             "value", "ci_lo", "ci_hi", "n", "source_file"]],
        pd.DataFrame(fin_rows),
    ], ignore_index=True)
    facts_rows = [{
        "figure": BASENAME, "panel": "ordering", "metric": "f4_ordering",
        "statistic": "cells_with_mean_ordering", "series": "P>V>Vn",
        "value": float(facts["n_ordered"]), "ci_lo": float(facts["n_ordered"]),
        "ci_hi": float(facts["n_ordered"]), "n": int(facts["n_pos"]),
        "source_file": src,
        "note": "cells where prestige > prestige_visibility > "
                "prestige_vanishing",
    }]
    for (a, b), (rho, pval) in facts["rho"].items():
        facts_rows.append({
            "figure": BASENAME, "panel": "ordering",
            "metric": "f4_cross_stock_rho", "statistic": "spearman_rho",
            "series": f"{a}|{b}", "value": rho, "ci_lo": rho, "ci_hi": rho,
            "n": int(facts["n_pos"]), "source_file": src,
            "note": f"p = {pval:.3g}"})
    sidecar = pd.concat([sidecar, pd.DataFrame(facts_rows)],
                        ignore_index=True)
    panel_specs = [
        # the matrix is an image: the rendered-value contract cannot see it,
        # so this panel is gated on the collision contract only (the verifier
        # re-derives every annotated value from the cells)
        {"ax": ax, "metric": "f4_wedge_cell", "skip_headroom": True},
        {"ax": ax2, "metric": "f4_stock_mean", "expect_lines": len(STOCKS),
         "points_per_line": 1, "sidecar": sidecar},
    ]
    report = visual_qa.assert_figure_clean(fig, panel_specs)

    out = Path(outdir)
    if not out.is_absolute():
        out = REPO / out
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{BASENAME}.png"
    pdf = out / f"{BASENAME}.pdf"
    fig.savefig(png, dpi=s["dpi"])
    fig.savefig(pdf)
    plt.close(fig)
    csv_path = T.write_figure_data(sidecar, png)
    print(f"QA: {report.get('_total', 0)} failure(s)")
    print(f"Wrote {png} / {pdf} / {csv_path}")
    print(sm.round(5).to_string(index=False))
    print(f"ordering {order} | holds in {facts['n_ordered']}/{facts['n_pos']}"
          f" | rho {rho_txt}")
    return {"png": str(png), "pdf": str(pdf), "data_csv": str(csv_path),
            "qa": report, "layout": layout, "sidecar_rows": len(sidecar)}


if __name__ == "__main__":
    build()
