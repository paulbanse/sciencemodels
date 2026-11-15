"""Path D typology figures (2026-11-15, P-0005): house-style figure module.

ADR-0002 one-home-per-measure: every figure is a function here, drawn only
from data/typology_v1 artefacts, exported as PDF + 300 dpi PNG into
data/typology_v1/figures/ (the commission's figure home; output/reports/
carries only the .md and .pdf per the write boundary).  Every figure
function returns {"name", "paths", "sources"}; main() prints the manifest
(provenance record for the report and later audits).

Source map (before any plotting code):
* fig1 quadrant P-G decay: data/typology_v1/cells/agents/cell_*.csv
  (final-step per-agent prestige / ideal_Merit / current_local_merit,
  pooled per quadrant) + runs.csv (quadrant mean correlations for the
  subtitle);
* fig2 measures by quadrant: data/typology_v1/runs.csv (per-run points,
  120 per quadrant);
* fig3 wedge_fin heatmap, fig4 ideal_loss_cum heatmap: runs.csv grouped
  to the 24 cells.

No house template module exists in the repo (the figures-v1 round's styles
live inside notebooks, which are outside this round's write boundary), so
_style() below is this round's own block, disclosed as such.

Run:
    MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/typology/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "data" / "typology_v1"
FIG_DIR = OUT_DIR / "figures"

# Okabe-Ito colorblind-safe palette, quadrant order fixed by the typology
# table (appearance order: Pure, +Destruction, +Counterforce, Full).
QUADRANT_ORDER = [
    ("Pure Distraction", (0.0, 0.0), "#0072B2"),
    ("Distraction + Destruction", (1.0, 0.0), "#E69F00"),
    ("Distraction + Counterforce", (0.0, 1.0), "#009E73"),
    ("Full Interaction", (1.0, 1.0), "#CC79A7"),
]
QUAD_COLOR = {name: c for name, _, c in QUADRANT_ORDER}

MEASURE_LABELS = {
    "Mean_of_Corr_prestige_ideal_merit": "Corr(prestige, ideal merit), time-mean",
    "Mean_of_Corr_prestige_local_merit": "Corr(prestige, local merit), time-mean",
    "wedge_fin": "wedge_fin = c_ideal - c_local (final step)",
    "ideal_loss_cum": "ideal_loss_cum = -sum(ideal_merit_cum)",
}


def _style():
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.4,
        "axes.axisbelow": True,
    })


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for ext in (".pdf", ".png"):
        p = FIG_DIR / (name + ext)
        fig.savefig(p, bbox_inches="tight")
        paths[ext] = str(p)
    return paths


def _footnote(fig, text, top_frac=0.015):
    import textwrap

    wrapped = textwrap.fill(text, width=118)
    fig.text(0.065, top_frac, wrapped, fontsize=6.3, ha="left", va="bottom",
             color="#333333")


def _quad_subtitle(runs, g, k):
    sub = runs[(runs["gamma"] == g) & (runs["kappa"] == k)]
    return (
        "c_ideal_tm=%.3f  c_local_tm=%.3f  wedge_fin=%.3f  n=%d runs"
        % (
            sub["Mean_of_Corr_prestige_ideal_merit"].mean(),
            sub["Mean_of_Corr_prestige_local_merit"].mean(),
            sub["wedge_fin"].mean(),
            len(sub),
        )
    )


def fig1_quadrant_pg_decay(runs, agents):
    """P-0005 Figure 1: per-quadrant prestige-vs-ideal-merit decay with the
    wedge arrow (mean ideal -> mean local at mean prestige)."""
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.9), sharex=False, sharey=False)
    # pooled hue scale across quadrants (comparability), 1-99 percentile
    # to keep outlier tiles from flattening the map
    hue_all = agents["current_local_merit"].to_numpy(dtype=float)
    vmin, vmax = np.percentile(hue_all, [1, 99])
    for ax, (name, (g, k), _c) in zip(axes.flat, QUADRANT_ORDER):
        sub = agents[(agents["gamma"] == g) & (agents["kappa"] == k)]
        x = sub["prestige"].to_numpy(dtype=float)
        y = sub["ideal_Merit"].to_numpy(dtype=float)
        hue = sub["current_local_merit"].to_numpy(dtype=float)
        sc = ax.scatter(x, y, c=hue, s=5, alpha=0.32, cmap="viridis",
                        edgecolors="none", rasterized=True, vmin=vmin, vmax=vmax)
        xbar, ybar_i, ybar_l = float(np.mean(x)), float(np.mean(y)), float(np.mean(sub["current_local_merit"]))
        ax.annotate("", xy=(xbar, ybar_l), xytext=(xbar, ybar_i),
                    arrowprops=dict(arrowstyle="->", color="#c02626", lw=1.4))
        ax.plot([xbar], [ybar_i], marker="o", ms=4, color="#c02626")
        ax.plot([xbar], [ybar_l], marker="v", ms=4, color="#c02626")
        rsub = runs[(runs["gamma"] == g) & (runs["kappa"] == k)]
        ax.set_title("%s ($\\gamma=%d$, $\\kappa=%d$)\nc_ideal=%.3f  c_local=%.3f  wedge_fin=%.3f"
                     % (name, g, k,
                        rsub["Mean_of_Corr_prestige_ideal_merit"].mean(),
                        rsub["Mean_of_Corr_prestige_local_merit"].mean(),
                        rsub["wedge_fin"].mean()), fontsize=8)
        ax.set_xlabel("prestige (final step)")
        ax.set_ylabel("ideal merit (final step)")
        ax.text(0.03, 0.97, "mean ideal$\\to$local gap: %+.4f" % (ybar_l - ybar_i),
                transform=ax.transAxes, fontsize=7, va="top", color="#c02626")
    cb = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
    cb.set_label("local merit (final step)", fontsize=7.5)
    _footnote(fig,
              "Sources: data/typology_v1/cells/agents/cell_*.csv (final-step per-agent "
              "rows pooled per quadrant; 120 runs per quadrant) and data/typology_v1/runs.csv "
              "(subtitle means). Aggregation: pooled agents, one point per agent-run; no "
              "smoothing. Model-internal experiment (P-0005), NOT calibrated to empirical "
              "data. Destruction: eps_eff = eps * m^gamma; counterforce: eps_i = eps_0 / "
              "(1 + kappa * P_sup).")
    fig.subplots_adjust(left=0.08, right=0.90, top=0.90, bottom=0.17, hspace=0.62, wspace=0.38)
    paths = _save(fig, "fig1_quadrant_pg_decay")
    plt.close(fig)
    return {"name": "fig1_quadrant_pg_decay", "paths": paths,
            "sources": ["data/typology_v1/cells/agents/cell_*.csv", "data/typology_v1/runs.csv"]}


def fig2_measures_by_quadrant(runs):
    """Per-run values of the four commission measures by quadrant."""
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.9))
    rng = np.random.default_rng(7)
    for ax, (src, label) in zip(axes.flat, MEASURE_LABELS.items()):
        for i, (name, (g, k), color) in enumerate(QUADRANT_ORDER):
            sub = runs[(runs["gamma"] == g) & (runs["kappa"] == k)][src].to_numpy(dtype=float)
            xj = i + rng.uniform(-0.16, 0.16, size=len(sub))
            ax.scatter(xj, sub, s=7, alpha=0.35, color=color, edgecolors="none")
            ax.plot([i], [sub.mean()], marker="D", ms=6, color="#222222", zorder=5)
            ax.annotate("%.4f" % sub.mean(), (i, sub.mean()),
                        textcoords="offset points", xytext=(8, 6), fontsize=6.8,
                        color="#222222")
        ax.set_xticks(range(4))
        ax.set_xticklabels(["Pure\nDistraction", "+Destruction", "+Counterforce", "Full\nInteraction"],
                           fontsize=7)
        ax.set_ylabel(label, fontsize=7.5)
        ax.set_xlim(-0.6, 3.6)
        ax.margins(y=0.12)  # headroom for outliers: no clipped points
    axes[0][0].set_title("per-run values (24 cells x 20 seeds = 480 runs; 120 per quadrant)", fontsize=8)
    _footnote(fig,
              "Source: data/typology_v1/runs.csv, one point per run (seed). Diamond: quadrant "
              "mean; annotation: mean value. Corr measures are the model's per-step "
              "prestige-merit correlations time-averaged (Mean_of_ convention); wedge_fin is "
              "the final-step correlation difference c_ideal - c_local; ideal_loss_cum is the "
              "negated cumulative ideal-merit shortfall. Model-internal, not calibrated.")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.17, hspace=0.62, wspace=0.30)
    paths = _save(fig, "fig2_measures_by_quadrant")
    plt.close(fig)
    return {"name": "fig2_measures_by_quadrant", "paths": paths,
            "sources": ["data/typology_v1/runs.csv"]}


def _heatmap(ax, runs, metric, g_entry, title):
    sub = runs[runs["g_entry"] == g_entry]
    quads = [(g, k) for _, (g, k), _ in QUADRANT_ORDER]
    mat = np.full((len(quads), len(CURIOSITIES := (0.3, 0.5, 0.7))), np.nan)
    for i, (g, k) in enumerate(quads):
        for j, c in enumerate(CURIOSITIES):
            cell = sub[(sub["gamma"] == g) & (sub["kappa"] == k) & (sub["curiosity"] == c)]
            if len(cell):
                mat[i, j] = cell[metric].mean()
    im = ax.imshow(mat, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(CURIOSITIES)))
    ax.set_xticklabels(["c=0.3", "c=0.5", "c=0.7"])
    ax.set_yticks(range(len(quads)))
    ax.set_yticklabels(["Pure\nDistraction", "+Destruction", "+Counterforce", "Full\nInteraction"],
                       fontsize=7)
    ax.set_title(title, fontsize=8.5)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, "%.3f" % mat[i, j], ha="center", va="center", fontsize=7,
                    color="#111111")
    ax.grid(False)
    return im


def fig3_wedge_heatmap(runs):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    im0 = _heatmap(axes[0], runs, "wedge_fin", 0.0, "wedge_fin, entry g=0.0\n(kappa columns identical by construction)")
    im1 = _heatmap(axes[1], runs, "wedge_fin", 0.1, "wedge_fin, entry g=0.1")
    fig.colorbar(im1, ax=axes.tolist(), shrink=0.85, pad=0.02, label="mean wedge_fin")
    _footnote(fig,
              "Source: data/typology_v1/runs.csv, cell means of wedge_fin = c_ideal_fin - "
              "c_local_fin (final-step correlations). Left panel: at g=0 no newcomers exist, "
              "so the counterforce has no object and the kappa=0 and kappa=1 rows are "
              "bit-identical (the round's built-in control arm). Model-internal, not calibrated.")
    fig.subplots_adjust(left=0.13, right=0.90, top=0.82, bottom=0.24, wspace=0.35)
    paths = _save(fig, "fig3_wedge_fin_heatmap")
    plt.close(fig)
    return {"name": "fig3_wedge_fin_heatmap", "paths": paths,
            "sources": ["data/typology_v1/runs.csv"]}


def fig4_ideal_loss_heatmap(runs):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    im0 = _heatmap(axes[0], runs, "ideal_loss_cum", 0.0, "ideal_loss_cum, entry g=0.0\n(kappa columns identical by construction)")
    im1 = _heatmap(axes[1], runs, "ideal_loss_cum", 0.1, "ideal_loss_cum, entry g=0.1")
    fig.colorbar(im1, ax=axes.tolist(), shrink=0.85, pad=0.02, label="mean ideal_loss_cum")
    _footnote(fig,
              "Source: data/typology_v1/runs.csv, cell means of ideal_loss_cum = -sum of the "
              "per-agent ideal_merit_cum accumulator over the final population (remeasure-round "
              "convention; more negative = larger cumulative shortfall vs the ideal harvest). "
              "Model-internal, not calibrated.")
    fig.subplots_adjust(left=0.13, right=0.90, top=0.82, bottom=0.24, wspace=0.35)
    paths = _save(fig, "fig4_ideal_loss_cum_heatmap")
    plt.close(fig)
    return {"name": "fig4_ideal_loss_cum_heatmap", "paths": paths,
            "sources": ["data/typology_v1/runs.csv"]}


def build_all():
    _style()
    runs = pd.read_csv(OUT_DIR / "runs.csv")
    agents = pd.concat(
        [pd.read_csv(p) for p in sorted((OUT_DIR / "cells" / "agents").glob("cell_*.csv"))],
        ignore_index=True,
    )
    manifest = [
        fig1_quadrant_pg_decay(runs, agents),
        fig2_measures_by_quadrant(runs),
        fig3_wedge_heatmap(runs),
        fig4_ideal_loss_heatmap(runs),
    ]
    return manifest


def main():
    manifest = build_all()
    for entry in manifest:
        sizes = {e: Path(p).stat().st_size for e, p in entry["paths"].items()}
        print("%s: %s sources=%s" % (entry["name"], sizes, entry["sources"]))
    with open(OUT_DIR / "figures_manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print("FIGURES OK -> data/typology_v1/figures/ (+ figures_manifest.json)")


if __name__ == "__main__":
    main()
