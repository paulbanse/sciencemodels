"""Attraction-trias 2D heatmap figure (task/attraction-trias-heatmap).

Publication-grade 2x2 heatmap over the ``run_attraction_heatmap.py`` grid
(data/attraction_heatmap_adr0003/, 90 cells x 20 seeds on the ADR-0003 clock):

    rows:    M0 (g=0, exit_mode="none")  /  M1 (g=0.25, fixed_span, tau_max=80)
    columns: Panel A = percentage_knowledge_harvested
             Panel B = time-mean Corr(P_out, M*)  (Mean_of_Corr_prestige_ideal_merit)
    x-axis:  xi = 1 - c in {0.0, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}
    y-axis:  w = prestige_weight_in_visibility in {0.0, 0.25, 0.5, 0.75, 1.0}

Each panel carries contour lines and one computed annotation answering the
panel's structural question: are the iso-lines vertical (the metric tracks xi
only -- proximity-dominated) or tilted by w (past-prestige-modulated)? The
test is an OLS plane fit on normalized axes, Z = a*x_t + b*w_t + c; the
iso-line direction is (b, -a), so its angle from vertical is
theta = atan2(|b|, |a|). theta = 0 deg is exactly vertical (w-inert);
90 deg is horizontal (pure w). The verdict word uses a 10-degree convention.

QA note (disclosed): visual_qa's rendered-artist data contract covers
bar/line/errorbar panels; heatmap meshes and contour collections sit outside
its artist inventory. assert_figure_clean therefore enforces the full
collision-free layout contract on this canvas (0 failures required), while
the data layer is verified here in the builder: full 9x5 cell coverage,
n = 20 seeds per cell, plotted matrix == checkpointed cell means, and the
exact numbers persisted as a sidecar CSV.

NOT CALIBRATED: model-internal only (owner ruling 2026-11-13).

Run:
    MPLBACKEND=Agg /opt/hermes/.venv/bin/python \
        scripts/attraction_heatmap/plot_attraction_heatmap.py
"""
from __future__ import annotations

import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "scripts", "report_notebook"))
import diagram_templates as T  # noqa: E402
import visual_qa  # noqa: E402

CELL_DIR = os.path.join(REPO, "data", "attraction_heatmap_adr0003", "cells")
OUT = os.path.join(REPO, "output", "report_notebook_v1",
                   "F_attraction_trias_heatmap.png")

XI_LEVELS = (0.0, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
W_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
TIERS = ("m0", "m1")
TIER_LABELS = {
    "m0": "M0  (g = 0, exit_mode = none)",
    "m1": "M1  (g = 0.25, fixed_span, $\\tau_{\\max}$ = 80)",
}
METRICS = (
    ("percentage_knowledge_harvested",
     "Panel A: percentage knowledge harvested"),
    ("Mean_of_Corr_prestige_ideal_merit",
     "Panel B: time-mean Corr($P_{out}$, $M^*$)"),
)
VERTICAL_TOL_DEG = 10.0


def load_grid() -> dict:
    """tier -> metric -> 5x9 matrix of 20-seed cell means (+ coverage checks)."""
    out = {(t, m): np.full((len(W_LEVELS), len(XI_LEVELS)), np.nan)
           for t in TIERS for m, _ in METRICS}
    n_ok = 0
    for ti, tier in enumerate(TIERS):
        for xi_i, xi in enumerate(XI_LEVELS):
            for w_i, w in enumerate(W_LEVELS):
                tag = (f"cell_xi{int(round(xi * 100)):03d}"
                       f"_w{int(round(w * 100)):03d}_{tier}.csv")
                path = os.path.join(CELL_DIR, tag)
                assert os.path.exists(path), f"missing checkpoint: {path}"
                df = pd.read_csv(path)
                assert len(df) == 20, f"{tag}: {len(df)} runs, expected 20"
                assert sorted(df.agent_seed.tolist()) == list(range(20)), tag
                for metric, _ in METRICS:
                    out[(tier, metric)][w_i, xi_i] = float(df[metric].mean())
                n_ok += 1
    for key, mat in out.items():
        assert not np.isnan(mat).any(), f"NaN cell in {key}"
    print(f"grid OK: {n_ok} cells x 20 seeds, all values finite")
    return out


def tilt_stats(z: np.ndarray) -> dict:
    """OLS plane on normalized axes -> iso-line angle from vertical (deg)."""
    x_t = (np.asarray(XI_LEVELS) - min(XI_LEVELS)) / (
        max(XI_LEVELS) - min(XI_LEVELS))
    w_t = (np.asarray(W_LEVELS) - min(W_LEVELS)) / (
        max(W_LEVELS) - min(W_LEVELS))
    xx, ww = np.meshgrid(x_t, w_t)
    a, b, c = np.linalg.lstsq(
        np.column_stack([xx.ravel(), ww.ravel(), np.ones(xx.size)]),
        z.ravel(), rcond=None)[0]
    theta = float(np.degrees(np.arctan2(abs(b), abs(a))))
    return {"a": float(a), "b": float(b), "c": float(c), "theta_deg": theta,
            "vertical": theta <= VERTICAL_TOL_DEG}


def build() -> dict:
    grids = load_grid()
    s = {**T.HOUSE_STYLE, "dpi": 180, "title_size": 11, "label_size": 9.5,
         "tick_size": 8.5, "note_size": 7.5}
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.6))

    xx, ww = np.meshgrid(np.asarray(XI_LEVELS), np.asarray(W_LEVELS))
    stats = {}
    for col, (metric, title) in enumerate(METRICS):
        mats = [grids[(t, metric)] for t in TIERS]
        vmin = min(m.min() for m in mats)
        vmax = max(m.max() for m in mats)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        for row, tier in enumerate(TIERS):
            ax = axes[row][col]
            z = grids[(tier, metric)]
            im = ax.pcolormesh(xx, ww, z, cmap="viridis", norm=norm,
                               shading="nearest")
            cs = ax.contour(xx, ww, z, levels=6, colors="white",
                            linewidths=0.8, alpha=0.85)
            ax.clabel(cs, fmt="%.2f", fontsize=6.5, colors="white")
            st = tilt_stats(z)
            stats[(tier, metric)] = st
            verdict = ("vertical: proximity-dominated" if st["vertical"]
                       else f"tilted {st['theta_deg']:.0f}$^\\circ$: "
                            "past-prestige-modulated")
            ax.text(0.03, 0.06,
                    f"iso-lines {st['theta_deg']:.1f}$^\\circ$ from vertical"
                    f"  \u2192  {verdict}",
                    transform=ax.transAxes, fontsize=s["note_size"] + 0.5,
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="black",
                              alpha=0.55, edgecolor="none"))
            ax.set_title(f"{TIER_LABELS[tier]}\n{title}",
                         fontsize=s["title_size"])
            ax.set_xlabel("$\\xi = 1 - c$", fontsize=s["label_size"])
            ax.set_ylabel("w = prestige weight in visibility",
                          fontsize=s["label_size"])
            ax.set_xticks(np.asarray(XI_LEVELS))
            ax.set_xticklabels([f"{x:.1f}" for x in XI_LEVELS])
            ax.set_yticks(np.asarray(W_LEVELS))
            ax.set_yticklabels([f"{w:.2f}" for w in W_LEVELS])
            ax.tick_params(labelsize=s["tick_size"])
            fig.colorbar(im, ax=ax, shrink=0.86, pad=0.02)

    fig.suptitle(
        "Attraction trias on the ADR-0003 clock: the $\\xi$ axis dominates, "
        "past-prestige weighting $w$ tilts but does not replace it",
        fontsize=s["title_size"] + 1, y=0.985)

    n_vert = sum(1 for st in stats.values() if st["vertical"])
    note = (
        "source: data/attraction_heatmap_adr0003/ (90 cells x 20 seeds, "
        "H = 200, N = 20, mgstd5, gamma = 0.933; M1: g = 0.25, fixed_span, "
        "tau_max = 80). Cell means over 20 seeds; contours at 6 levels. "
        "Iso-line angle from vertical = atan2(|b|, |a|) of the OLS plane "
        "Z = a*xi_t + b*w_t + c on normalized axes; 0 deg = vertical "
        "(w-inert, proximity-dominated), 90 deg = horizontal (pure w); "
        f"verdict threshold {VERTICAL_TOL_DEG:.0f} deg. "
        f"Panels vertical: {n_vert}/4. "
        "Model-internal, NOT calibrated."
    )
    fig.text(0.008, 0.012, textwrap.fill(note, width=150),
             fontsize=s["note_size"],
             color=T.PALETTE["gray"], va="bottom", linespacing=1.45)
    fig.tight_layout(rect=(0, 0.075, 1, 0.965))

    # data verification + sidecar (the QA data contract cannot see meshes)
    rows = []
    for tier in TIERS:
        for metric, _ in METRICS:
            z = grids[(tier, metric)]
            for w_i, w in enumerate(W_LEVELS):
                for xi_i, xi in enumerate(XI_LEVELS):
                    rows.append({"metric": metric, "tier": tier,
                                 "xi": xi, "w": w,
                                 "value": float(z[w_i, xi_i]), "n": 20,
                                 "theta_deg": stats[(tier, metric)]["theta_deg"]})
    sidecar = pd.DataFrame(rows)
    assert len(sidecar) == 2 * 2 * 5 * 9, sidecar.shape

    panels = [{"ax": ax, "metric": f"attraction_{row}_{col}"}
              for col in range(2) for row, ax in
              ((0, axes[0][col]), (1, axes[1][col]))]
    report = visual_qa.assert_figure_clean(fig, panels)

    outdir = os.path.dirname(OUT)
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(OUT, dpi=s["dpi"])
    pdf = OUT[:-4] + ".pdf"
    fig.savefig(pdf)
    plt.close(fig)
    csv = T.write_figure_data(sidecar, OUT)

    print(f"QA: {report.get('_total', 0)} failure(s)")
    if report.get("_relieved"):
        print("relieved:", report["_relieved"])
    print(f"Wrote {OUT} / {pdf} / {csv}")
    for (tier, metric), st in sorted(stats.items()):
        print(f"  {tier} {metric}: theta={st['theta_deg']:.2f} deg, "
              f"a={st['a']:+.4f}, b={st['b']:+.4f}, "
              f"{'vertical' if st['vertical'] else 'tilted'}")
    return {"png": OUT, "pdf": pdf, "data_csv": csv, "stats": stats}


if __name__ == "__main__":
    build()
