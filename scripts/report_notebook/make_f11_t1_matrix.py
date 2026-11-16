"""F11 - 36-cell T1 Sensitivity Matrix: Harvest Share vs. Corr(prestige, ideal merit).

Plots the full 36-cell post-audit T1 factorial (data/remeasure_v1/cell_summary.csv,
origin == 'T1') across M0 (g=0.0) and M1 (g=0.1), showing harvested_fin (top row)
and c_ideal_fin (bottom row) for every (landscape, vanishing, curiosity) cell.
"""
from __future__ import annotations

from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import diagram_templates as T

OUT_PNG = REPO / "output/report_notebook_v1/F11_t1_cell_matrix.png"


def main() -> None:
    df = pd.read_csv(REPO / "data/remeasure_v1/cell_summary.csv")
    t1 = df[df["origin"] == "T1"].copy()
    assert len(t1) == 36, f"Expected 36 T1 cells, got {len(t1)}"

    curiosities = [0.3, 0.5, 0.7]
    row_specs = [
        ("mgstd5", 0.5, "mgstd5  van=0.5"),
        ("mgstd5", 0.9, "mgstd5  van=0.9"),
        ("mgstd5", 1.0, "mgstd5  van=1.0"),
        ("noisypr5", 0.5, "noisypr5  van=0.5"),
        ("noisypr5", 0.9, "noisypr5  van=0.9"),
        ("noisypr5", 1.0, "noisypr5  van=1.0"),
    ]
    g_vals = [(0.0, "M0: Closed Cohort (g = 0.0)"),
              (0.1, "M1: Vertical Lineage (g = 0.1)")]

    def build_grid(col_name: str, gval: float) -> np.ndarray:
        mat = np.zeros((len(row_specs), len(curiosities)), dtype=float)
        for i, (land, van, _) in enumerate(row_specs):
            for j, c in enumerate(curiosities):
                sub = t1[(t1["agent_generation_time"] == gval)
                         & (t1["landscape"] == land)
                         & (t1["vanishing"] == van)
                         & (t1["curiosity"] == c)]
                assert len(sub) == 1
                mat[i, j] = float(sub[col_name].iloc[0])
        return mat

    s = T.HOUSE_STYLE
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.2), sharex=True, sharey=True)

    # Top row: harvested_fin
    h_mats = [build_grid("harvested_fin", g) for g, _ in g_vals]
    h_vmin, h_vmax = 0.20, 0.80
    for j, ((g, title), mat) in enumerate(zip(g_vals, h_mats)):
        ax = axes[0, j]
        im_h = ax.imshow(mat, cmap="YlGnBu", vmin=h_vmin, vmax=h_vmax, aspect="auto")
        ax.set_title(title, fontsize=s["title_size"], pad=8)
        ax.axhline(2.5, color="white", linewidth=2.0)
        for r in range(mat.shape[0]):
            for c_idx in range(mat.shape[1]):
                val = mat[r, c_idx]
                txt_col = "white" if val > 0.55 else "#222222"
                ax.text(c_idx, r, f"{val*100:.1f}%", ha="center", va="center",
                        fontsize=s["tick_size"], color=txt_col, fontweight="medium")
    axes[0, 0].set_ylabel("harvested knowledge share", fontsize=s["label_size"])

    # Bottom row: c_ideal_fin
    c_mats = [build_grid("c_ideal_fin", g) for g, _ in g_vals]
    norm = TwoSlopeNorm(vmin=-0.40, vcenter=0.0, vmax=0.40)
    for j, ((g, _), mat) in enumerate(zip(g_vals, c_mats)):
        ax = axes[1, j]
        im_c = ax.imshow(mat, cmap="RdBu", norm=norm, aspect="auto")
        ax.axhline(2.5, color="white", linewidth=2.0)
        for r in range(mat.shape[0]):
            for c_idx in range(mat.shape[1]):
                val = mat[r, c_idx]
                txt_col = "white" if abs(val) > 0.24 else "#222222"
                ax.text(c_idx, r, f"{val:+.3f}", ha="center", va="center",
                        fontsize=s["tick_size"], color=txt_col, fontweight="medium")
        ax.set_xticks(range(len(curiosities)))
        ax.set_xticklabels([f"c = {c}" for c in curiosities], fontsize=s["tick_size"])
        ax.set_xlabel("curiosity c", fontsize=s["label_size"])
    axes[1, 0].set_ylabel("Corr(prestige, ideal merit)", fontsize=s["label_size"])

    for r_idx in range(2):
        axes[r_idx, 0].set_yticks(range(len(row_specs)))
        axes[r_idx, 0].set_yticklabels([lbl for _, _, lbl in row_specs],
                                       fontsize=s["tick_size"] - 0.5)
        for c_idx in range(2):
            for spine in axes[r_idx, c_idx].spines.values():
                spine.set_visible(False)

    # Colorbars on right
    cbar_h = fig.colorbar(im_h, ax=axes[0, :].ravel().tolist(), fraction=0.025, pad=0.03)
    cbar_h.ax.tick_params(labelsize=s["tick_size"] - 1)
    cbar_c = fig.colorbar(im_c, ax=axes[1, :].ravel().tolist(), fraction=0.025, pad=0.03)
    cbar_c.ax.tick_params(labelsize=s["tick_size"] - 1)

    import textwrap
    note = textwrap.fill(
        "source: data/remeasure_v1/cell_summary.csv (origin = T1, 36 cells x 20 seeds = 720 runs; "
        "use_visibility_reward = True). Top: harvested knowledge share (M0 vs. M1). "
        "Bottom: final-step Corr(prestige, ideal merit) flips negative in 17/18 M1 cells. "
        "Model-internal, NOT calibrated.",
        width=118,
    )
    fig.text(0.01, 0.015, note, fontsize=s["note_size"], color=T.PALETTE["gray"], va="bottom")
    fig.subplots_adjust(left=0.17, right=0.88, top=0.92, bottom=0.14, hspace=0.14, wspace=0.08)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=s["dpi"])
    fig.savefig(OUT_PNG.with_suffix(".pdf"))
    plt.close(fig)

    t1[["agent_generation_time", "landscape", "vanishing", "curiosity",
        "harvested_fin", "ideal_loss_cum", "c_ideal_fin"]].to_csv(
        OUT_PNG.with_name("F11_t1_cell_matrix_data.csv"), index=False
    )
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
