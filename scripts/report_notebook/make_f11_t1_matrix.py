"""F11_v2 - 36-cell T1 Sensitivity Matrix, POST-BUG-0006.

Refresh of ``F11_t1_cell_matrix`` (owner brief 2026-11-19): reads the
re-measured archive ``data/bug0006_remeasure_v2/t1_runs.csv`` (720 runs,
36 cells x 20 seeds) instead of the pre-fix ``data/remeasure_v1/
cell_summary.csv``, grouped by (landscape, vanishing, curiosity,
agent_generation_time). The top row (``harvested_fin``) is invariant to the
fix; the bottom row (``c_ideal_fin``) now shows the post-fix truth: every
``mgstd5`` M1 cell is positive (+0.025 to +0.119, 0/9 negative) and the
negative cells sit only on ``noisypr5`` (5/9 negative).

Scales: top row sequential 0.20-0.80; bottom row ``TwoSlopeNorm`` centred
at 0 over [-0.30, +0.40].

Run:
    MPLBACKEND=Agg /opt/hermes/.venv/bin/python \
        scripts/report_notebook/make_f11_t1_matrix.py
"""
from __future__ import annotations

from pathlib import Path
import sys
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import diagram_templates as T
import visual_qa

OUT_PNG = REPO / "output/report_notebook_v1/F11_v2_t1_cell_matrix.png"
SRC = REPO / "data/bug0006_remeasure_v2/t1_runs.csv"


def load_cells() -> pd.DataFrame:
    """36 cell means from the post-fix run archive (20 seeds per cell)."""
    raw = pd.read_csv(SRC)
    assert len(raw) == 720, f"expected 720 runs, got {len(raw)}"
    keys = ["landscape", "vanishing", "curiosity", "agent_generation_time"]
    sizes = raw.groupby(keys).size()
    assert (sizes == 20).all(), f"seeds per cell: {sizes.unique()}"
    cells = raw.groupby(keys, as_index=False).agg(
        harvested_fin=("harvested_fin", "mean"),
        c_ideal_fin=("c_ideal_fin", "mean"),
        ideal_loss_cum=("ideal_loss_cum", "mean"),
        n_seeds=("seed", "size"),
    )
    assert len(cells) == 36, f"expected 36 cells, got {len(cells)}"

    # brief confirmations: post-fix sign structure
    m1 = cells[cells["agent_generation_time"] == 0.1]
    mg = m1[m1.landscape == "mgstd5"]["c_ideal_fin"]
    nz = m1[m1.landscape == "noisypr5"]["c_ideal_fin"]
    assert len(mg) == 9 and (mg > 0).all(), (
        f"mgstd5 M1 cells must be all positive, got "
        f"{sorted(mg.round(4))}")
    assert len(nz) == 9 and int((nz < 0).sum()) == 5, (
        f"noisypr5 M1 must carry exactly 5 negative cells, got "
        f"{sorted(nz.round(4))}")
    print(f"CONFIRM mgstd5 M1: 0/9 negative, range "
          f"[{mg.min():+.3f}, {mg.max():+.3f}]")
    print(f"CONFIRM noisypr5 M1: {int((nz < 0).sum())}/9 negative, range "
          f"[{nz.min():+.3f}, {nz.max():+.3f}]")
    return cells


def main() -> None:
    t1 = load_cells()
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
                assert len(sub) == 1, (land, van, c, gval, len(sub))
                mat[i, j] = float(sub[col_name].iloc[0])
        return mat

    s = T.HOUSE_STYLE
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.2), sharex=True, sharey=True)

    # Top row: harvested_fin (sequential, 0.20-0.80)
    h_mats = [build_grid("harvested_fin", g) for g, _ in g_vals]
    h_norm = Normalize(vmin=0.20, vmax=0.80)
    for j, ((g, title), mat) in enumerate(zip(g_vals, h_mats)):
        ax = axes[0, j]
        im_h = ax.imshow(mat, cmap="YlGnBu", norm=h_norm, aspect="auto")
        ax.set_title(title, fontsize=s["title_size"], pad=8)
        ax.axhline(2.5, color="white", linewidth=2.0)
        for r in range(mat.shape[0]):
            for c_idx in range(mat.shape[1]):
                val = mat[r, c_idx]
                txt_col = "white" if val > 0.55 else "#222222"
                ax.text(c_idx, r, f"{val*100:.1f}%", ha="center", va="center",
                        fontsize=s["tick_size"], color=txt_col,
                        fontweight="medium")
    axes[0, 0].set_ylabel("harvested knowledge share", fontsize=s["label_size"])

    # Bottom row: c_ideal_fin (TwoSlopeNorm centred at 0, -0.30..+0.40)
    c_mats = [build_grid("c_ideal_fin", g) for g, _ in g_vals]
    norm = TwoSlopeNorm(vmin=-0.30, vcenter=0.0, vmax=0.40)
    for j, ((g, _), mat) in enumerate(zip(g_vals, c_mats)):
        ax = axes[1, j]
        im_c = ax.imshow(mat, cmap="RdBu", norm=norm, aspect="auto")
        ax.axhline(2.5, color="white", linewidth=2.0)
        for r in range(mat.shape[0]):
            for c_idx in range(mat.shape[1]):
                val = mat[r, c_idx]
                txt_col = "white" if abs(val) > 0.24 else "#222222"
                ax.text(c_idx, r, f"{val:+.3f}", ha="center", va="center",
                        fontsize=s["tick_size"], color=txt_col,
                        fontweight="medium")
        ax.set_xticks(range(len(curiosities)))
        ax.set_xticklabels([f"c = {c}" for c in curiosities],
                           fontsize=s["tick_size"])
        ax.set_xlabel("curiosity c", fontsize=s["label_size"])
    axes[1, 0].set_ylabel("Corr(prestige, ideal merit)", fontsize=s["label_size"])

    for r_idx in range(2):
        axes[r_idx, 0].set_yticks(range(len(row_specs)))
        axes[r_idx, 0].set_yticklabels([lbl for _, _, lbl in row_specs],
                                       fontsize=s["tick_size"] - 0.5)
        for c_idx in range(2):
            for spine in axes[r_idx, c_idx].spines.values():
                spine.set_visible(False)

    cbar_h = fig.colorbar(im_h, ax=axes[0, :].ravel().tolist(),
                          fraction=0.025, pad=0.03)
    cbar_h.ax.tick_params(labelsize=s["tick_size"] - 1)
    cbar_c = fig.colorbar(im_c, ax=axes[1, :].ravel().tolist(),
                          fraction=0.025, pad=0.03)
    cbar_c.ax.tick_params(labelsize=s["tick_size"] - 1)

    note = textwrap.fill(
        "source: data/bug0006_remeasure_v2/t1_runs.csv (POST-BUG-0006 "
        "re-measure; 36 cells x 20 seeds = 720 runs, use_visibility_reward "
        "= True). Top: harvested knowledge share (invariant to the fix). "
        "Bottom: final-step Corr(prestige, ideal merit) — post-fix every "
        "mgstd5 M1 cell is positive (+0.025 to +0.119); the negative cells "
        "sit only on noisypr5 (5/9). Heatmap shows the full 36-cell "
        "factorial, matching Figure 6 (F1_v2). Model-internal, NOT "
        "calibrated.",
        width=118,
    )
    fig.text(0.01, 0.015, note, fontsize=s["note_size"],
             color=T.PALETTE["gray"], va="bottom")
    fig.subplots_adjust(left=0.20, right=0.88, top=0.92, bottom=0.24,
                        hspace=0.14, wspace=0.08)

    # Visual QA: heatmaps carry no line/bar series; the contract still runs
    # the collision pass over every cell label and title.
    qa_specs = [{"ax": ax, "expect_bars": 0, "skip_headroom": True}
                for ax in axes.ravel()]
    report = visual_qa.assert_figure_clean(fig, qa_specs)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=s["dpi"])
    fig.savefig(OUT_PNG.with_suffix(".pdf"))
    plt.close(fig)

    t1[["landscape", "vanishing", "curiosity", "agent_generation_time",
        "n_seeds", "harvested_fin", "c_ideal_fin", "ideal_loss_cum"]].to_csv(
        OUT_PNG.with_name("F11_v2_t1_cell_matrix_data.csv"), index=False
    )
    print(f"QA: {report.get('_total', 0)} failure(s)")
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
