"""F11_v2 - the ``T1`` sensitivity matrix on the ``ADR-0003`` clock.

Re-based (paper v3.5) on ``data/sensitivity_adr0003/cells``: 18 cells x 20
seeds = 360 runs, a 3 x 3 x 2 factorial over curiosity c in {0.3, 0.5, 0.7},
entry rate g in {0.125, 0.25, 0.5} and landscape in {mgstd5, noisypr5}
(``fixed_span`` replacement, ``tau_max = 80``, ``H = 200``).

The previous version drew 36 cells (720 runs) as a 6 x 3 grid split by
entry regime: vanishing tier x curiosity on the rows/columns, with the
``g = 0`` (M0) and ``g = 0.1`` (M1) archives as the two columns. The ADR-0003
T1 arm has neither the vanishing tier nor a ``g = 0`` control, so the matrix
is honestly re-laid out: columns = landscape, rows = g, columns-inside-panel
= curiosity. Every panel is therefore a full 3 x 3 grid of real cells; no
cell is interpolated or carried over, and the note states the reduction.

Scales are computed from the cells (sequential for the harvest share, a
two-slope normal centred at zero for the terminal correlation).

Run:
    MPLBACKEND=Agg /opt/hermes/.venv/bin/python \
        scripts/report_notebook/make_f11_t1_matrix.py
"""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import diagram_templates as T
import visual_qa

BASENAME = "F11_v2_t1_cell_matrix"
DATA_ROOT = REPO / "data" / "sensitivity_adr0003"

GS = [0.125, 0.25, 0.5]
CURIOSITY = [0.3, 0.5, 0.7]
LANDSCAPES = ["mgstd5", "noisypr5"]
TAU_MAX, HORIZON = 80, 200

M_HARV = "harvested knowledge share (final step)"
M_CIDEAL = "Corr(prestige, ideal merit), final step"

ADR_PINS = ("ADR-0003 clock: 1 step = 0.5 yr, entry age 25, tau_max 80 "
            "(40-yr careers), gamma = 0.5**(1/10) = 0.933, g in "
            "{0.125, 0.25, 0.5}, exit_mode fixed_span, H = 200")


def data_root_path(data_root=None) -> Path:
    return Path(data_root) if data_root is not None else DATA_ROOT


def load_cells(data_root=None) -> pd.DataFrame:
    """18 cell means from the ADR-0003 T1 archive (20 seeds per cell)."""
    root = data_root_path(data_root)
    files = sorted((root / "cells").glob("*.csv"))
    assert len(files) == 18, f"T1 gross cells=18 expected, found {len(files)}"
    frames = []
    for f in files:
        d = pd.read_csv(f)
        assert len(d) == 20, f"{f.name}: 20 seed rows expected, found {len(d)}"
        frames.append(d)
    raw = pd.concat(frames, ignore_index=True)
    assert len(raw) == 360, f"expected 360 runs, got {len(raw)}"
    keys = ["landscape", "g", "curiosity"]
    sizes = raw.groupby(keys).size()
    assert (sizes == 20).all(), f"seeds per cell: {sizes.unique()}"
    assert sorted(raw["landscape"].unique().tolist()) == LANDSCAPES, \
        "T1 landscape grid"
    assert sorted(raw["g"].unique().tolist()) == GS, "T1 g grid"
    assert sorted(raw["curiosity"].unique().tolist()) == CURIOSITY, \
        "T1 curiosity grid"
    assert raw["exit_mode"].unique().tolist() == ["fixed_span"], \
        "T1 exit_mode must be pinned to fixed_span"
    assert raw["tau_max"].unique().tolist() == [TAU_MAX], f"tau_max {TAU_MAX}"
    assert raw["horizon"].unique().tolist() == [HORIZON], f"H {HORIZON}"
    cells = raw.groupby(keys, as_index=False).agg(
        harvested_fin=("harvested_fin", "mean"),
        c_ideal_fin=("c_ideal_fin", "mean"),
        ideal_loss_cum=("ideal_loss_cum", "mean"),
        n_seeds=("seed", "size"),
    )
    assert len(cells) == 18, f"expected 18 cells, got {len(cells)}"

    # structural claims the two rows carry
    cid = cells["c_ideal_fin"]
    mg = cells.loc[cells["landscape"] == "mgstd5", "c_ideal_fin"]
    nz = cells.loc[cells["landscape"] == "noisypr5", "c_ideal_fin"]
    assert (mg > 0).all(), f"mgstd5 terminal cells must be positive, got {sorted(mg.round(4))}"
    assert int((nz < 0).sum()) > 0, "noisypr5 must carry negative terminal cells"
    print(f"CONFIRM mgstd5 terminal cells: 0/{len(mg)} negative, range "
          f"[{mg.min():+.3f}, {mg.max():+.3f}]")
    print(f"CONFIRM noisypr5 terminal cells: {int((nz < 0).sum())}/{len(nz)} "
          f"negative, range [{nz.min():+.3f}, {nz.max():+.3f}]")
    print(f"CONFIRM harvested share range [{cells['harvested_fin'].min():.3f}, "
          f"{cells['harvested_fin'].max():.3f}] over {len(cid)} cells")
    return cells


def build_grid(cells: pd.DataFrame, col: str, land: str) -> np.ndarray:
    mat = np.zeros((len(GS), len(CURIOSITY)), dtype=float)
    for i, gv in enumerate(GS):
        for j, c in enumerate(CURIOSITY):
            sub = cells[(cells["landscape"] == land)
                        & np.isclose(cells["g"], gv)
                        & np.isclose(cells["curiosity"], c)]
            assert len(sub) == 1, (land, gv, c, len(sub))
            mat[i, j] = float(sub[col].iloc[0])
    return mat


def build(outdir="output/report_notebook_v1", data_root=None,
          collector: dict | None = None) -> dict:
    out = Path(outdir)
    if not out.is_absolute():
        out = REPO / out
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{BASENAME}.png"
    pdf = out / f"{BASENAME}.pdf"

    cells = load_cells(data_root)
    src = "data/sensitivity_adr0003/cells/*.csv"

    s = T.HOUSE_STYLE
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.6), sharex=True, sharey=True)

    # --- scales computed from the cells --------------------------------
    h_all = cells["harvested_fin"]
    h_norm = Normalize(vmin=float(h_all.min()), vmax=float(h_all.max()))
    c_all = cells["c_ideal_fin"]
    c_norm = TwoSlopeNorm(vmin=float(min(c_all.min(), -1e-6)), vcenter=0.0,
                          vmax=float(max(c_all.max(), 1e-6)))

    rows = []
    for i, (col, metric) in enumerate((("harvested_fin", M_HARV),
                                      ("c_ideal_fin", M_CIDEAL))):
        for j, land in enumerate(LANDSCAPES):
            ax = axes[i, j]
            mat = build_grid(cells, col, land)
            norm = h_norm if col == "harvested_fin" else c_norm
            cmap = "YlGnBu" if col == "harvested_fin" else "RdBu"
            im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
            if i == 0:
                ax.set_title(land, fontsize=s["title_size"], pad=6)
            for r in range(mat.shape[0]):
                for c_idx in range(mat.shape[1]):
                    val = mat[r, c_idx]
                    if col == "harvested_fin":
                        frac = (val - h_norm.vmin) / (h_norm.vmax - h_norm.vmin)
                        txt_col = "white" if frac > 0.62 else "#222222"
                        label = f"{val * 100:.1f}%"
                    else:
                        span = max(abs(c_norm.vmin), abs(c_norm.vmax))
                        txt_col = "white" if abs(val) > 0.55 * span else "#222222"
                        label = f"{val:+.3f}"
                    ax.text(c_idx, r, label, ha="center", va="center",
                            fontsize=s["tick_size"], color=txt_col,
                            fontweight="medium")
                    rows.append(dict(
                        figure=BASENAME,
                        panel="harvest" if col == "harvested_fin" else "c_ideal",
                        metric=metric, landscape=land, g=GS[r],
                        curiosity=CURIOSITY[c_idx], value=float(val),
                        n_seeds=20, source_file=src,
                        note="cell mean over the 20 seeds"))
            ax.set_xticks(range(len(CURIOSITY)))
            ax.set_yticks(range(len(GS)))
            if i == 1:
                ax.set_xticklabels([f"{c:g}" for c in CURIOSITY],
                                   fontsize=s["tick_size"])
                ax.set_xlabel("curiosity c", fontsize=s["label_size"])
            if j == 0:
                ax.set_yticklabels([f"{g:g}" for g in GS],
                                   fontsize=s["tick_size"])
                ax.set_ylabel("entry rate g", fontsize=s["label_size"])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if i == 0:
                im_h = im
            else:
                im_c = im

    # row labels on the far left, one per metric row
    axes[0, 0].annotate(M_HARV, xy=(-0.30, 0.5), xycoords="axes fraction",
                        ha="center", va="center", rotation=90,
                        fontsize=s["label_size"], annotation_clip=False)
    axes[1, 0].annotate(M_CIDEAL, xy=(-0.30, 0.5), xycoords="axes fraction",
                        ha="center", va="center", rotation=90,
                        fontsize=s["label_size"], annotation_clip=False)

    cbar_h = fig.colorbar(im_h, ax=axes[0, :].ravel().tolist(),
                          fraction=0.025, pad=0.03)
    cbar_h.ax.tick_params(labelsize=s["tick_size"] - 1)
    cbar_c = fig.colorbar(im_c, ax=axes[1, :].ravel().tolist(),
                          fraction=0.025, pad=0.03)
    cbar_c.ax.tick_params(labelsize=s["tick_size"] - 1)

    n_neg_mg = int((cells.loc[cells["landscape"] == "mgstd5",
                              "c_ideal_fin"] < 0).sum())
    n_neg_nz = int((cells.loc[cells["landscape"] == "noisypr5",
                              "c_ideal_fin"] < 0).sum())
    per_land = len(GS) * len(CURIOSITY)
    note = [
        textwrap.fill(
            f"source: data/sensitivity_adr0003/cells ({len(cells)} cells x 20 "
            f"seeds = {len(cells) * 20} runs; curiosity 0.3/0.5/0.7 x "
            f"landscape x g). {ADR_PINS}. Top row: harvested knowledge share, "
            f"colour scale {h_norm.vmin:.3f}-{h_norm.vmax:.3f} from the cells. "
            f"Bottom row: final-step Corr(prestige, ideal merit), two-slope "
            f"colour scale centred at 0 over {c_norm.vmin:+.3f} to "
            f"{c_norm.vmax:+.3f}: every mgstd5 cell is positive "
            f"({n_neg_mg}/{per_land} negative) and the negative cells sit on "
            f"noisypr5 ({n_neg_nz}/{per_land}).", width=112),
        textwrap.fill(
            f"Reduction vs the pre-ADR matrix: 18 cells (360 runs) instead of "
            f"36 (720). The ADR-0003 T1 arm drops the vanishing tier and the "
            f"g = 0 control, so the old 6 x 3 grid split by entry regime is "
            f"replaced by columns = landscape and rows = g, with the "
            f"curiosity levels inside each panel. No cell is carried over or "
            f"interpolated. Model-internal, NOT calibrated.", width=112),
    ]
    lines: list[str] = []
    for block in note:
        lines.extend(block.split("\n"))
    top, floor = 0.168, 0.012
    step = min(0.019, (top - floor) / max(len(lines), 1))
    y = top
    for line in lines:
        fig.text(0.010, y, line, fontsize=s["note_size"] - 1.4,
                 color=T.PALETTE["gray"])
        y -= step
    fig.subplots_adjust(left=0.175, right=0.90, top=0.928, bottom=0.255,
                        hspace=0.16, wspace=0.10)

    qa_specs = [{"ax": ax, "expect_bars": 0, "expect_lines": 0,
                 "skip_headroom": True} for ax in axes.ravel()]
    report = visual_qa.assert_figure_clean(fig, qa_specs)

    fig.savefig(png, dpi=s["dpi"])
    fig.savefig(pdf)

    sidecar = pd.DataFrame(rows, columns=[
        "figure", "panel", "metric", "landscape", "g", "curiosity", "value",
        "n_seeds", "source_file", "note"])
    ann = pd.DataFrame([
        dict(figure=BASENAME, panel="note", metric="negative cells (mgstd5)",
             landscape="mgstd5", g=np.nan, curiosity=np.nan,
             value=float(n_neg_mg), n_seeds=np.nan, source_file=src,
             note=f"{n_neg_mg}/{per_land} terminal cells negative"),
        dict(figure=BASENAME, panel="note", metric="negative cells (noisypr5)",
             landscape="noisypr5", g=np.nan, curiosity=np.nan,
             value=float(n_neg_nz), n_seeds=np.nan, source_file=src,
             note=f"{n_neg_nz}/{per_land} terminal cells negative"),
        dict(figure=BASENAME, panel="note", metric="cells", landscape="",
             g=np.nan, curiosity=np.nan, value=float(len(cells)),
             n_seeds=20.0, source_file=src,
             note="18 cells x 20 seeds = 360 runs (pre-ADR: 36 x 20 = 720)"),
    ], columns=sidecar.columns)
    sidecar = pd.concat([sidecar, ann], ignore_index=True)
    csv_path = T.write_figure_data(sidecar, png)
    if collector is not None:
        collector["fig"] = fig
    plt.close(fig)
    print(f"QA: {report.get('_total', 0)} failure(s)")
    print(f"Wrote {png} | {pdf} | {csv_path}")
    return {"png": str(png), "pdf": str(pdf), "data_csv": str(csv_path),
            "qa": report, "sidecar_rows": len(sidecar)}


def verify(outdir="output/report_notebook_v1", data_root=None,
           tol: float = 1e-9) -> list[str]:
    """Recompute the matrices and compare with the published sidecar."""
    fails: list[str] = []
    out = Path(outdir)
    if not out.is_absolute():
        out = REPO / out
    csv_path = out / f"{BASENAME}_data.csv"
    cells = load_cells(data_root)
    if not csv_path.exists():
        return [f"sidecar missing: {csv_path}"]
    got = pd.read_csv(csv_path)
    got = got[got["panel"].isin(["harvest", "c_ideal"])]
    if len(got) != 2 * len(cells):
        fails.append(f"sidecar cell rows: {2 * len(cells)} expected, "
                     f"found {len(got)}")
        return fails
    for panel, col in (("harvest", "harvested_fin"),
                       ("c_ideal", "c_ideal_fin")):
        for land in LANDSCAPES:
            mat = build_grid(cells, col, land)
            sub = got[(got["panel"] == panel) & (got["landscape"] == land)]
            if len(sub) != mat.size:
                fails.append(f"{panel}/{land}: {mat.size} cells expected, "
                             f"found {len(sub)}")
                continue
            for i, gv in enumerate(GS):
                for j, c in enumerate(CURIOSITY):
                    row = sub[np.isclose(sub["g"], gv)
                              & np.isclose(sub["curiosity"], c)]
                    if len(row) != 1:
                        fails.append(f"{panel}/{land}/g={gv}/c={c}: no unique "
                                     f"row")
                        continue
                    v = float(row["value"].iloc[0])
                    if abs(v - float(mat[i, j])) > tol:
                        fails.append(f"{panel}/{land}/g={gv}/c={c}: sidecar "
                                     f"{v!r} != recomputed {float(mat[i, j])!r}")
    # rendered QA on a fresh canvas
    tmp = Path(tempfile.mkdtemp(prefix="f11_"))
    col: dict = {}
    res = build(tmp, data_root=data_root, collector=col)
    if res["qa"].get("_total", 0) != 0:
        fails.append(f"visual_qa re-run reported {res['qa']['_total']} "
                     f"failure(s)")
    regot = pd.read_csv(tmp / f"{BASENAME}_data.csv")
    regot = regot[regot["panel"].isin(["harvest", "c_ideal"])]
    key = ["panel", "landscape", "g", "curiosity"]
    j = got.merge(regot, on=key, suffixes=("_a", "_b"))
    if len(j) != len(got):
        fails.append("regenerated sidecar rows do not pair 1:1")
    d = np.abs(j["value_a"].to_numpy(float) - j["value_b"].to_numpy(float))
    if not np.all(np.nan_to_num(d) <= tol):
        fails.append("regenerated sidecar values differ from the rebuild")
    return fails


def main() -> int:
    build()
    fails = verify()
    if fails:
        for f in fails:
            print(f"FAIL {f}")
        return 1
    print("verify_F11_v2: all checks passed (18 cells x 2 metrics reproduced "
          "from the cells; visual_qa re-run clean)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
