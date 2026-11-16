"""F6 - Two-Stage Scale of Proxy Failure (Section 2.3, drafts/paper_sketch_v1.md).

Two panels, one figure, both across the full visibility axis xi = 1 - c in
[0.1, 0.9]:

  (a) Macro efficiency: terminal harvest share
  (b) Micro signal alignment: terminal Corr(prestige, ideal_merit)

Series in each panel:
  M0 MultG2 / MultG6   closed cohort, g = 0, 11 c x 10 seeds
  M1 Vertical Lineage  g = 0.1, xi = 0.1..0.9
  M1 + Turnover        age_hazard, H = 400, entry = 0.1, xi = 0.1..0.9
  M2 (lambda=0 / 1)    dyadic_move_pull = 1.0, logit, equal_split, H = 200,
                       entry = 0.1, xi = 0.1..0.9

Sources (read-only):
  data/figures_v1/run_number2.csv            MultG2 curiosity sweep (11 x 10)
  data/figures_v1/run_number4.csv            MultG6 curiosity sweep (11 x 10)
  data/remeasure_v1/w_v05_summary.csv        M0 / M1 terminal moments
  data/turnover_v1/cells/*.csv               M1 + turnover, xi = 0.3/0.5/0.7
  data/fig1_extension_v1/cells/m1_*.csv      M1 + turnover, xi = 0.1/0.9  (new)
  data/path_b_v2/cells/*.csv                 M2, xi = 0.5
  data/fig1_extension_v1/cells/m2_*.csv      M2, xi = 0.1/0.3/0.7/0.9    (new)

The two `fig1_extension_v1` legs were run through the archived round drivers
themselves (`run_turnover_v1.run_one`, `path_b_v2_common.build_params`), with a
seed-level control arm that reproduces one archived cell of each round
bit-exactly (max abs diff 0.0; see that round's manifest.json). So the only
difference between the xi points is the axis value.

Declared pins (unspecified levels, pinned canonical per superset-and-disclose):
  right panel landscape = mgstd5 (= MultG2 bundle), vanishing 0.9, visibility
  reward on; turnover leg entry = 0.1 (pairs with M1's g = 0.1); M2 arm is the
  ruled max-coupling cell (pull 1.0, logit, equal_split).

Style: house template (scripts/report_notebook/diagram_templates.py, copied
byte-identical beside this module), PALETTE + _cosmetics, dpi=180 PNG.
Model-internal; NOT calibrated (owner ruling 2026-11-13).

Writes only output/report_notebook_v1/F6_two_stage_proxy_failure.{png,pdf} and
F6_two_stage_proxy_failure_data.csv (plus this module's own console manifest).
"""
from __future__ import annotations

import glob
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
import sys

sys.path.insert(0, str(HERE))
import diagram_templates as dt
from diagram_templates import PALETTE, HOUSE_STYLE, _cosmetics, write_figure_data

OUT_PNG = HERE / "F6_two_stage_proxy_failure.png"
FAIL = dt._FAIL_COLOR  # house convention: the pathological regime reads red

MACRO_SRC = {2: "MultG2", 4: "MultG6"}
TURNOVER_PIN = {"exit_mode": "age_hazard", "entry": 0.1, "landscape": "mgstd5"}
M2_PIN = {"dyadic_move_pull": 1.0, "formulation": "logit",
          "credit_sharing_axis": "equal_split", "landscape": "mgstd5"}
XI_FULL = (0.1, 0.3, 0.5, 0.7, 0.9)
SEEDS_PER_CELL = 20


# ---------------------------------------------------------------- extraction
def _turnover_cells() -> pd.DataFrame:
    """M1 + turnover per-run rows for the pinned cell, all five xi points.

    Three interior points come from the archived turnover round, the two ends
    from this round's extension. Both are read as RUN-LEVEL cells (20 rows
    each) so every point on the curve carries the same n and the same
    aggregation, instead of mixing a pooled table with raw cells.
    """
    frames = []
    archived = sorted(glob.glob(str(REPO / "data/turnover_v1/cells/cell_*.csv")))
    for f in archived:
        d = pd.read_csv(f)
        d = d[(d.exit_mode == TURNOVER_PIN["exit_mode"])
              & np.isclose(d.entry, TURNOVER_PIN["entry"])
              & (d.landscape == TURNOVER_PIN["landscape"])]
        if len(d):
            d = d.assign(source_file=str(Path(f).relative_to(REPO)))
            frames.append(d)
    for f in sorted(glob.glob(str(REPO / "data/fig1_extension_v1/cells/m1_*.csv"))):
        d = pd.read_csv(f)
        d = d[(d.exit_mode == TURNOVER_PIN["exit_mode"])
              & np.isclose(d.entry, TURNOVER_PIN["entry"])
              & (d.landscape == TURNOVER_PIN["landscape"])]
        d = d.assign(source_file=str(Path(f).relative_to(REPO)))
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    # orientation proof: exactly the five axis points, 20 runs each
    counts = out.groupby(out.curiosity.round(2)).size()
    assert list(counts.index) == [0.1, 0.3, 0.5, 0.7, 0.9], counts.to_dict()
    assert (counts == SEEDS_PER_CELL).all(), counts.to_dict()
    assert out.horizon.nunique() == 1 and int(out.horizon.iloc[0]) == 400
    return out


def _m2_cells() -> pd.DataFrame:
    """M2 per-run rows for the pinned cell, five xi points x two lambda arms."""
    frame_list = []
    for f in sorted(glob.glob(str(REPO / "data/path_b_v2/cells/cell_*.csv"))):
        d = pd.read_csv(f)
        d = d[(d.landscape == M2_PIN["landscape"])
              & np.isclose(d.dyadic_move_pull, M2_PIN["dyadic_move_pull"])
              & (d.formulation == M2_PIN["formulation"])
              & (d.credit_sharing_axis == M2_PIN["credit_sharing_axis"])
              & (d.dyadic_lambda.isin([0.0, 1.0]))]
        if len(d):
            frame_list.append(d.assign(
                source_file=str(Path(f).relative_to(REPO)),
                curiosity=np.nan))     # not a column of the v2 cells: set below
    for f in sorted(glob.glob(str(REPO / "data/fig1_extension_v1/cells/m2_*.csv"))):
        d = pd.read_csv(f)
        d = d.assign(source_file=str(Path(f).relative_to(REPO)))
        # the extended cells carry xi; recover curiosity for a single axis key
        d["curiosity"] = 1.0 - d["xi"]
        frame_list.append(d)
    out = pd.concat(frame_list, ignore_index=True)
    # the archived c=0.5 cells predate the curiosity column; their pin is 0.5
    out["curiosity"] = out["curiosity"].fillna(0.5)
    out["xi"] = 1.0 - out["curiosity"]
    counts = out.groupby([out.curiosity.round(2), out.dyadic_lambda]).size()
    assert len(counts) == len(XI_FULL) * 2, counts.to_dict()
    assert (counts == SEEDS_PER_CELL).all(), counts.to_dict()
    return out


def macro_points() -> pd.DataFrame:
    """Terminal harvest share per xi for every macro series."""
    rows = []
    for n, fam in MACRO_SRC.items():
        df = pd.read_csv(REPO / f"data/figures_v1/run_number{n}.csv")
        # orientation proof: shape, curvature axis, closed cohort, family tag
        assert len(df) == 110, (fam, len(df))
        assert sorted(df.initial_curiosity.unique()) == [i / 10 for i in range(11)]
        assert (df.agent_generation_time == 0).all(), fam
        expect = "number_gaussians': 2" if fam == "MultG2" else "number_gaussians': 6"
        assert df.generation_params.str.contains(expect).all(), fam
        g = (df.groupby("initial_curiosity")["percentage_knowledge_harvested"]
             .agg(["mean", "std", "count"]).reset_index())
        for _, r in g.iterrows():
            rows.append(dict(
                panel="L", series=fam, c=float(r.initial_curiosity),
                xi=1.0 - float(r.initial_curiosity),
                value=float(r["mean"]), sd=float(r["std"]), n=int(r["count"]),
                metric="percentage_knowledge_harvested (terminal)",
                source_file=f"data/figures_v1/run_number{n}.csv"))
    # M1 macro efficiency (mgstd5, g=0.1, xi=0.1..0.9)
    w = pd.read_csv(REPO / "data/remeasure_v1/w_v05_summary.csv")
    m1 = w[(w.landscape == "mgstd5") & (w.use_visibility_reward == True)  # noqa: E712
           & (w.agent_generation_time == 0.1)]
    for _, r in m1.iterrows():
        rows.append(dict(
            panel="L", series="M1 lineage",
            c=float(r.curiosity), xi=1.0 - float(r.curiosity),
            value=float(r.harvested_fin), sd=np.nan, n=int(r.n_seeds),
            metric="harvested_knowledge_share (terminal)",
            source_file="data/remeasure_v1/w_v05_summary.csv"))
    # M1 + turnover, five xi points from run-level cells
    for c, d in _turnover_cells().groupby("curiosity"):
        rows.append(dict(
            panel="L", series="M1 + Turnover (age_hazard)",
            c=float(c), xi=1.0 - float(c),
            value=float(d.harvested_fin.mean()),
            sd=float(d.harvested_fin.std()), n=int(len(d)),
            metric="harvested_knowledge_share (terminal)",
            source_file=" + ".join(sorted(d.source_file.unique()))))
    # M2, five xi points x two lambda arms
    m2 = _m2_cells()
    for lam, tag in ((0.0, "M2 (lambda=0)"), (1.0, "M2 (lambda=1)")):
        for c, d in m2[m2.dyadic_lambda == lam].groupby("curiosity"):
            rows.append(dict(
                panel="L", series=tag, c=float(c), xi=1.0 - float(c),
                value=float(d.percentage_knowledge_harvested.mean()),
                sd=float(d.percentage_knowledge_harvested.std()), n=int(len(d)),
                metric="harvested_knowledge_share (terminal)",
                source_file=" + ".join(sorted(d.source_file.unique()))))
    return pd.DataFrame(rows)


def micro_points() -> pd.DataFrame:
    """Terminal c_ideal per xi for M0, M1, M1 + turnover and both M2 arms."""
    rows = []
    w = pd.read_csv(REPO / "data/remeasure_v1/w_v05_summary.csv")
    sel = w[(w.landscape == "mgstd5") & (w.use_visibility_reward == True)]  # noqa: E712
    assert len(sel) == 10, f"expected 10 mgstd5 cells in w_v05_summary, got {len(sel)}"
    names = {0.0: "M0 Closed Cohort", 0.1: "M1 Vertical Lineage"}
    for _, r in sel.iterrows():
        rows.append(dict(
            panel="R", series=names[float(r.agent_generation_time)],
            c=float(r.curiosity), xi=1.0 - float(r.curiosity),
            value=float(r.c_ideal_fin), sd=np.nan, n=int(r.n_seeds),
            metric="Corr_prestige_ideal_merit (terminal)",
            source_file="data/remeasure_v1/w_v05_summary.csv"))
    for c, d in _turnover_cells().groupby("curiosity"):
        rows.append(dict(
            panel="R", series="M1 + Turnover (age_hazard)",
            c=float(c), xi=1.0 - float(c),
            value=float(d.c_ideal_fin.mean()), sd=float(d.c_ideal_fin.std()),
            n=int(len(d)), metric="Corr_prestige_ideal_merit (terminal)",
            source_file=" + ".join(sorted(d.source_file.unique()))))
    m2 = _m2_cells()
    for lam, tag in ((0.0, "M2 (lambda=0)"), (1.0, "M2 (lambda=1)")):
        for c, d in m2[m2.dyadic_lambda == lam].groupby("curiosity"):
            rows.append(dict(
                panel="R", series=tag, c=float(c), xi=1.0 - float(c),
                value=float(d.Mean_of_Corr_prestige_ideal_merit.mean()),
                sd=float(d.Mean_of_Corr_prestige_ideal_merit.std()), n=int(len(d)),
                metric="Corr(prestige, ideal merit), final step",
                source_file=" + ".join(sorted(d.source_file.unique()))))
    return pd.DataFrame(rows)


def wedge_points() -> pd.DataFrame:
    """Terminal relational wedge (corr(prest,local) - corr(prest, ideal)) per xi."""
    rows = []
    w = pd.read_csv(REPO / "data/remeasure_v1/w_v05_summary.csv")
    sel = w[(w.landscape == "mgstd5") & (w.use_visibility_reward == True)]  # noqa: E712
    names = {0.0: "M0 Closed Cohort", 0.1: "M1 Vertical Lineage"}
    for _, r in sel.iterrows():
        rows.append(dict(
            panel="C", series=names[float(r.agent_generation_time)],
            c=float(r.curiosity), xi=1.0 - float(r.curiosity),
            value=float(r.wedge_local_minus_ideal_fin), sd=np.nan, n=int(r.n_seeds),
            metric="relational wedge (c_local - c_ideal), final step",
            source_file="data/remeasure_v1/w_v05_summary.csv"))
    for c, d in _turnover_cells().groupby("curiosity"):
        w_vals = d.c_local_fin - d.c_ideal_fin
        rows.append(dict(
            panel="C", series="M1 + Turnover (age_hazard)",
            c=float(c), xi=1.0 - float(c),
            value=float(w_vals.mean()), sd=float(w_vals.std()),
            n=int(len(d)), metric="relational wedge (c_local - c_ideal), final step",
            source_file=" + ".join(sorted(d.source_file.unique()))))
    m2 = _m2_cells()
    for lam, tag in ((0.0, "M2 (lambda=0)"), (1.0, "M2 (lambda=1)")):
        for c, d in m2[m2.dyadic_lambda == lam].groupby("curiosity"):
            w_vals = d.Mean_of_Corr_prestige_local_merit - d.Mean_of_Corr_prestige_ideal_merit
            rows.append(dict(
                panel="C", series=tag, c=float(c), xi=1.0 - float(c),
                value=float(w_vals.mean()), sd=float(w_vals.std()), n=int(len(d)),
                metric="relational wedge (c_local - c_ideal), final step",
                source_file=" + ".join(sorted(d.source_file.unique()))))
    return pd.DataFrame(rows)


def top10_points() -> pd.DataFrame:
    """Terminal top-10% prestige share per xi."""
    rows = []
    w = pd.read_csv(REPO / "data/remeasure_v1/w_v05_summary.csv")
    sel = w[(w.landscape == "mgstd5") & (w.use_visibility_reward == True)]  # noqa: E712
    names = {0.0: "M0 Closed Cohort", 0.1: "M1 Vertical Lineage"}
    for _, r in sel.iterrows():
        rows.append(dict(
            panel="D", series=names[float(r.agent_generation_time)],
            c=float(r.curiosity), xi=1.0 - float(r.curiosity),
            value=float(r.top10_fin), sd=np.nan, n=int(r.n_seeds),
            metric="top-10% prestige share (terminal)",
            source_file="data/remeasure_v1/w_v05_summary.csv"))
    for c, d in _turnover_cells().groupby("curiosity"):
        rows.append(dict(
            panel="D", series="M1 + Turnover (age_hazard)",
            c=float(c), xi=1.0 - float(c),
            value=float(d.top10_fin.mean()), sd=float(d.top10_fin.std()),
            n=int(len(d)), metric="top-10% prestige share (terminal)",
            source_file=" + ".join(sorted(d.source_file.unique()))))
    m2 = _m2_cells()
    for lam, tag in ((0.0, "M2 (lambda=0)"), (1.0, "M2 (lambda=1)")):
        for c, d in m2[m2.dyadic_lambda == lam].groupby("curiosity"):
            rows.append(dict(
                panel="D", series=tag, c=float(c), xi=1.0 - float(c),
                value=float(d.top10pct_prestige_share.mean()),
                sd=float(d.top10pct_prestige_share.std()), n=int(len(d)),
                metric="top-10% prestige share (terminal)",
                source_file=" + ".join(sorted(d.source_file.unique()))))
    return pd.DataFrame(rows)
def _bands(ax) -> None:
    """Shared two-stage bands: Stage 1, welfare-kink band, Stage 2."""
    ax.axvspan(0.3, 0.6, color=PALETTE["gray_light"], alpha=0.18, zorder=0)
    ax.axvspan(0.6, 0.8, facecolor=PALETTE["purple_light"], alpha=0.18,
               hatch="///", edgecolor=PALETTE["gray"], linewidth=0, zorder=0)
    for xv in (0.6, 0.8):
        ax.axvline(xv, ls="--", lw=0.9, color=PALETTE["gray"], zorder=1)
    ax.axvspan(0.8, 1.0, color=PALETTE["gray_light"], alpha=0.35, zorder=0)
    ax.text(0.45, 0.96, "Stage 1", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=HOUSE_STYLE["tick_size"], color=PALETTE["gray"])
    ax.text(0.70, 0.96, "kink", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=HOUSE_STYLE["tick_size"], color=PALETTE["gray"])
    ax.text(0.90, 0.96, "Stage 2", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=HOUSE_STYLE["tick_size"], color=PALETTE["gray"])


#: one home for the drawing grammar of every series: colour, linestyle, marker
SERIES_STYLE = {
    "MultG2": (PALETTE["purple"], "-", "o"),
    "MultG6": (PALETTE["orange"], "-", "o"),
    # two M1 arms must differ in HUE, not only in linestyle: red stays the
    # failure marker for the turnover regime; the pre-turnover arm is neutral
    # grey (it is the baseline that turnover is measured against) -- its markers
    # are hollow so the red curve stays readable where the two coincide
    "M1 lineage": (PALETTE["gray"], "--", "o"),
    "M1 + Turnover (age_hazard)": (FAIL, "-", "D"),
    "M2 (lambda=0)": (PALETTE["blue"], "--", "s"),
    "M2 (lambda=1)": (PALETTE["green"], "-", "^"),
    "M0 Closed Cohort": (PALETTE["purple"], "-", "o"),
    # same arm as "M1 lineage" in panel (a): one hue across both panels
    "M1 Vertical Lineage": (PALETTE["gray"], "--", "o"),
}
SHORT = {"MultG2": "M0 MultG2", "MultG6": "M0 MultG6",
         "M1 lineage": "M1 lineage", "M1 + Turnover (age_hazard)": "M1 + turnover",
         "M2 (lambda=0)": "M2 lambda=0", "M2 (lambda=1)": "M2 lambda=1",
         "M0 Closed Cohort": "M0 closed", "M1 Vertical Lineage": "M1 lineage"}


#: series whose markers are hollow -- they sit on top of another curve
HOLLOW = frozenset({"M1 lineage", "M1 Vertical Lineage"})


def _plot_series(ax, frame: pd.DataFrame, order: tuple[str, ...]) -> None:
    for name in order:
        d = frame[frame.series == name].sort_values("xi")
        color, ls, mk = SERIES_STYLE[name]
        ax.plot(d.xi.to_numpy(float), d.value.to_numpy(float),
                color=color, linestyle=ls, marker=mk, linewidth=1.7,
                markersize=4.2, label=SHORT[name], zorder=4,
                markerfacecolor="white" if name in HOLLOW else color,
                markeredgecolor=color, markeredgewidth=1.0)


def _footnote(macro: pd.DataFrame, micro: pd.DataFrame) -> str:
    """Verdict lines COMPOSED from the plotted values (never hardcoded)."""
    g2 = macro[(macro.series == "MultG2")].sort_values("xi")
    g6 = macro[(macro.series == "MultG6")].sort_values("xi")
    plateau2 = g2[g2.xi <= 0.5].value.max()
    plateau6 = g6[g6.xi <= 0.5].value.max()
    s1_2 = g2[(g2.xi >= 0.3) & (g2.xi <= 0.6)].value.min()
    s1_6 = g6[(g6.xi >= 0.3) & (g6.xi <= 0.6)].value.min()
    loss2, loss6 = 100 * (plateau2 - s1_2), 100 * (plateau6 - s1_6)
    m1 = micro[(micro.series == "M1 Vertical Lineage") & (micro.xi <= 0.6)]
    turn = micro[micro.series == "M1 + Turnover (age_hazard)"]
    m2a = micro[micro.series == "M2 (lambda=1)"].sort_values("xi")
    m2b = micro[micro.series == "M2 (lambda=0)"].sort_values("xi")
    lines = [
        "Sources: figures_v1/run_number{2,4} (M0, 11 c x 10 seeds); "
        "remeasure_v1/w_v05_summary (M0/M1); turnover_v1 + fig1_extension_v1 "
        "cells (M1+turnover); path_b_v2 + fig1_extension_v1 cells (M2). 20 seeds "
        "per cell, H=400 turnover / 200 else, pins mgstd5 + entry 0.1.",
        f"Measured: Stage-1 loss vs plateau {loss2:.1f} pp (MultG2), {loss6:.1f} pp "
        f"(MultG6); M1 lineage negative at {int((m1.value < 0).sum())}/{len(m1)} "
        f"Stage-1 points, M1+turnover at {int((turn.value < 0).sum())}/{len(turn)} "
        f"spanning [{turn.value.min():+.3f}, {turn.value.max():+.3f}]; M2 alignment "
        f"{m2b.value.min():+.3f}..{m2b.value.max():+.3f} (lambda=0), "
        f"{m2a.value.min():+.3f}..{m2a.value.max():+.3f} (lambda=1). "
        "Model-internal, not calibrated.",
    ]
    return "\n".join(textwrap.fill(t, width=118) for t in lines)


# -------------------------------------------------------------------- figure
def build_f6(macro: pd.DataFrame, micro: pd.DataFrame,
             wedge: pd.DataFrame, top10: pd.DataFrame) -> dict:
    def left(ax):
        _bands(ax)
        _plot_series(ax, macro, ("MultG2", "MultG6", "M1 lineage",
                                 "M1 + Turnover (age_hazard)",
                                 "M2 (lambda=0)", "M2 (lambda=1)"))
        ax.set_ylim(0, float(macro.value.max()) * 1.18)
        ax.set_xlim(0, 1)
        _cosmetics(ax, HOUSE_STYLE, "",
                   "",
                   "harvested knowledge share")

    def right(ax):
        _bands(ax)
        ax.axhline(0, color=PALETTE["gray"], lw=1.0, zorder=2)
        ax.text(0.01, 0, "0", transform=ax.get_yaxis_transform(),
                fontsize=HOUSE_STYLE["tick_size"], color=PALETTE["gray"],
                va="bottom")
        _plot_series(ax, micro, ("M0 Closed Cohort", "M1 Vertical Lineage",
                                 "M1 + Turnover (age_hazard)",
                                 "M2 (lambda=0)", "M2 (lambda=1)"))
        lo, hi = float(micro.value.min()), float(micro.value.max())
        span = hi - lo
        ax.set_ylim(lo - 0.12 * span, hi + 0.12 * span)
        ax.set_xlim(0, 1)
        _cosmetics(ax, HOUSE_STYLE, "",
                   "",
                   "Corr(prestige, ideal merit)")

    def panel_c(ax):
        _bands(ax)
        ax.axhline(0, color=PALETTE["gray"], lw=1.0, zorder=2)
        ax.text(0.01, 0, "0", transform=ax.get_yaxis_transform(),
                fontsize=HOUSE_STYLE["tick_size"], color=PALETTE["gray"],
                va="bottom")
        _plot_series(ax, wedge, ("M0 Closed Cohort", "M1 Vertical Lineage",
                                 "M1 + Turnover (age_hazard)",
                                 "M2 (lambda=0)", "M2 (lambda=1)"))
        lo, hi = float(wedge.value.min()), float(wedge.value.max())
        span = hi - lo
        ax.set_ylim(lo - 0.14 * span, hi + 0.14 * span)
        ax.set_xlim(0, 1)
        _cosmetics(ax, HOUSE_STYLE, "",
                   "visibility weight  $\\xi = 1 - c$",
                   "relational wedge (c_local - c_ideal)")

    def panel_d(ax):
        _bands(ax)
        ax.axhline(0.10, color=PALETTE["gray"], lw=0.9, ls=":", zorder=2)
        ax.text(0.01, 0.10, "0.10 equality", transform=ax.get_yaxis_transform(),
                fontsize=HOUSE_STYLE["tick_size"] - 1.5, color=PALETTE["gray"],
                va="bottom")
        _plot_series(ax, top10, ("M0 Closed Cohort", "M1 Vertical Lineage",
                                 "M1 + Turnover (age_hazard)",
                                 "M2 (lambda=0)", "M2 (lambda=1)"))
        ax.set_ylim(0.05, float(top10.value.max()) * 1.15)
        ax.set_xlim(0, 1)
        _cosmetics(ax, HOUSE_STYLE, "",
                   "visibility weight  $\\xi = 1 - c$",
                   "top-10% prestige share")

    fig, axes = dt.styled_panel(
        [left, right, panel_c, panel_d], nrows=2, ncols=2,
        titles=["(a) Macro efficiency", "(b) Micro signal alignment",
                "(c) Relational wedge", "(d) Elite concentration (top 10%)"],
        figsize=(8.0, 6.3))
    note = _footnote(macro, micro)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.94, bottom=0.295,
                        hspace=0.26, wspace=0.26)
    flat = np.atleast_1d(axes).ravel()
    h_a, lb_a = flat[0].get_legend_handles_labels()
    fig.legend(h_a, lb_a,
               loc="upper center", bbox_to_anchor=(0.53, 0.215),
               ncol=3, frameon=False,
               fontsize=HOUSE_STYLE["tick_size"] - 1.5, handlelength=1.7,
               columnspacing=1.6, labelspacing=0.35)
    fig.text(0.01, 0.015, note, fontsize=HOUSE_STYLE["note_size"],
             color=PALETTE["gray"], va="bottom")
    fig.savefig(OUT_PNG, dpi=HOUSE_STYLE["dpi"])
    fig.savefig(OUT_PNG.with_suffix(".pdf"))  # vector text for venue editing
    plt.close(fig)
    csv = write_figure_data(
        pd.concat([macro, micro, wedge, top10], ignore_index=True), OUT_PNG)
    return {"name": "F6_two_stage_proxy_failure",
            "paths": [str(OUT_PNG), str(OUT_PNG.with_suffix(".pdf")), str(csv)],
            "sources": sorted(set(
                macro.source_file.tolist() + micro.source_file.tolist()))}


def main() -> None:
    macro, micro, wedge, top10 = macro_points(), micro_points(), wedge_points(), top10_points()
    rep = build_f6(macro, micro, wedge, top10)
    for p in rep["paths"]:
        kb = Path(p).stat().st_size / 1024
        print(f"{Path(p).name}: {kb:.1f} kB")
    for s in rep["sources"]:
        print(f"source: {s}")
    print(f"plotted rows: macro {len(macro)}, micro {len(micro)}, wedge {len(wedge)}, top10 {len(top10)}")
    for name, d in (("macro", macro), ("micro", micro)):
        cov = d.groupby("series").xi.apply(lambda s: len(set(np.round(s, 2)))).to_dict()
        print(f"  xi points per series ({name}): {cov}")


if __name__ == "__main__":
    main()
