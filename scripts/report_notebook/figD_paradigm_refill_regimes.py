"""F13 -- paradigm refill regimes: lock-in, frontier stock, alignment (Section 5.3).

Figure D of the section-figures-v2 commission, replacing ``FC_refill_lockin``
(which plotted only top10 and contradicted its own caption).  From
``data/interventions_v1/cells/cellC_*.csv`` (18 cells x 20 seeds = 360 runs,
mgstd5; the 3 ACTUAL refill regimes -- off / occupied / frontier, no
'uniform' -- across M0 (g = agent_generation_time = 0.0) and M1 (g = 0.1),
each pooled over initial_curiosity c in {0.3, 0.5, 0.7} and 20 seeds):

  (a) Mainstream lock-in:           explored grid share
  (b) Unclaimed frontier knowledge: mean residual knowledge on grid (log y)
  (c) Merit alignment:              Corr(prestige, ideal merit) -- bars
        (final step) + M1 diamonds (time mean), both aggregations labelled
  (d) Inequality without currency-scale illusion: top-10% share and the
        normalized relative newcomer gap (newcomer - incumbent)/incumbent
        under M1, with the mean_prestige inflation annotated.

Run:  MPLBACKEND=Agg python3 scripts/report_notebook/figD_paradigm_refill_regimes.py
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

from diagram_templates import (  # noqa: E402
    HOUSE_STYLE,
    PALETTE,
    _cosmetics,
    write_figure_data,
)
from turnover_lifecycle_figure import assert_text_fits  # noqa: E402

CELLS = REPO / "data" / "interventions_v1" / "cells"
FNAME = "F13_paradigm_refill_regimes"

REGIMES = ("off", "occupied", "frontier")
GS = (0.0, 0.1)
G_LABEL = {0.0: "M0 (g = 0)", 0.1: "M1 (g = 0.1)"}
#: regime hue; _FAIL_COLOR marks the incumbent-bailout lock-in regime
REGIME_COLOR = {"off": PALETTE["purple"], "occupied": "#D73027",
                "frontier": PALETTE["green"]}

METRIC_COL = {
    "explored grid share": "explored_final",
    "mean residual knowledge on grid": "avg_knowledge_on_grid",
    "Corr(prestige, ideal merit) (final step)": "corr_prestige_ideal_merit_final",
    "Corr(prestige, ideal merit) (time mean)": "Mean_of_Corr_prestige_ideal_merit",
    "top-10% prestige share": "top10_final",
}


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


# ----------------------------------------------------------------- data ----
def load_runs() -> pd.DataFrame:
    files = sorted(CELLS.glob("cellC_*.csv"))
    _require(len(files) == 18, f"cellC files=18, found {len(files)}")
    frames = []
    for f in files:
        d = pd.read_csv(f)
        _require(len(d) == 20, f"{f.name}: 20 seeds expected, got {len(d)}")
        d["source_file"] = f.name
        frames.append(d)
    c = pd.concat(frames, ignore_index=True)
    _require(len(c) == 360, f"360 runs expected, got {len(c)}")
    _require(sorted(c["refill_mode"].unique()) == sorted(REGIMES),
             f"refill_mode grid (3 actual regimes, no 'uniform'): "
             f"{sorted(c['refill_mode'].unique())}")
    _require(sorted(c["agent_generation_time"].unique().tolist()) == list(GS),
             "agent_generation_time grid")
    c["gap_ratio"] = ((c["newcomer_prestige_mean"] - c["incumbent_prestige_mean"])
                      / c["incumbent_prestige_mean"])
    return c


def series_rows(c: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(panel: str, metric: str, regime: str, g: float, value: float,
            ci: float, n: int, statistic: str = "mean", **extra) -> None:
        r = dict(figure=FNAME, panel=panel, metric=metric,
                 statistic=statistic, refill_mode=regime, g_entry=g,
                 value=float(value), ci_lo=float(value - ci),
                 ci_hi=float(value + ci), n=int(n),
                 source_file="data/interventions_v1/cells/cellC_*.csv",
                 note="pooled over initial_curiosity {0.3, 0.5, 0.7} and 20 seeds"
                      "; 95% CI = 1.96*sd/sqrt(n)")
        r.update(extra)
        rows.append(r)

    for metric, col in METRIC_COL.items():
        panel = {"explored grid share": "a",
                 "mean residual knowledge on grid": "b",
                 "Corr(prestige, ideal merit) (final step)": "c",
                 "Corr(prestige, ideal merit) (time mean)": "c",
                 "top-10% prestige share": "d"}[metric]
        for g in GS:
            for regime in REGIMES:
                sub = c[(c["refill_mode"] == regime)
                        & (c["agent_generation_time"] == g)][col]
                _require(len(sub) == 60, f"{metric} {regime} g={g}: n={len(sub)}")
                ci = 1.96 * float(sub.std(ddof=1)) / np.sqrt(len(sub))
                add(panel, metric, regime, g, float(sub.mean()), ci, len(sub))
    # relative newcomer gap -- defined on M1 only (M0 has no newcomers)
    for regime in REGIMES:
        sub = c[(c["refill_mode"] == regime) & (c["agent_generation_time"] == 0.1)]
        v = sub["gap_ratio"]
        _require(len(v) == 60, f"gap_ratio {regime}: n={len(v)}")
        ci = 1.96 * float(v.std(ddof=1)) / np.sqrt(len(v))
        add("d", "relative newcomer gap ((newcomer-incumbent)/incumbent)",
            regime, 0.1, float(v.mean()), ci, len(v))
    return pd.DataFrame(rows)


def annotation_rows(c: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(metric: str, value: float, note: str) -> None:
        rows.append(dict(figure=FNAME, panel="annotation", metric=metric,
                         statistic="annotation", refill_mode="", g_entry=np.nan,
                         value=float(value), ci_lo=np.nan, ci_hi=np.nan, n=360,
                         source_file="data/interventions_v1/cells/cellC_*.csv",
                         note=note))

    m1 = c[c["agent_generation_time"] == 0.1]

    def m1mean(col: str, regime: str) -> float:
        return float(m1[m1["refill_mode"] == regime][col].mean())

    ex_off = m1mean("explored_final", "off")
    ex_occ = m1mean("explored_final", "occupied")
    ex_fro = m1mean("explored_final", "frontier")
    add("occupied_exploration_collapse_pct_M1",
        100.0 * (ex_occ / ex_off - 1.0),
        f"explored grid share off {ex_off:.4f} -> occupied {ex_occ:.4f} "
        "(commission: -83.8% from rounded labels 37.7->6.1)")
    add("frontier_exploration_pct_M1", 100.0 * ex_fro,
        "frontier explored grid share under M1 (commission: 33.0%)")

    ak_off = m1mean("avg_knowledge_on_grid", "off")
    ak_occ = m1mean("avg_knowledge_on_grid", "occupied")
    ak_fro = m1mean("avg_knowledge_on_grid", "frontier")
    add("frontier_residual_ratio_vs_off_M1", ak_fro / ak_off,
        f"residual knowledge frontier {ak_fro:.6f} / off {ak_off:.6f} "
        "(commission: 315x computed on rounded labels 0.1895/0.0006)")
    add("occupied_residual_ratio_vs_off_M1", ak_occ / ak_off,
        f"residual knowledge occupied {ak_occ:.6f} / off {ak_off:.6f}")

    for regime in REGIMES:
        add(f"corr_tm_M1_{regime}", m1mean("Mean_of_Corr_prestige_ideal_merit",
                                           regime),
            f"Corr(prestige, ideal merit) time mean, M1, {regime}")
        add(f"corr_final_M1_{regime}",
            m1mean("corr_prestige_ideal_merit_final", regime),
            f"Corr(prestige, ideal merit) final step, M1, {regime}")

    mp_off = m1mean("mean_prestige", "off")
    for regime in ("occupied", "frontier"):
        add(f"mean_prestige_inflation_{regime}_M1",
            m1mean("mean_prestige", regime) / mp_off,
            f"raw mean_prestige inflation vs off under M1 ({regime}; "
            "commission: 31x / 44x)")
    return pd.DataFrame(rows)


# --------------------------------------------------------------- drawing ---
def _no_text_overlap(fig) -> None:
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = []
    for t in fig.texts:
        if t.get_text().strip():
            boxes.append((t.get_window_extent(renderer=r), t.get_text()))
    for ax in fig.axes:
        for t in ax.texts:
            if t.get_text().strip():
                boxes.append((t.get_window_extent(renderer=r), t.get_text()))
    for leg in fig.legends:
        for t in leg.get_texts():
            if t.get_text().strip():
                boxes.append((t.get_window_extent(renderer=r), t.get_text()))
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                if t.get_text().strip():
                    boxes.append((t.get_window_extent(renderer=r), t.get_text()))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (b1, s1), (b2, s2) = boxes[i], boxes[j]
            ov = (min(b1.x1, b2.x1) - max(b1.x0, b2.x0),
                  min(b1.y1, b2.y1) - max(b1.y0, b2.y0))
            if ov[0] > 4 and ov[1] > 4:
                raise AssertionError(
                    f"overlapping text artists: {s1[:30]!r} vs {s2[:30]!r}")


def _grouped_bars(ax, S: pd.DataFrame, metric: str, *, hatch: bool = True,
                  logy: bool = False) -> dict:
    """Bars per regime, M0 hollow vs M1 solid (hue = regime)."""
    width = 0.36
    xs = np.arange(len(REGIMES))
    out = {}
    for gi, g in enumerate(GS):
        vals, los, his = [], [], []
        for regime in REGIMES:
            r = S[(S["metric"] == metric) & (S["statistic"] == "mean")
                  & (S["refill_mode"] == regime) & (S["g_entry"] == g)].iloc[0]
            vals.append(r["value"])
            los.append(r["value"] - r["ci_lo"])
            his.append(r["ci_hi"] - r["value"])
            out[(metric, regime, g)] = r
        x = xs + (gi - 0.5) * width
        bars = ax.bar(x, vals, width=width,
                      color=["white"] * 3 if g == 0.0 else
                      [REGIME_COLOR[r] for r in REGIMES],
                      edgecolor=[REGIME_COLOR[r] for r in REGIMES],
                      linewidth=1.2, zorder=3,
                      hatch="///" if (hatch and g == 0.0) else None)
        for bb, col in zip(bars, [REGIME_COLOR[r] for r in REGIMES]):
            bb.set_edgecolor(col)
        ax.errorbar(x, vals, yerr=[los, his], fmt="none", ecolor="black",
                    elinewidth=0.9, capsize=3.0, zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels(REGIMES)
    return out


def build(outdir: str = "output/report_notebook_v1") -> dict:
    c = load_runs()
    series = series_rows(c)
    anns = annotation_rows(c)
    sidecar = pd.concat([series, anns], ignore_index=True)
    A = {r["metric"]: r["value"] for _, r in anns.iterrows()}
    S = series

    fig = plt.figure(figsize=(11.2, 7.6))
    gs = fig.add_gridspec(2, 2)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    gs_d = gs[1, 1].subgridspec(1, 2, wspace=0.42)
    ax_d1 = fig.add_subplot(gs_d[0, 0])
    ax_d2 = fig.add_subplot(gs_d[0, 1])

    # ---- (a) explored grid share ---------------------------------------
    _grouped_bars(ax_a, S, "explored grid share")
    _cosmetics(ax_a, HOUSE_STYLE, "A · Mainstream lock-in",
               "refill regime", "explored grid share")
    ax_a.text(0.03, 0.97,
              f"occupied collapses exploration "
              f"{A['occupied_exploration_collapse_pct_M1']:.1f}% (M1)\n"
              f"off → occupied: "
              f"{100 * S[(S.metric == 'explored grid share') & (S.refill_mode == 'off') & (S.g_entry == 0.1)].value.iloc[0]:.1f}%"
              f" → "
              f"{100 * S[(S.metric == 'explored grid share') & (S.refill_mode == 'occupied') & (S.g_entry == 0.1)].value.iloc[0]:.1f}%; "
              f"frontier {A['frontier_exploration_pct_M1']:.1f}%",
              transform=ax_a.transAxes, va="top", ha="left",
              fontsize=HOUSE_STYLE["note_size"], color=PALETTE["gray"])
    ax_a.set_ylim(0, 0.52)

    # ---- (b) residual knowledge on grid, log scale ---------------------
    rows_b = _grouped_bars(ax_b, S, "mean residual knowledge on grid",
                           logy=True)
    ax_b.set_yscale("log")
    _cosmetics(ax_b, HOUSE_STYLE,
               "B · Unclaimed frontier knowledge (log scale)",
               "refill regime", "mean residual knowledge on grid")
    for (metric, regime, g), r in rows_b.items():
        r = rows_b[(metric, regime, g)]
        x = np.arange(len(REGIMES))[REGIMES.index(regime)] + \
            (0.0 if g == 0.1 else -0.18)
        ax_b.text(x, r["ci_hi"] * 1.25, f"{r['value']:.6f}", ha="center",
                  va="bottom", fontsize=HOUSE_STYLE["tick_size"] - 1,
                  color=PALETTE["gray"], rotation=90, zorder=5)
    ax_b.text(0.03, 0.97,
              f"frontier = {A['frontier_residual_ratio_vs_off_M1']:.0f}× off "
              f"baseline (M1)\noccupied = "
              f"{A['occupied_residual_ratio_vs_off_M1']:.1f}× off",
              transform=ax_b.transAxes, va="top", ha="left",
              fontsize=HOUSE_STYLE["note_size"], color=PALETTE["gray"])

    # ---- (c) merit alignment: bars final step + M1 diamonds time mean --
    width = 0.36
    xs = np.arange(len(REGIMES))
    for gi, g in enumerate(GS):
        vals, los, his = [], [], []
        for regime in REGIMES:
            r = S[(S.metric.str.startswith("Corr(prestige, ideal merit) (final"))
                  & (S.refill_mode == regime) & (S.g_entry == g)].iloc[0]
            vals.append(r["value"])
            los.append(r["value"] - r["ci_lo"])
            his.append(r["ci_hi"] - r["value"])
        x = xs + (gi - 0.5) * width
        ax_c.bar(x, vals, width=width,
                 color=["white"] * 3 if g == 0.0 else
                 [REGIME_COLOR[r] for r in REGIMES],
                 edgecolor=[REGIME_COLOR[r] for r in REGIMES],
                 linewidth=1.2, zorder=3,
                 hatch="///" if g == 0.0 else None)
        ax_c.errorbar(x, vals, yerr=[los, his], fmt="none", ecolor="black",
                      elinewidth=0.9, capsize=3.0, zorder=4)
    dm = []
    for regime in REGIMES:
        r = S[(S.metric.str.startswith("Corr(prestige, ideal merit) (time mean"))
              & (S.refill_mode == regime) & (S.g_entry == 0.1)].iloc[0]
        dm.append(r["value"])
    ax_c.plot(xs, dm, marker="D", markersize=6, linestyle="none",
              color=PALETTE["blue"], markeredgecolor="black", zorder=6,
              label="M1 time mean (diamonds)")
    for x, v in zip(xs, dm):
        ax_c.text(x, v + 0.018, f"{v:+.3f}", ha="center", va="bottom",
                  fontsize=HOUSE_STYLE["tick_size"] - 1, color=PALETTE["blue"],
                  zorder=6)
    r_fro = S[(S.metric.str.startswith("Corr(prestige, ideal merit) (final"))
              & (S.refill_mode == "frontier") & (S.g_entry == 0.1)].iloc[0]
    ax_c.text(-0.42, 0.74,
              f"frontier terminal {r_fro['value']:+.3f}\nonly positive under "
              "entry", fontsize=HOUSE_STYLE["note_size"],
              color=PALETTE["gray"], va="top", ha="left")
    ax_c.axhline(0.0, color=PALETTE["gray"], ls="--", lw=1.0, zorder=2)
    ax_c.set_xticks(xs)
    ax_c.set_xticklabels(REGIMES)
    _cosmetics(ax_c, HOUSE_STYLE,
               "C · Merit alignment  (bars: final step; diamonds: time mean)",
               "refill regime",
               "Corr(prestige, ideal merit)\n(bars (final step); "
               "diamonds (time mean))")
    ax_c.legend(fontsize=HOUSE_STYLE["tick_size"] - 1, frameon=False,
                loc="upper right")

    # ---- (d) inequality + normalized newcomer gap ----------------------
    t10 = []
    for regime in REGIMES:
        r = S[(S.metric == "top-10% prestige share") & (S.refill_mode == regime)
              & (S.g_entry == 0.1)].iloc[0]
        t10.append(r)
    ax_d1.bar(xs, [r["value"] for r in t10], width=0.62,
              color=[REGIME_COLOR[regime] for regime in REGIMES], zorder=3)
    ax_d1.errorbar(xs, [r["value"] for r in t10],
                   yerr=[[r["value"] - r["ci_lo"] for r in t10],
                         [r["ci_hi"] - r["value"] for r in t10]],
                   fmt="none", ecolor="black", elinewidth=0.9, capsize=3.0,
                   zorder=4)
    for x, r in zip(xs, t10):
        ax_d1.text(x, r["ci_hi"] + 0.004, f"{r['value']:.3f}", ha="center",
                   va="bottom", fontsize=HOUSE_STYLE["tick_size"] - 1,
                   color=PALETTE["gray"], zorder=5)
    ax_d1.axhline(0.10, color=PALETTE["gray"], ls=":", lw=1.1, zorder=2)
    ax_d1.set_xlim(-0.55, 2.80)
    ax_d1.text(2.78, 0.108, "0.10\nequality", ha="right", va="bottom",
               fontsize=HOUSE_STYLE["tick_size"] - 1, color=PALETTE["gray"])
    ax_d1.set_xticks(xs)
    ax_d1.set_xticklabels(REGIMES, fontsize=HOUSE_STYLE["tick_size"] - 1)
    _cosmetics(ax_d1, HOUSE_STYLE, "D1 · top-10% share (M1)",
               "refill regime", "top-10% prestige share")
    ax_d1.yaxis.label.set_fontsize(HOUSE_STYLE["tick_size"])
    ax_d1.set_ylim(0, 0.33)

    gaps = []
    for regime in REGIMES:
        r = S[(S.metric.str.startswith("relative newcomer gap"))
              & (S.refill_mode == regime)].iloc[0]
        gaps.append(r)
    ax_d2.bar(xs, [r["value"] for r in gaps], width=0.62,
              color=[REGIME_COLOR[regime] for regime in REGIMES], zorder=3)
    ax_d2.errorbar(xs, [r["value"] for r in gaps],
                   yerr=[[r["value"] - r["ci_lo"] for r in gaps],
                         [r["ci_hi"] - r["value"] for r in gaps]],
                   fmt="none", ecolor="black", elinewidth=0.9, capsize=3.0,
                   zorder=4)
    for x, r in zip(xs, gaps):
        ax_d2.text(x, r["value"] - 0.02, f"{100 * r['value']:.1f}%", ha="center",
                   va="top", fontsize=HOUSE_STYLE["tick_size"] - 1,
                   color=PALETTE["gray"], zorder=5)
    ax_d2.axhline(0.0, color=PALETTE["gray"], ls="--", lw=1.0, zorder=2)
    ax_d2.set_xticks(xs)
    ax_d2.set_xticklabels(REGIMES, fontsize=HOUSE_STYLE["tick_size"] - 1)
    _cosmetics(ax_d2, HOUSE_STYLE, "D2 · newcomer gap (M1)",
               "refill regime", "(newcomer − incumbent) / incumbent")
    ax_d2.yaxis.label.set_fontsize(HOUSE_STYLE["tick_size"])
    ax_d2.text(0.03, 0.97,
               "mean_prestige inflates\n"
               f"{A['mean_prestige_inflation_occupied_M1']:.1f}× (occupied)\n"
               f"{A['mean_prestige_inflation_frontier_M1']:.1f}× (frontier)",
               transform=ax_d2.transAxes, va="top", ha="left",
               fontsize=HOUSE_STYLE["note_size"] - 1, color=PALETTE["gray"])

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=REGIME_COLOR[r],
                             edgecolor=REGIME_COLOR[r]) for r in REGIMES]
    labels = [f"{r} (refill)" for r in REGIMES]
    handles += [plt.Rectangle((0, 0), 1, 1, facecolor="white",
                              edgecolor=PALETTE["gray"], hatch="///"),
                plt.Rectangle((0, 0), 1, 1, facecolor=PALETTE["gray"],
                              edgecolor=PALETTE["gray"]),
                plt.Line2D([], [], marker="D", linestyle="none",
                           color=PALETTE["blue"], markeredgecolor="black")]
    labels += ["M0 (g = 0), hollow", "M1 (g = 0.1), solid",
               "M1 time mean (diamonds)"]
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, 0.10))

    footnote = (
        "Sources: data/interventions_v1/cells/cellC_*.csv (18 cells x 20 seeds "
        "= 360 runs, mgstd5; refill_mode in {off, occupied, frontier} -- no "
        "'uniform' regime exists in the archive). Bars pool initial_curiosity "
        "{0.3, 0.5, 0.7} and 20 seeds (n=60 per bar), mean +/- 95% CI "
        "(1.96*sd/sqrt(n)); M0 = g = 0.0, M1 = g = 0.1. "
        f"Measured: occupied exploration collapse "
        f"{A['occupied_exploration_collapse_pct_M1']:.1f}% (M1; the commission's "
        "-83.8% is computed on rounded labels 37.7->6.1); frontier residual "
        f"knowledge {A['frontier_residual_ratio_vs_off_M1']:.0f}x the off "
        "baseline (commission's 315x is 0.1895/0.0006 on rounded labels); "
        f"mean_prestige inflates "
        f"{A['mean_prestige_inflation_occupied_M1']:.1f}x / "
        f"{A['mean_prestige_inflation_frontier_M1']:.1f}x (occupied / frontier, "
        "commission: 31x / 44x). Model-internal, not calibrated."
    )
    fig.text(0.01, 0.005, "\n".join(textwrap.fill(x, width=150)
                                    for x in footnote.split("\n")),
             fontsize=HOUSE_STYLE["note_size"], color=PALETTE["gray"],
             va="bottom")
    fig.subplots_adjust(left=0.075, right=0.975, top=0.92, bottom=0.24,
                        hspace=0.42, wspace=0.32)

    w, h = fig.get_size_inches() * fig.dpi
    assert_text_fits(fig, w, h)
    _no_text_overlap(fig)

    out = (REPO / outdir) if not Path(outdir).is_absolute() else Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{FNAME}.png"
    pdf = out / f"{FNAME}.pdf"
    fig.savefig(png, dpi=HOUSE_STYLE["dpi"])
    fig.savefig(pdf)
    plt.close(fig)
    csv_path = write_figure_data(sidecar, png)
    return {"png": str(png), "pdf": str(pdf), "data_csv": str(csv_path)}


if __name__ == "__main__":
    print(build())
