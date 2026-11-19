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

v3 repairs (Team 2, visual QA).  Every ylim is derived from the plotted
sidecar rows, the annotation blocks are moved into white space, and the
shared ``visual_qa.assert_figure_clean`` contract runs on the rendered canvas
before every ``savefig``:

  * Panel B: log top raised so the rotated 0.186078 / 0.189537 labels stop
    touching the title; the M0 label column now sits at the M0 bar offset
    (-0.18, previously 0.0 -> labels printed between the bars).
  * Panel C: top padded so the frontier callout clears the title, bottom
    padded so the off/occupied negative caps clear the spine, diamond labels
    offset right of the hatched bars instead of inside them.
  * Panel D1: ylim top raised for the value labels; the "0.10 equality" tag
    moved into the empty band right of the last bar (it was printed on the
    frontier bar).
  * Panel D2: ylim bottom expanded below the -67.0% cap and value labels
    dropped below their caps; the "mean_prestige inflates ..." note moved
    into the white space above the zero line.

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
import matplotlib.ticker as mticker  # noqa: E402

from diagram_templates import (  # noqa: E402
    HOUSE_STYLE,
    PALETTE,
    _cosmetics,
    write_figure_data,
)
from turnover_lifecycle_figure import assert_text_fits  # noqa: E402

import visual_qa  # noqa: E402

#: default archive; switch to ``data/interventions_v2`` in one line when it lands
DATA_ROOT = REPO / "data" / "interventions_v1" / "cells"
FNAME = "F13_paradigm_refill_regimes"

REGIMES = ("off", "occupied", "frontier")
GS = (0.0, 0.1)
G_LABEL = {0.0: "M0 (g = 0)", 0.1: "M1 (g = 0.1)"}
#: regime hue; _FAIL_COLOR marks the incumbent-bailout lock-in regime
REGIME_COLOR = {"off": PALETTE["purple"], "occupied": "#D73027",
                "frontier": PALETTE["green"]}

M_LOCKIN = "explored grid share"
M_FRONTIER = "mean residual knowledge on grid"
M_CORR_FIN = "Corr(prestige, ideal merit) (final step)"
M_CORR_TM = "Corr(prestige, ideal merit) (time mean)"
M_TOP10 = "top-10% prestige share"
#: x tick type size of the half-width D1/D2 panels ("occupied"/"frontier"
#: do not fit side by side at the panel default of 9 pt)
D_TICK = 7.0

M_GAP = "relative newcomer gap ((newcomer-incumbent)/incumbent)"

METRIC_COL = {
    M_LOCKIN: "explored_final",
    M_FRONTIER: "avg_knowledge_on_grid",
    M_CORR_FIN: "corr_prestige_ideal_merit_final",
    M_CORR_TM: "Mean_of_Corr_prestige_ideal_merit",
    M_TOP10: "top10_final",
}


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def data_root_path(data_root=None) -> Path:
    """Directory holding ``cellC_*.csv`` (archive dir or its ``cells`` subdir)."""
    root = Path(data_root) if data_root is not None else DATA_ROOT
    if not list(root.glob("cellC_*.csv")) and list((root / "cells").glob("cellC_*.csv")):
        root = root / "cells"
    return root


# ----------------------------------------------------------------- data ----
def load_runs(data_root=None) -> pd.DataFrame:
    files = sorted(data_root_path(data_root).glob("cellC_*.csv"))
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


def series_rows(c: pd.DataFrame, data_root=None) -> pd.DataFrame:
    rows = []
    try:
        src = str(data_root_path(data_root).relative_to(REPO)) + "/cellC_*.csv"
    except ValueError:  # pragma: no cover - root outside the repo
        src = str(data_root_path(data_root) / "cellC_*.csv")

    def add(panel: str, metric: str, regime: str, g: float, value: float,
            ci: float, n: int, statistic: str = "mean", **extra) -> None:
        r = dict(figure=FNAME, panel=panel, metric=metric,
                 statistic=statistic, refill_mode=regime, g_entry=g,
                 value=float(value), ci_lo=float(value - ci),
                 ci_hi=float(value + ci), n=int(n), source_file=src,
                 note="pooled over initial_curiosity {0.3, 0.5, 0.7} and 20 seeds"
                      "; 95% CI = 1.96*sd/sqrt(n)")
        r.update(extra)
        rows.append(r)

    for metric, col in METRIC_COL.items():
        panel = {M_LOCKIN: "a", M_FRONTIER: "b", M_CORR_FIN: "c",
                 M_CORR_TM: "c", M_TOP10: "d"}[metric]
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
        add("d", M_GAP, regime, 0.1, float(v.mean()), ci, len(v))
    return pd.DataFrame(rows)


def annotation_rows(c: pd.DataFrame, data_root=None) -> pd.DataFrame:
    rows = []
    try:
        src = str(data_root_path(data_root).relative_to(REPO)) + "/cellC_*.csv"
    except ValueError:  # pragma: no cover
        src = str(data_root_path(data_root) / "cellC_*.csv")

    def add(metric: str, value: float, note: str) -> None:
        rows.append(dict(figure=FNAME, panel="annotation", metric=metric,
                         statistic="annotation", refill_mode="", g_entry=np.nan,
                         value=float(value), ci_lo=np.nan, ci_hi=np.nan, n=360,
                         source_file=src, note=note))

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
    """Text-vs-text guard (kept alongside visual_qa: fails fast, local)."""
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


def _clean_yticks(ax, nbins: int = 6) -> None:
    """Re-seat the y ticks inside the (re)padded view.

    matplotlib keeps the ticks of the pre-padding view alive: their labels
    stay non-empty, they can sit outside the canvas, and when the old view was
    narrow they are too dense to read.  A fresh MaxNLocator tick set fixes
    both.
    """
    lo, hi = ax.get_ylim()
    ticks = mticker.MaxNLocator(nbins=nbins).tick_values(lo, hi)
    ax.set_yticks([t for t in ticks if lo - 1e-12 <= t <= hi + 1e-12])


def _pad_ylim(ax, spans, *, pad_bottom: float = 0.12, pad_top: float = 0.12,
              extra=(), log: bool = False) -> tuple[float, float]:
    """ylim from the plotted (lo, hi) pairs; pad by a fraction of the span.

    ``log=True`` pads in decades (a linear 3%-of-span pad is meaningless on a
    log axis, which is why the log panel passes ``headroom=0`` to visual_qa
    and is checked in log space here).
    """
    los = [float(s[0]) for s in spans]
    his = [float(s[1]) for s in spans]
    if log:  # log axis: a non-positive anchor is meaningless
        los += [float(v) for v in extra if v > 0]
        his += [float(v) for v in extra if v > 0]
    else:
        los += [float(v) for v in extra]
        his += [float(v) for v in extra]
    if log:
        lo = 10.0 ** np.floor(np.log10(min(los)))
        hi = 10.0 ** (np.log10(max(his)) + pad_top)  # pad_top = decades
        ax.set_ylim(lo, hi)
        return ax.get_ylim()
    lo, hi = min(los), max(his)
    span = hi - lo
    if span <= 0:
        span = max(abs(hi), 1.0)
    ax.set_ylim(lo - pad_bottom * span, hi + pad_top * span)
    _clean_yticks(ax)
    return ax.get_ylim()


def _label(ax, x, y, text, *, below=False, ha="center", va=None,
           fontsize=None, color=None, rotation=0, extra_pt=4.0) -> None:
    """Value label anchored `extra_pt` points clear of a bar cap/marker."""
    ax.annotate(text, xy=(x, y), xycoords="data",
                xytext=(0, -extra_pt if below else extra_pt),
                textcoords="offset points", ha=ha,
                va=va or ("top" if below else "bottom"),
                fontsize=fontsize or HOUSE_STYLE["tick_size"] - 1,
                color=color or PALETTE["gray"], rotation=rotation, zorder=6)


def _grouped_bars(ax, S: pd.DataFrame, metric: str, *, hatch: bool = True) -> dict:
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


def _figure(sidecar: pd.DataFrame):
    """Draw F13 from its sidecar; return (fig, visual_qa panel specs)."""
    A = {r["metric"]: r["value"] for _, r in
         sidecar[sidecar["statistic"] == "annotation"].iterrows()}
    S = sidecar[sidecar["statistic"] == "mean"]
    xs = np.arange(len(REGIMES))
    width = 0.36

    fig = plt.figure(figsize=(12.0, 7.8))
    gs = fig.add_gridspec(2, 2)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    gs_d = gs[1, 1].subgridspec(1, 2, wspace=0.34)
    ax_d1 = fig.add_subplot(gs_d[0, 0])
    ax_d2 = fig.add_subplot(gs_d[0, 1])
    fig.subplots_adjust(left=0.075, right=0.975, top=0.93, bottom=0.235,
                        hspace=0.45, wspace=0.30)

    # ---- (a) explored grid share ---------------------------------------
    rows_a = _grouped_bars(ax_a, S, M_LOCKIN)
    _cosmetics(ax_a, HOUSE_STYLE, "A · Mainstream lock-in",
               "refill regime", "explored grid share")
    ax_a.text(0.03, 0.97,
              f"occupied collapses exploration "
              f"{A['occupied_exploration_collapse_pct_M1']:.1f}% (M1)\n"
              f"off → occupied: "
              f"{100 * rows_a[(M_LOCKIN, 'off', 0.1)]['value']:.1f}%"
              f" → "
              f"{100 * rows_a[(M_LOCKIN, 'occupied', 0.1)]['value']:.1f}%; "
              f"frontier {A['frontier_exploration_pct_M1']:.1f}%",
              transform=ax_a.transAxes, va="top", ha="left",
              fontsize=HOUSE_STYLE["note_size"], color=PALETTE["gray"])
    _pad_ylim(ax_a, [(r["ci_lo"], r["ci_hi"]) for r in rows_a.values()],
              pad_bottom=0.0, pad_top=0.34, extra=(0.0,))
    ax_a.set_ylim(0.0, ax_a.get_ylim()[1])

    # ---- (b) residual knowledge on grid, log scale ---------------------
    rows_b = _grouped_bars(ax_b, S, M_FRONTIER)
    ax_b.set_yscale("log")
    _cosmetics(ax_b, HOUSE_STYLE,
               "B · Unclaimed frontier knowledge (log scale)",
               "refill regime", "mean residual knowledge on grid")
    for (metric, regime, g), r in rows_b.items():
        # M0 sits at the -0.18 bar offset, M1 at +0.18 (was 0.0 for M1)
        x = xs[REGIMES.index(regime)] + (-0.18 if g == 0.0 else 0.18)
        _label(ax_b, x, r["ci_hi"], f"{r['value']:.6f}", rotation=90,
               fontsize=7, extra_pt=3.0)
    ax_b.text(0.03, 0.965,
              f"frontier = {A['frontier_residual_ratio_vs_off_M1']:.0f}× off "
              f"baseline (M1)\noccupied = "
              f"{A['occupied_residual_ratio_vs_off_M1']:.1f}× off",
              transform=ax_b.transAxes, va="top", ha="left",
              fontsize=HOUSE_STYLE["note_size"], color=PALETTE["gray"])
    # 1.15 decades of headroom above the tallest cap: enough for the rotated
    # 6-decimal labels to clear the title (the v2 defect)
    _pad_ylim(ax_b, [(r["ci_lo"], r["ci_hi"]) for r in rows_b.values()],
              pad_top=1.15, log=True)

    # ---- (c) merit alignment: bars final step + M1 diamonds time mean --
    spans_c = []
    for gi, g in enumerate(GS):
        vals, los, his = [], [], []
        for regime in REGIMES:
            r = S[(S["metric"] == M_CORR_FIN) & (S["refill_mode"] == regime)
                  & (S["g_entry"] == g)].iloc[0]
            vals.append(r["value"])
            los.append(r["value"] - r["ci_lo"])
            his.append(r["ci_hi"] - r["value"])
            spans_c.append((r["ci_lo"], r["ci_hi"]))
        x = xs + (gi - 0.5) * width
        ax_c.bar(x, vals, width=width,
                 color=["white"] * 3 if g == 0.0 else
                 [REGIME_COLOR[r] for r in REGIMES],
                 edgecolor=[REGIME_COLOR[r] for r in REGIMES],
                 linewidth=1.2, zorder=3,
                 hatch="///" if g == 0.0 else None)
        ax_c.errorbar(x, vals, yerr=[los, his], fmt="none", ecolor="black",
                      elinewidth=0.9, capsize=3.0, zorder=4)
    dm = [S[(S["metric"] == M_CORR_TM) & (S["refill_mode"] == regime)
            & (S["g_entry"] == 0.1)].iloc[0] for regime in REGIMES]
    ax_c.plot(xs, [r["value"] for r in dm], marker="D", markersize=6,
              linestyle="none", color=PALETTE["blue"],
              markeredgecolor="black", zorder=6,
              label="M1 time mean (diamonds)")
    for x, r in zip(xs, dm):
        # right of the bar pair: the diamond itself sits between the bars
        _label(ax_c, x + 0.20, r["value"], f"{r['value']:+.3f}", ha="left",
               va="center", color=PALETTE["blue"],
               fontsize=HOUSE_STYLE["tick_size"] - 1)
        spans_c.append((r["value"], r["value"]))
    r_fro = S[(S["metric"] == M_CORR_FIN) & (S["refill_mode"] == "frontier")
              & (S["g_entry"] == 0.1)].iloc[0]
    ax_c.text(0.03, 0.97,
              f"frontier terminal {r_fro['value']:+.3f}\nonly positive under "
              "entry", fontsize=HOUSE_STYLE["note_size"],
              transform=ax_c.transAxes, color=PALETTE["gray"], va="top",
              ha="left")
    ax_c.axhline(0.0, color=PALETTE["gray"], ls="--", lw=1.0, zorder=2)
    ax_c.set_xticks(xs)
    ax_c.set_xticklabels(REGIMES)
    _cosmetics(ax_c, HOUSE_STYLE,
               "C · Merit alignment  (bars: final step; diamonds: time mean)",
               "refill regime",
               "Corr(prestige, ideal merit)\n(bars (final step); "
               "diamonds (time mean))")
    _pad_ylim(ax_c, spans_c, pad_bottom=0.10, pad_top=0.16, extra=(0.0,))
    ax_c.legend(fontsize=HOUSE_STYLE["tick_size"] - 1, frameon=False,
                loc="upper right", borderaxespad=0.2)

    # ---- (d) inequality + normalized newcomer gap ----------------------
    t10 = [S[(S["metric"] == M_TOP10) & (S["refill_mode"] == regime)
             & (S["g_entry"] == 0.1)].iloc[0] for regime in REGIMES]
    ax_d1.bar(xs, [r["value"] for r in t10], width=0.62,
              color=[REGIME_COLOR[regime] for regime in REGIMES], zorder=3)
    ax_d1.errorbar(xs, [r["value"] for r in t10],
                   yerr=[[r["value"] - r["ci_lo"] for r in t10],
                         [r["ci_hi"] - r["value"] for r in t10]],
                   fmt="none", ecolor="black", elinewidth=0.9, capsize=3.0,
                   zorder=4)
    for x, r in zip(xs, t10):
        _label(ax_d1, x, r["ci_hi"], f"{r['value']:.3f}",
               fontsize=HOUSE_STYLE["note_size"])
    ax_d1.axhline(0.10, color=PALETTE["gray"], ls=":", lw=1.1, zorder=2)
    ax_d1.set_xlim(-0.62, len(REGIMES) - 0.38)
    ax_d1.set_xticks(xs)
    _cosmetics(ax_d1, HOUSE_STYLE, "D1 · top-10% share (M1)",
               "refill regime", "top-10% prestige share")
    ax_d1.yaxis.label.set_fontsize(HOUSE_STYLE["tick_size"])
    # half-width panel: the three regime words only fit at reduced type size
    ax_d1.set_xticklabels(REGIMES, fontsize=D_TICK)
    _pad_ylim(ax_d1, [(r["ci_lo"], r["ci_hi"]) for r in t10],
              pad_bottom=0.0, pad_top=0.34, extra=(0.0,))
    ax_d1.set_ylim(0.0, ax_d1.get_ylim()[1])
    # the dotted 0.10 equality line is named in the white band above the bars:
    # printing the tag next to the line put it on the frontier bar (v2 defect)
    ax_d1.text(0.97, 0.965, "dotted: 0.10 equality", transform=ax_d1.transAxes,
               ha="right", va="top", fontsize=HOUSE_STYLE["note_size"],
               color=PALETTE["gray"])

    gaps = [S[(S["metric"] == M_GAP) & (S["refill_mode"] == regime)].iloc[0]
            for regime in REGIMES]
    ax_d2.bar(xs, [r["value"] for r in gaps], width=0.62,
              color=[REGIME_COLOR[regime] for regime in REGIMES], zorder=3)
    ax_d2.errorbar(xs, [r["value"] for r in gaps],
                   yerr=[[r["value"] - r["ci_lo"] for r in gaps],
                         [r["ci_hi"] - r["value"] for r in gaps]],
                   fmt="none", ecolor="black", elinewidth=0.9, capsize=3.0,
                   zorder=4)
    for x, r in zip(xs, gaps):
        # 9 pt below the cap, so the label reads as BELOW the bar and not as
        # an in-bar percentage (the v2 defect); pad_bottom leaves room
        _label(ax_d2, x, r["ci_lo"], f"{100 * r['value']:.1f}%", below=True,
               fontsize=HOUSE_STYLE["note_size"], extra_pt=9.0)
    ax_d2.axhline(0.0, color=PALETTE["gray"], ls="--", lw=1.0, zorder=2)
    ax_d2.set_xticks(xs)
    _cosmetics(ax_d2, HOUSE_STYLE, "D2 · newcomer gap (M1)",
               "refill regime", "(newcomer − incumbent) / incumbent")
    ax_d2.yaxis.label.set_fontsize(HOUSE_STYLE["tick_size"])
    ax_d2.set_xticklabels(REGIMES, fontsize=D_TICK)
    # top pad sized for the 3-line mean_prestige note in the white space
    # above the zero line (all gap bars are negative)
    _pad_ylim(ax_d2, [(r["ci_lo"], r["ci_hi"]) for r in gaps],
              pad_bottom=0.30, pad_top=0.46, extra=(0.0,))
    # white space above the zero line: the bars are all negative
    ax_d2.text(0.04, 0.96,
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
               frameon=False, bbox_to_anchor=(0.5, 0.095))

    footnote = (
        f"Sources: {S['source_file'].iloc[0]} (18 cells x 20 seeds "
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
    fig.text(0.01, 0.005, "\n".join(textwrap.fill(x, width=160)
                                    for x in footnote.split("\n")),
             fontsize=HOUSE_STYLE["note_size"], color=PALETTE["gray"],
             va="bottom")
    # (layout was fixed at axes creation: placements are measured on it)

    # panel C renders two statistics of ONE measure (bars = final step,
    # diamonds = time mean); the QA's single-metric contract needs both under
    # one name, so give it a QA-only view of the sidecar (the persisted CSV
    # keeps the two metric labels separate).
    qa_side = sidecar.copy()
    qa_side.loc[qa_side["metric"] == M_CORR_TM, "metric"] = M_CORR_FIN

    panels = [
        {"ax": ax_a, "metric": M_LOCKIN, "expect_lines": 2,
         "points_per_line": 3, "expect_bars": 6, "sidecar": S},
        {"ax": ax_b, "metric": M_FRONTIER, "expect_lines": 2,
         "points_per_line": 3, "expect_bars": 6, "sidecar": S,
         "headroom": 0.0},  # log axis: headroom asserted in decades above
        {"ax": ax_c, "metric": M_CORR_FIN, "expect_lines": 2,
         "expect_bars": 6, "sidecar": qa_side},
        {"ax": ax_d1, "metric": M_TOP10, "expect_lines": 1,
         "points_per_line": 3, "expect_bars": 3, "sidecar": S,
         "filters": {"g_entry": 0.1}},
        {"ax": ax_d2, "metric": M_GAP, "expect_lines": 1,
         "points_per_line": 3, "expect_bars": 3, "sidecar": S,
         "filters": {"g_entry": 0.1}},
    ]
    return fig, panels


def build(outdir: str = "output/report_notebook_v1",
          data_root=None) -> dict:
    c = load_runs(data_root)
    series = series_rows(c, data_root=data_root)
    anns = annotation_rows(c, data_root=data_root)
    sidecar = pd.concat([series, anns], ignore_index=True)

    fig, panels = _figure(sidecar)

    w, h = fig.get_size_inches() * fig.dpi
    assert_text_fits(fig, w, h)
    _no_text_overlap(fig)
    visual_qa.assert_figure_clean(fig, panels)  # v3 hardened contract

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
