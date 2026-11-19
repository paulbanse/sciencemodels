"""F10a - M2 choice-to-production dashboard (Section 6, Part 1).

Replaces the raw ``path_b_v2_gate2.png`` and the inconsistently pooled
``F10_m2_wedge_and_team_size.png`` with one coherent 2x2 mechanism
dashboard: partner choice (lambda) -> movement coupling
(``dyadic_move_pull``) -> teams, alignment, repetition - explicitly split
by ``mgstd5`` vs ``noisypr5`` so the Gate-2 PARTIAL verdict is
self-explanatory.

House contract: ``PALETTE`` / ``HOUSE_STYLE`` / ``_cosmetics`` /
``write_figure_data`` from ``diagram_templates``; zero hardcoded model
numbers - every plotted value, error bar and annotation is computed here
from the raw CSVs and written into the ``*_data.csv`` sidecar.

v3 repair (Team 2, F10a defects):
  * Panel B: one legend entry per movement pull (the pooled-label
    duplication is gone - only the ``mgstd5`` arm carries
    ``label=f"pull = {pull:g}"``), plus an explicit 2-item linestyle key
    (solid = ``mgstd5``, dashed = ``noisypr5``); the legend sits in the
    clear band above the ``pull = 0`` error bars / M1 reference line, and
    the grey reference lines are identified in the panel note instead of
    the legend.
  * Panel C: the 6-item legend is moved OUT of the curves (below the
    axes), the top-right note keeps a clear band above the highest curve.
  * Panel D: bar value labels are lifted ABOVE ``ci_hi`` (never on the
    error bar) with ylim headroom reserved for them.
  * ``build(outdir, data_root=None)`` / ``compute_table(data_root=None)``
    are parameterized (default: ``data/path_b_v1`` + ``data/path_b_v2``;
    one-line switch to the post-BUG-0006 archives once they land), and
    ``visual_qa.assert_figure_clean`` runs on every panel before
    ``savefig``.

Aggregation convention (stated in every axis label): panels (a)-(c) pool
over formulation x credit_sharing (4 cells x 20 seeds = 80 runs per
point); panel (d) shows the 2x2 formulation x credit breakdown at
``pull = 1.0`` on ``mgstd5``.  Error bars are normal-approximation 95%
CIs over runs (mean +/- 1.96 SE).  ``M1 (dyadic = off)`` references come
from ``data/path_b_v1/baseline_off.csv``.  Klann empirical benchmarks
(team size 2.65, repeated share 0.34) are unfitted diagnostic reference
lines only, labelled ``Empirical benchmark (unfitted)``.  lambda = 1 is
ALWAYS labelled ``closeness x local merit``, never "epistemic
complementarity".
"""

from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from diagram_templates import PALETTE, HOUSE_STYLE, _cosmetics, write_figure_data

REPO = _HERE.parent.parent
DATA = REPO / "data"

FNAME = "F10a_m2_choice_to_production"

#: unfitted Klann empirical benchmarks (diagnostic reference lines ONLY)
KLANN_TEAM_SIZE = 2.65
KLANN_REPEATED_SHARE = 0.34
BENCH_LABEL = "Empirical benchmark (unfitted)"

Z95 = 1.96  # normal-approximation 95% CI multiplier over runs

# visual grammar: hue = movement pull, linestyle = landscape, and in
# panels (c)/(d) hue = lambda with the tier palette (M2 lam=0 blue,
# M2 lam=1 green); M1 (dyadic = off) references are grey dashed.
PULL_COLOR = {0.0: PALETTE["purple"], 0.5: PALETTE["orange"], 1.0: PALETTE["pink"]}
LAM_COLOR = {0.0: PALETTE["blue"], 1.0: PALETTE["green"]}
LAND_LS = {"mgstd5": "-", "noisypr5": "--"}
M1_COLOR = PALETTE["gray"]
LAM_LABEL = {0.0: "λ = 0 (prestige matching)",
             1.0: "λ = 1 (closeness × local merit)"}
#: compact legend wording (the full names live in the axis labels/notes)
LAM_SHORT = {0.0: "λ = 0 · prestige matching",
             1.0: "λ = 1 · closeness × local merit"}

V1_NAME = re.compile(
    r"cell_lam(\d+)_(logit|threshold)_(equal|struct)_(mgstd5|noisypr5)\.csv")
V2_NAME = re.compile(
    r"cell_lam(\d+)_pull(\d+)_(logit|threshold)_(equal|struct)_(mgstd5|noisypr5)\.csv")


# ----------------------------------------------------------------- helpers
def _mean_ci(vals) -> tuple[float, float, float, int]:
    """Mean and normal-approximation 95% CI over runs (mean +/- 1.96 SE)."""
    v = np.asarray(vals, dtype=float)
    n = int(v.size)
    m = float(v.mean())
    half = Z95 * float(v.std(ddof=1)) / np.sqrt(n) if n > 1 else float("nan")
    return m, m - half, m + half, n


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _paths(data_root=None) -> dict:
    """Resolve the Path B archives (one place to switch archives).

    ``data_root`` is the directory that holds ``path_b_v1/`` and
    ``path_b_v2/`` (default: ``data/``).  Pass
    ``data/bug0006_remeasure_v2``-style roots here once Team 1 publishes
    the post-BUG-0006 Path B archives.
    """
    base = Path(data_root) if data_root is not None else DATA
    if not base.is_absolute():
        base = REPO / base
    return {
        "base": base,
        "off": base / "path_b_v1" / "baseline_off.csv",
        "off_summary": base / "path_b_v1" / "baseline_off_summary.csv",
        "v1": base / "path_b_v1" / "cells",
        "v2": base / "path_b_v2" / "cells",
    }


def assert_text_fits(fig, canvas_w: float, canvas_h: float) -> int:
    """Zero clipped AND zero overlapping text artists on the canvas.

    ``assert_text_fits`` pattern from ``turnover_lifecycle_figure.py``
    (tick labels are skipped: matplotlib keeps off-view ticks as invisible
    artists), extended with a pairwise text-text overlap audit among all
    checked artists.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    offenders = []
    items: list[tuple[str, str, object]] = []
    checked = [0]

    def check(artist, where):
        txt = artist.get_text()
        if not txt.strip():
            return
        checked[0] += 1
        bb = artist.get_window_extent(renderer=r)
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > canvas_w + 1 or bb.y1 > canvas_h + 1:
            offenders.append((where, txt[:50].replace("\n", " "),
                              "outside canvas "
                              f"(x0={bb.x0:.0f} y0={bb.y0:.0f} "
                              f"x1={bb.x1:.0f} y1={bb.y1:.0f})"))
        items.append((where, txt[:50].replace("\n", " "), bb))

    for t in fig.texts:
        check(t, "fig")
    for i, ax in enumerate(fig.axes):
        for loc, artist in (("center", ax.title),
                            ("left", getattr(ax, "_left_title", None)),
                            ("right", getattr(ax, "_right_title", None))):
            if artist is not None:
                check(artist, f"ax{i}.title({loc})")
        check(ax.xaxis.label, f"ax{i}.xlabel")
        check(ax.yaxis.label, f"ax{i}.ylabel")
        if ax.get_legend():
            for t in ax.get_legend().get_texts():
                check(t, f"ax{i}.legend")
        for t in ax.texts:
            check(t, f"ax{i}.text")
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            w1, s1, b1 = items[i]
            w2, s2, b2 = items[j]
            dx = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
            dy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
            if dx > 2 and dy > 2:
                offenders.append((f"{w1}|{w2}", f"{s1!r} vs {s2!r}",
                                  f"overlap {dx:.1f}x{dy:.1f} px"))
    _require(not offenders,
             "text fit violations: "
             + "; ".join(f"{w}: {s!r} ({d})" for w, s, d in offenders))
    return checked[0]


# ------------------------------------------------------------------- input
def _load_off(data_root=None) -> pd.DataFrame:
    """M1 (dyadic = off) reference runs: 40 runs, c = 0.5, g = 0.1."""
    paths = _paths(data_root)
    df = pd.read_csv(paths["off"])
    _require(len(df) == 40, f"baseline_off.csv: expected 40 runs, got {len(df)}")
    _require(set(df["land"].unique()) == {"mgstd5", "noisypr5"},
             f"baseline_off.csv: unexpected landscapes {sorted(df['land'].unique())}")
    _require((df.groupby("land").size() == 20).all(),
             "baseline_off.csv: expected 20 runs per landscape")
    _require((df["dyadic_mode"] == "off").all(),
             "baseline_off.csv: dyadic_mode must be 'off' everywhere")
    _require((df["initial_curiosity"] == 0.5).all(),
             "baseline_off.csv: initial_curiosity must be 0.5 everywhere")
    _require((df["agent_generation_time"] == 0.1).all(),
             "baseline_off.csv: agent_generation_time must be 0.1 everywhere")
    # landscape column / filename check against the archived summary
    summ = pd.read_csv(paths["off_summary"])
    for land, g in df.groupby("land"):
        row = summ[summ["land"] == land]
        _require(len(row) == 1, f"baseline_off_summary.csv: no row for {land}")
        for col in ("team_tm", "rep_tm", "links_tm", "c_ideal_tm",
                    "link_assort_tm"):
            _require(abs(float(g[col].mean()) - float(row[col].iloc[0])) <= 1e-9,
                     f"baseline_off vs summary drift {land}.{col}")
    return df


def _load_v1(data_root=None) -> pd.DataFrame:
    """Path B v1 lambda sweep at pull = 0: 40 cells x 20 seeds = 800 runs."""
    paths = _paths(data_root)
    rows = []
    files = sorted(paths["v1"].glob("cell_lam*.csv"))
    _require(len(files) == 40, f"v1 cells: expected 40 files, got {len(files)}")
    for f in files:
        m = V1_NAME.match(f.name)
        _require(m is not None, f"v1 cell name not parsable: {f.name}")
        d = pd.read_csv(f).copy()
        _require(len(d) == 20, f"{f.name}: expected 20 runs, got {len(d)}")
        d["dyadic_lambda"] = int(m.group(1)) / 100.0
        d["dyadic_move_pull"] = 0.0
        d["formulation"] = m.group(2)
        d["credit_sharing_axis"] = "equal_split" if m.group(3) == "equal" else "structural"
        d["landscape"] = m.group(4)
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def _load_v2(data_root=None) -> pd.DataFrame:
    """Path B v2 pull sweep: 48 cells x 20 seeds = 960 runs, pull in {0.5, 1}."""
    paths = _paths(data_root)
    rows = []
    files = sorted(paths["v2"].glob("cell_lam*_pull*.csv"))
    _require(len(files) == 48, f"v2 cells: expected 48 files, got {len(files)}")
    for f in files:
        m = V2_NAME.match(f.name)
        _require(m is not None, f"v2 cell name not parsable: {f.name}")
        d = pd.read_csv(f).copy()
        _require(len(d) == 20, f"{f.name}: expected 20 runs, got {len(d)}")
        _require((d["dyadic_lambda"] == int(m.group(1)) / 100.0).all(),
                 f"{f.name}: dyadic_lambda column disagrees with filename")
        _require((d["dyadic_move_pull"] == int(m.group(2)) / 100.0).all(),
                 f"{f.name}: dyadic_move_pull column disagrees with filename")
        _require((d["formulation"] == m.group(3)).all(),
                 f"{f.name}: formulation column disagrees with filename")
        _require((d["credit_sharing_axis"] == (
            "equal_split" if m.group(4) == "equal" else "structural")).all(),
            f"{f.name}: credit_sharing_axis column disagrees with filename")
        _require((d["landscape"] == m.group(5)).all(),
                 f"{f.name}: landscape column disagrees with filename")
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------- sidecar
def compute_table(data_root=None) -> pd.DataFrame:
    """Every plotted/annotated quantity as one tidy long table.

    Columns: panel, role ('point' | 'reference' | 'annotation'), series,
    metric, x, value, ci_lo, ci_hi, n, note.  Panels (a)-(c) pool over
    formulation x credit (4 cells x 20 seeds = 80 runs per point).
    """
    off = _load_off(data_root)
    v1 = _load_v1(data_root)
    v2 = _load_v2(data_root)
    _require(len(v1) + len(v2) + len(off) == 1800,
             "Path B v1+v2+off must total 1,800 runs")
    runs = pd.concat([v1, v2], ignore_index=True)

    rows: list[dict] = []

    def add(panel, role, series, metric, x, *, vals=None, value=None,
            note=""):
        if vals is not None:
            v, lo, hi, n = _mean_ci(vals)
        else:
            v, lo, hi, n = float(value), float("nan"), float("nan"), 0
        rows.append(dict(panel=panel, role=role, series=series, metric=metric,
                         x=x, value=v, ci_lo=lo, ci_hi=hi, n=n, note=note))

    lands = ["mgstd5", "noisypr5"]
    pulls = sorted(float(p) for p in runs["dyadic_move_pull"].unique())

    # ---- panel A/B points: pooled over formulation x credit ----
    for pull in pulls:
        sub_p = runs[runs["dyadic_move_pull"] == pull]
        lams = sorted(float(l) for l in sub_p["dyadic_lambda"].unique())
        for land in lands:
            for lam in lams:
                sel = sub_p[(sub_p["landscape"] == land)
                            & (sub_p["dyadic_lambda"] == lam)]
                _require(len(sel) == 80,
                         f"pull={pull} lam={lam} {land}: expected 80 runs "
                         f"(4 cells x 20 seeds), got {len(sel)}")
                series = f"pull = {pull:g} | {land}"
                add("A", "point", series, "links_tm", lam, vals=sel["links_tm"])
                add("B", "point", series, "team_tm", lam, vals=sel["team_tm"])

    # ---- panel A/B references: M1 (dyadic = off) per landscape ----
    for land in lands:
        sel = off[off["land"] == land]
        series = f"M1 (dyadic = off) | {land}"
        add("A", "reference", series, "links_tm", float("nan"),
            vals=sel["links_tm"],
            note="no declared links under dyadic_mode = off")
        add("B", "reference", series, "team_tm", float("nan"),
            vals=sel["team_tm"], note="baseline_off.csv, c = 0.5, g = 0.1")

    # ---- panel B reference: unfitted Klann team-size benchmark ----
    add("B", "reference", BENCH_LABEL, "team_tm", float("nan"),
        value=KLANN_TEAM_SIZE, note=BENCH_LABEL)

    # ---- panel A annotations: link_assort_tm on mgstd5 ----
    for pull in (0.0, 1.0):
        for lam in (0.0, 1.0):
            sel = runs[(runs["dyadic_move_pull"] == pull)
                       & (runs["landscape"] == "mgstd5")
                       & (runs["dyadic_lambda"] == lam)]
            add("A", "annotation", f"pull = {pull:g} | mgstd5",
                "link_assort_tm", lam, vals=sel["link_assort_tm"],
                note="declared-link prestige assortativity, mgstd5")

    # ---- panel B annotations: pull = 0 flatness range per landscape ----
    for land in lands:
        lams = sorted(float(l) for l in
                      runs[runs["dyadic_move_pull"] == 0.0]["dyadic_lambda"].unique())
        means = []
        for lam in lams:
            sel = runs[(runs["dyadic_move_pull"] == 0.0)
                       & (runs["landscape"] == land)
                       & (runs["dyadic_lambda"] == lam)]
            means.append(float(sel["team_tm"].mean()))
        add("B", "annotation", f"pull = 0 | {land}", "team_tm_pull0_min",
            float("nan"), value=min(means),
            note="min over lambda of pooled team_tm means at pull = 0")
        add("B", "annotation", f"pull = 0 | {land}", "team_tm_pull0_max",
            float("nan"), value=max(means),
            note="max over lambda of pooled team_tm means at pull = 0")

    # ---- panel C: c_ideal_tm and wedge_tm for lambda in {0, 1} ----
    for pull in pulls:
        for lam in (0.0, 1.0):
            for land in lands:
                sel = runs[(runs["dyadic_move_pull"] == pull)
                           & (runs["dyadic_lambda"] == lam)
                           & (runs["landscape"] == land)]
                _require(len(sel) == 80, f"panel C pull={pull} lam={lam} {land}")
                series = f"{LAM_LABEL[lam]} | {land}"
                add("C", "point", series, "c_ideal_tm", pull,
                    vals=sel["c_ideal_tm"])
                wedge = sel["c_local_tm"] - sel["c_ideal_tm"]
                add("C", "annotation", series, "wedge_tm", pull, vals=wedge,
                    note="relational wedge c_local_tm - c_ideal_tm "
                         "(positive = proxy failure)")
    for land in lands:
        sel = off[off["land"] == land]
        add("C", "reference", f"M1 (dyadic = off) | {land}", "c_ideal_tm",
            float("nan"), vals=sel["c_ideal_tm"],
            note="baseline_off.csv, c = 0.5, g = 0.1")

    # ---- panel D: formulation x credit breakdown at pull = 1.0, mgstd5 ----
    sub_d = v2[(v2["dyadic_move_pull"] == 1.0) & (v2["landscape"] == "mgstd5")]
    for formulation in sorted(sub_d["formulation"].unique()):
        for credit in sorted(sub_d["credit_sharing_axis"].unique()):
            for lam in (0.0, 1.0):
                sel = sub_d[(sub_d["formulation"] == formulation)
                            & (sub_d["credit_sharing_axis"] == credit)
                            & (sub_d["dyadic_lambda"] == lam)]
                _require(len(sel) == 20, f"panel D {formulation}/{credit}/lam={lam}")
                series = f"{formulation} + {credit} | {LAM_LABEL[lam]}"
                add("D", "point", series, "team_tm", float("nan"),
                    vals=sel["team_tm"], note="pull = 1.0, mgstd5")
                add("D", "point", series, "rep_tm", float("nan"),
                    vals=sel["rep_tm"], note="pull = 1.0, mgstd5")
    add("D", "reference", BENCH_LABEL, "rep_tm", float("nan"),
        value=KLANN_REPEATED_SHARE, note=BENCH_LABEL)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figure
def _errband(ax, g: pd.DataFrame, *, color: str, ls: str, marker: str,
             label: str | None, markerfacecolor=None) -> None:
    x = g["x"].to_numpy(float)
    y = g["value"].to_numpy(float)
    lo = g["ci_lo"].to_numpy(float)
    hi = g["ci_hi"].to_numpy(float)
    kw = {} if markerfacecolor is None else {"markerfacecolor": markerfacecolor}
    ax.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), color=color,
                linestyle=ls, marker=marker, markersize=5, linewidth=1.6,
                capsize=2.5, label=label, **kw)


def build(outdir: str = "output/report_notebook_v1",
          data_root=None) -> dict:
    out = Path(outdir)
    if not out.is_absolute():
        out = REPO / out
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{FNAME}.png"

    table = compute_table(data_root)
    off = _load_off(data_root)
    v1 = _load_v1(data_root)
    v2 = _load_v2(data_root)
    n_off, n_v1, n_v2 = len(off), len(v1), len(v2)

    def t(panel, role, metric, series, x=None):
        g = table[(table["panel"] == panel) & (table["role"] == role)
                  & (table["metric"] == metric) & (table["series"] == series)]
        if x is not None:
            g = g[g["x"] == x]
        return g

    fig = plt.figure(figsize=(11.6, 8.0))
    gs = fig.add_gridspec(2, 2, left=0.075, right=0.985, top=0.885,
                          bottom=0.245, hspace=0.52, wspace=0.30)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    gs_d = gs[1, 1].subgridspec(1, 2, width_ratios=[3.0, 1.0], wspace=0.40)
    axD = fig.add_subplot(gs_d[0])
    axRep = fig.add_subplot(gs_d[1])

    # ---- A: choice layer, links_tm vs lambda ----
    for pull in (0.0, 0.5, 1.0):
        for land in ("mgstd5", "noisypr5"):
            g = table[(table["panel"] == "A") & (table["role"] == "point")
                      & (table["metric"] == "links_tm")
                      & (table["series"] == f"pull = {pull:g} | {land}")]
            _errband(axA, g, color=PULL_COLOR[pull], ls=LAND_LS[land],
                     marker="o", label=f"pull = {pull:g}" if land == "mgstd5"
                     else None)
    axA.axhline(0.0, color=M1_COLOR, linestyle="--", linewidth=1.2, zorder=2)
    la = (float(t("A", "annotation", "link_assort_tm",
                  "pull = 0 | mgstd5", 0.0)["value"].iloc[0]),
          float(t("A", "annotation", "link_assort_tm",
                  "pull = 0 | mgstd5", 1.0)["value"].iloc[0]),
          float(t("A", "annotation", "link_assort_tm",
                  "pull = 1 | mgstd5", 0.0)["value"].iloc[0]),
          float(t("A", "annotation", "link_assort_tm",
                  "pull = 1 | mgstd5", 1.0)["value"].iloc[0]))
    m1_links = {land: float(t("A", "reference", "links_tm",
                              f"M1 (dyadic = off) | {land}")["value"].iloc[0])
                for land in ("mgstd5", "noisypr5")}
    axA.text(0.985, 0.045, (
        f"M1 (dyadic = off): {m1_links['mgstd5']:.2f} links (both landscapes)\n"
        f"declared-link assortativity link_assort_tm (mgstd5):\n"
        f"pull = 0 {la[0]:+.2f} → {la[1]:+.2f}; pull = 1 {la[2]:+.2f} → {la[3]:+.2f}"),
        transform=axA.transAxes, fontsize=HOUSE_STYLE["note_size"],
        color=PALETTE["gray"], va="bottom", ha="right",
        bbox=dict(facecolor="white", edgecolor="none", pad=1.6))
    axA.legend(loc="upper left", fontsize=HOUSE_STYLE["tick_size"] - 1,
               title="movement pull", title_fontsize=HOUSE_STYLE["tick_size"] - 1,
               frameon=True, facecolor="white", edgecolor="none",
               framealpha=1.0, handlelength=2.2, labelspacing=0.3,
               borderpad=0.3)
    _cosmetics(axA, HOUSE_STYLE,
               "A · Choice layer: active declared links (links_tm)",
               "partner choice  λ = dyadic_lambda",
               "links_tm (time mean · run mean)")
    axA.set_xticks([0.0, 0.5, 1.0])
    axA.set_xlim(-0.06, 1.06)
    axA.set_ylim(0, 11.0)
    axA.yaxis.labelpad = 9.0

    # ---- B: production layer, team_tm vs lambda ----
    for pull in (0.0, 0.5, 1.0):
        for land in ("mgstd5", "noisypr5"):
            g = table[(table["panel"] == "B") & (table["role"] == "point")
                      & (table["metric"] == "team_tm")
                      & (table["series"] == f"pull = {pull:g} | {land}")]
            # v3: ONE legend entry per pull (mgstd5 arm only) - the earlier
            # duplicate "pull = x" labels came from labelling both arms.
            _errband(axB, g, color=PULL_COLOR[pull], ls=LAND_LS[land],
                     marker="o", label=f"pull = {pull:g}" if land == "mgstd5"
                     else None)
    m1_team = {land: float(t("B", "reference", "team_tm",
                             f"M1 (dyadic = off) | {land}")["value"].iloc[0])
               for land in ("mgstd5", "noisypr5")}
    # reference lines: identified in the panel note (no legend clutter)
    axB.axhline(m1_team["mgstd5"], color=M1_COLOR, linestyle="--",
                linewidth=1.2, zorder=2)
    axB.axhline(m1_team["noisypr5"], color=M1_COLOR, linestyle=":",
                linewidth=1.2, zorder=2)
    axB.axhline(KLANN_TEAM_SIZE, color=M1_COLOR, linestyle=":", linewidth=1.4,
                zorder=2)
    b0 = {land: (float(t("B", "annotation", "team_tm_pull0_min",
                         f"pull = 0 | {land}")["value"].iloc[0]),
                 float(t("B", "annotation", "team_tm_pull0_max",
                         f"pull = 0 | {land}")["value"].iloc[0]))
          for land in ("mgstd5", "noisypr5")}
    b1 = {(land, lam): float(t("B", "point", "team_tm",
                               f"pull = 1 | {land}", lam)["value"].iloc[0])
          for land in ("mgstd5", "noisypr5") for lam in (0.0, 1.0)}
    axB.text(0.015, 0.025, (
        f"pull = 0 inert (flat): {b0['mgstd5'][0]:.2f}–{b0['mgstd5'][1]:.2f} "
        f"(mgstd5), {b0['noisypr5'][0]:.2f}–{b0['noisypr5'][1]:.2f} (noisypr5)\n"
        f"pull = 1 (mgstd5): {m1_team['mgstd5']:.2f} (M1) → {b1[('mgstd5', 0.0)]:.2f} "
        f"(λ = 0) → {b1[('mgstd5', 1.0)]:.2f} (λ = 1)\n"
        f"pull = 1 (noisypr5): {b1[('noisypr5', 0.0)]:.2f} → {b1[('noisypr5', 1.0)]:.2f}, "
        f"below the {KLANN_TEAM_SIZE:.2f} benchmark"),
        transform=axB.transAxes, fontsize=HOUSE_STYLE["note_size"] - 0.5,
        color=PALETTE["gray"], va="bottom", ha="left",
        bbox=dict(facecolor="white", edgecolor="none", pad=1.6))
    # v3 legend: 3 movement-pull entries (hue) + a clean 2-item linestyle
    # key (solid = mgstd5, dashed = noisypr5), in the empty band above the
    # pull = 0 error bars and the M1 reference line.
    land_key = [
        Line2D([0], [0], color=PALETTE["gray"], linestyle="-", linewidth=1.6,
               label="mgstd5 (solid)"),
        Line2D([0], [0], color=PALETTE["gray"], linestyle="--", linewidth=1.6,
               label="noisypr5 (dashed)"),
    ]
    pull_key = [
        Line2D([0], [0], color=PULL_COLOR[p], linestyle="-", marker="o",
               markersize=5, linewidth=1.6, label=f"pull = {p:g}")
        for p in (0.0, 0.5, 1.0)
    ]
    axB.legend(handles=pull_key + land_key, loc="upper center",
               bbox_to_anchor=(0.5, 0.995), ncol=2, frameon=True,
               facecolor="white", edgecolor="none", framealpha=1.0,
               fontsize=HOUSE_STYLE["tick_size"] - 1.5,
               handlelength=1.8, columnspacing=1.2, labelspacing=0.25,
               borderpad=0.1)
    _cosmetics(axB, HOUSE_STYLE,
               "B · Production layer: co-harvest team size (team_tm)",
               "partner choice  λ = dyadic_lambda",
               "team_tm (time mean · run mean)")
    axB.set_xticks([0.0, 0.5, 1.0])
    axB.set_xlim(-0.06, 1.06)
    axB.set_ylim(0, 12.8)
    axB.yaxis.labelpad = 9.0

    # ---- C: signal alignment, c_ideal_tm vs pull ----
    for lam in (0.0, 1.0):
        for land in ("mgstd5", "noisypr5"):
            series = f"{LAM_LABEL[lam]} | {land}"
            g = table[(table["panel"] == "C") & (table["role"] == "point")
                      & (table["metric"] == "c_ideal_tm")
                      & (table["series"] == series)]
            _errband(axC, g, color=LAM_COLOR[lam], ls=LAND_LS[land],
                     marker="o", label=f"λ={lam:g} · {land}")
    m1_ci = {land: float(t("C", "reference", "c_ideal_tm",
                           f"M1 (dyadic = off) | {land}")["value"].iloc[0])
             for land in ("mgstd5", "noisypr5")}
    for land in ("mgstd5", "noisypr5"):
        axC.axhline(m1_ci[land], color=M1_COLOR, linestyle="--" if land == "mgstd5"
                    else ":", linewidth=1.2, zorder=2,
                    label=f"M1 (dyadic off) · {land}")
    c_end = {(lam, land): float(t("C", "point", "c_ideal_tm",
                                  f"{LAM_LABEL[lam]} | {land}", 1.0)["value"].iloc[0])
             for lam in (0.0, 1.0) for land in ("mgstd5", "noisypr5")}
    w_end = {(lam, land): float(t("C", "annotation", "wedge_tm",
                                  f"{LAM_LABEL[lam]} | {land}", 1.0)["value"].iloc[0])
             for lam in (0.0, 1.0) for land in ("mgstd5", "noisypr5")}
    axC.text(0.985, 0.985, (
        f"at pull = 1: mgstd5 λ=0 {m1_ci['mgstd5']:+.3f} → "
        f"{c_end[(0.0, 'mgstd5')]:+.3f} (below M1), λ=1 → "
        f"{c_end[(1.0, 'mgstd5')]:+.3f} (above M1)\n"
        f"noisypr5 λ=0 {m1_ci['noisypr5']:+.3f} → {c_end[(0.0, 'noisypr5')]:+.3f}, "
        f"λ=1 → {c_end[(1.0, 'noisypr5')]:+.3f}\n"
        f"wedge_tm (mgstd5): λ=0 {w_end[(0.0, 'mgstd5')]:+.3f} → "
        f"λ=1 {w_end[(1.0, 'mgstd5')]:+.3f} (sign inversion)"),
        transform=axC.transAxes, fontsize=HOUSE_STYLE["note_size"] - 0.5,
        color=PALETTE["gray"], va="top", ha="right",
        bbox=dict(facecolor="white", edgecolor="none", pad=1.6))
    _cosmetics(axC, HOUSE_STYLE,
               "C · Signal alignment vs movement coupling",
               "movement coupling  dyadic_move_pull",
               "Corr(prestige, ideal merit) (time mean · run mean)")
    axC.set_xticks([0.0, 0.5, 1.0])
    axC.set_xlim(-0.08, 1.08)
    axC.yaxis.labelpad = 9.0
    # v3: top band keeps the note clear of the highest curve and of the y
    # ticks; the 6-item legend is moved OUT of the curves (below the axes).
    axC.set_ylim(-0.035, 0.56)
    axC.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3,
               frameon=False, fontsize=HOUSE_STYLE["tick_size"] - 2,
               handlelength=2.2, columnspacing=1.6, labelspacing=0.3,
               borderpad=0.1)

    # ---- D: formulation x credit at pull = 1.0, mgstd5 + rep_tm strip ----
    combos = [("logit", "equal_split"), ("logit", "structural"),
              ("threshold", "equal_split"), ("threshold", "structural")]
    xs = np.arange(len(combos), dtype=float)
    vmax_hi = 0.0
    width = 0.36
    bar_data: dict[float, tuple] = {}
    for k, lam in enumerate((0.0, 1.0)):
        vals, los, his = [], [], []
        for formulation, credit in combos:
            g = table[(table["panel"] == "D") & (table["role"] == "point")
                      & (table["metric"] == "team_tm")
                      & (table["series"] == f"{formulation} + {credit} | {LAM_LABEL[lam]}")]
            vals.append(float(g["value"].iloc[0]))
            los.append(float(g["ci_lo"].iloc[0]))
            his.append(float(g["ci_hi"].iloc[0]))
        vmax_hi = max(vmax_hi, max(his))
        pos = xs + (k - 0.5) * width
        axD.bar(pos, vals, width=width * 0.92, color=LAM_COLOR[lam],
                yerr=np.vstack([np.array(vals) - np.array(los),
                                np.array(his) - np.array(vals)]),
                capsize=3, label=LAM_SHORT[lam], edgecolor="white")
        bar_data[lam] = (pos, vals, los, his)
    axD.set_xticks(xs)
    axD.set_xticklabels(["logit\n+ equal", "logit\n+ structural",
                         "threshold\n+ equal", "threshold\n+ structural"],
                        fontsize=HOUSE_STYLE["tick_size"] - 2)
    _cosmetics(axD, HOUSE_STYLE,
               "D · Robustness at pull = 1.0",
               "formulation + credit_sharing (mgstd5)",
               "team_tm (time mean · run mean)")
    axD.set_ylim(0, vmax_hi * 1.40)
    axD.yaxis.labelpad = 9.0
    # v3: value labels ABOVE the upper CI cap (they sat on the error bar
    # before), with the ylim headroom reserved for exactly that.
    label_pad = 0.025 * (axD.get_ylim()[1] - axD.get_ylim()[0])
    for lam in (0.0, 1.0):
        pos, vals, _los, his = bar_data[lam]
        for x_, v_, hi_ in zip(pos, vals, his):
            axD.text(x_, hi_ + label_pad, f"{v_:.2f}", ha="center", va="bottom",
                     fontsize=HOUSE_STYLE["tick_size"] - 1.5,
                     color=PALETTE["gray"])
    axD.legend(loc="upper left", bbox_to_anchor=(0.0, 0.995), ncol=1,
               fontsize=HOUSE_STYLE["tick_size"] - 2, frameon=False,
               handlelength=1.6, labelspacing=0.3, borderpad=0.1)

    # rep_tm strip paired with the bars (same 8 cells)
    jitter = (-0.12, -0.04, 0.04, 0.12)
    for k, lam in enumerate((0.0, 1.0)):
        ys = []
        for formulation, credit in combos:
            g = table[(table["panel"] == "D") & (table["role"] == "point")
                      & (table["metric"] == "rep_tm")
                      & (table["series"] == f"{formulation} + {credit} | {LAM_LABEL[lam]}")]
            ys.append(float(g["value"].iloc[0]))
        axRep.plot([k + j for j in jitter], ys, marker="D", linestyle="none",
                   color=LAM_COLOR[lam], markersize=4.5,
                   label=f"λ={lam:g} · 4 cells")
    axRep.axhline(KLANN_REPEATED_SHARE, color=M1_COLOR, linestyle=":",
                  linewidth=1.2)
    # v3: the strip is narrow - keep the benchmark label inside the axes
    # (small type, two lines) instead of letting it run into the margin.
    axRep.text(0.02, KLANN_REPEATED_SHARE + 0.006,
               f"{KLANN_REPEATED_SHARE:.2f} Empirical\nbenchmark (unfitted)",
               transform=axRep.get_yaxis_transform(), fontsize=6.0,
               color=PALETTE["gray"], va="bottom", ha="left",
               bbox=dict(facecolor="white", edgecolor="none", pad=1.2))
    rep_vals = table[(table["panel"] == "D") & (table["metric"] == "rep_tm")
                     & (table["role"] == "point")]["value"]
    rep_lo, rep_hi = float(rep_vals.min()), float(rep_vals.max())
    axRep.set_xticks([0, 1])
    axRep.set_xticklabels(["λ = 0", "λ = 1"], fontsize=HOUSE_STYLE["tick_size"] - 1)
    _cosmetics(axRep, HOUSE_STYLE,
               "D′ · Repeated share",
               "partner choice",
               "rep_tm (time mean · run mean)")
    axRep.title.set_fontsize(HOUSE_STYLE["note_size"])
    axRep.set_xlim(-0.55, 1.55)
    axRep.yaxis.labelpad = 8.0
    axRep.set_ylim(KLANN_REPEATED_SHARE * 0.8, 1.06)
    axRep.text(0.5, 0.99, f"flat: {rep_lo:.3f}–{rep_hi:.3f}",
               transform=axRep.transAxes, fontsize=6.5,
               color=PALETTE["gray"], va="top", ha="center")

    # ---- figure frame, legend, footnote ----
    fig.text(0.5, 0.995, "F10a · How M2 works end-to-end: partner choice (λ) → "
             "movement coupling → teams, alignment, repetition\n"
             "(panels a–c pooled over formulation × credit, 4 cells × 20 seeds)",
             fontsize=HOUSE_STYLE["title_size"] + 1, ha="center", va="top")
    note_lines = [
        f"Sources (read-only): data/path_b_v1/baseline_off.csv ({n_off} runs, M1 dyadic = off), "
        f"data/path_b_v1/cells ({n_v1} runs), data/path_b_v2/cells ({n_v2} runs) — "
        f"{n_off + n_v1 + n_v2:,} Path B runs total. Panels (a)–(c): means over runs pooled over "
        f"formulation × credit (4 cells × 20 seeds = 80 runs per point); error bars 95% CI = "
        f"mean ± 1.96 SE over runs. Panel (d): pull = 1.0, mgstd5, per cell (20 seeds).",
        "Solid = mgstd5, dashed = noisypr5. λ = 1 (closeness × local merit); λ = 0 "
        "(prestige matching). Grey dashed/dotted = M1 (dyadic = off) references from "
        "baseline_off.csv. Klann team size 2.65 and repeated share 0.34: Empirical benchmark "
        "(unfitted) — model-internal, not calibrated.",
    ]
    note = "\n".join(textwrap.fill(t, width=176) for t in note_lines)
    fig.text(0.01, 0.010, note, fontsize=HOUSE_STYLE["note_size"],
             color=PALETTE["gray"], va="bottom")

    n_text = assert_text_fits(fig, fig.get_size_inches()[0] * fig.dpi,
                              fig.get_size_inches()[1] * fig.dpi)

    import visual_qa  # v3 hardened contract (rendered artists + collisions)
    visual_qa.assert_figure_clean(fig, [
        {"ax": axA, "metric": "links_tm", "expect_lines": 6,
         "sidecar": table, "filters": {"panel": "A", "role": "point"}},
        {"ax": axB, "metric": "team_tm", "expect_lines": 6,
         "sidecar": table, "filters": {"panel": "B", "role": "point"}},
        {"ax": axC, "metric": "c_ideal_tm", "expect_lines": 4,
         "sidecar": table, "filters": {"panel": "C", "role": "point"}},
        {"ax": axD, "metric": "team_tm", "expect_bars": 8,
         "sidecar": table, "filters": {"panel": "D", "role": "point"}},
        {"ax": axRep, "metric": "rep_tm", "expect_lines": 2,
         "points_per_line": 4, "sidecar": table,
         "ci_cols": ("value", "value"),  # markers only: no CI whiskers drawn
         "filters": {"panel": "D", "role": "point"}},
    ])

    fig.savefig(png, dpi=HOUSE_STYLE["dpi"])
    fig.savefig(png.with_suffix(".pdf"))
    plt.close(fig)

    csv = write_figure_data(table, png)
    return {"png": str(png), "pdf": str(png.with_suffix(".pdf")),
            "data_csv": str(csv), "n_text_artists_checked": n_text}


if __name__ == "__main__":
    print(build())
