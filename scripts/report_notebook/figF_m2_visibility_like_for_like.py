"""F10b - M2 on the full visibility axis, like-for-like (Section 6, Part 2).

Places ``M2`` (``pull = 1.0``, ``logit + equal_split``) directly alongside
``M0`` and ``M1`` across the full visibility axis ``xi = 1 - c`` in
``[0.1, 0.9]`` on ``mgstd5``, using STRICTLY like-for-like time-mean
alignment (``c_ideal_tm``) - resolving Figure 1 (F6)'s mixed-statistic
distortion and exposing M2's Stage-2 mob collapse.

Inputs (read-only):
  * ``data/remeasure_v1/w_v05_summary.csv`` (M0 g=0, M1 g=0.1; vanishing
    = 0.5 pin) + ``w_v05_runs.csv`` for the 95% CIs over the 20 seeds;
  * ``data/fig1_extension_v1/cells/m2_*.csv`` (xi in {0.1, 0.3, 0.7, 0.9});
  * ``data/path_b_v2/cells/cell_lam{000,100}_pull100_logit_equal_mgstd5.csv``
    (xi = 0.5); 200 M2 runs total.

House contract: ``PALETTE`` / ``HOUSE_STYLE`` / ``_cosmetics`` /
``write_figure_data`` from ``diagram_templates``; F6's exact Stage 1 /
kink / Stage 2 bands; F6's exact tier palette (M0 purple, M1 gray dashed
hollow, M2 lambda=0 blue dashed, M2 lambda=1 green); zero hardcoded model
numbers - every plotted value and annotation is computed from the raw
CSVs and written into the ``*_data.csv`` sidecar; ``assert_text_fits``
guards the canvas.  Klann's 2.65 team-size line is labelled ``Empirical
benchmark (unfitted)``.  lambda = 1 is ALWAYS labelled ``closeness x
local merit``.  The footnote states the vanishing = 0.5 pin transparently
and the cross-check +0.1077 (here) vs +0.1065 (path_b_v1, vanishing 0.9).
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

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from diagram_templates import PALETTE, HOUSE_STYLE, _cosmetics, write_figure_data

REPO = _HERE.parent.parent
DATA = REPO / "data"

KLANN_TEAM_SIZE = 2.65
BENCH_LABEL = "Empirical benchmark (unfitted)"
Z95 = 1.96  # normal-approximation 95% CI multiplier over runs

# F6's tier grammar (exact)
SERIES_STYLE = {
    "M0 (g = 0)": (PALETTE["purple"], "-", "o", None),
    "M1 (g = 0.1)": (PALETTE["gray"], "--", "o", "none"),
    "M2 λ = 0 (prestige matching)": (PALETTE["blue"], "--", "s", None),
    "M2 λ = 1 (closeness × local merit)": (PALETTE["green"], "-", "^", None),
}
M0_SERIES = "M0 (g = 0)"
M1_SERIES = "M1 (g = 0.1)"
M2_SERIES = {0.0: "M2 λ = 0 (prestige matching)",
             1.0: "M2 λ = 1 (closeness × local merit)"}

M2_NAME = re.compile(
    r"m2_cell_lam(\d+)_pull(\d+)_(logit|threshold)_(equal|struct)_(\w+)_cur(\d+)\.csv")


def _mean_ci(vals) -> tuple[float, float, float, int]:
    v = np.asarray(vals, dtype=float)
    n = int(v.size)
    m = float(v.mean())
    half = Z95 * float(v.std(ddof=1)) / np.sqrt(n) if n > 1 else float("nan")
    return m, m - half, m + half, n


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def assert_text_fits(fig, canvas_w: float, canvas_h: float) -> int:
    """Zero clipped AND zero overlapping text artists on the canvas."""
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
def _load_remeasure() -> tuple[pd.DataFrame, pd.DataFrame]:
    """M0/M1 rows: w_v05_summary.csv (means) + w_v05_runs.csv (CIs)."""
    summ = pd.read_csv(DATA / "remeasure_v1" / "w_v05_summary.csv")
    runs = pd.read_csv(DATA / "remeasure_v1" / "w_v05_runs.csv")
    sel_s = summ[(summ["landscape"] == "mgstd5")
                 & (summ["use_visibility_reward"] == True)]  # noqa: E712
    sel_r = runs[(runs["landscape"] == "mgstd5")
                 & (runs["use_visibility_reward"] == True)]  # noqa: E712
    _require(len(sel_s) == 10,
             f"w_v05_summary mgstd5+vis: expected 10 rows, got {len(sel_s)}")
    _require(len(sel_r) == 200,
             f"w_v05_runs mgstd5+vis: expected 200 runs, got {len(sel_r)}")
    _require(set(sel_s["agent_generation_time"].unique()) == {0.0, 0.1},
             "w_v05_summary: expected g in {0.0 (M0), 0.1 (M1)}")
    _require((sel_s["n_seeds"] == 20).all(), "w_v05_summary: n_seeds != 20")
    _require(set(np.round(sel_s["curiosity"], 1)) == {0.1, 0.3, 0.5, 0.7, 0.9},
             "w_v05_summary: unexpected curiosity grid")
    _require((sel_r.groupby(["agent_generation_time", "curiosity"]).size()
              == 20).all(), "w_v05_runs: expected 20 seeds per cell")
    # summary means must equal run means (aggregation identity)
    agg = sel_r.groupby(["agent_generation_time", "curiosity"]).mean(numeric_only=True)
    for _, row in sel_s.iterrows():
        a = agg.loc[(row["agent_generation_time"], row["curiosity"])]
        for col in ("harvested_fin", "c_ideal_tm", "wedge_local_minus_ideal_tm"):
            _require(abs(float(a[col]) - float(row[col])) <= 1e-9,
                     f"w_v05 summary vs runs drift: {col}")
    return sel_s, sel_r


def _load_m2() -> pd.DataFrame:
    """M2 (pull = 1.0, logit + equal_split) across xi = 0.1..0.9: 200 runs."""
    rows = []
    files = sorted((DATA / "fig1_extension_v1" / "cells").glob("m2_*.csv"))
    _require(len(files) == 8,
             f"fig1_extension m2 cells: expected 8 files, got {len(files)}")
    for f in files:
        m = M2_NAME.match(f.name)
        _require(m is not None, f"m2 cell name not parsable: {f.name}")
        d = pd.read_csv(f).copy()
        _require(len(d) == 20, f"{f.name}: expected 20 runs, got {len(d)}")
        _require((d["dyadic_lambda"] == int(m.group(1)) / 100.0).all(),
                 f"{f.name}: dyadic_lambda mismatch")
        _require((d["dyadic_move_pull"] == 1.0).all(),
                 f"{f.name}: expected pull = 1.0")
        _require((d["formulation"] == "logit").all(), f"{f.name}: formulation")
        _require((d["credit_sharing_axis"] == "equal_split").all(),
                 f"{f.name}: credit_sharing_axis")
        _require((d["landscape"] == "mgstd5").all(), f"{f.name}: landscape")
        d["xi"] = np.round(1.0 - d["initial_curiosity"], 1)
        rows.append(d)
    # xi = 0.5 anchor cells from path_b_v2
    for lam in (0.0, 1.0):
        tag = "000" if lam == 0.0 else "100"
        f = DATA / "path_b_v2" / "cells" / f"cell_lam{tag}_pull100_logit_equal_mgstd5.csv"
        d = pd.read_csv(f).copy()
        _require(len(d) == 20, f"{f.name}: expected 20 runs, got {len(d)}")
        _require((d["dyadic_lambda"] == lam).all(), f"{f.name}: dyadic_lambda")
        _require((d["dyadic_move_pull"] == 1.0).all(), f"{f.name}: pull")
        _require((d["initial_curiosity"] == 0.5).all(), f"{f.name}: curiosity")
        d["xi"] = 0.5
        rows.append(d)
    m2 = pd.concat(rows, ignore_index=True)
    _require(len(m2) == 200, f"M2 xi-extension: expected 200 runs, got {len(m2)}")
    return m2


# --------------------------------------------------------------- sidecar
def compute_table() -> pd.DataFrame:
    """Every plotted/annotated quantity as one tidy long table.

    Columns: panel, role ('point' | 'reference' | 'annotation'), series,
    metric, x (xi), value, ci_lo, ci_hi, n, note.
    """
    summ, runs = _load_remeasure()
    m2 = _load_m2()

    rows: list[dict] = []

    def add(panel, role, series, metric, x, *, vals=None, value=None,
            note=""):
        if vals is not None:
            v, lo, hi, n = _mean_ci(vals)
        else:
            v, lo, hi, n = float(value), float("nan"), float("nan"), 0
        rows.append(dict(panel=panel, role=role, series=series, metric=metric,
                         x=x, value=v, ci_lo=lo, ci_hi=hi, n=n, note=note))

    xis = [0.1, 0.3, 0.5, 0.7, 0.9]

    # ---- M0 / M1 points: means from w_v05_summary, CIs from runs ----
    for g, series in ((0.0, M0_SERIES), (0.1, M1_SERIES)):
        for xi in xis:
            cur = round(1.0 - xi, 1)
            row = summ[(np.round(summ["agent_generation_time"], 2) == g)
                       & (np.round(summ["curiosity"], 1) == cur)]
            _require(len(row) == 1, f"w_v05_summary row g={g} cur={cur}")
            sel = runs[(np.round(runs["agent_generation_time"], 2) == g)
                       & (np.round(runs["curiosity"], 1) == cur)]
            for metric, col in (("harvested_share", "harvested_fin"),
                                ("c_ideal_tm", "c_ideal_tm")):
                m, lo, hi, n = _mean_ci(sel[col])
                rows.append(dict(
                    panel="A" if metric == "harvested_share" else "B",
                    role="point", series=series, metric=metric, x=xi,
                    value=float(row[col].iloc[0]), ci_lo=lo, ci_hi=hi, n=n,
                    note="w_v05_summary mean; CI over w_v05_runs (20 seeds); "
                         "vanishing = 0.5"))

    # ---- M2 points from raw cell runs ----
    for lam, series in M2_SERIES.items():
        for xi in xis:
            sel = m2[(m2["dyadic_lambda"] == lam) & (m2["xi"] == xi)]
            _require(len(sel) == 20, f"M2 lam={lam} xi={xi}: expected 20 runs")
            for panel, metric, col in (("A", "harvested_share",
                                        "percentage_knowledge_harvested"),
                                       ("B", "c_ideal_tm", "c_ideal_tm"),
                                       ("C", "team_tm", "team_tm")):
                add(panel, "point", series, metric, xi, vals=sel[col],
                    note="M2 pull = 1.0, logit + equal_split, mgstd5")

    # ---- reference: unfitted Klann team-size benchmark ----
    add("C", "reference", BENCH_LABEL, "team_tm", float("nan"),
        value=KLANN_TEAM_SIZE, note=BENCH_LABEL)

    # ---- annotations computed from the plotted aggregates ----
    entry_arms = (M1_SERIES, M2_SERIES[0.0], M2_SERIES[1.0])

    def arm_values(panel, metric, series_list, xi_list):
        g = [r for r in rows if r["panel"] == panel and r["role"] == "point"
             and r["metric"] == metric and r["series"] in series_list
             and r["x"] in xi_list]
        return [r["value"] for r in g]

    plateau = arm_values("A", "harvested_share", entry_arms,
                         [0.1, 0.3, 0.5])
    collapse = arm_values("A", "harvested_share", entry_arms, [0.9])
    add("A", "annotation", "entry arms (M1 + M2)", "plateau_harvested_min",
        float("nan"), value=min(plateau),
        note="min harvested_share over entry arms at xi <= 0.5")
    add("A", "annotation", "entry arms (M1 + M2)", "plateau_harvested_max",
        float("nan"), value=max(plateau),
        note="max harvested_share over entry arms at xi <= 0.5")
    add("A", "annotation", "entry arms (M1 + M2)", "stage2_collapse_min",
        float("nan"), value=min(collapse),
        note="min harvested_share over entry arms at xi = 0.9")
    add("A", "annotation", "entry arms (M1 + M2)", "stage2_collapse_max",
        float("nan"), value=max(collapse),
        note="max harvested_share over entry arms at xi = 0.9")

    small = arm_values("C", "team_tm", list(M2_SERIES.values()),
                       [0.1, 0.3, 0.5])
    mob = arm_values("C", "team_tm", list(M2_SERIES.values()), [0.7, 0.9])
    add("C", "annotation", "M2 (both λ)", "team_small_min", float("nan"),
        value=min(small), note="min team_tm at xi <= 0.5")
    add("C", "annotation", "M2 (both λ)", "team_small_max", float("nan"),
        value=max(small), note="max team_tm at xi <= 0.5")
    add("C", "annotation", "M2 (both λ)", "team_mob_min", float("nan"),
        value=min(mob), note="min team_tm at xi in {0.7, 0.9}")
    add("C", "annotation", "M2 (both λ)", "team_mob_max", float("nan"),
        value=max(mob), note="max team_tm at xi in {0.7, 0.9}")

    # ---- cross-check rows: M1 time-mean alignment at xi = 0.5 ----
    m1_xi05 = float(summ[(np.round(summ["agent_generation_time"], 2) == 0.1)
                         & (np.round(summ["curiosity"], 1) == 0.5)]
                    ["c_ideal_tm"].iloc[0])
    add("B", "annotation", M1_SERIES, "crosscheck_xi05_remeasure_vanishing05",
        0.5, value=m1_xi05,
        note="c_ideal_tm at xi = 0.5, remeasure_v1 w_v05 (vanishing = 0.5)")
    pb = pd.read_csv(DATA / "path_b_v1" / "baseline_off_summary.csv")
    m1_pb = float(pb[pb["land"] == "mgstd5"]["c_ideal_tm"].iloc[0])
    add("B", "annotation", M1_SERIES, "crosscheck_xi05_pathb_vanishing09",
        0.5, value=m1_pb,
        note="c_ideal_tm at xi = 0.5, path_b_v1 baseline_off "
             "(vanishing = 0.9)")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figure
def _bands(ax) -> None:
    """F6's exact two-stage bands: Stage 1, welfare-kink band, Stage 2."""
    ax.axvspan(0.3, 0.6, color=PALETTE["gray_light"], alpha=0.18, zorder=0)
    ax.axvspan(0.6, 0.8, facecolor=PALETTE["purple_light"], alpha=0.18,
               hatch="///", edgecolor=PALETTE["gray"], linewidth=0, zorder=0)
    for xv in (0.6, 0.8):
        ax.axvline(xv, ls="--", lw=0.9, color=PALETTE["gray"], zorder=1)
    ax.axvspan(0.8, 1.0, color=PALETTE["gray_light"], alpha=0.35, zorder=0)
    ax.text(0.45, 0.96, "Stage 1", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=HOUSE_STYLE["tick_size"],
            color=PALETTE["gray"])
    ax.text(0.70, 0.96, "kink", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=HOUSE_STYLE["tick_size"],
            color=PALETTE["gray"])
    ax.text(0.90, 0.96, "Stage 2", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=HOUSE_STYLE["tick_size"],
            color=PALETTE["gray"])


def build(outdir: str = "output/report_notebook_v1") -> dict:
    out = Path(outdir)
    if not out.is_absolute():
        out = REPO / out
    out.mkdir(parents=True, exist_ok=True)
    png = out / "F10b_m2_visibility_like_for_like.png"

    table = compute_table()
    summ, runs = _load_remeasure()
    m2 = _load_m2()
    n_m2 = len(m2)

    def series_rows(panel, metric, series):
        return table[(table["panel"] == panel) & (table["role"] == "point")
                     & (table["metric"] == metric)
                     & (table["series"] == series)].sort_values("x")

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(13.2, 4.6))

    order = [M0_SERIES, M1_SERIES, M2_SERIES[0.0], M2_SERIES[1.0]]

    # ---- (a) macro efficiency: harvested knowledge share ----
    for s in order:
        g = series_rows("A", "harvested_share", s)
        color, ls, marker, mfc = SERIES_STYLE[s]
        kw = {} if mfc is None else {"markerfacecolor": mfc}
        axA.errorbar(g["x"], g["value"],
                     yerr=np.vstack([g["value"] - g["ci_lo"],
                                     g["ci_hi"] - g["value"]]),
                     color=color, linestyle=ls, marker=marker, markersize=5.5,
                     linewidth=1.7, capsize=2.5, label=s, **kw)
    _bands(axA)
    p_lo = float(table[(table["panel"] == "A")
                       & (table["metric"] == "plateau_harvested_min")
                       & (table["role"] == "annotation")]["value"].iloc[0])
    p_hi = float(table[(table["panel"] == "A")
                       & (table["metric"] == "plateau_harvested_max")
                       & (table["role"] == "annotation")]["value"].iloc[0])
    c_lo = float(table[(table["panel"] == "A")
                       & (table["metric"] == "stage2_collapse_min")
                       & (table["role"] == "annotation")]["value"].iloc[0])
    c_hi = float(table[(table["panel"] == "A")
                       & (table["metric"] == "stage2_collapse_max")
                       & (table["role"] == "annotation")]["value"].iloc[0])
    m0_xi09 = float(series_rows("A", "harvested_share", M0_SERIES)
                    .set_index("x").loc[0.9, "value"])
    axA.text(0.03, 0.045, (
        f"entry arms flat to ξ = 0.5: {p_lo * 100:.1f}–{p_hi * 100:.1f}%\n"
        f"Stage-2 collapse at ξ = 0.9: {c_lo * 100:.1f}–{c_hi * 100:.1f}% "
        f"(entry arms)\nM0 falls further: {m0_xi09 * 100:.1f}%"),
        transform=axA.transAxes, fontsize=HOUSE_STYLE["note_size"],
        color=PALETTE["gray"], va="bottom", ha="left")
    _cosmetics(axA, HOUSE_STYLE,
               "A · Macro efficiency",
               "visibility weight  $\\xi = 1 - c$",
               "harvested knowledge share\n(final step · run mean)")
    axA.set_xlim(0.05, 0.95)
    axA.set_ylim(0.2, 1.0)

    # ---- (b) like-for-like signal alignment ----
    for s in order:
        g = series_rows("B", "c_ideal_tm", s)
        color, ls, marker, mfc = SERIES_STYLE[s]
        kw = {} if mfc is None else {"markerfacecolor": mfc}
        axB.errorbar(g["x"], g["value"],
                     yerr=np.vstack([g["value"] - g["ci_lo"],
                                     g["ci_hi"] - g["value"]]),
                     color=color, linestyle=ls, marker=marker, markersize=5.5,
                     linewidth=1.7, capsize=2.5, label=s, **kw)
    _bands(axB)
    axB.axhline(0, color=PALETTE["gray"], lw=1.0, zorder=2)
    axB.text(0.012, 0, "0", transform=axB.get_yaxis_transform(),
             fontsize=HOUSE_STYLE["tick_size"], color=PALETTE["gray"],
             va="bottom")
    ends = {s: (float(series_rows("B", "c_ideal_tm", s).set_index("x").loc[0.1, "value"]),
                float(series_rows("B", "c_ideal_tm", s).set_index("x").loc[0.9, "value"]))
            for s in order}
    order_sorted = sorted(order, key=lambda s: -ends[s][0])
    hl = [f"{s.split(' (')[0]} {ends[s][0]:+.3f} → {ends[s][1]:+.3f}"
          for s in order_sorted]
    for k, line in enumerate(["like-for-like hierarchy (ξ = 0.1 → 0.9):",
                              hl[0], f"› {hl[1]}", f"› {hl[2]}", f"› {hl[3]}"]):
        axB.text(0.97, 0.885 - 0.05 * k, line,
                 transform=axB.transAxes, fontsize=HOUSE_STYLE["note_size"],
                 color=PALETTE["gray"], va="top", ha="right")
    _cosmetics(axB, HOUSE_STYLE,
               "B · Like-for-like signal alignment",
               "visibility weight  $\\xi = 1 - c$",
               "Corr(prestige, ideal merit)\n(time mean · run mean)")
    axB.set_xlim(0.05, 0.95)
    axB.set_ylim(-0.06, 0.75)

    # ---- (c) M2 co-harvest team size across xi ----
    for lam in (0.0, 1.0):
        s = M2_SERIES[lam]
        g = series_rows("C", "team_tm", s)
        color, ls, marker, mfc = SERIES_STYLE[s]
        axC.errorbar(g["x"], g["value"],
                     yerr=np.vstack([g["value"] - g["ci_lo"],
                                     g["ci_hi"] - g["value"]]),
                     color=color, linestyle=ls, marker=marker, markersize=5.5,
                     linewidth=1.7, capsize=2.5, label=s)
    _bands(axC)
    axC.axhline(KLANN_TEAM_SIZE, color=PALETTE["gray"], linestyle=":",
                linewidth=1.3, zorder=2,
                label=f"{KLANN_TEAM_SIZE:.2f} Klann team size —\n{BENCH_LABEL}")
    s_lo = float(table[(table["panel"] == "C") & (table["metric"] == "team_small_min")
                       & (table["role"] == "annotation")]["value"].iloc[0])
    s_hi = float(table[(table["panel"] == "C") & (table["metric"] == "team_small_max")
                       & (table["role"] == "annotation")]["value"].iloc[0])
    m_lo = float(table[(table["panel"] == "C") & (table["metric"] == "team_mob_min")
                       & (table["role"] == "annotation")]["value"].iloc[0])
    m_hi = float(table[(table["panel"] == "C") & (table["metric"] == "team_mob_max")
                       & (table["role"] == "annotation")]["value"].iloc[0])
    axC.text(0.03, 0.88, (
        f"ξ ≤ 0.5: teams {s_lo:.1f}–{s_hi:.1f} agents\n"
        f"ξ ∈ {{0.7, 0.9}}: {m_lo:.1f}–{m_hi:.1f} agents (mobs)"),
        transform=axC.transAxes, fontsize=HOUSE_STYLE["note_size"],
        color=PALETTE["gray"], va="top", ha="left")
    axC.legend(loc="center left", fontsize=HOUSE_STYLE["tick_size"] - 1.5,
               frameon=False)
    _cosmetics(axC, HOUSE_STYLE,
               "C · M2 co-harvest team size (mob collapse)",
               "visibility weight  $\\xi = 1 - c$",
               "co-harvest team size team_tm\n(time mean · run mean)")
    axC.set_xlim(0.05, 0.95)
    axC.set_ylim(0, 32)

    # ---- figure frame, shared legend, footnote ----
    fig.suptitle("F10b · M2 (pull = 1.0, logit + equal_split) on the full "
                 "visibility axis — strictly like-for-like time-mean alignment "
                 "(mgstd5)",
                 fontsize=HOUSE_STYLE["title_size"] + 1, y=0.99)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.82, bottom=0.33,
                        wspace=0.28)
    handles, labels = axA.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 0.925), ncol=4, frameon=False,
               fontsize=HOUSE_STYLE["tick_size"] - 1, handlelength=1.9,
               columnspacing=1.8)

    x05 = float(table[(table["panel"] == "B")
                      & (table["metric"] == "crosscheck_xi05_remeasure_vanishing05")]
                ["value"].iloc[0])
    p09 = float(table[(table["panel"] == "B")
                      & (table["metric"] == "crosscheck_xi05_pathb_vanishing09")]
                ["value"].iloc[0])
    note_lines = [
        f"Sources (read-only): data/remeasure_v1/w_v05_summary.csv (+ w_v05_runs.csv for CIs) for "
        f"M0 (g = 0) and M1 (g = 0.1); data/fig1_extension_v1/cells/m2_*.csv plus "
        f"data/path_b_v2/cells/cell_lam{{000,100}}_pull100_logit_equal_mgstd5.csv for M2 "
        f"(pull = 1.0, logit + equal_split) — {n_m2} M2 runs, 20 seeds per cell. All panels: mgstd5, "
        "error bars 95% CI = mean ± 1.96 SE over runs.",
        f"M0/M1 are pinned at vanishing factor 0.5 in remeasure_v1 (the path_b archives use 0.9); "
        f"cross-check of M1 time-mean alignment at ξ = 0.5: {x05:+.4f} (here, vanishing 0.5) vs "
        f"{p09:+.4f} (path_b_v1 baseline_off, vanishing 0.9). Alignment is strictly like-for-like "
        "time-mean c_ideal_tm in every arm — no time-mean/final-step mixing. In Stage 2, visibility "
        "herding swamps local dyadic pull and M2 teams explode into mobs. Klann team size 2.65: "
        f"{BENCH_LABEL} — model-internal, not calibrated.",
    ]
    note = "\n".join(textwrap.fill(t, width=182) for t in note_lines)
    fig.text(0.01, 0.012, note, fontsize=HOUSE_STYLE["note_size"],
             color=PALETTE["gray"], va="bottom")

    n_text = assert_text_fits(fig, fig.get_size_inches()[0] * fig.dpi,
                              fig.get_size_inches()[1] * fig.dpi)
    fig.savefig(png, dpi=HOUSE_STYLE["dpi"])
    fig.savefig(png.with_suffix(".pdf"))
    plt.close(fig)

    csv = write_figure_data(table, png)
    return {"png": str(png), "pdf": str(png.with_suffix(".pdf")),
            "data_csv": str(csv), "n_text_artists_checked": n_text}


if __name__ == "__main__":
    print(build())
