"""F10b_v2 -- M2 on the full visibility axis, ADR-0003 clock (Section 6, part 2).

Fig 16 of the paper (``F10b_v2_m2_visibility_like_for_like``).

All five tiers now live in ONE archive, so the like-for-like comparison no
longer stitches two runs tables together::

    data/fig1_xi_adr0003/cells/fig1_tier-*_xi-*.csv
        55 cells = 5 tiers x 11 xi points x 20 seeds = 1,100 runs, mgstd5,
        visibility reward on, tau_max = 80, gamma = 0.933, fixed_span except
        M0 (no retirement) and the M1 staggered tier; H = 400 for the two M1
        tiers, H = 200 for M0 and M2.
    data/bug0006_remeasure_v2/{w_v05_runs,m2_xi_runs}.csv   (pre-ADR pair)

The dense xi grid is {0.1, 0.3, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
0.95} -- dense exactly where the two-stage collapse happens.  Tier labels are
derived from the archive's own pins (entry rate, exit rule, horizon), so a
pre-ADR rebuild does not inherit the ADR-0003 g pin in its legend.

Panels:
  (a) macro efficiency        : harvested knowledge share (``harvested_fin``)
  (b) like-for-like hierarchy : time-mean ``c_ideal_tm`` for M0, both M1
      exit rules and M2 lambda = 0 / lambda = 1 -- one statistic in every arm
  (c) co-harvest team size    : ``team_tm`` for the tiers whose population is
      the same as the modal one; the staggered M1 tier (N = 121 by
      construction) is EXCLUDED and its population is printed in the panel
      note, because team size is a level, not a share, and a 121-agent pool is
      not comparable with a 20-agent one.

The F6 stage bands (Stage 1 / kink / Stage 2) are the shared fig_F6_v2 shaded
regions -- imported read-only, labelled once in panel A, and never drawn as
boundary rules across the series.

Every plotted value, error bar and annotation is computed from the raw runs
and written into ``*_data.csv``; ``visual_qa.assert_figure_clean`` and the
local display-space text-vs-data audit run on the rendered canvas before
``savefig``.  lambda = 1 is always labelled ``closeness x local merit``;
Klann's 2.65 is labelled ``Empirical benchmark (unfitted)``.

Run:  MPLBACKEND=Agg /opt/hermes/.venv/bin/python \
          scripts/report_notebook/figF_v2_m2_visibility_postbug.py
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from diagram_templates import (  # noqa: E402
    HOUSE_STYLE,
    PALETTE,
    _cosmetics,
    write_figure_data,
)
import visual_qa  # noqa: E402
#: shared band / placement / ylim / audit grammar (one implementation; F6 owns
#: it and is imported READ-ONLY)
from fig_F6_v2 import (  # noqa: E402
    _auto_ylim,
    _bands,
    _band_labels,
    _mean_ci,
    _place_text,
    _require,
    audit_text_over_data,
    place_figure_legend,
)

FNAME = "F10b_v2_m2_visibility_like_for_like"
DEFAULT_DATA_ROOT = REPO / "data" / "fig1_xi_adr0003"
PRE_DATA_ROOT = REPO / "data" / "bug0006_remeasure_v2"
LANDSCAPE = "mgstd5"
Z95 = 1.96
KLANN_TEAM_SIZE = 2.65
BENCH_LABEL = "Empirical benchmark (unfitted)"

M_HARVEST = "harvested knowledge share"
M_CORR = "Corr(prestige, ideal merit) (time mean)"
M_TEAM = "co-harvest team size (time mean)"
PANELS = {"a": M_HARVEST, "b": M_CORR, "c": M_TEAM}
VALUE_COL = {"a": "harvested_fin", "b": "c_ideal_tm", "c": "team_tm"}

#: tier CODES (the archive's own vocabulary) -- display labels are derived from
#: the archive pins, never hard-typed, so a rebuild keeps its own g pin
C0, C1F, C1S, C20, C21 = "M0", "M1_fixed", "M1_staggered", "M2_l0", "M2_l1"
ORDER = (C0, C1F, C1S, C20, C21)
#: tier grammar (F6 hues/markers): M0 purple, M1 fixed grey dashed hollow,
#: M1 staggered orange dash-dot, M2 lambda = 0 blue dashed squares,
#: M2 lambda = 1 green triangles
STYLE = {
    C0: dict(color=PALETTE["purple"], ls="-", marker="o", mfc=PALETTE["purple"]),
    C1F: dict(color=PALETTE["gray"], ls="--", marker="o", mfc="white"),
    C1S: dict(color=PALETTE["orange"], ls="-.", marker="D", mfc=PALETTE["orange"]),
    C20: dict(color=PALETTE["blue"], ls="--", marker="s", mfc=PALETTE["blue"]),
    C21: dict(color=PALETTE["green"], ls="-", marker="^", mfc=PALETTE["green"]),
}
SHORT = {C0: "M0 (g=0)", C1F: "M1 fixed", C1S: "M1 staggered",
         C20: "M2 λ = 0", C21: "M2 λ = 1"}
ENTRY_ARMS = (C1F, C1S, C20, C21)


def tier_label(code: str, pins: dict) -> str:
    """Display label from the archive pins (g, exit rule)."""
    tp = pins["tier_pins"].get(code, {})
    g = tp.get("g", float("nan"))
    if code == C0:
        return f"M0 (g = 0)"
    if code == C1F:
        return f"M1 (g = {g:g}), fixed span"
    if code == C1S:
        return f"M1 (g = {g:g}), staggered"
    if code == C20:
        return "M2 λ = 0 (prestige matching)"
    return "M2 λ = 1 (closeness × local merit)"


def data_root_path(data_root=None) -> Path:
    root = DEFAULT_DATA_ROOT if data_root is None else Path(data_root)
    if not root.is_absolute():
        root = REPO / root
    return root


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------
def load_runs(data_root=None) -> tuple[pd.DataFrame, dict]:
    """Canonical (tier code, xi) run table + pins, from either archive layout."""
    root = data_root_path(data_root)
    cells = sorted((root / "cells").glob("fig1_*.csv"))
    if cells:
        frames = []
        for f in cells:
            d = pd.read_csv(f)
            _require(len(d) == 20, f"{f.name}: 20 seed rows expected")
            frames.append(d)
        raw = pd.concat(frames, ignore_index=True)
        _require(set(raw["tier"].unique()) <= set(ORDER),
                 f"unexpected tier codes {sorted(raw['tier'].unique())}")
        runs = pd.DataFrame({
            "tier": raw["tier"].astype(str), "xi": raw["xi"].astype(float),
            "harvested_fin": raw["harvested_fin"].astype(float),
            "c_ideal_tm": raw["c_ideal_tm"].astype(float),
            "team_tm": raw["team_tm"].astype(float),
            "n_final": raw["n_final"].astype(float),
            "seed": raw["seed"].astype(int),
        })
        pins = {
            "layout": "single archive (5 tiers)",
            "cells": len(cells), "runs": int(len(runs)), "per_cell": 20,
            "xs": sorted(runs["xi"].unique().tolist()),
            "tier_pins": raw.groupby("tier").agg(
                g=("g", "first"), exit_mode=("exit_mode", "first"),
                horizon=("horizon", "first"), n_final=("n_final", "mean"),
                dyadic_lambda=("dyadic_lambda", "first"),
                pull=("dyadic_move_pull", "first")).to_dict("index"),
        }
        _require(runs.groupby(["tier", "xi"]).size().eq(20).all(),
                 "expected 20 seed rows per (tier, xi) cell")
        return runs, pins

    # ---- pre-ADR pair: w_v05_runs.csv (M0/M1) + m2_xi_runs.csv (M2) ------
    f_w, f_m2 = root / "w_v05_runs.csv", root / "m2_xi_runs.csv"
    for f in (f_w, f_m2):
        _require(f.exists(), f"missing archive {f} (and no fig1_*.csv cells)")
    w = pd.read_csv(f_w)
    wv = w[(w["landscape"] == LANDSCAPE) & (w["use_visibility_reward"] == True)].copy()  # noqa: E712
    wv["xi"] = np.round(1.0 - wv["curiosity"], 2)
    wv["tier"] = np.where(np.round(wv["agent_generation_time"], 2) == 0.0, C0, C1F)
    m = pd.read_csv(f_m2)
    m["xi"] = np.round(m["xi"], 2)
    m["tier"] = np.where(np.round(m["dyadic_lambda"], 1) == 0.0, C20, C21)
    keep = ["tier", "xi", "harvested_fin", "c_ideal_tm", "team_tm", "seed"]
    wv_keep = wv.reindex(columns=keep).copy()
    wv_keep["n_final"] = np.nan          # the pre-ADR tables carry no team stats
    wv_keep["team_tm"] = np.nan
    m_keep = m.reindex(columns=keep).copy()
    m_keep["n_final"] = np.nan
    runs = pd.concat([wv_keep, m_keep], ignore_index=True)
    _require(runs.groupby(["tier", "xi"]).size().eq(20).all(),
             "pre-ADR pair: 20 seed rows per (tier, xi) cell")
    def _first(s_, *names):
        for n in names:
            if n in s_.columns:
                return float(s_[n].iloc[0])
        return float("nan")

    tp = {}
    for frame in (wv, m):
        for tier, s_ in frame.groupby("tier"):
            tp[str(tier)] = dict(
                g=_first(s_, "agent_generation_time", "g"),
                exit_mode="n/a", horizon=np.nan, n_final=np.nan,
                dyadic_lambda=_first(s_, "dyadic_lambda"),
                pull=_first(s_, "dyadic_move_pull"))
    pins = {"layout": "pre-ADR pair (w_v05 + m2_xi)", "cells": len(runs) // 20,
            "runs": int(len(runs)), "per_cell": 20,
            "xs": sorted(runs["xi"].unique().tolist()), "tier_pins": tp}
    return runs, pins


def _team_tiers(runs: pd.DataFrame, pins: dict) -> tuple[tuple, dict]:
    """Tiers whose population matches the modal one (team size is a level)."""
    with_team = tuple(c for c in ORDER
                      if runs.loc[runs["tier"] == c, "team_tm"].notna().any())
    pop = runs.dropna(subset=["team_tm"]).groupby("tier")["n_final"].mean()
    pop = pop.dropna()
    if pop.empty or not len(pop):
        return with_team, {"modal": np.nan, "excluded": {}}
    modal = float(pop.round().mode().iloc[0])
    keep = tuple(c for c in with_team
                 if c in pop.index and abs(pop[c] - modal) <= 0.75 * modal)
    excluded = {c: float(pop[c]) for c in with_team if c not in keep}
    return keep, {"modal": modal, "excluded": excluded}


def series_rows(runs: pd.DataFrame, pins: dict) -> pd.DataFrame:
    rows = []
    labels = {c: tier_label(c, pins) for c in ORDER}
    for panel, metric in PANELS.items():
        col = VALUE_COL[panel]
        tiers = ORDER if panel != "c" else _team_tiers(runs, pins)[0]
        for code in tiers:
            for xi in pins["xs"]:
                sel = runs[(runs["tier"] == code) & (runs["xi"] == xi)][col]
                if sel.notna().sum() == 0:
                    continue
                _require(len(sel) == 20, f"{code} xi={xi}: {len(sel)} runs")
                mean, lo, hi, n = _mean_ci(sel)
                rows.append(dict(figure=FNAME, panel=panel, metric=metric,
                                 statistic="mean", tier=labels[code], xi=xi,
                                 landscape=LANDSCAPE, value=mean, ci_lo=lo,
                                 ci_hi=hi, n=n,
                                 source_file=f"{pins['layout']}: {col}"))
    rows.append(dict(figure=FNAME, panel="c", metric=M_TEAM,
                     statistic="reference", tier=BENCH_LABEL, xi=np.nan,
                     landscape=LANDSCAPE, value=KLANN_TEAM_SIZE, ci_lo=np.nan,
                     ci_hi=np.nan, n=0,
                     source_file="Klann et al. team-size benchmark (unfitted)"))
    return pd.DataFrame(rows)


def annotation_rows(series: pd.DataFrame, pins: dict, data_root=None) -> pd.DataFrame:
    labels = {c: tier_label(c, pins) for c in ORDER}
    V = {}
    for _, r in series[series["statistic"] == "mean"].iterrows():
        V[(r["panel"], r["tier"], float(r["xi"]))] = float(r["value"])
    xs = pins["xs"]
    rows = []

    def add(panel, metric, value, note):
        rows.append(dict(figure=FNAME, panel=panel, metric=metric,
                         statistic="annotation", tier="", xi=np.nan,
                         landscape=LANDSCAPE, value=float(value), ci_lo=np.nan,
                         ci_hi=np.nan, n=100, source_file=note))

    # -- (a) plateau over the entry arms, then the Stage-2 collapse ---------
    plateau = [V[("a", labels[c], xi)] for c in ENTRY_ARMS for xi in xs
               if xi <= 0.5 and ("a", labels[c], xi) in V]
    tail = [V[("a", labels[c], xi)] for c in ENTRY_ARMS for xi in xs
            if xi >= 0.9 and ("a", labels[c], xi) in V]
    add("a", "entry_plateau_harvested_min", min(plateau),
        "min harvested share over the entry arms at xi <= 0.5")
    add("a", "entry_plateau_harvested_max", max(plateau),
        "max harvested share over the entry arms at xi <= 0.5")
    add("a", "stage2_collapse_min", min(tail),
        "min harvested share over the entry arms at xi >= 0.9")
    add("a", "stage2_collapse_max", max(tail),
        "max harvested share over the entry arms at xi >= 0.9")
    add("a", "m0_min_harvested", min(V[("a", labels[C0], xi)] for xi in xs),
        "lowest harvested share of the M0 tier over the grid")

    # -- (b) like-for-like endpoints per tier ------------------------------
    corr_codes = tuple(c for c in ORDER if ("b", labels[c], xs[0]) in V)
    for code in corr_codes:
        add("b", f"c_tm_first::{labels[code]}", V[("b", labels[code], xs[0])],
            f"time-mean c_ideal_tm, {labels[code]}, xi = {xs[0]:g}")
        add("b", f"c_tm_last::{labels[code]}", V[("b", labels[code], xs[-1])],
            f"time-mean c_ideal_tm, {labels[code]}, xi = {xs[-1]:g}")

    # -- (c) team size: young-team range and the two mob stages ------------
    small = [v for (p, t, x), v in V.items() if p == "c" and x <= 0.5]
    add("c", "team_small_min", min(small),
        "min team_tm over the fixed-population tiers at xi <= 0.5")
    add("c", "team_small_max", max(small),
        "max team_tm over the fixed-population tiers at xi <= 0.5")
    for lab, thr in (("xi07", 0.7), ("xi09", 0.9)):
        vals = [v for (p, t, x), v in V.items() if p == "c" and x == thr]
        if not vals:
            continue
        add("c", f"team_{lab}_min", min(vals), f"min team_tm at xi = {thr:g}")
        add("c", f"team_{lab}_max", max(vals), f"max team_tm at xi = {thr:g}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figure ---
def _plot_series(ax, s: pd.DataFrame, metric: str, label: str, n_x: int,
                 code: str) -> None:
    d = s[(s["metric"] == metric) & (s["tier"] == label)].sort_values("xi")
    _require(len(d) == n_x, f"{label}: {len(d)} points, expected {n_x}")
    st = STYLE[code]
    ax.errorbar(d["xi"].to_numpy(float), d["value"].to_numpy(float),
                yerr=np.vstack([d["value"] - d["ci_lo"],
                                d["ci_hi"] - d["value"]]),
                color=st["color"], linestyle=st["ls"], marker=st["marker"],
                markersize=5.0, linewidth=1.7, capsize=2.2, elinewidth=0.85,
                label=label, markerfacecolor=st["mfc"],
                markeredgecolor=st["color"], markeredgewidth=1.0, zorder=4)


def build_figure(data_root=None):
    runs, pins = load_runs(data_root)
    labels = {c: tier_label(c, pins) for c in ORDER}
    avail = tuple(c for c in ORDER if (runs["tier"] == c).any())
    series = series_rows(runs, pins)
    anns = annotation_rows(series, pins, data_root)
    sidecar = pd.concat([series, anns], ignore_index=True)
    A = {r["metric"]: float(r["value"]) for _, r in anns.iterrows()}
    xs = pins["xs"]
    n_x = len(xs)
    team_codes, pop_info = _team_tiers(runs, pins)

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(13.8, 5.6))
    fig.subplots_adjust(left=0.052, right=0.99, top=0.845, bottom=0.470,
                        wspace=0.24)
    fig.text(0.5, 0.978,
             "F10b_v2 · M2 (pull = 1.0, logit + equal_split) on the full "
             "visibility axis — five tiers in one ADR-0003 archive, mgstd5",
             ha="center", va="top", fontsize=HOUSE_STYLE["title_size"] + 1)

    # ---- (a) macro efficiency -------------------------------------------
    _bands(ax_a)
    for code in avail:
        _plot_series(ax_a, series, M_HARVEST, labels[code], n_x, code)
    _auto_ylim(ax_a, pad_hi=0.22)
    _place_text(ax_a, (
        f"entry arms flat to ξ ≤ 0.5: "
        f"{A['entry_plateau_harvested_min'] * 100:.1f}–"
        f"{A['entry_plateau_harvested_max'] * 100:.1f}%\n"
        f"ξ ≥ 0.9: {A['stage2_collapse_min'] * 100:.1f}–"
        f"{A['stage2_collapse_max'] * 100:.1f}% (entry arms)\n"
        f"M0 is lower still: {A['m0_min_harvested'] * 100:.1f}% at ξ = {xs[-1]:g}"),
        ((0.03, 0.05, "left", "bottom"), (0.03, 0.18, "left", "bottom"),
         (0.97, 0.05, "right", "bottom")))
    _band_labels(ax_a, prefer=("top", "bottom"))
    _cosmetics(ax_a, HOUSE_STYLE, "A · Macro efficiency",
               "visibility weight  $\\xi = 1 - c$",
               "harvested knowledge share\n(final step · run mean)")
    ax_a.set_xlim(0.04, xs[-1] + 0.06)
    ax_a.set_xticks(xs, [f"{x:g}" for x in xs],
                    fontsize=HOUSE_STYLE["tick_size"] - 2)

    # ---- (b) like-for-like hierarchy ------------------------------------
    _bands(ax_b)
    for code in avail:
        _plot_series(ax_b, series, M_CORR, labels[code], n_x, code)
    _auto_ylim(ax_b, include=(0.0,), pad_hi=0.42, pad_lo=0.10)
    ax_b.axhline(0.0, color=PALETTE["gray"], linewidth=1.0, zorder=2)
    lo_b, hi_b = ax_b.get_ylim()
    ax_b.set_yticks([t for t in ax_b.get_yticks() if lo_b < t < hi_b])
    corr = [c for c in avail if f"c_tm_first::{labels[c]}" in A]
    order = sorted(corr, key=lambda c: -A[f"c_tm_first::{labels[c]}"])
    _place_text(ax_b, (
        f"order at ξ = {xs[0]:g} (high → low)\n"
        + " > ".join(SHORT[c] for c in order[:3]) + " >\n"
        + " > ".join(SHORT[c] for c in order[3:]) + "\n"
        + (f"(M2 λ = 0 is negative at low ξ: "
           f"{A[f'c_tm_first::{labels[C20]}']:+.3f})"
           if f"c_tm_first::{labels[C20]}" in A else
           f"(ξ = {xs[0]:g} ordering)")),
        ((0.97, 0.99, "right", "top"), (0.03, 0.99, "left", "top")))
    _cosmetics(ax_b, HOUSE_STYLE, "B · Like-for-like signal alignment",
               "visibility weight  $\\xi = 1 - c$",
               "Corr(prestige, ideal merit)\n(time mean · run mean)")
    ax_b.set_xlim(0.04, xs[-1] + 0.06)
    ax_b.set_xticks(xs, [f"{x:g}" for x in xs],
                    fontsize=HOUSE_STYLE["tick_size"] - 2)

    # ---- (c) co-harvest team size ---------------------------------------
    _bands(ax_c)
    for code in team_codes:
        _plot_series(ax_c, series, M_TEAM, labels[code], n_x, code)
    ax_c.axhline(KLANN_TEAM_SIZE, color=PALETTE["gray"], linestyle=":",
                 linewidth=1.3, zorder=2)
    _auto_ylim(ax_c, include=(KLANN_TEAM_SIZE, 0.0), pad_hi=0.36, floor=0.0)
    # the Klann rule is identified in the shared legend, not in the panel:
    # the team curves cross the rule, so no inline label can clear the data
    klann_handle = plt.Line2D([], [], color=PALETTE["gray"], linestyle=":",
                              linewidth=1.3,
                              label=f"{KLANN_TEAM_SIZE:.2f} Klann team size "
                                    f"({BENCH_LABEL})")
    exc = pop_info.get("excluded", {})
    exc_txt = ("\n".join(f"{SHORT[c]} (N = {n:.0f}) not drawn"
                         for c, n in exc.items())
               if exc else "same population in every tier drawn")
    _place_text(ax_c, (
        f"teams at ξ ≤ 0.5: {A['team_small_min']:.1f}–{A['team_small_max']:.1f}\n"
        f"ξ = 0.7: {A['team_xi07_min']:.1f}–{A['team_xi07_max']:.1f}\n"
        f"ξ = 0.9: {A['team_xi09_min']:.1f}–{A['team_xi09_max']:.1f} (mobs)\n"
        + exc_txt),
        tuple((0.97, y, "right", "top") for y in (0.99, 0.78))
        + tuple((0.03, y, "left", "top") for y in (0.99, 0.72)))
    _cosmetics(ax_c, HOUSE_STYLE, "C · Co-harvest team size",
               "visibility weight  $\\xi = 1 - c$",
               "co-harvest team size team_tm\n(time mean · run mean)")
    ax_c.set_xlim(0.04, xs[-1] + 0.06)
    ax_c.set_xticks(xs, [f"{x:g}" for x in xs],
                    fontsize=HOUSE_STYLE["tick_size"] - 2)

    # ---- frame, shared legend, footnote ----------------------------------
    root = data_root_path(data_root)
    rel = root.relative_to(REPO) if root.is_relative_to(REPO) else root
    tp = pins["tier_pins"]
    pin_bits = []
    for code in avail:
        p_ = tp.get(code, {})
        if p_ and not pd.isna(p_.get("g", np.nan)):
            pin_bits.append(f"{SHORT[code]}: g = {p_['g']:g}, "
                            f"{p_['exit_mode']}, H = {p_['horizon']}")
    pin_txt = "; ".join(pin_bits) if pin_bits else "pins read from the archive"
    note_lines = [
        f"Sources (read-only): {rel}/cells/fig1_tier-*_xi-*.csv — "
        f"{pins['cells']} cells × {pins['per_cell']} seeds = {pins['runs']} runs, "
        f"mgstd5, visibility reward on, xi in {{{', '.join(f'{x:g}' for x in xs)}}} "
        f"(dense where the collapse happens). Tier pins: {pin_txt}. ADR-0003 "
        "clock: 1 step = 0.5 yr, entry age 25, tau_max = 80 (40-yr careers), "
        "gamma = 0.5**(1/10) = 0.933 (5-yr retention half-life). Error bars 95% "
        "CI = mean ± 1.96 SE over runs.",
        "Panels: (a) harvested knowledge share, every tier; (b) time-mean "
        "c_ideal_tm for all five tiers — one statistic in every arm, so the "
        "comparison is like-for-like; (c) co-harvest team size only for tiers "
        "whose population matches the modal one — team size is a level, not a "
        "share, and the staggered M1 tier carries N = "
        f"{exc.get(C1S, float('nan')):.0f} agents (scheduled entry plus "
        "one-for-one retirement) against N = "
        f"{pop_info.get('modal', float('nan')):.0f} elsewhere, so it is excluded "
        "and named in the panel instead of drawn as if comparable. The pre-ADR "
        "figure could show M2 only in (c) because the M0/M1 tables carried no "
        "team statistics; in this archive they do.",
        "Time-mean alignment endpoints (ξ = " + f"{xs[0]:g} → {xs[-1]:g}" + ", "
        "from the sidecar rows): " + "; ".join(
            f"{SHORT[c]} {A[f'c_tm_first::{labels[c]}']:+.3f} → "
            f"{A[f'c_tm_last::{labels[c]}']:+.3f}" for c in corr) + ".",
        "Stage 1 / kink / Stage 2 are the shared F6 shaded bands, labelled once "
        "in panel A (no boundary rule is drawn across the series, so nothing "
        f"crosses the annotations). Klann team size {KLANN_TEAM_SIZE:.2f}: "
        "Empirical benchmark (unfitted). Every plotted value, CI and annotation "
        f"is computed from the raw CSVs and stored in {FNAME}_data.csv. "
        "MODEL-INTERNAL, NOT CALIBRATED.",
    ]
    note = fig.text(0.008, 0.006, "\n".join(textwrap.fill(t, width=196)
                                            for t in note_lines),
                    fontsize=HOUSE_STYLE["note_size"] - 1.5,
                    color=PALETTE["gray"], va="bottom")
    handles, labels_ = ax_a.get_legend_handles_labels()
    handles, labels_ = list(handles) + [klann_handle], list(labels_) + [
        klann_handle.get_label()]
    place_figure_legend(fig, handles, labels_, len(handles), note)

    audit_text_over_data(fig, (ax_a, ax_b, ax_c))
    visual_qa.assert_figure_clean(fig, [
        {"ax": ax_a, "metric": M_HARVEST, "expect_lines": len(avail),
         "points_per_line": n_x, "sidecar": sidecar,
         "filters": {"statistic": "mean"}},
        {"ax": ax_b, "metric": M_CORR, "expect_lines": len(avail),
         "points_per_line": n_x, "sidecar": sidecar,
         "filters": {"statistic": "mean"}},
        {"ax": ax_c, "metric": M_TEAM, "expect_lines": len(team_codes),
         "points_per_line": n_x, "sidecar": sidecar,
         "filters": {"statistic": "mean"}},
    ])
    return fig, sidecar, runs, pins


def build(outdir="output/report_notebook_v1", data_root=None) -> dict:
    fig, sidecar, _runs, _pins = build_figure(data_root)
    out = Path(outdir)
    if not out.is_absolute():
        out = REPO / out
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{FNAME}.png"
    pdf = out / f"{FNAME}.pdf"
    fig.savefig(png, dpi=HOUSE_STYLE["dpi"])
    fig.savefig(pdf)
    plt.close(fig)
    csv_path = write_figure_data(sidecar, png)
    print(f"F10b_v2: {png} | {pdf} | {csv_path}")
    print(f"sidecar rows: {len(sidecar)}")
    return {"png": str(png), "pdf": str(pdf), "data_csv": str(csv_path),
            "sidecar_rows": len(sidecar)}


if __name__ == "__main__":
    print(build())
