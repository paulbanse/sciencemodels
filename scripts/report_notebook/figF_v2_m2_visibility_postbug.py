"""F10b_v2 -- M2 on the full visibility axis, post-BUG-0006 (Section 6, part 2).

Same three-panel architecture as the pre-fix ``F10b`` (kept intact for
contrast), rebuilt from the post-``BUG-0006`` archives so the M2 numbers are
no longer the ones the bug corrupted:

  (a) macro efficiency        : harvested knowledge share (``harvested_fin``)
  (b) like-for-like hierarchy : time-mean ``c_ideal_tm`` for M0, M1, M2
      lambda = 0 and M2 lambda = 1 -- one statistic in every arm
  (c) M2 co-harvest team size : ``team_tm`` with the unfitted Klann
      team-size benchmark

Inputs (read-only, ``data_root`` defaults to ``data/bug0006_remeasure_v2``):
  * ``w_v05_runs.csv``  -- M0 (g = 0) and M1 (g = 0.1), mgstd5, visibility
    reward on, vanishing 0.5 (the pin is read from the archive);
  * ``m2_xi_runs.csv``  -- M2 (pull = 1.0, logit + equal_split), xi in
    0.1 ... 0.9, vanishing 0.9;
  * ``tables/m2_xi_bug0006_delta.csv`` / ``tables/w_v05_bug0006_delta.csv``
    for the pre-vs-post contrast quoted in the footnote (read, not typed).

Two rendering defects of the pre-fix figure are fixed here:
  1. the stray ``"0"`` text label sitting just above the zero line in panel B
     is gone -- zero is now a real y-tick plus the reference rule;
  2. the dashed vertical band-boundary lines that crossed every series and
     ran straight through panel B's annotation block are gone -- the Stage
     1 / kink / Stage 2 bands are shaded REGIONS only.

Every plotted value, error bar and annotation is computed from the raw runs
and written into ``*_data.csv``; ``visual_qa.assert_figure_clean`` runs on
the rendered canvas before ``savefig``.  lambda = 1 is always labelled
``closeness x local merit``; Klann's 2.65 is labelled
``Empirical benchmark (unfitted)``.

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
# shared band / placement / ylim grammar (one implementation, both
# post-BUG-0006 refreshes)
from fig_F6_v2 import (  # noqa: E402
    _auto_ylim,
    _band_labels,
    _bands,
    _mean_ci,
    audit_text_over_data,
    place_figure_legend,
    _place_text,
    _plot_tier,
    _require,
)

FNAME = "F10b_v2_m2_visibility_like_for_like"
DEFAULT_DATA_ROOT = REPO / "data" / "bug0006_remeasure_v2"
LANDSCAPE = "mgstd5"
XIS = (0.1, 0.3, 0.5, 0.7, 0.9)
KLANN_TEAM_SIZE = 2.65
BENCH_LABEL = "Empirical benchmark (unfitted)"

M_HARVEST = "harvested knowledge share"
M_CORR = "Corr(prestige, ideal merit) (time mean)"
M_TEAM = "co-harvest team size (time mean)"
PANELS = {"a": M_HARVEST, "b": M_CORR, "c": M_TEAM}
VALUE_COL = {"a": "harvested_fin", "b": "c_ideal_tm", "c": "team_tm"}

M0_TIER = "M0 (g = 0)"
M1_TIER = "M1 (g = 0.1)"
M2_TIERS = {0.0: "M2 λ = 0 (prestige matching)",
            1.0: "M2 λ = 1 (closeness × local merit)"}
TIERS = (M0_TIER, M1_TIER, M2_TIERS[0.0], M2_TIERS[1.0])
ENTRY_ARMS = (M1_TIER, M2_TIERS[0.0], M2_TIERS[1.0])


def data_root_path(data_root=None) -> Path:
    return Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT


def load_runs(data_root=None) -> tuple[pd.DataFrame, dict]:
    """M0/M1 from w_v05_runs.csv + M2 from m2_xi_runs.csv (400 runs)."""
    root = data_root_path(data_root)
    f_w = root / "w_v05_runs.csv"
    f_m2 = root / "m2_xi_runs.csv"
    for f in (f_w, f_m2):
        _require(f.exists(), f"missing archive {f}")

    w = pd.read_csv(f_w)
    wv = w[(w["landscape"] == LANDSCAPE)
           & (w["use_visibility_reward"] == True)].copy()  # noqa: E712
    _require(len(wv) == 200,
             f"w_v05 mgstd5+visibility: expected 200 runs, got {len(wv)}")
    _require(set(np.round(wv["vanishing"], 3)) == {0.5},
             "w_v05: vanishing must be pinned at 0.5")
    wv["xi"] = np.round(1.0 - wv["curiosity"], 1)
    _require(sorted(wv["xi"].unique()) == list(XIS), "w_v05: xi grid")
    _require((wv.groupby(["agent_generation_time", "xi"]).size() == 20).all(),
             "w_v05: expected 20 seeds per (g, xi) cell")
    wv["tier"] = np.where(np.round(wv["agent_generation_time"], 2) == 0.0,
                          M0_TIER, M1_TIER)

    m = pd.read_csv(f_m2)
    _require(len(m) == 200, f"m2_xi: expected 200 runs, got {len(m)}")
    _require(set(np.round(m["vanishing"], 3)) == {0.9},
             "m2_xi: vanishing must be pinned at 0.9")
    _require((m["landscape"] == LANDSCAPE).all(), "m2_xi: landscape")
    _require((m["use_visibility_reward"] == True).all(),  # noqa: E712
             "m2_xi: visibility reward must be on")
    _require(set(np.round(m["dyadic_move_pull"], 3)) == {1.0},
             "m2_xi: pull must be 1.0")
    _require((m["formulation"] == "logit").all(), "m2_xi: formulation")
    _require((m["credit_sharing_axis"] == "equal_split").all(),
             "m2_xi: credit_sharing_axis")
    m["xi"] = np.round(m["xi"], 1)
    _require(np.allclose(m["xi"], np.round(1.0 - m["curiosity"], 1)),
             "m2_xi: xi must equal 1 - curiosity")
    _require(sorted(m["xi"].unique()) == list(XIS), "m2_xi: xi grid")
    _require((m.groupby(["dyadic_lambda", "xi"]).size() == 20).all(),
             "m2_xi: expected 20 seeds per (lambda, xi) cell")
    _require(set(np.round(m["dyadic_lambda"], 1)) == {0.0, 1.0},
             "m2_xi: lambda grid")
    _require(len(set(np.round(m["agent_generation_time"], 2))) == 1,
             "m2_xi: expected one g pin")
    m["tier"] = np.where(np.round(m["dyadic_lambda"], 1) == 0.0,
                         M2_TIERS[0.0], M2_TIERS[1.0])

    keep = ["tier", "xi", "harvested_fin", "c_ideal_tm", "team_tm", "seed"]
    wv_keep = wv[["tier", "xi", "harvested_fin", "c_ideal_tm", "seed"]].copy()
    wv_keep["team_tm"] = np.nan   # w_v05 does not carry the W-layer team stats
    runs = pd.concat([wv_keep.assign(archive=f_w.name),
                      m[keep].assign(archive=f_m2.name)], ignore_index=True)
    _require(len(runs) == 400,
             f"M0/M1/M2: expected 400 runs, got {len(runs)}")
    pins = {M0_TIER: dict(vanishing=0.5, archive=f_w.name, runs=100),
            M1_TIER: dict(vanishing=0.5, archive=f_w.name, runs=100),
            M2_TIERS[0.0]: dict(vanishing=0.9, archive=f_m2.name, runs=100),
            M2_TIERS[1.0]: dict(vanishing=0.9, archive=f_m2.name, runs=100),
            "m2_g_pin": float(np.round(m["agent_generation_time"].iloc[0], 3)),
            "n_w_v05": len(wv), "n_m2": len(m)}
    return runs, pins


def series_rows(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean / 95% CI per (panel, tier, xi) -- one row per plotted statistic."""
    rows = []
    for panel, metric in PANELS.items():
        col = VALUE_COL[panel]
        tiers = TIERS if panel != "c" else tuple(M2_TIERS.values())
        for tier in tiers:
            for xi in XIS:
                sel = runs[(runs["tier"] == tier) & (runs["xi"] == xi)][col]
                _require(len(sel) == 20,
                         f"{tier} xi={xi}: expected 20 runs, got {len(sel)}")
                mean, lo, hi, n = _mean_ci(sel)
                rows.append(dict(
                    figure=FNAME, panel=panel, metric=metric,
                    statistic="mean", tier=tier, xi=xi, landscape=LANDSCAPE,
                    value=mean, ci_lo=lo, ci_hi=hi, n=n,
                    source_file=sorted(
                        runs[runs["tier"] == tier]["archive"].unique())[0],
                ))
    # the unfitted Klann team-size benchmark (panel c reference line)
    rows.append(dict(figure=FNAME, panel="c", metric=M_TEAM,
                     statistic="reference", tier=BENCH_LABEL,
                     xi=np.nan, landscape=LANDSCAPE,
                     value=KLANN_TEAM_SIZE, ci_lo=np.nan, ci_hi=np.nan, n=0,
                     source_file="Klann et al. team-size benchmark (unfitted)"))
    return pd.DataFrame(rows)


def _value_table(series: pd.DataFrame) -> dict:
    out = {}
    for _, r in series[series["statistic"] == "mean"].iterrows():
        out[(r["panel"], r["tier"], float(r["xi"]))] = float(r["value"])
    return out


def annotation_rows(series: pd.DataFrame, pins: dict,
                    data_root=None) -> pd.DataFrame:
    V = _value_table(series)
    rows = []

    def add(panel: str, metric: str, value: float, note: str) -> None:
        rows.append(dict(figure=FNAME, panel=panel, metric=metric,
                         statistic="annotation", tier="", xi=np.nan,
                         landscape=LANDSCAPE, value=float(value),
                         ci_lo=np.nan, ci_hi=np.nan, n=100, source_file=note))

    # -- (a) macro: plateau, Stage-2 collapse ------------------------------
    plateau = [V[("a", t, xi)] for t in ENTRY_ARMS for xi in (0.1, 0.3, 0.5)]
    collapse = [V[("a", t, 0.9)] for t in ENTRY_ARMS]
    add("a", "entry_plateau_harvested_min", min(plateau),
        "min harvested share over entry arms (M1 + M2) at xi <= 0.5")
    add("a", "entry_plateau_harvested_max", max(plateau),
        "max harvested share over entry arms at xi <= 0.5")
    add("a", "stage2_collapse_min", min(collapse),
        "min harvested share over entry arms at xi = 0.9")
    add("a", "stage2_collapse_max", max(collapse),
        "max harvested share over entry arms at xi = 0.9")
    add("a", "m0_xi09_harvested", V[("a", M0_TIER, 0.9)],
        "M0 harvested share at xi = 0.9")

    # -- (b) like-for-like hierarchy endpoints ------------------------------
    for tier in TIERS:
        add("b", f"c_tm_xi01::{tier}", V[("b", tier, 0.1)],
            f"time-mean c_ideal_tm, {tier}, xi = 0.1")
        add("b", f"c_tm_xi09::{tier}", V[("b", tier, 0.9)],
            f"time-mean c_ideal_tm, {tier}, xi = 0.9")
    lam0 = [V[("b", M2_TIERS[0.0], xi)] for xi in XIS]
    add("b", "m2_lambda0_flat_min", min(lam0),
        "min time-mean alignment over xi, M2 lambda = 0")
    add("b", "m2_lambda0_flat_max", max(lam0),
        "max time-mean alignment over xi, M2 lambda = 0")

    # -- (c) team size: young-team range and the two mob stages ------------
    small = [V[("c", t, xi)] for t in M2_TIERS.values() for xi in (0.1, 0.3, 0.5)]
    add("c", "team_small_min", min(small), "min team_tm over M2 at xi <= 0.5")
    add("c", "team_small_max", max(small), "max team_tm over M2 at xi <= 0.5")
    mob7 = [V[("c", t, 0.7)] for t in M2_TIERS.values()]
    mob9 = [V[("c", t, 0.9)] for t in M2_TIERS.values()]
    add("c", "team_xi07_min", min(mob7), "min team_tm over M2 at xi = 0.7")
    add("c", "team_xi07_max", max(mob7), "max team_tm over M2 at xi = 0.7")
    add("c", "team_xi09_min", min(mob9), "min team_tm over M2 at xi = 0.9")
    add("c", "team_xi09_max", max(mob9), "max team_tm over M2 at xi = 0.9")

    # -- BUG-0006 pre-vs-post contrast, read from the archived delta tables --
    delta = pd.read_csv(data_root_path(data_root) / "tables"
                        / "m2_xi_bug0006_delta.csv")
    d_tm = delta[(delta["arm"] == "M2_xi") & (delta["scope"] == "all")
                 & (delta["metric"] == "c_ideal_tm")]
    _require(len(d_tm) == 1, "m2_xi_bug0006_delta: c_ideal_tm row")
    add("b", "bug0006_m2_c_tm_delta", float(d_tm["mean_delta"].iloc[0]),
        "Measurement-correction mean change in time-mean alignment over all 200 M2 runs "
        "(tables/m2_xi_bug0006_delta.csv)")
    dw = pd.read_csv(data_root_path(data_root) / "tables"
                     / "w_v05_bug0006_delta.csv")

    def _w(scope, metric, col):
        q = dw[(dw["scope"] == scope) & (dw["metric"] == metric)]
        _require(len(q) == 1, f"w_v05 delta row {scope}/{metric}")
        return float(q[col].iloc[0])

    add("a", "bug0006_w_v05_harvest_max_abs_diff",
        float(dw[dw["metric"] == "harvested_fin"]["max_abs_diff"].abs().max()),
        "max |pre-post| harvested_fin over the whole w_v05 arm")
    add("b", "bug0006_w_v05_m0_c_tm_max_abs_diff",
        _w("g=0.0 (control)", "c_ideal_tm", "max_abs_diff"),
        "max |pre-post| c_ideal_tm, M0 (g = 0) control")
    add("b", "bug0006_w_v05_m1_mgstd5_c_tm_mean_delta",
        _w("g=0.1 (mgstd5)", "c_ideal_tm", "mean_delta"),
        "mean pre-post change in c_ideal_tm, M1 (g = 0.1), mgstd5")
    add("b", "bug0006_w_v05_m1_mgstd5_c_tm_max_abs_diff",
        _w("g=0.1 (mgstd5)", "c_ideal_tm", "max_abs_diff"),
        "max |pre-post| c_ideal_tm, M1 (g = 0.1), mgstd5")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figure ---
def build_figure(data_root=None):
    runs, pins = load_runs(data_root)
    series = series_rows(runs)
    anns = annotation_rows(series, pins, data_root)
    sidecar = pd.concat([series, anns], ignore_index=True)
    A = {r["metric"]: float(r["value"]) for _, r in anns.iterrows()}

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(13.4, 5.2))
    # final geometry BEFORE any text placement: callout clearance is tested
    # on the axes people will actually see
    fig.subplots_adjust(left=0.055, right=0.99, top=0.845, bottom=0.40,
                        wspace=0.26)
    fig.text(0.5, 0.958,
             "F10b_v2 · M2 (pull = 1.0, logit + equal_split) on the full visibility "
             "axis — strictly like-for-like time-mean alignment "
             "(mgstd5)",
             ha="center", va="top", fontsize=HOUSE_STYLE["title_size"] + 1)

    # ---- (a) macro efficiency -------------------------------------------
    _bands(ax_a)
    for tier in TIERS:
        _plot_tier(ax_a, series[series["metric"] == M_HARVEST], tier)
    _auto_ylim(ax_a, pad_hi=0.16)
    _place_text(ax_a, (
        f"entry arms flat to ξ = 0.5: "
        f"{A['entry_plateau_harvested_min'] * 100:.1f}–"
        f"{A['entry_plateau_harvested_max'] * 100:.1f}%\n"
        f"Stage-2 collapse at ξ = 0.9:\n"
        f"{A['stage2_collapse_min'] * 100:.1f}–"
        f"{A['stage2_collapse_max'] * 100:.1f}% (entry arms)\n"
        f"M0 falls further: {A['m0_xi09_harvested'] * 100:.1f}%"),
        tuple((0.03, y, "left", "bottom") for y in (0.04, 0.22, 0.40))
        + tuple((0.97, y, "right", "top") for y in (0.97, 0.79)))
    _band_labels(ax_a, prefer=("top", "bottom"))
    _cosmetics(ax_a, HOUSE_STYLE, "A · Macro efficiency",
               "visibility weight  $\\xi = 1 - c$",
               "harvested knowledge share\n(final step · run mean)")
    ax_a.set_xlim(0.02, 0.98)
    ax_a.set_xticks(list(XIS))

    # ---- (b) like-for-like hierarchy -------------------------------------
    _bands(ax_b)
    for tier in TIERS:
        _plot_tier(ax_b, series[series["metric"] == M_CORR], tier)
    _auto_ylim(ax_b, include=(0.0,), pad_hi=0.22)
    ax_b.axhline(0.0, color=PALETTE["gray"], linewidth=1.0, zorder=2)
    lo_b, hi_b = ax_b.get_ylim()
    ax_b.set_yticks([t for t in ax_b.get_yticks() if lo_b < t < hi_b])
    order = sorted(TIERS, key=lambda t: -A[f"c_tm_xi01::{t}"])
    SHORT = {M0_TIER: "M0 (g=0)", M1_TIER: "M1 (g=.1)",
             M2_TIERS[0.0]: "M2 λ = 0", M2_TIERS[1.0]: "M2 λ = 1"}
    lines = [f"{SHORT[t]}  {A[f'c_tm_xi01::{t}']:+.3f} → "
             f"{A[f'c_tm_xi09::{t}']:+.3f}" for t in order]
    _place_text(ax_b, "like-for-like hierarchy\ntime-mean c_ideal_tm\n"
                      "ξ = 0.1 → 0.9\n"
                      + "\n".join(("›  " if k else "    ") + ln
                                  for k, ln in enumerate(lines)),
                tuple((0.97, y, "right", "top") for y in (0.97, 0.79))
                + tuple((0.03, y, "left", "top") for y in (0.97, 0.52)))
    _band_labels(ax_b, prefer=("bottom", "top"))
    _cosmetics(ax_b, HOUSE_STYLE, "B · Like-for-like signal alignment",
               "visibility weight  $\\xi = 1 - c$",
               "Corr(prestige, ideal merit)\n(time mean · run mean)")
    ax_b.set_xlim(0.02, 0.98)
    ax_b.set_xticks(list(XIS))

    # ---- (c) M2 co-harvest team size -------------------------------------
    _bands(ax_c)
    for lam in (0.0, 1.0):
        _plot_tier(ax_c, series[series["metric"] == M_TEAM], M2_TIERS[lam])
    ax_c.axhline(KLANN_TEAM_SIZE, color=PALETTE["gray"], linestyle=":",
                 linewidth=1.3, zorder=2)
    _auto_ylim(ax_c, include=(KLANN_TEAM_SIZE, 0.0), pad_hi=0.16,
               floor=0.0)
    lo_c, hi_c = ax_c.get_ylim()
    kf = (KLANN_TEAM_SIZE - lo_c) / (hi_c - lo_c)
    _place_text(ax_c, (
        f"teams at ξ ≤ 0.5: {A['team_small_min']:.1f}–"
        f"{A['team_small_max']:.1f}\n"
        f"ξ = 0.7: {A['team_xi07_min']:.1f}–{A['team_xi07_max']:.1f}\n"
        f"ξ = 0.9: {A['team_xi09_min']:.1f}–{A['team_xi09_max']:.1f} (mobs)"),
        ((0.03, 0.62, "left", "top"), (0.03, 0.45, "left", "top"),
         (0.97, 0.55, "right", "top"), (0.03, 0.97, "left", "top")))
    # the unfitted benchmark is labelled inline at the rule's left end (no
    # third legend entry, so nothing sits on the mob curves)
    _place_text(ax_c, f"{KLANN_TEAM_SIZE:.2f} Klann team size\n{BENCH_LABEL}",
                ((0.03, kf + 0.012, "left", "bottom"),
                 (0.97, kf + 0.012, "right", "bottom"),
                 (0.03, kf - 0.012, "left", "top")))
    _band_labels(ax_c, prefer=("top", "bottom"))  # C rises: top is free
    _cosmetics(ax_c, HOUSE_STYLE, "C · M2 co-harvest team size",
               "visibility weight  $\\xi = 1 - c$",
               "co-harvest team size team_tm\n(time mean · run mean)")
    ax_c.set_xlim(0.02, 0.98)
    ax_c.set_xticks(list(XIS))

    # ---- frame, shared legend, footnote ----------------------------------
    root = data_root_path(data_root)
    rel = root.relative_to(REPO) if root.is_relative_to(REPO) else root
    note_lines = [
        f"Sources (read-only): {rel}/w_v05_runs.csv for M0 (g = 0) and M1 (g = 0.1) — "
        f"{pins['n_w_v05']} runs; {rel}/m2_xi_runs.csv for M2 (pull = 1.0, "
        f"logit + equal_split) — {pins['n_m2']} runs; 20 seeds per cell, mgstd5, "
        "visibility reward on; error bars 95% CI = mean ± 1.96 SE over runs.",
        f"Vanishing-factor pins (read from the archives): M0/M1 "
        f"{pins[M0_TIER]['vanishing']:g} (w_v05), M2 λ = 0/1 "
        f"{pins[M2_TIERS[0.0]]['vanishing']:g} (m2_xi). Panel A plots harvested_fin for "
        "every tier — the corrected archives expose one uniform statistic (identical "
        "to the early-draft percentage_knowledge_harvested; the remeasure leaves harvest "
        f"invariant, max |Δ| = {A['bug0006_w_v05_harvest_max_abs_diff']:.1e}).",
        f"The measurement correction moved M2 time-mean alignment by "
        f"{A['bug0006_m2_c_tm_delta']:+.4f} on average over its 200 runs "
        "(tables/m2_xi_bug0006_delta.csv) and the M1 (g = 0.1) rows by "
        f"{A['bug0006_w_v05_m1_mgstd5_c_tm_mean_delta']:+.4f} on mgstd5 (max |Δ| "
        f"{A['bug0006_w_v05_m1_mgstd5_c_tm_max_abs_diff']:.4f}), because the bug "
        "dropped unscored step-t newborns from the correlation; the M0 (g = 0) "
        f"control is unchanged (max |Δ| = "
        f"{A['bug0006_w_v05_m0_c_tm_max_abs_diff']:.1e}). Panel B therefore "
        "supersedes the early-draft hierarchy. Zero is a y-tick plus the reference "
        "rule — no stray label; the Stage 1 / kink / Stage 2 bands are shaded "
        "regions only, so no boundary rule crosses the series or the annotation. "
        "Klann team size 2.65: Empirical benchmark (unfitted). Model-internal, "
        "not calibrated.",
    ]
    note = fig.text(0.008, 0.006, "\n".join(textwrap.fill(t, width=178)
                                            for t in note_lines),
                    fontsize=HOUSE_STYLE["note_size"], color=PALETTE["gray"],
                    va="bottom")
    handles, labels = ax_a.get_legend_handles_labels()
    place_figure_legend(fig, handles, labels, 4, note)

    audit_text_over_data(fig, (ax_a, ax_b, ax_c))
    visual_qa.assert_figure_clean(fig, [
        {"ax": ax_a, "metric": M_HARVEST, "expect_lines": len(TIERS),
         "points_per_line": len(XIS), "sidecar": sidecar,
         "filters": {"statistic": "mean"}},
        {"ax": ax_b, "metric": M_CORR, "expect_lines": len(TIERS),
         "points_per_line": len(XIS), "sidecar": sidecar,
         "filters": {"statistic": "mean"}},
        {"ax": ax_c, "metric": M_TEAM, "expect_lines": len(M2_TIERS),
         "points_per_line": len(XIS), "sidecar": sidecar,
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
    return {"png": str(png), "pdf": str(pdf), "data_csv": str(csv_path)}


if __name__ == "__main__":
    print(build())
