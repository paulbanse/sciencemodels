"""F6_v2 -- two-stage proxy failure, post-BUG-0006 refresh (Figure 1, sec. 4.1).

Rebuilds the old ``F6_two_stage_proxy_failure`` from the post-``BUG-0006``
archives on one common visibility axis ``xi = 1 - c`` (mgstd5, visibility
reward on, 20 seeds per cell):

  tier                     archive / cell                            pin
  ------------------------------------------------------------------------
  M0 (g = 0)               bug0006_remeasure_v2/w_v05_runs.csv        vanishing 0.5
  M1 (g = 0.1)             bug0006_remeasure_v2/w_v05_runs.csv        vanishing 0.5
  M1 + turnover_staggered  turnover_v2/runs.csv (age_hazard_staggered,
                           entry = 0.10 = the M1 g pin)               vanishing 0.9
  M2 lambda = 0            bug0006_remeasure_v2/m2_xi_runs.csv        vanishing 0.9
  M2 lambda = 1            bug0006_remeasure_v2/m2_xi_runs.csv        vanishing 0.9

Panels:
  (a) macro efficiency      : harvested knowledge share (``harvested_fin``)
  (b) strictly like-for-like: time-mean ``c_ideal_tm`` in every arm (no
      time-mean / final-step mixing -- the distortion F6 originally suffered)

Stage bands are F6's exact grammar: Stage 1 (xi <= 0.6), welfare kink
[0.6, 0.8], Stage 2 (xi >= 0.8).  The bands are shaded REGIONS only: the
old dashed band-boundary lines crossed every series (and, in F10b, the
annotation block); v3 drops them so no rule cuts the data.

Provenance is data-driven: the vanishing pins are read from the run
archives (w_v05, m2_xi) and from ``tables/four_tier_xi_summary.csv`` (the
turnover arm, whose runs.csv has no vanishing column); the horizon pins are
read from the archives / ``turnover_v2/design.json``.  The builder also
asserts its own per-cell means against ``tables/four_tier_xi_summary.csv``
at 1e-9, so the figure and the archived tier table cannot drift apart.

Rendering contract: the shared ``visual_qa.assert_figure_clean`` runs on the
rendered canvas, plus ``audit_text_over_data`` -- a local display-space
text-vs-data gate, because ``visual_qa`` compares line / error-bar segments
in DATA coordinates against DISPLAY-space text boxes and therefore cannot
see a label sitting on a curve.  Every callout and band label is placed by
measured clearance (largest note font that still clears every segment, rule
and other text artist), never by a guessed anchor.

Vocabulary: hue+marker = tier; every plotted value, error bar and
annotation is computed here from the raw CSVs and written into the
``*_data.csv`` sidecar.  ``visual_qa.assert_figure_clean`` runs on the
RENDERED canvas before ``savefig`` (rendered-artist data contract +
collision-free layout contract).

Run:  MPLBACKEND=Agg /opt/hermes/.venv/bin/python \
          scripts/report_notebook/fig_F6_v2.py
"""

from __future__ import annotations

import json
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

FNAME = "F6_v2_two_stage_proxy_failure"
DEFAULT_DATA_ROOT = REPO / "data"
LANDSCAPE = "mgstd5"
XIS = (0.1, 0.3, 0.5, 0.7, 0.9)
Z95 = 1.96                       # normal-approximation 95% CI over runs
BAND_STAGE1, BAND_KINK, BAND_STAGE2 = (0.3, 0.6), (0.6, 0.8), (0.8, 0.9)

#: panel metrics (sidecar vocabulary)
M_HARVEST = "harvested knowledge share"
M_CORR = "Corr(prestige, ideal merit) (time mean)"
PANELS = {"a": M_HARVEST, "b": M_CORR}

#: tier grammar: M0 purple, M1 grey dashed hollow, staggered orange
#: dash-dot diamond, M2 lambda=0 blue dashed squares, lambda=1 green triangles
TIERS = (
    "M0 (g = 0)",
    "M1 (g = 0.1)",
    "M1 + turnover_staggered",
    "M2 λ = 0 (prestige matching)",
    "M2 λ = 1 (closeness × local merit)",
)
STYLE = {
    "M0 (g = 0)": dict(color=PALETTE["purple"], ls="-", marker="o",
                        mfc=PALETTE["purple"]),
    "M1 (g = 0.1)": dict(color=PALETTE["gray"], ls="--", marker="o",
                         mfc="white"),
    "M1 + turnover_staggered": dict(color=PALETTE["orange"], ls="-.",
                                    marker="D", mfc=PALETTE["orange"]),
    "M2 λ = 0 (prestige matching)": dict(color=PALETTE["blue"], ls="--",
                                         marker="s", mfc=PALETTE["blue"]),
    "M2 λ = 1 (closeness × local merit)": dict(color=PALETTE["green"], ls="-",
                                               marker="^", mfc=PALETTE["green"]),
}
ENTRY_ARMS = ("M1 (g = 0.1)", "M2 λ = 0 (prestige matching)",
              "M2 λ = 1 (closeness × local merit)")
TURNOVER_TIER = "M1 + turnover_staggered"
M0_TIER, M1_TIER = TIERS[0], TIERS[1]

#: four_tier_xi_summary.csv tier ids, for the aggregation identity check
TABLE_TIER = {
    "M0 (g = 0)": "M0_closed_Wv05",
    "M1 (g = 0.1)": "M1_lineage_Wv05",
    TURNOVER_TIER: "M1_turnover_staggered",
    "M2 λ = 0 (prestige matching)": "M2_lambda0",
    "M2 λ = 1 (closeness × local merit)": "M2_lambda1",
}

VALUE_COL = {"a": "harvested_fin", "b": "c_ideal_tm"}


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _mean_ci(vals) -> tuple[float, float, float, int]:
    v = np.asarray(vals, dtype=float)
    n = int(v.size)
    m = float(v.mean())
    half = Z95 * float(v.std(ddof=1)) / np.sqrt(n) if n > 1 else float("nan")
    return m, m - half, m + half, n


# ------------------------------------------------------------------ data ---
def data_root_path(data_root=None) -> Path:
    return Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT


def _arm_paths(data_root=None) -> tuple[Path, Path, Path]:
    root = data_root_path(data_root)
    return (root / "bug0006_remeasure_v2" / "w_v05_runs.csv",
            root / "bug0006_remeasure_v2" / "m2_xi_runs.csv",
            root / "turnover_v2" / "runs.csv")


def load_runs(data_root=None) -> tuple[pd.DataFrame, dict]:
    """Tidy runs for the five tiers, plus provenance pins.

    Returns ``(runs, pins)``; ``runs`` has one row per run with columns
    ``tier, xi, harvested_fin, c_ideal_tm, seed, archive``; ``pins`` maps
    tier -> {vanishing, horizon, archive, cells, runs}.
    """
    f_w, f_m2, f_t = _arm_paths(data_root)
    for f in (f_w, f_m2, f_t):
        _require(f.exists(), f"missing archive {f}")

    # ---- W_v05: M0 (g = 0) and M1 (g = 0.1), visibility reward on ---------
    w = pd.read_csv(f_w)
    _require({"mgstd5", "mg6"} <= set(w["landscape"].unique()),
             f"w_v05: expected mgstd5 + mg6, got {sorted(w['landscape'].unique())}")
    wv = w[(w["landscape"] == LANDSCAPE)
           & (w["use_visibility_reward"] == True)].copy()  # noqa: E712
    _require(len(wv) == 200,
             f"w_v05 mgstd5+visibility: expected 200 runs, got {len(wv)}")
    _require(set(np.round(wv["vanishing"], 3)) == {0.5},
             "w_v05: vanishing must be pinned at 0.5 across the arm")
    wv["xi"] = np.round(1.0 - wv["curiosity"], 1)
    _require(sorted(wv["xi"].unique()) == list(XIS),
             f"w_v05: xi grid {sorted(wv['xi'].unique())}")
    _require((wv.groupby(["agent_generation_time", "xi"]).size() == 20).all(),
             "w_v05: expected 20 seeds per (g, xi) cell")
    wv["tier"] = np.where(np.round(wv["agent_generation_time"], 2) == 0.0,
                          M0_TIER, M1_TIER)
    _require(set(wv["tier"].unique()) == {M0_TIER, M1_TIER},
             "w_v05: unexpected agent_generation_time grid")

    # ---- M2 xi-extension: lambda in {0, 1}, vanishing 0.9 -----------------
    m = pd.read_csv(f_m2)
    _require(len(m) == 200, f"m2_xi: expected 200 runs, got {len(m)}")
    _require(set(np.round(m["vanishing"], 3)) == {0.9},
             "m2_xi: vanishing must be pinned at 0.9")
    _require((m["landscape"] == LANDSCAPE).all(), "m2_xi: landscape")
    _require((m["use_visibility_reward"] == True).all(),  # noqa: E712
             "m2_xi: visibility reward must be on")
    _require(set(np.round(m["dyadic_move_pull"], 3)) == {1.0},
             "m2_xi: pull must be 1.0")
    m["xi"] = np.round(m["xi"], 1)
    _require(np.allclose(m["xi"], np.round(1.0 - m["curiosity"], 1)),
             "m2_xi: xi must equal 1 - curiosity")
    _require(sorted(m["xi"].unique()) == list(XIS), "m2_xi: xi grid")
    _require((m.groupby(["dyadic_lambda", "xi"]).size() == 20).all(),
             "m2_xi: expected 20 seeds per (lambda, xi) cell")
    m["tier"] = np.where(m["dyadic_lambda"] == 0.0,
                         "M2 λ = 0 (prestige matching)",
                         "M2 λ = 1 (closeness × local merit)")

    # ---- turnover arm: age_hazard_staggered, entry = M1's g pin -----------
    t = pd.read_csv(f_t)
    _require(len(t) == 1520, f"turnover_v2: expected 1,520 runs, got {len(t)}")
    _require("age_hazard_staggered" in set(t["exit_mode"].unique()),
             "turnover_v2: age_hazard_staggered mode missing")
    # the turnover entry rate is matched to M1's g pin (w_v05), and the M2
    # arm must carry the same pin -- both checked, neither assumed
    g_pin = float(np.max(np.round(wv["agent_generation_time"], 3)))
    _require(len(np.unique(np.round(m["agent_generation_time"], 3))) == 1,
             "m2_xi: expected one g pin")
    _require(abs(float(np.round(m["agent_generation_time"].iloc[0], 3))
                 - g_pin) <= 1e-9,
             f"m2_xi g pin {m['agent_generation_time'].iloc[0]} != M1 g {g_pin}")
    ts = t[(t["exit_mode"] == "age_hazard_staggered")
           & (t["landscape"] == LANDSCAPE)
           & (np.round(t["entry"], 2) == g_pin)].copy()
    _require(len(ts) == 100,
             f"turnover_v2 staggered entry={g_pin}: expected 100 runs "
             f"(5 xi x 20 seeds), got {len(ts)}")
    ts["xi"] = np.round(ts["xi"], 1)
    _require(sorted(ts["xi"].unique()) == list(XIS),
             f"turnover_v2 staggered: xi grid {sorted(ts['xi'].unique())}")
    _require((ts.groupby("xi").size() == 20).all(),
             "turnover_v2 staggered: expected 20 seeds per xi")
    ts["tier"] = TURNOVER_TIER

    keep = ["tier", "xi", "harvested_fin", "c_ideal_tm", "seed"]
    runs = pd.concat([wv[keep].assign(archive=f_w.name),
                      m[keep].assign(archive=f_m2.name),
                      ts[keep].assign(archive=f_t.name)], ignore_index=True)
    _require(len(runs) == 500,
             f"five tiers: expected 500 runs (200 + 200 + 100), got {len(runs)}")

    # horizon pins: w_v05/m2 arms are 200-step runs (design.json), the
    # turnover archive is 400-step (design.json); never assume, read them.
    design_w = json.loads((data_root_path(data_root) / "bug0006_remeasure_v2"
                           / "design.json").read_text())
    design_t = json.loads((data_root_path(data_root) / "turnover_v2"
                           / "design.json").read_text())
    pins = {
        "M0 (g = 0)": dict(vanishing=0.5, horizon=int(design_w["horizon"]),
                           archive=f_w.name, cells=5, runs=100),
        "M1 (g = 0.1)": dict(vanishing=0.5, horizon=int(design_w["horizon"]),
                             archive=f_w.name, cells=5, runs=100),
        TURNOVER_TIER: dict(vanishing=float("nan"),
                            horizon=int(design_t["horizon"]),
                            archive=f_t.name, cells=5, runs=100,
                            entry=g_pin, mode="age_hazard_staggered"),
        "M2 λ = 0 (prestige matching)": dict(vanishing=0.9,
                                             horizon=int(design_w["horizon"]),
                                             archive=f_m2.name, cells=5,
                                             runs=100),
        "M2 λ = 1 (closeness × local merit)": dict(
            vanishing=0.9, horizon=int(design_w["horizon"]),
            archive=f_m2.name, cells=5, runs=100),
    }
    # the turnover archive carries no vanishing column: take the pin from
    # the archived tier table instead of inventing one.
    four = pd.read_csv(data_root_path(data_root) / "bug0006_remeasure_v2"
                       / "tables" / "four_tier_xi_summary.csv")
    tv = four[four["tier"] == TABLE_TIER[TURNOVER_TIER]]["vanishing"].unique()
    _require(len(tv) == 1, "four_tier_xi_summary: turnover vanishing pin")
    pins[TURNOVER_TIER]["vanishing"] = float(tv[0])
    return runs, pins


def series_rows(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean / 95% CI per (panel, tier, xi) -- one row per plotted statistic."""
    rows = []
    for panel, metric in PANELS.items():
        col = VALUE_COL[panel]
        for tier in TIERS:
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
    return pd.DataFrame(rows)


def _value_table(series: pd.DataFrame) -> dict:
    """(panel, tier, xi) -> value, from the plotted aggregates only."""
    out = {}
    for _, r in series[series["statistic"] == "mean"].iterrows():
        out[(r["panel"], r["tier"], float(r["xi"]))] = float(r["value"])
    return out


def annotation_rows(series: pd.DataFrame, pins: dict,
                    data_root=None) -> pd.DataFrame:
    """Every number written on the canvas, derived from the plotted means."""
    V = _value_table(series)
    rows = []

    def add(panel: str, metric: str, value: float, note: str) -> None:
        rows.append(dict(figure=FNAME, panel=panel, metric=metric,
                         statistic="annotation", tier="", xi=np.nan,
                         landscape=LANDSCAPE, value=float(value),
                         ci_lo=np.nan, ci_hi=np.nan, n=100, source_file=note))

    # -- (a) macro: plateau, Stage-2 collapse, staggered rescue -------------
    plateau = [V[("a", t, xi)] for t in ENTRY_ARMS for xi in (0.1, 0.3, 0.5)]
    collapse = [V[("a", t, 0.9)] for t in ENTRY_ARMS]
    add("a", "entry_plateau_harvested_min", min(plateau),
        "min harvested share over entry arms (M1, M2 lam=0/1) at xi <= 0.5")
    add("a", "entry_plateau_harvested_max", max(plateau),
        "max harvested share over entry arms at xi <= 0.5")
    add("a", "stage2_harvest_min", min(collapse),
        "min harvested share over entry arms at xi = 0.9")
    add("a", "stage2_harvest_max", max(collapse),
        "max harvested share over entry arms at xi = 0.9")
    add("a", "m0_xi09_harvested", V[("a", M0_TIER, 0.9)],
        "M0 (g = 0) harvested share at xi = 0.9")
    add("a", "staggered_xi01_harvested", V[("a", TURNOVER_TIER, 0.1)],
        "M1 + turnover_staggered harvested share at xi = 0.1")
    add("a", "staggered_xi09_harvested", V[("a", TURNOVER_TIER, 0.9)],
        "M1 + turnover_staggered harvested share at xi = 0.9")
    add("a", "staggered_rescue_pp",
        100.0 * (V[("a", TURNOVER_TIER, 0.9)] - V[("a", M1_TIER, 0.9)]),
        "staggered minus M1 harvested share at xi = 0.9, percentage points")

    # -- (b) like-for-like alignment: endpoints and flatness ----------------
    for tier in TIERS:
        add("b", f"c_tm_xi01::{tier}", V[("b", tier, 0.1)],
            f"time-mean c_ideal_tm, {tier}, xi = 0.1")
        add("b", f"c_tm_xi09::{tier}", V[("b", tier, 0.9)],
            f"time-mean c_ideal_tm, {tier}, xi = 0.9")
    lam0 = [V[("b", "M2 λ = 0 (prestige matching)", xi)] for xi in XIS]
    add("b", "m2_lambda0_flat_min", min(lam0),
        "min time-mean alignment over xi, M2 lambda = 0")
    add("b", "m2_lambda0_flat_max", max(lam0),
        "max time-mean alignment over xi, M2 lambda = 0")
    add("b", "m0_alignment_drop_pct",
        100.0 * (V[("b", M0_TIER, 0.9)] / V[("b", M0_TIER, 0.1)] - 1.0),
        "M0 time-mean alignment change xi = 0.1 -> 0.9, percent")
    add("b", "turnover_vanishing_pin", pins[TURNOVER_TIER]["vanishing"],
        "vanishing factor of the turnover arm (turnover_v2 has no column; "
        "pin read from tables/four_tier_xi_summary.csv)")

    # -- BUG-0006 effect on this figure's own statistic (read, not typed) ---
    root = data_root_path(data_root)
    dw = pd.read_csv(root / "bug0006_remeasure_v2" / "tables"
                     / "w_v05_bug0006_delta.csv")
    dm = pd.read_csv(root / "bug0006_remeasure_v2" / "tables"
                     / "m2_xi_bug0006_delta.csv")

    def _w(scope, metric, col):
        q = dw[(dw["scope"] == scope) & (dw["metric"] == metric)]
        _require(len(q) == 1, f"w_v05 delta row {scope}/{metric}")
        return float(q[col].iloc[0])

    add("b", "bug0006_m0_c_tm_max_abs_diff",
        _w("g=0.0 (control)", "c_ideal_tm", "max_abs_diff"),
        "max |pre-post| c_ideal_tm, M0 (g = 0) control (no unscored newborns)")
    add("b", "bug0006_m1_mgstd5_c_tm_mean_delta",
        _w("g=0.1 (mgstd5)", "c_ideal_tm", "mean_delta"),
        "mean pre-post change in c_ideal_tm, M1 (g = 0.1), mgstd5")
    add("b", "bug0006_m2_c_tm_mean_delta",
        float(dm[(dm["arm"] == "M2_xi") & (dm["scope"] == "all")
                 & (dm["metric"] == "c_ideal_tm")]["mean_delta"].iloc[0]),
        "mean pre-post change in c_ideal_tm over the 200 M2 runs")
    return pd.DataFrame(rows)


def check_against_tier_table(series: pd.DataFrame, data_root=None) -> None:
    """Aggregation identity: our per-cell means == archived tier table."""
    four = pd.read_csv(data_root_path(data_root) / "bug0006_remeasure_v2"
                       / "tables" / "four_tier_xi_summary.csv")
    col = {"a": "harvested_fin", "b": "c_ideal_tm"}
    for _, r in series[series["statistic"] == "mean"].iterrows():
        tid = TABLE_TIER[r["tier"]]
        q = four[(four["tier"] == tid) & (np.round(four["xi"], 1) == r["xi"])]
        _require(len(q) == 1, f"four_tier_xi_summary: no row {tid} xi={r['xi']}")
        ref = float(q[col[r["panel"]]].iloc[0])
        _require(abs(ref - float(r["value"])) <= 1e-9,
                 f"tier table drift {tid} xi={r['xi']} panel={r['panel']}: "
                 f"{r['value']:.12g} vs {ref:.12g}")


# --------------------------------------------------------------- drawing ---
def _bands(ax) -> None:
    """F6's exact two-stage bands -- shaded regions, no boundary rules."""
    ax.axvspan(*BAND_STAGE1, color=PALETTE["gray_light"], alpha=0.16, zorder=0)
    ax.axvspan(*BAND_KINK, facecolor=PALETTE["purple_light"], alpha=0.16,
               hatch="///", edgecolor=PALETTE["gray"], linewidth=0, zorder=0)
    ax.axvspan(*BAND_STAGE2, color=PALETTE["gray_light"], alpha=0.32, zorder=0)


def _plotted_extent(ax) -> tuple[float, float]:
    vals = visual_qa._series_values(ax)
    parts = [v for k in ("line_values", "line_endpoints", "bar_values")
             for v in vals[k] if len(v)]
    if not parts:
        return 0.0, 1.0
    allv = np.concatenate(parts)
    return float(allv.min()), float(allv.max())


def _auto_ylim(ax, include=(), pad_lo: float = 0.08,
               pad_hi: float = 0.10, floor: float | None = None
               ) -> tuple[float, float]:
    """ylim from the rendered values (incl. error endpoints) + headroom.

    ``floor`` clamps the lower limit (e.g. 0 for a strictly non-negative
    quantity) AFTER padding, so the axis still keeps its 3% headroom.
    """
    lo, hi = _plotted_extent(ax)
    for v in include:
        lo, hi = min(lo, float(v)), max(hi, float(v))
    span = hi - lo if hi > lo else max(abs(hi), 1.0)
    lo, hi = lo - pad_lo * span, hi + pad_hi * span
    if floor is not None:
        lo = max(lo, float(floor))
    ax.set_ylim(lo, hi)
    return lo, hi


def _display_segments(ax) -> list:
    """Plotted geometry in DISPLAY pixels (bars, markers, error bars, refs).

    ``visual_qa._segments_from_artist`` returns data coordinates for lines
    and error bars, which cannot be compared against a display-space text
    bbox; this helper applies ``ax.transData`` first so the text-vs-data
    contract actually binds (see the module docstring / report note).
    """
    art = visual_qa.collect_panel_artists(ax)
    tr = ax.transData
    segs: list = []

    def _pts(xy):
        xy = np.asarray(xy, dtype=float)
        if xy.ndim == 1:
            xy = xy.reshape(-1, 2)
        return tr.transform(xy)

    for a in art["lines"]:
        pts = _pts(a.get_xydata())
        if len(pts) == 1:
            segs.append((tuple(pts[0]), tuple(pts[0])))
        else:
            segs.extend(zip(pts[:-1], pts[1:]))
    for cont in art["errorbars"]:
        dl, caps, bcs = visual_qa._eb_parts(cont)
        for bc in bcs:
            for s in bc.get_segments():
                p = _pts(s)
                segs.append((tuple(p[0]), tuple(p[-1])))
        for cap in caps:
            p = _pts(cap.get_xydata())
            segs.append((tuple(p[0]), tuple(p[-1] if len(p) > 1 else p[0])))
        if dl is not None:
            p = _pts(dl.get_xydata())
            if len(p) == 1:
                segs.append((tuple(p[0]), tuple(p[0])))
    for bc in art["bars"]:
        for patch in bc.patches:
            bb = patch.get_window_extent()
            segs.append(((bb.x0, bb.y0), (bb.x1, bb.y1)))
    for ln in art["refs"]:
        p = _pts(ln.get_xydata())
        segs.extend(zip(p[:-1], p[1:]))
    return segs


def _text_hits_data(ax, t) -> bool:
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bb = t.get_window_extent(renderer=r)
    return any(visual_qa._seg_hits_box(s, bb) for s in _display_segments(ax))


def audit_text_over_data(fig, axes) -> None:
    """Local strict text-vs-data gate in display space (zero tolerance).

    Complements ``visual_qa.assert_figure_clean``: that contract compares
    line/error-bar segments in data coordinates against display-space text
    boxes, so it cannot see a label sitting on a curve. This one can, and
    it runs here for every text artist of every panel (excluding tick
    labels, which live outside the axes).
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    offenders = []
    for ax in axes:
        segs = _display_segments(ax)
        ab = ax.bbox
        for t in visual_qa._text_artists(fig, ax):
            if t is ax.title or t is ax.xaxis.label or t is ax.yaxis.label:
                continue
            bb = visual_qa._bbox(t, r)
            if bb is None:
                continue
            if bb.x1 < ab.x0 or bb.x0 > ab.x1 or bb.y1 < ab.y0 or bb.y0 > ab.y1:
                continue
            if any(visual_qa._seg_hits_box(s, bb) for s in segs):
                offenders.append(f"panel[{ax.get_title()[:24]}]: "
                                 f"{t.get_text()[:40]!r} over plotted data")
    _require(not offenders, "; ".join(offenders))


def _text_hits_text(ax, t) -> bool:
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bb = t.get_window_extent(renderer=r)
    others = [x for x in visual_qa._text_artists(fig, ax) if x is not t]
    for o in others:
        ob = visual_qa._bbox(o, r)
        if ob is not None and visual_qa._boxes_overlap(bb, ob):
            return True
    return False


def _place_text(ax, text: str, candidates, *, zorder: int = 6,
                sizes=(8, 7.4, 6.8)):
    """First (anchor, size) where the block touches no data and no text.

    Size is a fallback axis, not a cosmetic choice: a long annotation block
    in a narrow panel gets the largest house-note size that still clears
    every plotted segment, error bar, rule and other text artist.
    """
    for size in sizes:
        for xf, yf, ha, va in candidates:
            t = ax.text(xf, yf, text, transform=ax.transAxes,
                        fontsize=size, color=PALETTE["gray"],
                        ha=ha, va=va, zorder=zorder)
            if not _text_hits_data(ax, t) and not _text_hits_text(ax, t):
                return t
            t.remove()
    raise AssertionError(f"no free placement for callout {text[:40]!r}")


def _band_labels(ax, prefer=("top", "bottom")) -> None:
    """Stage labels, each placed where it touches neither data nor text."""
    slots = {"top": ((0.965, "top"), (0.90, "top")),
             "bottom": ((0.035, "bottom"), (0.10, "bottom"))}
    candidates = tuple(s for side in prefer for s in slots[side])
    for x, lab in ((0.45, "Stage 1"), (0.70, "kink"), (0.90, "Stage 2")):
        for yf, va in candidates:
            t = ax.text(x, yf, lab, transform=ax.get_xaxis_transform(),
                        ha="center", va=va, fontsize=HOUSE_STYLE["tick_size"],
                        color=PALETTE["gray"], zorder=5)
            if not _text_hits_data(ax, t) and not _text_hits_text(ax, t):
                break
            t.remove()
        else:
            raise AssertionError(f"no free slot for band label {lab!r}")


def place_figure_legend(fig, handles, labels, ncol: int, note_text, *,
                        gap: float = 8.0):
    """Figure legend in the band between the x-axis labels and the footnote.

    Positioned from measured geometry (never a guessed anchor) and asserted
    against both neighbours — a gap ``visual_qa`` leaves open, since it does
    not put ``fig.legends`` texts into its collision set.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    xl = min(ax.xaxis.label.get_window_extent(r).y0 for ax in fig.axes)
    fn = note_text.get_window_extent(r).y1
    y_top = (xl - gap) / fig.bbox.height
    leg = fig.legend(handles, labels, loc="upper center",
                     bbox_to_anchor=(0.5, y_top), ncol=ncol, frameon=False,
                     fontsize=HOUSE_STYLE["tick_size"] - 1, handlelength=2.2,
                     columnspacing=1.7)
    fig.canvas.draw()
    bb = leg.get_window_extent(r)
    _require(bb.y0 > fn + 1.0,
             f"figure legend overlaps the footnote ({bb.y0:.0f} <= {fn:.0f} px)")
    _require(bb.y1 < xl - 1.0,
             "figure legend overlaps the x-axis labels")
    return leg


def _plot_tier(ax, s: pd.DataFrame, tier: str) -> None:
    d = s[s["tier"] == tier].sort_values("xi")
    _require(len(d) == len(XIS), f"{tier}: {len(d)} points, expected {len(XIS)}")
    st = STYLE[tier]
    ax.errorbar(d["xi"].to_numpy(float), d["value"].to_numpy(float),
                yerr=np.vstack([d["value"] - d["ci_lo"],
                                d["ci_hi"] - d["value"]]),
                color=st["color"], linestyle=st["ls"], marker=st["marker"],
                markersize=5.2, linewidth=1.7, capsize=2.4, elinewidth=0.9,
                label=tier, markerfacecolor=st["mfc"],
                markeredgecolor=st["color"], markeredgewidth=1.0, zorder=4)


# ---------------------------------------------------------------- figure ---
def build_figure(data_root=None):
    """Build the figure in memory: returns (fig, sidecar, runs, pins)."""
    runs, pins = load_runs(data_root)
    series = series_rows(runs)
    check_against_tier_table(series, data_root)
    anns = annotation_rows(series, pins, data_root)
    sidecar = pd.concat([series, anns], ignore_index=True)
    A = {r["metric"]: float(r["value"]) for _, r in anns.iterrows()}

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.4, 5.9))
    # final geometry BEFORE any text placement: callout clearance is tested
    # on the axes people will actually see
    fig.subplots_adjust(left=0.062, right=0.988, top=0.845, bottom=0.40,
                        wspace=0.21)
    fig.text(0.5, 0.955,
             "F6_v2 · Two-stage proxy failure — macro harvest and "
             "strictly like-for-like time-mean alignment across five tiers "
             "(mgstd5)",
             ha="center", va="top",
             fontsize=HOUSE_STYLE["title_size"] + 1)

    for ax, panel, title, ylabel in (
            (ax_a, "a", "A · Macro efficiency: harvested knowledge share",
             "harvested knowledge share\n(final step · run mean)"),
            (ax_b, "b", "B · Like-for-like signal alignment",
             "Corr(prestige, ideal merit)\n(time mean · run mean)")):
        _bands(ax)
        s_metric = series[(series["metric"] == PANELS[panel])]  # single metric
        for tier in TIERS:
            _plot_tier(ax, s_metric, tier)
        ax.set_xlim(0.02, 0.98)
        ax.set_xticks(list(XIS))
        _cosmetics(ax, HOUSE_STYLE, title,
                   "visibility weight  $\\xi = 1 - c$", ylabel)

    _auto_ylim(ax_a)
    _auto_ylim(ax_b, include=(0.0,))
    # zero-alignment reference in panel b: a rule, never a stray text label
    ax_b.axhline(0.0, color=PALETTE["gray"], linewidth=1.0, zorder=2)
    ticks_b = [t for t in ax_b.get_yticks() if ax_b.get_ylim()[0] < t < ax_b.get_ylim()[1]]
    ax_b.set_yticks(ticks_b)

    # callouts first (they need the largest free block), then the band
    # labels, which are placed around whatever the callouts took
    _place_text(ax_a, (
        f"entry arms flat to ξ = 0.5: "
        f"{A['entry_plateau_harvested_min'] * 100:.1f}–"
        f"{A['entry_plateau_harvested_max'] * 100:.1f}%\n"
        f"Stage-2 collapse (ξ = 0.9): "
        f"{A['stage2_harvest_min'] * 100:.1f}–"
        f"{A['stage2_harvest_max'] * 100:.1f}% (entry arms);\n"
        f"M0 {A['m0_xi09_harvested'] * 100:.1f}%; "
        f"turnover_staggered {A['staggered_xi09_harvested'] * 100:.1f}% "
        f"({A['staggered_rescue_pp']:+.1f} pp vs M1)"),
        tuple((0.03, y, "left", "bottom") for y in (0.04, 0.22, 0.40))
        + tuple((0.97, y, "right", "top") for y in (0.97, 0.79, 0.61)))

    order = sorted(TIERS, key=lambda t: -A[f"c_tm_xi01::{t}"])
    SHORT = {M0_TIER: "M0 (g = 0)", M1_TIER: "M1 (g = 0.1)",
             TURNOVER_TIER: "M1 + staggered",
             "M2 λ = 0 (prestige matching)": "M2 λ = 0",
             "M2 λ = 1 (closeness × local merit)": "M2 λ = 1"}
    lines = [f"{SHORT[t]}  {A[f'c_tm_xi01::{t}']:+.3f} → "
             f"{A[f'c_tm_xi09::{t}']:+.3f}" for t in order]
    header = "strictly like-for-like c_ideal_tm\n(ξ = 0.1 → 0.9)"
    _place_text(ax_b, header + "\n" + "\n".join(
        ("" if k == 0 else "›   ") + ln for k, ln in enumerate(lines)),
        tuple((0.97, y, "right", "top") for y in (0.97, 0.79))
        + tuple((0.03, y, "left", "top") for y in (0.97, 0.52, 0.30)))
    _band_labels(ax_a, prefer=("top", "bottom"))
    _band_labels(ax_b, prefer=("bottom", "top"))

    src = data_root_path(data_root)
    rel = src.relative_to(REPO) if src.is_relative_to(REPO) else src
    note_lines = [
        f"Sources (read-only): {rel}/bug0006_remeasure_v2/w_v05_runs.csv (M0 g = 0, "
        f"M1 g = 0.1), {rel}/bug0006_remeasure_v2/m2_xi_runs.csv (M2 pull = 1.0, "
        f"logit + equal_split) and {rel}/turnover_v2/runs.csv (age_hazard_staggered, "
        f"entry = {pins[TURNOVER_TIER]['entry']:g} = the M1 g pin). 500 runs over 25 "
        "cells, 20 seeds each, mgstd5, visibility reward on; error bars 95% CI = "
        "mean ± 1.96 SE over runs.",
        f"Vanishing-factor pins (read from the archives): M0/M1 (w_v05) "
        f"{pins[M0_TIER]['vanishing']:g}; M1 + turnover_staggered "
        f"{pins[TURNOVER_TIER]['vanishing']:g} (turnover_v2 carries no column — pin "
        f"taken from tables/four_tier_xi_summary.csv); M2 λ = 0/1 (m2_xi) "
        f"{pins['M2 λ = 0 (prestige matching)']['vanishing']:g}. Horizon pins: M0/M1/M2 "
        f"{pins[M0_TIER]['horizon']} steps, turnover arm "
        f"{pins[TURNOVER_TIER]['horizon']} steps — the staggered level is therefore "
        "not horizon-comparable to the other tiers, only its shape is (see F8_v3).",
        "Stage 1 (ξ ≤ 0.6), welfare kink [0.6, 0.8], Stage 2 (ξ ≥ 0.8) are shaded "
        "regions only — no boundary rule crosses a series. Panel B uses the SAME "
        "statistic in every arm (time-mean c_ideal_tm), so no time-mean / final-step "
        f"mixing can distort the hierarchy. The measurement correction moved this statistic by "
        f"{A['bug0006_m1_mgstd5_c_tm_mean_delta']:+.4f} (M1, mgstd5) and "
        f"{A['bug0006_m2_c_tm_mean_delta']:+.4f} (M2) while the M0 control is "
        f"unchanged (max |Δ| = {A['bug0006_m0_c_tm_max_abs_diff']:.1e}); the "
        "the early-draft F6 is therefore superseded. Model-internal, not calibrated "
        "(owner ruling 2026-11-13).",
    ]
    note = fig.text(0.008, 0.006, "\n".join(textwrap.fill(t, width=178)
                                            for t in note_lines),
                    fontsize=HOUSE_STYLE["note_size"], color=PALETTE["gray"],
                    va="bottom")
    handles, labels = ax_a.get_legend_handles_labels()
    place_figure_legend(fig, handles, labels, 3, note)
    audit_text_over_data(fig, (ax_a, ax_b))
    visual_qa.assert_figure_clean(fig, [
        {"ax": ax_a, "metric": PANELS["a"], "expect_lines": len(TIERS),
         "points_per_line": len(XIS), "sidecar": sidecar,
         "filters": {"statistic": "mean"}},
        {"ax": ax_b, "metric": PANELS["b"], "expect_lines": len(TIERS),
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
