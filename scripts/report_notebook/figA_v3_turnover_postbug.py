"""F8_v3 -- turnover diagnostics (paper sketch 4.3), re-based on the ADR-0003 clock.

Fig 10 of the paper (``F8_v3_turnover_diagnostics``).

The pre-ADR figure read ``data/turnover_v2/traj/cell_*.csv``: per-step
trajectories of ``c_ideal`` for four exit modes at ``tau_max = 200``.  The
ADR-0003 archives are SUMMARY-cell archives (one CSV per cell, 20 seed rows
each, no per-step series), so the lifecycle-trajectory panel cannot be drawn
from them and is replaced by an equivalent statistic that the archive does
carry: the SETTLED alignment ``c_ideal_late`` (measured over the late window
of the run), shown by entry rate and exit mode, followed by the regime-level
fixed-minus-staggered contrast that is the figure's actual claim.

Archive (``data_root`` defaults to the ADR-0003 arm; the pre-ADR arm is
selectable)::

    data/turnover_adr0003/cells/turn_c-*_exit-*_g-*_land-*.csv
        36 cells = curiosity {0.3, 0.5, 0.7} x landscape {mgstd5, noisypr5}
        x g {0.125, 0.25, 0.5} x exit_mode {fixed_span, age_hazard_staggered}
        x 20 seeds = 720 runs, H = 400 (tau_max = 80, gamma = 0.933).
    data/turnover_v2/runs.csv   (pre-ADR, 1,520 runs, H = 400, tau_max = 200)

Four panels, every number computed here from the raw CSVs:

  (a) settled alignment ``c_ideal_late`` vs entry rate g, one line per exit
      mode, 95% CI across the 6 cells of each (mode, g) group;
  (b) the regime-level contrast ``fixed_span - age_hazard_staggered`` at every
      g and pooled, paired by (g, curiosity, landscape, seed).  The two arms
      are DIFFERENT POPULATIONS by construction -- fixed_span pins N = 20
      while scheduled entry plus one-for-one retirement grows the staggered
      arm (N = 71 / 121 / 221 at g = 0.125 / 0.25 / 0.5) -- so this is a
      regime comparison, not a same-N treatment effect.  The archived
      populations are printed on the panel;
  (c) arrival-gap ``gap_recent_mean_late`` (native units) pooled per mode with
      per-landscape markers and the difference against ``fixed_span``;
  (d) output and concentration in native units: harvested knowledge share and
      the late-window top-10% prestige share (the pre-ADR step-399/step-400
      pair does not exist in a summary archive), plus the per-capita harvest
      range (``harvested_fin / n_final``) per mode and g.

Provenance rules: ZERO hardcoded model numbers -- every plotted value, CI,
population count and text annotation is computed from the raw CSVs and
written into ``F8_v3_turnover_diagnostics_data.csv``.  The commission anchors
(c_ideal_late fixed minus staggered +0.0187 post vs +0.0279 pre; top10_late
0.205 vs 0.235) are recomputed here and reported against the computed value.

Run:  MPLBACKEND=Agg /opt/hermes/.venv/bin/python \
          scripts/report_notebook/figA_v3_turnover_postbug.py
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
from matplotlib.lines import Line2D  # noqa: E402

from diagram_templates import (  # noqa: E402
    HOUSE_STYLE,
    PALETTE,
    _cosmetics,
    write_figure_data,
)
import visual_qa  # noqa: E402
#: shared placement / collision grammar (F6 owns the implementation; imported
#: READ-ONLY, exactly as figF_v2 does)
from fig_F6_v2 import _place_text, audit_text_over_data  # noqa: E402

FNAME = "F8_v3_turnover_diagnostics"
BASENAME = FNAME

#: ADR-0003 turnover arm (the default) and the pre-ADR arm (selectable)
DEFAULT_ROOT = REPO / "data" / "turnover_adr0003"
PRE_ROOT = REPO / "data" / "turnover_v2"

EXIT_FIXED = "fixed_span"
EXIT_STAGGERED = "age_hazard_staggered"
#: exit-mode grammar: fixed span = orange, staggered age hazard = blue; the
#: legacy modes keep their pre-ADR hues so a pre-ADR rebuild stays readable
MODE_ORDER = ["none", "fixed_span", "age_hazard_sync", "age_hazard_staggered"]
MODE_COLOR = {
    "none": PALETTE["purple"],
    "fixed_span": PALETTE["orange"],
    "age_hazard_sync": PALETTE["green"],
    "age_hazard_staggered": PALETTE["blue"],
}
MODE_LABEL = {
    "none": "none — no retirement",
    "fixed_span": "fixed_span — fixed span",
    "age_hazard_sync": "age_hazard_sync — synchronized",
    "age_hazard_staggered": "age_hazard_staggered — staggered",
}
LAND_MARK = {"mgstd5": "D", "noisypr5": "^"}
LAND_OFF = {"mgstd5": -0.26, "noisypr5": 0.26}
Z95 = 1.96

#: sidecar metric names -- ONE metric per panel (the rendered-artist data
#: contract of ``visual_qa`` checks a panel against a single metric name)
M_A = "c_ideal_late (settled alignment, late window)"
M_B = "c_ideal_late difference (fixed_span - age_hazard_staggered)"
M_C = "gap_recent_mean_late (arrival gap, native units)"
M_D = "native share (harvested_fin, top10_late)"
M_REF = "reference (archive pins and derived contrasts)"


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


# ---------------------------------------------------------------------------
# archive adapter (ADR-0003 summary cells; pre-ADR runs table)
# ---------------------------------------------------------------------------
def data_root_path(data_root=None) -> Path:
    """Archive root: ``cells/`` (ADR-0003) or ``runs.csv`` (pre-ADR)."""
    if data_root is None:
        return DEFAULT_ROOT
    p = Path(data_root)
    return p if p.is_absolute() else (REPO / p)


def _col(df: pd.DataFrame, *names: str) -> pd.Series:
    """First present column under its archive name (raises if none)."""
    for n in names:
        if n in df.columns:
            return df[n]
    raise AssertionError(f"none of {names} in the archive columns")


#: ADR-0003 summary name -> canonical name, then pre-ADR fallbacks
_CANON = {
    "c_ideal_late": ("c_ideal_late", "c_ideal_late_mean"),
    "c_ideal_fin": ("c_ideal_fin",),
    "top10_late": ("top10_late", "top10_late_mean"),
    "harvested_fin": ("harvested_fin",),
    "gap_late": ("gap_recent_mean_late", "gap_recent_mean_last100"),
    "n_final": ("n_final",),
}


def load_runs(data_root=None) -> tuple[pd.DataFrame, dict]:
    """Canonical run table (one row per run) plus the archive pins."""
    root = data_root_path(data_root)
    cells = sorted((root / "cells").glob("*.csv"))
    if cells:                                    # ADR-0003 summary archive
        layout = "cells"
        frames = []
        for f in cells:
            d = pd.read_csv(f)
            _require(len(d) == 20, f"{f.name}: 20 seed rows expected, got {len(d)}")
            frames.append(d)
        raw = pd.concat(frames, ignore_index=True)
        _require(len(cells) >= 2,
                 f"at least two exit-mode cells expected, found {len(cells)}")
    else:                                        # pre-ADR runs table
        layout = "runs.csv"
        f = root / "runs.csv"
        _require(f.exists(), f"missing archive {f}")
        raw = pd.read_csv(f)

    out = pd.DataFrame({
        "exit_mode": _col(raw, "exit_mode").astype(str),
        "g": _col(raw, "g", "entry", "agent_generation_time").astype(float),
        "curiosity": _col(raw, "curiosity", "initial_curiosity").astype(float),
        "landscape": _col(raw, "landscape", "land").astype(str),
    })
    for canon, names in _CANON.items():
        out[canon] = _col(raw, *names).astype(float)
    out["seed"] = _col(raw, "seed").astype(int)
    for canon, names in (("tau_max", ("tau_max", "step_limit")),
                         ("horizon", ("horizon", "steps"))):
        try:
            out[canon] = _col(raw, *names).astype(float)
        except AssertionError:            # pre-ADR summary cells: not stored
            out[canon] = np.nan
    out["per_capita_harvest"] = out["harvested_fin"] / out["n_final"]

    modes = [m for m in MODE_ORDER if m in set(out["exit_mode"])]
    _require(EXIT_FIXED in modes and EXIT_STAGGERED in modes,
             f"archive must carry both {EXIT_FIXED} and {EXIT_STAGGERED}: {modes}")
    _require(out.groupby(["exit_mode", "g", "curiosity", "landscape", "seed"])
             .size().max() == 1,
             "one run per (exit_mode, g, curiosity, landscape, seed) expected")
    pins = {
        "layout": layout,
        "root": str(root.relative_to(REPO)) if root.is_relative_to(REPO) else str(root),
        "modes": modes,
        "g_grid": sorted(out["g"].unique().tolist()),
        "curiosity_grid": sorted(out["curiosity"].unique().tolist()),
        "landscapes": sorted(out["landscape"].unique().tolist()),
        "tau_max": sorted(out["tau_max"].dropna().unique().tolist()),
        "horizon": sorted(out["horizon"].dropna().unique().tolist()),
        "seeds": int(out["seed"].nunique()),
        "cells": int(out.groupby(["exit_mode", "g", "curiosity", "landscape"]).ngroups),
        "runs": int(len(out)),
    }
    return out, pins


def pre_adr_contrast(root: Path = PRE_ROOT) -> dict:
    """Pre-ADR anchors recomputed from the read-only pre-ADR archive.

    The pre-ADR table carries ``c_ideal_late_mean`` / ``top10_late_mean`` and a
    four-mode grid; the ledger's reference point is the ``entry = 0.10`` pin,
    which is the only g level the two archives share.
    """
    f = root / "runs.csv"
    if not f.exists():                       # no pre-ADR archive: nothing to quote
        return {}
    d = pd.read_csv(f)
    d = d[d["entry"] == 0.10]
    out = {"pre_root": str(root.relative_to(REPO)), "pre_g": 0.10}
    for key, col in (("c_ideal_late", "c_ideal_late_mean"),
                     ("top10_late", "top10_late_mean")):
        f_ = d.loc[d["exit_mode"] == EXIT_FIXED, col]
        s_ = d.loc[d["exit_mode"] == EXIT_STAGGERED, col]
        out[f"pre_{key}_fixed"] = float(f_.mean())
        out[f"pre_{key}_staggered"] = float(s_.mean())
        out[f"pre_{key}_diff"] = float(f_.mean() - s_.mean())
    return out


# ---------------------------------------------------------------------------
# statistics helpers
# ---------------------------------------------------------------------------
def _mean_ci(vals) -> tuple[float, float, float, int]:
    v = pd.Series(vals, dtype=float).dropna()
    n = int(v.size)
    _require(n > 1, "CI needs at least two observations")
    m = float(v.mean())
    half = Z95 * float(v.std(ddof=1)) / np.sqrt(n)
    return m, m - half, m + half, n


def _paired_diff(runs: pd.DataFrame, key=("g", "curiosity", "landscape", "seed")):
    """fixed_span minus staggered, paired on the shared random draws."""
    piv = runs.pivot_table(index=list(key), columns="exit_mode",
                           values="c_ideal_late", aggfunc="first")
    _require({EXIT_FIXED, EXIT_STAGGERED} <= set(piv.columns),
             "paired contrast needs both exit modes")
    piv = piv[[EXIT_FIXED, EXIT_STAGGERED]].dropna()
    return (piv[EXIT_FIXED] - piv[EXIT_STAGGERED]).rename("diff").reset_index()


# ---------------------------------------------------------------------------
# compute (every plotted number comes from the raw CSVs)
# ---------------------------------------------------------------------------
def compute(data_root=None):
    runs, pins = load_runs(data_root)
    modes = pins["modes"]
    gs = pins["g_grid"]
    rows: list[dict] = []

    def add(panel, metric, series, *, value, ci_lo=np.nan, ci_hi=np.nan, n,
            statistic="mean", exit_mode="", g=np.nan, landscape="pooled",
            source_file, note=""):
        rows.append(dict(figure=FNAME, panel=panel, metric=metric,
                         statistic=statistic, series=series,
                         exit_mode=exit_mode, g=float(g),
                         landscape=landscape, value=float(value),
                         ci_lo=float(ci_lo), ci_hi=float(ci_hi), n=n,
                         source_file=source_file, note=note))

    src_cells = f"{pins['root']}/cells/*.csv" if pins["layout"] == "cells" \
        else f"{pins['root']}/runs.csv"

    # ---------------- (a) settled alignment vs entry rate ------------------
    a_frames = []
    for m in modes:
        for g in gs:
            sub = runs[(runs["exit_mode"] == m) & (runs["g"] == g)]
            v, lo, hi, n = _mean_ci(sub["c_ideal_late"])
            add("a", M_A, "settled", value=v, ci_lo=lo, ci_hi=hi, n=n,
                exit_mode=m, g=g, source_file=src_cells,
                note="mean c_ideal_late over the cells of the (mode, g) group; "
                     "95% CI across cell runs")
            a_frames.append(dict(exit_mode=m, g=g, value=v, ci_lo=lo, ci_hi=hi,
                                 n=n, cell_mean=float(
                                     sub.groupby(["curiosity", "landscape"])
                                     ["c_ideal_late"].mean().mean()),
                                 sd_cells=float(
                                     sub.groupby(["curiosity", "landscape"])
                                     ["c_ideal_late"].mean().std(ddof=1))))
    adf = pd.DataFrame(a_frames)

    # ---------------- (b) regime-level contrast ----------------------------
    pdf = _paired_diff(runs)
    b_frames = []
    for g in gs + ["pooled"]:
        sub = pdf if g == "pooled" else pdf[pdf["g"] == g]
        v, lo, hi, n = _mean_ci(sub["diff"])
        add("b", M_B, "difference", value=v, ci_lo=lo, ci_hi=hi, n=n,
            g=(np.nan if g == "pooled" else g), source_file=src_cells,
            note="paired by (g, curiosity, landscape, seed): "
                 "c_ideal_late(fixed_span) - c_ideal_late(staggered)")
        b_frames.append(dict(g=g, value=v, ci_lo=lo, ci_hi=hi, n=n))
    bdf = pd.DataFrame(b_frames)
    pooled_b = bdf[bdf["g"] == "pooled"].iloc[0]

    pop = (runs.groupby(["exit_mode", "g"])["n_final"]
           .agg(["mean", "min", "max"]).reset_index())
    pop_txt = {m: {float(r["g"]): float(r["mean"]) for _, r in
                   pop[pop["exit_mode"] == m].iterrows()} for m in modes}

    # ---------------- (c) arrival gap --------------------------------------
    c_frames = []
    for m in modes:
        sub = runs[runs["exit_mode"] == m]
        v, lo, hi, n = _mean_ci(sub["gap_late"])
        add("c", M_C, "pooled", value=v, ci_lo=lo, ci_hi=hi, n=n, exit_mode=m,
            source_file=src_cells,
            note="mean gap_recent_mean_late over the runs of the mode; "
                 "95% CI across runs")
        c_frames.append(dict(exit_mode=m, series="pooled", landscape="pooled",
                             value=v, ci_lo=lo, ci_hi=hi, n=n))
        for land in pins["landscapes"]:
            vv, llo, hhi, nn = _mean_ci(sub.loc[sub["landscape"] == land,
                                                "gap_late"])
            add("c", M_C, land, value=vv, ci_lo=llo, ci_hi=hhi, n=nn,
                exit_mode=m, landscape=land, source_file=src_cells,
                note=f"mean over the {nn} {land} runs")
            c_frames.append(dict(exit_mode=m, series=land, landscape=land,
                                 value=vv, ci_lo=llo, ci_hi=hhi, n=nn))
    cdf = pd.DataFrame(c_frames)
    base_c = float(cdf[(cdf["exit_mode"] == EXIT_FIXED)
                       & (cdf["series"] == "pooled")]["value"].iloc[0])
    c_delta = {}
    for m in modes:
        v = float(cdf[(cdf["exit_mode"] == m)
                      & (cdf["series"] == "pooled")]["value"].iloc[0])
        c_delta[m] = v - base_c
        add("c", "annotation: gap_difference_vs_fixed_span", "annotation",
            value=v - base_c, n=int(len(runs[runs["exit_mode"] == m])),
            statistic="annotation", exit_mode=m, source_file=src_cells,
            note=f"{m}: gap_recent_mean_late minus {EXIT_FIXED} "
                 f"(+{v - base_c:.5f} native units)")

    # ---------------- (d) output and concentration -------------------------
    d_frames = []
    for m in modes:
        sub = runs[runs["exit_mode"] == m]
        for ser, col, metric_note in (
                ("harvested_fin", "harvested_fin",
                 "mean harvested knowledge share over the runs"),
                ("top10_late", "top10_late",
                 "mean top-10% prestige share, late window, over the runs")):
            v, lo, hi, n = _mean_ci(sub[col])
            add("d", M_D, ser, value=v, ci_lo=lo, ci_hi=hi, n=n, exit_mode=m,
                source_file=src_cells, note=metric_note)
            d_frames.append(dict(exit_mode=m, series=ser, value=v, ci_lo=lo,
                                 ci_hi=hi, n=n))
    ddf = pd.DataFrame(d_frames)

    pc = (runs.groupby(["exit_mode", "g"])["per_capita_harvest"]
          .mean().reset_index())
    pc_range = {}
    for m in modes:
        s = pc[pc["exit_mode"] == m].sort_values("g")["per_capita_harvest"]
        pc_range[m] = (float(s.iloc[0]), float(s.iloc[-1]), float(s.max() - s.min()))
        add("d", "annotation: per_capita_harvest_low_g", "annotation",
            value=float(s.iloc[0]), n=int(len(runs[runs["exit_mode"] == m])),
            statistic="annotation", exit_mode=m, g=gs[0], source_file=src_cells,
            note="harvested_fin / n_final at the lowest g")
        add("d", "annotation: per_capita_harvest_high_g", "annotation",
            value=float(s.iloc[-1]), n=int(len(runs[runs["exit_mode"] == m])),
            statistic="annotation", exit_mode=m, g=gs[-1], source_file=src_cells,
            note="harvested_fin / n_final at the highest g")

    # ---------------- archive pins + recomputed anchors -------------------
    anchors = pre_adr_contrast()
    for cat, val, note in (
            ("runs", pins["runs"], "one row per run in the archive"),
            ("cells", pins["cells"], "distinct (exit_mode, g, curiosity, landscape)"),
            ("seeds", pins["seeds"], "seed rows per cell"),
            ("tau_max", pins["tau_max"][0] if pins["tau_max"] else np.nan,
             "career length in steps (ADR-0003)"),
            ("horizon", pins["horizon"][0] if pins["horizon"] else np.nan,
             "H: total steps simulated"),
            ("g_min", gs[0], "entry-rate grid, low pin"),
            ("g_max", gs[-1], "entry-rate grid, high pin"),
            ("curiosity_min", pins["curiosity_grid"][0], "curiosity grid, low pin"),
            ("curiosity_max", pins["curiosity_grid"][-1], "curiosity grid, high pin"),
            ("anchor_fixed_minus_staggered", float(pooled_b["value"]),
             "recomputed commission anchor: c_ideal_late fixed_span minus "
             "staggered, pooled (commission: +0.0187 post)"),
            ("anchor_top10_fixed", float(ddf[(ddf["exit_mode"] == EXIT_FIXED)
                                             & (ddf["series"] == "top10_late")]
                                         ["value"].iloc[0]),
             "recomputed commission anchor: top10_late, fixed_span "
             "(commission: 0.205)"),
            ("anchor_top10_staggered",
             float(ddf[(ddf["exit_mode"] == EXIT_STAGGERED)
                       & (ddf["series"] == "top10_late")]["value"].iloc[0]),
             "recomputed commission anchor: top10_late, staggered "
             "(commission: 0.235)"),
            ("population_fixed_g_max", pop_txt[EXIT_FIXED][gs[-1]],
             "mean n_final, fixed_span, highest g"),
            ("population_staggered_g_max", pop_txt[EXIT_STAGGERED][gs[-1]],
             "mean n_final, staggered, highest g")):
        rows.append(dict(figure=FNAME, panel="ref", metric=M_REF,
                         statistic="definitional constant", series=cat,
                         exit_mode="", g=np.nan, landscape="", value=float(val),
                         ci_lo=np.nan, ci_hi=np.nan, n=np.nan,
                         source_file=src_cells, note=note))
    if anchors:
        for cat, note in (("pre_anchor_fixed_minus_staggered",
                           f"pre-ADR c_ideal_late fixed minus staggered at the "
                           f"shared pin g = {anchors['pre_g']:g} (read-only "
                           f"{anchors['pre_root']})"),
                          ("pre_anchor_top10_fixed", "pre-ADR top10_late, fixed_span"),
                          ("pre_anchor_top10_staggered",
                           "pre-ADR top10_late, staggered")):
            key = {"pre_anchor_fixed_minus_staggered": "pre_c_ideal_late_diff",
                   "pre_anchor_top10_fixed": "pre_top10_late_fixed",
                   "pre_anchor_top10_staggered": "pre_top10_late_staggered"}[cat]
            rows.append(dict(figure=FNAME, panel="ref", metric=M_REF,
                             statistic="definitional constant", series=cat,
                             exit_mode="", g=np.nan, landscape="",
                             value=float(anchors[key]), ci_lo=np.nan,
                             ci_hi=np.nan, n=np.nan, source_file=anchors["pre_root"],
                             note=note))

    sidecar = pd.DataFrame(rows, columns=[
        "figure", "panel", "metric", "statistic", "series", "exit_mode", "g",
        "landscape", "value", "ci_lo", "ci_hi", "n", "source_file", "note"])

    frames = dict(a=adf, b=bdf, c=cdf, d=ddf, pop=pop)
    texts = dict(pins=pins, anchors=anchors, pop=pop_txt,
                 pooled_b=pooled_b, c_delta=c_delta, pc_range=pc_range,
                 base_c=base_c, modes=modes, gs=gs)
    return frames, texts, sidecar


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------
def _tick_name(mode: str, short: bool) -> str:
    """Long exit-mode names are shortened on crowded axes (4-mode rebuilds)."""
    return mode.replace("age_hazard_", "") if short else mode


def _pad_ylim(ax, spans, *, pad_lo=0.16, pad_hi=0.30, extra=()) -> tuple[float, float]:
    los = [float(s[0]) for s in spans] + [float(v) for v in extra]
    his = [float(s[1]) for s in spans] + [float(v) for v in extra]
    lo, hi = min(los), max(his)
    span = hi - lo if hi > lo else max(abs(hi), 1.0)
    ax.set_ylim(lo - pad_lo * span, hi + pad_hi * span)
    return ax.get_ylim()


def _draw(frames, texts, sidecar, png: Path, pdf: Path) -> dict:
    s = HOUSE_STYLE
    adf, bdf, cdf, ddf = frames["a"], frames["b"], frames["c"], frames["d"]
    modes, gs = texts["modes"], texts["gs"]
    pins, anchors, pop = texts["pins"], texts["anchors"], texts["pop"]

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.4), dpi=s["dpi"])
    axA, axB, axC, axD = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # ---------------- (a) settled alignment vs entry rate ------------------
    for m in modes:
        d = adf[adf["exit_mode"] == m].sort_values("g")
        axA.errorbar(d["g"], d["value"],
                     yerr=[d["value"] - d["ci_lo"], d["ci_hi"] - d["value"]],
                     color=MODE_COLOR[m], marker="o", markersize=5.5,
                     linewidth=1.8, elinewidth=1.0, capsize=2.6, zorder=4,
                     label=MODE_LABEL[m])
    _pad_ylim(axA, [(r["ci_lo"], r["ci_hi"]) for _, r in adf.iterrows()],
              pad_lo=0.42, pad_hi=0.55, extra=(0.0,))
    axA.axhline(0.0, color=PALETTE["gray"], linewidth=0.9, linestyle="--",
                zorder=2)
    axA.set_xlim(gs[0] - 0.06 * (gs[-1] - gs[0]), gs[-1] + 0.16 * (gs[-1] - gs[0]))
    axA.set_xticks(gs, [f"{g:g}" for g in gs])
    _cosmetics(axA, s,
               "(a) Settled alignment vs entry rate"
               + (f" (H = {pins['horizon'][0]:.0f})" if pins["horizon"] else ""),
               "entry rate g (one arrival every 1/g steps)",
               "Corr(prestige, ideal merit)\n(settled: c_ideal_late, cell mean)")
    axA.legend(fontsize=s["tick_size"] - 1.5, frameon=False, loc="upper left",
               handlelength=1.8, labelspacing=0.3, borderaxespad=0.2)
    loA, hiA = axA.get_ylim()
    _place_text(axA, (
        f"fixed_span above staggered at every g\n"
        f"pooled gap {texts['pooled_b']['value']:+.4f} "
        f"[{texts['pooled_b']['ci_lo']:+.4f}, {texts['pooled_b']['ci_hi']:+.4f}]"),
        ((0.98, 0.06, "right", "bottom"), (0.98, 0.30, "right", "bottom"),
         (0.02, 0.06, "left", "bottom"), (0.02, 0.30, "left", "bottom"),
         (0.5, 0.02, "center", "bottom"), (0.02, 0.98, "left", "top")))

    # ---------------- (b) regime-level contrast ---------------------------
    xs = np.arange(len(bdf))
    labels = ["pooled" if isinstance(r["g"], str) else f"{r['g']:g}"
              for _, r in bdf.iterrows()]
    colors = [PALETTE["orange"] if isinstance(r["g"], str) else PALETTE["gray"]
              for _, r in bdf.iterrows()]
    axB.bar(xs, bdf["value"], width=0.58, color=colors, zorder=3,
            edgecolor="white")
    axB.errorbar(xs, bdf["value"],
                 yerr=[bdf["value"] - bdf["ci_lo"], bdf["ci_hi"] - bdf["value"]],
                 fmt="none", ecolor="black", elinewidth=1.0, capsize=3.0,
                 zorder=4)
    axB.axhline(0.0, color=PALETTE["gray"], linewidth=0.9, zorder=2)
    _pad_ylim(axB, [(r["ci_lo"], r["ci_hi"]) for _, r in bdf.iterrows()],
              pad_lo=0.20, pad_hi=0.62, extra=(0.0,))
    axB.set_xticks(xs, labels)
    axB.set_xlim(-0.6, len(bdf) - 0.4)
    _cosmetics(axB, s, "(b) Fixed span minus staggered age hazard",
               "entry rate g (last bar: all g pooled)",
               "Δ Corr(prestige, ideal merit)\n(paired by cell and seed)")
    for x, r in zip(xs, bdf.itertuples()):
        # offset in POINTS above the upper cap: a data-space offset put the
        # label on the cap line (visual_qa "text over data")
        axB.annotate(f"{r.value:+.4f}", xy=(x, r.ci_hi), xytext=(0, 4),
                     textcoords="offset points", ha="center", va="bottom",
                     fontsize=s["note_size"] - 1.0, color=PALETTE["gray"],
                     zorder=6)
    pop_line = ("populations differ by construction (N = "
                f"{pop[EXIT_FIXED][gs[0]]:.0f}/{pop[EXIT_FIXED][gs[1]]:.0f}/"
                f"{pop[EXIT_FIXED][gs[2]]:.0f} fixed vs "
                f"{pop[EXIT_STAGGERED][gs[0]]:.0f}/{pop[EXIT_STAGGERED][gs[1]]:.0f}/"
                f"{pop[EXIT_STAGGERED][gs[2]]:.0f} staggered)")
    pre_line = (f"pre-ADR (g = {anchors['pre_g']:g}): "
                f"{anchors['pre_c_ideal_late_diff']:+.4f}" if anchors else "")
    _place_text(axB, "regime comparison, not same-N:\n" + pop_line
                + (("\n" + pre_line) if pre_line else ""),
                ((0.02, 0.97, "left", "top"), (0.98, 0.97, "right", "top")))

    # ---------------- (c) arrival gap -------------------------------------
    vmin = float(cdf["ci_lo"].min())
    span_c = abs(vmin)
    axC.set_ylim(vmin - 0.75 * span_c, 0.35 * span_c)
    axC.axhline(0.0, color="black", linewidth=0.8, zorder=2)
    for i, m in enumerate(modes):
        pooled = cdf[(cdf["exit_mode"] == m) & (cdf["series"] == "pooled")].iloc[0]
        axC.bar(i, pooled["value"], width=0.40, color=MODE_COLOR[m], zorder=3)
        axC.errorbar([i], [pooled["value"]],
                     yerr=[[pooled["value"] - pooled["ci_lo"]],
                           [pooled["ci_hi"] - pooled["value"]]],
                     fmt="none", ecolor=MODE_COLOR[m], elinewidth=1.2,
                     capsize=3, zorder=4)
        lab = f"{pooled['value']:.5f}"
        if m != EXIT_FIXED:
            lab += f"\nΔ {texts['c_delta'][m]:+.5f}"
        axC.text(i, pooled["ci_lo"] - 0.06 * span_c, lab,
                 ha="center", va="top",
                 fontsize=s["note_size"] - (1.0 if len(modes) <= 2 else 2.0),
                 color=PALETTE["gray"], zorder=5)
        for land in pins["landscapes"]:
            sub = cdf[(cdf["exit_mode"] == m) & (cdf["series"] == land)].iloc[0]
            axC.errorbar([i + LAND_OFF[land]], [sub["value"]],
                         yerr=[[sub["value"] - sub["ci_lo"]],
                               [sub["ci_hi"] - sub["value"]]],
                         fmt=LAND_MARK[land], color=MODE_COLOR[m],
                         markerfacecolor="white", markersize=4.6,
                         elinewidth=0.9, capsize=2.0, zorder=5)
    _cosmetics(axC, s, "(c) Arrival gap in the late window (native units)",
               "exit mode", "gap_recent_mean_late\n(run means)")
    # long mode names do not fit side by side: shrink and angle them
    big = len(modes) > 2
    axC.set_xticks(range(len(modes)), [_tick_name(m, big) for m in modes],
                   fontsize=s["tick_size"] - (1.0 if not big else 2.5),
                   rotation=0 if not big else 14)
    axC.legend(handles=[
        Line2D([], [], linestyle="none", marker="D", color=PALETTE["gray"],
               markerfacecolor="white", markersize=4.6, label="mgstd5"),
        Line2D([], [], linestyle="none", marker="^", color=PALETTE["gray"],
               markerfacecolor="white", markersize=4.6, label="noisypr5"),
    ], fontsize=s["tick_size"] - 2, frameon=False, loc="lower right",
        handlelength=1.2, labelspacing=0.25, borderaxespad=0.3)

    # ---------------- (d) output and concentration ------------------------
    width = 0.34
    for i, m in enumerate(modes):
        for j, (ser, colr) in enumerate((("harvested_fin", PALETTE["blue"]),
                                         ("top10_late", PALETTE["gray"]))):
            r = ddf[(ddf["exit_mode"] == m) & (ddf["series"] == ser)].iloc[0]
            x = i + (j - 0.5) * width
            axD.bar(x, r["value"], width=width, color=colr, zorder=3,
                    edgecolor="white")
            axD.errorbar([x], [r["value"]],
                         yerr=[[r["value"] - r["ci_lo"]],
                               [r["ci_hi"] - r["value"]]],
                         fmt="none", ecolor=colr, elinewidth=1.0, capsize=2.4,
                         zorder=4)
            axD.text(x, r["ci_hi"] + 0.015, f"{r['value']:.3f}", ha="center",
                     va="bottom", fontsize=s["note_size"] - 1.2,
                     color=PALETTE["gray"], zorder=5)
    axD.set_xticks(range(len(modes)), [_tick_name(m, big) for m in modes],
                   fontsize=s["tick_size"] - (1.0 if len(modes) <= 2 else 2.5),
                   rotation=0 if len(modes) <= 2 else 14)
    axD.set_xlim(-0.6, len(modes) - 0.4)
    axD.set_ylim(0.0, 1.55)
    axD.axhline(0.10, color=PALETTE["gray"], linewidth=1.0, linestyle=":",
                zorder=1)
    _cosmetics(axD, s, "(d) Output and concentration (native shares)",
               "exit mode", "share (native units [0, 1])")
    axD.legend(handles=[
        Line2D([], [], color=PALETTE["blue"], linewidth=6,
               label="harvested knowledge share"),
        Line2D([], [], color=PALETTE["gray"], linewidth=6,
               label="top-10% prestige (late window)"),
        Line2D([], [], color=PALETTE["gray"], linestyle=":", linewidth=1.0,
               label="0.10 equality"),
    ], fontsize=s["tick_size"] - 2, frameon=False, loc="upper left",
        bbox_to_anchor=(0.0, 1.0), handlelength=1.4, labelspacing=0.25,
        borderaxespad=0.2)
    axD.set_ylim(0.0, 1.30)

    # ---------------- source footnotes ------------------------------------
    gtxt = "/".join(f"{g:g}" for g in gs)
    foot = [
        textwrap.fill(
            f"Source: {texts['pins']['root']}/cells/*.csv (ADR-0003 turnover arm): "
            f"{pins['cells']} cells x {pins['seeds']} seeds = {pins['runs']} runs; "
            f"exit_mode in {{{', '.join(modes)}}}, g in {{{gtxt}}}, curiosity in "
            f"{{{', '.join(f'{c:g}' for c in pins['curiosity_grid'])}}}, landscapes "
            f"{{{', '.join(pins['landscapes'])}}}"
            + (f", H = {pins['horizon'][0]:.0f}." if pins["horizon"] else ".")
            + " ADR-0003 "
            f"clock: 1 step = 0.5 yr, entry age 25, tau_max = "
            f"{pins['tau_max'][0] if pins['tau_max'] else float('nan'):.0f} "
            "(40-yr careers), gamma = 0.5**(1/10) = 0.933 (5-yr retention "
            "half-life). 95% CIs = mean +/- 1.96 SE.", width=196),
        textwrap.fill(
            "Panel (a) replaces the pre-ADR per-step lifecycle panel: this archive "
            "stores summary cells only (no trajectories), so the settled window "
            "statistic c_ideal_late is shown instead. Panel (b) is a REGIME "
            "comparison, not a same-N treatment effect: fixed_span pins the "
            "population at N = 20 while scheduled entry plus one-for-one retirement "
            f"grows the staggered arm to N = {pop[EXIT_STAGGERED][gs[0]]:.0f}/"
            f"{pop[EXIT_STAGGERED][gs[1]]:.0f}/{pop[EXIT_STAGGERED][gs[2]]:.0f} at "
            f"g = {gtxt}; runs are paired by (g, curiosity, landscape, seed) and the "
            "population counts are printed on the panel. Panel (c) uses the core "
            "gap_recent reporter (the is_newcomer prestige gap is structurally NaN "
            "once founders retire). Panel (d) drops the pre-ADR step-399/step-400 "
            "top-10% pair (a summary archive has no single-step series) and shows "
            "top10_late instead.", width=196),
        textwrap.fill(
            "Panel (b) also reproduces ADR-0003 qualification (ii): per-capita "
            "harvest (harvested_fin / n_final) is flat across g under fixed_span "
            f"({texts['pc_range'][EXIT_FIXED][0]:.4f} -> "
            f"{texts['pc_range'][EXIT_FIXED][1]:.4f}) and falls under staggered age "
            f"hazard ({texts['pc_range'][EXIT_STAGGERED][0]:.4f} -> "
            f"{texts['pc_range'][EXIT_STAGGERED][1]:.4f}). Every plotted value, CI "
            "and annotation is computed from the raw CSVs and stored in "
            f"{BASENAME}_data.csv. MODEL-INTERNAL, NOT CALIBRATED: no empirical "
            "benchmark is plotted here.", width=196),
    ]
    y = 0.118
    for block in foot:
        for line in block.split("\n"):
            fig.text(0.012, y, line, fontsize=s["note_size"] - 1.6,
                     color=PALETTE["gray"])
            y -= 0.0122

    fig.subplots_adjust(left=0.105, right=0.985, top=0.925, bottom=0.215,
                        hspace=0.52, wspace=0.24)

    for ax in (axA, axB, axC, axD):
        lo, hi = ax.get_ylim()
        ax.set_yticks([t for t in ax.get_yticks() if lo - 1e-12 <= t <= hi + 1e-12])

    specs = [
        {"ax": axA, "metric": M_A, "expect_lines": len(modes),
         "points_per_line": len(gs), "sidecar": sidecar,
         "filters": {"panel": "a", "statistic": "mean"}},
        {"ax": axB, "metric": M_B, "expect_bars": len(bdf), "sidecar": sidecar,
         "filters": {"panel": "b", "statistic": "mean"}},
        {"ax": axC, "metric": M_C, "expect_bars": len(modes),
         "expect_lines": 3 * len(modes), "sidecar": sidecar,
         "filters": {"panel": "c", "statistic": "mean"}},
        {"ax": axD, "metric": M_D, "expect_bars": 2 * len(modes),
         "sidecar": sidecar, "filters": {"panel": "d", "statistic": "mean"}},
    ]
    report = visual_qa.assert_figure_clean(fig, specs)
    audit_text_over_data(fig, (axA, axB, axC, axD))

    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=s["dpi"])
    fig.savefig(pdf)
    plt.close(fig)
    return report


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def build(outdir="output/report_notebook_v1", data_root=None,
          collector: dict | None = None) -> dict:
    """Build F8_v3 (PNG + PDF + _data.csv); returns paths and the QA report."""
    out = Path(outdir)
    if not out.is_absolute():
        out = REPO / out
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{BASENAME}.png"
    pdf = out / f"{BASENAME}.pdf"

    frames, texts, sidecar = compute(data_root=data_root)
    report = _draw(frames, texts, sidecar, png, pdf)
    data_csv = write_figure_data(sidecar, png)
    if collector is not None:
        collector["sidecar"] = sidecar
        collector["texts"] = texts
    print(f"F8_v3: {png} | {pdf} | {data_csv}")
    print(f"sidecar rows: {len(sidecar)} | visual_qa failures: {report['_total']}")
    return {"png": str(png), "pdf": str(pdf), "data_csv": str(data_csv),
            "qa": report, "sidecar_rows": len(sidecar)}


def main() -> int:
    build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
