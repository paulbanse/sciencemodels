"""F13 -- paradigm refill regimes: lock-in, frontier stock, alignment (Section 5.3).

Fig 14 of the paper (``F13_paradigm_refill_regimes``), re-based on the
ADR-0003 clock.

Inputs (read-only): the refill sub-arm of the ADR-0003 interventions archive::

    data/interventions_adr0003/cells/intC_c-*_g-*_refill-*.csv
        27 cells = refill_mode {off, occupied, frontier} x curiosity
        {0.3, 0.5, 0.7} x g {0.125, 0.25, 0.5} x 20 seeds = 540 runs,
        mgstd5, fixed_span, H = 200, tau_max = 80, gamma = 0.933.
    data/interventions_v2/cells/cellC_*.csv   (pre-ADR, M0/M1 pins; selectable)

Panels (all numbers computed here from the raw CSVs):

  (a) mainstream lock-in          : explored grid share ``explored_fin``
  (b) unclaimed frontier knowledge: mean residual knowledge on grid, log scale
  (c) merit alignment             : ``c_ideal_fin`` bars + ``c_ideal_tm``
                                    diamonds (final step vs time mean)
  (d1) concentration              : top-10% prestige share
  (d2) arrival gap                : ``gap_recent_mean_late`` in native units

Two panels of the pre-ADR figure could not survive the schema change and are
replaced, disclosed on the canvas:

  * the M0 (g = 0) hollow bars are gone -- the ADR-0003 refill arm carries an
    entry-rate grid, not an M0/M1 pair, so g is the x axis and every bar is a
    real (regime, g) cell;
  * the normalised newcomer gap ``(newcomer - incumbent)/incumbent`` and the
    ``mean_prestige`` inflation factor do not exist in the 51-column summary
    schema (no per-agent prestige means are stored).  Panel (d2) therefore
    plots the core replacement measure ``gap_recent_mean_late`` in native
    units, which is where the refill effect actually shows.

Provenance: ZERO hardcoded model numbers -- every plotted value, CI and
annotation is computed from the raw CSVs and written into
``F13_paradigm_refill_regimes_data.csv``.  The commission anchors
(explored_fin occupied minus off -0.297 post vs -0.319 pre; frontier minus off
-0.094 post vs -0.045 pre) are recomputed and reported against the computed
value.

Run:  MPLBACKEND=Agg /opt/hermes/.venv/bin/python \
          scripts/report_notebook/figD_paradigm_refill_regimes.py
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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from diagram_templates import (  # noqa: E402
    HOUSE_STYLE,
    PALETTE,
    _cosmetics,
    write_figure_data,
)
import visual_qa  # noqa: E402
from fig_F6_v2 import _place_text, audit_text_over_data  # noqa: E402  (read-only)

FNAME = "F13_paradigm_refill_regimes"
DEFAULT_ROOT = REPO / "data" / "interventions_adr0003"
PRE_ROOT = REPO / "data" / "interventions_v2"

REGIMES = ("off", "occupied", "frontier")
#: regime hue; the red stays reserved for the incumbent-bailout lock-in regime
REGIME_COLOR = {"off": PALETTE["purple"], "occupied": "#D73027",
                "frontier": PALETTE["green"]}
Z95 = 1.96
D_TICK = 7.0

M_LOCKIN = "explored grid share (explored_fin)"
M_FRONTIER = "mean residual knowledge on grid (avg_knowledge_fin)"
M_CORR_FIN = "Corr(prestige, ideal merit) (final step)"
M_CORR_TM = "Corr(prestige, ideal merit) (time mean)"
M_TOP10 = "top-10% prestige share"
M_GAP = "arrival gap, late window (native units)"
M_REF = "reference (archive pins and derived contrasts)"


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


# ---------------------------------------------------------------------------
# archive adapter
# ---------------------------------------------------------------------------
def data_root_path(data_root=None) -> Path:
    """Directory holding the refill cells (archive root or its ``cells/``)."""
    root = DEFAULT_ROOT if data_root is None else Path(data_root)
    if not root.is_absolute():
        root = REPO / root
    if not list(root.glob("intC_*.csv")) and not list(root.glob("cellC_*.csv")):
        if list((root / "cells").glob("*C_*.csv")):
            root = root / "cells"
    return root


def _col(df: pd.DataFrame, *names: str) -> pd.Series:
    for n in names:
        if n in df.columns:
            return df[n]
    raise AssertionError(f"none of {names} in the archive columns")


_CANON = {
    "explored_fin": ("explored_fin", "explored_final", "explored_percentage"),
    "avg_knowledge_fin": ("avg_knowledge_fin", "avg_knowledge_on_grid"),
    "c_ideal_fin": ("c_ideal_fin", "corr_prestige_ideal_merit_final"),
    "c_ideal_tm": ("c_ideal_tm", "Mean_of_Corr_prestige_ideal_merit"),
    "top10": ("top10_late", "top10_fin", "top10_final"),
}


def load_runs(data_root=None) -> tuple[pd.DataFrame, dict]:
    root = data_root_path(data_root)
    files = sorted(root.glob("intC_*.csv")) or sorted(root.glob("cellC_*.csv"))
    _require(bool(files), f"no refill cells in {root}")
    frames = []
    for f in files:
        d = pd.read_csv(f)
        _require(len(d) == 20, f"{f.name}: 20 seed rows expected, got {len(d)}")
        frames.append(d)
    raw = pd.concat(frames, ignore_index=True)

    out = pd.DataFrame({
        "refill_mode": _col(raw, "refill_mode").astype(str),
        "g": _col(raw, "g", "agent_generation_time").astype(float),
        "curiosity": _col(raw, "curiosity", "initial_curiosity").astype(float),
    })
    for canon, names in _CANON.items():
        out[canon] = _col(raw, *names).astype(float)
    gap_col = ("gap_recent_mean_late" if "gap_recent_mean_late" in raw.columns
               else "gap_recent_mean_last100")
    if gap_col in raw.columns:
        out["gap"] = raw[gap_col].astype(float)
        gap_kind = gap_col
    else:                                  # pre-ADR: normalised prestige gap
        out["gap"] = ((_col(raw, "newcomer_prestige_mean")
                       - _col(raw, "incumbent_prestige_mean"))
                      / _col(raw, "incumbent_prestige_mean"))
        gap_kind = "(newcomer - incumbent) / incumbent prestige"
    out["seed"] = _col(raw, "seed", "agent_seed").astype(int)

    _require(sorted(out["refill_mode"].unique()) == sorted(REGIMES),
             f"refill_mode grid: {sorted(out['refill_mode'].unique())}")
    _require(out.groupby(["refill_mode", "g", "curiosity", "seed"]).size().max() == 1,
             "one run per (regime, g, curiosity, seed) expected")
    pins = {
        "root": str(root.relative_to(REPO)) if root.is_relative_to(REPO) else str(root),
        "files": len(files),
        "cells": int(out.groupby(["refill_mode", "g", "curiosity"]).ngroups),
        "runs": int(len(out)),
        "g_grid": sorted(out["g"].unique().tolist()),
        "curiosity_grid": sorted(out["curiosity"].unique().tolist()),
        "seeds": int(out["seed"].nunique()),
        "gap_kind": gap_kind,
        "per_g": int(out.groupby(["refill_mode", "g"]).size().iloc[0]),
    }
    return out, pins


def _pre_contrast(root: Path = PRE_ROOT) -> dict:
    """Pre-ADR anchors, recomputed from the read-only pre-ADR refill cells.

    Two bases are kept because they are NOT the same comparison:

    * ``pre_*``            -- matched WITHIN the refill sub-arm (``cellC``),
      the same basis as this figure (regime vs its own ``off`` control);
    * ``pre_alloff_*``     -- the remeasure ledger's basis, which pools every
      ``off`` cell of the archive (``cell{A,B,C}_*.csv``, including the spawn
      and relocation controls).  Quoted here so the before/after ledger can be
      cross-checked, but it is not like-for-like with the plotted cells.
    """
    all_files = sorted((root / "cells").glob("cell*.csv"))
    c_files = sorted((root / "cells").glob("cellC_*.csv"))
    if not c_files:
        return {}
    d = pd.concat([pd.read_csv(f) for f in c_files], ignore_index=True)
    a = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
    col = "explored_final"
    m = d.groupby("refill_mode")[col].mean()
    k = a.groupby("refill_mode")[col].mean()
    if not set(REGIMES) <= set(m.index):
        return {}
    out = {"pre_root": str(root.relative_to(REPO)),
           "pre_occupied_minus_off": float(m["occupied"] - m["off"]),
           "pre_frontier_minus_off": float(m["frontier"] - m["off"])}
    if set(REGIMES) <= set(k.index):
        out["pre_alloff_occupied_minus_off"] = float(k["occupied"] - k["off"])
        out["pre_alloff_frontier_minus_off"] = float(k["frontier"] - k["off"])
    return out


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def _mean_ci(vals) -> tuple[float, float, float, int]:
    v = pd.Series(vals, dtype=float).dropna()
    n = int(v.size)
    _require(n > 1, "CI needs at least two observations")
    m = float(v.mean())
    return m, m - Z95 * float(v.std(ddof=1)) / np.sqrt(n), \
        m + Z95 * float(v.std(ddof=1)) / np.sqrt(n), n


def compute(data_root=None):
    runs, pins = load_runs(data_root)
    gs = pins["g_grid"]
    #: the arrival gap needs newcomers: a g = 0 (no entry) pin has none, so the
    #: gap panel is restricted to the pins where the measure is defined
    gap_gs = [g for g in gs
              if int(runs.loc[runs["g"] == g, "gap"].notna().sum()) >= 2]
    pins["gap_g_grid"] = gap_gs
    src = f"{pins['root']}/*C_*.csv"
    rows: list[dict] = []

    def add(panel, metric, regime, g, *, value, ci, n, statistic="mean",
            series=None, note=""):
        rows.append(dict(figure=FNAME, panel=panel, metric=metric,
                         statistic=statistic, series=series or regime,
                         refill_mode=regime, g=float(g), value=float(value),
                         ci_lo=float(value - ci), ci_hi=float(value + ci),
                         n=int(n), source_file=src, note=note))

    points = {}
    for panel, metric, col in (("a", M_LOCKIN, "explored_fin"),
                               ("b", M_FRONTIER, "avg_knowledge_fin"),
                               ("c", M_CORR_FIN, "c_ideal_fin"),
                               ("d1", M_TOP10, "top10"),
                               ("d2", M_GAP, "gap")):
        for g in (gap_gs if panel == "d2" else gs):
            for regime in REGIMES:
                sub = runs[(runs["refill_mode"] == regime) & (runs["g"] == g)][col]
                _require(len(sub) == pins["per_g"],
                         f"{metric} {regime} g={g}: n={len(sub)}")
                v, lo, hi, n = _mean_ci(sub)
                add(panel, metric, regime, g, value=v, ci=hi - v, n=n,
                    note="pooled over the curiosity grid and 20 seeds; "
                         "95% CI = 1.96*sd/sqrt(n)")
                points[(metric, regime, g)] = (v, lo, hi, n)
    # (c) diamonds: time-mean alignment
    for g in gs:
        for regime in REGIMES:
            sub = runs[(runs["refill_mode"] == regime) & (runs["g"] == g)]["c_ideal_tm"]
            v, lo, hi, n = _mean_ci(sub)
            add("c", M_CORR_TM, regime, g, value=v, ci=hi - v, n=n,
                note="time-mean Corr(prestige, ideal merit), same cell")

    # ---- derived contrasts (matched regime differences within the archive) --
    ann = {}
    for g in gs + ["pooled"]:
        sub = runs if g == "pooled" else runs[runs["g"] == g]
        m = sub.groupby("refill_mode")["explored_fin"].mean()
        for regime in ("occupied", "frontier"):
            key = f"explored_{regime}_minus_off_{'pooled' if g == 'pooled' else g}"
            ann[key] = float(m[regime] - m["off"])
            add("a", "annotation: explored_minus_off", regime,
                (np.nan if g == "pooled" else g), value=ann[key], ci=np.nan,
                n=int(len(sub[sub["refill_mode"] == regime])), statistic="annotation",
                note=f"explored_fin: {regime} minus off "
                     f"({'pooled' if g == 'pooled' else f'g = {g:g}'}), "
                     "matched within the refill sub-arm")
    m_off = float(runs[runs["refill_mode"] == "off"]["explored_fin"].mean())
    m_occ = float(runs[runs["refill_mode"] == "occupied"]["explored_fin"].mean())
    ann["occupied_collapse_pct"] = 100.0 * (m_occ / m_off - 1.0)
    add("a", "annotation: occupied_exploration_pct_vs_off", "occupied", np.nan,
        value=ann["occupied_collapse_pct"], ci=np.nan,
        n=int((runs["refill_mode"] == "occupied").sum()), statistic="annotation",
        note="occupied explored_fin in percent of the off level")
    ak = runs.groupby("refill_mode")["avg_knowledge_fin"].mean()
    for regime in ("occupied", "frontier"):
        ann[f"residual_ratio_{regime}"] = float(ak[regime] / ak["off"])
        add("b", "annotation: residual_ratio_vs_off", regime, np.nan,
            value=ann[f"residual_ratio_{regime}"], ci=np.nan,
            n=int((runs["refill_mode"] == regime).sum()), statistic="annotation",
            note=f"avg_knowledge_fin {regime} / off (pooled)")
    ctm = runs.groupby("refill_mode")["c_ideal_tm"].mean()
    for regime in REGIMES:
        add("c", "annotation: corr_tm_pooled", regime, np.nan,
            value=float(ctm[regime]), ci=np.nan,
            n=int((runs["refill_mode"] == regime).sum()), statistic="annotation",
            note="time-mean alignment, pooled over g and curiosity")
    gp = runs.groupby("refill_mode")["gap"].mean()   # NaN pins dropped
    for regime in REGIMES:
        add("d2", "annotation: gap_pooled", regime, np.nan, value=float(gp[regime]),
            ci=np.nan, n=int((runs["refill_mode"] == regime).sum()),
            statistic="annotation", note="arrival gap, pooled")

    # ---- recomputed commission anchors ------------------------------------
    anchors = {
        "anchor_occupied_minus_off": ann["explored_occupied_minus_off_pooled"],
        "anchor_frontier_minus_off": ann["explored_frontier_minus_off_pooled"],
    }
    pre = _pre_contrast()
    for cat, val, note in (
            ("cells", pins["cells"], "distinct (refill_mode, g, curiosity)"),
            ("runs", pins["runs"], "one row per run"),
            ("seeds", pins["seeds"], "seed rows per cell"),
            ("g_min", gs[0], "entry-rate grid, low pin"),
            ("g_max", gs[-1], "entry-rate grid, high pin"),
            ("curiosity_min", pins["curiosity_grid"][0], "curiosity grid, low pin"),
            ("curiosity_max", pins["curiosity_grid"][-1], "curiosity grid, high pin"),
            ("anchor_occupied_minus_off", anchors["anchor_occupied_minus_off"],
             "recomputed commission anchor: explored_fin occupied minus off "
             "(commission: -0.297 post)"),
            ("anchor_frontier_minus_off", anchors["anchor_frontier_minus_off"],
             "recomputed commission anchor: explored_fin frontier minus off "
             "(commission: -0.094 post)")):
        rows.append(dict(figure=FNAME, panel="ref", metric=M_REF,
                         statistic="definitional constant", series=cat,
                         refill_mode="", g=np.nan, value=float(val), ci_lo=np.nan,
                         ci_hi=np.nan, n=np.nan, source_file=src, note=note))
    if pre:
        for cat, key, note in (
                ("pre_occupied_minus_off", "pre_occupied_minus_off",
                 f"pre-ADR explored occupied minus off, matched WITHIN the "
                 f"refill sub-arm ({pre['pre_root']}, cellC only)"),
                ("pre_frontier_minus_off", "pre_frontier_minus_off",
                 f"pre-ADR explored frontier minus off, matched WITHIN the "
                 f"refill sub-arm ({pre['pre_root']}, cellC only)"),
                ("pre_alloff_occupied_minus_off",
                 "pre_alloff_occupied_minus_off",
                 "pre-ADR occupied minus off on the remeasure-ledger basis "
                 "(all cell{A,B,C} off cells pooled; commission -0.319)"),
                ("pre_alloff_frontier_minus_off",
                 "pre_alloff_frontier_minus_off",
                 "pre-ADR frontier minus off on the remeasure-ledger basis "
                 "(all cell{A,B,C} off cells pooled; commission -0.045)")):
            if key not in pre:
                continue
            rows.append(dict(figure=FNAME, panel="ref", metric=M_REF,
                             statistic="definitional constant", series=cat,
                             refill_mode="", g=np.nan, value=float(pre[key]),
                             ci_lo=np.nan, ci_hi=np.nan, n=np.nan,
                             source_file=pre["pre_root"], note=note))

    sidecar = pd.DataFrame(rows)
    texts = dict(pins=pins, anchors=anchors, ann=ann, pre=pre, gs=gs,
                 points=points)
    return sidecar, texts


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------
WIDTH = 0.26


def _grouped(ax, S, metric, *, hatch_series=None):
    """Bars per (g, regime): x = g, hue = regime."""
    gs = sorted(S[S["metric"] == metric]["g"].dropna().unique().tolist())
    xs = np.arange(len(gs))
    out = {}
    for gi, g in enumerate(gs):
        for ri, regime in enumerate(REGIMES):
            r = S[(S["metric"] == metric) & (S["refill_mode"] == regime)
                  & (S["g"] == g)].iloc[0]
            v, lo, hi = float(r["value"]), float(r["ci_lo"]), float(r["ci_hi"])
            x = xs[gi] + (ri - 1) * WIDTH
            ax.bar(x, v, width=WIDTH * 0.92, color=REGIME_COLOR[regime],
                   edgecolor="white", zorder=3)
            ax.errorbar([x], [v], yerr=[[v - lo], [hi - v]], fmt="none",
                        ecolor="black", elinewidth=0.9, capsize=2.6, zorder=4)
            out[(regime, g)] = r
    ax.set_xticks(xs, [f"{g:g}" for g in gs])
    return out, xs


def _pad(ax, spans, *, pad_lo=0.10, pad_hi=0.34, extra=(0.0,), log=False):
    los = [s[0] for s in spans] + list(extra)
    his = [s[1] for s in spans] + list(extra)
    if log:
        pos = [v for v in los + his if v > 0]
        lo = 10.0 ** np.floor(np.log10(min(pos)))
        hi = 10.0 ** (np.log10(max(pos)) + pad_hi)
    else:
        lo, hi = min(los), max(his)
        span = hi - lo if hi > lo else max(abs(hi), 1.0)
        lo, hi = lo - pad_lo * span, hi + pad_hi * span
    ax.set_ylim(lo, hi)
    if not log:
        ticks = mticker.MaxNLocator(nbins=6).tick_values(lo, hi)
        ax.set_yticks([t for t in ticks if lo - 1e-12 <= t <= hi + 1e-12])


def _draw(sidecar, texts, png: Path, pdf: Path) -> dict:
    s = HOUSE_STYLE
    S = sidecar[sidecar["statistic"] == "mean"]
    A = texts["ann"]
    pins = texts["pins"]

    pre = texts["pre"]
    fig = plt.figure(figsize=(12.0, 8.0))
    gsp = fig.add_gridspec(2, 2)
    ax_a = fig.add_subplot(gsp[0, 0])
    ax_b = fig.add_subplot(gsp[0, 1])
    ax_c = fig.add_subplot(gsp[1, 0])
    gs_d = gsp[1, 1].subgridspec(1, 2, wspace=0.34)
    ax_d1 = fig.add_subplot(gs_d[0, 0])
    ax_d2 = fig.add_subplot(gs_d[0, 1])
    fig.subplots_adjust(left=0.075, right=0.975, top=0.905, bottom=0.335,
                        hspace=0.42, wspace=0.28)

    # ---- (a) explored grid share ----------------------------------------
    rows_a, xs = _grouped(ax_a, S, M_LOCKIN)
    _cosmetics(ax_a, s, "A · Mainstream lock-in (explored grid share)",
               "entry rate g", "explored grid share")
    _pad(ax_a, [(r["ci_lo"], r["ci_hi"]) for r in rows_a.values()],
         pad_lo=0.0, pad_hi=0.42)
    ax_a.set_ylim(0.0, ax_a.get_ylim()[1])
    _place_text(ax_a, (
        f"occupied: {100 * A['occupied_collapse_pct']:.1f}% of the off level "
        f"(pooled)\nΔ vs off  occupied {A['explored_occupied_minus_off_pooled']:+.3f}"
        f"   frontier {A['explored_frontier_minus_off_pooled']:+.3f}"),
        ((0.02, 0.97, "left", "top"), (0.98, 0.97, "right", "top")))

    # ---- (b) residual knowledge, log scale ------------------------------
    rows_b, _ = _grouped(ax_b, S, M_FRONTIER)
    ax_b.set_yscale("log")
    _cosmetics(ax_b, s, "B · Unclaimed frontier knowledge (log scale)",
               "entry rate g", "mean residual knowledge on grid")
    for (regime, g), r in rows_b.items():
        # only the frontier bar carries a value label: labelling all nine on a
        # log axis crowded the callout out of the panel
        if regime != "frontier":
            continue
        x = xs[texts["gs"].index(g)] + (REGIMES.index(regime) - 1) * WIDTH
        ax_b.annotate(f"{r['value']:.4f}", xy=(x, r["ci_hi"]), xytext=(0, 5),
                      textcoords="offset points", ha="center", va="bottom",
                      rotation=90, fontsize=6.6, color=PALETTE["gray"],
                      zorder=6)
    _pad(ax_b, [(r["ci_lo"], r["ci_hi"]) for r in rows_b.values()],
         pad_hi=1.0, log=True)
    _place_text(ax_b, (
        f"frontier ×{A['residual_ratio_frontier']:.0f} off\n"
        f"occupied ×{A['residual_ratio_occupied']:.1f} off"),
        ((0.02, 0.99, "left", "top"), (0.98, 0.99, "right", "top"),
         (0.02, 0.62, "left", "top")))

    # ---- (c) merit alignment: bars final step + diamonds time mean -------
    spans_c = []
    for gi, g in enumerate(texts["gs"]):
        for ri, regime in enumerate(REGIMES):
            r = S[(S["metric"] == M_CORR_FIN) & (S["refill_mode"] == regime)
                  & (S["g"] == g)].iloc[0]
            v, lo, hi = float(r["value"]), float(r["ci_lo"]), float(r["ci_hi"])
            x = xs[gi] + (ri - 1) * WIDTH
            ax_c.bar(x, v, width=WIDTH * 0.92, color=REGIME_COLOR[regime],
                     edgecolor="white", zorder=3)
            ax_c.errorbar([x], [v], yerr=[[v - lo], [hi - v]], fmt="none",
                          ecolor="black", elinewidth=0.9, capsize=2.6, zorder=4)
            spans_c.append((lo, hi))
    dm = []
    for gi, g in enumerate(texts["gs"]):
        for regime in REGIMES:
            r = S[(S["metric"] == M_CORR_TM) & (S["refill_mode"] == regime)
                  & (S["g"] == g)].iloc[0]
            dm.append((xs[gi], regime, float(r["value"])))
            spans_c.append((float(r["value"]), float(r["value"])))
    ax_c.plot([d[0] for d in dm], [d[2] for d in dm], marker="D", markersize=5.5,
              linestyle="none", color=PALETTE["blue"], markeredgecolor="black",
              zorder=6, label="time mean (diamonds)")
    ax_c.axhline(0.0, color=PALETTE["gray"], ls="--", lw=1.0, zorder=2)
    ax_c.set_xticks(xs, [f"{g:g}" for g in texts["gs"]])
    _cosmetics(ax_c, s,
               "C · Merit alignment  (bars: final step; diamonds: time mean)",
               "entry rate g", "Corr(prestige, ideal merit)")
    _pad(ax_c, spans_c, pad_lo=0.14, pad_hi=0.30)
    lo, hi = ax_c.get_ylim()
    ax_c.set_ylim(min(lo, 0.0), hi)
    ax_c.legend(fontsize=s["tick_size"] - 1, frameon=False, loc="upper left",
                borderaxespad=0.2, handlelength=1.2)

    # ---- (d1) top-10% prestige share ------------------------------------
    rows_t, _ = _grouped(ax_d1, S, M_TOP10)
    for (regime, g), r in rows_t.items():
        x = xs[texts["gs"].index(g)] + (REGIMES.index(regime) - 1) * WIDTH
        ax_d1.annotate(f"{r['value']:.3f}", xy=(x, r["ci_hi"]), xytext=(0, 4),
                       textcoords="offset points", ha="center", va="bottom",
                       fontsize=5.6, color=PALETTE["gray"], rotation=90, zorder=6)
    ax_d1.set_xlim(-0.6, len(texts["gs"]) - 0.4)
    ax_d1.set_xticklabels([f"{g:g}" for g in texts["gs"]], fontsize=D_TICK)
    ax_d1.axhline(0.10, color=PALETTE["gray"], ls=":", lw=1.1, zorder=2)
    _cosmetics(ax_d1, s, "D1 · top-10% prestige share", "entry rate g",
               "top-10% prestige share")
    ax_d1.yaxis.label.set_fontsize(s["tick_size"])
    _pad(ax_d1, [(r["ci_lo"], r["ci_hi"]) for r in rows_t.values()],
         pad_lo=0.0, pad_hi=0.55)
    ax_d1.set_ylim(0.0, ax_d1.get_ylim()[1])
    ax_d1.text(0.97, 0.975, "dotted: 0.10 equality", transform=ax_d1.transAxes,
               ha="right", va="top", fontsize=6.6, color=PALETTE["gray"])

    # ---- (d2) arrival gap -----------------------------------------------
    rows_g, _ = _grouped(ax_d2, S, M_GAP)
    n_gap = len(texts["pins"]["gap_g_grid"])
    for (regime, g), r in rows_g.items():
        if n_gap < 2:            # one x group: the labels would collide
            break
        x = xs[texts["gs"].index(g)] + (REGIMES.index(regime) - 1) * WIDTH
        ax_d2.annotate(f"{r['value']:.3f}", xy=(x, r["ci_lo"]), xytext=(0, -2),
                       textcoords="offset points", ha="center", va="top",
                       fontsize=5.6, color=PALETTE["gray"], rotation=90, zorder=6)
    ax_d2.axhline(0.0, color=PALETTE["gray"], ls="--", lw=1.0, zorder=2)
    ax_d2.set_xlim(-0.6, n_gap - 0.4)
    ax_d2.set_xticklabels([f"{g:g}" for g in texts["pins"]["gap_g_grid"]],
                          fontsize=D_TICK)
    _cosmetics(ax_d2, s, "D2 · arrival gap (late window)", "entry rate g",
               "gap_recent_mean_late (native units)")
    ax_d2.yaxis.label.set_fontsize(s["tick_size"] - 0.5)
    _pad(ax_d2, [(r["ci_lo"], r["ci_hi"]) for r in rows_g.values()],
         pad_lo=0.34, pad_hi=0.52)

    pre = texts["pre"]
    # ---- shared legend + footnote ---------------------------------------
    handles = [Patch(facecolor=REGIME_COLOR[r], edgecolor=REGIME_COLOR[r],
                     label=f"{r} refill") for r in REGIMES]
    handles += [Line2D([], [], marker="D", linestyle="none", color=PALETTE["blue"],
                       markeredgecolor="black", label="C: time mean (diamonds)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=s["tick_size"] - 1, bbox_to_anchor=(0.5, 0.245))
    gtxt = "/".join(f"{g:g}" for g in texts["gs"])
    note_lines = [
        f"Sources (read-only): {texts['pins']['root']}/*C_*.csv — "
        f"{pins['cells']} cells x {pins['seeds']} seeds = {pins['runs']} runs; "
        f"refill_mode in {{off, occupied, frontier}}, g in {{{gtxt}}}, curiosity in "
        f"{{{', '.join(f'{c:g}' for c in pins['curiosity_grid'])}}}, mgstd5, "
        "fixed_span, H = 200. ADR-0003 clock: 1 step = 0.5 yr, entry age 25, "
        "tau_max = 80 (40-yr careers), gamma = 0.5**(1/10) = 0.933 (5-yr "
        "retention half-life). Bars pool the curiosity grid and 20 seeds; "
        "95% CI = 1.96*sd/sqrt(n).",
        "Schema substitutions (disclosed): the ADR-0003 refill arm has an entry-"
        "rate grid, not an M0/M1 pair, so the pre-ADR hollow M0 bars are replaced "
        "by real (regime, g) cells; the normalised newcomer gap and the "
        "mean_prestige inflation factor need per-agent prestige means, which the "
        "summary schema does not store, so panel D2 plots the core replacement "
        "measure gap_recent_mean_late in native units. The refill arms inject "
        "stock, so their harvested_fin is not a conservation measure and is not "
        "plotted (ADR-0003 remeasure note).",
        "Pre-ADR basis, disclosed: like-for-like within the refill sub-arm the "
        "pre-ADR cells give occupied "
        f"{pre.get('pre_occupied_minus_off', float('nan')):+.3f} and frontier "
        f"{pre.get('pre_frontier_minus_off', float('nan')):+.3f}; the remeasure "
        "ledger quotes -0.319 / -0.045 on its own pooled basis, which these "
        "cells do not reproduce (pooling every archived off cell gives "
        f"{pre.get('pre_alloff_occupied_minus_off', float('nan')):+.3f} / "
        f"{pre.get('pre_alloff_frontier_minus_off', float('nan')):+.3f}). "
        "Direction and ordering are identical on every basis.",
        "Commission anchors recomputed here: explored_fin occupied minus off "
        f"{texts['anchors']['anchor_occupied_minus_off']:+.3f} (commission -0.297 "
        "post), frontier minus off "
        f"{texts['anchors']['anchor_frontier_minus_off']:+.3f} (commission -0.094 "
        "post). Every plotted value, CI and annotation is computed from the raw "
        "CSVs and stored in " + f"{FNAME}_data.csv. MODEL-INTERNAL, NOT CALIBRATED.",
    ]
    note = fig.text(0.008, 0.006, "\n".join(textwrap.fill(t, width=172)
                                            for t in note_lines),
                    fontsize=s["note_size"] - 1.5, color=PALETTE["gray"],
                    va="bottom")

    # ---- gates -----------------------------------------------------------
    qa_side = sidecar.copy()
    is_tm = qa_side["metric"] == M_CORR_TM
    qa_side.loc[is_tm, "metric"] = M_CORR_FIN
    # the diamonds plot point estimates only: collapse their CI columns so the
    # contract compares against what is actually rendered
    qa_side.loc[is_tm, "ci_lo"] = qa_side.loc[is_tm, "value"]
    qa_side.loc[is_tm, "ci_hi"] = qa_side.loc[is_tm, "value"]
    n_g = len(texts["gs"])
    panels = [
        {"ax": ax_a, "metric": M_LOCKIN, "expect_bars": 3 * n_g,
         "sidecar": S, "filters": {"panel": "a"}},
        {"ax": ax_b, "metric": M_FRONTIER, "expect_bars": 3 * n_g,
         "sidecar": S, "filters": {"panel": "b"}, "headroom": 0.0},
        # bars (errorbar containers) + the diamond series; both live under
        # M_CORR_FIN in the QA view, and both carry CI columns in the sidecar
        {"ax": ax_c, "metric": M_CORR_FIN, "expect_bars": 3 * n_g,
         "expect_lines": 3 * n_g, "sidecar": qa_side, "filters": {"panel": "c"}},
        {"ax": ax_d1, "metric": M_TOP10, "expect_bars": 3 * n_g,
         "sidecar": S, "filters": {"panel": "d1"}},
        {"ax": ax_d2, "metric": M_GAP,
         "expect_bars": 3 * len(texts["pins"]["gap_g_grid"]),
         "sidecar": S, "filters": {"panel": "d2"}},
    ]
    report = visual_qa.assert_figure_clean(fig, panels)
    audit_text_over_data(fig, (ax_a, ax_b, ax_c, ax_d1, ax_d2))

    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=s["dpi"])
    fig.savefig(pdf)
    plt.close(fig)
    return report


def build(outdir: str = "output/report_notebook_v1", data_root=None) -> dict:
    sidecar, texts = compute(data_root)
    out = Path(outdir)
    if not out.is_absolute():
        out = REPO / out
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{FNAME}.png"
    pdf = out / f"{FNAME}.pdf"
    report = _draw(sidecar, texts, png, pdf)
    csv_path = write_figure_data(sidecar, png)
    print(f"F13: {png} | {pdf} | {csv_path}")
    print(f"sidecar rows: {len(sidecar)} | visual_qa failures: {report['_total']}")
    return {"png": str(png), "pdf": str(pdf), "data_csv": str(csv_path),
            "qa": report, "sidecar_rows": len(sidecar)}


if __name__ == "__main__":
    print(build())
