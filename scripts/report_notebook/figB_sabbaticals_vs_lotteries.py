"""F12 -- sabbatical shocks vs. PhD entry lotteries (Section 5.2 Part A).

Figure B of the section-figures-v2 commission: an exhaustive, F6-aligned
comparison of the six ``(alpha, p_reloc)`` intervention arms of
``data/interventions_v1/cells/cellB_*.csv`` (36 cells x 20 seeds = 720 runs)
along the visibility axis ``xi = 1 - c`` in {0.5, 0.7, 0.9}:

  (a) Macro efficiency:        harvested knowledge share
  (b) Mechanism:               explored grid share (explored_final)
  (c) Epistemic shortfall:     terminal 1-step loss (-ideal_merit_sum_final)
  (d) Micro alignment:         Corr(prestige, ideal merit) (time mean)

Vocabulary (commission sec. 1): hue = p_reloc (0.00 baseline / 0.02
sabbatical), marker+linestyle = alpha (1.0 linear roulette / 0.5 damped /
0.0 pure lottery).  Curves pool over mgstd5 + noisypr5 with 95% CIs
(1.96 * sd / sqrt(n)); per-landscape rows live in the ``_data.csv`` sidecar.

Every annotation value is computed here from the raw cell CSVs and written
to the sidecar (commission rule 3).  Known drift, pinned exactly:
the commission's "+20.2 pp mgstd5 Stage-2 harvest rescue" reproduces on
run means at alpha = 1.0 (20.22 pp); pooled over all alpha the same
contrast is 20.30 pp; the coordinator's spot-check figure of 20.1 pp is
not reproducible under any run-mean aggregation tried (see sidecar rows
``mgstd5_stage2_harvest_rescue_pp*``).

Run:  MPLBACKEND=Agg python3 scripts/report_notebook/figB_sabbaticals_vs_lotteries.py
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
import visual_qa  # noqa: E402

CELLS = REPO / "data" / "interventions_v1" / "cells"
FNAME = "F12_sabbaticals_vs_lotteries"

XIS = (0.5, 0.7, 0.9)
ALPHAS = (0.0, 0.5, 1.0)
PRELOCS = (0.00, 0.02)
LANDSCAPES = ("pooled", "mgstd5", "noisypr5")

#: hue = p_reloc (commission sec. 2 Figure B)
HUE = {0.00: PALETTE["purple"], 0.02: PALETTE["green"]}
#: marker/linestyle = alpha: 1.0 linear roulette, 0.5 damped, 0.0 pure lottery
MARK = {1.0: ("-", "o"), 0.5: ("--", "s"), 0.0: (":", "^")}
SHORT = {
    (0.00, 1.0): "baseline p=0.00 · α=1.0 linear roulette",
    (0.00, 0.5): "baseline p=0.00 · α=0.5 damped",
    (0.00, 0.0): "baseline p=0.00 · α=0.0 pure lottery",
    (0.02, 1.0): "sabbatical p=0.02 · α=1.0 linear roulette",
    (0.02, 0.5): "sabbatical p=0.02 · α=0.5 damped",
    (0.02, 0.0): "sabbatical p=0.02 · α=0.0 pure lottery",
}

METRIC_COL = {
    "harvested knowledge share": "percentage_knowledge_harvested",
    "explored grid share": "explored_final",
    "terminal 1-step loss (-ideal_merit_sum_final)": "ideal_merit_sum_final",
    "Corr(prestige, ideal merit) (time mean)": "Mean_of_Corr_prestige_ideal_merit",
    "top-10% prestige share": "top10_final",
}


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


# ----------------------------------------------------------------- data ----
def data_root_path(data_root=None) -> Path:
    return Path(data_root) if data_root is not None else CELLS


def load_runs(data_root=None) -> pd.DataFrame:
    files = sorted(data_root_path(data_root).glob("cellB_*.csv"))
    _require(len(files) == 36, f"cellB files=36, found {len(files)}")
    frames = []
    for f in files:
        d = pd.read_csv(f)
        _require(len(d) == 20, f"{f.name}: 20 seeds expected, got {len(d)}")
        d["source_file"] = f.name
        frames.append(d)
    b = pd.concat(frames, ignore_index=True)
    _require(len(b) == 720, f"720 runs expected, got {len(b)}")
    _require(sorted(b["alpha"].unique().tolist()) == list(ALPHAS), "alpha grid")
    _require(sorted(b["p_reloc"].unique().tolist()) == list(PRELOCS), "p_reloc grid")
    _require(sorted(b["visibility_weight"].unique().tolist()) == list(XIS), "xi grid")
    _require(bool(np.allclose(b["visibility_weight"], 1.0 - b["initial_curiosity"])),
             "xi must equal 1 - initial_curiosity")
    return b


def series_rows(b: pd.DataFrame) -> pd.DataFrame:
    """Mean / 95% CI for every (panel metric, alpha, p_reloc, xi, landscape)."""
    rows = []
    for metric, col in METRIC_COL.items():
        vals = -b[col] if col == "ideal_merit_sum_final" else b[col]
        d = b.assign(_v=vals)
        panel = {"harvested knowledge share": "a",
                 "explored grid share": "b",
                 "terminal 1-step loss (-ideal_merit_sum_final)": "c",
                 "Corr(prestige, ideal merit) (time mean)": "d",
                 "top-10% prestige share": "d"}[metric]
        for xi in XIS:
            for alpha in ALPHAS:
                for preloc in PRELOCS:
                    for land in LANDSCAPES:
                        sel = (d["visibility_weight"] == xi) & (d["alpha"] == alpha) \
                            & (d["p_reloc"] == preloc)
                        if land != "pooled":
                            sel &= d["landscape"] == land
                        sub = d.loc[sel, "_v"]
                        _require(len(sub) == (40 if land == "pooled" else 20),
                                 f"{metric} {xi} {alpha} {preloc} {land}: n={len(sub)}")
                        mean = float(sub.mean())
                        ci = 1.96 * float(sub.std(ddof=1)) / np.sqrt(len(sub))
                        rows.append(dict(
                            figure=FNAME, panel=panel, metric=metric,
                            statistic="mean", alpha=alpha, p_reloc=preloc,
                            xi=xi, landscape=land, value=mean,
                            ci_lo=mean - ci, ci_hi=mean + ci, n=int(len(sub)),
                            source_file=" + ".join(sorted(d.loc[sel, "source_file"].unique())),
                        ))
    return pd.DataFrame(rows)


def _sel(b: pd.DataFrame, *, xi: float, alpha: float, preloc: float,
         land: str = "pooled") -> pd.DataFrame:
    s = (b["visibility_weight"] == xi) & (b["alpha"] == alpha) & (b["p_reloc"] == preloc)
    if land != "pooled":
        s &= b["landscape"] == land
    return b[s]


def _mean(b: pd.DataFrame, col: str, **kw) -> float:
    d = _sel(b, **kw)
    v = -d[col] if col == "ideal_merit_sum_final" else d[col]
    return float(v.mean())


def annotation_rows(b: pd.DataFrame) -> pd.DataFrame:
    """Every number written on the canvas, computed from the raw cell CSVs."""
    rows = []

    def add(panel: str, metric: str, value: float, note: str) -> None:
        rows.append(dict(figure=FNAME, panel=panel, metric=metric,
                         statistic="annotation", alpha=np.nan, p_reloc=np.nan,
                         xi=np.nan, landscape="pooled", value=float(value),
                         ci_lo=np.nan, ci_hi=np.nan, n=720,
                         source_file="data/interventions_v1/cells/cellB_*.csv",
                         note=note))

    h = "percentage_knowledge_harvested"
    # -- (a) harvest rescue at xi = 0.9, percentage points ------------------
    r_mg_a1 = 100.0 * (_mean(b, h, xi=0.9, alpha=1.0, preloc=0.02, land="mgstd5")
                       - _mean(b, h, xi=0.9, alpha=1.0, preloc=0.00, land="mgstd5"))
    add("a", "mgstd5_stage2_harvest_rescue_pp", r_mg_a1,
        "mgstd5 Stage-2 harvest rescue at xi=0.9, p_reloc 0->0.02, alpha=1.0, "
        "run means (commission quotes +20.2 pp)")
    # same contrast pooled over all alpha (drift pin)
    for land, key in (("mgstd5", "mgstd5_stage2_harvest_rescue_pp_all_alpha"),
                      ("pooled", "pooled_stage2_harvest_rescue_pp_all_alpha")):
        s = b[(b["visibility_weight"] == 0.9)]
        if land != "pooled":
            s = s[s["landscape"] == land]
        v = 100.0 * (s[s["p_reloc"] == 0.02][h].mean()
                     - s[s["p_reloc"] == 0.00][h].mean())
        add("a", key, v,
            f"{land} Stage-2 harvest rescue at xi=0.9 pooled over all alpha, "
            "run means (drift pin)")
    r_pool_a1 = 100.0 * (_mean(b, h, xi=0.9, alpha=1.0, preloc=0.02)
                         - _mean(b, h, xi=0.9, alpha=1.0, preloc=0.00))
    add("a", "pooled_stage2_harvest_rescue_pp", r_pool_a1,
        "pooled-landscape Stage-2 harvest rescue at xi=0.9, alpha=1.0 "
        "(commission quotes +16.7 pp)")
    lot = max(abs(100.0 * (_mean(b, h, xi=0.9, alpha=0.0, preloc=p)
                           - _mean(b, h, xi=0.9, alpha=1.0, preloc=p)))
              for p in PRELOCS)
    add("a", "lottery_harvest_effect_pp_max", lot,
        "max |harvest(alpha=0) - harvest(alpha=1)| at xi=0.9 over p_reloc "
        "(commission: lotteries <1 pp)")

    # -- (b) explored grid share tripling at xi = 0.9 ----------------------
    e_b = _mean(b, "explored_final", xi=0.9, alpha=1.0, preloc=0.00)
    e_s = _mean(b, "explored_final", xi=0.9, alpha=1.0, preloc=0.02)
    add("b", "explored_xi09_baseline", e_b, "explored_final, xi=0.9, alpha=1.0, p=0")
    add("b", "explored_xi09_sabbatical", e_s, "explored_final, xi=0.9, alpha=1.0, p=0.02")
    add("b", "explored_xi09_fold", e_s / e_b,
        "explored grid share fold change, xi=0.9, alpha=1.0 (commission: 31.5%->97.6%)")

    # -- (c) high-visibility penalty and its reductions --------------------
    L = "ideal_merit_sum_final"
    pen_b = _mean(b, L, xi=0.9, alpha=1.0, preloc=0.00) - _mean(b, L, xi=0.5, alpha=1.0, preloc=0.00)
    pen_s = _mean(b, L, xi=0.9, alpha=1.0, preloc=0.02) - _mean(b, L, xi=0.5, alpha=1.0, preloc=0.02)
    pen_l = _mean(b, L, xi=0.9, alpha=0.0, preloc=0.00) - _mean(b, L, xi=0.5, alpha=0.0, preloc=0.00)
    add("c", "high_visibility_penalty_baseline", pen_b,
        "L(0.9)-L(0.5) terminal 1-step loss, alpha=1.0, p=0")
    add("c", "high_visibility_penalty_sabbatical", pen_s,
        "L(0.9)-L(0.5) terminal 1-step loss, alpha=1.0, p=0.02")
    add("c", "penalty_reduction_pct_sabbatical", 100.0 * (pen_s / pen_b - 1.0),
        "high-visibility penalty reduction, sabbatical alpha=1.0 "
        "(commission: -40.0%)")
    l9_b = _mean(b, L, xi=0.9, alpha=1.0, preloc=0.00)
    l9_s = _mean(b, L, xi=0.9, alpha=1.0, preloc=0.02)
    add("c", "loss_xi09_reduction_pct_sabbatical", 100.0 * (l9_s / l9_b - 1.0),
        "terminal 1-step loss reduction at xi=0.9, sabbatical alpha=1.0 "
        "(commission: -41.5%)")
    add("c", "penalty_reduction_pct_lottery", 100.0 * (pen_l / pen_b - 1.0),
        "high-visibility penalty reduction, lottery alpha=0 vs alpha=1, p=0 "
        "(commission: -0.9%)")

    # -- (d) alignment lift and concentration note ------------------------
    c_b = _mean(b, "Mean_of_Corr_prestige_ideal_merit", xi=0.9, alpha=1.0, preloc=0.00)
    c_s = _mean(b, "Mean_of_Corr_prestige_ideal_merit", xi=0.9, alpha=1.0, preloc=0.02)
    add("d", "corr_tm_xi09_baseline", c_b, "Corr time mean, xi=0.9, alpha=1.0, p=0")
    add("d", "corr_tm_xi09_sabbatical", c_s, "Corr time mean, xi=0.9, alpha=1.0, p=0.02")
    t_b = _mean(b, "top10_final", xi=0.9, alpha=1.0, preloc=0.00)
    t_l = _mean(b, "top10_final", xi=0.9, alpha=0.0, preloc=0.00)
    t_s = _mean(b, "top10_final", xi=0.9, alpha=1.0, preloc=0.02)
    add("d", "top10_xi09_baseline", t_b, "top10_final, xi=0.9, alpha=1.0, p=0")
    add("d", "top10_xi09_lottery", t_l, "top10_final, xi=0.9, alpha=0.0, p=0")
    add("d", "top10_xi09_sabbatical", t_s, "top10_final, xi=0.9, alpha=1.0, p=0.02")
    return pd.DataFrame(rows)


# --------------------------------------------------------------- drawing ---
def _bands(ax) -> None:
    """F6's exact two-stage bands: Stage 1 [0.3, 0.6], kink [0.6, 0.8], Stage 2."""
    ax.axvspan(0.3, 0.6, color=PALETTE["gray_light"], alpha=0.18, zorder=0)
    ax.axvspan(0.6, 0.8, facecolor=PALETTE["purple_light"], alpha=0.18,
               hatch="///", edgecolor=PALETTE["gray"], linewidth=0, zorder=0)
    for xv in (0.6, 0.8):
        ax.axvline(xv, ls="--", lw=0.9, color=PALETTE["gray"], zorder=1)
    ax.axvspan(0.8, 1.0, color=PALETTE["gray_light"], alpha=0.35, zorder=0)
    # v3: band labels are placed by _band_labels() AFTER the series are drawn,
    # because their position must be free of data (they used to collide with
    # curves reaching the top of a panel).


def _band_labels(ax) -> None:
    """Place the three band labels in the first data-free band of the panel."""
    for x, lab in ((0.45, "Stage 1"), (0.70, "kink"), (0.90, "Stage 2")):
        placed = False
        for y, va in ((0.965, "top"), (0.500, "center"), (0.035, "bottom"),
                      (0.250, "center"), (0.750, "center")):
            t = ax.text(x, y, lab, transform=ax.get_xaxis_transform(),
                        ha="center", va=va,
                        fontsize=HOUSE_STYLE["tick_size"] - 1,
                        color=PALETTE["gray"], zorder=5)
            try:
                _no_text_overlap(ax.figure)
            except AssertionError:
                t.remove()
                continue
            if visual_qa.text_hits_data(ax, t):
                t.remove()
                continue
            placed = True
            break
        if not placed:
            raise AssertionError(f"no data-free band-label position for {lab!r}")


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


def _points_axes_coords(ax) -> np.ndarray:
    """Every plotted point of `ax` in axes-fraction coordinates."""
    inv = ax.transAxes.inverted()
    pts = []
    for ln in ax.lines:
        xy = ln.get_xydata()
        if len(xy):
            pts.append(inv.transform(ax.transData.transform(xy)))
    return np.vstack(pts) if pts else np.empty((0, 2))


def _place_callout(ax, text: str, prefer: tuple[str, ...] = ()) -> None:
    """Put `text` in the emptiest corner band of `ax` (no data, no text)."""
    anchors = {
        "tl": (0.04, 0.94, "left", "top"),
        "tr": (0.96, 0.94, "right", "top"),
        "bl": (0.04, 0.06, "left", "bottom"),
        "br": (0.96, 0.06, "right", "bottom"),
        "ml": (0.04, 0.50, "left", "center"),
        "mr": (0.96, 0.50, "right", "center"),
        "tr2": (0.96, 0.76, "right", "top"),
        "tl2": (0.04, 0.76, "left", "top"),
        "br2": (0.96, 0.24, "right", "bottom"),
        "bl2": (0.04, 0.24, "left", "bottom"),
    }
    order = tuple(prefer) + tuple(k for k in anchors if k not in prefer)
    pts = _points_axes_coords(ax)
    # v3: wrap to a narrow block so a long callout can fit beside the curves
    text = "\n".join(textwrap.fill(t, width=42) for t in text.split("\n"))
    nlines = len(text.split("\n"))
    w = max(len(t) for t in text.split("\n")) * 0.0072
    h = 0.075 * nlines + 0.02
    for key in order:
        x, y, ha, va = anchors[key]
        x0, x1 = (x, x + w) if ha == "left" else (x - w, x)
        if va == "top":
            y0, y1 = y - h, y
        elif va == "bottom":
            y0, y1 = y, y + h
        else:
            y0, y1 = y - h / 2, y + h / 2
        box = (x0 - 0.01, y0 - 0.01, x1 + 0.01, y1 + 0.01)
        if len(pts):
            hit = ((pts[:, 0] >= box[0]) & (pts[:, 0] <= box[2])
                   & (pts[:, 1] >= box[1]) & (pts[:, 1] <= box[3])).any()
        else:
            hit = False
        if hit:
            continue
        t = ax.text(x, y, text, transform=ax.transAxes,
                    fontsize=HOUSE_STYLE["note_size"], va=va, ha=ha,
                    color=PALETTE["gray"])
        try:
            _no_text_overlap(ax.figure)
        except AssertionError:
            t.remove()
            continue
        if visual_qa.text_hits_data(ax, t):   # v3: caps/whiskers, not only points
            t.remove()
            continue
        return
    raise AssertionError(f"no free placement found for callout: {text[:40]!r}")


def _plot_arm(ax, s: pd.DataFrame, alpha: float, preloc: float) -> None:
    d = s[(s["alpha"] == alpha) & (s["p_reloc"] == preloc)
          & (s["landscape"] == "pooled")].sort_values("xi")
    ls, mk = MARK[alpha]
    ax.errorbar(d["xi"].to_numpy(float), d["value"].to_numpy(float),
                yerr=np.vstack([d["value"] - d["ci_lo"],
                                d["ci_hi"] - d["value"]]),
                color=HUE[preloc], linestyle=ls, marker=mk, linewidth=1.7,
                markersize=4.6, capsize=2.2, elinewidth=0.9,
                label=SHORT[(preloc, alpha)], zorder=4,
                markerfacecolor="white" if preloc == 0.00 else HUE[preloc],
                markeredgecolor=HUE[preloc], markeredgewidth=1.0)


def build(outdir: str = "output/report_notebook_v1",
          data_root=None) -> dict:
    b = load_runs(data_root)
    series = series_rows(b)
    anns = annotation_rows(b)
    sidecar = pd.concat([series, anns], ignore_index=True)

    A = {r["metric"]: r["value"] for _, r in anns.iterrows()}

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6))
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    panels = (
        (ax_a, "harvested knowledge share", "harvested knowledge share"),
        (ax_b, "explored grid share", "explored grid share"),
        (ax_c, "terminal 1-step loss (-ideal_merit_sum_final)",
         "terminal 1-step loss"),
        (ax_d, "Corr(prestige, ideal merit) (time mean)",
         "Corr(prestige, ideal merit)\n(time mean)"),
    )
    titles = {
        ax_a: "A · Macro efficiency: harvested knowledge share",
        ax_b: "B · Mechanism: explored grid share",
        ax_c: "C · Epistemic shortfall: terminal 1-step loss",
        ax_d: "D · Micro alignment: Corr(prestige, ideal merit)",
    }
    for ax, metric, ylabel in panels:
        _bands(ax)
        s_metric = series[series["metric"] == metric]  # v3: single-metric panel
        for alpha in ALPHAS:
            for preloc in PRELOCS:
                _plot_arm(ax, s_metric, alpha, preloc)
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks(list(XIS))
        _cosmetics(ax, HOUSE_STYLE, titles[ax],
                   "visibility weight  $\\xi = 1 - c$", ylabel)

    # computed callouts (every number below comes from `anns`)
    _place_callout(ax_a,
                   f"mgstd5 Stage-2 rescue: +{A['mgstd5_stage2_harvest_rescue_pp']:.1f} pp "
                   f"(α=1)\npooled-landscape: +{A['pooled_stage2_harvest_rescue_pp']:.1f} pp; "
                   f"lotteries <{max(1.0, np.ceil(A['lottery_harvest_effect_pp_max'])):.0f} pp",
                   prefer=("tr", "br", "tl"))
    _place_callout(ax_b,
                   f"explored {100 * A['explored_xi09_baseline']:.1f}% → "
                   f"{100 * A['explored_xi09_sabbatical']:.1f}% "
                   f"(×{A['explored_xi09_fold']:.1f}, sabbatical, α=1, ξ=0.9)",
                   prefer=("ml", "bl", "mr"))
    _place_callout(ax_c,
                   f"penalty L(0.9)−L(0.5): "
                   f"{A['penalty_reduction_pct_sabbatical']:.1f}% (sabbatical, α=1)\n"
                   f"L(0.9): {A['loss_xi09_reduction_pct_sabbatical']:.1f}%; "
                   f"lottery penalty: {A['penalty_reduction_pct_lottery']:.1f}%",
                   prefer=("tl", "bl", "ml"))
    _place_callout(ax_d,
                   f"corr {A['corr_tm_xi09_baseline']:+.3f} → "
                   f"{A['corr_tm_xi09_sabbatical']:+.3f} (ξ=0.9, α=1)\n"
                   f"top10: {A['top10_xi09_baseline']:.3f} → "
                   f"{A['top10_xi09_lottery']:.3f} (lottery) vs "
                   f"{A['top10_xi09_sabbatical']:.3f} (sabbatical)",
                   prefer=("tr", "tl", "mr"))

    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 0.082))
    footnote = (
        "Sources: "
        + str(data_root_path(data_root).relative_to(REPO))
        + "/cellB_*.csv (36 cells x 20 seeds = "
        "720 runs; curves pool mgstd5+noisypr5, per-landscape rows in the data "
        "CSV). Curves: run means +/- 95% CI (1.96*sd/sqrt(n)); xi = 1 - c = "
        "visibility_weight. "
        f"Measured: mgstd5 Stage-2 harvest rescue at xi=0.9, alpha=1.0 "
        f"{A['mgstd5_stage2_harvest_rescue_pp']:+.1f} pp (pooled-landscape "
        f"{A['pooled_stage2_harvest_rescue_pp']:+.1f} pp); pooled over all alpha "
        f"{A['mgstd5_stage2_harvest_rescue_pp_all_alpha']:+.1f} pp (mgstd5) -- the "
        "commission's +20.2 pp reproduces at alpha=1.0; the 20.1 pp spot-check "
        "figure matches no run-mean aggregation. "
        f"Penalty reduction {A['penalty_reduction_pct_sabbatical']:.1f}% "
        f"(loss at 0.9 {A['loss_xi09_reduction_pct_sabbatical']:.1f}%), lottery "
        f"{A['penalty_reduction_pct_lottery']:.1f}%. Model-internal, not "
        "calibrated."
    )
    fig.text(0.01, 0.005, "\n".join(textwrap.fill(t, width=150)
                                    for t in footnote.split("\n")),
             fontsize=HOUSE_STYLE["note_size"], color=PALETTE["gray"],
             va="bottom")
    fig.subplots_adjust(left=0.075, right=0.975, top=0.93, bottom=0.215,
                        hspace=0.42, wspace=0.26)

    # v3: band labels LAST, after subplots_adjust -- fixed-point text boxes
    # grow relative to a shrinking axes, so a placement that passes before the
    # final layout can collide afterwards (panel D's callout vs 'Stage 1').
    for ax in (ax_a, ax_b, ax_c, ax_d):
        _band_labels(ax)

    w, h = fig.get_size_inches() * fig.dpi
    assert_text_fits(fig, w, h)
    _no_text_overlap(fig)

    import visual_qa  # v3 hardened contract (rendered artists + collisions)
    _qa_common = {"expect_lines": 6, "points_per_line": 3,
                  "sidecar": series, "filters": {"landscape": "pooled"}}
    visual_qa.assert_figure_clean(fig, [
        {"ax": ax_a, "metric": "harvested knowledge share", **_qa_common},
        {"ax": ax_b, "metric": "explored grid share", **_qa_common},
        {"ax": ax_c, "metric": "terminal 1-step loss (-ideal_merit_sum_final)",
         **_qa_common},
        {"ax": ax_d, "metric": "Corr(prestige, ideal merit) (time mean)",
         **_qa_common},
    ])

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
