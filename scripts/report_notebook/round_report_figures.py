"""House-style figures for the round-results report notebook.

ADR-0002: this is the one shared plotting helper for the report notebook.
Every figure is drawn through the borrowed CodeLearn templates
(``diagram_templates.py``, byte-identical copy -- sha256 verified against
``/workspace/git/ai/CodeLearn/P_diagram_templates/``), so the palette,
sizes and input contracts are the project's house style rather than a
per-notebook restyling.

Each builder loads through ``round_report_data`` (which asserts the gross
shape), draws, writes one PNG into ``output/report_notebook_v1/`` and
returns the figure so the notebook can embed it.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import textwrap

import diagram_templates as T
import round_report_data as D
import visual_qa

REPO = D.REPO
OUT = REPO / "output" / "report_notebook_v1"

SOURCE_T3 = "source: careerB_analysis.csv + noisyB_analysis.csv (9 cells x 20 seeds each), curiosity averaged"
SOURCE_T4 = "source: data/welfare_v1 (40 cells x 20 seeds), means over the visibility toggle"
SOURCE_WA = "source: data/prestige_stock_v1/summary_cell.csv (36 cells x 20 seeds)"
SOURCE_RM = "source: data/sensitivity_v1 x data/remeasure_v1, 720 seed-matched T1 pairs"


def _outdir(outdir: Path | str | None) -> Path:
    out = Path(outdir) if outdir else OUT
    out.mkdir(parents=True, exist_ok=True)
    return out


# --------------------------------------------------------------------------
# F1 -- T1 sensitivity: the wedge opens under entry
# --------------------------------------------------------------------------
def figure_sensitivity(outdir: Path | str | None = None):
    d = D.sensitivity_summary()
    cap = D.sensitivity_caption()
    out = _outdir(outdir) / "F1_sensitivity_entry_curiosity.png"
    fig, ax = T.styled_linechart(
        d,
        x="entry",
        y="c_ideal_fin",
        group="curiosity",
        title="Prestige no longer tracks ideal merit once entry is on",
        xlabel="entry rate (agent_generation_time)",
        ylabel="Corr(prestige, ideal merit), final step",
        source=(f"post-audit: entry=0 {cap['entry0_mean']:+.3f}, "
                f"entry=0.1 {cap['entry01_mean']:+.3f}, "
                f"{cap['entry01_negative']}/{cap['entry01_cells']} cells "
                f"negative | source: data/remeasure_v1/runs.csv, 720 runs"),
        outpath=out,
    )
    return fig, out, cap


# --------------------------------------------------------------------------
# F2 -- T3 career flows: the newcomer-incumbent gap never closes
# --------------------------------------------------------------------------
def figure_career_flows(outdir: Path | str | None = None):
    d = D.career_flows_comparison()
    out = _outdir(outdir) / "F2_career_flows_gap_ratio.png"
    fig, ax = T.styled_linechart(
        d,
        x="entry",
        y="gap_ratio",
        group="series",
        title="Newcomer prestige stays below incumbent prestige on both terrains",
        xlabel="entry rate (agent_generation_time)",
        ylabel="newcomer / incumbent prestige gap ratio",
        source=SOURCE_T3,
        outpath=out,
    )
    return fig, out


def figure_career_flows_ideal_corr(outdir: Path | str | None = None):
    d = D.career_flows_ideal_corr()
    out = _outdir(outdir) / "F2b_career_flows_noisy_vs_clean.png"
    fig, ax = T.styled_linechart(
        d,
        x="entry",
        y="ideal_corr",
        group="landscape",
        title="Slower entry decouples prestige from merit more strongly",
        xlabel="entry rate (agent_generation_time)",
        ylabel="Corr(prestige, ideal merit), final step",
        source=SOURCE_T3,
        outpath=out,
    )
    return fig, out


# --------------------------------------------------------------------------
# F3 -- W_v05 welfare on the ADR-0003 clock: the price of visibility
#       pressure, its knee, and the (reversed) entry gradient of cumulative
#       loss.
#
# ADR-0003 (paper v3.5): this builder read data/welfare_v1 through
# ``round_report_data`` (40 cells, entry 0/0.1, two gaussian legs). The
# ADR-0003 archive keeps a different Cartesian product -- curiosity x
# visibility-reward toggle x landscape leg x g -- on the 6-month clock, so
# the loader lives here (parameterized ``data_root``) instead of in the
# shared data module, and only this figure is touched by it.
# --------------------------------------------------------------------------
ADR_WELFARE_ROOT = "data/welfare_adr0003"
#: pre-ADR reference (read-only) for the before/after note and its drift check
PRE_WELFARE_RUNS = "data/remeasure_v1/w_v05_runs.csv"

WELFARE_WEIGHTS = [0.1, 0.3, 0.5, 0.7, 0.9]
WELFARE_GS = [0.125, 0.25, 0.5]
WELFARE_MEASURES = ["ideal_merit_sum", "ideal_loss_cum", "harvested_fin"]
Z95 = 1.96

#: ADR-0003 pins quoted in the provenance note (clock, not numbers)
ADR_PINS = ("ADR-0003 clock: 1 step = 0.5 yr, entry age 25, tau_max 80 "
            "(40-yr careers), gamma = 0.5**(1/10) = 0.933, g in "
            "{0.125, 0.25, 0.5}, exit_mode fixed_span, H = 200")

M_TERM = "terminal ideal-merit loss (-sum over agents, positive)"
M_CUM = "cumulative ideal-merit loss (sum over agents, positive)"
M_REF = "reference (definitional constants and archive pins)"


def welfare_root(data_root: Path | str | None = None) -> Path:
    """Resolve the welfare archive root (default = the ADR-0003 archive)."""
    return Path(data_root) if data_root is not None else REPO / ADR_WELFARE_ROOT


def welfare_cells(data_root: Path | str | None = None) -> pd.DataFrame:
    """One row per run of the W_v05 welfare arm (60 cells x 20 seeds)."""
    root = welfare_root(data_root)
    files = sorted((root / "cells").glob("*.csv"))
    if not files:
        # pre-ADR archives carry a single flat frame instead of cells/
        flat = root / "w_v05_runs.csv"
        if flat.exists():
            d = pd.read_csv(flat)
            d["visibility_weight"] = 1.0 - d["curiosity"]
            d["term_loss"] = -d["ideal_merit_sum"]
            return d
        raise AssertionError(f"no welfare cells under {root}")
    D._require(len(files) == 60, f"W_v05 gross cells=60, found {len(files)}")
    frames = []
    for f in files:
        d = pd.read_csv(f)
        D._require(len(d) == 20, f"{f.name}: gross seeds=20, got {len(d)}")
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    D._require(len(out) == 1200, f"W_v05 net runs=1200, got {len(out)}")
    D._require(sorted(out["curiosity"].unique().tolist()) == WELFARE_WEIGHTS,
               "W_v05 curiosity grid must be 0.1..0.9 by 0.2")
    D._require(sorted(out["g"].unique().tolist()) == WELFARE_GS,
               "W_v05 g grid must be the ADR-0003 entry grid")
    D._require(sorted(out["leg"].unique().tolist()) == [2, 6],
               "W_v05 landscape-leg tags must be {2, 6}")
    D._require(sorted(out["use_visibility_reward"].unique().tolist())
               == [False, True], "W_v05 visibility toggle must be boolean")
    D._require(out["exit_mode"].unique().tolist() == ["fixed_span"],
               "W_v05 exit_mode is pinned to the fixed_span regime")
    D._require(out["tau_max"].unique().tolist() == [80], "W_v05 tau_max=80")
    D._require(out["horizon"].unique().tolist() == [200], "W_v05 H=200")
    D._require(out["seed"].nunique() == 20, "W_v05 seeds must be 0..19")
    out = out.copy()
    out["visibility_weight"] = (1.0 - out["curiosity"]).round(2)
    out["term_loss"] = -out["ideal_merit_sum"]
    # the two landscape-leg tags both resolve to the mgstd5 landscape in this
    # archive; if they ever stop being seed-identical the figure must draw
    # them as separate series instead of averaging them away.
    if out["landscape"].nunique() == 1 and set(out["leg"].unique()) == {2, 6}:
        keys = ["curiosity", "g", "use_visibility_reward", "seed"]
        a = out[out["leg"] == 2].set_index(keys)
        b = out[out["leg"] == 6].set_index(keys)
        D._require(a.index.equals(b.index), "leg-2 / leg-6 cell keys differ")
        for col in WELFARE_MEASURES:
            av = a[col].to_numpy(float)
            bv = b[col].to_numpy(float)
            D._require(np.array_equal(av, bv, equal_nan=True),
                       f"leg-2 / leg-6 not seed-identical on {col}")
        out.attrs["legs_duplicated_landscape"] = str(out["landscape"].iloc[0])
    return out


def welfare_summary(data_root: Path | str | None = None) -> pd.DataFrame:
    """F3's input: one row per (visibility weight, g), seed-level 95% CI.

    Design axes folded into the plotted mean: the visibility-reward toggle and
    the duplicated landscape-leg tag (verified seed-identical above). The
    per-seed value is the mean over the folded cells, so the CI is seed
    variation, not a design blend.
    """
    d = welfare_cells(data_root)
    per_seed = d.groupby(["visibility_weight", "g", "seed"], as_index=False).agg(
        term_loss=("term_loss", "mean"),
        cum_loss=("ideal_loss_cum", "mean"),
        harvested=("harvested_fin", "mean"),
        n_cells=("term_loss", "size"),
    )
    g = d.groupby(["visibility_weight", "g"], as_index=False).agg(
        term_loss=("term_loss", "mean"),
        cum_loss=("ideal_loss_cum", "mean"),
        harvested=("harvested_fin", "mean"),
        n_runs=("seed", "size"),
    )
    for col in ("term_loss", "cum_loss"):
        stat = per_seed.groupby(["visibility_weight", "g"])[col].agg(
            ["mean", "std", "size"])
        ci = Z95 * stat["std"] / np.sqrt(stat["size"])
        g[f"{col}_ci"] = [float(ci.loc[(r.visibility_weight, r.g)])
                          for r in g.itertuples()]
    g["series"] = [f"g = {v:.3f}" for v in g["g"]]
    g = g.sort_values(["visibility_weight", "g"]).reset_index(drop=True)
    D._require(len(g) == len(WELFARE_WEIGHTS) * len(WELFARE_GS),
               f"F3 grid 5 weights x 3 g = 15 rows, got {len(g)}")
    D._require((g["n_runs"] == 2 * 2 * 20).all(),
               "F3 rows must carry 2 legs x 2 toggles x 20 seeds = 80 runs")
    return g


def welfare_toggle_gap(data_root: Path | str | None = None) -> dict:
    """Largest |toggle effect| on any plotted point (per metric)."""
    d = welfare_cells(data_root)
    out = {}
    for metric, col in (("term_loss", "term_loss"), ("cum_loss", "ideal_loss_cum")):
        piv = (d.groupby(["visibility_weight", "g", "use_visibility_reward"])
               [col].mean().unstack("use_visibility_reward"))
        out[metric] = float((piv[True] - piv[False]).abs().max())
    return out


def welfare_reference() -> dict | None:
    """Pre-ADR reference values, recomputed from the read-only archive.

    Returns ``None`` (and the note omits the comparison) when the pre-ADR
    archive is not present -- never a hardcoded fallback.
    """
    path = REPO / PRE_WELFARE_RUNS
    if not path.exists():
        return None
    d = pd.read_csv(path)
    if "origin" in d.columns:
        origin = d["origin"].astype(str).str.upper()
        d = d[origin.str.startswith("W")]
    if not len(d):
        return None
    d = d.copy()
    d["visibility_weight"] = (1.0 - d["curiosity"]).round(2)
    g = d.groupby("agent_generation_time")["ideal_loss_cum"].mean()
    hi, lo = float(g.index.max()), float(g.index.min())
    return {
        "path": PRE_WELFARE_RUNS,
        "n_runs": int(len(d)),
        "corr_term": float(np.corrcoef(d["visibility_weight"],
                                       -d["ideal_merit_sum"])[0, 1]),
        "corr_cum": float(np.corrcoef(d["visibility_weight"],
                                      d["ideal_loss_cum"])[0, 1]),
        "corr_harvest": float(np.corrcoef(d["visibility_weight"],
                                          d["harvested_fin"])[0, 1]),
        "loss_gradient": float(g.loc[hi] - g.loc[lo]),
        "g_pair": (lo, hi),
    }


def _welfare_note(g: pd.DataFrame, corr: dict, gap: dict,
                  ref: dict | None, spread: dict) -> list[str]:
    """Provenance + qualification (i), every number computed from the cells."""
    grad = float(g.loc[g["g"] == max(WELFARE_GS), "cum_loss"].mean()
                 - g.loc[g["g"] == min(WELFARE_GS), "cum_loss"].mean())
    src = (f"source: {ADR_WELFARE_ROOT}/cells ({len(WELFARE_WEIGHTS) * 2 * 2 * len(WELFARE_GS)}"
           f" cells x 20 seeds = {len(g) * 80:,} runs; curiosity c x visibility-"
           f"reward toggle x landscape leg x g). {ADR_PINS}. "
           f"Model-internal, not calibrated (no empirical benchmark plotted).")
    lines = [src]
    cmp_txt = ""
    if ref is not None:
        cmp_txt = (f" (pre-ADR: corr {ref['corr_cum']:+.3f} / "
                   f"{ref['corr_harvest']:+.3f}, {ref['n_runs']:,} runs in "
                   f"{ref['path']})")
    lines.append(
        f"Visibility weight xi = 1 - c against cumulative ideal loss: "
        f"corr {corr['cum']:+.3f}; against per-capita harvest: corr "
        f"{corr['harvest']:+.3f}{cmp_txt}.")
    flip = ""
    if ref is not None:
        flip = (f", and {ref['loss_gradient']:+.3f} pre-ADR (g "
                f"{ref['g_pair'][1]:g} minus {ref['g_pair'][0]:g}: the sign "
                f"flips)")
    lines.append(
        f"Qualification (i), adr0003_remeasure_v1.md: the entry gradient of "
        f"cumulative ideal loss is {grad:+.3f} on the ADR-0003 clock (g "
        f"{max(WELFARE_GS):g} minus {min(WELFARE_GS):g}){flip} -- fixed-span "
        f"replacement removes the accumulation of long-lived depleted careers.")
    lines.append(
        f"Terminal loss converges across entry rates as visibility pressure "
        f"rises: the three g levels sit within {spread['rel_hi']:.1f}% of their "
        f"mean at xi={spread['w_hi']:g} and {spread['rel_mid']:.1f}% at "
        f"xi={spread['w_mid']:g}, and separate only at low pressure where the "
        f"loss is negligible (xi={spread['w_lo']:g}: {spread['lo_min']:.1e} to "
        f"{spread['lo_max']:.1e}), so panel (a) draws one pooled line with the "
        f"per-g means as open markers. Panel (b) is where g separates "
        f"(factor {spread['cum_ratio']:.1f} at xi={spread['w_hi']:g}). Both "
        f"landscape-leg tags (leg-2, leg-6) resolve to the same mgstd5 "
        f"landscape and are seed-identical in this archive, so they are folded "
        f"into one series; the visibility-reward toggle is averaged away (its "
        f"largest effect on any point is "
        f"{max(gap['term_loss'], gap['cum_loss']):.1e}).")
    return lines


def figure_welfare(outdir: Path | str | None = None,
                   data_root: Path | str | None = None,
                   collector: dict | None = None,
                   close: bool = False):
    """Draw F3 (Figure 2): welfare loss against visibility weight.

    Panel (a) plots the terminal ideal-merit loss, panel (b) the cumulative
    one; both are computed from the archive named by ``data_root``.
    """
    g = welfare_summary(data_root)
    gap = welfare_toggle_gap(data_root)
    ref = welfare_reference()
    d = welfare_cells(data_root)
    dedup = d[d["leg"] == 2] if set(d["leg"].unique()) == {2, 6} else d
    corr = {
        "term": float(np.corrcoef(dedup["visibility_weight"],
                                  dedup["term_loss"])[0, 1]),
        "cum": float(np.corrcoef(dedup["visibility_weight"],
                                 dedup["ideal_loss_cum"])[0, 1]),
        "harvest": float(np.corrcoef(dedup["visibility_weight"],
                                     dedup["harvested_fin"])[0, 1]),
    }
    # pooled terminal loss (mean over g) + how far apart the g levels run
    pooled_term = (g.groupby("visibility_weight")["term_loss"]
                   .agg(["mean", "min", "max"]))
    span = pooled_term["max"] - pooled_term["min"]
    rel = span / pooled_term["mean"]
    w_hi, w_mid, w_lo = (max(WELFARE_WEIGHTS), sorted(WELFARE_WEIGHTS)[-2],
                         min(WELFARE_WEIGHTS))
    cum_hi = g[g["visibility_weight"] == w_hi]["cum_loss"]
    spread = {
        "rel_hi": float(rel.loc[w_hi] * 100.0),
        "rel_mid": float(rel.loc[w_mid] * 100.0),
        "w_hi": w_hi,
        "w_mid": w_mid,
        "w_lo": w_lo,
        "lo_min": float(pooled_term.loc[w_lo, "min"]),
        "lo_max": float(pooled_term.loc[w_lo, "max"]),
        "cum_ratio": float(cum_hi.max() / cum_hi.min()),
    }
    note = _welfare_note(g, corr, gap, ref, spread)

    s = {**T.HOUSE_STYLE}
    colors = {v: T.HOUSE_STYLE["colors"][i]
              for i, v in enumerate(WELFARE_GS)}
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.6, 5.5), dpi=s["dpi"])

    # ---------------- (a) terminal loss, pooled over g -------------------
    for gv in WELFARE_GS:
        sub = g[g["g"] == gv]
        axA.plot(sub["visibility_weight"], sub["term_loss"], linestyle="none",
                 marker="o", markersize=5.5, markerfacecolor="white",
                 markeredgecolor=colors[gv], markeredgewidth=1.3, zorder=4,
                 label=f"g = {gv:g} (mean)")
    axA.plot(pooled_term.index, pooled_term["mean"], marker="s", markersize=5,
             linewidth=2.0, color=T.PALETTE["blue"], zorder=5,
             label="mean over g")
    T._cosmetics(axA, s,
                 f"(a) Terminal ideal-merit loss: corr(xi, loss) = "
                 f"{corr['term']:+.3f}",
                 "visibility weight xi = 1 - curiosity",
                 M_TERM)

    # ---------------- (b) cumulative loss, one line per g ----------------
    for gv in WELFARE_GS:
        sub = g[g["g"] == gv]
        axB.plot(sub["visibility_weight"], sub["cum_loss"], marker="o",
                 markersize=5, linewidth=1.8, color=colors[gv],
                 label=f"g = {gv:g}")
    T._cosmetics(axB, s,
                 f"(b) Cumulative ideal loss: corr(xi, loss_cum) = "
                 f"{corr['cum']:+.3f}",
                 "visibility weight xi = 1 - curiosity", M_CUM)

    for ax, col in ((axA, "term_loss"), (axB, "cum_loss")):
        ax.set_xlim(min(WELFARE_WEIGHTS) - 0.05, max(WELFARE_WEIGHTS) + 0.05)
        ax.set_xticks(WELFARE_WEIGHTS,
                      [f"{v:g}" for v in WELFARE_WEIGHTS],
                      fontsize=s["tick_size"])
        # headroom: the lowest points sit an order of magnitude above zero, so
        # the axis floor drops below zero instead of clipping them onto the
        # spine (visual_qa's 3% headroom rule)
        vmax = float(g[col].max())
        ax.set_ylim(-0.08 * vmax, 1.30 * vmax)
        lo, hi = ax.get_ylim()
        ax.set_yticks([t for t in ax.get_yticks()
                       if lo - 1e-12 <= t <= hi + 1e-12])
    axA.legend(fontsize=s["tick_size"] - 0.5, frameon=False, loc="upper left",
               labelspacing=0.25, handlelength=1.6)

    # qualification (i) callout: the flip, computed from the cells
    grad = float(g.loc[g["g"] == max(WELFARE_GS), "cum_loss"].mean()
                 - g.loc[g["g"] == min(WELFARE_GS), "cum_loss"].mean())
    flip_txt = (f"entry gradient of cumulative loss\n"
                f"g {max(WELFARE_GS):g} - g {min(WELFARE_GS):g} = "
                f"{grad:+.3f}")
    if ref is not None:
        flip_txt += f"\npre-ADR: {ref['loss_gradient']:+.3f} (sign flip)"
    axB.text(0.025, 0.975, flip_txt, transform=axB.transAxes, ha="left",
             va="top", fontsize=s["note_size"] - 0.5, zorder=6,
             bbox=dict(boxstyle="round,pad=0.32", fc="white",
                       ec=T.PALETTE["gray_light"], alpha=0.95))

    lines: list[str] = []
    for block in note:
        lines.extend(textwrap.fill(block, width=188).split("\n"))
    top, floor = 0.183, 0.014
    step = min(0.021, (top - floor) / max(len(lines), 1))
    y = top
    for line in lines:
        fig.text(0.008, y, line, fontsize=s["note_size"] - 1.5,
                 color=T.PALETTE["gray"])
        y -= step
    fig.subplots_adjust(left=0.068, right=0.988, top=0.915, bottom=0.300,
                        wspace=0.20)

    rows = []
    src = f"{ADR_WELFARE_ROOT}/cells/*.csv"
    # panel (a): one pooled line (statistic "mean") + the per-g markers
    for w, r in pooled_term.iterrows():
        rows.append(dict(
            figure="F3_welfare_loss_vs_visibility_weight", panel="a",
            metric=M_TERM, statistic="mean", series="mean over g",
            visibility_weight=float(w), g=np.nan, landscape="mgstd5",
            leg="2+6 (seed-identical)", curiosity=1.0 - float(w),
            use_visibility_reward="both (averaged)", value=float(r["mean"]),
            ci_lo=np.nan, ci_hi=np.nan, n=60, source_file=src,
            note="pooled terminal loss: mean over the three entry rates, the "
                 "visibility toggle and the duplicated landscape-leg tag"))
    for r in g.itertuples():
        rows.append(dict(
            figure="F3_welfare_loss_vs_visibility_weight", panel="a",
            metric=M_TERM, statistic="cell_mean", series=f"g = {r.g:.3f}",
            visibility_weight=float(r.visibility_weight), g=float(r.g),
            landscape="mgstd5", leg="2+6 (seed-identical)",
            curiosity=1.0 - float(r.visibility_weight),
            use_visibility_reward="both (averaged)",
            value=float(r.term_loss),
            ci_lo=float(r.term_loss - r.term_loss_ci),
            ci_hi=float(r.term_loss + r.term_loss_ci),
            n=20, source_file=src,
            note="per-g mean (open marker in panel a); seed mean folded over "
                 "the visibility toggle and the duplicated landscape-leg tag"))
    # panel (b): one line per g
    for r in g.itertuples():
        rows.append(dict(
            figure="F3_welfare_loss_vs_visibility_weight", panel="b",
            metric=M_CUM, statistic="mean", series=f"g = {r.g:.3f}",
            visibility_weight=float(r.visibility_weight), g=float(r.g),
            landscape="mgstd5", leg="2+6 (seed-identical)",
            curiosity=1.0 - float(r.visibility_weight),
            use_visibility_reward="both (averaged)",
            value=float(r.cum_loss),
            ci_lo=float(r.cum_loss - r.cum_loss_ci),
            ci_hi=float(r.cum_loss + r.cum_loss_ci),
            n=20, source_file=src,
            note="seed mean folded over the visibility toggle and the "
                 "duplicated landscape-leg tag; 95% CI = 1.96*SE over "
                 "the 20 per-seed values"))
    ann = [
        ("annotation: corr_xi_terminal_loss", corr["term"], M_TERM,
         f"run-level corr(1 - curiosity, -ideal_merit_sum) = {corr['term']:+.4f}"),
        ("annotation: corr_xi_cumulative_loss", corr["cum"], M_CUM,
         f"run-level corr(1 - curiosity, ideal_loss_cum) = {corr['cum']:+.4f}"),
        ("annotation: corr_xi_harvest", corr["harvest"], M_CUM,
         f"run-level corr(1 - curiosity, harvested_fin) = {corr['harvest']:+.4f}"),
        ("annotation: loss_gradient_high_minus_low_g", grad, M_CUM,
         f"mean ideal_loss_cum at g={max(WELFARE_GS):g} minus "
         f"g={min(WELFARE_GS):g} = {grad:+.4f} (qualification i)"),
        ("annotation: max_toggle_gap_terminal", gap["term_loss"], M_TERM,
         f"largest |visibility-toggle effect| on a plotted point = "
         f"{gap['term_loss']:.3e}"),
        ("annotation: max_toggle_gap_cumulative", gap["cum_loss"], M_CUM,
         f"largest |visibility-toggle effect| on a plotted point = "
         f"{gap['cum_loss']:.3e}"),
    ]
    if ref is not None:
        ann += [
            ("annotation: pre_adr_corr_xi_cumulative_loss", ref["corr_cum"],
             M_REF, f"pre-ADR corr(1 - c, ideal_loss_cum) = {ref['corr_cum']:+.4f}"),
            ("annotation: pre_adr_corr_xi_harvest", ref["corr_harvest"],
             M_REF, f"pre-ADR corr(1 - c, harvested_fin) = {ref['corr_harvest']:+.4f}"),
            ("annotation: pre_adr_loss_gradient", ref["loss_gradient"], M_REF,
             f"pre-ADR mean ideal_loss_cum at g={ref['g_pair'][1]:g} minus "
             f"g={ref['g_pair'][0]:g} = {ref['loss_gradient']:+.4f}"),
        ]
    for series, value, metric, text in ann:
        rows.append(dict(
            figure="F3_welfare_loss_vs_visibility_weight", panel="note",
            metric=metric, statistic="annotation" if metric != M_REF
            else "definitional constant", series=series,
            visibility_weight=np.nan, g=np.nan, landscape="", leg="",
            curiosity=np.nan, use_visibility_reward="", value=float(value),
            ci_lo=np.nan, ci_hi=np.nan, n=np.nan,
            source_file=src if metric != M_REF else PRE_WELFARE_RUNS,
            note=text))
    for cat, val, text in [
        ("weights", len(WELFARE_WEIGHTS), "visibility weights 0.1..0.9"),
        ("g_levels", len(WELFARE_GS), "entry rates 0.125/0.25/0.5"),
        ("cells", 60, "cell CSVs in the archive"),
        ("seeds_per_cell", 20, "seed rows per cell CSV"),
        ("tau_max", 80, "ADR-0003 career horizon in steps"),
        ("horizon", 200, "run horizon in steps"),
        ("gamma", 0.5 ** (1.0 / 10.0), "5-yr retention half-life"),
        ("z95", Z95, "95% CI multiplier (house convention)"),
    ]:
        rows.append(dict(
            figure="F3_welfare_loss_vs_visibility_weight", panel="ref",
            metric=M_REF, statistic="definitional constant", series=cat,
            visibility_weight=np.nan, g=np.nan, landscape="", leg="",
            curiosity=np.nan, use_visibility_reward="", value=float(val),
            ci_lo=np.nan, ci_hi=np.nan, n=np.nan, source_file="ADR-0003 pins",
            note=text))
    sidecar = pd.DataFrame(rows)

    specs = [
        {"ax": axA, "metric": M_TERM,
         "expect_lines": len(WELFARE_GS) + 1,
         "points_per_line": len(WELFARE_WEIGHTS), "sidecar": sidecar,
         "filters": {"panel": "a"}, "ci_cols": ("value", "value")},
        {"ax": axB, "metric": M_CUM, "expect_lines": len(WELFARE_GS),
         "points_per_line": len(WELFARE_WEIGHTS), "sidecar": sidecar,
         "filters": {"panel": "b"}, "ci_cols": ("value", "value")},
    ]
    report = visual_qa.assert_figure_clean(fig, specs)

    out = _outdir(outdir) / "F3_welfare_loss_vs_visibility_weight.png"
    fig.savefig(out, dpi=s["dpi"])
    fig.savefig(out.with_suffix(".pdf"))
    csv_path = T.write_figure_data(sidecar, out)
    if collector is not None:
        collector["fig"] = fig
        collector["sidecar"] = sidecar
    if close:
        plt.close(fig)
    print(f"F3: {out} | {out.with_suffix('.pdf')} | {csv_path} | "
          f"visual_qa failures: {report['_total']}")
    return fig, out



# --------------------------------------------------------------------------
# F4 -- W3-A prestige stock: the wedge matrix, entry-on cells only
# --------------------------------------------------------------------------
def figure_prestige_stock(outdir: Path | str | None = None):
    d = D.prestige_stock_summary()
    d = d[d["agent_generation_time"] > 0].copy()
    _require = D._require
    _require(len(d) == 18,
             f"entry-on prestige-stock cells=18 (6 x 3 stocks), got {len(d)}")
    d["cell"] = ("cur" + d["initial_curiosity"].map(lambda v: f"{v:.1f}")
                 + " " + d["landscape"].map(
                     lambda s: "mg" if s == "mgstd5" else "noisy"))
    d["stock"] = d["stock"].map({
        "prestige": "prestige (raw)",
        "prestige_vanishing": "prestige_vanishing",
        "prestige_visibility": "prestige_visibility",
    })
    out = _outdir(outdir) / "F4_prestige_stock_wedge_matrix.png"
    fig, ax = T.styled_barchart(
        d,
        x="cell",
        y="wedge_fin_mean",
        group="stock",
        title="The terminal wedge barely moves with the prestige stock",
        xlabel="entry 0.1 cell (curiosity x landscape legibility)",
        ylabel="relational wedge (c_local - c_ideal), final step",
        source=SOURCE_WA,
        value_fmt="{:.2f}",
        outpath=out,
    )
    return fig, out


# --------------------------------------------------------------------------
# F5 -- N1/N2 re-measure: what the pairing audit moved, and what it did not
# --------------------------------------------------------------------------
def figure_remeasure_delta(outdir: Path | str | None = None):
    d, counts = D.remeasure_pairs()
    d = d.copy()
    d["reporter"] = d["reporter"].map({
        "c_ideal_fin": "corr(prestige, ideal) final",
        "c_ideal_tm": "corr(prestige, ideal) time-mean",
        "wedge_fin": "wedge final",
        "wedge_tm": "wedge time-mean",
        "top10_fin": "top-10% share",
        "c_local_fin": "corr(prestige, local) final",
        "c_local_tm": "corr(prestige, local) time-mean",
    })
    # first row at the top: movers first, anchors last
    order = ["corr(prestige, ideal) final", "corr(prestige, ideal) time-mean",
             "wedge final", "wedge time-mean", "top-10% share",
             "corr(prestige, local) final", "corr(prestige, local) time-mean"]
    d["reporter"] = pd.Categorical(d["reporter"], categories=order,
                                   ordered=True)
    d = d.sort_values("reporter")
    arms = list(dict.fromkeys(d["arm"]))          # appearance order
    two_arms = len(arms) > 1
    out = _outdir(outdir) / "F5_pairing_audit_paired_delta.png"
    n_pairs = int(d.groupby("arm")["n"].max().sum())
    fig, ax = T.styled_coefplot(
        d,
        y="reporter",
        x="delta",
        xerr="se",
        group="arm" if two_arms else None,
        level=0.95,
        annotate=not two_arms,
        value_fmt="{:+.4f}",
        title="Pairing audit moved the ideal-side levels; local side untouched",
        xlabel=f"mean paired delta (post-fix minus pre-fix), 95% CI over "
               f"{n_pairs} seed-matched runs",
        source=(f"arms: {', '.join(arms)} | {SOURCE_RM}"),
        outpath=out,
    )
    return fig, out, counts



# --------------------------------------------------------------------------
# F7 -- M0 cognitive division of labour: diversity buys alignment, not speed
# --------------------------------------------------------------------------
F7_STEM = "F7_v2_m0_cognitive_division_of_labor"

#: Arms in appearance order: the two substantive cohorts only. The pooled
#: sweep reference was removed in the v2 streamline (owner brief
#: 2026-11-19): an average over populations is not a cohort, and Panel A
#: must read as the matched mixed-vs-homogeneous comparison it is.
F7_ARMS_A = ("mixed", "homogeneous")
F7_ARMS_B = ("mixed", "homogeneous")


def _f7_colors(s: dict) -> dict:
    return {"mixed": T.PALETTE["purple"], "homogeneous": T.PALETTE["orange"],
            "sweep average": T.PALETTE["gray_light"]}


def _f7_panel(ax, wide: pd.DataFrame, lands: list[str], arms: tuple[str, ...],
              s: dict, *, fmt: str, ylabel: str, title: str,
              tick_second: dict[str, str] | None = None) -> None:
    """One grouped-bar panel, drawn from an already validated wide frame.

    The templates build one axes per figure, so a two-panel file composes them
    from the template's own parts: ``_wide`` (duplicate/gap/level checks),
    ``_cosmetics`` (title, spines, grid) and ``HOUSE_STYLE`` (sizes, dpi).
    """
    colors = _f7_colors(s)
    n_grp = len(arms)
    width = s["bar_width"] / n_grp
    x = np.arange(len(lands))
    tops = []
    for j, arm in enumerate(arms):
        vals = wide[arm].reindex(lands).to_numpy(float)
        off = (j - (n_grp - 1) / 2) * width
        hatch = "//" if arm == "sweep average" else None
        bars = ax.bar(x + off, vals, width=width * 0.94,
                      color=colors[arm], hatch=hatch,
                      edgecolor=T.PALETTE["gray"] if hatch else "none",
                      label=arm)
        ax.bar_label(bars, fmt=lambda v: fmt.format(v), fontsize=s["note_size"],
                     padding=2)
        # 95 % CI of the run-level mean, drawn only where the frame carries one
        ci = wide.attrs.get(arm)
        if ci is not None:
            lo, hi = ci
            ax.errorbar(x + off, vals, yerr=[vals - lo, hi - vals], fmt="none",
                        ecolor=T.PALETTE["gray"], elinewidth=1.0, capsize=2.4)
        tops.append(np.nanmax(vals))
    ax.set_xticks(x, [f"{land}\n{tick_second.get(land, '')}" if tick_second
                      else land for land in lands], fontsize=s["tick_size"])
    T._cosmetics(ax, s, title, "", ylabel)
    hi = float(np.nanmax(tops)) * 1.28
    ax.set_ylim(0.0, hi if hi > 0 else 1.0)
    ax.legend(fontsize=s["tick_size"], frameon=False, loc="upper right")


def figure_cognitive_division_of_labor(outdir: Path | str | None = None):
    """F7_v2 -- the two M0 claims of the sketch, drawn against their artifacts.

    Panel A is the sketch's "early breakthrough acceleration" claim and Panel B
    its "merit-prestige alignment" claim, both measured on the same four
    landscape pairs so the reader can see that the second survives and the
    first does not: the matched comparison (mixed vs the swept file's mu=0.5
    cohort) shows no speed-up, while the alignment gain is large on MultG2 and
    reverses on Noisy2. v2 (owner brief 2026-11-19) removes the pooled sweep
    average from Panel A -- only `mixed` vs `homogeneous` remain, matching
    Panel B -- and asserts exactly two bars per landscape.
    """
    tidy, caps = D.m0_division_of_labor()
    lands = list(D.M0_FILES)
    s = {**T.HOUSE_STYLE, "figsize": (11.4, 5.0)}
    out = _outdir(outdir) / f"{F7_STEM}.png"

    fig, (axA, axB) = plt.subplots(1, 2, figsize=s["figsize"])

    # ---- Panel A: steps to 10 % harvested --------------------------------
    a = tidy[tidy["metric"] == "h10_steps"]
    wide_a = T._wide(a.rename(columns={"value": "h10_steps"}),
                     "landscape", "h10_steps", "arm")
    for arm in F7_ARMS_A:
        sub = a[a["arm"] == arm].set_index("landscape").reindex(lands)
        wide_a.attrs[arm] = (sub["ci_lo"].to_numpy(float),
                             sub["ci_hi"].to_numpy(float))
    verdict_a = ("no speed-up from a mixed cohort"
                 if caps["speed_claim"] == "no acceleration"
                 else f"mixed cohort {caps['speed_claim']}")
    _f7_panel(axA, wide_a, lands, F7_ARMS_A, s, fmt="{:.1f}",
              ylabel="steps to 10 % knowledge harvested",
              title=f"A · Harvest speed: {verdict_a}",
              tick_second={land: f"×{caps['ratios'][land]['homog_over_mixed']:.2f}"
                           for land in lands})

    # ---- Panel B: merit-prestige alignment -------------------------------
    b = tidy[tidy["metric"] == "c_ideal"]
    wide_b = T._wide(b.rename(columns={"value": "c_ideal"}),
                     "landscape", "c_ideal", "arm")
    for arm in F7_ARMS_B:
        sub = b[b["arm"] == arm].set_index("landscape").reindex(lands)
        wide_b.attrs[arm] = (sub["ci_lo"].to_numpy(float),
                             sub["ci_hi"].to_numpy(float))
    _f7_panel(axB, wide_b, lands, F7_ARMS_B, s, fmt="{:+.3f}",
              ylabel="Corr(prestige, ideal merit), run mean",
              title=f"B · Alignment: diversity helps in "
                    f"{caps['align_better']} of {caps['n_landscapes']} landscapes",
              tick_second={land: f"Δ {caps['deltas'][land]:+.3f}"
                           for land in lands})

    note = textwrap.fill(
        "source: data/figures_v1/run_number0-7.csv — the 8 canonical BULK_SETUPS, "
        f"{D.M0_REPEATS} repeats per cell, step_limit 200. mixed: newcomers draw "
        "c ~ U(0,1) (files 1/3/5/7). homogeneous: the swept file (0/2/4/6) at "
        "μ = 0.5, where newcomers inherit c = 0.5. Both panels compare only "
        "these two matched cohorts (v2: the pooled 11-point sweep reference "
        "was removed — an average over populations is not a cohort). "
        "× under a landscape = steps ratio "
        "homogeneous/mixed; Δ = mixed minus homogeneous. No MultG2_no_dist artifact "
        "exists in the repo (no sweep ran use_distance=False), so the fourth pair is "
        "randominit. Model-internal, no empirical calibration.",
        width=176)
    fig.text(0.006, 0.012, note, fontsize=s["note_size"], color=T.PALETTE["gray"],
             va="bottom", linespacing=1.45)
    fig.tight_layout(rect=(0, 0.195, 1, 1))

    # v2 contract: Panel A carries exactly two bars per landscape
    import visual_qa
    n_bars_a = sum(len(bc.patches) for bc in
                   visual_qa.collect_panel_artists(axA)["bars"])
    assert n_bars_a == 2 * len(lands), (
        f"Panel A must show 2 bars per landscape, found {n_bars_a} "
        f"for {len(lands)} landscapes")

    fig.savefig(out, dpi=s["dpi"])
    pdf = out.with_suffix(".pdf")
    fig.savefig(pdf)

    export = tidy[tidy["metric"].isin(["h10_steps", "c_ideal", "harvest_pct"])].copy()
    export["kind"] = "drawn"
    extra = []
    for land in lands:
        r = caps["ratios"][land]
        extra += [
            {"landscape": land, "arm": "homogeneous/mixed", "metric": "ratio_h10_steps",
             "value": r["homog_over_mixed"], "kind": "annotated"},
            {"landscape": land, "arm": "mixed-homogeneous", "metric": "delta_c_ideal",
             "value": caps["deltas"][land], "kind": "annotated"},
            {"landscape": land, "arm": "mixed-homogeneous", "metric": "harvest_gap_pp",
             "value": caps["harvest_gap_pp"][land], "kind": "annotated"},
        ]
    export = pd.concat([export, pd.DataFrame(extra)], ignore_index=True)
    csv_path = T.write_figure_data(export, out)

    # v2 visual QA on the rendered canvas (drawn arms only)
    qa_a = export[(export["metric"] == "h10_steps")
                  & (export["arm"].isin(F7_ARMS_A))].copy()
    qa_a["metric"] = "f7a"
    qa_b = export[(export["metric"] == "c_ideal")
                  & (export["arm"].isin(F7_ARMS_B))].copy()
    qa_b["metric"] = "f7b"
    visual_qa.assert_figure_clean(fig, [
        {"ax": axA, "metric": "f7a", "expect_bars": 8, "sidecar": qa_a},
        {"ax": axB, "metric": "f7b", "expect_bars": 8, "sidecar": qa_b},
    ])
    return fig, out, {"caps": caps, "csv": csv_path, "pdf": pdf}


# --------------------------------------------------------------------------
# F10 -- Tier M2: the pull narrows the wedge and binds team size
# --------------------------------------------------------------------------
SOURCE_M2 = ("source: data/path_b_v2/lambda_trend.csv (48 cells x 20 seeds, "
             "pull in {0.5, 1.0}) and data/path_b_v1/cells (the pull = 0 arm, "
             "40 cells x 20 seeds)")

#: Klann panel stylized benchmark, the paper's unfitted reference line.
TEAM_BENCHMARK = 2.65


def _m2_pull0() -> pd.DataFrame:
    """The pull = 0 arm, rebuilt seed-by-seed from the Path B v1 cells.

    The v2 grid sweeps the pull at 0.5 and 1.0 only; ``pull = 0`` is the v1
    archive, whose lambda lives in the filename. Rebuilding it here (rather
    than reading v1's wide metrics table) keeps one code path for both arms
    and gives the per-seed values the export carries.
    """
    frames = []
    for path in sorted((REPO / "data" / "path_b_v1" / "cells").glob("cell_lam*_*.csv")):
        m = re.match(
            r"cell_lam(\d{3})_(logit|threshold)_(struct|equal)_(\w+)\.csv$",
            path.name,
        )
        if not m:
            continue
        d = pd.read_csv(path)
        d["dyadic_lambda"] = int(m.group(1)) / 100.0
        d["formulation"] = m.group(2)
        d["credit_sharing_axis"] = "structural" if m.group(3) == "struct" else "equal_split"
        d["landscape"] = m.group(4)
        frames.append(d)
    if not frames:
        raise SystemExit("F10: no Path B v1 cells found for the pull=0 arm")
    out = pd.concat(frames, ignore_index=True)
    out["dyadic_move_pull"] = 0.0
    return out


def _m2_v2_cells() -> pd.DataFrame:
    """The v2 arm at seed grain, so both arms carry the same 80 seeds/cell-mean.

    Reading ``lambda_trend.csv`` instead would give four CELL means per point
    and an error bar over 4 observations, against 80 seeds for the v1 arm: two
    different bases drawn on one axis. The cells are the same runs the trend
    table aggregates, so the means agree (asserted in ``main``).
    """
    frames = []
    for path in sorted((REPO / "data" / "path_b_v2" / "cells").glob("cell_lam*_pull*.csv")):
        frames.append(pd.read_csv(path))
    if not frames:
        raise SystemExit("F10: no Path B v2 cells found")
    return pd.concat(frames, ignore_index=True)


def _m2_cells() -> pd.DataFrame:
    """Both arms stacked at seed grain, with the wedge attached."""
    both = pd.concat([_m2_pull0(), _m2_v2_cells()], ignore_index=True)
    both["wedge"] = both["c_local_tm"] - both["c_ideal_tm"]
    return both


def figure_m2_wedge_and_team_size(outdir: Path | str | None = None):
    """F10: panel A the prestige wedge vs the pull, panel B team size at pull=1.

    Panel A is drawn on ``mgstd5`` alone. That is not a convenience: the
    narrowing is a property of that landscape. On ``noisypr5`` the lambda=1
    wedge stays positive at every pull, so pooling the two would cancel the
    very sign flip the panel exists to show. Panel B pools the two landscapes,
    as the paper's Figure 7 does, and its export carries the per-landscape
    components so the pooling is recoverable.
    """
    s = T.HOUSE_STYLE
    out = _outdir(outdir) / "F10_m2_wedge_and_team_size.png"
    cells = _m2_cells()

    # ---- Panel A: the wedge on mgstd5 across the pull ---------------------
    a = (
        cells[(cells["landscape"] == "mgstd5") & cells["dyadic_lambda"].isin([0.0, 1.0])]
        .groupby(["dyadic_move_pull", "dyadic_lambda"], as_index=False)
        .agg(mean=("wedge", "mean"), sem=("wedge", "sem"),
             n=("wedge", "size"))
    )
    a["series"] = a["dyadic_lambda"].map({0.0: "lambda=0 (prestige matching)",
                                          1.0: "lambda=1 (epistemic complementarity)"})

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.4, 4.7))
    for j, lam in enumerate((0.0, 1.0)):
        sub = a[a["dyadic_lambda"] == lam].sort_values("dyadic_move_pull")
        axA.errorbar(
            sub["dyadic_move_pull"], sub["mean"], yerr=sub["sem"],
            marker="o", markersize=5, linewidth=1.8, capsize=3,
            color=T.PALETTE["purple" if lam == 0.0 else "orange"],
            label=sub["series"].iloc[0],
        )
    axA.axhline(0.0, color=T.PALETTE["gray"], linewidth=0.9, linestyle=":")
    T._cosmetics(
        axA, s,
        "A - the pull narrows the prestige wedge",
        "dyadic_move_pull (spatial co-movement toward the partner)",
        "relational wedge  $c_{local} - c_{ideal}$",
    )
    axA.legend(fontsize=s["tick_size"], frameon=False, loc="lower left")
    # the two endpoints the text quotes, read off the frame, never typed
    for lam, col in ((0.0, T.PALETTE["purple"]), (1.0, T.PALETTE["orange"])):
        sub = a[(a["dyadic_lambda"] == lam)].sort_values("dyadic_move_pull")
        last = sub.iloc[-1]
        axA.annotate(
            f"{last['mean']:+.3f}",
            (last["dyadic_move_pull"], last["mean"]),
            textcoords="offset points", xytext=(6, 4 if lam == 0.0 else -12),
            fontsize=s["tick_size"], color=col,
        )

    # ---- Panel B: team size at pull = 1 -----------------------------------
    b_all = cells[(cells["dyadic_move_pull"] == 1.0) & cells["dyadic_lambda"].isin([0.0, 1.0])]
    b = (
        b_all.groupby(["formulation", "credit_sharing_axis", "dyadic_lambda"],
                      as_index=False)["team_tm"].mean()
    )
    groups = [
        ("logit", "structural"), ("logit", "equal_split"),
        ("threshold", "structural"), ("threshold", "equal_split"),
    ]
    labels = [f"{f}\n{c.replace('_', ' ')}" for f, c in groups]
    x = np.arange(len(groups))
    width = 0.36
    bars_by_lam = {}
    for j, (lam, key, lab) in enumerate(
        ((0.0, "purple", "lambda=0 (prestige matching)"),
         (1.0, "orange", "lambda=1 (epistemic complementarity)"))
    ):
        vals = [
            float(b[(b["formulation"] == f) & (b["credit_sharing_axis"] == c)
                     & (b["dyadic_lambda"] == lam)]["team_tm"].iloc[0])
            for f, c in groups
        ]
        bars = axB.bar(
            x + (j - 0.5) * width, vals, width,
            color=T.PALETTE[key], label=lab,
        )
        bars_by_lam[lam] = bars
        if s["value_label"]:
            for rect, v in zip(bars, vals):
                axB.text(
                    rect.get_x() + rect.get_width() / 2, v + 0.06, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=s["tick_size"] - 1,
                )
    axB.axhline(TEAM_BENCHMARK, ls="--", lw=1.1, color=T.PALETTE["gray"],
                label=f"Empirical benchmark (unfitted): {TEAM_BENCHMARK:.2f}")
    # the benchmark is annotated through the legend, in the clear band above
    # the logit bars: a floating label at y=2.65 sits inside the threshold bars
    # and needs a white box punched through the data to stay readable.
    axB.set_ylim(0, 6.0)
    # the cell the paper's prose quotes, located in the frame rather than
    # assumed: outlined, not re-annotated -- the bar already carries its own
    # value label and a second "2.95" stacked above it reads as two numbers.
    tgt_idx = groups.index(("threshold", "equal_split"))
    tgt = b[(b["formulation"] == "threshold") & (b["credit_sharing_axis"] == "equal_split")
            & (b["dyadic_lambda"] == 1.0)]["team_tm"].iloc[0]
    bars_by_lam[1.0][tgt_idx].set_edgecolor(T.PALETTE["gray"])
    bars_by_lam[1.0][tgt_idx].set_linewidth(1.3)
    axB.set_xticks(x, labels, fontsize=s["tick_size"])
    T._cosmetics(
        axB, s,
        "B - team size converges at full pull",
        "dyadic_mode x credit_sharing (pull = 1.0)",
        "mean co-harvest team size",
    )
    h, l = axB.get_legend_handles_labels()
    order = [i for i in range(len(l)) if not l[i].startswith("Empirical")] + \
            [i for i in range(len(l)) if l[i].startswith("Empirical")]
    axB.legend([h[i] for i in order], [l[i] for i in order],
               fontsize=s["tick_size"] - 1, frameon=False, loc="upper left",
               labelspacing=0.3, handlelength=1.6)

    split = (b_all.groupby(["landscape", "dyadic_lambda"])["team_tm"].mean()
             .unstack("dyadic_lambda"))
    note = textwrap.fill(
        f"A: mgstd5 only. On noisypr5 the lambda=1 wedge never crosses zero, so "
        f"pooling would hide the sign flip. Points, bars: mean; whiskers: s.e.m. "
        f"over 80 seeds (4 cells x 20). pull = 0 is the Path B v1 archive, where "
        f"lambda changes links but not movement, so the two lines coincide there "
        f"by construction. B: pooled over mgstd5 and noisypr5 as in Figure 7 - "
        f"lambda=1 team size is {split.loc['mgstd5', 1.0]:.2f} on mgstd5 against "
        f"{split.loc['noisypr5', 1.0]:.2f} on noisypr5 (lambda=0: "
        f"{split.loc['mgstd5', 0.0]:.2f} / {split.loc['noisypr5', 0.0]:.2f}); the "
        f"per-landscape values are in the _data.csv. Benchmark 2.65 is the Klann "
        f"stylized moment, a diagnostic and never a fitting target. " + SOURCE_M2 +
        " Model-internal, not calibrated.",
        width=198,
    )
    fig.text(0.006, 0.010, note, fontsize=s["note_size"], color=T.PALETTE["gray"],
             va="bottom", linespacing=1.5)
    fig.tight_layout(rect=(0, 0.125, 1, 1))
    fig.savefig(out, dpi=s["dpi"])
    pdf = out.with_suffix(".pdf")
    fig.savefig(pdf)

    export = pd.concat(
        [
            a.assign(panel="A", metric="wedge", scope="mgstd5",
                     formulation="pooled", credit_sharing_axis="pooled",
                     landscape="mgstd5", kind="drawn")
            [["panel", "scope", "metric", "dyadic_move_pull", "dyadic_lambda",
              "formulation", "credit_sharing_axis", "landscape", "mean", "sem", "n", "kind"]]
            .rename(columns={"mean": "value"}),
            b_all.assign(panel="B", metric="team_tm", scope="pooled", kind="drawn",
                         landscape="pooled")
            .groupby(["panel", "scope", "metric", "dyadic_move_pull", "dyadic_lambda",
                      "formulation", "credit_sharing_axis", "landscape", "kind"],
                     as_index=False)["team_tm"].mean()
            .rename(columns={"team_tm": "value"}),
            b_all.assign(panel="B", metric="team_tm", scope="by_landscape", kind="component")
            .groupby(["panel", "scope", "metric", "dyadic_move_pull", "dyadic_lambda",
                      "formulation", "credit_sharing_axis", "landscape", "kind"],
                     as_index=False)["team_tm"].mean()
            .rename(columns={"team_tm": "value"}),
        ],
        ignore_index=True,
    )
    csv_path = T.write_figure_data(export, out)
    return fig, out, {"csv": csv_path, "pdf": pdf, "target_cell": float(tgt)}


BUILDERS = [
    ("F1", figure_sensitivity, "T1 sensitivity (post-audit arm)"),
    ("F2", figure_career_flows, "T3 career flows (clean + noisy)"),
    ("F2b", figure_career_flows_ideal_corr, "T3 slow-entry gradient"),
    ("F3", figure_welfare, "T4 welfare probe"),
    ("F4", figure_prestige_stock, "W3-A prestige stock"),
    ("F5", figure_remeasure_delta, "N1/N2 re-measure"),
    ("F7", figure_cognitive_division_of_labor, "M0 cognitive division of labour"),
    ("F10", figure_m2_wedge_and_team_size, "M2 dyadic choice (Path B v1+v2)"),
]


def build_all(outdir: Path | str | None = None) -> pd.DataFrame:
    """Draw every figure once; return the gross/net ledger of PNGs."""
    rows = []
    for tag, fn, round_name in BUILDERS:
        res = fn(outdir)
        fig, out = res[0], res[1]
        rows.append({"figure": tag, "round": round_name,
                     "png": str(out.relative_to(REPO)),
                     "bytes": out.stat().st_size})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# F3 verifier: recompute every plotted value from the cells, independently of
# the builder, and re-run the visual QA contract on a fresh canvas.
# --------------------------------------------------------------------------
F3_FILENAME = "F3_welfare_loss_vs_visibility_weight"


def verify_welfare(outdir: Path | str | None = None,
                   data_root: Path | str | None = None,
                   tol: float = 1e-9) -> list[str]:
    """Self-contained check of the F3 sidecar against the raw cells."""
    import tempfile

    fails: list[str] = []
    out = Path(outdir) if outdir else OUT
    if not out.is_absolute():
        out = REPO / out
    csv_path = out / f"{F3_FILENAME}_data.csv"
    root = welfare_root(data_root)
    files = sorted((root / "cells").glob("*.csv"))
    if len(files) != 60:
        return [f"W_v05 gross cells=60 expected, found {len(files)}"]
    frames = []
    for f in files:
        d = pd.read_csv(f)
        if len(d) != 20:
            fails.append(f"{f.name}: 20 seed rows expected, found {len(d)}")
        frames.append(d)
    if fails:
        return fails
    d = pd.concat(frames, ignore_index=True)
    if len(d) != 1200:
        fails.append(f"W_v05 runs: 1200 expected, found {len(d)}")
    if sorted(d["curiosity"].unique().tolist()) != WELFARE_WEIGHTS:
        fails.append("W_v05 curiosity grid is not 0.1..0.9 by 0.2")
    if sorted(d["g"].unique().tolist()) != WELFARE_GS:
        fails.append("W_v05 g grid is not the ADR-0003 grid")
    d = d.copy()
    d["w"] = 1.0 - d["curiosity"]
    d["term"] = -d["ideal_merit_sum"]

    # structural claims carried by the figure
    corr_cum = float(np.corrcoef(d["w"], d["ideal_loss_cum"])[0, 1])
    corr_harv = float(np.corrcoef(d["w"], d["harvested_fin"])[0, 1])
    if not corr_cum > 0.5:
        fails.append(f"visibility-pressure claim broken: corr(xi, loss_cum) "
                     f"{corr_cum:+.4f} is not clearly positive")
    if not corr_harv < -0.4:
        fails.append(f"visibility-pressure claim broken: corr(xi, harvest) "
                     f"{corr_harv:+.4f} is not clearly negative")
    gmean = d.groupby("g")["ideal_loss_cum"].mean()
    grad = float(gmean.loc[max(WELFARE_GS)] - gmean.loc[min(WELFARE_GS)])
    if not grad < 0:
        fails.append(f"qualification (i) broken: cumulative-loss entry "
                     f"gradient {grad:+.4f} does not fall with g")

    exp = (d.groupby(["w", "g"], as_index=False)
           .agg(term=("term", "mean"), cum=("ideal_loss_cum", "mean")))
    exp_pooled = d.groupby("w")["term"].mean()
    if not csv_path.exists():
        return fails + [f"sidecar missing: {csv_path}"]
    full = pd.read_csv(csv_path)
    got = full[full["statistic"].isin(["mean", "cell_mean"])]
    n_expect = len(WELFARE_WEIGHTS) + 2 * len(WELFARE_WEIGHTS) * len(WELFARE_GS)
    if len(got) != n_expect:
        fails.append(f"sidecar plotted rows: {n_expect} expected, "
                     f"found {len(got)}")
        return fails

    # panel (a): the pooled line
    pool = got[(got["panel"] == "a") & (got["statistic"] == "mean")]
    if len(pool) != len(WELFARE_WEIGHTS):
        fails.append(f"panel a: {len(WELFARE_WEIGHTS)} pooled points "
                     f"expected, found {len(pool)}")
    for w, ref in exp_pooled.items():
        row = pool[np.isclose(pool["visibility_weight"], w)]
        if len(row) != 1:
            fails.append(f"panel a: no unique pooled row for w={w:g}")
            continue
        v = float(row["value"].iloc[0])
        if abs(v - float(ref)) > tol:
            fails.append(f"panel a pooled w={w:g}: sidecar {v!r} != "
                         f"recomputed {float(ref)!r}")
    # panel (a): the per-g markers
    cells_a = got[(got["panel"] == "a") & (got["statistic"] == "cell_mean")]
    if len(cells_a) != len(WELFARE_WEIGHTS) * len(WELFARE_GS):
        fails.append(f"panel a: 15 per-g markers expected, "
                     f"found {len(cells_a)}")
    # panel (b): one line per g
    cells_b = got[(got["panel"] == "b") & (got["statistic"] == "mean")]
    if len(cells_b) != len(WELFARE_WEIGHTS) * len(WELFARE_GS):
        fails.append(f"panel b: 15 plotted points expected, "
                     f"found {len(cells_b)}")
    if not (cells_b["metric"] == M_CUM).all():
        fails.append("panel b: metric label mismatch")
    if not (cells_a["metric"] == M_TERM).all():
        fails.append("panel a: metric label mismatch")

    for sub, col, panel in ((cells_a, "term", "a"), (cells_b, "cum", "b")):
        for r in exp.itertuples():
            row = sub[(np.isclose(sub["visibility_weight"], r.w))
                      & (np.isclose(sub["g"], r.g))]
            if len(row) != 1:
                fails.append(f"panel {panel}: no unique row for "
                             f"w={r.w}/g={r.g}")
                continue
            v = float(row["value"].iloc[0])
            ref = float(getattr(r, col))
            if abs(v - ref) > tol:
                fails.append(f"panel {panel} w={r.w}/g={r.g}: sidecar {v!r} "
                             f"!= recomputed {ref!r}")
            lo, hi = float(row["ci_lo"].iloc[0]), float(row["ci_hi"].iloc[0])
            if not (lo <= v <= hi):
                fails.append(f"panel {panel} w={r.w}/g={r.g}: CI does not "
                             f"bracket the point")
            if int(row["n"].iloc[0]) != 20:
                fails.append(f"panel {panel} w={r.w}/g={r.g}: n must be 20")

    # the rendered callout must carry the recomputed gradient
    ann = full[(full["statistic"] == "annotation")
               & (full["series"] == "annotation: loss_gradient_high_minus_low_g")]
    if len(ann) != 1:
        fails.append("sidecar: the qualification-(i) gradient annotation is "
                     "missing")
    elif abs(float(ann["value"].iloc[0]) - grad) > tol:
        fails.append(f"gradient annotation {float(ann['value'].iloc[0])!r} "
                     f"!= recomputed {grad!r}")

    # visual QA re-run on a fresh canvas, plus a sidecar round-trip
    tmp = Path(tempfile.mkdtemp(prefix="f3_"))
    col: dict = {}
    figure_welfare(tmp, data_root=data_root, collector=col, close=False)
    try:
        visual_qa.assert_figure_clean(col["fig"], [
            {"ax": ax, "metric": m,
             "expect_lines": len(WELFARE_GS) + (1 if p == "a" else 0),
             "points_per_line": len(WELFARE_WEIGHTS), "sidecar": full,
             "filters": {"panel": p}, "ci_cols": ("value", "value")}
            for ax, m, p in zip(col["fig"].axes, (M_TERM, M_CUM), ("a", "b"))
        ])
    except AssertionError as exc:
        fails.append(f"visual QA re-run failed: {exc}")
    regot = pd.read_csv(tmp / f"{F3_FILENAME}_data.csv")
    regot = regot[regot["statistic"].isin(["mean", "cell_mean"])]
    if len(regot) != len(got):
        fails.append(f"regenerated sidecar has {len(regot)} plotted rows, "
                     f"expected {len(got)}")
    else:
        key = ["panel", "statistic", "series", "visibility_weight", "g"]
        j = got.merge(regot, on=key, suffixes=("_a", "_b"))
        if len(j) != len(got):
            fails.append("regenerated sidecar rows do not pair 1:1 with the "
                         "published sidecar")
        dv = np.abs(j["value_a"].to_numpy(float) - j["value_b"].to_numpy(float))
        if not np.all(np.nan_to_num(dv) <= tol):
            fails.append("regenerated sidecar values differ from the rebuild")
    plt.close(col["fig"])
    return fails


if __name__ == "__main__":  # pragma: no cover
    # F3 ONLY: building the whole module would rewrite every legacy figure.
    _fig, _out = figure_welfare(close=False)
    _fails = verify_welfare()
    print("verify_F3_welfare: " +
          ("all checks passed" if not _fails else f"{len(_fails)} FAIL(s)"))
    for _f in _fails:
        print(f"FAIL {_f}")
    raise SystemExit(1 if _fails else 0)

