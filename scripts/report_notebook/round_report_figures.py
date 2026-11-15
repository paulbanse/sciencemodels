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
import textwrap

import diagram_templates as T
import round_report_data as D

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
        ylabel="corr(prestige, ideal merit), final step",
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
        ylabel="corr(prestige, ideal merit), final step",
        source=SOURCE_T3,
        outpath=out,
    )
    return fig, out


# --------------------------------------------------------------------------
# F3 -- T4 welfare: the price of visibility pressure, and its knee
# --------------------------------------------------------------------------
def figure_welfare(outdir: Path | str | None = None):
    d = D.welfare_summary().copy()
    d["loss_pos"] = -d["loss"]            # ideal_Merit is negative: loss = -sum
    out = _outdir(outdir) / "F3_welfare_loss_vs_visibility_weight.png"
    worst_gap = float(d["toggle_gap"].max())
    fig, ax = T.styled_linechart(
        d,
        x="visibility_weight",
        y="loss_pos",
        group="series",
        title=(f"Visibility pressure raises loss; toggle gap "
               f"at most {worst_gap:.1e}"),
        xlabel="visibility weight (1 - curiosity)",
        ylabel="ideal-merit loss (sum over agents, positive)",
        source=SOURCE_T4,
        outpath=out,
    )
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
        ylabel="wedge = corr(local) - corr(ideal), final step",
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
F7_STEM = "F7_m0_cognitive_division_of_labor"

#: Arms in appearance order: the two substantive cohorts, then the pooled
#: sweep reference. Assigned explicitly so no sort decides which arm is purple.
F7_ARMS_A = ("mixed", "homogeneous", "sweep average")
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
    """F7 -- the two M0 claims of the sketch, drawn against their artifacts.

    Panel A is the sketch's "early breakthrough acceleration" claim and Panel B
    its "merit-prestige alignment" claim, both measured on the same four
    landscape pairs so the reader can see that the second survives and the
    first does not: the matched comparison (mixed vs the swept file's mu=0.5
    cohort) shows no speed-up, while the alignment gain is large on MultG2 and
    reverses on Noisy2. The pooled sweep average -- the only comparator under
    which a 2.6x ratio exists -- is drawn as a hatched grey reference so the
    sketch's number can be located rather than restated.
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
              ylabel="corr(prestige, ideal merit), run mean",
              title=f"B · Alignment: diversity helps in "
                    f"{caps['align_better']} of {caps['n_landscapes']} landscapes",
              tick_second={land: f"Δ {caps['deltas'][land]:+.3f}"
                           for land in lands})

    note = textwrap.fill(
        "source: data/figures_v1/run_number0-7.csv — the 8 canonical BULK_SETUPS, "
        f"{D.M0_REPEATS} repeats per cell, step_limit 200. mixed: newcomers draw "
        "c ~ U(0,1) (files 1/3/5/7). homogeneous: the swept file (0/2/4/6) at "
        "μ = 0.5, where newcomers inherit c = 0.5. Grey hatched: all 11 swept "
        "cohorts pooled (μ = 0, .1, …, 1) — an average over populations, not a "
        "cohort, and the only comparator under which the sketch's 2.6× exists "
        f"(censored at μ ≤ 0.1: {caps['reach']['MultG2|sweep average']['censored']}/110 "
        "runs on MultG2 never reach 10 %). × under a landscape = steps ratio "
        "homogeneous/mixed; Δ = mixed minus homogeneous. No MultG2_no_dist artifact "
        "exists in the repo (no sweep ran use_distance=False), so the fourth pair is "
        "randominit. Model-internal, no empirical calibration.",
        width=176)
    fig.text(0.006, 0.012, note, fontsize=s["note_size"], color=T.PALETTE["gray"],
             va="bottom", linespacing=1.45)
    fig.tight_layout(rect=(0, 0.105, 1, 1))
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
            {"landscape": land, "arm": "sweep/mixed", "metric": "ratio_h10_steps",
             "value": r["sweep_over_mixed"], "kind": "annotated"},
            {"landscape": land, "arm": "mixed-homogeneous", "metric": "delta_c_ideal",
             "value": caps["deltas"][land], "kind": "annotated"},
            {"landscape": land, "arm": "mixed-homogeneous", "metric": "harvest_gap_pp",
             "value": caps["harvest_gap_pp"][land], "kind": "annotated"},
        ]
    export = pd.concat([export, pd.DataFrame(extra)], ignore_index=True)
    csv_path = T.write_figure_data(export, out)
    return fig, out, {"caps": caps, "csv": csv_path, "pdf": pdf}


BUILDERS = [
    ("F1", figure_sensitivity, "T1 sensitivity (post-audit arm)"),
    ("F2", figure_career_flows, "T3 career flows (clean + noisy)"),
    ("F2b", figure_career_flows_ideal_corr, "T3 slow-entry gradient"),
    ("F3", figure_welfare, "T4 welfare probe"),
    ("F4", figure_prestige_stock, "W3-A prestige stock"),
    ("F5", figure_remeasure_delta, "N1/N2 re-measure"),
    ("F7", figure_cognitive_division_of_labor, "M0 cognitive division of labour"),
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
