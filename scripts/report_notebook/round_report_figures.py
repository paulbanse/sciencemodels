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
