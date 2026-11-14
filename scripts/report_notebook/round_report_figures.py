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

import pandas as pd

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


BUILDERS = [
    ("F1", figure_sensitivity, "T1 sensitivity (post-audit arm)"),
    ("F2", figure_career_flows, "T3 career flows (clean + noisy)"),
    ("F2b", figure_career_flows_ideal_corr, "T3 slow-entry gradient"),
    ("F3", figure_welfare, "T4 welfare probe"),
    ("F4", figure_prestige_stock, "W3-A prestige stock"),
    ("F5", figure_remeasure_delta, "N1/N2 re-measure"),
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
