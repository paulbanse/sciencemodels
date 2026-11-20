"""F2 - Breadth vs. Depth: raw spatial coverage misleads, discovery does not.

Section 3.2 (`Finding L2`) of the paper sketch. Two panels over curiosity
c in [0, 1]:

  (a) raw spatial coverage (``explored_percentage``) under BOTH search modes,
      showing the paradox: under global jump search the pure visibility
      chasers (c = 0) touch ~99.3% of the grid while the most curious agents
      (c = 1.0) touch only ~81.2%; friction-on step search visits far less
      of the grid at all.
  (b) knowledge discovery (``percentage_knowledge_harvested``) under both
      modes, plus jump-search peak-weighted coverage
      (``explored_weighted_by_initial_knowledge``, dotted): c = 1.0 harvests
      ~76.7% while still covering >=98% of the *initial knowledge* -- the
      tiles the curious agents skip are the empty ones.

Sources (read-only):
  friction on (use_distance = 1, H = 200, 10 repeats/c):
      data/figures_v1/run_number{0,2,4,6}.csv = randominit, MultG2,
      MultG6, Noisy2 (the swept homogeneous files, D.M0_FILES).
  jump search (use_distance = 0, H = 400, 40 seeds/c):
      data/legacy_2025_2026/testrun_harmonizedKnowledge_*.csv, paired by
      initCellFunc: seed4 <-> randominit, std5_number_gaussians2 <->
      MultG2, std5_number_gaussians6 <-> MultG6, prop_random5 <->
      Noisy2 (its std_curiosity = 0.2, disclosed in the footnote).

DISCLOSED CONFOUND: the two archives differ in horizon (H = 200 vs 400)
and replicate count (10 vs 40). Cross-mode LEVELS are not horizon-matched;
the within-mode paradox (coverage falls with curiosity under jump search)
is unaffected. Model-internal, not calibrated.

Run:
    MPLBACKEND=Agg /opt/hermes/.venv/bin/python \
        scripts/report_notebook/fig_3_2_breadth_vs_depth.py
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import diagram_templates as T
import visual_qa

OUT = REPO / "output/report_notebook_v1/F2_breadth_vs_depth.png"

#: landscape -> (swept friction file index, jump-search legacy file, hue)
PAIRS = [
    ("MultG2", 2, "testrun_harmonizedKnowledge_seed4_std5_number_gaussians2.csv",
     T.PALETTE["purple"]),
    ("MultG6", 4, "testrun_harmonizedKnowledge_seed4_std5_number_gaussians6.csv",
     T.PALETTE["orange"]),
    ("Noisy2", 6,
     "testrun_harmonizedKnowledge_prop_random5_seed4_std5_std_curiosity0.2.csv",
     T.PALETTE["green"]),
    ("randominit", 0, "testrun_harmonizedKnowledge_seed4.csv",
     T.PALETTE["blue"]),
]
C_GRID = [round(0.1 * i, 10) for i in range(11)]


def _mean_ci(s: pd.Series) -> tuple[float, float, float]:
    m = float(s.mean())
    se = float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else 0.0
    return m, m - 1.96 * se, m + 1.96 * se


def load() -> pd.DataFrame:
    """Tidy frame: one row per (mode, landscape, c) with run-level stats."""
    rows = []
    for land, idx, jump_file, _hue in PAIRS:
        fr = pd.read_csv(REPO / f"data/figures_v1/run_number{idx}.csv")
        ju = pd.read_csv(REPO / "data/legacy_2025_2026" / jump_file)
        for mode, df, n in (("friction", fr, 10), ("jump", ju, 40)):
            counts = df.groupby("initial_curiosity").size()
            assert (counts == n).all(), (
                f"{land}/{mode}: expected {n} runs per c, got "
                f"{counts.to_dict()}")
            for c in C_GRID:
                sub = df[df["initial_curiosity"] == c]
                assert len(sub) == n, f"{land}/{mode} c={c}: {len(sub)} runs"
                for metric in ("explored_percentage",
                               "percentage_knowledge_harvested",
                               "explored_weighted_by_initial_knowledge"):
                    m, lo, hi = _mean_ci(sub[metric])
                    rows.append(dict(mode=mode, landscape=land,
                                     curiosity=c, metric=metric,
                                     value=m, ci_lo=lo, ci_hi=hi, n=n))
    out = pd.DataFrame(rows)
    assert len(out) == 4 * 2 * 11 * 3, f"tidy rows {len(out)} != 264"
    return out


def build(outdir: Path | str | None = None) -> dict:
    tidy = load()
    s = {**T.HOUSE_STYLE, "figsize": (11.4, 4.9)}
    outdir = Path(outdir) if outdir else OUT.parent
    out = outdir / OUT.name

    fig, (axA, axB) = plt.subplots(1, 2, figsize=s["figsize"])

    def series(metric, mode):
        for land, _i, _j, hue in PAIRS:
            # NB: tidy["mode"] -- tidy.mode would resolve to DataFrame.mode()
            sel = tidy[(tidy.metric == metric) & (tidy["mode"] == mode)
                       & (tidy.landscape == land)].sort_values("curiosity")
            yield land, hue, sel

    # ---- (a) raw spatial coverage ---------------------------------------
    n_a = 0
    for land, hue, sel in series("explored_percentage", "friction"):
        axA.plot(sel.curiosity, sel.value, color=hue, lw=1.9, ls="-",
                 marker="o", ms=3.4, label=f"{land} · friction")
        n_a += 1
    for land, hue, sel in series("explored_percentage", "jump"):
        axA.plot(sel.curiosity, sel.value, color=hue, lw=1.9, ls="--",
                 marker="s", ms=3.4, label=f"{land} · jump")
        n_a += 1
    # the paradox, annotated from the drawn values
    j2 = tidy[(tidy.metric == "explored_percentage") & (tidy["mode"] == "jump")
              & (tidy.landscape == "MultG2")].set_index("curiosity")
    assert len(j2) == 11, f"jump MultG2 explored rows: {len(j2)}"
    axA.annotate(
        f"jump search: {j2.loc[0.0, 'value'] * 100:.1f}% at c=0\n"
        f"$\\to$ {j2.loc[1.0, 'value'] * 100:.1f}% at c=1.0",
        xy=(1.0, j2.loc[1.0, "value"]), xytext=(0.34, 0.62),
        fontsize=s["note_size"], color=T.PALETTE["gray"],
        ha="left", va="bottom")
    # NB: text-only callout -- an arrow would join this artist's window
    # extent and every arrow crossing a plotted line fails the canvas QA
    T._cosmetics(axA, s,
                 "A · Raw spatial coverage falls with curiosity (jump search)",
                 "curiosity c", "explored grid share")
    axA.set_xlim(-0.03, 1.03)
    axA.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])  # pin: AutoLocator also
    # emits out-of-view ticks (1.2), which the canvas QA rightly flags
    axA.set_ylim(0.0, 1.09)

    # ---- (b) discovery + peak-weighted coverage --------------------------
    n_b = 0
    for land, hue, sel in series("percentage_knowledge_harvested", "friction"):
        axB.plot(sel.curiosity, sel.value, color=hue, lw=1.9, ls="-",
                 marker="o", ms=3.4, label=f"{land} · friction")
        n_b += 1
    for land, hue, sel in series("percentage_knowledge_harvested", "jump"):
        axB.plot(sel.curiosity, sel.value, color=hue, lw=1.9, ls="--",
                 marker="s", ms=3.4, label=f"{land} · jump")
        n_b += 1
    for land, hue, sel in series("explored_weighted_by_initial_knowledge",
                                 "jump"):
        axB.plot(sel.curiosity, sel.value, color=hue, lw=1.2, ls=":",
                 marker=None, alpha=0.85,
                 label=f"{land} · jump, knowledge-weighted")
        n_b += 1
    j2h = tidy[(tidy.metric == "percentage_knowledge_harvested")
               & (tidy["mode"] == "jump")
               & (tidy.landscape == "MultG2")].set_index("curiosity")
    j2w = tidy[(tidy.metric == "explored_weighted_by_initial_knowledge")
               & (tidy["mode"] == "jump")
               & (tidy.landscape == "MultG2")].set_index("curiosity")
    assert len(j2h) == 11 and len(j2w) == 11, (len(j2h), len(j2w))
    axB.annotate(
        f"c=1.0: {j2h.loc[1.0, 'value'] * 100:.1f}% harvested,\n"
        f"{j2w.loc[1.0, 'value'] * 100:.1f}% of initial knowledge covered",
        xy=(1.0, j2h.loc[1.0, "value"]), xytext=(0.02, 0.78),
        fontsize=s["note_size"], color=T.PALETTE["gray"],
        ha="left", va="bottom")
    # text-only callout in the empty band between jump harvest (<=0.74)
    T._cosmetics(axB, s,
                 "B · Discovery rises with curiosity under both modes",
                 "curiosity c", "harvested knowledge share")
    axB.set_xlim(-0.03, 1.03)
    axB.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # small negative floor = headroom for the c=0 friction points (min 0.013),
    # which the QA rightly keeps 3% clear of the bottom spine
    axB.set_ylim(-0.05, 1.10)

    # one shared legend above both panels: inside-panel legends sat on the
    # dense line bundles and the canvas QA flagged them
    handles, labels = axB.get_legend_handles_labels()
    assert len(handles) == 12, f"expected 12 panel-b handles, got {len(handles)}"
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=6, fontsize=s["tick_size"] - 1.5, frameon=False,
               columnspacing=1.0, handlelength=1.6)

    note = textwrap.fill(
        "source: friction-on = data/figures_v1/run_number0/2/4/6.csv "
        "(swept homogeneous cohorts, 10 repeats/cell, H = 200, "
        "use_distance = 1); jump = data/legacy_2025_2026/"
        "testrun_harmonizedKnowledge_*.csv (40 seeds/c, H = 400, "
        "use_distance = 0; Noisy2 jump file uses std_curiosity = 0.2). "
        "DISCLOSED CONFOUND: horizons differ (200 vs 400 steps), so "
        "cross-mode levels are not horizon-matched; the within-mode "
        "paradox is. Lines are run means; per-cell n in the sidecar. "
        "Model-internal, NOT calibrated.", width=176)
    fig.text(0.006, 0.012, note, fontsize=s["note_size"],
             color=T.PALETTE["gray"], va="bottom", linespacing=1.45)
    fig.tight_layout(rect=(0, 0.11, 1, 0.86))

    # ---- sidecar + QA ----------------------------------------------------
    export = tidy.copy()
    qa_a = export[export.metric == "explored_percentage"].copy()
    qa_a["metric"] = "panel_a"
    qa_b = export[export.metric.isin(
        ["percentage_knowledge_harvested",
         "explored_weighted_by_initial_knowledge"])].copy()
    qa_b["metric"] = "panel_b"

    report = visual_qa.assert_figure_clean(fig, [
        {"ax": axA, "metric": "panel_a", "expect_lines": n_a,
         "points_per_line": 11, "sidecar": qa_a,
         "ci_cols": ("value", "value")},
        {"ax": axB, "metric": "panel_b", "expect_lines": n_b,
         "points_per_line": 11, "sidecar": qa_b,
         "ci_cols": ("value", "value")},
    ])
    assert n_a == 8 and n_b == 12, f"series counts a={n_a} b={n_b}"

    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=s["dpi"])
    pdf = out.with_suffix(".pdf")
    fig.savefig(pdf)
    plt.close(fig)
    csv = T.write_figure_data(export, out)
    print(f"QA: {report.get('_total', 0)} failure(s)")
    print(f"Wrote {out} / {pdf} / {csv}")
    return {"png": str(out), "pdf": str(pdf), "data_csv": str(csv)}


if __name__ == "__main__":
    build()
