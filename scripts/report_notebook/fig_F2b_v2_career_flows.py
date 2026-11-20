"""F2b_v2 - Career-flows decoupling, post-BUG-0006 (final-step ideal corr).

Section 4.3 of the paper sketch. Final-step Corr(prestige, ideal merit)
against the entry rate g in {0.05, 0.1, 0.2}, for mgstd5 vs noisypr5,
curiosity-averaged (pooled 3 c x 20 seeds = 60 runs per point), from the
post-fix re-run of the 360-run sweep B:

    data/career_flows_v2/careerB_*.csv  (18 cells, H = 200, vanishing 0.9)

VERIFICATION (brief): clean terrain must NOT dip into negative
correlations -- asserted on the CI lows of every mgstd5 point. Pre-fix
(v1) the same cells read negative (the newborn zero-padding artifact);
the v1 curiosity-averaged means are computed here for the note.

Replaces the pre-fix F2b_career_flows_noisy_vs_clean.png. Model-internal,
NOT calibrated.

Run:
    MPLBACKEND=Agg /opt/hermes/.venv/bin/python \
        scripts/report_notebook/fig_F2b_v2_career_flows.py
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

OUT = REPO / "output/report_notebook_v1/F2b_v2_career_flows_noisy_vs_clean.png"

G_LEVELS = (0.05, 0.1, 0.2)
LANDSCAPES = ("mgstd5", "noisypr5")
COL = "Corr_prestige_ideal_merit"


def _final_step_frames() -> pd.DataFrame:
    files = sorted(p for p in
                   (REPO / "data/career_flows_v2").glob("careerB_*.csv")
                   if not p.name.endswith("_career.csv"))
    assert len(files) == 18, f"expected 18 career-flow cells, got {len(files)}"
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    fin = df[df.Step == df.Step.max()].copy()
    assert len(fin) == 360, f"expected 360 final-step rows, got {len(fin)}"
    return fin


def _summarise(fin: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (land, g), sub in fin.groupby(["landscape", "agent_generation_time"]):
        v = sub[COL].to_numpy(float)
        mean, se = float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))
        rows.append({"landscape": land, "entry": g, "value": mean, "se": se,
                     "ci_lo": mean - 1.96 * se, "ci_hi": mean + 1.96 * se,
                     "n": len(v)})
    out = pd.DataFrame(rows)
    assert len(out) == 6, f"expected 6 points, got {len(out)}"
    # brief verification: clean terrain must not dip negative (CI lows too)
    clean = out[out.landscape == "mgstd5"]
    assert (clean.ci_lo > 0).all(), \
        f"clean-terrain CI low dips <= 0: {clean[['entry', 'ci_lo']].to_dict('records')}"
    return out.sort_values(["landscape", "entry"]).reset_index(drop=True)


def _v1_means() -> dict:
    """Curiosity-averaged pre-fix means per (landscape, g) for the note."""
    out = {}
    for land, path in [("mgstd5", "career_flows_v1/careerB_analysis.csv"),
                       ("noisypr5", "career_flows_noisy_v1/noisyB_analysis.csv")]:
        p = REPO / "data" / path
        if not p.is_file():
            continue
        d = pd.read_csv(p)
        out[land] = d.groupby("agent_generation_time")["ideal_corr_final"].mean()
    return out


def build():
    fin = _final_step_frames()
    df = _summarise(fin)
    v1 = _v1_means()
    s = {**T.HOUSE_STYLE, "figsize": (9.6, 4.9)}
    fig, ax = plt.subplots(figsize=s["figsize"])

    colors = {"mgstd5": T.PALETTE["purple"], "noisypr5": T.PALETTE["orange"]}
    labels = {"mgstd5": "mgstd5 (clean)", "noisypr5": "noisypr5"}
    for land in LANDSCAPES:
        sub = df[df.landscape == land]
        ax.errorbar(sub.entry, sub.value,
                    yerr=[sub.value - sub.ci_lo, sub.ci_hi - sub.value],
                    fmt="o-", color=colors[land], label=labels[land],
                    linewidth=2, markersize=6, capsize=4, elinewidth=1.4)
        for r in sub.itertuples():
            ax.text(r.entry, r.ci_hi + 0.007, f"{r.value:+.3f}",
                    ha="center", va="bottom", fontsize=s["tick_size"] - 1)
    ax.axhline(0, color=T.PALETTE["gray"], linewidth=1.2)

    ax.set_xticks(list(G_LEVELS))
    ax.set_xticklabels([f"{g:.2f}" for g in G_LEVELS])
    ax.set_xlim(0.02, 0.232)
    ax.set_ylim(-0.06, 0.215)
    ax.set_yticks([0.0, 0.05, 0.10, 0.15, 0.20])
    T._cosmetics(ax, s,
                 "Every entry cell aligns after the fix; slower entry "
                 "aligns more strongly",
                 "entry rate (agent_generation_time)",
                 "Corr(prestige, ideal merit), final step")
    ax.legend(loc="upper right", fontsize=s["tick_size"], frameon=False)

    clean_lo = float(df[df.landscape == "mgstd5"].ci_lo.min())
    v1txt = ""
    if v1:
        parts = []
        for land in LANDSCAPES:
            if land in v1:
                m = v1[land]
                parts.append(f"{land} " + "/".join(f"{m[g]:+.3f}" for g in G_LEVELS))
        v1txt = (" Pre-fix v1 curiosity-averaged means: "
                 + "; ".join(parts) + " (zero-padding artifact).")
    note = textwrap.fill(
        "source: data/career_flows_v2/careerB_*.csv (post-fix re-run of "
        "sweep B: g x c x landscape x 20 seeds = 360 runs, H = 200, "
        "vanishing 0.9), final step, curiosity-averaged (pooled 60 runs "
        "per point, 95% CI = mean +/- 1.96 SE). VERIFICATION: all mgstd5 "
        f"CI lows > 0 (min {clean_lo:+.3f})." + v1txt +
        " Model-internal, NOT calibrated.", width=150)
    fig.text(0.006, 0.012, note, fontsize=s["note_size"],
             color=T.PALETTE["gray"], va="bottom", linespacing=1.45)
    fig.tight_layout(rect=(0, 0.21, 1, 1))

    export = df.copy()
    export.insert(0, "metric", "f2bv2")
    report = visual_qa.assert_figure_clean(fig, [
        {"ax": ax, "metric": "f2bv2", "expect_lines": 2,
         "points_per_line": 3, "sidecar": export,
         "ci_cols": ("ci_lo", "ci_hi")},
    ])

    outdir = OUT.parent
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=s["dpi"])
    pdf = OUT.with_suffix(".pdf")
    fig.savefig(pdf)
    plt.close(fig)
    csv = T.write_figure_data(export, OUT)
    print(f"QA: {report.get('_total', 0)} failure(s)")
    print(f"Wrote {OUT} / {pdf} / {csv}")
    print(df.to_string(index=False))
    return {"png": str(OUT), "pdf": str(pdf), "data_csv": str(csv)}


if __name__ == "__main__":
    build()
