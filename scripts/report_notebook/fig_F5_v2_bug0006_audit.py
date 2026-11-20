"""F5_v2 - BUG-0006 paired audit: what the fix moved, and what it did not.

Section 4.1 of the paper sketch. Paired coefficient plot (vertical): mean
post-fix minus pre-fix delta with 95% CI over 720 seed-matched T1 pairs
(data/bug0006_remeasure_v2/t1_runs.csv vs data/remeasure_v1/runs.csv,
origin = T1, joined on curiosity x vanishing x landscape x entry rate x
use_visibility_reward x seed):

  * corr(prestige, ideal merit), final      -> positive level shift
  * corr(prestige, ideal merit), time-mean  -> positive level shift
  * corr(prestige, local merit), final      -> exactly 0 (anchor)
  * top-10% prestige share, final           -> exactly 0 (anchor)

The g = 0 half of the pairs is exactly zero on the ideal channel too
(BUG-0006 only touched the birth step, which fires at g > 0) -- asserted.

Replaces the pre-fix F5_pairing_audit_paired_delta.png (BUG-0004 audit,
T1 + W_v05 arms). Model-internal, NOT calibrated.

Run:
    MPLBACKEND=Agg /opt/hermes/.venv/bin/python \
        scripts/report_notebook/fig_F5_v2_bug0006_audit.py
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

OUT = REPO / "output/report_notebook_v1/F5_v2_pairing_audit_paired_delta.png"

KEYS = ["curiosity", "vanishing", "landscape", "agent_generation_time",
        "use_visibility_reward", "seed"]
METRICS = [
    ("c_ideal_fin", "corr(prestige,\nideal), final"),
    ("c_ideal_tm", "corr(prestige,\nideal), time-mean"),
    ("c_local_fin", "corr(prestige,\nlocal), final"),
    ("top10_fin", "top-10% share,\nfinal"),
]
ANCHORS = ("c_local_fin", "top10_fin")


def _paired() -> tuple[pd.DataFrame, dict]:
    pre = pd.read_csv(REPO / "data/remeasure_v1/runs.csv")
    pre = pre[pre.origin == "T1"][KEYS + [m for m, _ in METRICS]].copy()
    post = pd.read_csv(REPO / "data/bug0006_remeasure_v2/t1_runs.csv")
    post = post[KEYS + [m for m, _ in METRICS]].copy()
    assert not pre.duplicated(KEYS).any(), "pre archive has duplicate keys"
    assert not post.duplicated(KEYS).any(), "post archive has duplicate keys"
    m = pre.merge(post, on=KEYS, suffixes=("_pre", "_post"), validate="1:1")
    assert len(m) == 720, f"expected 720 seed-matched T1 pairs, got {len(m)}"

    stats = {"n_pairs": len(m)}
    rows = []
    for col, label in METRICS:
        d = m[f"{col}_post"] - m[f"{col}_pre"]
        mean, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
        rows.append({"metric": col, "reporter": label, "value": mean, "se": se,
                     "ci_lo": mean - 1.96 * se, "ci_hi": mean + 1.96 * se,
                     "n_pairs": len(d),
                     "max_abs_delta": float(d.abs().max())})
        if col in ANCHORS:
            assert abs(mean) < 1e-12 and float(d.abs().max()) < 1e-9, \
                f"{col} anchor moved: mean={mean} max|d|={d.abs().max()}"
        else:
            assert mean > 0, f"{col}: expected a positive shift, got {mean}"
        if col == "c_ideal_fin":
            # the g = 0 half must be exactly zero: only birth steps moved
            g0 = m[m.agent_generation_time == 0]
            d0 = (g0[f"{col}_post"] - g0[f"{col}_pre"]).abs().max()
            stats["n_g0_pairs"] = len(g0)
            stats["g0_max_abs_delta"] = float(d0)
            assert d0 < 1e-12, f"g=0 ideal channel moved: max|d|={d0}"
    return pd.DataFrame(rows), stats


def build():
    df, stats = _paired()
    s = {**T.HOUSE_STYLE, "figsize": (9.6, 4.9)}
    fig, ax = plt.subplots(figsize=s["figsize"])

    x = np.arange(len(df))
    ax.errorbar(x, df.value,
                yerr=[df.value - df.ci_lo, df.ci_hi - df.value],
                fmt="o", color=T.PALETTE["purple"],
                ecolor=T.PALETTE["purple"], elinewidth=1.6, capsize=5,
                markersize=7, linestyle="none")
    ax.axhline(0, color=T.PALETTE["gray"], linewidth=1.2)

    for i, r in enumerate(df.itertuples()):
        txt = "0.0000" if abs(r.value) < 5e-5 else f"{r.value:+.4f}"
        ax.text(i, r.ci_hi + 0.004, txt, va="bottom", ha="center",
                fontsize=s["tick_size"] + 0.5,
                fontweight="bold" if r.metric not in ANCHORS else "normal")

    ax.set_xticks(x)
    ax.set_xticklabels(df.reporter, fontsize=s["tick_size"] - 0.5)
    ax.set_xlim(-0.6, len(df) - 0.4)
    ax.set_ylim(-0.016, float(df.ci_hi.max()) + 0.034)
    # pinned ticks: AutoLocator emitted an out-of-view -0.02 whose label
    # collided with the y-axis label (same ghost-tick pattern as F4)
    ax.set_yticks([0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12])
    T._cosmetics(ax, s,
                 "Metric invariance audit: ideal-side levels move; "
                 "local side and top-10 stay exactly zero",
                 "T1 cell measure (720 pairs; bars = 95% CI)",
                 "mean paired delta (corrected \u2212 early-draft)")

    note = textwrap.fill(
        f"source: data/bug0006_remeasure_v2/t1_runs.csv paired against "
        f"data/remeasure_v1/runs.csv (origin = T1), joined on "
        f"{' x '.join(KEYS)} -> {stats['n_pairs']} seed-matched pairs. "
        f"ideal final {df[df.metric=='c_ideal_fin'].value.iloc[0]:+.4f} "
        f"(SE {df[df.metric=='c_ideal_fin'].se.iloc[0]:.4f}), ideal "
        f"time-mean {df[df.metric=='c_ideal_tm'].value.iloc[0]:+.4f} "
        f"(SE {df[df.metric=='c_ideal_tm'].se.iloc[0]:.4f}); anchors "
        f"max|delta| = {df[df.metric.isin(ANCHORS)].max_abs_delta.max():.1e} "
        f"(float noise). The g = 0 half ({stats['n_g0_pairs']} pairs) is "
        f"exactly zero on the ideal channel (max|delta| = "
        f"{stats['g0_max_abs_delta']:.1e}): the birth step only fires at "
        f"g > 0. Model-internal, NOT calibrated.", width=150)
    fig.text(0.006, 0.012, note, fontsize=s["note_size"],
             color=T.PALETTE["gray"], va="bottom", linespacing=1.45)
    fig.tight_layout(rect=(0, 0.23, 1, 1))

    report = visual_qa.assert_figure_clean(fig, [
        {"ax": ax, "metric": "f5v2", "expect_lines": 1,
         "points_per_line": 4,
         "sidecar": df.assign(metric="f5v2"),
         "ci_cols": ("ci_lo", "ci_hi")},
    ])

    outdir = OUT.parent
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=s["dpi"])
    pdf = OUT.with_suffix(".pdf")
    fig.savefig(pdf)
    plt.close(fig)
    csv = T.write_figure_data(df, OUT)
    print(f"QA: {report.get('_total', 0)} failure(s)")
    print(f"Wrote {OUT} / {pdf} / {csv}")
    print(df[["metric", "value", "se", "ci_lo", "ci_hi"]].to_string(index=False))
    return {"png": str(OUT), "pdf": str(pdf), "data_csv": str(csv)}


if __name__ == "__main__":
    build()
