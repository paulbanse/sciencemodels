"""FA2_v2 - Arm A birth-tile decomposition, post-BUG-0006 (mgstd5, g = 0.1).

Section 4.2 (birth-tile falsification) of the paper sketch. Bar chart of the
two decomposition shares of the entry-induced quality drop, for curiosity
c in {0.3, 0.5, 0.7} on mgstd5:

    drop      = M0 - exact                      (entry-induced quality drop)
    starvation share = (random - exact) / drop  (negative: exact spawn BEATS
                                                 random on quality)
    herding share    = (M0  - random)   / drop  (>100%: the random-spawn arm
                                                 already sits below M0)
    identity: starvation + herding == 100%

Sources (read-only):
  levels:  data/interventions_v2/tables/arm_a_spawn.csv
           (post-fix exact/random terminal c_ideal, n = 20/seed cell,
           entry rate g = 0.1; the table's ring1 row belongs to a later
           refill round and is NOT part of this exact-vs-random split)
  M0:      data/interventions_v2/runs.csv, rows with arm == "C",
           agent_generation_time == 0, landscape == "mgstd5",
           refill_mode == "off"  (n = 20 per curiosity)

Replaces the pre-fix data/interventions_v1/figures/FA2_drop_decomposition.png
(106%-153% herding range). Model-internal, NOT calibrated.

Run:
    MPLBACKEND=Agg /opt/hermes/.venv/bin/python \
        scripts/report_notebook/fig_FA2_v2_arm_a_decomposition.py
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

OUT = REPO / "output/report_notebook_v1/FA2_v2_drop_decomposition.png"

SPAWNS = ("exact", "random")
CURIOUSITIES = (0.3, 0.5, 0.7)


def _load() -> pd.DataFrame:
    """One row per curiosity: M0 / exact / random levels + both shares."""
    sp = pd.read_csv(REPO / "data/interventions_v2/tables/arm_a_spawn.csv")
    sp = sp[(sp.landscape == "mgstd5") & (sp.r_spawn.isin(SPAWNS))]
    assert len(sp) == len(CURIOUSITIES) * len(SPAWNS), \
        f"expected {len(CURIOUSITIES)*len(SPAWNS)} mgstd5 spawn rows, got {len(sp)}"

    runs = pd.read_csv(REPO / "data/interventions_v2/runs.csv")
    m0 = runs[(runs.arm == "C") & (runs.agent_generation_time == 0)
              & (runs.landscape == "mgstd5") & (runs.refill_mode == "off")]
    assert len(m0) == len(CURIOUSITIES) * 20, \
        f"expected {len(CURIOUSITIES)*20} M0 rows, got {len(m0)}"

    rows = []
    for c in CURIOUSITIES:
        m0c = m0[m0.initial_curiosity == c]
        assert len(m0c) == 20, f"c={c}: M0 n={len(m0c)}, expected 20"
        off = float(m0c["corr_prestige_ideal_merit_final"].mean())
        lvl = {s: float(sp[(sp.curiosity == c)
                           & (sp.r_spawn == s)].c_ideal_fin_post.iloc[0])
               for s in SPAWNS}
        drop = off - lvl["exact"]
        assert abs(drop) > 1e-9, f"c={c}: degenerate M0-exact gap"
        rows.append({
            "curiosity": c,
            "m0_level": off,
            "exact_level": lvl["exact"],
            "random_level": lvl["random"],
            "n_m0": len(m0c),
            "n_spawn": int(sp[sp.curiosity == c].n.iloc[0]),
            "starvation_share": 100.0 * (lvl["random"] - lvl["exact"]) / drop,
            "herding_share": 100.0 * (off - lvl["random"]) / drop,
        })
    df = pd.DataFrame(rows)
    # decomposition identity: the two shares sum to exactly 100%
    assert np.allclose(df.starvation_share + df.herding_share, 100.0,
                       atol=1e-9), "shares do not sum to 100%"
    return df


def build():
    df = _load()
    s = {**T.HOUSE_STYLE, "figsize": (9.6, 4.9)}
    fig, ax = plt.subplots(figsize=s["figsize"])

    x = np.arange(len(df))
    w = 0.34
    colors = {"starvation": T.PALETTE["orange"], "herding": T.PALETTE["purple"]}
    bars_s = ax.bar(x - w / 2, df.starvation_share, width=w,
                    color=colors["starvation"], label="starvation share")
    bars_h = ax.bar(x + w / 2, df.herding_share, width=w,
                    color=colors["herding"], label="herding share")
    for bars in (bars_s, bars_h):
        ax.bar_label(bars, labels=[f"{v:+.1f}%" for v in
                                   (df.starvation_share if bars is bars_s
                                    else df.herding_share)],
                     padding=3, fontsize=s["tick_size"] - 0.5)
    ax.axhline(0, color=T.PALETTE["gray"], linewidth=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels([f"c = {c:.1f}" for c in df.curiosity])
    # headroom: tallest bar + label ~195, deepest bar + label ~-97
    ax.set_ylim(-118, 218)
    T._cosmetics(ax, s,
                 "Herding more than explains the entry drop; "
                 "starvation runs negative",
                 "curiosity c (entry rate g = 0.1, mgstd5)",
                 "share of the M0 \u2192 exact gap (%)")

    handles, labels = ax.get_legend_handles_labels()
    assert len(handles) == 2, f"expected 2 bar containers, got {len(handles)}"
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=2, fontsize=s["tick_size"], frameon=False)

    lv = "; ".join(
        f"c={r.curiosity:.1f} {r.m0_level:+.4f}/{r.exact_level:+.4f}/"
        f"{r.random_level:+.4f}" for r in df.itertuples())
    note = textwrap.fill(
        "source: interventions_v2 arm_a_spawn.csv (exact/random levels, "
        "g = 0.1, n = 20/cell) + runs.csv M0 (arm C, g = 0, refill off, "
        "n = 20/c); shares = starvation (random-exact)/(M0-exact), herding "
        "(M0-random)/(M0-exact), sum = 100% asserted. M0/exact/random: "
        + lv + ". ring1 excluded (later refill round). "
        "Model-internal, NOT calibrated.", width=150)
    fig.text(0.006, 0.012, note, fontsize=s["note_size"],
             color=T.PALETTE["gray"], va="bottom", linespacing=1.45)
    fig.tight_layout(rect=(0, 0.21, 1, 0.88))

    export = pd.DataFrame({
        "metric": "fa2v2",
        "curiosity": list(df.curiosity) + list(df.curiosity),
        "arm": ["starvation"] * len(df) + ["herding"] * len(df),
        "value": list(df.starvation_share) + list(df.herding_share),
        "ci_lo": list(df.starvation_share) + list(df.herding_share),
        "ci_hi": list(df.starvation_share) + list(df.herding_share),
    })
    # level columns ride along for the sidecar (per-arm duplication avoided:
    # keep one tidy block of shares; levels live in the figure note + this)
    lvblock = df[["curiosity", "m0_level", "exact_level", "random_level",
                  "n_m0", "n_spawn"]]
    export = export.merge(lvblock, on="curiosity", how="left")

    report = visual_qa.assert_figure_clean(fig, [
        {"ax": ax, "metric": "fa2v2", "expect_bars": 6,
         "sidecar": export, "ci_cols": ("ci_lo", "ci_hi")},
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
