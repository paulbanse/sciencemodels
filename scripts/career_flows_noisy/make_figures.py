"""Figures for the noisy career sweep: noisy arm against the clean arm.

Four panels, all pooled means over the three curiosity cells and 20 seeds:

1. newcomer-incumbent gap ratio by entry rate (steps 50-200; the ratio is
   degenerate while incumbent prestige is ~0, as in T2),
2. per-capita cumulative harvest by entry rate,
3. top-10% prestige share trajectory by entry rate,
4. final ideal and local correlation by entry rate (dots, both channels).

Reads `data/career_flows_noisy_v1/noisyB_pooled.csv` (own) and the read-only
clean arm `data/career_flows_v1/careerB_pooled.csv`. Writes PNGs into the run
folder. Text stays plain ASCII: matplotlib mathtext raises on many symbols and
the exception only arrives at save time.

    MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/career_flows_noisy/make_figures.py
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = "/workspace/git/sciencemodels"
OUT = os.path.join(REPO, "data", "career_flows_noisy_v1")
CLEAN = os.path.join(REPO, "data", "career_flows_v1")
FIG = os.path.join(OUT, "figures")

NOISY = pd.read_csv(os.path.join(OUT, "noisyB_pooled.csv"))
CLEANDF = pd.read_csv(os.path.join(CLEAN, "careerB_pooled.csv"))

# T2's pooled file carries the ideal channel only; the local (benchmark-free)
# channel is recovered from the clean arm's raw per-cell CSVs, read-only.
import glob

_loc = pd.concat([
    pd.read_csv(p, usecols=["agent_generation_time", "Step",
                            "Corr_prestige_local_merit"])
    for p in sorted(glob.glob(os.path.join(CLEAN, "careerB_*_mgstd5_gen0p*_h200.csv")))
    if not p.endswith("_career.csv")
])
CLEANDF = CLEANDF.merge(
    _loc.groupby(["agent_generation_time", "Step"])["Corr_prestige_local_merit"].mean().reset_index(),
    on=["agent_generation_time", "Step"], how="left")

GS = [0.05, 0.1, 0.2]
COLORS = {0.05: "#1f77b4", 0.1: "#ff7f0e", 0.2: "#2ca02c"}


def pooled(df, col):
    """Mean over curiosity cells, per (gen_time, Step)."""
    return df.groupby(["agent_generation_time", "Step"])[col].mean().reset_index()


def main():
    os.makedirs(FIG, exist_ok=True)
    figs = []

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0][0]
    for g in GS:
        for df, style, lab in ((CLEANDF, "--", "clean"), (NOISY, "-", "noisy")):
            d = pooled(df, "prestige_gap_ratio")
            d = d[(d.agent_generation_time == g) & (d.Step >= 50)]
            ax.plot(d.Step, d.prestige_gap_ratio, style, color=COLORS[g], lw=1.4,
                    label=f"g={g} {lab}")
    ax.set_title("A  young-minus-old prestige gap ratio (steps 50-200)")
    ax.set_xlabel("step")
    ax.set_ylabel("gap ratio (newcomer - incumbent) / incumbent")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[0][1]
    for g in GS:
        for df, style, lab in ((CLEANDF, "--", "clean"), (NOISY, "-", "noisy")):
            d = pooled(df, "percap_harvest_cum")
            d = d[d.agent_generation_time == g]
            ax.plot(d.Step, d.percap_harvest_cum, style, color=COLORS[g], lw=1.4,
                    label=f"g={g} {lab}")
    ax.set_title("B  per-capita cumulative harvest")
    ax.set_xlabel("step")
    ax.set_ylabel("harvested knowledge per scientist")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1][0]
    for g in GS:
        for df, style, lab in ((CLEANDF, "--", "clean"), (NOISY, "-", "noisy")):
            d = pooled(df, "top10pct_prestige_share")
            d = d[d.agent_generation_time == g]
            ax.plot(d.Step, d.top10pct_prestige_share, style, color=COLORS[g], lw=1.4,
                    label=f"g={g} {lab}")
    ax.set_title("C  top-10% prestige share")
    ax.set_xlabel("step")
    ax.set_ylabel("share of prestige held by top decile")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1][1]
    for df, marker, lab in ((CLEANDF, "o", "clean"), (NOISY, "s", "noisy")):
        final = df[df.Step == 200]
        for col, ls in (("Corr_prestige_ideal_merit", "-"),
                        ("Corr_prestige_local_merit", ":")):
            d = final.groupby("agent_generation_time")[col].mean()
            ax.plot(GS, [d.loc[g] for g in GS], marker=marker, ls=ls,
                    label=f"{lab}, {'ideal' if 'ideal' in col else 'local'} channel")
    ax.set_title("D  final correlation with merit by entry rate")
    ax.set_xlabel("agent_generation_time g")
    ax.set_ylabel("mean correlation at step 200")
    ax.axhline(0, color="k", lw=0.8)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    fig.suptitle("Career flows on the noisy landscape (solid) vs the clean one (dashed); "
                 "model-internal, not calibrated", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = os.path.join(FIG, "career_noisy_vs_clean.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs.append(p)

    print(json.dumps({"figures": figs, "noisy_rows": len(NOISY), "clean_rows": len(CLEANDF)},
                     indent=1))


if __name__ == "__main__":
    main()
