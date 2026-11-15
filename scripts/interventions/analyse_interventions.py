"""Analysis for interventions v1 (P-0002 / P-0003 / P-0004).

Reads ONLY ``data/interventions_v1/cells/``, asserts the gross shape before
computing anything, and answers the owner's three questions:

  (i)   how much of the entry-induced quality drop and of the newcomer gap
        comes from birth-tile starvation vs visibility herding (Arm A),
  (ii)  whether entry lotteries or sabbatical shocks flatten the high-
        visibility loss penalty that carries the xi* kink (Arm B),
  (iii) how incumbent bailouts vs frontier refills reshape lock-in (Arm C).

Control arm inside the round (cross-round rule): Arm A's ``exact`` cells and
Arm C's ``off`` cells at g = 0.1 on mgstd5 carry identical parameters, so the
same seed must reproduce the same run bit-for-bit.  The assert below proves it
instead of assuming it, which is what makes the cross-arm comparison in (i)
attributable to the spawn rule alone.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "report_notebook"))

OUT = REPO / "data" / "interventions_v1"
CELLS = OUT / "cells"
FIGS = OUT / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

N_CELLS, N_SEEDS = 72, 20
LOSS = "ideal_merit_sum_final"       # terminal cross-section (negative)
QUALITY = "corr_prestige_ideal_merit_final"


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def load() -> pd.DataFrame:
    files = sorted(CELLS.glob("cell*.csv"))
    _require(len(files) == N_CELLS, f"gross cells={N_CELLS}, found {len(files)}")
    frames = []
    for f in files:
        d = pd.read_csv(f)
        _require(len(d) == N_SEEDS,
                 f"{f.name}: expected {N_SEEDS} rows, got {len(d)}")
        _require(d["agent_seed"].nunique() == N_SEEDS,
                 f"{f.name}: seed identity not unique")
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    _require(len(out) == N_CELLS * N_SEEDS,
             f"net runs={N_CELLS * N_SEEDS}, got {len(out)}")
    counts = out.groupby("arm").size().to_dict()
    _require(counts == {"A": 360, "B": 720, "C": 360},
             f"arm counts wrong: {counts}")
    return out


def cross_arm_control(df: pd.DataFrame) -> dict:
    """Arm A exact == Arm C off at g=0.1 on mgstd5, seed for seed."""
    a = df[(df.arm == "A") & (df.r_spawn == "exact")
           & (df.landscape == "mgstd5") & (df.agent_generation_time == 0.1)]
    c = df[(df.arm == "C") & (df.refill_mode == "off")
           & (df.landscape == "mgstd5") & (df.agent_generation_time == 0.1)]
    keys = ["initial_curiosity", "agent_seed"]
    m = a.merge(c, on=keys, suffixes=("_a", "_c"))
    _require(len(m) == 60, f"control pairs=60 (3 c x 20 seeds), got {len(m)}")
    worst = 0.0
    for col in (QUALITY, "corr_prestige_local_merit_final",
                "top10_final", "newcomer_gap", "ideal_merit_cum_final",
                "percentage_knowledge_harvested", "explored_percentage"):
        worst = max(worst, float(np.nanmax(np.abs(
            m[f"{col}_a"].to_numpy(float) - m[f"{col}_c"].to_numpy(float)))))
    _require(worst == 0.0,
             f"cross-arm control not bit-identical: worst diff {worst}")
    return {"pairs": int(len(m)), "worst_diff": worst}


# ------------------------------------------------------------- (i) Arm A
def question_i(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decompose the entry-induced quality drop and the newcomer gap."""
    a = df[df.arm == "A"]
    c = df[df.arm == "C"]
    rows, gap_rows = [], []
    for cur in (0.3, 0.5, 0.7):
        # in-round entry-off reference (Arm C, off, g=0, mgstd5)
        off = c[(c.refill_mode == "off") & (c.landscape == "mgstd5")
                & (c.agent_generation_time == 0.0)
                & (c.initial_curiosity == cur)][QUALITY].mean()
        level = {m: a[(a.r_spawn == m) & (a.landscape == "mgstd5")
                      & (a.initial_curiosity == cur)][QUALITY].mean()
                 for m in ("exact", "ring1", "random")}
        d_exact = off - level["exact"]          # total entry-induced drop
        d_random = off - level["random"]        # what dispersal leaves behind
        # D_total = (off - random) + (random - exact): the residual under
        # dispersal is the herding part, the improvement from dispersing is
        # the starvation part.  A negative starvation share therefore means
        # dispersal made the correlation WORSE, not that the split failed.
        starv = (level["random"] - level["exact"]) / d_exact if d_exact else np.nan
        herding = d_random / d_exact if d_exact else np.nan
        _require(abs(starv + herding - 1.0) < 1e-9,
                 f"shares must sum to 1, got {starv + herding}")
        rows.append(dict(
            curiosity=cur, corr_entry_off=float(off),
            corr_exact=float(level["exact"]), corr_ring1=float(level["ring1"]),
            corr_random=float(level["random"]),
            drop_total=float(d_exact), drop_after_dispersal=float(d_random),
            starvation_share=float(starv),
            herding_share=float(herding),
        ))
        # newcomer gap: no entry-off reference exists (no newcomers at g=0),
        # so the share is taken against the exact-mode gap itself.  Both gaps
        # are negative, so the share is computed on magnitudes: positive =
        # dispersal closed the gap, negative = dispersal widened it.
        for land in ("mgstd5", "noisypr5"):
            g = {m: a[(a.r_spawn == m) & (a.landscape == land)
                      & (a.initial_curiosity == cur)]["newcomer_gap"].mean()
                 for m in ("exact", "ring1", "random")}
            closed = (abs(g["exact"]) - abs(g["random"])) / abs(g["exact"]) \
                if g["exact"] else np.nan
            gap_rows.append(dict(
                curiosity=cur, landscape=land,
                gap_exact=float(g["exact"]), gap_ring1=float(g["ring1"]),
                gap_random=float(g["random"]),
                share_of_gap_closed_by_dispersal=float(closed)))
    return pd.DataFrame(rows), pd.DataFrame(gap_rows)


# ------------------------------------------------------------- (ii) Arm B
def question_ii(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """High-weight loss penalty and its convexity, per intervention."""
    b = df[df.arm == "B"].copy()
    b["loss"] = -b[LOSS]                        # positive = worse
    curve = (b.groupby(["alpha", "p_reloc", "visibility_weight"], as_index=False)
               .agg(loss=("loss", "mean"), loss_sd=("loss", "std"),
                    quality=(QUALITY, "mean"),
                    top10=("top10_final", "mean"),
                    wedge=("corr_prestige_local_merit_final", "mean"),
                    n=("agent_seed", "size")))
    _require(len(curve) == 18, f"B curve rows=3x2x3=18, got {len(curve)}")
    summaries = []
    for (alpha, preloc), g in curve.groupby(["alpha", "p_reloc"]):
        g = g.set_index("visibility_weight").sort_index()
        l5, l7, l9 = (float(g.loc[w, "loss"]) for w in (0.5, 0.7, 0.9))
        summaries.append(dict(
            alpha=alpha, p_reloc=preloc,
            loss_w05=l5, loss_w07=l7, loss_w09=l9,
            high_weight_penalty=l9 - l5,          # a rightward kink lowers this
            convexity=l9 - 2 * l7 + l5,           # discrete second difference
            quality_w09=float(g.loc[0.9, "quality"]),
            top10_w09=float(g.loc[0.9, "top10"]),
        ))
    s = pd.DataFrame(summaries)
    base = s[(s.alpha == 1.0) & (s.p_reloc == 0.0)].iloc[0]
    s["penalty_vs_baseline"] = s["high_weight_penalty"] - base["high_weight_penalty"]
    s["convexity_vs_baseline"] = s["convexity"] - base["convexity"]
    s["loss09_vs_baseline"] = s["loss_w09"] - base["loss_w09"]
    return curve, s


# ------------------------------------------------------------- (iii) Arm C
def question_iii(df: pd.DataFrame) -> pd.DataFrame:
    """Lock-in across refill regimes and entry rates."""
    c = df[df.arm == "C"].copy()
    c["loss"] = -c[LOSS]
    out = (c.groupby(["refill_mode", "agent_generation_time"], as_index=False)
             .agg(quality=(QUALITY, "mean"),
                  top10=("top10_final", "mean"),
                  explored=("explored_percentage", "mean"),
                  harvested=("percentage_knowledge_harvested", "mean"),
                  loss=("loss", "mean"),
                  best_knowledge=("best_knowledge", "mean"),
                  mean_knowledge=("avg_knowledge_on_grid", "mean"),
                  gap=("newcomer_gap", "mean"),
                  n=("agent_seed", "size")))
    _require(len(out) == 6, f"C rows=3 modes x 2 entry=6, got {len(out)}")
    return out


# ---------------------------------------------------------------- figures
def figures(curve: pd.DataFrame, s: pd.DataFrame, i_tbl: pd.DataFrame,
            iii_tbl: pd.DataFrame, df: pd.DataFrame) -> list[Path]:
    import diagram_templates as T
    out = []

    # FA1 -- quality by spawn mode (mgstd5, entry on)
    a = df[(df.arm == "A") & (df.landscape == "mgstd5")]
    t = (a.groupby(["initial_curiosity", "r_spawn"], as_index=False)
           .agg(q=(QUALITY, "mean")))
    order = ["exact", "ring1", "random"]
    t["r_spawn"] = pd.Categorical(t["r_spawn"], order, ordered=True)
    t = t.sort_values(["initial_curiosity", "r_spawn"])
    p = FIGS / "FA1_spawn_mode_quality.png"
    T.styled_linechart(
        t, x="initial_curiosity", y="q", group="r_spawn",
        title="Exact spawn keeps the quality correlation highest",
        xlabel="curiosity c", ylabel="corr(prestige, ideal merit), final step",
        source="Arm A, mgstd5, entry 0.1, 20 seeds (40 runs per point)",
        outpath=p)
    out.append(p)

    # FA2 -- decomposition shares (mgstd5); shares sum to 1 per column and
    # may be negative when dispersal worsens the outcome (see question_i)
    d = i_tbl.melt(id_vars=["curiosity", "drop_total"],
                   value_vars=["starvation_share", "herding_share"],
                   var_name="component", value_name="share")
    d["component"] = d["component"].map(
        {"starvation_share": "birth-tile starvation",
         "herding_share": "visibility herding (residual)"})
    p = FIGS / "FA2_drop_decomposition.png"
    T.styled_barchart(
        d, x="curiosity", y="share", group="component",
        title="Dispersal removes little of the entry-induced quality drop",
        xlabel="curiosity c", ylabel="share of drop (sums to 1 per column)",
        source="Arm A exact vs random vs Arm C entry-off, mgstd5, 20 seeds",
        value_fmt="{:.2f}", ylim=(-0.75, 1.75), outpath=p)
    out.append(p)

    # FB -- loss curves per intervention
    labels = {
        (1.0, 0.0): "baseline (linear roulette)",
        (0.5, 0.0): "damped roulette (alpha 0.5)",
        (0.0, 0.0): "entry lottery (alpha 0)",
        (1.0, 0.02): "sabbatical shocks (p 0.02)",
    }
    curve = curve.copy()
    curve["intervention"] = [labels.get((r.alpha, r.p_reloc),
                                        f"a{r.alpha}/r{r.p_reloc}")
                             for r in curve.itertuples()]
    keep = [k for k in labels if ((curve.alpha == k[0])
                                  & (curve.p_reloc == k[1])).any()]
    curve = curve[[(r.alpha, r.p_reloc) in keep for r in curve.itertuples()]]
    curve["intervention"] = pd.Categorical(
        curve["intervention"], [labels[k] for k in keep], ordered=True)
    curve = curve.sort_values(["visibility_weight", "intervention"])
    p = FIGS / "FB_welfare_loss_curves.png"
    T.styled_linechart(
        curve, x="visibility_weight", y="loss", group="intervention",
        title="Sabbatical shocks flatten the high-visibility loss penalty",
        xlabel="visibility weight (1 - curiosity)",
        ylabel="ideal-merit loss (terminal, positive)",
        source="Arm B, 2 landscapes x 20 seeds per point, entry 0.1, H=200",
        outpath=p)
    out.append(p)

    # FC -- lock-in across refill regimes
    t = iii_tbl.copy()
    t["entry"] = t["agent_generation_time"].map({0.0: "no entry", 0.1: "entry 0.1"})
    p = FIGS / "FC_refill_lockin.png"
    T.styled_linechart(
        t, x="entry", y="top10", group="refill_mode",
        title="Frontier refills raise prestige concentration under entry",
        xlabel="entry rate", ylabel="top-10% prestige share, final step",
        source="Arm C, mgstd5, 3 curiosities x 20 seeds per point, H=200",
        outpath=p)
    out.append(p)
    return out


def main() -> int:
    df = load()
    print(f"loaded {len(df)} runs across {df.cell_tag.nunique()} cells "
          f"(arm counts: {df.groupby('arm').size().to_dict()})")
    ctrl = cross_arm_control(df)
    print(f"cross-arm control: {ctrl}")
    i_tbl, gap_tbl = question_i(df)
    curve, ii_tbl = question_ii(df)
    iii_tbl = question_iii(df)

    i_tbl.to_csv(OUT / "q1_spawn_decomposition.csv", index=False)
    gap_tbl.to_csv(OUT / "q1_newcomer_gap.csv", index=False)
    curve.to_csv(OUT / "q2_loss_curve.csv", index=False)
    ii_tbl.to_csv(OUT / "q2_intervention_summary.csv", index=False)
    iii_tbl.to_csv(OUT / "q3_refill_lockin.csv", index=False)

    print("\n=== (i) quality-drop decomposition (mgstd5) ===")
    print(i_tbl.to_string(index=False))
    print("\n=== (i) newcomer gap ===")
    print(gap_tbl.to_string(index=False))
    print("\n=== (ii) intervention summary ===")
    print(ii_tbl.to_string(index=False))
    print("\n=== (iii) refill regimes ===")
    print(iii_tbl.to_string(index=False))

    paths = figures(curve, ii_tbl, i_tbl, iii_tbl, df)
    print("\nfigures:")
    for p in paths:
        print("  ", p.relative_to(REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
