"""Consolidate career_flows_v1: oscillation statistics (sweep A) and career-flow
measures (sweep B). Reads only the per-cell CSVs + checkpoint under
data/career_flows_v1/ (plus read-only T1 artifacts for the h200 contrast).

Oscillation definitions (declared a priori, applied mechanically):
- sign change: consecutive pooled steps with strictly opposite signs
  (x[t-1] * x[t] < 0); exact zeros (the correlation reporter's zero-guard)
  break neither.
- neg share: fraction of steps 1..H with pooled value < 0.
- final sign rate: fraction of seeds whose final-step value < 0.
- verdict per cell: oscillates = sign_changes >= 3 and 0.05 < neg_share < 0.95;
  ends_negative = pooled final < 0; mean_reverted = pooled final >= 0.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = "/workspace/git/sciencemodels"
OUT = os.path.join(REPO, "data", "career_flows_v1")
T1 = os.path.join(REPO, "data", "sensitivity_v1")

IDEAL = "Corr_prestige_ideal_merit"
LOCAL = "Corr_prestige_local_merit"


def oscillation_stats(series):
    """series: pooled per-step values, index 1..H (step 0 excluded)."""
    x = series.to_numpy(dtype=float)
    sign_changes = int(sum(1 for i in range(1, len(x)) if x[i-1] * x[i] < 0))
    neg_share = float((x < 0).mean())
    zero_share = float((x == 0).mean())
    return sign_changes, neg_share, zero_share


def main():
    with open(os.path.join(OUT, "checkpoint.json")) as f:
        cp = json.load(f)

    # ---------------- sweep A: oscillation confirmation ----------------
    a_cells = {t: c for t, c in cp["cells"].items() if c["sweep"] == "A"}
    pooled_rows, analysis_rows = [], []
    for tag, entry in sorted(a_cells.items()):
        d = pd.read_csv(os.path.join(OUT, entry["csv"]))
        H = int(d["Step"].max())
        per_step = d.groupby("Step")[IDEAL].mean()
        per_step_local = d.groupby("Step")[LOCAL].mean()
        steps = per_step.index
        pooled_rows.append(pd.DataFrame({
            "cell_tag": tag, "step": steps,
            "ideal_mean": per_step.to_numpy(),
            "local_mean": per_step_local.to_numpy(),
            "top10_mean": d.groupby("Step")["top10pct_prestige_share"].mean().to_numpy(),
            "harvested_mean": d.groupby("Step")["percentage_knowledge_harvested"].mean().to_numpy(),
            "mean_age_mean": d.groupby("Step")["mean_age"].mean().to_numpy(),
        }))
        sc, neg, zero = oscillation_stats(per_step.loc[1:])
        per_seed_sc, per_seed_neg, per_seed_final = [], [], []
        for s, g in d.groupby("seed"):
            g = g.set_index("Step").sort_index()
            a_sc, a_neg, _ = oscillation_stats(g[IDEAL].loc[1:])
            per_seed_sc.append(a_sc)
            per_seed_neg.append(a_neg)
            per_seed_final.append(g[IDEAL].iloc[-1])
        per_seed_final = np.array(per_seed_final)
        per_seed_sc = np.array(per_seed_sc)
        seeds_osc_ge3 = int((per_seed_sc >= 3).sum())
        last_third = per_step.loc[per_step.index > H * 2 / 3]
        # zero-guard / depletion context
        final_rows = d[d.Step == H]
        exact0_final = float((final_rows[IDEAL] == 0).mean())
        rec = {
            "cell_tag": tag,
            "curiosity": entry["params"]["curiosity"],
            "landscape": entry["params"]["landscape"],
            "vanishing": entry["params"]["vanishing_factor_for_prestige"],
            "gen_time": entry["params"]["agent_generation_time"],
            "step_limit": H,
            "status": entry["status"], "seeds_ok": entry["seeds_ok"],
            "sign_changes_pooled": sc,
            "neg_share_pooled": round(neg, 3),
            "zero_share_pooled": round(zero, 3),
            "sign_changes_per_seed_mean": round(float(np.mean(per_seed_sc)), 2),
            "sign_changes_per_seed_min": int(np.min(per_seed_sc)),
            "sign_changes_per_seed_max": int(np.max(per_seed_sc)),
            "seeds_osc_ge3": seeds_osc_ge3,
            "seed_level_final_sign_rate": round(float((per_seed_final < 0).mean()), 3),
            "neg_share_per_seed_mean": round(float(np.mean(per_seed_neg)), 3),
            "final_pooled": round(float(per_step.iloc[-1]), 4),
            "final_sign_rate": round(float((per_seed_final < 0).mean()), 3),
            "final_mean_over_seeds": round(float(np.mean(per_seed_final)), 4),
            "last_third_pooled_mean": round(float(last_third.mean()), 4),
            "mid_run_pooled_min": round(float(per_step.loc[1:].min()), 4),
            "exact0_final_share": round(exact0_final, 3),
            "harvested_final": round(float(final_rows["percentage_knowledge_harvested"].mean()), 3),
            "mean_age_final": round(float(final_rows["mean_age"].mean()), 1),
            "n_agents_final_expected": 21 + int(H * entry["params"]["agent_generation_time"]),
        }
        oscillates = sc >= 3 and 0.05 < neg < 0.95
        if rec["exact0_final_share"] > 0.5:
            verdict = "degenerate_zero_guard"
        elif oscillates and rec["final_pooled"] < 0:
            verdict = "oscillates_ends_negative"
        elif oscillates:
            verdict = "oscillates_mean_reverts"
        elif rec["final_pooled"] < 0:
            verdict = "monotone_negative"
        else:
            verdict = "stays_positive"
        rec["verdict"] = verdict
        analysis_rows.append(rec)

    a_pooled = pd.concat(pooled_rows, ignore_index=True)
    a_pooled.to_csv(os.path.join(OUT, "confirmA_pooled.csv"), index=False)
    a_analysis = pd.DataFrame(analysis_rows)
    a_analysis.to_csv(os.path.join(OUT, "confirmA_analysis.csv"), index=False)
    print(f"confirmA: {len(a_analysis)} cells, pooled {len(a_pooled)} rows")
    print(a_analysis[["cell_tag", "sign_changes_pooled", "neg_share_pooled",
                      "final_pooled", "final_sign_rate", "last_third_pooled_mean",
                      "exact0_final_share", "verdict"]].to_string(index=False))

    # T1 contrast (h200, 20 seeds) for the same 6 cells, from T1 raw CSVs (read-only)
    t1_rows = []
    for tag, entry in sorted(a_cells.items()):
        cur = entry["params"]["curiosity"]
        land = entry["params"]["landscape"]
        t1_tag = f"cell_cur{int(round(cur*10)):02d}_van09_{land}_gen01"
        t1_path = os.path.join(T1, f"{t1_tag}.csv")
        if not os.path.isfile(t1_path):
            continue
        d = pd.read_csv(t1_path)
        H = int(d["Step"].max())
        per_step = d.groupby("Step")[IDEAL].mean()
        finals = d[d.Step == H].groupby("seed")[IDEAL].last()
        sc, neg, _ = oscillation_stats(per_step.loc[1:])
        t1_rows.append({"cell_tag": tag, "t1_tag": t1_tag,
                        "t1_horizon": H, "t1_seeds": len(finals),
                        "t1_sign_changes_pooled": sc,
                        "t1_neg_share_pooled": round(float(neg), 3),
                        "t1_final_pooled": round(float(per_step.iloc[-1]), 4),
                        "t1_final_sign_rate": round(float((finals < 0).mean()), 3)})
    t1 = pd.DataFrame(t1_rows)
    if len(t1):
        t1.to_csv(os.path.join(OUT, "confirmA_t1_contrast.csv"), index=False)
        print("T1 contrast (h200/20 seeds) written:", len(t1))

    # ---------------- sweep B: career flows ----------------
    b_cells = {t: c for t, c in cp["cells"].items() if c["sweep"] == "B"}
    b_pooled_rows, b_analysis_rows = [], []
    for tag, entry in sorted(b_cells.items()):
        career = pd.read_csv(os.path.join(OUT, entry["career_csv"]))
        model = pd.read_csv(os.path.join(OUT, entry["csv"]))
        merged = career.merge(
            model[["seed", "Step", IDEAL, "top10pct_prestige_share",
                   "percentage_knowledge_harvested", "mean_age", "mean_prestige"]],
            on=["seed", "Step"], how="left")
        per_step = merged.drop(columns=["seed"], errors="ignore").groupby("Step").mean(numeric_only=True)
        pooled = per_step.reset_index()
        pooled.insert(0, "cell_tag", tag)
        pooled.insert(1, "curiosity", entry["params"]["curiosity"])
        pooled.insert(2, "agent_generation_time", entry["params"]["agent_generation_time"])
        pooled.insert(3, "landscape", entry["params"]["landscape"])
        b_pooled_rows.append(pooled)
        H = int(merged["Step"].max())
        late = per_step.loc[per_step.index >= 50]
        rec = {
            "cell_tag": tag,
            "agent_generation_time": entry["params"]["agent_generation_time"],
            "curiosity": entry["params"]["curiosity"],
            "status": entry["status"], "seeds_ok": entry["seeds_ok"],
            "pop_final": round(float(per_step["n_agents"].iloc[-1]), 1),
            "entrants_final": round(float(per_step["n_agents"].iloc[-1]) - 21, 1),
            "pop_growth_x": round(float(per_step["n_agents"].iloc[-1] / 21.0), 3),
            "top10_step10": round(float(per_step["top10pct_prestige_share"].loc[10]), 4),
            "top10_final": round(float(per_step["top10pct_prestige_share"].iloc[-1]), 4),
            "top10_delta": round(float(per_step["top10pct_prestige_share"].iloc[-1]
                                       - per_step["top10pct_prestige_share"].loc[10]), 4),
            "gap_final": round(float(per_step["prestige_gap"].iloc[-1]), 5),
            "gap_ratio_final": round(float(per_step["prestige_gap_ratio"].iloc[-1]), 3),
            "gap_ratio_mean_late": round(float(late["prestige_gap_ratio"].mean()), 3),
            "gap_sign_late": "neg" if late["prestige_gap"].mean() < 0 else "nonneg",
            "percap_harvest_final": round(float(per_step["percap_harvest_cum"].iloc[-1]), 5),
            "harvest_flow_mean_last50": round(float(per_step["harvest_flow"].iloc[-50:].mean()), 6),
            "percap_flow_mean_last50": round(float(per_step["percap_harvest_flow"].iloc[-50:].mean()), 7),
            "mean_age_final": round(float(per_step["mean_age"].iloc[-1]), 1),
            "ideal_corr_final": round(float(per_step[IDEAL].iloc[-1]), 4),
            "harvested_final": round(float(per_step["percentage_knowledge_harvested"].iloc[-1]), 3),
        }
        b_analysis_rows.append(rec)

    b_pooled = pd.concat(b_pooled_rows, ignore_index=True)
    b_pooled.to_csv(os.path.join(OUT, "careerB_pooled.csv"), index=False)
    b_analysis = pd.DataFrame(b_analysis_rows)
    b_analysis.to_csv(os.path.join(OUT, "careerB_analysis.csv"), index=False)
    print(f"careerB: {len(b_analysis)} cells, pooled {len(b_pooled)} rows")
    print(b_analysis[["cell_tag", "pop_final", "entrants_final", "top10_delta",
                      "gap_final", "gap_ratio_final", "gap_ratio_mean_late",
                      "percap_harvest_final", "ideal_corr_final"]].to_string(index=False))


if __name__ == "__main__":
    main()
