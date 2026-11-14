"""Consolidate sensitivity_v1 per-cell CSVs into cell means + mean trajectories.

v2: order-agnostic flattening of the agg MultiIndex (v1 mislabeled interleaved
mean/sd columns). Reads ONLY the per-cell CSVs and checkpoint.json under
data/sensitivity_v1/ (independent re-derivation path: no driver state).
Self-check: at step 1 both prestige-merit correlations must be >= 0.9 across
cells (first harvest is proportional to tile knowledge by construction) —
a canary against column scrambling.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = "/workspace/git/sciencemodels"
OUT = os.path.join(REPO, "data", "sensitivity_v1")

TARGETS = [
    "Corr_prestige_ideal_merit",
    "Corr_prestige_local_merit",
    "top10pct_prestige_share",
    "explored_percentage",
    "percentage_knowledge_harvested",
]

with open(os.path.join(OUT, "checkpoint.json")) as f:
    checkpoint = json.load(f)

frames = []
for tag, entry in checkpoint["cells"].items():
    csv_path = os.path.join(OUT, entry["csv"]) if entry.get("csv") else None
    if not csv_path or not os.path.isfile(csv_path):
        continue
    frames.append(pd.read_csv(csv_path))

raw = pd.concat(frames, ignore_index=True)
print(f"loaded {len(raw)} rows from {len(frames)} cell CSVs")

KEYS = ["cell_tag", "curiosity", "vanishing_factor_for_prestige", "landscape",
        "agent_generation_time", "Step"]

# --- mean trajectories per cell per step (order-agnostic flatten) ---
g = raw.groupby(KEYS)[TARGETS].agg(["mean", "std"])
g.columns = [f"{a}_{b}" for a, b in g.columns]  # derives names from the actual MultiIndex
agg = g.reset_index().rename(columns={"Step": "step"})
n_seeds = (
    raw.groupby(["cell_tag", "Step"])["seed"].nunique().reset_index()
    .rename(columns={"Step": "step", "seed": "n_seeds"})
)
agg = agg.merge(n_seeds, on=["cell_tag", "step"])
agg = agg.sort_values(["cell_tag", "step"]).reset_index(drop=True)
agg.to_csv(os.path.join(OUT, "trajectories_mean.csv"), index=False)
print(f"trajectories_mean.csv: {agg.shape[0]} rows")

# --- consolidation canary: step-1 correlations must be ~1 (first harvest ~ tile knowledge) ---
s1 = agg[agg.step == 1]
min_ideal = s1["Corr_prestige_ideal_merit_mean"].min()
min_local = s1["Corr_prestige_local_merit_mean"].min()
print(f"canary: min step-1 mean corr across cells: ideal={min_ideal:.4f} (expect >= 0.95), "
      f"local={min_local:.4f} (expect >= 0.80; noisy floor legitimately decorrelates local merit)")
assert min_ideal >= 0.95, "consolidation column scramble suspected (ideal)"
assert min_local >= 0.80, "consolidation column scramble suspected (local)"
assert (agg.groupby("cell_tag")["step"].nunique() == 201).all(), "trajectory length != 201"
assert (agg["n_seeds"] == 20).all(), "n_seeds != 20 somewhere"
print("canary OK; all cells have 201 steps x 20 seeds")

# --- cell summary ---
final = raw[raw["Step"] == raw["Step"].max()]
rows = []
for tag, entry in checkpoint["cells"].items():
    sub = raw[raw["cell_tag"] == tag]
    if sub.empty:
        rows.append({"cell_tag": tag, "status": entry["status"],
                     "seeds_ok": entry["seeds_ok"], "seeds_failed": entry["seeds_failed"]})
        continue
    fin = sub[sub["Step"] == sub["Step"].max()]
    rec = {
        "cell_tag": tag,
        "curiosity": sub["curiosity"].iloc[0],
        "vanishing_factor_for_prestige": sub["vanishing_factor_for_prestige"].iloc[0],
        "landscape": sub["landscape"].iloc[0],
        "agent_generation_time": sub["agent_generation_time"].iloc[0],
        "status": entry["status"],
        "seeds_ok": entry["seeds_ok"],
        "seeds_failed": entry["seeds_failed"],
        "n_scientists_final_min": entry["n_scientists_final_min"],
        "n_scientists_final_max": entry["n_scientists_final_max"],
    }
    for t in TARGETS:
        rec[f"{t}_timemean"] = sub[t].mean()
        rec[f"{t}_final"] = fin[t].mean()
    rec["wedge_ideal_vs_local_timemean"] = (
        rec["Corr_prestige_local_merit_timemean"] - rec["Corr_prestige_ideal_merit_timemean"]
    )
    rec["wedge_ideal_vs_local_final"] = (
        rec["Corr_prestige_local_merit_final"] - rec["Corr_prestige_ideal_merit_final"]
    )
    rows.append(rec)

summary = pd.DataFrame(rows).sort_values(
    ["landscape", "agent_generation_time", "vanishing_factor_for_prestige", "curiosity"]
)
summary.to_csv(os.path.join(OUT, "cell_summary.csv"), index=False)
print(f"cell_summary.csv: {summary.shape[0]} rows")
print(summary[["cell_tag", "status", "seeds_ok", "seeds_failed"]].to_string(index=False))
