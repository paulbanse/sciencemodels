"""Consolidate the noisy career sweep and test each clean-landscape finding.

Reads only `data/career_flows_noisy_v1/` (own run folder) plus the read-only
clean arm `data/career_flows_v1/careerB_*.csv` for comparison.

Per-cell scalars use the same definitions as the T2 consolidator
(`scripts/career_flows/consolidate.py`), so the two arms line up field by field.

Declared verdict rules (fixed BEFORE the noisy numbers were read; a finding is
tested on the noisy arm and compared with the clean arm):

F1 population formula   pop_final == 21 + floor(g*H) in every cell (exact;
                        holds/fails only).
F2 newcomer gap         pooled prestige_gap < 0 at every step >= 50 in all 9
                        cells AND mean late gap ratio < 0. Magnitude tolerance:
                        |mean late ratio (noisy) - mean late ratio (clean)| <=
                        0.10 -> holds; sign holds but tolerance missed ->
                        qualifies; any non-negative late gap -> fails.
F3 per-capita harvest   percap_harvest_final strictly decreasing in g within
                        each curiosity cell (3/3). Tolerance: the g=0.05 / g=0.2
                        ratio moves by <= 25% versus the clean arm -> holds;
                        ordering holds but the ratio moves more -> qualifies;
                        ordering broken -> fails.
F4 concentration decay  top10_delta (step 200 - step 10) < 0 in all 9 cells AND
                        the decay is weakest at g=0.2 in at least 2 of 3
                        curiosity cells -> holds; delta < 0 everywhere but the
                        high-g ordering does not hold -> qualifies; any delta
                        >= 0 -> fails.
F5 decoupling ordering  mean ideal_corr_final over curiosity is increasing in g
                        (0.05 < 0.1 < 0.2). Tolerance: the spread (max - min
                        over g) stays within 0.10 of the clean spread -> holds;
                        ordering holds, spread moves more -> qualifies;
                        ordering broken -> fails.

Writes `*_analysis.csv`, `*_pooled.csv`, `finding_tests.csv`,
`control_equivalence.csv`, `manifest.json` under the run folder.

    /opt/hermes/.venv/bin/python scripts/career_flows_noisy/consolidate.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

REPO = "/workspace/git/sciencemodels"
OUT = os.path.join(REPO, "data", "career_flows_noisy_v1")
CLEAN = os.path.join(REPO, "data", "career_flows_v1")

IDEAL = "Corr_prestige_ideal_merit"
LOCAL = "Corr_prestige_local_merit"

GAP_TOL = 0.10
PERCAP_TOL = 0.25
SPREAD_TOL = 0.10


def cell_scalars(merged, entry):
    per_step = merged.drop(columns=["seed"], errors="ignore").groupby("Step").mean(numeric_only=True)
    H = int(merged["Step"].max())
    late = per_step.loc[per_step.index >= 50]
    return per_step, {
        "cell_tag": entry["cell_tag"],
        "agent_generation_time": entry["params"]["agent_generation_time"],
        "curiosity": entry["params"]["curiosity"],
        "landscape": entry["params"]["landscape"],
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
        "gap_min_late": round(float(late["prestige_gap"].min()), 5),
        "gap_max_late": round(float(late["prestige_gap"].max()), 5),
        "gap_sign_late": "neg" if late["prestige_gap"].mean() < 0 else "nonneg",
        "percap_harvest_final": round(float(per_step["percap_harvest_cum"].iloc[-1]), 5),
        "harvest_flow_mean_last50": round(float(per_step["harvest_flow"].iloc[-50:].mean()), 6),
        "percap_flow_mean_last50": round(float(per_step["percap_harvest_flow"].iloc[-50:].mean()), 7),
        "mean_age_final": round(float(per_step["mean_age"].iloc[-1]), 1),
        "ideal_corr_final": round(float(per_step[IDEAL].iloc[-1]), 4),
        "ideal_corr_last_third": round(float(per_step[IDEAL].loc[per_step.index > H * 2 / 3].mean()), 4),
        "local_corr_final": round(float(per_step[LOCAL].iloc[-1]), 4),
        "harvested_final": round(float(per_step["percentage_knowledge_harvested"].iloc[-1]), 3),
        "n_agents_expected": 21 + int(H * entry["params"]["agent_generation_time"]),
        "late_gap_steps_negative": int((late["prestige_gap"] < 0).sum()),
        "late_gap_steps_total": int(len(late)),
        "step_limit": H,
    }


def verdict(value_ok, tolerance_ok, fail_msg):
    if not value_ok:
        return "fails"
    return "holds" if tolerance_ok else "qualifies"


def main():
    with open(os.path.join(OUT, "checkpoint.json")) as f:
        cp = json.load(f)

    rows, pooled_rows = [], []
    for tag, entry in sorted(cp["cells"].items()):
        entry = dict(entry)
        entry["cell_tag"] = tag
        career = pd.read_csv(os.path.join(OUT, entry["career_csv"]))
        model = pd.read_csv(os.path.join(OUT, entry["csv"]))
        merged = career.merge(
            model[["seed", "Step", IDEAL, LOCAL, "top10pct_prestige_share",
                   "percentage_knowledge_harvested", "mean_age", "mean_prestige"]],
            on=["seed", "Step"], how="left")
        per_step, rec = cell_scalars(merged, entry)
        rows.append(rec)
        pooled = per_step.reset_index()
        pooled.insert(0, "cell_tag", tag)
        pooled.insert(1, "curiosity", entry["params"]["curiosity"])
        pooled.insert(2, "agent_generation_time", entry["params"]["agent_generation_time"])
        pooled.insert(3, "landscape", entry["params"]["landscape"])
        pooled_rows.append(pooled)

    noisy = pd.DataFrame([r for r in rows if r["landscape"] == "noisypr5"]).sort_values(
        ["agent_generation_time", "curiosity"]).reset_index(drop=True)
    control = pd.DataFrame([r for r in rows if r["landscape"] == "mgstd5"])
    pooled = pd.concat(pooled_rows, ignore_index=True)

    noisy.to_csv(os.path.join(OUT, "noisyB_analysis.csv"), index=False)
    pooled.to_csv(os.path.join(OUT, "pooled_all_cells.csv"), index=False)
    # the analysis arm only: the control cell is clean-landscape data and must
    # not slip into noisy-arm pooled means
    pooled[pooled["landscape"] == "noisypr5"].to_csv(
        os.path.join(OUT, "noisyB_pooled.csv"), index=False)

    # ---------- control arm: is the clean cell reproduced bit-exactly? ----------
    ct = control.iloc[0]["cell_tag"]
    orig_tag = "careerB_cur5_van9_mgstd5_gen0p1_h200"
    mine = pd.read_csv(os.path.join(OUT, f"{ct}.csv"))
    theirs = pd.read_csv(os.path.join(CLEAN, f"{orig_tag}.csv"))
    key = ["seed", "Step"]
    a = mine.sort_values(key).reset_index(drop=True)
    b = theirs.sort_values(key).reset_index(drop=True)
    target_cols = ["Corr_prestige_ideal_merit", "Corr_prestige_local_merit",
                   "top10pct_prestige_share", "explored_percentage",
                   "percentage_knowledge_harvested", "mean_age", "mean_prestige"]
    ctl = {"rows_mine": len(a), "rows_clean": len(b),
           "shape_match": list(a[target_cols].shape) == list(b[target_cols].shape)}
    for c in target_cols:
        ctl[f"max_abs_diff_{c}"] = float(np.max(np.abs(a[c].to_numpy() - b[c].to_numpy())))
    cm = pd.read_csv(os.path.join(OUT, f"{ct}_career.csv")).sort_values(key).reset_index(drop=True)
    cc = pd.read_csv(os.path.join(CLEAN, f"{orig_tag}_career.csv")).sort_values(key).reset_index(drop=True)
    for c in ["n_agents", "incumbent_mean_prestige", "newcomer_mean_prestige",
              "prestige_gap", "prestige_gap_ratio", "percap_harvest_cum"]:
        ctl[f"career_max_abs_diff_{c}"] = float(np.max(np.abs(cm[c].to_numpy() - cc[c].to_numpy())))
    ctl["bit_identical_model"] = all(v == 0.0 for k, v in ctl.items() if k.startswith("max_abs_diff_"))
    ctl["bit_identical_career"] = all(v == 0.0 for k, v in ctl.items() if k.startswith("career_max_abs_diff_"))
    pd.DataFrame([ctl]).to_csv(os.path.join(OUT, "control_equivalence.csv"), index=False)
    print("control equivalence vs clean arm:", json.dumps(
        {k: ctl[k] for k in ("bit_identical_model", "bit_identical_career", "rows_mine", "rows_clean")}))

    # ---------- clean arm, same scalars, for comparison ----------
    clean = pd.read_csv(os.path.join(CLEAN, "careerB_analysis.csv"))
    clean_mean_gap = float(clean["gap_ratio_mean_late"].mean())
    noisy_mean_gap = float(noisy["gap_ratio_mean_late"].mean())

    def by_g(df, col):
        return df.groupby("agent_generation_time")[col].mean()

    # F1 population formula
    f1 = verdict(bool((noisy["pop_final"] == noisy["n_agents_expected"]).all()), True,
                 "population formula mismatch")

    # F2 newcomer-incumbent gap
    neg_steps = bool((noisy["late_gap_steps_negative"] == noisy["late_gap_steps_total"]).all())
    gap_ok = neg_steps and noisy_mean_gap < 0
    f2 = verdict(gap_ok, abs(noisy_mean_gap - clean_mean_gap) <= GAP_TOL, "")

    # F3 per-capita harvest decreasing in g, within each curiosity cell
    mono = []
    for cur, g in noisy.groupby("curiosity"):
        g = g.sort_values("agent_generation_time")
        v = g["percap_harvest_final"].to_numpy()
        mono.append(bool((np.diff(v) < 0).all()))
    ratio_noisy = by_g(noisy, "percap_harvest_final")
    ratio_clean = by_g(clean, "percap_harvest_final")
    ratio_noisy = float(ratio_noisy.loc[0.05] / ratio_noisy.loc[0.2])
    ratio_clean = float(ratio_clean.loc[0.05] / ratio_clean.loc[0.2])
    f3 = verdict(all(mono), abs(ratio_noisy / ratio_clean - 1.0) <= PERCAP_TOL, "")

    # F4 concentration decay and its high-g weakness
    decay_ok = bool((noisy["top10_delta"] < 0).all())
    weakest_high_g = 0
    for cur, g in noisy.groupby("curiosity"):
        g = g.sort_values("agent_generation_time")
        absdelta = g["top10_delta"].abs().to_numpy()
        if absdelta[-1] == absdelta.min():
            weakest_high_g += 1
    f4 = verdict(decay_ok, weakest_high_g >= 2, "")

    # F5 decoupling ordering across entry rates
    ideal_by_g = by_g(noisy, "ideal_corr_final")
    ideal_by_g_clean = by_g(clean, "ideal_corr_final")
    order_ok = bool(ideal_by_g.loc[0.05] < ideal_by_g.loc[0.1] < ideal_by_g.loc[0.2])
    spread_noisy = float(ideal_by_g.max() - ideal_by_g.min())
    spread_clean = float(ideal_by_g_clean.max() - ideal_by_g_clean.min())
    f5 = verdict(order_ok, abs(spread_noisy - spread_clean) <= SPREAD_TOL, "")

    tests = pd.DataFrame([
        {"finding": "F1 population formula 21+floor(g*t)",
         "clean_arm": "31/41/61 scientists at g=0.05/0.1/0.2",
         "noisy_arm": "/".join(str(int(v)) for v in
                               noisy.groupby("agent_generation_time")["pop_final"].mean()),
         "verdict": f1},
        {"finding": "F2 newcomer-incumbent gap persistent and negative",
         "clean_arm": f"late ratio mean {clean_mean_gap:.3f}",
         "noisy_arm": f"late ratio mean {noisy_mean_gap:.3f}; "
                      f"{int(noisy['late_gap_steps_negative'].sum())}/"
                      f"{int(noisy['late_gap_steps_total'].sum())} cell-steps negative",
         "verdict": f2},
        {"finding": "F3 per-capita harvest falls with entry rate",
         "clean_arm": f"ratio (g=.05 / g=.2) {ratio_clean:.3f}",
         "noisy_arm": f"ratio (g=.05 / g=.2) {ratio_noisy:.3f}; "
                      f"monotone in {sum(mono)}/3 curiosity cells",
         "verdict": f3},
        {"finding": "F4 concentration decays within entry cells, weakest at high g",
         "clean_arm": "top10 delta<0 in 9/9; weakest at g=0.2",
         "noisy_arm": f"top10 delta<0 in {int((noisy['top10_delta'] < 0).sum())}/9; "
                      f"weakest at g=0.2 in {weakest_high_g}/3 curiosity cells",
         "verdict": f4},
        {"finding": "F5 decoupling strongest at slow entry (ideal corr rising in g)",
         "clean_arm": "ideal final " + ", ".join(f"{k}: {v:.4f}" for k, v in ideal_by_g_clean.items()),
         "noisy_arm": "ideal final " + ", ".join(f"{k}: {v:.4f}" for k, v in ideal_by_g.items()),
         "verdict": f5},
    ])
    tests["tolerance"] = [f"exact", f"{GAP_TOL:.2f} on late gap ratio",
                          f"{PERCAP_TOL:.0%} on the g-ratio", f"2/3 high-g cells",
                          f"{SPREAD_TOL:.2f} on the spread"]
    tests.to_csv(os.path.join(OUT, "finding_tests.csv"), index=False)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cells_gross": 10, "cells_net": int(len(cp["cells"])),
        "runs_gross": 200, "runs_net": int(sum(c["seeds_ok"] for c in cp["cells"].values())),
        "noisy_cells": int(len(noisy)), "control_cells": int(len(control)),
        "control_equivalence": {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v)
                                for k, v in ctl.items() if not isinstance(v, list)},
        "verdicts": dict(zip(tests["finding"], tests["verdict"])),
        "sources_sha256_16": {
            "driver": hashlib.sha256(open(os.path.join(REPO, "scripts", "career_flows_noisy", "driver.py"), "rb").read()).hexdigest()[:16],
            "consolidator": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16],
        },
        "environment": {"python": sys.version.split()[0],
                        "numpy": np.__version__, "pandas": pd.__version__},
    }
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, default=str)

    pd.set_option("display.width", 200)
    print(noisy[["cell_tag", "pop_final", "top10_delta", "gap_final",
                 "gap_ratio_mean_late", "percap_harvest_final",
                 "ideal_corr_final", "local_corr_final"]].to_string(index=False))
    print()
    print(tests[["finding", "noisy_arm", "verdict"]].to_string(index=False))
    print(f"\npooled rows {len(pooled)}; gross 200 runs / net "
          f"{manifest['runs_net']} runs over {manifest['cells_net']} cells")


if __name__ == "__main__":
    main()
