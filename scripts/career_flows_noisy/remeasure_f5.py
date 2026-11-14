"""Re-measure finding F5 (decoupling ordering in entry rate) under the CURRENT code.

Background: this round's sweep ran 23:54-00:05 UTC. While it ran, a sibling round
replaced the ideal-merit pairing (sorted-to-sorted instead of the static-head
pairing) and made the ideal benchmark follow the live population, so every
`Corr_prestige_ideal_merit` value produced before that change belongs to the old
implementation. The control-arm bit-exactness check proves the sweep matches the
clean round, but not that either matches today's tree.

F5 is the one verdict that rests on the ideal channel, so it is re-measured here
on fresh runs rather than argued from the old ones: 2 seeds x 3 entry rates x
2 landscapes = 12 runs, through the same `run_one`, on the current code.

Reads nothing from the sweep's CSVs except for comparison; writes
`f5_remeasure.csv` and `f5_remeasure.json` into the run folder.

    /opt/hermes/.venv/bin/python scripts/career_flows_noisy/remeasure_f5.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

REPO = "/workspace/git/sciencemodels"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "career_flows_noisy"))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from driver import run_one
from sciencemodels import landscapes as L

OUT = os.path.join(REPO, "data", "career_flows_noisy_v1")
SEEDS = [0, 1]
GS = [0.05, 0.1, 0.2]
ARMS = {"mgstd5": (L.multipleGaussians, {"std": 5, "number_gaussians": 2}),
        "noisypr5": (L.noisyGaussian, {"prop_random": 5, "std": 5, "std_curiosity": 0.1})}


def one(arm, g, seed):
    fn, gp = ARMS[arm]
    status, payload, n_final, _av = run_one(fn, dict(gp), 0.5, 0.9, g, 200, seed)
    if status != "ok":
        raise RuntimeError(f"{arm} g={g} seed={seed}: {payload[-1200:]}")
    return {"arm": arm, "agent_generation_time": g, "seed": seed, "n_final": n_final,
            "ideal_final": float(payload["Corr_prestige_ideal_merit"].iloc[-1]),
            "local_final": float(payload["Corr_prestige_local_merit"].iloc[-1]),
            "top10_final": float(payload["top10pct_prestige_share"].iloc[-1]),
            "harvested_final": float(payload["percentage_knowledge_harvested"].iloc[-1])}


def main():
    jobs = [(a, g, s) for a in ARMS for g in GS for s in SEEDS]
    rows = Parallel(n_jobs=2, return_as="list")(delayed(one)(a, g, s) for a, g, s in jobs)
    df = pd.DataFrame(rows).sort_values(["arm", "agent_generation_time", "seed"])
    df.to_csv(os.path.join(OUT, "f5_remeasure.csv"), index=False)
    res = {}
    for arm, sub in df.groupby("arm"):
        m = sub.groupby("agent_generation_time")[["ideal_final", "local_final"]].mean()
        order_ideal = bool(m.loc[0.05, "ideal_final"] < m.loc[0.1, "ideal_final"] < m.loc[0.2, "ideal_final"])
        order_local = bool(m.loc[0.05, "local_final"] < m.loc[0.1, "local_final"] < m.loc[0.2, "local_final"])
        res[arm] = {
            "ideal_final_by_g": {str(k): round(float(v), 4) for k, v in m["ideal_final"].items()},
            "local_final_by_g": {str(k): round(float(v), 4) for k, v in m["local_final"].items()},
            "ideal_spread": round(float(m["ideal_final"].max() - m["ideal_final"].min()), 4),
            "order_ideal_increasing_in_g": order_ideal,
            "order_local_increasing_in_g": order_local,
            "seeds_per_cell": len(SEEDS),
        }
    res["checked_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    res["note"] = ("2 seeds per cell on the current code; verdict is the ordering test, "
                   "not a level estimate")
    with open(os.path.join(OUT, "f5_remeasure.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
