"""Measure the code delta between this round's sweep and the current model tree.

The sweep ran 2026-11-13T23:54-00:05 UTC. While it ran, a sibling round landed
an ideal-merit pairing audit and an entry-prestige-stock change
(`sciencemodels/{io,metrics,model,plotting}.py` and the smoke test were modified
between 23:54 and 00:29). This round's numbers therefore belong to the
pre-audit code. Instead of forcing the old values, this script re-runs a
sample of the same cells and seeds through the CURRENT code and measures the
delta on every reporter the analysis uses.

Sample: the whole control cell (mgstd5, c=0.5, g=0.1, seeds 0-19) plus three
seeds of one noisy cell (c=0.5, g=0.1, seeds 0-2), both compared against the
CSVs this round already wrote.

Writes `code_delta.csv` and `code_delta.json` into the run folder. Nothing is
overwritten.

    /opt/hermes/.venv/bin/python scripts/career_flows_noisy/audit_code_delta.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone

REPO = "/workspace/git/sciencemodels"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "career_flows_noisy"))

import numpy as np
import pandas as pd

from driver import FIXED, MODEL_TARGETS, career_measures_from_agent_vars, run_one
from sciencemodels import landscapes as L

OUT = os.path.join(REPO, "data", "career_flows_noisy_v1")

CASES = [
    # (landscape label, cell tag already on disk, gen_time, seeds to re-run)
    ("mgstd5", "control_cur5_van9_mgstd5_gen0p1_h200", 0.1, list(range(5))),
    ("noisypr5", "noisyB_cur5_van9_noisypr5_gen0p1_h200", 0.1, [0, 1, 2]),
]
GP = {
    "mgstd5": {"std": 5, "number_gaussians": 2},
    "noisypr5": {"prop_random": 5, "std": 5, "std_curiosity": 0.1},
}
FN = {"mgstd5": L.multipleGaussians, "noisypr5": L.noisyGaussian}
CAREER_COLS = ["n_agents", "incumbent_mean_prestige", "newcomer_mean_prestige",
               "prestige_gap", "prestige_gap_ratio", "percap_harvest_cum"]


def sha16(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def main():
    rows = []
    for land, tag, gen, seeds in CASES:
        model_now, career_now = [], []
        for s in seeds:
            status, payload, n_final, av = run_one(
                FN[land], dict(GP[land]), 0.5, 0.9, gen, 200, s)
            if status != "ok":
                raise RuntimeError(f"{tag} seed {s} failed:\n{payload[-1500:]}")
            model_now.append(payload)
            cm = career_measures_from_agent_vars(av, payload)
            cm.insert(0, "seed", s)
            career_now.append(cm)
        now_m = pd.concat(model_now, ignore_index=True)
        now_c = pd.concat(career_now, ignore_index=True)
        old_m = pd.read_csv(os.path.join(OUT, f"{tag}.csv"))
        old_c = pd.read_csv(os.path.join(OUT, f"{tag}_career.csv"))
        old_m = old_m[old_m["seed"].isin(seeds)]
        old_c = old_c[old_c["seed"].isin(seeds)]
        key = ["seed", "Step"]
        now_m = now_m.sort_values(key).reset_index(drop=True)
        old_m = old_m.sort_values(key).reset_index(drop=True)
        now_c = now_c.sort_values(key).reset_index(drop=True)
        old_c = old_c.sort_values(key).reset_index(drop=True)
        if len(now_m) != len(old_m):
            raise RuntimeError(f"{tag}: row mismatch {len(now_m)} vs {len(old_m)}")
        rec = {"cell": tag, "seeds_rerun": len(seeds), "rows": len(now_m), "_delta_keys": []}
        for c in MODEL_TARGETS + ["Corr_prestige_local_merit"]:
            if c not in now_m.columns:
                continue
            d = float(np.max(np.abs(now_m[c].to_numpy() - old_m[c].to_numpy())))
            rec[f"model_max_abs_delta_{c}"] = d
            rec[f"model_mean_old_{c}"] = round(float(old_m[c].mean()), 6)
            rec[f"model_mean_new_{c}"] = round(float(now_m[c].mean()), 6)
            rec["_delta_keys"].append(f"model_max_abs_delta_{c}")
        for c in CAREER_COLS:
            d = float(np.max(np.abs(now_c[c].to_numpy() - old_c[c].to_numpy())))
            rec[f"career_max_abs_delta_{c}"] = d
            rec[f"career_mean_old_{c}"] = round(float(old_c[c].mean()), 6)
            rec[f"career_mean_new_{c}"] = round(float(now_c[c].mean()), 6)
            rec["_delta_keys"].append(f"career_max_abs_delta_{c}")
        rec["identical"] = all(rec[k] == 0.0 for k in rec["_delta_keys"])
        rec["max_abs_delta_overall"] = max(rec[k] for k in rec["_delta_keys"])
        del rec["_delta_keys"]
        rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "code_delta.csv"), index=False)
    summary = {
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources_sha256_16_now": {p: sha16(os.path.join(REPO, p)) for p in
                                  ["sciencemodels/metrics.py", "sciencemodels/model.py",
                                   "sciencemodels/io.py", "tests/test_smoke.py"]},
        "cases": rows,
    }
    with open(os.path.join(OUT, "code_delta.json"), "w") as f:
        json.dump(summary, f, indent=1)
    for r in rows:
        print(f"\n{r['cell']}  seeds_rerun={r['seeds_rerun']} rows={r['rows']}")
        for k, v in r.items():
            if k.endswith(tuple(f"_delta_{c}" for c in MODEL_TARGETS + CAREER_COLS + ["Corr_prestige_local_merit"])):
                print(f"   {k}: {v}")


if __name__ == "__main__":
    main()
