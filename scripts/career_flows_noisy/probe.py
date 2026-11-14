"""Critical-path probe for the noisy career sweep (run before the real grid).

Three proofs, two seeds per arm, through the same functions the driver uses:

- schema: every column the analysis will read is present in a CSV written to
  disk and read back (model targets + career measures).
- binding: at a fixed seed, the noisy landscape and the clean one must differ on
  a headline outcome --- otherwise the swept axis is inert and the whole sweep
  would be a confident no-op.
- timing: wall time per unit, so the grid budget is arithmetic.

Writes only to /tmp (a probe must never sit inside the run folder):

    /opt/hermes/.venv/bin/python scripts/career_flows_noisy/probe.py
"""
from __future__ import annotations

import os
import sys
import time

REPO = "/workspace/git/sciencemodels"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "career_flows_noisy"))

import pandas as pd

from driver import LANDSCAPES, career_measures_from_agent_vars, run_one
from sciencemodels import landscapes as L

PROBE_DIR = "/tmp/career_noisy_probe"
FN = {"mgstd5": L.multipleGaussians, "noisypr5": L.noisyGaussian}
NEEDED_MODEL = ["cell_tag", "curiosity", "vanishing_factor_for_prestige",
                "landscape", "agent_generation_time", "step_limit", "seed", "Step",
                "Corr_prestige_ideal_merit", "Corr_prestige_local_merit",
                "top10pct_prestige_share", "explored_percentage",
                "percentage_knowledge_harvested", "mean_age", "mean_prestige"]
NEEDED_CAREER = ["seed", "Step", "n_agents", "incumbent_mean_prestige",
                 "newcomer_mean_prestige", "prestige_gap", "prestige_gap_ratio",
                 "harvested_cum", "harvest_flow", "percap_harvest_cum",
                 "percap_harvest_flow"]


def main():
    os.makedirs(PROBE_DIR, exist_ok=True)
    out = {}
    for land, seeds in (("noisypr5", [0, 1]), ("mgstd5", [0, 1])):
        lp = dict(LANDSCAPES[land][1])
        frames, careers = [], []
        for s in seeds:
            t0 = time.time()
            status, payload, n_final, av = run_one(
                FN[land], dict(lp), 0.5, 0.9, 0.1, 200, s)
            assert status == "ok", payload[-2000:]
            out.setdefault("seconds_per_run", []).append(round(time.time() - t0, 2))
            payload.insert(0, "cell_tag", f"probe_{land}")
            payload.insert(1, "curiosity", 0.5)
            payload.insert(2, "vanishing_factor_for_prestige", 0.9)
            payload.insert(3, "landscape", land)
            payload.insert(4, "agent_generation_time", 0.1)
            payload.insert(5, "step_limit", 200)
            frames.append(payload)
            cm = career_measures_from_agent_vars(av, payload)
            cm.insert(0, "seed", s)
            careers.append(cm)
            print(f"{land} seed {s}: ok, n_final={n_final}, "
                  f"ideal={payload['Corr_prestige_ideal_merit'].iloc[-1]:.4f}, "
                  f"top10={payload['top10pct_prestige_share'].iloc[-1]:.4f}")
        mp = os.path.join(PROBE_DIR, f"probe_{land}.csv")
        cp = os.path.join(PROBE_DIR, f"probe_{land}_career.csv")
        pd.concat(frames, ignore_index=True).to_csv(mp, index=False)
        pd.concat(careers, ignore_index=True).to_csv(cp, index=False)
        m = pd.read_csv(mp)
        c = pd.read_csv(cp)
        assert set(NEEDED_MODEL) <= set(m.columns), set(NEEDED_MODEL) - set(m.columns)
        assert set(NEEDED_CAREER) <= set(c.columns), set(NEEDED_CAREER) - set(c.columns)
        assert m.seed.nunique() == len(seeds) and m.Step.max() == 200
        assert c.n_agents.iloc[-1] == 41, c.n_agents.iloc[-1]
        out[f"{land}_rows"] = [len(m), len(c)]
        out[f"{land}_final"] = {
            "ideal": round(float(m.iloc[-1]["Corr_prestige_ideal_merit"]), 6),
            "local": round(float(m.iloc[-1]["Corr_prestige_local_merit"]), 6),
            "top10": round(float(m.iloc[-1]["top10pct_prestige_share"]), 6),
            "harvested": round(float(m.iloc[-1]["percentage_knowledge_harvested"]), 6),
            "gap": round(float(c.iloc[-1]["prestige_gap"]), 6),
        }

    # binding proof: same seed, same parameters, landscape switched
    a = out["noisypr5_final"]
    b = out["mgstd5_final"]
    deltas = {k: round(abs(a[k] - b[k]), 6) for k in a}
    out["binding_deltas_noisy_vs_clean"] = deltas
    out["binding_ok"] = all(v > 0 for v in deltas.values())
    out["total_probe_seconds"] = sum(out["seconds_per_run"])
    print(out)
    assert out["binding_ok"], f"swept axis inert: {deltas}"
    print("PROBE OK: schema present, axis binds, timing recorded")


if __name__ == "__main__":
    main()
