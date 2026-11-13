"""Longitudinal companion dataset for the figures-v1 regeneration (ADR-0002 driver).

``notebooks/longitudinalcomparisions.ipynb`` needs two shapes of data that the
report-mode bulk sweep does not produce:

1. per-agent trajectories (``plot_agent_based_data``): one CSV per landscape
   family, columns ``Step, AgentID, prestige, prestige_vanishing,
   prestige_visibility, ...`` -- written to
   ``data/figures_v1/agent_long_<family>.csv``.
2. per-step model frames per initial curiosity
   (``plot_all_curiosities``): one CSV per (curiosity, agent seed) whose
   filename carries ``curiosity<c>`` -- written to
   ``data/figures_v1/longitudinal_random/run_curiosity<c>_agent_seed<s>.csv``.

Both come from the same production model class, 200 steps, the figures-v1
parameter set (``step_limit`` shared with ``run_fresh_sweep.py`` cosmetically:
this dataset is longitudinal, so the step count *is* the series length).

Run from anywhere:

    MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/figures/make_longitudinal_data.py
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from joblib import Parallel, delayed

from sciencemodels import landscapes
from sciencemodels.model import MyModel

OUT_DIR = "data/figures_v1"
LONG_DIR = f"{OUT_DIR}/longitudinal_random"
STEPS = 200
N_JOBS = 3

# legacy notebook labels -> current landscape generators (2 / 6 gaussians as in
# the bulk_runner design; the legacy notebook names MGauss1/MGauss4 predate it)
FAMILIES = {
    "random": (landscapes.randominit, {}),
    "MGauss1": (landscapes.multipleGaussians, {"std": 5, "number_gaussians": 2}),
    "MGauss4": (landscapes.multipleGaussians, {"std": 5, "number_gaussians": 6}),
    "NGauss": (landscapes.noisyGaussian, {"prop_random": 5, "std": 5, "std_curiosity": 0.1}),
}


def params(curiosity, seed, generation_function, generation_params):
    return {
        "n_agents": 20,
        "initial_curiosity": curiosity,
        "use_distance": True,
        "epsilon": 0.1,
        "harvest": 0.02,
        "grid_size": 20,
        "cell_init_function": generation_function,
        "generation_params": generation_params | {"seed": seed},
        "agent_generation_time": 0,
        "new_questions": 0,
        "agent_seed": seed,
        "step_limit": STEPS,
        "agent_generation_function": lambda cur, _rng, _p: cur,
        "vanishing_factor_for_prestige": 0.9,
        "use_visibility_reward": True,
        "knowledge_normalization": "share",
        "prestige_accrual": "harvested",
    }


def agent_trajectories(family):
    path = f"{OUT_DIR}/agent_long_{family}.csv"
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return {"artifact": path, "status": "skipped-present"}
    fn, gp = FAMILIES[family]
    model = MyModel(**params(0.5, 0, fn, gp))
    model.run_mode_for_bulk()
    model.datacollector.get_agent_vars_dataframe().to_csv(path)
    return {"artifact": path, "status": "written"}


def curiosity_series(curiosity, seed):
    path = f"{LONG_DIR}/run_curiosity{curiosity}_agent_seed{seed}.csv"
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return {"artifact": path, "status": "skipped-present"}
    fn, gp = FAMILIES["random"]
    model = MyModel(**params(curiosity, seed, fn, gp))
    model.run_mode_for_bulk()
    model.datacollector.get_model_vars_dataframe().to_csv(path, index=False)
    return {"artifact": path, "status": "written"}


def main():
    os.makedirs(LONG_DIR, exist_ok=True)
    t0 = time.time()
    traj = Parallel(n_jobs=N_JOBS, return_as="list")(
        delayed(agent_trajectories)(f) for f in FAMILIES
    )
    cells = [(k / 10, s) for k in range(11) for s in range(2)]
    series = Parallel(n_jobs=N_JOBS, return_as="list")(
        delayed(curiosity_series)(c, s) for c, s in cells
    )
    manifest = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "steps": STEPS,
        "n_jobs": N_JOBS,
        "seconds": round(time.time() - t0, 1),
        "agent_trajectories": {"gross": len(FAMILIES),
                               "net": sum(r["status"] == "written" for r in traj),
                               "rows": traj},
        "curiosity_series": {"gross": len(cells),
                             "net": sum(r["status"] == "written" for r in series)},
    }
    with open(f"{OUT_DIR}/longitudinal_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({k: v for k, v in manifest.items() if k != "agent_trajectories"}, indent=2))
    print("agent trajectories:", [(r["artifact"], r["status"]) for r in traj])


if __name__ == "__main__":
    main()
