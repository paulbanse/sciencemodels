"""Re-measure pass (2026-11-14): rerun the load-bearing wave-1/2 cells under the
fixed ideal-merit pairing (N1 round, d3503ac) and summarize per cell for
paired comparison against the archived wave-1/2 summaries.

Model-internal; NOT calibrated (owner ruling 2026-11-13).
Grids: T1 sensitivity grid (36 cells) + welfare grid (40 cells), 20 seeds
each = 1520 runs. Outputs: data/remeasure_v1/ (runs.csv, cell_summary.csv,
manifest.json). One home for measures: everything comes from the model's own
datacollector.
"""
import itertools
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sciencemodels import landscapes as lf
from sciencemodels.model import MyModel

OUT = os.path.join(ROOT, "data", "remeasure_v1")

LAND = {
    "mgstd5": (lf.multipleGaussians, {"std": 5, "number_gaussians": 2}),
    "mg6": (lf.multipleGaussians, {"std": 5, "number_gaussians": 6}),
    "noisypr5": (lf.noisyGaussian, {"prop_random": 5, "std": 5, "std_curiosity": 0.1}),
}


def identity(cur, rng, params):
    return cur


def cells():
    grid = []
    # T1 sensitivity grid: 3 cur x 3 van x 2 land x 2 gen = 36 cells
    for cur, van, land, gen in itertools.product(
        (0.3, 0.5, 0.7), (0.5, 0.9, 1.0), ("mgstd5", "noisypr5"), (0.0, 0.1)
    ):
        grid.append(("T1", cur, van, land, gen, True))
    # Welfare grid: 5 cur x 2 toggle x 2 gen x 2 land = 40 cells
    for cur, vis, gen, land in itertools.product(
        (0.1, 0.3, 0.5, 0.7, 0.9), (True, False), (0.0, 0.1), ("mgstd5", "mg6")
    ):
        grid.append(("W", cur, 0.9, land, gen, vis))
    return grid


def run_one(origin, cur, van, land, gen, vis, seed):
    f, gp = LAND[land]
    m = MyModel(
        n_agents=20,
        initial_curiosity=cur,
        epsilon=0.1,
        harvest=0.02,
        grid_size=20,
        cell_init_function=f,
        generation_params=dict(gp),
        use_distance=True,
        agent_generation_time=gen,
        new_questions=0,
        agent_seed=seed,
        step_limit=200,
        agent_generation_function=identity,
        vanishing_factor_for_prestige=van,
        use_visibility_reward=vis,
        knowledge_normalization="share",
        prestige_accrual="harvested",
    )
    for _ in range(200):
        m.step(True)
    dc = m.datacollector.get_model_vars_dataframe()
    last = dc.iloc[-1]
    row = dict(
        origin=origin,
        curiosity=cur,
        vanishing=van,
        landscape=land,
        agent_generation_time=gen,
        use_visibility_reward=vis,
        seed=seed,
    )
    for src, dst in (
        ("Corr_prestige_ideal_merit", "c_ideal"),
        ("Corr_prestige_local_merit", "c_local"),
        ("top10pct_prestige_share", "top10"),
    ):
        row[dst + "_fin"] = float(last.get(src, np.nan))
        row[dst + "_tm"] = float(dc[src].mean())
    row["ideal_loss_cum"] = -sum(
        getattr(a, "ideal_merit_cum", 0.0) for a in m._scientists
    )
    row["harvested_fin"] = float(last.get("percentage_knowledge_harvested", np.nan))
    return row


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    canary = os.system(
        "MPLBACKEND=Agg %s %s > %s/canary.txt 2>&1"
        % (sys.executable, os.path.join(ROOT, "tests", "test_smoke.py"), OUT)
    )
    assert canary == 0, "canary failed - aborting before any run"
    grid = cells()
    t0 = time.time()
    rows = Parallel(n_jobs=3)(
        delayed(run_one)(*c, s) for c in grid for s in range(20)
    )
    df = pd.DataFrame(rows)
    df["wedge_fin"] = df["c_ideal_fin"] - df["c_local_fin"]
    df["wedge_tm"] = df["c_ideal_tm"] - df["c_local_tm"]
    df.to_csv(os.path.join(OUT, "runs.csv"), index=False)
    keys = [
        "origin",
        "curiosity",
        "vanishing",
        "landscape",
        "agent_generation_time",
        "use_visibility_reward",
    ]
    summ = df.groupby(keys).mean(numeric_only=True).reset_index()
    summ.to_csv(os.path.join(OUT, "cell_summary.csv"), index=False)
    net = int(df["c_ideal_fin"].notna().sum())
    json.dump(
        {
            "grid": "T1 36 cells + welfare 40 cells",
            "cells": len(grid),
            "gross_runs": len(grid) * 20,
            "net_runs_ok": net,
            "wall_s": round(time.time() - t0, 1),
            "code_base": "d3503ac (post pairing-fix)",
        },
        open(os.path.join(OUT, "manifest.json"), "w"),
        indent=1,
    )
    print(
        "GROSS cells=%d runs=%d | NET ok=%d | wall=%.0fs"
        % (len(grid), len(grid) * 20, net, time.time() - t0)
    )
