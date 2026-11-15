"""Fresh 8-setup parametric sweep for the figures-v1 regeneration (ADR-0002 driver).

Replicates the setup design of ``scripts/bulk_runner.py`` (4 landscape families x
fixed/mixed curiosity market, ``chosen_setup = 0..7``) but with the figures-v1
run parameters:

    repeats_per_setting = 10, step_limit = 200, change_landscape_seed = True

Outputs (one CSV per setup) land in ``data/figures_v1/run_number<k>.csv``.

Restart safety: each setup is written to ``.tmp_run_number<k>.csv`` through the
production interface ``generate_data_parametric_exploration`` and renamed onto
``run_number<k>.csv`` only after the setup's row count matches the gross
expectation, so a timeout loses at most one setup and never mixes partial rows
into a finished file. A setup whose final file already holds the expected rows
is skipped.

The setup table is not carried here: it is ``sciencemodels.sweeps.BULK_SETUPS``,
imported, and shared with ``scripts/bulk_runner.py`` (ADR-0002: one home per
parameter setup). This driver adds only its own repeats and step limit.

Run from anywhere:

    MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/figures/run_fresh_sweep.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from joblib import Parallel, delayed

from sciencemodels.model import generate_data_parametric_exploration
from sciencemodels.sweeps import BULK_SETUPS, bulk_param_grid, bulk_setup

OUT_DIR = "data/figures_v1"
REPEATS = 10
STEP_LIMIT = 200
N_JOBS = 3


def param_grid(chosen_setup):
    """The shared bulk parameter dict for one setup, with the figures-v1 step limit."""
    return bulk_param_grid(bulk_setup(chosen_setup), step_limit=STEP_LIMIT)


def n_param_sets(grid):
    """Number of distinct parameter combinations, as the production loop counts them."""
    lists = [a for a in grid if type(grid[a]) == list]
    varying = [a for a in lists if len(grid[a]) > 1]
    if not varying:
        return 1
    n = 1
    for a in varying:
        n *= len(grid[a])
    return n


def gross_runs(chosen_setup):
    return n_param_sets(param_grid(chosen_setup)) * REPEATS


def completed_rows(path):
    if not os.path.isfile(path):
        return 0
    with open(path, newline="") as f:
        return sum(1 for _ in csv.reader(f)) - 1  # minus header


def run_setup(chosen_setup):
    final = f"{OUT_DIR}/run_number{chosen_setup}.csv"
    tmp_prefix = f"{OUT_DIR}/.tmp_run_number{chosen_setup}"
    tmp = tmp_prefix + ".csv"
    expected = gross_runs(chosen_setup)
    if completed_rows(final) == expected:
        return {"setup": chosen_setup, "family": bulk_setup(chosen_setup).family,
                "status": "skipped-complete", "gross": expected,
                "net": completed_rows(final), "seconds": 0.0}
    if os.path.exists(tmp):
        os.remove(tmp)
    t0 = time.time()
    generate_data_parametric_exploration(
        tmp_prefix,
        param_grid=param_grid(chosen_setup),
        repeats_per_setting=REPEATS,
        change_landscape_seed=True,
        intention="w",
        skip_to=0,
        longitudinal=False,
    )
    net = completed_rows(tmp)
    if net != expected:
        raise RuntimeError(
            f"setup {chosen_setup}: wrote {net} rows, expected {expected}"
        )
    os.replace(tmp, final)
    return {"setup": chosen_setup, "family": bulk_setup(chosen_setup).family,
            "status": "run", "gross": expected, "net": net,
            "seconds": round(time.time() - t0, 1)}


def main():
    os.makedirs(f"{OUT_DIR}/figures", exist_ok=True)
    results = Parallel(n_jobs=N_JOBS, return_as="list")(
        delayed(run_setup)(k) for k in range(len(BULK_SETUPS))
    )
    manifest = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_parameters": {
            "repeats_per_setting": REPEATS,
            "step_limit": STEP_LIMIT,
            "change_landscape_seed": True,
            "n_jobs": N_JOBS,
        },
        "gross_runs_expected": sum(r["gross"] for r in results),
        "net_runs_completed": sum(r["net"] for r in results),
        "setups": results,
    }
    with open(f"{OUT_DIR}/sweep_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
