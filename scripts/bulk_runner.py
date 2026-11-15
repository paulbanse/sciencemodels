"""Thin entry point: bulk parametric sweep (ADR-0002).

The 8-setup design is not carried here: it lives in
``sciencemodels.sweeps.BULK_SETUPS`` (ADR-0002: one home per parameter setup).
This script keeps only what is runner-specific - the output folder, the repeat
count and the step limit it passes in. Run from anywhere:

    python scripts/bulk_runner.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from joblib import Parallel, delayed

from sciencemodels.model import generate_data_parametric_exploration
from sciencemodels.sweeps import BULK_SETUPS, bulk_param_grid, bulk_setup

warnings.simplefilter(action="ignore", category=FutureWarning)

# setup map: 0|1 randominit, 2|3 two gaussians, 4|5 six gaussians, 6|7 noisy gaussian
chosen_setup = 3

def generate_on_run(chosen_setup, foldername=""):
    print(f"Starting setup {chosen_setup}")
    if not 0 <= chosen_setup < len(BULK_SETUPS):
        print("chosen setup number non recognized")
        exit()
    setup = bulk_setup(chosen_setup)
    all_params = bulk_param_grid(setup, step_limit=400)

    filename = "data/" + foldername + "/run_number" + str(chosen_setup)
    print(filename, setup.generation_params)
    generate_data_parametric_exploration(
        filename,
        param_grid=all_params,
        repeats_per_setting=40,
        change_landscape_seed=True,
        intention="w",
        skip_to=0,
        longitudinal=False,
    )

foldername = "large_run_with_ideal"
os.makedirs("data/" + foldername, exist_ok=True)
with (
    open(__file__) as fp,
    open("data/" + foldername + "/creationscript.txt", "w") as tp,
):
    tp.writelines(fp.readlines())

Parallel(n_jobs=10, return_as="list")(
    [delayed(generate_on_run)(i, foldername) for i in range(len(BULK_SETUPS))]
)
