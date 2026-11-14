"""Off-mode structural baseline for the Gate 2 evaluation (Path B).

The same grid skeleton as the sweep (2 landscapes x 20 seeds, step_limit
200, agent_generation_time 0.1, initial_curiosity 0.5) with
dyadic_mode="off" and credit_sharing="structural": co-harvest teams,
prestige dynamics and the wedge WITHOUT any choice layer. This is the
counterfactual the lambda response is read against, and it doubles as the
production-layer moments' structural floor (co-location under the existing
visibility coupling alone).

Writes data/path_b_v1/baseline_off.csv. Simulation code writing a data/ CSV
(ADR-0001 separation); analysis reads it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sciencemodels.model import MyModel  # noqa: E402

import run_path_b_sweep as drv  # noqa: E402

OUT_CSV = drv.OUT_DIR / "baseline_off.csv"


def run_one_off(land, seed):
    f, gp = drv.LANDSCAPE_BUNDLES[land]
    params = dict(drv.BASE_PARAMS)
    params.update(
        cell_init_function=f,
        generation_params=dict(gp),
        agent_seed=seed,
        step_limit=drv.STEP_LIMIT,
        dyadic_mode="off",
        credit_sharing="structural",
    )
    m = MyModel(**params)
    for _ in range(drv.STEP_LIMIT):
        m.step(True)
    row = m.get_report()
    row = drv.report_extras(m, row)
    row["land"] = land
    return row


def main():
    drv.OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for land in drv.LANDSCAPES:
        for seed in range(drv.SEEDS):
            rows.append(run_one_off(land, seed))
            print("baseline off: %s seed %d done" % (land, seed))
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print("wrote %s (%d rows)" % (OUT_CSV, len(rows)))


if __name__ == "__main__":
    main()
