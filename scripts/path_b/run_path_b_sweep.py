"""Path B Gate 2 sweep (2026-11-14): the dyadic collaborator-choice layer.

ADR-0002: driver code lives in scripts/path_b/, run outputs in
data/path_b_v1/. NOT CALIBRATED: model-internal only (owner ruling
2026-11-13); the Klann moments (team size mean 2.65 / median 2, repeated-
collab share 0.34) are diagnostic reference values for the Gate 2
evaluation, never fitting targets.

Grid: lambda in {0, 0.25, 0.5, 0.75, 1.0} x formulation {logit, threshold}
(Q1: both, primary + robustness twin) x credit {structural, equal_split}
(Q8) x landscape {mgstd5, noisypr5} = 40 cells, 20 seeds, 800 runs,
step_limit 200, agent_generation_time 0.1, initial_curiosity 0.5.

Checkpointed per cell (ADR-0001 field-execution hygiene): atomic per-cell
CSVs under data/path_b_v1/cells/ (tmp -> row-count check is implicit in the
skip-on-restart line count -> os.replace), append-only run log, manifest.
A crashed seed is recorded in the log and manifest, never silently dropped.

Run:
    MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/path_b/run_path_b_sweep.py
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sciencemodels import landscapes
from sciencemodels.model import MyModel

OUT_DIR = REPO / "data" / "path_b_v1"
CELLS_DIR = OUT_DIR / "cells"
LOG_PATH = OUT_DIR / "run_log.txt"

LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
FORMULATIONS = ("logit", "threshold")
CREDITS = ("structural", "equal_split")
LANDSCAPES = ("mgstd5", "noisypr5")
LANDSCAPE_BUNDLES = {
    # waves 1-2 bundles, mirroring bulk_runner setups 2/3 and 6/7
    "mgstd5": (landscapes.multipleGaussians, {"std": 5, "number_gaussians": 2}),
    "noisypr5": (
        landscapes.noisyGaussian,
        {"prop_random": 5, "std": 5, "std_curiosity": 0.1},
    ),
}
SEEDS = 20
STEP_LIMIT = 200

BASE_PARAMS = {
    "n_agents": 20,
    "initial_curiosity": 0.5,
    "epsilon": 0.1,
    "harvest": 0.02,
    "grid_size": 20,
    "use_distance": True,
    "agent_generation_time": 0.1,
    "new_questions": 0,
    "agent_generation_function": lambda cur, _rng, _params: cur,
    "vanishing_factor_for_prestige": 0.9,
    "use_visibility_reward": True,
    "knowledge_normalization": "share",
    "prestige_accrual": "harvested",
    "entry_prestige_stock": "prestige",
    "dyadic_proposal_regime": "newcomer",
    "closeness_mode": "grid",
    "dyadic_visibility_stock": "prestige_visibility",
}

# pins that never reach the CSV as such (constants of the design)
DESIGN_PINS = {
    "n_agents": 20,
    "grid_size": 20,
    "step_limit": 200,
    "epsilon": 0.1,
    "harvest": 0.02,
    "initial_curiosity": 0.5,
    "agent_generation_time": 0.1,
    "vanishing_factor_for_prestige": 0.9,
    "use_distance": True,
    "use_visibility_reward": True,
    "new_questions": 0,
    "knowledge_normalization": "share",
    "prestige_accrual": "harvested",
    "entry_prestige_stock": "prestige",
    "agent_generation_function": "identity (curiosity pinned at 0.5)",
    "dyadic_proposal_regime": "newcomer",
    "closeness_mode": "grid",
    "dyadic_visibility_stock": "prestige_visibility",
    "seed_semantics": "agent_seed 0..19 drives agent and landscape RNG jointly (model injects agent_seed into generation_params)",
    "canary_anchor_mean_prestige": 0.025399760810316363,
    "team_definition": "co-harvest groups: agents sharing a tile at end of step, size >= 2; newborns count with their supervisor (they land on the supervisor's tile)",
    "repeated_definition": "per step, share of active co-harvest pairs seen active in any earlier step; 0.0 when no active pairs",
}


def cell_tag(lam, form, credit, land):
    return "cell_lam%03d_%s_%s_%s" % (
        int(round(lam * 100)),
        form,
        "struct" if credit == "structural" else "equal",
        land,
    )


def build_params(lam, form, credit, land, seed):
    f, gp = LANDSCAPE_BUNDLES[land]
    p = dict(BASE_PARAMS)
    p.update(
        cell_init_function=f,
        generation_params=dict(gp),
        agent_seed=seed,
        step_limit=STEP_LIMIT,
        dyadic_mode=form,
        dyadic_lambda=lam,
        credit_sharing=credit,
    )
    return p


def report_extras(m, row):
    """Driver-side per-run columns computed from the model's own
    datacollector (one home per measure: the model reports, the driver
    aggregates the trajectory). Shared with run_baseline_off.py."""
    dc = m.datacollector.get_model_vars_dataframe()
    row["agents_final"] = len(m._scientists)
    row["newcomers_final"] = int(sum(1 for a in m._scientists if a.is_newcomer))
    for src, dst in (
        ("mean_team_size", "team_tm"),
        ("repeated_collab_share", "rep_tm"),
        ("dyadic_assortativity_prestige", "assort_tm"),
        ("n_active_links", "links_tm"),
        ("link_repeated_share", "link_rep_tm"),
        ("link_assortativity_prestige", "link_assort_tm"),
        ("Corr_prestige_ideal_merit", "c_ideal_tm"),
        ("Corr_prestige_local_merit", "c_local_tm"),
    ):
        row[dst] = float(dc[src].mean())
    row["rep_mass0"] = float((dc["repeated_collab_share"] == 0).mean())
    row["rep_mass1"] = float((dc["repeated_collab_share"] == 1).mean())
    row["ideal_loss_cum"] = -float(
        sum(getattr(a, "ideal_merit_cum", 0.0) for a in m._scientists)
    )
    row["ideal_loss_mean"] = -float(
        np.mean([getattr(a, "ideal_merit_cum", 0.0) for a in m._scientists])
    )
    return row


def run_one(lam, form, credit, land, seed):
    m = MyModel(**build_params(lam, form, credit, land, seed))
    for _ in range(STEP_LIMIT):
        m.step(True)
    row = m.get_report()
    return report_extras(m, row)


def run_cell(spec):
    lam, form, credit, land = spec
    tag = cell_tag(lam, form, credit, land)
    path = CELLS_DIR / (tag + ".csv")
    if path.exists() and sum(1 for _ in open(path)) == SEEDS + 1:
        return tag, "skipped", []
    rows, failed = [], []
    for seed in range(SEEDS):
        try:
            rows.append(run_one(lam, form, credit, land, seed))
        except Exception as exc:  # a crashed seed is reported, not dropped
            failed.append((seed, repr(exc)[:300]))
    if rows:
        df = pd.DataFrame(rows)
        tmp = str(path) + ".tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)  # atomic: a crash leaves no partial cell
    with open(LOG_PATH, "a") as f:
        f.write(
            "%s %s rows=%d failed_seeds=%d\n"
            % (tag, "ok" if rows else "FAILED", len(rows), len(failed))
        )
        for seed, err in failed:
            f.write("  seed %d: %s\n" % (seed, err))
    return tag, ("ok" if rows else "FAILED"), failed


def main():
    os.makedirs(CELLS_DIR, exist_ok=True)
    canary = os.system(
        "MPLBACKEND=Agg %s %s > %s/canary.txt 2>&1"
        % (sys.executable, REPO / "tests" / "test_smoke.py", OUT_DIR)
    )
    assert canary == 0, "canary failed - aborting before any run"
    grid = list(itertools.product(LAMBDAS, FORMULATIONS, CREDITS, LANDSCAPES))
    t0 = time.time()
    results = Parallel(n_jobs=3, verbose=5)(delayed(run_cell)(c) for c in grid)
    ok = sum(1 for _, st, _ in results if st == "ok")
    skipped = sum(1 for _, st, _ in results if st == "skipped")
    failed_cells = sorted(t for t, st, _ in results if st == "FAILED")
    failed_seeds = sum(len(fl) for _, _, fl in results)
    # recount from disk: evidence over memory
    rows_net = 0
    for f in sorted(CELLS_DIR.glob("cell_*.csv")):
        rows_net += sum(1 for _ in open(f)) - 1
    manifest = {
        "grid": "5 lambda x 2 formulation x 2 credit x 2 landscape = 40 cells, 20 seeds, 800 runs",
        "cells_gross": len(grid),
        "cells_new": ok,
        "cells_skipped": skipped,
        "cells_failed": failed_cells,
        "runs_gross": len(grid) * SEEDS,
        "rows_net": rows_net,
        "failed_seeds": failed_seeds,
        "wall_s": round(time.time() - t0, 1),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "design_pins": DESIGN_PINS,
        "code_base": "Path B dyadic layer (dyadic_mode/logit+threshold, dyadic_lambda, credit_sharing, closeness_mode, dyadic_visibility_stock; N1 pairing fix d3503ac underneath)",
        "canary": "tests/test_smoke.py incl. Check 10 (off-mode bit-identity + anchor), gate exit 0",
    }
    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print(
        "GROSS cells=%d runs=%d | NET ok=%d skipped=%d failed_cells=%d"
        " failed_seeds=%d rows_net=%d/%d | wall=%.0fs"
        % (
            len(grid),
            len(grid) * SEEDS,
            ok,
            skipped,
            len(failed_cells),
            failed_seeds,
            rows_net,
            len(grid) * SEEDS,
            time.time() - t0,
        )
    )
    print("EXIT=0")


if __name__ == "__main__":
    main()
