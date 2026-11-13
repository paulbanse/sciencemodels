"""Welfare probe v1 - Path C prototype sweep driver.

ADR-0002: driver/analysis code lives in scripts/, run outputs in data/.
NOT CALIBRATED: model-internal exploration only (owner ruling 2026-11-13).

Design (see docs/welfare_probe_v1.md):

* 40 cells = initial_curiosity {0.1,0.3,0.5,0.7,0.9}
              x use_visibility_reward {True,False}
              x agent_generation_time {0,0.1}
              x landscape leg {2,6 bumps, std 5}
* 20 seeds each (agent_seed = repeat; the model derives the landscape seed
  from agent_seed, so seed s draws one landscape+agent draw per cell),
  step_limit 200, grid 20x20, n_agents 20 -> 800 runs.
* Sweep interface: sciencemodels.model.generate_data_parametric_exploration,
  one run-tagged CSV per cell under data/welfare_v1/.
* The model does not report ideal merit or the final wedge correlation, so
  this driver wraps sciencemodels.io.build_report (module attribute, no repo
  file is edited) with four extra row keys:
      ideal_merit_sum, ideal_merit_mean,
      corr_prestige_ideal_merit_final, corr_prestige_local_merit_final
* Checkpointing: one state/<tag>.json per cell, written atomically after the
  CSV row count is verified; a completed cell is skipped on restart.

Run:
    MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/welfare/run_welfare_sweep.py --probe
    MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/welfare/run_welfare_sweep.py
"""
from __future__ import annotations

import argparse
import csv
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
from joblib import Parallel, delayed

from sciencemodels import landscapes
import sciencemodels.io as _io
from sciencemodels.model import generate_data_parametric_exploration

# --- report patch (driver-local; no repository file is modified) ------------
# joblib/cloudpickle ships `run_cell` to the worker by value and does NOT
# re-execute this module there, so a module-level patch would silently never
# install. The patch is therefore installed from inside `run_cell`, once per
# process, against the module attribute `model.get_report` resolves at call
# time (`from . import io` in sciencemodels/model.py).
def _ensure_patch() -> None:
    if getattr(_io, "_welfare_patch_installed", False):
        return
    original = _io.build_report

    def _welfare_build_report(model):
        row = original(model)
        df = model.datacollector.get_model_vars_dataframe()
        merits = [float(a.ideal_Merit) for a in model.agents]
        row["ideal_merit_sum"] = float(np.sum(merits))
        row["ideal_merit_mean"] = float(np.mean(merits))
        row["corr_prestige_ideal_merit_final"] = float(
            df["Corr_prestige_ideal_merit"].iloc[-1]
        )
        row["corr_prestige_local_merit_final"] = float(
            df["Corr_prestige_local_merit"].iloc[-1]
        )
        return row

    _io.build_report = _welfare_build_report
    _io._welfare_patch_installed = True

# --- pinned design ----------------------------------------------------------
CURIOSITIES = (0.1, 0.3, 0.5, 0.7, 0.9)
VISIBILITY = (True, False)
ENTRY = (0.0, 0.1)
LEGS = (2, 6)
GAUSSIAN_STD = 5
STEP_LIMIT = 200
GRID_SIZE = 20
N_AGENTS = 20
SEEDS = 20

def curiosity_is_initial(curiosity, _rng, _params):
    """`agent_generation_function` that lets `initial_curiosity` bind.

    `landscapes.uniform(_mu, ...)` deliberately ignores its first argument, so
    under it every agent is redrawn from U(0,1) and `initial_curiosity` never
    enters the dynamics - phase 1 of this probe produced five curiosity cells
    with bit-identical outcomes in every metric. `scripts/bulk_runner.py` uses
    this same identity form whenever it sweeps curiosity; the sweep here needs
    the visibility weight (1 - curiosity) to be the cell's x-axis, so it uses
    the identity form too.
    """
    return curiosity


OUT_DIR = REPO / "data" / "welfare_v1"
STATE_DIR = OUT_DIR / "state"
FIG_DIR = OUT_DIR / "figures"
LOG_PATH = OUT_DIR / "run_log.txt"

BASE_PARAMS = {
    "n_agents": N_AGENTS,
    "epsilon": 0.1,
    "harvest": 0.02,
    "grid_size": GRID_SIZE,
    "cell_init_function": landscapes.multipleGaussians,
    "use_distance": True,
    "new_questions": 0,
    "agent_seed": 0,
    "step_limit": STEP_LIMIT,
    "agent_generation_function": curiosity_is_initial,
    "vanishing_factor_for_prestige": 0.5,
    "knowledge_normalization": "share",
    "prestige_accrual": "harvested",
}


def tag_for(curiosity: float, visibility: bool, entry: float, leg: int) -> str:
    return (
        f"cell_cur{int(round(curiosity * 10)):02d}"
        f"_vis{1 if visibility else 0}"
        f"_gen{int(round(entry * 10)):02d}"
        f"_mg{leg}"
    )


def build_cells():
    cells = []
    for curiosity in CURIOSITIES:
        for visibility in VISIBILITY:
            for entry in ENTRY:
                for leg in LEGS:
                    cells.append(
                        {
                            "tag": tag_for(curiosity, visibility, entry, leg),
                            "curiosity": curiosity,
                            "visibility_weight": 1.0 - curiosity,
                            "use_visibility_reward": visibility,
                            "agent_generation_time": entry,
                            "number_gaussians": leg,
                        }
                    )
    return cells


def param_grid_for(cell) -> dict:
    grid = dict(BASE_PARAMS)
    grid.update(
        {
            "initial_curiosity": cell["curiosity"],
            "use_visibility_reward": cell["use_visibility_reward"],
            "agent_generation_time": cell["agent_generation_time"],
            "generation_params": {
                "std": GAUSSIAN_STD,
                "number_gaussians": cell["number_gaussians"],
            },
        }
    )
    return grid


def count_rows(path: Path) -> int:
    """Data rows (excluding the header) of a generated CSV."""
    with open(path, newline="") as fh:
        n = sum(1 for _ in csv.reader(fh))
    return max(n - 1, 0)


def is_done(cell, seeds: int) -> bool:
    csv_path = OUT_DIR / f"{cell['tag']}.csv"
    state_path = STATE_DIR / f"{cell['tag']}.json"
    if not (csv_path.exists() and state_path.exists()):
        return False
    try:
        state = json.loads(state_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return state.get("status") == "ok" and state.get("rows") == seeds == count_rows(
        csv_path
    )


def run_cell(cell, seeds: int) -> dict:
    _ensure_patch()
    tag = cell["tag"]
    csv_path = OUT_DIR / f"{tag}.csv"
    state_path = STATE_DIR / f"{tag}.json"
    if is_done(cell, seeds):
        return {"tag": tag, "status": "skipped", "rows": seeds, "wall_s": 0.0}

    started = time.time()
    generate_data_parametric_exploration(
        str(OUT_DIR / tag),
        param_grid_for(cell),
        repeats_per_setting=seeds,
        change_landscape_seed=False,
        intention="w",
        skip_to=0,
        longitudinal=False,
    )
    wall = time.time() - started
    rows = count_rows(csv_path) if csv_path.exists() else -1
    status = "ok" if rows == seeds else "row_mismatch"

    state = {
        "tag": tag,
        "status": status,
        "rows": rows,
        "expected_rows": seeds,
        "wall_s": round(wall, 2),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **{
            k: cell[k]
            for k in (
                "curiosity",
                "visibility_weight",
                "use_visibility_reward",
                "agent_generation_time",
                "number_gaussians",
            )
        },
    }
    tmp = state_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, state_path)

    line = (
        f"[{state['finished_at']}] {tag} -> {status} rows={rows}/{seeds} "
        f"wall={wall:.1f}s\n"
    )
    with open(LOG_PATH, "a") as fh:
        fh.write(line)
    return {"tag": tag, "status": status, "rows": rows, "wall_s": round(wall, 2)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Welfare probe v1 sweep")
    parser.add_argument("--probe", action="store_true",
                        help="one cell, two seeds (critical-path check)")
    parser.add_argument("--seeds", type=int, default=SEEDS)
    parser.add_argument("--n-jobs", type=int, default=3,
                        help="cap 3 (concurrent-sweep hygiene)")
    parser.add_argument("--limit", type=int, default=None,
                        help="run only the first N cells")
    args = parser.parse_args()

    seeds = 2 if args.probe else args.seeds
    n_jobs = min(args.n_jobs, 3)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # self-describing outputs: constants that do not appear in the CSVs
    design = {
        "run": "welfare_v1",
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "not_calibrated": True,
        "cells": "curiosity {0.1,0.3,0.5,0.7,0.9} x use_visibility_reward "
                 "{True,False} x agent_generation_time {0,0.1} x "
                 "number_gaussians {2,6}",
        "visibility_weight": "1 - initial_curiosity",
        "seeds_per_cell": seeds,
        "seed": "agent_seed = repeat 0..n-1; the model derives the landscape "
                "seed from agent_seed",
        "fixed_params": {
            k: (v.__name__ if callable(v) else v)
            for k, v in BASE_PARAMS.items()
            if k not in ("cell_init_function", "agent_generation_function")
        }
        | {
            "cell_init_function": BASE_PARAMS["cell_init_function"].__name__,
            "gaussian_std": GAUSSIAN_STD,
            "initial_curiosity": "varied per cell (see filename tag)",
            "use_visibility_reward": "varied per cell (see filename tag)",
            "agent_generation_time": "varied per cell (see filename tag)",
            "number_gaussians": "varied per cell (see filename tag)",
        },
        "note": "phase 1 of this run folder used "
                "agent_generation_function=landscapes.uniform, under which "
                "initial_curiosity is inert; it is archived in "
                "superseded_uniform/ and superseded by this design.",
    }
    (OUT_DIR / "design.json").write_text(json.dumps(design, indent=2))

    cells = build_cells()
    if args.probe:
        cells = [c for c in cells if c["tag"] == "cell_cur05_vis1_gen00_mg2"]
    if args.limit is not None:
        cells = cells[: args.limit]

    todo = [c for c in cells if not is_done(c, seeds)]
    print(
        f"design: {len(cells)} cells x {seeds} seeds "
        f"({len(cells) * seeds} runs); already complete: "
        f"{len(cells) - len(todo)}; to run: {len(todo)}; n_jobs={n_jobs}",
        flush=True,
    )

    results = Parallel(n_jobs=n_jobs)(delayed(run_cell)(c, seeds) for c in todo)

    ok = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skipped"]
    bad = [r for r in results if r["status"] not in ("ok", "skipped")]
    gross = len(cells) * seeds
    net = sum(r["rows"] for r in results if r["rows"] > 0) + (
        len(cells) - len(todo)
    ) * seeds
    print(
        f"GROSS cells={len(cells)} runs={gross} | NET ok={len(ok)} "
        f"skipped={len(skipped)} failed={len(bad)} rows_net={net}/{gross}",
        flush=True,
    )
    for r in bad:
        print(f"  FAILED {r}", flush=True)
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
