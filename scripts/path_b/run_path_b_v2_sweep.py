"""Path B Gate 2 v2 sweep (2026-11-15): the dyadic MOVEMENT coupling.

ADR-0002: driver code lives in scripts/path_b/, run outputs in
data/path_b_v2/. NOT CALIBRATED: model-internal only (owner ruling
2026-11-13); the Klann moments (team size mean 2.65 / median 2,
repeated-collab share 0.34) are diagnostic reference values for the Gate 2
evaluation, never fitting targets.

Grid: lambda in {0.0, 0.5, 1.0} x dyadic_move_pull in {0.5, 1.0} x
formulation in {logit, threshold} x credit in {structural, equal_split} x
landscape in {mgstd5, noisypr5} = 48 cells, 20 seeds, 960 runs,
step_limit 200, agent_generation_time 0.1, initial_curiosity 0.5.

Control arm inside the driver (parameter-sweep-rounds: a cross-round
comparison needs its control in the driver, so a regeneration re-proves
equivalence instead of assuming it): before the grid, two Path B v1 cells x
two seeds are re-run at dyadic_move_pull=0.0 and every numeric column is
diffed against the stored v1 CSV row with the same agent_seed. Any difference
aborts the launch.

Checkpointed per cell (ADR-0001): atomic per-cell CSV (tmp -> os.replace),
skip-with-verified-row-count on restart, append-only run log, manifest with
gross-vs-net recount from disk and a source snapshot (sha256 of every module
the dynamics touch, checked again after the grid).

Run:
    MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/path_b/run_path_b_v2_sweep.py
"""
from __future__ import annotations

import hashlib
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
sys.path.insert(0, str(REPO / "scripts" / "path_b"))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import path_b_v2_common as C

OUT_DIR = REPO / "data" / "path_b_v2"
CELLS_DIR = OUT_DIR / "cells"
LOG_PATH = OUT_DIR / "run_log.txt"
V1_DIR = REPO / "data" / "path_b_v1" / "cells"

LAMBDAS = (0.0, 0.5, 1.0)
PULLS = (0.5, 1.0)
FORMULATIONS = ("logit", "threshold")
CREDITS = ("structural", "equal_split")
LANDSCAPES = ("mgstd5", "noisypr5")
SEEDS = C.SEEDS
STEP_LIMIT = C.STEP_LIMIT

#: modules whose content the dynamics depend on; hashed before and after
SOURCE_FILES = (
    "sciencemodels/agents.py",
    "sciencemodels/model.py",
    "sciencemodels/metrics.py",
    "scripts/path_b/path_b_v2_common.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_sources() -> dict:
    return {f: sha256(REPO / f) for f in SOURCE_FILES}


def v1_tag(lam, form, credit, land):
    return "cell_lam%03d_%s_%s_%s" % (
        int(round(lam * 100)),
        form,
        "struct" if credit == "structural" else "equal",
        land,
    )


def control_arm() -> dict:
    """Re-run v1 cells at pull=0.0 and diff bit-exactly against v1 artifacts."""
    cases = [
        (1.0, "logit", "structural", "mgstd5"),
        (0.0, "threshold", "equal_split", "noisypr5"),
    ]
    rows = []
    for lam, form, credit, land in cases:
        path = V1_DIR / (v1_tag(lam, form, credit, land) + ".csv")
        if not path.exists():
            raise SystemExit(
                f"control arm: v1 reference missing ({path}); cannot prove identity"
            )
        v1 = pd.read_csv(path)
        for seed in (0, 1):
            ref = v1[v1["agent_seed"] == seed].iloc[0]
            got = C.run_one(lam, 0.0, form, credit, land, seed)
            diffs = {
                col: abs(float(ref[col]) - float(got[col]))
                for col in v1.columns
                if col in got
                and isinstance(ref[col], (int, float, np.integer, np.floating))
                and isinstance(got[col], (int, float, np.integer, np.floating))
            }
            worst_col = max(diffs, key=lambda k: diffs[k])
            rows.append(
                {
                    "v1_cell": v1_tag(lam, form, credit, land),
                    "agent_seed": seed,
                    "columns_compared": len(diffs),
                    "max_abs_diff": diffs[worst_col],
                    "worst_column": worst_col,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "control_v1_recheck.csv", index=False)
    worst = float(df["max_abs_diff"].max())
    if worst != 0.0:
        raise SystemExit(f"control arm FAILED: max abs diff {worst!r} vs Path B v1")
    print(
        "control arm OK: %d v1 reruns x %d columns, max abs diff 0.0"
        % (len(df), int(df["columns_compared"].min()))
    )
    return {
        "control_reference": "data/path_b_v1/cells/*.csv",
        "control_cases": int(len(df)),
        "control_columns_min": int(df["columns_compared"].min()),
        "control_max_abs_diff": worst,
    }


def run_cell(spec):
    lam, pull, form, credit, land = spec
    tag = C.cell_tag(lam, pull, form, credit, land)
    path = CELLS_DIR / (tag + ".csv")
    if path.exists() and sum(1 for _ in open(path)) == SEEDS + 1:
        return tag, "skipped", []
    rows, failed = [], []
    for seed in range(SEEDS):
        try:
            rows.append(C.run_one(lam, pull, form, credit, land, seed))
        except Exception as exc:  # a crashed seed is reported, never dropped
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


def main() -> int:
    os.makedirs(CELLS_DIR, exist_ok=True)
    os.makedirs(OUT_DIR / "probe", exist_ok=True)

    canary_rc = os.system(
        "MPLBACKEND=Agg %s %s > %s/canary.txt 2>&1"
        % (sys.executable, REPO / "tests" / "test_smoke.py", OUT_DIR)
    )
    if canary_rc != 0:
        raise SystemExit("canary failed - aborting before any run")

    snap_before = snapshot_sources()
    control = control_arm()

    grid = list(
        itertools.product(LAMBDAS, PULLS, FORMULATIONS, CREDITS, LANDSCAPES)
    )
    t0 = time.time()
    results = Parallel(n_jobs=3, verbose=5)(delayed(run_cell)(c) for c in grid)
    wall = time.time() - t0

    ok = sum(1 for _, st, _ in results if st == "ok")
    skipped = sum(1 for _, st, _ in results if st == "skipped")
    failed_cells = sorted(t for t, st, _ in results if st == "FAILED")
    failed_seeds = sum(len(fl) for _, _, fl in results)

    rows_net = 0
    for f in sorted(CELLS_DIR.glob("cell_*.csv")):
        rows_net += sum(1 for _ in open(f)) - 1

    snap_after = snapshot_sources()
    moved = [f for f in SOURCE_FILES if snap_before[f] != snap_after[f]]

    manifest = {
        "round": "Path B Gate 2 v2 (movement coupling)",
        "grid": "3 lambda x 2 dyadic_move_pull x 2 formulation x 2 credit x 2 landscape = 48 cells, 20 seeds, 960 runs",
        "cells_gross": len(grid),
        "cells_new": ok,
        "cells_skipped": skipped,
        "cells_failed": failed_cells,
        "runs_gross": len(grid) * SEEDS,
        "rows_net": rows_net,
        "failed_seeds": failed_seeds,
        "wall_s": round(wall, 1),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "design_pins": C.DESIGN_PINS,
        "control_arm": control,
        "source_snapshot_before": snap_before,
        "source_snapshot_after": snap_after,
        "sources_moved_during_run": moved,
        "code_base": "Path B dyadic layer + dyadic_move_pull (this round); N1 pairing fix d3503ac underneath",
        "canary": "tests/test_smoke.py incl. Check 10 + the movement-coupling checks, gate exit 0",
    }
    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)

    print(
        "GROSS cells=%d runs=%d | NET ok=%d skipped=%d failed_cells=%d"
        " failed_seeds=%d rows_net=%d/%d | wall=%.0fs | sources_moved=%s"
        % (
            len(grid),
            len(grid) * SEEDS,
            ok,
            skipped,
            len(failed_cells),
            failed_seeds,
            rows_net,
            len(grid) * SEEDS,
            wall,
            moved or "none",
        )
    )
    print("EXIT=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
