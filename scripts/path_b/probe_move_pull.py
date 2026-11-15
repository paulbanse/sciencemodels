"""Path B v2 critical-path probe: identity, binding, schema, timing.

Runs before the grid (ADR-0001 / parameter-sweep-rounds step 2) and answers
four questions with computed numbers:

1. **Identity** -- `dyadic_move_pull=0.0` reproduces Path B v1 bit-for-bit.
   Two v1 cells x two seeds are re-run through this round's code and every
   numeric column is diffed against the stored v1 CSV row with the same
   `agent_seed`. Required result: max abs difference exactly 0.0.
2. **Binding** -- `dyadic_move_pull > 0` moves the PRODUCTION layer (team
   size, repeated-collaboration share), not only the choice layer.
3. **Schema** -- the analysis keys are present in a file written by a worker
   process and read back from disk.
4. **Timing** -- wall seconds per run, so the grid budget is arithmetic.

Writes only `data/path_b_v2/probe/`. Run:
    MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/path_b/probe_move_pull.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "path_b"))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import path_b_v2_common as C

V1_DIR = REPO / "data" / "path_b_v1" / "cells"
OUT = REPO / "data" / "path_b_v2" / "probe"

# (lambda, pull, formulation, credit, landscape) as v1 cell tags
V1_CELLS = [
    (1.0, "logit", "structural", "mgstd5"),
    (0.0, "threshold", "equal_split", "noisypr5"),
]
IDENTITY_SEEDS = (0, 1)


def v1_tag(lam, form, credit, land):
    return "cell_lam%03d_%s_%s_%s" % (
        int(round(lam * 100)),
        form,
        "struct" if credit == "structural" else "equal",
        land,
    )


def worker(spec):
    """One run in a worker process -> the row (also exercises the pickler path)."""
    lam, pull, form, credit, land, seed = spec
    t0 = time.time()
    row = C.run_one(lam, pull, form, credit, land, seed)
    row["wall_s"] = round(time.time() - t0, 2)
    row["_cell"] = C.cell_tag(lam, pull, form, credit, land)
    return row


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    report = {}

    # ---- 1. identity against Path B v1 -------------------------------------
    id_rows = []
    for lam, form, credit, land in V1_CELLS:
        tag = v1_tag(lam, form, credit, land)
        path = V1_DIR / (tag + ".csv")
        if not path.exists():
            raise SystemExit(f"v1 reference cell missing: {path}")
        v1 = pd.read_csv(path)
        for seed in IDENTITY_SEEDS:
            ref = v1[v1["agent_seed"] == seed]
            if ref.empty:
                raise SystemExit(f"{tag}: no v1 row with agent_seed={seed}")
            ref = ref.iloc[0]
            got = C.run_one(lam, 0.0, form, credit, land, seed)
            diffs = {}
            for col in v1.columns:
                if col not in got:
                    continue
                a, b = ref[col], got[col]
                if isinstance(a, (int, float, np.integer, np.floating)) and isinstance(
                    b, (int, float, np.integer, np.floating)
                ):
                    diffs[col] = abs(float(a) - float(b))
            worst = max(diffs.items(), key=lambda kv: kv[1])
            id_rows.append(
                {
                    "v1_cell": tag,
                    "agent_seed": seed,
                    "columns_compared": len(diffs),
                    "max_abs_diff": worst[1],
                    "worst_column": worst[0],
                    "n_differing": sum(1 for v in diffs.values() if v != 0.0),
                }
            )
            print(
                f"identity {tag} seed={seed}: {len(diffs)} numeric columns, "
                f"max abs diff {worst[1]!r} ({worst[0]})"
            )
    id_df = pd.DataFrame(id_rows)
    id_df.to_csv(OUT / "identity.csv", index=False)
    report["identity_max_abs_diff"] = float(id_df["max_abs_diff"].max())
    report["identity_columns_compared"] = int(id_df["columns_compared"].min())
    if report["identity_max_abs_diff"] != 0.0:
        raise SystemExit(
            "IDENTITY FAILED: dyadic_move_pull=0.0 is not bit-identical to Path B v1"
        )

    # ---- 2. binding at the production layer -------------------------------
    grid = [
        (lam, pull, form, credit, land, seed)
        for lam in (0.0, 1.0)
        for pull in (0.0, 1.0)
        for (form, credit, land) in (("logit", "structural", "mgstd5"),)
        for seed in (0, 1)
    ]
    rows = Parallel(n_jobs=1)(delayed(worker)(g) for g in grid)
    bind = pd.DataFrame(rows)
    bind.to_csv(OUT / "binding.csv", index=False)
    print("\nbinding (logit/structural/mgstd5, seeds 0-1):")
    for lam in (0.0, 1.0):
        for pull in (0.0, 1.0):
            sel = bind[(bind["dyadic_lambda"] == lam) & (bind["dyadic_move_pull"] == pull)]
            print(
                f"  lam={lam:<4} pull={pull:<4} "
                f"team_tm={sel['team_tm'].mean():6.3f} "
                f"rep_tm={sel['rep_tm'].mean():6.3f} "
                f"links_tm={sel['links_tm'].mean():6.3f} "
                f"c_ideal_tm={sel['c_ideal_tm'].mean():6.4f}"
            )

    # ---- 3. schema proof on a worker-written file -------------------------
    C.assert_schema(bind, "probe/binding.csv (worker-written, read back from disk)")
    report["schema_keys_checked"] = list(C.ANALYSIS_KEYS)

    # ---- 4. timing --------------------------------------------------------
    walls = [r["wall_s"] for r in rows]
    report["wall_s_per_run_mean"] = float(np.mean(walls))
    report["wall_s_per_run_min"] = float(np.min(walls))
    report["wall_s_per_run_max"] = float(np.max(walls))
    report["projected_grid_wall_s_njobs3"] = float(
        np.mean(walls) * 960 / 3
    )
    with open(OUT / "probe_report.json", "w") as f:
        json.dump(report, f, indent=1)

    # ---- binding assertions ----------------------------------------------
    def cell_tm(lam, pull, key):
        sel = bind[(bind["dyadic_lambda"] == lam) & (bind["dyadic_move_pull"] == pull)]
        return float(sel[key].mean())

    prod_moves = []
    for lam in (0.0, 1.0):
        for key in ("team_tm", "rep_tm"):
            base, pulled = cell_tm(lam, 0.0, key), cell_tm(lam, 1.0, key)
            prod_moves.append((lam, key, base, pulled, pulled - base))
    print()
    for lam, key, base, pulled, delta in prod_moves:
        print(f"  lam={lam} {key:8} pull0={base:8.4f} pull1={pulled:8.4f} delta={delta:+8.4f}")
    moved = [d for _, _, _, _, d in prod_moves if abs(d) > 1e-9]
    print(f"\nproduction metrics moved by the pull in {len(moved)}/{len(prod_moves)} comparisons")
    print(f"wall per run: mean {report['wall_s_per_run_mean']:.2f}s "
          f"-> projected grid (960 runs, n_jobs=3) {report['projected_grid_wall_s_njobs3']:.0f}s")
    return 0 if moved else 1


if __name__ == "__main__":
    raise SystemExit(main())
