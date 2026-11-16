"""Figure 1 extension: the missing xi = 1 - c grid points on mgstd5.

The two legs of `F6_two_stage_proxy_failure` stopped short of the ends of the
visibility axis. This round fills them in WITHOUT touching `sciencemodels/`:

- **M1 + turnover** (`age_hazard`, H=400, entry=0.1): two new curiosity cells,
  c in {0.1, 0.9} (xi in {0.9, 0.1}), 20 seeds each. The three interior cells
  (c in {0.3, 0.5, 0.7}) already exist in `data/turnover_v1/`; the figure reads
  them from there and these two from here.
- **M2** (`dyadic_move_pull=1.0`, `logit`, `equal_split`, H=200, entry=0.1):
  c in {0.1, 0.3, 0.7, 0.9} x lambda in {0.0, 1.0}, 20 seeds each (160 runs).
  The c = 0.5 cells already exist in `data/path_b_v2/`.

Every run goes through the ARCHIVED round drivers' own entry points --
`run_turnover_v1.run_one` and `path_b_v2_common.build_params` -- so the setup
is the archived setup by construction rather than by re-derivation. The single
driver-side difference is the one this round exists to make: the curiosity of
the cell (Path B v2 pinned it at 0.5). It is applied as a one-key override of
the round's parameter dict and recorded in `design.json`.

Two arms, two processes. `install_exit_rule` patches a CLASS attribute on
`MyModel` and a joblib worker outlives a cell, so the turnover patch would
otherwise follow a worker into an M2 cell. The arms therefore run as separate
invocations (`--arm m1` / `--arm m2`), and every M2 cell re-installs the
pristine original first -- belt and braces, both checked in the control arm.

Writes ONLY `data/fig1_extension_v1/`. Nothing in `data/turnover_v1/`,
`data/path_b_v2/` or `sciencemodels/` is modified.

    MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/fig1_extension/run_fig1_extension.py --arm both
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "turnover"))
sys.path.insert(0, str(REPO / "scripts" / "path_b"))

from joblib import Parallel, delayed  # noqa: E402

import run_turnover_v1 as TO  # noqa: E402  the archived M1 driver
import path_b_v2_common as PB  # noqa: E402  the archived M2 driver
from sciencemodels.model import MyModel  # noqa: E402

OUT = REPO / "data" / "fig1_extension_v1"
CELLS = OUT / "cells"
LOG_PATH = OUT / "run_log.txt"
DRIVER_VERSION = "fig1_extension_v1 / driver 1"

SEEDS = 20
#: the archived constants, restated here only to be ASSERTED against the
#: imported modules at startup -- if a round's constant moves, this round must
#: stop rather than silently change horizon in the middle of a figure series.
M1_HORIZON, M1_ENTRY, M1_LAND, M1_MODE = 400, 0.10, "mgstd5", "age_hazard"
M1_NEW_CURIOSITY = (0.1, 0.9)
M1_CONTROL_CURIOSITY = 0.5
M2_STEP_LIMIT, M2_ENTRY, M2_LAND = 200, 0.1, "mgstd5"
M2_PULL, M2_FORM, M2_CREDIT = 1.0, "logit", "equal_split"
M2_NEW_CURIOSITY = (0.1, 0.3, 0.7, 0.9)
M2_CONTROL_CURIOSITY = 0.5
LAMBDAS = (0.0, 1.0)

SOURCE_FILES = (
    "sciencemodels/model.py",
    "sciencemodels/agents.py",
    "sciencemodels/metrics.py",
    "sciencemodels/landscapes.py",
    "scripts/turnover/run_turnover_v1.py",
    "scripts/path_b/path_b_v2_common.py",
)


# --------------------------------------------------------------- provenance
def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot_sources() -> dict:
    return {f: sha256(REPO / f) for f in SOURCE_FILES}


def assert_archived_constants() -> None:
    """Refuse to run against a round whose constants moved underneath us."""
    checks = [
        (TO.HORIZON == M1_HORIZON, f"turnover horizon {TO.HORIZON} != {M1_HORIZON}"),
        (TO.SEEDS == SEEDS, f"turnover seeds {TO.SEEDS} != {SEEDS}"),
        (TO.TAU_MAX == 200, f"turnover tau_max {TO.TAU_MAX} != 200"),
        (M1_ENTRY in TO.ENTRY, f"entry {M1_ENTRY} not in turnover ENTRY {TO.ENTRY}"),
        (M1_MODE in TO.MODES, f"mode {M1_MODE} not in turnover MODES {TO.MODES}"),
        (M1_LAND in TO.LAND, f"landscape {M1_LAND} not in turnover LAND"),
        (PB.STEP_LIMIT == M2_STEP_LIMIT, f"path_b step_limit {PB.STEP_LIMIT} != {M2_STEP_LIMIT}"),
        (PB.SEEDS == SEEDS, f"path_b seeds {PB.SEEDS} != {SEEDS}"),
        (PB.BASE_PARAMS["agent_generation_time"] == M2_ENTRY,
         f"path_b entry {PB.BASE_PARAMS['agent_generation_time']} != {M2_ENTRY}"),
        (PB.BASE_PARAMS["initial_curiosity"] == M2_CONTROL_CURIOSITY,
         "path_b base curiosity is no longer 0.5; the control arm no longer "
         "reproduces the archived c=0.5 cells"),
        (M2_LAND in PB.LANDSCAPE_BUNDLES, f"landscape {M2_LAND} not in path_b bundles"),
    ]
    bad = [msg for cond, msg in checks if not cond]
    if bad:
        raise SystemExit("archived-constant divergence (refusing to run):\n  " + "\n  ".join(bad))


def restore_model() -> None:
    """Put the unpatched ``__generate_new_agents`` back on ``MyModel``.

    `install_exit_rule` writes a CLASS attribute, and a loky worker serves many
    cells; without this an M2 cell could inherit the turnover exit rule and the
    M2 leg would silently be an M1 run with a dyadic layer. `_turnover_original`
    is captured by the turnover driver on its first install.
    """
    attr = "_MyModel__generate_new_agents"
    original = getattr(MyModel, "_turnover_original", None)
    if original is not None:
        setattr(MyModel, attr, original)
    MyModel._turnover_mode = None


# --------------------------------------------------------------------- cells
def m1_specs() -> list[tuple]:
    return [(M1_MODE, M1_ENTRY, c, M1_LAND) for c in M1_NEW_CURIOSITY]


def m2_specs() -> list[tuple]:
    return [(lam, c) for c, lam in itertools.product(M2_NEW_CURIOSITY, LAMBDAS)]


def m1_tag(spec: tuple) -> str:
    return "m1_" + TO.cell_tag(spec)


def m2_tag(lam: float, cur: float) -> str:
    return "m2_%s_cur%02d" % (
        PB.cell_tag(lam, M2_PULL, M2_FORM, M2_CREDIT, M2_LAND),
        int(round(cur * 10)),
    )


def m1_run_one(mode, g, cur, land, seed):
    """One turnover run -- the archived driver's own entry point."""
    row, _traj = TO.run_one(mode, g, cur, land, seed)
    row["xi"] = 1.0 - float(cur)
    row["driver_version"] = DRIVER_VERSION
    return row


def m2_run_one(lam: float, cur: float, seed: int):
    """One Path B v2 run with the cell's curiosity -- the one override.

    `build_params` is the archived builder; the ONLY key replaced is
    `initial_curiosity`, which Path B v2 pinned at 0.5 because it swept lambda,
    not curiosity. Everything else (pull 1.0, logit, equal_split, H=200,
    entry 0.1, identity newcomer rule) comes from the round unchanged.
    """
    restore_model()
    params = PB.build_params(lam, M2_PULL, M2_FORM, M2_CREDIT, M2_LAND, seed)
    assert params["initial_curiosity"] == M2_CONTROL_CURIOSITY, params["initial_curiosity"]
    params["initial_curiosity"] = float(cur)
    m = MyModel(**params)
    for _ in range(PB.STEP_LIMIT):
        m.step(True)
    row = PB.report_extras(m, m.get_report())
    row.update(
        dyadic_lambda=float(lam),
        dyadic_move_pull=float(M2_PULL),
        formulation=M2_FORM,
        credit_sharing_axis=M2_CREDIT,
        landscape=M2_LAND,
        xi=1.0 - float(cur),
        driver_version=DRIVER_VERSION,
    )
    return row


def _log(line: str) -> None:
    with open(LOG_PATH, "a") as f:
        f.write(line.rstrip("\n") + "\n")


def run_cell(kind: str, spec: tuple) -> tuple[str, str, list]:
    """One cell: 20 seeds, atomic write, restart-safe, append-only log."""
    if kind == "m1":
        tag = m1_tag(spec)
        runner = lambda s: m1_run_one(*spec, seed=s)  # noqa: E731
    else:
        lam, cur = spec
        tag = m2_tag(lam, cur)
        runner = lambda s: m2_run_one(lam, cur, s)  # noqa: E731
    path = CELLS / (tag + ".csv")
    if path.exists() and sum(1 for _ in open(path)) == SEEDS + 1:
        return tag, "skipped", []
    rows, failed = [], []
    for seed in range(SEEDS):
        try:
            rows.append(runner(seed))
        except Exception as exc:  # a crashed seed is reported, never dropped
            failed.append((seed, repr(exc)[:300]))
    if rows:
        df = pd.DataFrame(rows)
        tmp = str(path) + ".tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)  # atomic: a crash leaves no partial cell
    _log("%s %s rows=%d failed_seeds=%d" % (tag, "ok" if rows else "FAILED", len(rows), len(failed)))
    for seed, err in failed:
        _log("  seed %d: %s" % (seed, err))
    return tag, ("ok" if rows else "FAILED"), failed


# ------------------------------------------------------------- proof stages
def probe() -> dict:
    """Two seeds per arm through the production entry, read back from disk."""
    print("PROBE: one cell x 2 seeds per arm, read back from the worker file")
    out = {"m1": {}, "m2": {}}
    t0 = time.time()
    for cur in M1_NEW_CURIOSITY:
        spec = (M1_MODE, M1_ENTRY, cur, M1_LAND)
        rows = [m1_run_one(*spec, seed=s) for s in range(2)]
        p = CELLS / (m1_tag(spec) + "_probe.csv")
        pd.DataFrame(rows).to_csv(p, index=False)
        back = pd.read_csv(p)
        for key in ("curiosity", "xi", "harvested_fin", "c_ideal_fin", "n_final",
                    "mean_age_final", "seed"):
            assert key in back.columns, f"probe m1 c={cur}: missing {key!r}"
        out["m1"][cur] = dict(rows=len(back),
                              c_ideal_fin=float(back.c_ideal_fin.mean()),
                              harvested_fin=float(back.harvested_fin.mean()))
        p.unlink()
    for lam in LAMBDAS:
        for cur in (0.1, 0.9):
            rows = [m2_run_one(lam, cur, s) for s in range(2)]
            p = CELLS / (m2_tag(lam, cur) + "_probe.csv")
            pd.DataFrame(rows).to_csv(p, index=False)
            back = pd.read_csv(p)
            PB.assert_schema(back, f"probe m2 lam={lam} c={cur}")
            for key in ("percentage_knowledge_harvested",
                        "Mean_of_Corr_prestige_ideal_merit", "xi", "agent_seed"):
                assert key in back.columns, f"probe m2 lam={lam} c={cur}: missing {key!r}"
            out["m2"][f"lam{lam}_c{cur}"] = dict(
                rows=len(back),
                harvest=float(back.percentage_knowledge_harvested.mean()),
                c_ideal=float(back.Mean_of_Corr_prestige_ideal_merit.mean()))
            p.unlink()
    wall = time.time() - t0
    # binding: the two ends of the axis must differ on the headline outcome
    b1 = abs(out["m1"][0.1]["c_ideal_fin"] - out["m1"][0.9]["c_ideal_fin"])
    b2 = abs(out["m2"]["lam1.0_c0.1"]["c_ideal"] - out["m2"]["lam1.0_c0.9"]["c_ideal"])
    assert b1 > 1e-6 and b2 > 1e-6, f"binding failed: m1 {b1}, m2 {b2}"
    out["binding"] = {"m1_c_ideal_gap": b1, "m2_c_ideal_gap": b2}
    out["wall_s_probe"] = round(wall, 1)
    out["wall_per_run_s"] = round(wall / 8.0, 2)
    print("  binding: m1 d(c_ideal) %.4f, m2 d(c_ideal) %.4f | %.2f s/run"
          % (b1, b2, out["wall_per_run_s"]))
    return out


def _diff_cell(new: pd.DataFrame, old_path: Path, keys: tuple) -> dict:
    """Bit-level diff of a rerun against an archived cell, seed by seed.

    The archive holds all 20 seeds and the control re-runs only a few, so the
    stored cell is FILTERED to the re-run's seed list before the comparison --
    an unfiltered diff reports a row-count error and hides the delta actually
    being measured.
    """
    old = pd.read_csv(old_path)
    seed_col = keys[0]
    wanted = set(new[seed_col].astype(int))
    old = old[old[seed_col].astype(int).isin(wanted)].copy()
    num = [c for c in new.columns if c in old.columns
           and pd.api.types.is_numeric_dtype(new[c])
           and pd.api.types.is_numeric_dtype(old[c])]
    a = new.sort_values(list(keys)).reset_index(drop=True)[num]
    b = old.sort_values(list(keys)).reset_index(drop=True)[num]
    assert len(a) == len(b), f"{old_path.name}: {len(a)} vs {len(b)} rows"
    worst, col = 0.0, None
    for c in num:
        delta = np.abs(a[c].to_numpy(float) - b[c].to_numpy(float))
        if not np.isfinite(delta).any():   # both sides all-NaN in these seeds
            continue
        d = float(np.nanmax(delta))
        if d > worst:
            worst, col = d, c
    return {"reference": str(old_path.relative_to(REPO)), "seeds": int(len(a)),
            "columns_compared": len(num), "max_abs_diff": worst, "worst_column": col}


def control_arm() -> dict:
    """Reproduce one archived cell per arm, seed by seed, on every shared column.

    Without this the new xi points are only *similar* to the archived ones; with
    it the difference between arms is attributable to curiosity alone.
    """
    print("CONTROL: rerun one archived cell per arm and diff it bit-exactly")
    seeds = (0, 1, 2)
    m1 = pd.DataFrame([m1_run_one(M1_MODE, M1_ENTRY, M1_CONTROL_CURIOSITY, M1_LAND, s)
                       for s in seeds])
    d1 = _diff_cell(m1, REPO / "data/turnover_v1/cells/cell_age_hazard_g10_cur05_mgstd5.csv",
                    ("seed",))
    m2 = pd.DataFrame([m2_run_one(1.0, M2_CONTROL_CURIOSITY, s) for s in seeds])
    d2 = _diff_cell(m2, REPO / "data/path_b_v2/cells/cell_lam100_pull100_logit_equal_mgstd5.csv",
                    ("agent_seed",))
    for d in (d1, d2):
        print("  %-58s seeds=%d cols=%d max|diff|=%.1e (%s)"
              % (d["reference"].split("/")[-1], d["seeds"], d["columns_compared"],
                 d["max_abs_diff"], d["worst_column"]))
        assert d["max_abs_diff"] == 0.0, f"control arm diverged: {d}"
    return {"m1": d1, "m2": d2}


# -------------------------------------------------------------------- driver
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("m1", "m2", "both"), default="both")
    ap.add_argument("--n-jobs", type=int, default=3)
    args = ap.parse_args()

    assert args.n_jobs <= 3, "round policy: n_jobs <= 3 on a shared box"
    CELLS.mkdir(parents=True, exist_ok=True)
    assert_archived_constants()

    canary_rc = os.system(
        "MPLBACKEND=Agg %s %s > %s/canary_closing.txt 2>&1"
        % (sys.executable, REPO / "tests" / "test_smoke.py", OUT))
    if canary_rc != 0:
        raise SystemExit("canary failed - aborting before any run")

    snap_before = snapshot_sources()
    probe_out = probe()
    control = control_arm()

    grid = []
    if args.arm in ("m1", "both"):
        grid += [("m1", s) for s in m1_specs()]
    if args.arm in ("m2", "both"):
        grid += [("m2", s) for s in m2_specs()]
    print("GRID: %d cells x %d seeds = %d runs, n_jobs=%d"
          % (len(grid), SEEDS, len(grid) * SEEDS, args.n_jobs))

    t0 = time.time()
    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(run_cell)(kind, spec) for kind, spec in grid)
    wall = time.time() - t0

    ok = sum(1 for _, st, _ in results if st == "ok")
    skipped = sum(1 for _, st, _ in results if st == "skipped")
    failed_cells = sorted(t for t, st, _ in results if st == "FAILED")
    failed_seeds = sum(len(fl) for _, _, fl in results)
    ran = [t for t, st, _ in results if st == "ok"]
    rows_this = sum(sum(1 for _ in open(CELLS / (t + ".csv"))) - 1 for t in ran)
    folder_cells = sorted(CELLS.glob("*.csv"))
    rows_net = sum(sum(1 for _ in open(f)) - 1 for f in folder_cells)

    snap_after = snapshot_sources()
    moved = [f for f in SOURCE_FILES if snap_before[f] != snap_after[f]]

    manifest = {
        "round": "Figure 1 extension (xi ends of the visibility axis, mgstd5)",
        "driver_version": DRIVER_VERSION,
        "config": {
            "m1": {"exit_mode": M1_MODE, "horizon": M1_HORIZON, "entry": M1_ENTRY,
                   "landscape": M1_LAND, "new_curiosity": list(M1_NEW_CURIOSITY)},
            "m2": {"dyadic_move_pull": M2_PULL, "formulation": M2_FORM,
                   "credit_sharing": M2_CREDIT, "step_limit": M2_STEP_LIMIT,
                   "entry": M2_ENTRY, "landscape": M2_LAND,
                   "new_curiosity": list(M2_NEW_CURIOSITY),
                   "lambdas": list(LAMBDAS)},
            "seeds_per_cell": SEEDS,
            "n_jobs": args.n_jobs,
            "arms": args.arm,
        },
        "this_invocation": {
            "arms": args.arm, "cells_gross": len(grid), "cells_new": ok,
            "cells_skipped": skipped, "cells_failed": failed_cells,
            "runs_gross": len(grid) * SEEDS, "rows_net": rows_this,
            "failed_seeds": failed_seeds, "wall_s": round(wall, 1),
        },
        "folder_totals": {
            "cells": len(folder_cells), "runs": rows_net,
            "expected": (len(m1_specs()) + len(m2_specs())) * SEEDS,
            "m1_cells": len(m1_specs()), "m2_cells": len(m2_specs()),
        },
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "probe": probe_out,
        "control_arm": control,
        "source_snapshot_before": snap_before,
        "source_snapshot_after": snap_after,
        "sources_moved_during_run": moved,
        "override_disclosure": (
            "path_b_v2_common.build_params pins initial_curiosity=0.5; this round "
            "replaces that ONE key with the cell's curiosity and asserts the "
            "replacement before constructing the model. Every other key comes "
            "from the archived builder unchanged. The M1 leg calls "
            "run_turnover_v1.run_one unmodified."
        ),
        "exit_rule_isolation": (
            "install_exit_rule patches MyModel._MyModel__generate_new_agents at "
            "class level; the arms run as separate processes and every M2 cell "
            "re-installs the captured original first."
        ),
        "canary": "tests/test_smoke.py, closing gate exit 0 (canary_closing.txt)",
    }
    with open(OUT / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    with open(OUT / "design.json", "w") as f:
        json.dump({k: manifest[k] for k in
                   ("round", "driver_version", "config", "override_disclosure",
                    "exit_rule_isolation", "source_snapshot_before")}, f, indent=1)

    print("GROSS cells=%d runs=%d | NET ok=%d skipped=%d failed_cells=%d "
          "failed_seeds=%d rows_net=%d/%d | wall=%.0fs | sources_moved=%s"
          % (len(grid), len(grid) * SEEDS, ok, skipped, len(failed_cells),
             failed_seeds, rows_this, len(grid) * SEEDS, wall, moved or "none"))
    print("FOLDER: %d cells, %d run rows (expected %d)"
          % (len(folder_cells), rows_net,
             (len(m1_specs()) + len(m2_specs())) * SEEDS))
    print("EXIT=0" if not failed_cells and not failed_seeds else "EXIT=1")
    return 0 if not failed_cells and not failed_seeds else 1


if __name__ == "__main__":
    sys.exit(main())
