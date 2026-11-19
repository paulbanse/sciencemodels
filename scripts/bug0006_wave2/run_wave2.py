"""BUG-0006 wave 2 -- re-run the remaining g>0 experimental archives under
the fixed code (Team 1 brief, 2026-11-19).

Wave 1 (`scripts/bug0006/`, `data/bug0006_remeasure_v2/`, `data/turnover_v2/`)
re-measured T1 / W_v05 / M2_xi / turnover.  This wave re-runs the archives
that wave 1 did not touch, in NEW namespaces, matching each source round's
exact grids, RNG seed sequences, CSV column schemas and filenames:

  data/interventions_v2/   Arm A + Arm B + Arm C   (72 cells x 20 seeds)
  data/typology_v2/        24 cells x 20 seeds + contrast_table.csv
  data/path_b_postbug/     baseline_off 40 + v1 grid 800 + v2 grid 960
                           + m0_m2_probe 80 (P-0011: g=0, pull=1.0,
                           lambda in {0,1}, logit + equal_split, 2 landscapes)

Engineering rules (Team 1 brief): stdlib ``concurrent.futures`` process pool
with top-level picklable worker functions (spawn context), atomic per-cell
checkpoint writes (tmp -> row-count check -> os.replace), and a
``manifest.json`` per output folder recording git_head,
sources_moved_during_sweep, gross/net counts and seed lists.  Source rounds
are READ-ONLY; their runners are imported, not copied.

One deliberate convention (disclosed in every manifest and in the summary):
the source interventions driver computes the terminal yardstick cross-section
``ideal_merit_sum_final`` as a RAW sum over ``a.ideal_Merit``.  Under the
BUG-0006 fix, last-step newborns carry NaN, so the raw channel now yields NaN
for every run with a step-200 birth.  Wave 1 established the intended
convention -- sum only FINITE merits (``scripts/bug0006/run_bug0006_remeasure.py:209``);
pre-fix unscored newborns contributed exactly 0, so the finite cross-section
is bit-identical to the pre-fix value.  The wave-2 rows therefore carry the
finite-convention value in the same column (schema unchanged), and the raw
NaN is recorded in the manifest as a driver-side defect to file.

Run:
    MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/bug0006_wave2/run_wave2.py --probe
    MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/bug0006_wave2/run_wave2.py --family all
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "scripts" / "interventions",
           REPO / "scripts" / "typology", REPO / "scripts" / "path_b"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402  (workers need it; imported here so spawn
                    # re-imports of this module are self-contained)

SEEDS = list(range(20))
N_SEEDS = len(SEEDS)

OUT_INTERV = REPO / "data" / "interventions_v2"
OUT_TYPO = REPO / "data" / "typology_v2"
OUT_PATHB = REPO / "data" / "path_b_postbug"

#: modules the dynamics depend on; hashed before and after every sweep
SOURCE_FILES = (
    "sciencemodels/agents.py",
    "sciencemodels/model.py",
    "sciencemodels/metrics.py",
    "scripts/interventions/run_interventions_v1.py",
    "scripts/typology/run_typology_v1.py",
    "scripts/typology/typology_layers.py",
    "scripts/path_b/run_path_b_sweep.py",
    "scripts/path_b/run_baseline_off.py",
    "scripts/path_b/path_b_v2_common.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_sources() -> dict:
    return {f: sha256(REPO / f) for f in SOURCE_FILES}


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:                                # noqa: BLE001
        return "unknown"


def line_count(path: Path) -> int:
    with open(path) as fh:
        return sum(1 for _ in fh)


# ============================================================ worker functions
# Top-level, picklable (spawn context).  Source modules are imported INSIDE
# the workers: their BASE_PARAMS carry a lambda (agent_generation_function),
# which never crosses the pickle boundary because params are rebuilt here.
def _w_interv(args):
    cell, seed = args
    import run_interventions_v1 as drv

    class Wave2Model(drv.InterventionsModel):
        """InterventionsModel + the wave-1 finite-cross-section convention for
        the terminal yardstick (see module docstring)."""

        def run_mode_for_bulk(self, longitudinal=False):
            row = super().run_mode_for_bulk(longitudinal=longitudinal)
            finite = [float(a.ideal_Merit) for a in self.agents
                      if np.isfinite(getattr(a, "ideal_Merit", np.nan))]
            row["ideal_merit_sum_final"] = (
                float(np.sum(finite)) if finite else float("nan"))
            return row

    return drv.run_one(cell, seed, model_cls=Wave2Model)


def _w_typology(args):
    spec, seed = args
    import warnings
    # same disclosure as the source round: the destruction channel's literal
    # formula yields NaN rewards on crowded tiles; the regime is NOT silenced
    warnings.filterwarnings(
        "ignore", message="invalid value encountered in scalar power")
    import run_typology_v1 as drv
    import typology_layers as tl
    gamma, kappa, curiosity, g_entry = spec
    try:
        m = tl.TypologyModel(
            **drv.build_params(gamma, kappa, curiosity, g_entry, seed))
        for _ in range(drv.STEP_LIMIT):
            m.step(True)
        row = tl._build_row(m)
        row.update(gamma=gamma, kappa=kappa, curiosity=curiosity,
                   g_entry=g_entry, seed=seed)
        arows = drv.agent_finals(m, gamma, kappa, curiosity, g_entry, seed)
        return "ok", row, arows
    except Exception:                                # noqa: BLE001
        return "fail", traceback.format_exc(), []


def _w_pathb(args):
    mode, spec, seed = args
    try:
        if mode == "off":
            import run_baseline_off as bo
            return "ok", bo.run_one_off(*spec, seed)
        if mode == "v1":
            import run_path_b_sweep as drv
            return "ok", drv.run_one(*spec, seed)
        if mode == "v2":
            import path_b_v2_common as C
            return "ok", C.run_one(*spec, seed)
        if mode == "probe":
            # P-0011: M0 + M2 without demographic entry -- ONE key replaced
            # (agent_generation_time 0.1 -> 0.0), asserted before construction.
            import path_b_v2_common as C
            from sciencemodels.model import MyModel
            lam, pull, form, credit, land = spec
            params = C.build_params(lam, pull, form, credit, land, seed)
            params["agent_generation_time"] = 0.0
            assert params["agent_generation_time"] == 0.0, "probe override lost"
            m = MyModel(**params)
            for _ in range(C.STEP_LIMIT):
                m.step(True)
            row = C.report_extras(m, m.get_report())
            row["dyadic_lambda"] = float(lam)
            row["dyadic_move_pull"] = float(pull)
            row["formulation"] = form
            row["credit_sharing_axis"] = credit
            row["landscape"] = land
            return "ok", row
        raise ValueError(f"unknown mode {mode!r}")
    except Exception:                                # noqa: BLE001
        return "fail", traceback.format_exc()


# ============================================================ parent helpers
def _log_factory(path: Path):
    def log(msg):
        line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        with open(path, "a") as fh:
            fh.write(line + "\n")
    return log


def _write_rows(path: Path, rows: list, style: str) -> None:
    """Atomic per-cell write with a read-back row-count check."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    if style == "dictwriter":
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=sorted(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    else:  # pandas, matching the source rounds' to_csv convention
        import pandas as pd
        pd.DataFrame(rows).to_csv(tmp, index=False)
    n_back = line_count(tmp) - 1
    assert n_back == len(rows), (
        f"{path.name}: wrote {len(rows)} rows, read back {n_back}")
    os.replace(tmp, path)


def _dispatch(executor, worker, tasks):
    """Run tasks through the pool; returns list of payloads (status first)."""
    return list(executor.map(worker, tasks, chunksize=1))


def _finish_manifest(folder: Path, family: str, sub: dict, t0: float,
                     hashes_before: dict) -> dict:
    hashes_after = snapshot_sources()
    moved = {k: [hashes_before[k], hashes_after[k]]
             for k in hashes_before if hashes_before[k] != hashes_after[k]}
    manifest = {
        "schema": f"bug0006_wave2_{family}",
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": git_head(),
        "environment": {
            "python": sys.version.split()[0],
            "pool": "concurrent.futures.ProcessPoolExecutor, spawn context",
            "n_jobs_note": "see run_log.txt",
        },
        "seed_semantics": "agent_seed 0..19 drives agent and landscape RNG jointly "
                          "(identical to the source rounds)",
        "seeds": SEEDS,
        "sources_snapshot_before": hashes_before,
        "sources_snapshot_after": hashes_after,
        "sources_moved_during_sweep": moved,
        "gross_net": sub,
        "wall_s": round(time.time() - t0, 1),
        "ideal_merit_sum_final_convention": (
            "finite-only cross-section sum (wave-1 convention, "
            "run_bug0006_remeasure.py:209); the source interventions driver's "
            "raw channel yields NaN post-fix for runs with a step-200 birth "
            "(disclosed driver-side defect, not written to the CSVs)"),
    }
    with open(folder / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=1, default=str)
    return manifest


# ============================================================ families
def sweep_interventions(executor, log, n_jobs) -> dict:
    import run_interventions_v1 as drv
    cells = drv.cell_list()
    cells_dir = OUT_INTERV / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    pending, tags_of_task = [], []
    for cell in cells:
        tag = drv.cell_tag(cell)
        path = cells_dir / (tag + ".csv")
        if path.exists() and line_count(path) == N_SEEDS + 1:
            log(f"SKIP {tag} (complete)")
            continue
        for seed in SEEDS:
            pending.append((cell, seed))
            tags_of_task.append(tag)
    log(f"interventions_v2: {len(cells)} cells, "
        f"{len(pending)} runs to execute (n_jobs={n_jobs})")
    t0 = time.time()
    results = _dispatch(executor, _w_interv, pending)
    grouped: dict[str, list] = {}
    fails: dict[str, str] = {}
    for tag, (status, payload) in zip(tags_of_task, results):
        if status == "ok":
            grouped.setdefault(tag, []).append(payload)
        else:
            fails[tag] = str(payload)[-2000:]
    n_written = 0
    for tag, rows in grouped.items():
        # cell order preserved: rows arrive in SEEDS order per cell
        _write_rows(cells_dir / (tag + ".csv"), rows, "dictwriter")
        n_written += len(rows)
        log(f"CELL {tag} -> ok rows={len(rows)}")
    for tag, err in fails.items():
        log(f"CELL {tag} FAILED: {err[:300]}")
    rows_net = sum(line_count(p) - 1 for p in sorted(cells_dir.glob("*.csv")))
    sub = {
        "cells_gross": len(cells),
        "cells_written": len(grouped),
        "cells_failed": sorted(fails),
        "runs_gross": len(cells) * N_SEEDS,
        "rows_net": rows_net,
        "failed_runs": len(fails),
        "grid": "Arm A 18 + Arm B 36 + Arm C 18 = 72 cells x 20 seeds "
                "= 1,440 runs, H=200 (exact interventions_v1 grid)",
    }
    _finish_manifest(OUT_INTERV, "interventions", sub, t0, snapshot_sources())
    return sub


def sweep_typology(executor, log, n_jobs) -> dict:
    import run_typology_v1 as drv
    cells_dir = OUT_TYPO / "cells"
    agents_dir = cells_dir / "agents"
    cells_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)
    grid = list(itertools.product(drv.GAMMAS, drv.KAPPAS,
                                  drv.CURIOSITIES, drv.ENTRIES))
    pending, tags_of_task = [], []
    for spec in grid:
        tag = drv.cell_tag(*spec)
        path = cells_dir / (tag + ".csv")
        if path.exists() and line_count(path) == N_SEEDS + 1:
            log(f"SKIP {tag} (complete)")
            continue
        for seed in SEEDS:
            pending.append((spec, seed))
            tags_of_task.append(tag)
    log(f"typology_v2: {len(grid)} cells, {len(pending)} runs "
        f"(n_jobs={n_jobs})")
    t0 = time.time()
    results = _dispatch(executor, _w_typology, pending)
    grouped: dict[str, list] = {}
    grouped_a: dict[str, list] = {}
    fails: dict[str, str] = {}
    for tag, (status, payload, arows) in zip(tags_of_task, results):
        if status == "ok":
            grouped.setdefault(tag, []).append(payload)
            grouped_a.setdefault(tag, []).extend(arows)
        else:
            fails[tag] = str(payload)[-2000:]
    for tag, rows in grouped.items():
        _write_rows(cells_dir / (tag + ".csv"), rows, "pandas")
        _write_rows(agents_dir / (tag + ".csv"), grouped_a[tag], "pandas")
        log(f"CELL {tag} -> ok rows={len(rows)} agent_rows="
            f"{len(grouped_a[tag])}")
    for tag, err in fails.items():
        log(f"CELL {tag} FAILED: {err[:300]}")
    # consolidate runs.csv from disk (evidence over memory), source convention
    import pandas as pd
    frames = [pd.read_csv(p) for p in sorted(cells_dir.glob("cell_*.csv"))]
    runs = pd.concat(frames, ignore_index=True)
    runs.to_csv(OUT_TYPO / "runs.csv", index=False)
    rows_net = sum(line_count(p) - 1
                   for p in sorted(cells_dir.glob("cell_*.csv")))
    sub = {
        "cells_gross": len(grid),
        "cells_written": len(grouped),
        "cells_failed": sorted(fails),
        "runs_gross": len(grid) * N_SEEDS,
        "rows_net": rows_net,
        "failed_runs": len(fails),
        "grid": "2 gamma x 2 kappa x 3 curiosity x 2 entry = 24 cells on "
                "mgstd5, 20 seeds, 480 runs, H=200 (exact typology_v1 grid)",
    }
    _finish_manifest(OUT_TYPO, "typology", sub, t0, snapshot_sources())
    return sub


def sweep_path_b(executor, log, n_jobs) -> dict:
    import run_path_b_sweep as v1
    import path_b_v2_common as C
    t0 = time.time()
    sub_all = {}

    # ---- baseline_off (40 runs, single CSV, source convention) ----------
    off_path = OUT_PATHB / "baseline_off.csv"
    if not (off_path.exists() and line_count(off_path) == 40):
        tasks = [("off", (land,), seed)
                 for land in v1.LANDSCAPES for seed in SEEDS]
        results = _dispatch(executor, _w_pathb, tasks)
        rows = [r for st, r in results if st == "ok"]
        fails = [r for st, r in results if st != "ok"]
        assert len(rows) == 40 and not fails, f"baseline_off: {len(rows)}/40"
        _write_rows(off_path, rows, "pandas")
        log(f"baseline_off -> 40 rows")
    else:
        log("SKIP baseline_off (complete)")
    sub_all["baseline_off"] = {"runs_gross": 40,
                               "rows_net": line_count(off_path) - 1}

    # ---- v1 grid (40 cells x 20 = 800) ---------------------------------
    cells_dir = OUT_PATHB / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    grid_v1 = list(itertools.product(v1.LAMBDAS, v1.FORMULATIONS,
                                     v1.CREDITS, v1.LANDSCAPES))
    pending, tags_of_task = [], []
    for spec in grid_v1:
        tag = v1.cell_tag(*spec)
        path = cells_dir / (tag + ".csv")
        if path.exists() and line_count(path) == N_SEEDS + 1:
            log(f"SKIP v1 {tag} (complete)")
            continue
        for seed in SEEDS:
            pending.append(("v1", spec, seed))
            tags_of_task.append(tag)
    if pending:
        results = _dispatch(executor, _w_pathb, pending)
        grouped: dict[str, list] = {}
        fails: dict[str, str] = {}
        for tag, (status, payload) in zip(tags_of_task, results):
            if status == "ok":
                grouped.setdefault(tag, []).append(payload)
            else:
                fails[tag] = str(payload)[-2000:]
        for tag, rows in grouped.items():
            _write_rows(cells_dir / (tag + ".csv"), rows, "pandas")
            log(f"CELL v1 {tag} -> ok rows={len(rows)}")
        for tag, err in fails.items():
            log(f"CELL v1 {tag} FAILED: {err[:300]}")
    rows_net = sum(line_count(p) - 1 for p in sorted(cells_dir.glob("*.csv")))
    sub_all["v1_cells"] = {"cells_gross": len(grid_v1), "rows_net": rows_net,
                           "runs_gross": len(grid_v1) * N_SEEDS}

    # ---- v2 grid (48 cells x 20 = 960) ---------------------------------
    cells_v2_dir = OUT_PATHB / "cells_v2"
    cells_v2_dir.mkdir(parents=True, exist_ok=True)
    # v2 sweep constants: lambda {0,0.5,1} x pull {0.5,1.0} (source round)
    grid_v2 = list(itertools.product((0.0, 0.5, 1.0), (0.5, 1.0),
                                     v1.FORMULATIONS, v1.CREDITS,
                                     v1.LANDSCAPES))
    assert len(grid_v2) == 48, f"v2 grid should be 48 cells, got {len(grid_v2)}"
    pending, tags_of_task = [], []
    for spec in grid_v2:
        tag = C.cell_tag(*spec)
        path = cells_v2_dir / (tag + ".csv")
        if path.exists() and line_count(path) == N_SEEDS + 1:
            log(f"SKIP v2 {tag} (complete)")
            continue
        for seed in SEEDS:
            pending.append(("v2", spec, seed))
            tags_of_task.append(tag)
    if pending:
        results = _dispatch(executor, _w_pathb, pending)
        grouped = {}
        fails = {}
        for tag, (status, payload) in zip(tags_of_task, results):
            if status == "ok":
                grouped.setdefault(tag, []).append(payload)
            else:
                fails[tag] = str(payload)[-2000:]
        for tag, rows in grouped.items():
            _write_rows(cells_v2_dir / (tag + ".csv"), rows, "pandas")
            log(f"CELL v2 {tag} -> ok rows={len(rows)}")
        for tag, err in fails.items():
            log(f"CELL v2 {tag} FAILED: {err[:300]}")
    rows_net = sum(line_count(p) - 1
                   for p in sorted(cells_v2_dir.glob("*.csv")))
    sub_all["v2_cells"] = {"cells_gross": len(grid_v2), "rows_net": rows_net,
                           "runs_gross": len(grid_v2) * N_SEEDS}

    # ---- m0_m2_probe (P-0011: g=0, pull=1.0, lam {0,1}, logit+equal,
    #      2 landscapes, 20 seeds = 80 runs) -----------------------------
    probe_dir = OUT_PATHB / "m0_m2_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    grid_probe = list(itertools.product((0.0, 1.0), (1.0,),
                                        ("logit",), ("equal_split",),
                                        v1.LANDSCAPES))
    pending, tags_of_task = [], []
    for spec in grid_probe:
        tag = C.cell_tag(*spec)
        path = probe_dir / (tag + ".csv")
        if path.exists() and line_count(path) == N_SEEDS + 1:
            log(f"SKIP probe {tag} (complete)")
            continue
        for seed in SEEDS:
            pending.append(("probe", spec, seed))
            tags_of_task.append(tag)
    if pending:
        results = _dispatch(executor, _w_pathb, pending)
        grouped = {}
        fails = {}
        for tag, (status, payload) in zip(tags_of_task, results):
            if status == "ok":
                grouped.setdefault(tag, []).append(payload)
            else:
                fails[tag] = str(payload)[-2000:]
        for tag, rows in grouped.items():
            _write_rows(probe_dir / (tag + ".csv"), rows, "pandas")
            log(f"CELL probe {tag} -> ok rows={len(rows)}")
        for tag, err in fails.items():
            log(f"CELL probe {tag} FAILED: {err[:300]}")
    rows_net = sum(line_count(p) - 1 for p in sorted(probe_dir.glob("*.csv")))
    sub_all["m0_m2_probe"] = {"cells_gross": len(grid_probe),
                              "rows_net": rows_net,
                              "runs_gross": len(grid_probe) * N_SEEDS,
                              "grid": "g=0, dyadic_move_pull=1.0, "
                                      "dyadic_lambda in {0.0,1.0}, logit + "
                                      "equal_split, mgstd5 + noisypr5"}

    _finish_manifest(OUT_PATHB, "path_b", sub_all, t0, snapshot_sources())
    return sub_all


# ============================================================ probe
def probe(executor, log, n_jobs) -> int:
    """Critical path before the grid: one cell x 2 seeds per family through
    the REAL spawn pool, schema equality vs the v1 archives, and paired
    spot-checks (g=0 all-metrics; g>0 non-ideal invariance + ideal shift)."""
    import pandas as pd
    import run_interventions_v1 as iv
    import run_typology_v1 as tv
    import run_path_b_sweep as pb
    import path_b_v2_common as C

    failures = []

    def _f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    def _ideal_channel(k):
        """The BUG-0006 channel: merit-correlation / wedge columns only.
        Cumulative accumulators (ideal_loss_cum, ideal_merit_cum_*,
        Mean_of_Ideal_merit_cum) are DYNAMICS and stay invariant."""
        kl = k.lower()
        return (("corr" in kl and "ideal_merit" in kl)
                or kl in ("c_ideal_tm", "c_ideal_fin")
                or kl.startswith("wedge"))

    def check(name, ok, detail=""):
        status = "PASS" if ok else "FAIL"
        log(f"PROBE {status}: {name} {detail}")
        if not ok:
            failures.append(name)

    # --- canary -----------------------------------------------------------
    can = subprocess.run(
        [sys.executable, str(REPO / "tests" / "test_smoke.py")],
        capture_output=True, text=True, env={**os.environ,
                                             "MPLBACKEND": "Agg"})
    check("canary tests/test_smoke.py", can.returncode == 0,
          f"exit={can.returncode}")

    # --- interventions: Arm C g=0 (all metrics) + Arm B g>0 ---------------
    cell_g0 = dict(arm="C", refill_mode="off", initial_curiosity=0.5,
                   landscape="mgstd5", r_spawn="exact", alpha=1.0,
                   p_reloc=0.0, agent_generation_time=0.0)
    cell_g1 = dict(arm="B", alpha=1.0, p_reloc=0.02, visibility_weight=0.7,
                   initial_curiosity=0.3, landscape="noisypr5",
                   r_spawn="exact", refill_mode="off",
                   agent_generation_time=0.1)
    res = _dispatch(executor, _w_interv, [(cell_g0, 0), (cell_g1, 0)])
    (st0, row0), (st1, row1) = res
    check("interventions worker path (spawn)", st0 == st1 == "ok")

    def archive_row(cell, seed):
        p = REPO / "data" / "interventions_v1" / "cells" / (
            iv.cell_tag(cell) + ".csv")
        with open(p) as fh:
            return {r["agent_seed"]: r for r in csv.DictReader(fh)}[str(seed)]

    old0 = archive_row(cell_g0, 0)
    shared = sorted(set(old0) & set(row0))
    keys_num = [k for k in shared if _f(old0[k]) is not None
                and _f(row0[k]) is not None]
    keys_cat = [k for k in shared if k not in keys_num]
    diffs0 = {k: abs(_f(old0[k]) - _f(row0[k])) for k in keys_num}
    cat_ok = all(str(old0[k]) == str(row0[k]) for k in keys_cat)
    check("Arm C g=0 all-metrics invariance (<=1e-11)",
          max(diffs0.values(), default=0.0) <= 1e-11 and cat_ok,
          f"max={max(diffs0.values(), default=0.0):.3e} "
          f"cat_equal={cat_ok} ({len(keys_num)} numeric, {len(keys_cat)} "
          f"categorical)")
    check("Arm C g=0 schema equality",
          set(old0) == set(row0),
          f"v1-only={sorted(set(old0) - set(row0))} "
          f"v2-only={sorted(set(row0) - set(old0))}")

    old1 = archive_row(cell_g1, 0)
    nonideal = [k for k in sorted(set(old1) & set(row1))
                if not _ideal_channel(k)
                and _f(old1[k]) is not None and _f(row1[k]) is not None]
    d_ni = max(abs(_f(old1[k]) - _f(row1[k])) for k in nonideal)
    d_id = max(abs(_f(old1[k]) - _f(row1[k]))
               for k in sorted(set(old1) & set(row1))
               if _ideal_channel(k) and _f(old1[k]) is not None)
    check("Arm B g>0 non-ideal invariance (<=1e-11)", d_ni <= 1e-11,
          f"max={d_ni:.3e}")
    check("Arm B g>0 ideal channel MOVES (sanity: bug was live)",
          d_id > 1e-3, f"max={d_id:.3e}")

    # --- typology: g=0 all-metrics + g>0 ideal moves ----------------------
    res = _dispatch(executor, _w_typology,
                    [((0.0, 0.0, 0.5, 0.0), 0), ((1.0, 0.0, 0.5, 0.1), 0)])
    (st, row_t0, _), (st, row_t1, _) = res
    check("typology worker path (spawn)", st == "ok")
    old_t0 = pd.read_csv(REPO / "data" / "typology_v1" / "cells" /
                         (tv.cell_tag(0.0, 0.0, 0.5, 0.0) + ".csv"))
    old_t0 = old_t0[old_t0["seed"] == 0].iloc[0]
    shared_t = [k for k in old_t0.index if k in row_t0
                and _f(old_t0[k]) is not None and _f(row_t0[k]) is not None]
    d_t0 = max(abs(_f(old_t0[k]) - _f(row_t0[k])) for k in shared_t)
    check("typology g=0 all-metrics invariance (<=1e-11)", d_t0 <= 1e-11,
          f"max={d_t0:.3e}")
    old_t1 = pd.read_csv(REPO / "data" / "typology_v1" / "cells" /
                         (tv.cell_tag(1.0, 0.0, 0.5, 0.1) + ".csv"))
    old_t1 = old_t1[old_t1["seed"] == 0].iloc[0]
    d_w = abs(float(old_t1["wedge_fin"]) - float(row_t1["wedge_fin"]))
    check("typology g>0 wedge_fin MOVES (sanity)", d_w > 1e-3,
          f"max={d_w:.3e}")

    # --- path_b: v1 cell (non-ideal invariance) + probe path --------------
    spec_v1 = (1.0, "logit", "structural", "mgstd5")
    res = _dispatch(executor, _w_pathb, [("v1", spec_v1, 0)])
    st, row_pb = res[0]
    check("path_b v1 worker path (spawn)", st == "ok")
    old_pb = pd.read_csv(REPO / "data" / "path_b_v1" / "cells" /
                         (pb.cell_tag(*spec_v1) + ".csv"))
    old_pb = old_pb[old_pb["agent_seed"] == 0].iloc[0]
    shared_pb = [k for k in old_pb.index if k in row_pb
                 and not _ideal_channel(k)
                 and _f(old_pb[k]) is not None and _f(row_pb[k]) is not None]
    d_pb = max(abs(_f(old_pb[k]) - _f(row_pb[k])) for k in shared_pb)
    check("path_b v1 non-ideal invariance (<=1e-11)", d_pb <= 1e-11,
          f"max={d_pb:.3e}")
    res = _dispatch(executor, _w_pathb,
                    [("probe", (0.0, 1.0, "logit", "equal_split",
                                "mgstd5"), 0)])
    st, row_pr = res[0]
    check("m0_m2_probe worker path (spawn, g=0 override)",
          st == "ok" and row_pr.get("dyadic_move_pull") == 1.0
          and row_pr.get("dyadic_lambda") == 0.0)

    if failures:
        log(f"PROBE FAILED: {failures}")
        return 1
    log("PROBE ALL PASS")
    return 0


# ============================================================ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="all",
                    choices=["all", "interventions", "typology", "path_b"])
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--probe", action="store_true",
                    help="canary + schema + paired spot-checks only")
    args = ap.parse_args()

    log = _log_factory(REPO / "scripts" / "bug0006_wave2" / "run_log.txt")
    hashes_before = snapshot_sources()
    log(f"launch family={args.family} n_jobs={args.n_jobs} "
        f"git_head={git_head()[:12]}")

    ctx = get_context("spawn")  # brief: top-level workers, macOS spawn
    with ProcessPoolExecutor(max_workers=args.n_jobs,
                             mp_context=ctx) as executor:
        if args.probe:
            return probe(executor, log, args.n_jobs)
        subs = {}
        if args.family in ("all", "interventions"):
            subs["interventions"] = sweep_interventions(executor, log,
                                                        args.n_jobs)
        if args.family in ("all", "typology"):
            subs["typology"] = sweep_typology(executor, log, args.n_jobs)
        if args.family in ("all", "path_b"):
            subs["path_b"] = sweep_path_b(executor, log, args.n_jobs)
        log(f"DONE {json.dumps(subs, default=str)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
