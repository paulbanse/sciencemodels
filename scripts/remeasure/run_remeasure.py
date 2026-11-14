"""Re-measure pass (2026-11-14, v2): rerun the load-bearing wave-1/2 cells under
the fixed ideal-merit pairing (N1 round, d3503ac) and summarize per cell for
paired comparison against the archived wave-1/2 summaries.

v2 adds per-cell checkpointing (ADR-0001 field-execution hygiene: an
interruption loses at most one cell) after the v1 in-memory driver lost its
entire pass to a stalled joblib parent. Model-internal; NOT calibrated
(owner ruling 2026-11-13). Grids: T1 sensitivity grid (36 cells) + welfare
grid (40 cells), 20 seeds each. Outputs: data/remeasure_v1/cells/<tag>.csv
(atomic), runs.csv + cell_summary.csv + manifest.json (consolidated).
Measures come from the model's own datacollector (one home).

v3 (2026-11-14) adds the W_v05 arm: the 40-cell welfare grid re-run at
``vanishing_factor_for_prestige = 0.5``, the value the archived welfare sweep
(``data/welfare_v1``) actually used, plus terminal-step ``ideal_merit_sum`` /
``ideal_merit_mean`` recorded the way that sweep recorded them
(``[float(a.ideal_Merit) for a in model.agents]`` after the last step). The
legacy ``W`` arm stays at 0.9 as a bonus sensitivity bridge.

WRITE CONTRACT -- do not break. The legacy outputs ``runs.csv`` and
``cell_summary.csv`` are read by ``scripts/report_notebook/round_report_data.py``,
which asserts ``len(runs.csv) == 1520`` and ``wedge_fin == c_ideal_fin -
c_local_fin`` on the raw file (it flips the sign in memory) and asserts the
W-arm vanishing mismatch ``[0.5] != [0.9]``. This driver therefore never
rewrites those two files: the W_v05 arm writes only ``cells_w_v05/``,
``w_v05_runs.csv``, ``w_v05_summary.csv`` and ``w_v05_manifest.json``, and its
wedge columns are explicitly named ``wedge_local_minus_ideal_*`` in the house
convention (``c_local - c_ideal``), documented in the W_v05 manifest.
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
CELLS = os.path.join(OUT, "cells")
# v3: own checkpoint dir and own consolidated files, so the banked legacy
# runs.csv / cell_summary.csv are never rewritten.
CELLS_WV05 = os.path.join(OUT, "cells_w_v05")

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


def cell_tag(spec):
    origin, cur, van, land, gen, vis = spec
    return "cell_%s_cur%02d_van%02d_%s_gen%02d_vis%d" % (
        origin,
        int(round(cur * 10)),
        int(round(van * 10)),
        land,
        int(round(gen * 10)),
        int(vis),
    )


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
    # v3: terminal-step ideal merit, same definition the archived welfare cells
    # used (run_welfare_sweep.py wraps io.build_report with
    # [float(a.ideal_Merit) for a in model.agents]). Recorded for every new run;
    # the banked legacy cells predate these two columns.
    merits = [float(a.ideal_Merit) for a in m.agents]
    row["ideal_merit_sum"] = float(np.sum(merits))
    row["ideal_merit_mean"] = float(np.mean(merits))
    return row


def run_cell(spec, cells_dir=None, log_name="run_log.txt"):
    origin, cur, van, land, gen, vis = spec
    tag = cell_tag(spec)
    path = os.path.join(cells_dir or CELLS, tag + ".csv")
    if os.path.exists(path) and sum(1 for _ in open(path)) == 21:
        return tag, "skipped"
    rows = [run_one(origin, cur, van, land, gen, vis, s) for s in range(20)]
    df = pd.DataFrame(rows)
    assert df["c_ideal_fin"].notna().sum() == 20, tag + " has missing runs"
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)  # atomic: a crash leaves either no cell or a whole cell
    with open(os.path.join(OUT, log_name), "a") as f:
        f.write("%s ok rows=20\n" % tag)
    return tag, "ok"


def run_cell_w_v05(spec):
    """Same checkpointed writer, own directory and own log."""
    return run_cell(spec, cells_dir=CELLS_WV05, log_name="w_v05_run_log.txt")


def consolidate():
    frames = [
        pd.read_csv(os.path.join(CELLS, f))
        for f in sorted(os.listdir(CELLS))
        if f.endswith(".csv")
    ]
    df = pd.concat(frames, ignore_index=True)
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
    return df, summ


def cells_w_v05():
    """W_v05 arm: the 40-cell welfare grid at vanishing = 0.5.

    ``data/welfare_v1`` ran at 0.5; the legacy W arm above re-ran the same grid
    at 0.9, so a W-versus-archive delta carries the vanishing unification on top
    of the ideal-merit pairing fix and cannot be attributed to the fix alone.
    This arm reproduces the archived configuration exactly, so the pairing-fix
    delta is isolated.
    """
    grid = []
    for cur, vis, gen, land in itertools.product(
        (0.1, 0.3, 0.5, 0.7, 0.9), (True, False), (0.0, 0.1), ("mgstd5", "mg6")
    ):
        grid.append(("Wv05", cur, 0.5, land, gen, vis))
    return grid


def consolidate_w_v05():
    """Consolidate only the W_v05 cells; never touches the legacy outputs."""
    frames = [
        pd.read_csv(os.path.join(CELLS_WV05, f))
        for f in sorted(os.listdir(CELLS_WV05))
        if f.endswith(".csv")
    ]
    df = pd.concat(frames, ignore_index=True)
    # House convention, explicitly named to avoid the legacy file's opposite
    # sign: positive when local merit outruns ideal merit, i.e. prestige tracks
    # the visible proxy better than it tracks quality.
    df["wedge_local_minus_ideal_fin"] = df["c_local_fin"] - df["c_ideal_fin"]
    df["wedge_local_minus_ideal_tm"] = df["c_local_tm"] - df["c_ideal_tm"]
    df.to_csv(os.path.join(OUT, "w_v05_runs.csv"), index=False)
    keys = [
        "origin",
        "curiosity",
        "vanishing",
        "landscape",
        "agent_generation_time",
        "use_visibility_reward",
    ]
    numeric = [c for c in df.columns if c not in keys + ["seed"]]
    summ = df.groupby(keys)[numeric].mean().reset_index()
    summ["n_seeds"] = df.groupby(keys).size().to_numpy()
    summ.to_csv(os.path.join(OUT, "w_v05_summary.csv"), index=False)
    return df, summ


if __name__ == "__main__":
    arm = sys.argv[1] if len(sys.argv) > 1 else "w_v05"
    canary = os.system(
        "MPLBACKEND=Agg %s %s > %s/canary.txt 2>&1"
        % (sys.executable, os.path.join(ROOT, "tests", "test_smoke.py"), OUT)
    )
    assert canary == 0, "canary failed - aborting before any run"

    if arm == "legacy":
        # Kept for a from-scratch rebuild only. Both outputs are consumed by
        # scripts/report_notebook/round_report_data.py under a row-count and
        # sign assertion, so a rebuild over banked files is refused here.
        for name in ("runs.csv", "cell_summary.csv"):
            if os.path.exists(os.path.join(OUT, name)):
                sys.exit(
                    "refusing to rewrite %s: banked and read by "
                    "round_report_data.py (len==1520, wedge_fin==c_ideal-c_local). "
                    "Delete it deliberately or build in a fresh folder." % name
                )
        os.makedirs(CELLS, exist_ok=True)
        grid = cells()
        t0 = time.time()
        results = Parallel(n_jobs=2, verbose=5)(delayed(run_cell)(c) for c in grid)
        df, summ = consolidate()
        done = sum(1 for _, st in results if st == "ok")
        skipped = sum(1 for _, st in results if st == "skipped")
        json.dump(
            {
                "grid": "T1 36 cells + welfare 40 cells",
                "cells_gross": len(grid),
                "cells_new": done,
                "cells_skipped": skipped,
                "gross_runs": len(grid) * 20,
                "net_runs_ok": int(df["c_ideal_fin"].notna().sum()),
                "wall_s": round(time.time() - t0, 1),
                "code_base": "post d3503ac (pairing-fixed); driver v3 checkpointed",
            },
            open(os.path.join(OUT, "manifest.json"), "w"),
            indent=1,
        )
        print(
            "LEGACY GROSS cells=%d runs=%d | NEW %d SKIPPED %d | NET ok=%d | wall=%.0fs"
            % (
                len(grid),
                len(grid) * 20,
                done,
                skipped,
                int(df["c_ideal_fin"].notna().sum()),
                time.time() - t0,
            )
        )
    elif arm == "w_v05":
        os.makedirs(CELLS_WV05, exist_ok=True)
        grid = cells_w_v05()
        t0 = time.time()
        results = Parallel(n_jobs=3, verbose=5)(
            delayed(run_cell_w_v05)(c) for c in grid
        )
        df, summ = consolidate_w_v05()
        done = sum(1 for _, st in results if st == "ok")
        skipped = sum(1 for _, st in results if st == "skipped")
        json.dump(
            {
                "arm": "W_v05",
                "why": (
                    "the archived welfare sweep ran at vanishing 0.5; the "
                    "legacy W arm re-ran the grid at 0.9, so only this arm "
                    "isolates the ideal-merit pairing fix from the vanishing "
                    "unification"
                ),
                "grid": "5 curiosity x 2 visibility toggle x 2 entry x 2 landscape",
                "cells_gross": len(grid),
                "cells_new": done,
                "cells_skipped": skipped,
                "gross_runs": len(grid) * 20,
                "net_runs_ok": int(df["c_ideal_fin"].notna().sum()),
                "wall_s": round(time.time() - t0, 1),
                "n_jobs": 3,
                "wedge_convention": "c_local - c_ideal (wedge_local_minus_ideal_*)",
                "outputs": [
                    "cells_w_v05/",
                    "w_v05_runs.csv",
                    "w_v05_summary.csv",
                    "w_v05_run_log.txt",
                ],
                "legacy_outputs_untouched": ["runs.csv", "cell_summary.csv"],
                "ideal_merit_definition": (
                    "terminal step, [float(a.ideal_Merit) for a in model.agents], "
                    "as in data/welfare_v1"
                ),
                "code_base": "post d3503ac (pairing-fixed); driver v3",
            },
            open(os.path.join(OUT, "w_v05_manifest.json"), "w"),
            indent=1,
        )
        print(
            "W_V05 GROSS cells=%d runs=%d | NEW %d SKIPPED %d | NET ok=%d | wall=%.0fs"
            % (
                len(grid),
                len(grid) * 20,
                done,
                skipped,
                int(df["c_ideal_fin"].notna().sum()),
                time.time() - t0,
            )
        )
    else:
        sys.exit("unknown arm %r (expected 'w_v05' or 'legacy')" % arm)
