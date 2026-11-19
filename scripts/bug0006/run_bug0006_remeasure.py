"""Post-BUG-0006 re-measure sweep (`T1`, `W_v05`, `M2_xi`).

Re-runs the three load-bearing 200-step parameter grids after fixing BUG-0006
(unscored step-t newborns carry ``ideal_Merit = np.nan`` and are excluded from
``correlation_reporter`` until ``update_ideal_merit`` scores them on step t+1):

- ``T1`` (36 cells x 20 seeds = 720 runs): ``cur in {0.3, 0.5, 0.7}``,
  ``van in {0.5, 0.9, 1.0}``, ``land in {mgstd5, noisypr5}``, ``g in {0.0, 0.1}``
  with ``use_visibility_reward = True``, H = 200. The 18 ``g=0.0`` cells form
  an in-sweep bit-identity control arm against ``data/remeasure_v1/runs.csv``.
- ``W_v05`` (40 cells x 20 seeds = 800 runs): ``cur in {0.1, 0.3, 0.5, 0.7, 0.9}``,
  ``vis in {True, False}``, ``g in {0.0, 0.1}``, ``land in {mgstd5, mg6}``
  with ``vanishing = 0.5``, H = 200 (mirror of ``data/remeasure_v1/w_v05_runs.csv``).
- ``M2_xi`` (10 cells x 20 seeds = 200 runs): ``cur in {0.1, 0.3, 0.5, 0.7, 0.9}``,
  ``dyadic_lambda in {0.0, 1.0}`` on ``mgstd5``, ``g = 0.1``, ``vanishing = 0.9``,
  ``dyadic_mode = "logit"``, ``credit_sharing = "equal_split"``,
  ``dyadic_move_pull = 1.0``, H = 200 (mirror of ``path_b_v2`` + ``fig1_extension_v1``).

Writes strictly to ``data/bug0006_remeasure_v2/`` with atomic per-cell
checkpoints and skip-on-restart. Uses stdlib ``ProcessPoolExecutor`` (macOS
spawn-safe, zero lambdas). Model-internal; NOT calibrated (owner ruling 2026-11-13).

Usage:
    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 scripts/bug0006/run_bug0006_remeasure.py --probe
    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 scripts/bug0006/run_bug0006_remeasure.py --arm all --n-jobs 6
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sciencemodels import landscapes as lf  # noqa: E402
from sciencemodels.model import MyModel  # noqa: E402

OUT = ROOT / "data" / "bug0006_remeasure_v2"
CELLS_T1 = OUT / "cells_t1"
CELLS_WV05 = OUT / "cells_w_v05"
CELLS_M2XI = OUT / "cells_m2_xi"

DRIVER_VERSION = "bug0006_remeasure_v2 / driver 1"
HORIZON = 200
SEEDS = 20

LAND = {
    "mgstd5": (lf.multipleGaussians, {"std": 5, "number_gaussians": 2}),
    "mg6": (lf.multipleGaussians, {"std": 5, "number_gaussians": 6}),
    "noisypr5": (
        lf.noisyGaussian,
        {"prop_random": 5, "std": 5, "std_curiosity": 0.1},
    ),
}

SOURCE_FILES = (
    "sciencemodels/model.py",
    "sciencemodels/agents.py",
    "sciencemodels/metrics.py",
    "sciencemodels/landscapes.py",
    "sciencemodels/io.py",
)

REQUIRED_COMMON_COLS = (
    "origin",
    "curiosity",
    "vanishing",
    "landscape",
    "agent_generation_time",
    "use_visibility_reward",
    "seed",
    "c_ideal_fin",
    "c_ideal_tm",
    "c_local_fin",
    "c_local_tm",
    "top10_fin",
    "top10_tm",
    "ideal_loss_cum",
    "harvested_fin",
    "ideal_merit_sum",
    "ideal_merit_mean",
    "n_unscored_ideal",
    "wedge_local_minus_ideal_fin",
    "wedge_local_minus_ideal_tm",
    "c_ideal_step199",
    "c_local_step199",
    "top10_step199",
)


def identity(cur, rng, params):
    """Top-level picklable curiosity generator (spawn-safe)."""
    return cur


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_sources() -> dict[str, str]:
    return {rel: _sha256(ROOT / rel) for rel in SOURCE_FILES}


# ---------------------------------------------------------------------------
# Grid specs & tags
# ---------------------------------------------------------------------------
def t1_specs() -> list[tuple]:
    return [
        ("T1", cur, van, land, gen, True)
        for cur, van, land, gen in itertools.product(
            (0.3, 0.5, 0.7),
            (0.5, 0.9, 1.0),
            ("mgstd5", "noisypr5"),
            (0.0, 0.1),
        )
    ]


def w_v05_specs() -> list[tuple]:
    return [
        ("Wv05", cur, 0.5, land, gen, vis)
        for cur, vis, gen, land in itertools.product(
            (0.1, 0.3, 0.5, 0.7, 0.9),
            (True, False),
            (0.0, 0.1),
            ("mgstd5", "mg6"),
        )
    ]


def m2_xi_specs() -> list[tuple]:
    return [
        ("M2_xi", cur, lam)
        for cur, lam in itertools.product(
            (0.1, 0.3, 0.5, 0.7, 0.9),
            (0.0, 1.0),
        )
    ]


def cell_tag(arm: str, spec: tuple) -> str:
    if arm in ("T1", "W_v05"):
        origin, cur, van, land, gen, vis = spec
        return "cell_%s_cur%02d_van%02d_%s_gen%02d_vis%d" % (
            origin,
            int(round(cur * 10)),
            int(round(van * 10)),
            land,
            int(round(gen * 10)),
            int(vis),
        )
    if arm == "M2_xi":
        _, cur, lam = spec
        return "cell_M2_cur%02d_lam%03d_pull100_logit_equal_mgstd5" % (
            int(round(cur * 10)),
            int(round(lam * 100)),
        )
    raise ValueError(f"Unknown arm {arm!r}")


# ---------------------------------------------------------------------------
# Single-run runners
# ---------------------------------------------------------------------------
def _extract_common(m: MyModel, origin: str, cur: float, van: float, land: str,
                    gen: float, vis: bool, seed: int) -> dict:
    dc = m.datacollector.get_model_vars_dataframe()
    last = dc.iloc[-1]
    prev = dc.iloc[-2]
    row = dict(
        origin=origin,
        curiosity=float(cur),
        vanishing=float(van),
        landscape=land,
        agent_generation_time=float(gen),
        use_visibility_reward=bool(vis),
        seed=int(seed),
    )
    for src, dst in (
        ("Corr_prestige_ideal_merit", "c_ideal"),
        ("Corr_prestige_local_merit", "c_local"),
        ("top10pct_prestige_share", "top10"),
    ):
        row[dst + "_fin"] = float(last[src])
        row[dst + "_tm"] = float(dc[src].mean())
        row[dst + "_step199"] = float(prev[src])
    row["ideal_loss_cum"] = -float(
        sum(getattr(a, "ideal_merit_cum", 0.0) for a in m._scientists)
    )
    row["harvested_fin"] = float(last["percentage_knowledge_harvested"])
    finite_merits = [
        float(a.ideal_Merit) for a in m.agents if np.isfinite(a.ideal_Merit)
    ]
    n_unscored = sum(
        1 for a in m.agents if not np.isfinite(a.ideal_Merit)
    )
    row["ideal_merit_sum"] = float(np.sum(finite_merits)) if finite_merits else np.nan
    row["ideal_merit_mean"] = float(np.mean(finite_merits)) if finite_merits else np.nan
    row["n_unscored_ideal"] = int(n_unscored)
    row["wedge_local_minus_ideal_fin"] = row["c_local_fin"] - row["c_ideal_fin"]
    row["wedge_local_minus_ideal_tm"] = row["c_local_tm"] - row["c_ideal_tm"]
    row["agents_final"] = len(m._scientists)
    row["newcomers_final"] = int(
        sum(1 for a in m._scientists if getattr(a, "is_newcomer", False))
    )
    row["mean_prestige_final"] = float(last["mean_prestige"])
    return row


def run_one_base(spec: tuple, seed: int) -> dict:
    origin, cur, van, land, gen, vis = spec
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
        step_limit=HORIZON,
        agent_generation_function=identity,
        vanishing_factor_for_prestige=van,
        use_visibility_reward=vis,
        knowledge_normalization="share",
        prestige_accrual="harvested",
    )
    for _ in range(HORIZON):
        m.step(True)
    return _extract_common(m, origin, cur, van, land, gen, vis, seed)


def run_one_m2(spec: tuple, seed: int) -> dict:
    _, cur, lam = spec
    f, gp = LAND["mgstd5"]
    m = MyModel(
        n_agents=20,
        initial_curiosity=cur,
        epsilon=0.1,
        harvest=0.02,
        grid_size=20,
        cell_init_function=f,
        generation_params=dict(gp),
        use_distance=True,
        agent_generation_time=0.1,
        new_questions=0,
        agent_seed=seed,
        step_limit=HORIZON,
        agent_generation_function=identity,
        vanishing_factor_for_prestige=0.9,
        use_visibility_reward=True,
        knowledge_normalization="share",
        prestige_accrual="harvested",
        entry_prestige_stock="prestige",
        dyadic_mode="logit",
        dyadic_lambda=lam,
        dyadic_proposal_regime="newcomer",
        credit_sharing="equal_split",
        closeness_mode="grid",
        dyadic_visibility_stock="prestige_visibility",
        dyadic_move_pull=1.0,
    )
    for _ in range(HORIZON):
        m.step(True)
    row = _extract_common(m, "M2_xi", cur, 0.9, "mgstd5", 0.1, True, seed)
    dc = m.datacollector.get_model_vars_dataframe()
    last = dc.iloc[-1]
    row["xi"] = 1.0 - float(cur)
    row["dyadic_lambda"] = float(lam)
    row["dyadic_move_pull"] = 1.0
    row["formulation"] = "logit"
    row["credit_sharing_axis"] = "equal_split"
    for src, dst in (
        ("mean_team_size", "team_tm"),
        ("repeated_collab_share", "rep_tm"),
        ("dyadic_assortativity_prestige", "assort_tm"),
        ("n_active_links", "links_tm"),
        ("link_repeated_share", "link_rep_tm"),
        ("link_assortativity_prestige", "link_assort_tm"),
    ):
        row[dst] = float(dc[src].mean())
    row["mean_team_size_fin"] = float(last["mean_team_size"])
    row["repeated_collab_share_fin"] = float(last["repeated_collab_share"])
    row["n_active_links_fin"] = int(last["n_active_links"])
    return row


def run_cell_task(task: tuple[str, tuple, str]) -> tuple[str, str, int]:
    """Worker entry point (top-level, picklable on macOS spawn)."""
    arm, spec, out_dir_str = task
    out_dir = Path(out_dir_str)
    tag = cell_tag(arm, spec)
    cpath = out_dir / f"{tag}.csv"
    if cpath.exists():
        with cpath.open() as fh:
            if sum(1 for _ in fh) == SEEDS + 1:
                return tag, "skipped", SEEDS
    rows = []
    for s in range(SEEDS):
        if arm == "M2_xi":
            rows.append(run_one_m2(spec, s))
        else:
            rows.append(run_one_base(spec, s))
    df = pd.DataFrame(rows)
    assert len(df) == SEEDS, f"{tag}: expected {SEEDS} rows, got {len(df)}"
    assert df["c_ideal_fin"].notna().all(), f"{tag}: NaN c_ideal_fin"
    tmp = out_dir / f"{tag}.csv.tmp.{os.getpid()}"
    df.to_csv(tmp, index=False)
    back = pd.read_csv(tmp)
    assert len(back) == SEEDS, f"{tag}: readback length {len(back)}"
    os.replace(tmp, cpath)
    return tag, "ok", len(back)


# ---------------------------------------------------------------------------
# Proofs & probe
# ---------------------------------------------------------------------------
def run_canary() -> str:
    env = dict(os.environ, MPLBACKEND="Agg")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tests" / "test_smoke.py")],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Smoke test failed:\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout.strip()


def run_probe(n_jobs: int) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    canary_txt = run_canary()
    (OUT / "canary.txt").write_text(canary_txt + "\n")

    # 1. g=0 bit-identity control vs data/remeasure_v1/runs.csv
    arch_t1 = pd.read_csv(ROOT / "data" / "remeasure_v1" / "runs.csv")
    arch_t1 = arch_t1[arch_t1["origin"] == "T1"]
    ctrl_spec = ("T1", 0.5, 0.9, "mgstd5", 0.0, True)
    ctrl_diffs = {}
    t0 = time.time()
    for seed in (0, 1):
        got = run_one_base(ctrl_spec, seed)
        ref = arch_t1[
            (arch_t1["curiosity"] == 0.5)
            & (arch_t1["vanishing"] == 0.9)
            & (arch_t1["landscape"] == "mgstd5")
            & (arch_t1["agent_generation_time"] == 0.0)
            & (arch_t1["seed"] == seed)
        ].iloc[0]
        diffs = {
            k: abs(float(got[k]) - float(ref[k]))
            for k in ("c_ideal_fin", "c_ideal_tm", "c_local_fin", "c_local_tm",
                      "top10_fin", "harvested_fin", "ideal_loss_cum")
        }
        ctrl_diffs[f"seed_{seed}"] = diffs
        for k, d in diffs.items():
            assert d <= 1e-12, f"g=0 control drift on {k} seed={seed}: {d}"
    t_g0_per_run = (time.time() - t0) / 2.0

    # 2. g=0.1 binding proof vs data/remeasure_v1/runs.csv & w_v05_runs.csv
    t1 = time.time()
    g1_t1 = run_one_base(("T1", 0.5, 0.9, "mgstd5", 0.1, True), 0)
    t_g1_per_run = time.time() - t1
    ref_g1_t1 = arch_t1[
        (arch_t1["curiosity"] == 0.5)
        & (arch_t1["vanishing"] == 0.9)
        & (arch_t1["landscape"] == "mgstd5")
        & (arch_t1["agent_generation_time"] == 0.1)
        & (arch_t1["seed"] == 0)
    ].iloc[0]
    non_ideal_t1 = {
        k: abs(float(g1_t1[k]) - float(ref_g1_t1[k]))
        for k in ("c_local_fin", "c_local_tm", "top10_fin", "harvested_fin", "ideal_loss_cum")
    }
    for k, d in non_ideal_t1.items():
        assert d <= 1e-12, f"g=0.1 non-ideal drift on {k}: {d}"
    assert g1_t1["n_unscored_ideal"] == 1
    assert g1_t1["c_ideal_fin"] != float(ref_g1_t1["c_ideal_fin"])

    arch_w = pd.read_csv(ROOT / "data" / "remeasure_v1" / "w_v05_runs.csv")
    g1_w = run_one_base(("Wv05", 0.5, 0.5, "mgstd5", 0.1, True), 0)
    ref_g1_w = arch_w[
        (arch_w["curiosity"] == 0.5)
        & (arch_w["landscape"] == "mgstd5")
        & (arch_w["agent_generation_time"] == 0.1)
        & (arch_w["use_visibility_reward"] == True)  # noqa: E712
        & (arch_w["seed"] == 0)
    ].iloc[0]
    non_ideal_w = {
        k: abs(float(g1_w[k]) - float(ref_g1_w[k]))
        for k in ("c_local_fin", "top10_fin", "harvested_fin", "ideal_loss_cum")
    }
    for k, d in non_ideal_w.items():
        assert d <= 1e-12, f"W_v05 non-ideal drift on {k}: {d}"

    t2 = time.time()
    m2_probe = run_one_m2(("M2_xi", 0.5, 1.0), 0)
    t_m2_per_run = time.time() - t2
    ref_m2 = pd.read_csv(
        ROOT / "data" / "path_b_v2" / "cells" / "cell_lam100_pull100_logit_equal_mgstd5.csv"
    )
    ref_m2_0 = ref_m2[ref_m2["agent_seed"] == 0].iloc[0]
    non_ideal_m2 = {
        "team_tm": abs(float(m2_probe["team_tm"]) - float(ref_m2_0["team_tm"])),
        "links_tm": abs(float(m2_probe["links_tm"]) - float(ref_m2_0["links_tm"])),
        "c_local_tm": abs(float(m2_probe["c_local_tm"]) - float(ref_m2_0["c_local_tm"])),
        "top10_fin": abs(float(m2_probe["top10_fin"]) - float(ref_m2_0["top10pct_prestige_share"])),
    }
    for k, d in non_ideal_m2.items():
        assert d <= 1e-12, f"M2_xi non-ideal drift on {k}: {d}"

    # 3. Worker-written schema proof
    with tempfile.TemporaryDirectory(prefix="bug0006_schema_", dir="/tmp") as tmpd:
        with ProcessPoolExecutor(max_workers=min(2, n_jobs)) as ex:
            fut = ex.submit(run_one_m2, ("M2_xi", 0.5, 1.0), 0)
            worker_row = fut.result()
        p = Path(tmpd) / "worker_schema.csv"
        pd.DataFrame([worker_row]).to_csv(p, index=False)
        back = pd.read_csv(p)
        missing = [c for c in REQUIRED_COMMON_COLS if c not in back.columns]
        assert not missing, f"worker CSV missing columns: {missing}"

    est_total_cpu_s = 720 * ((t_g0_per_run + t_g1_per_run) / 2.0) + 800 * ((t_g0_per_run + t_g1_per_run) / 2.0) + 200 * t_m2_per_run
    probe_report = {
        "driver": DRIVER_VERSION,
        "canary_ok": True,
        "g0_control_max_diff": max(max(d.values()) for d in ctrl_diffs.values()),
        "g0_control_diffs": ctrl_diffs,
        "t1_g01_binding": {
            "pre_c_ideal_fin": float(ref_g1_t1["c_ideal_fin"]),
            "post_c_ideal_fin": float(g1_t1["c_ideal_fin"]),
            "pre_c_ideal_tm": float(ref_g1_t1["c_ideal_tm"]),
            "post_c_ideal_tm": float(g1_t1["c_ideal_tm"]),
            "non_ideal_max_diff": max(non_ideal_t1.values()),
        },
        "w_v05_g01_binding": {
            "pre_c_ideal_fin": float(ref_g1_w["c_ideal_fin"]),
            "post_c_ideal_fin": float(g1_w["c_ideal_fin"]),
            "non_ideal_max_diff": max(non_ideal_w.values()),
        },
        "m2_xi_binding": {
            "pre_c_ideal_tm": float(ref_m2_0["c_ideal_tm"]),
            "post_c_ideal_tm": float(m2_probe["c_ideal_tm"]),
            "post_c_ideal_fin": float(m2_probe["c_ideal_fin"]),
            "non_ideal_max_diff": max(non_ideal_m2.values()),
        },
        "timing_s_per_run": {
            "base_g0": round(t_g0_per_run, 3),
            "base_g01": round(t_g1_per_run, 3),
            "m2_g01": round(t_m2_per_run, 3),
        },
        "est_wall_s_at_n_jobs": {str(n_jobs): round(est_total_cpu_s / n_jobs, 1)},
    }
    (OUT / "probe_results.json").write_text(json.dumps(probe_report, indent=2))
    print(json.dumps(probe_report, indent=2))
    return probe_report


# ---------------------------------------------------------------------------
# Sweep execution & consolidation
# ---------------------------------------------------------------------------
def run_arm(arm: str, specs: list[tuple], cells_dir: Path, n_jobs: int) -> dict:
    cells_dir.mkdir(parents=True, exist_ok=True)
    log_path = OUT / f"{arm.lower()}_run_log.txt"
    t0 = time.time()
    tasks = [(arm, spec, str(cells_dir)) for spec in specs]
    done = 0
    skipped = 0
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futs = {ex.submit(run_cell_task, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            tag, status, nrows = fut.result()
            if status == "ok":
                done += 1
            else:
                skipped += 1
            msg = f"[{arm}] {i}/{len(tasks)} {tag} -> {status} ({nrows} rows)"
            print(msg, flush=True)
            with log_path.open("a") as fh:
                fh.write(msg + "\n")
    wall = round(time.time() - t0, 1)
    return {"arm": arm, "cells_gross": len(specs), "cells_new": done,
            "cells_skipped": skipped, "wall_s": wall}


def consolidate_arm(arm: str, cells_dir: Path, runs_out: Path, summary_out: Path,
                    group_keys: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted(cells_dir.glob("cell_*.csv"))
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(runs_out, index=False)
    numeric = [c for c in df.columns if c not in group_keys + ["seed"] and pd.api.types.is_numeric_dtype(df[c])]
    summ = df.groupby(group_keys, as_index=False)[numeric].mean()
    counts = df.groupby(group_keys, as_index=False).size().rename(columns={"size": "n_seeds"})
    summ = summ.merge(counts, on=group_keys, how="left")
    summ.to_csv(summary_out, index=False)
    return df, summ


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--arm", default="all", choices=("all", "T1", "W_v05", "M2_xi"))
    ap.add_argument("--n-jobs", type=int, default=6)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if args.probe:
        run_probe(args.n_jobs)
        return 0

    canary_txt = run_canary()
    (OUT / "canary.txt").write_text(canary_txt + "\n")
    src_before = snapshot_sources()

    design = {
        "driver_version": DRIVER_VERSION,
        "bug": "BUG-0006 post-fix re-measure",
        "horizon": HORIZON,
        "seeds": SEEDS,
        "n_jobs": args.n_jobs,
        "wedge_convention": "c_local - c_ideal (wedge_local_minus_ideal_*)",
        "arms": {
            "T1": {"cells": 36, "runs": 720, "vanishing": [0.5, 0.9, 1.0]},
            "W_v05": {"cells": 40, "runs": 800, "vanishing": 0.5},
            "M2_xi": {"cells": 10, "runs": 200, "vanishing": 0.9, "pull": 1.0,
                      "formulation": "logit", "credit": "equal_split"},
        },
        "sources_sha256": src_before,
        "not_calibrated": "Model-internal; NOT calibrated (owner ruling 2026-11-13)",
    }
    (OUT / "design.json").write_text(json.dumps(design, indent=2))

    arm_stats = {}
    if args.arm in ("all", "T1"):
        arm_stats["T1"] = run_arm("T1", t1_specs(), CELLS_T1, args.n_jobs)
        df_t1, _ = consolidate_arm(
            "T1", CELLS_T1, OUT / "t1_runs.csv", OUT / "t1_summary.csv",
            ["origin", "curiosity", "vanishing", "landscape", "agent_generation_time", "use_visibility_reward"],
        )
        arm_stats["T1"]["rows_net"] = len(df_t1)
        assert len(df_t1) == 720, f"T1 expected 720 rows, got {len(df_t1)}"

    if args.arm in ("all", "W_v05"):
        arm_stats["W_v05"] = run_arm("W_v05", w_v05_specs(), CELLS_WV05, args.n_jobs)
        df_w, _ = consolidate_arm(
            "W_v05", CELLS_WV05, OUT / "w_v05_runs.csv", OUT / "w_v05_summary.csv",
            ["origin", "curiosity", "vanishing", "landscape", "agent_generation_time", "use_visibility_reward"],
        )
        arm_stats["W_v05"]["rows_net"] = len(df_w)
        assert len(df_w) == 800, f"W_v05 expected 800 rows, got {len(df_w)}"

    if args.arm in ("all", "M2_xi"):
        arm_stats["M2_xi"] = run_arm("M2_xi", m2_xi_specs(), CELLS_M2XI, args.n_jobs)
        df_m2, _ = consolidate_arm(
            "M2_xi", CELLS_M2XI, OUT / "m2_xi_runs.csv", OUT / "m2_xi_summary.csv",
            ["origin", "curiosity", "xi", "dyadic_lambda", "dyadic_move_pull", "formulation", "credit_sharing_axis", "landscape"],
        )
        arm_stats["M2_xi"]["rows_net"] = len(df_m2)
        assert len(df_m2) == 200, f"M2_xi expected 200 rows, got {len(df_m2)}"

    src_after = snapshot_sources()
    assert src_before == src_after, "model source files mutated during sweep"
    manifest = {
        "driver_version": DRIVER_VERSION,
        "arms": arm_stats,
        "sources_moved_during_sweep": False,
        "sources_sha256": src_after,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
