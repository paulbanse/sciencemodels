"""Demographic turnover v2 (post-BUG-0006 + staggered cohort replacement).

Extends ``scripts/turnover/run_turnover_v1.py`` cleanly in a new namespace
(``data/turnover_v2/``) after fixing BUG-0006 in core ``sciencemodels/``.
Adds a clean 2-step causal decomposition over 4 exit modes:

1. ``none``: immortal founders, ``N(t) = 21 + floor(g * t)``.
2. ``fixed_span``: every scheduled arrival retires the oldest active scientist
   so ``N`` stays pinned at ``N_KEEP = 20``.
3. ``age_hazard_sync``: exact ``turnover_v1`` rule (all generation-0 agents
   start at ``birth_step = 0`` and retire simultaneously at tenure ``>= 200``),
   isolating the pure ``BUG-0006`` fix relative to ``data/turnover_v1/``.
4. ``age_hazard_staggered``: identical ``tau_max = 200`` tenure-cap replacement
   rule, except generation-0 agents receive deterministic evenly staggered
   initial ages ``birth_step = -((k * TAU_MAX) // n0)`` (zero RNG calls),
   eliminating the synchronized step-200 / step-400 replacement shock.

Grid:
- Main grid: 4 modes x 3 entry (0.05, 0.10, 0.20) x 3 curiosities (0.3, 0.5, 0.7)
  x 2 landscapes (mgstd5, noisypr5) = 72 cells x 20 seeds = 1,440 runs (H=400).
- Spine extension (``--include-xi-ends``, default on): 2 modes
  (``age_hazard_sync``, ``age_hazard_staggered``) x entry 0.10 x cur in {0.1, 0.9}
  x mgstd5 = 4 cells x 20 seeds = 80 runs (H=400) for complete 5-point
  ``xi in [0.1, 0.9]`` turnover series.
Total: 76 cells, 1,520 runs. Uses stdlib ``ProcessPoolExecutor`` (macOS
spawn-safe, zero lambdas) and atomic per-cell checkpoints.

Model-internal; NOT calibrated (owner ruling 2026-11-13).

Usage:
    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 scripts/turnover/run_turnover_v2.py --probe
    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 scripts/turnover/run_turnover_v2.py --n-jobs 6
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import inspect
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sciencemodels import landscapes as lf  # noqa: E402
from sciencemodels.agents import Scientist  # noqa: E402
from sciencemodels.model import MyModel  # noqa: E402

OUT = ROOT / "data" / "turnover_v2"
CELLS = OUT / "cells"
TRAJ = OUT / "traj"

DRIVER_VERSION = "turnover_v2 / driver 1 (post-BUG-0006 + staggered)"
HORIZON = 400
SEEDS = 20
TAU_MAX = 200
N_KEEP = 20
T_WINDOW = 20

MODES = ("none", "fixed_span", "age_hazard_sync", "age_hazard_staggered")
ENTRY = (0.05, 0.10, 0.20)
CURIOSITY = (0.3, 0.5, 0.7)

LAND = {
    "mgstd5": (lf.multipleGaussians, {"std": 5, "number_gaussians": 2}),
    "noisypr5": (
        lf.noisyGaussian,
        {"prop_random": 5, "std": 5, "std_curiosity": 0.1},
    ),
}

CANARY_TOKENS = (
    "agents_to_generate",
    "entry_stock = self.entry_prestige_stock",
    "self._scientists.append(a)",
)

SOURCE_FILES = (
    "sciencemodels/model.py",
    "sciencemodels/agents.py",
    "sciencemodels/metrics.py",
    "sciencemodels/landscapes.py",
    "sciencemodels/io.py",
)


def identity(cur, rng, params):
    """Top-level picklable curiosity generator (spawn-safe)."""
    return cur


def _nanmean(values) -> float:
    a = np.asarray(values, dtype=float)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_sources() -> dict[str, str]:
    return {rel: _sha256(ROOT / rel) for rel in SOURCE_FILES}


# ---------------------------------------------------------------------------
# Driver-side exit rules
# ---------------------------------------------------------------------------
def _retire(model, agent) -> None:
    if agent in model._scientists:
        model._scientists.remove(agent)
    try:
        model.grid.remove_agent(agent)
    except Exception:
        pass
    try:
        agent.remove()
    except Exception:
        pass


def _spawn_from_roulette(model, agent_generation_function):
    entry_stock = model.entry_prestige_stock
    tuples = [(getattr(a, entry_stock), a) for a in model._scientists]
    total = sum(getattr(a, entry_stock) for a in model._scientists)
    val = model.rng.uniform(0, total)
    sup = model._scientists[0] if model._scientists else None
    for p, a in tuples:
        if val <= p:
            sup = a
            break
        val -= p
    w = agent_generation_function(
        model.initial_curiosity, model.rng, model.generation_params
    )
    a = Scientist(model, w, model.epsilon)
    a.ideal_merit_cum = np.float64(0.0)
    a.is_newcomer = True
    a.birth_step = model.steps
    model.grid.place_agent(a, sup.pos)
    model._scientists.append(a)
    return a


def install_exit_rule(mode: str, agent_generation_function) -> None:
    if mode not in MODES:
        raise ValueError(f"unknown exit mode {mode!r}")

    attr = "_MyModel__generate_new_agents"
    original = getattr(MyModel, "_turnover_original", None)
    if original is None:
        original = getattr(MyModel, attr)
        src = inspect.getsource(original)
        missing = [t for t in CANARY_TOKENS if t not in src]
        if missing:
            raise RuntimeError(
                f"divergence canary: sciencemodels/model.py missing {missing!r}"
            )
        MyModel._turnover_original = original

    if getattr(MyModel, "_turnover_mode", None) == mode:
        return

    def patched(self):
        n_before = len(self._scientists)
        original(self)
        born = self._scientists[n_before:]
        for a in born:
            a.birth_step = self.steps

        if mode == "none":
            return

        if mode == "fixed_span":
            for _ in born:
                if len(self._scientists) > N_KEEP:
                    victim = max(
                        self._scientists, key=lambda a: (a.age, -a.unique_id)
                    )
                    _retire(self, victim)
            return

        if mode in ("age_hazard_sync", "age_hazard_staggered"):
            due = [
                a
                for a in list(self._scientists)
                if self.steps - getattr(a, "birth_step", 0) >= TAU_MAX
            ]
            for a in due:
                _retire(self, a)
                _spawn_from_roulette(self, agent_generation_function)

    patched._turnover_mode = mode
    setattr(MyModel, attr, patched)
    MyModel._turnover_mode = mode


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------
def run_one(mode: str, g: float, cur: float, land: str, seed: int,
            horizon: int = HORIZON) -> tuple[dict, pd.DataFrame]:
    install_exit_rule(mode, identity)
    f, gp = LAND[land]
    m = MyModel(
        n_agents=N_KEEP,
        initial_curiosity=cur,
        epsilon=0.1,
        harvest=0.02,
        grid_size=20,
        cell_init_function=f,
        generation_params=dict(gp),
        use_distance=True,
        agent_generation_time=g,
        new_questions=0,
        agent_seed=seed,
        step_limit=horizon,
        agent_generation_function=identity,
        vanishing_factor_for_prestige=0.9,
        use_visibility_reward=True,
        knowledge_normalization="share",
        prestige_accrual="harvested",
    )

    # Generation-0 birth_step assignment (zero RNG calls).
    # Note: MyModel.__init__ runs end_loop_update() at steps==0, so m._scientists
    # contains N_KEEP + 1 = 21 agents for g in (0, 1].
    n0 = len(m._scientists)
    if mode == "age_hazard_staggered":
        for k, a in enumerate(m._scientists):
            a.birth_step = -((k * TAU_MAX) // n0)
            a._is_gen0 = True
    else:
        for a in m._scientists:
            a.birth_step = 0
            a._is_gen0 = True

    gap, gap_recent, gap_recent_ratio = [], [], []
    n_active, n_founders, mean_age, mean_tenure = [], [], [], []
    for _ in range(horizon):
        m.step(True)
        sci = m._scientists
        n_active.append(len(sci))
        n_founders.append(sum(1 for a in sci if getattr(a, "_is_gen0", False)))
        mean_age.append(float(np.mean([a.age for a in sci])))
        ten = [m.steps - getattr(a, "birth_step", 0) for a in sci]
        mean_tenure.append(float(np.mean(ten)))

        new = [a.prestige for a in sci if getattr(a, "is_newcomer", False)]
        old = [a.prestige for a in sci if not getattr(a, "is_newcomer", False)]
        gap.append(float(np.mean(new) - np.mean(old)) if new and old else np.nan)

        recent = [a.prestige for a, t in zip(sci, ten) if t < T_WINDOW]
        est = [a.prestige for a, t in zip(sci, ten) if t >= T_WINDOW]
        if recent and est:
            gr = float(np.mean(recent) - np.mean(est))
            est_m = float(np.mean(est))
            grr = float(gr / est_m) if est_m > 0 else np.nan
        else:
            gr = np.nan
            grr = np.nan
        gap_recent.append(gr)
        gap_recent_ratio.append(grr)

    dc = m.datacollector.get_model_vars_dataframe()
    dcs = dc.iloc[-horizon:]
    last = dcs.iloc[-1]
    prev = dcs.iloc[-2]
    late = dcs.iloc[-100:]

    finite_merits = [
        float(a.ideal_Merit) for a in m.agents if np.isfinite(a.ideal_Merit)
    ]
    n_unscored = sum(
        1 for a in m.agents if not np.isfinite(a.ideal_Merit)
    )
    retired_step = next((i + 1 for i, k in enumerate(n_founders) if k == 0), None)

    c_ideal_fin = float(last["Corr_prestige_ideal_merit"])
    c_local_fin = float(last["Corr_prestige_local_merit"])
    c_ideal_tm = float(dc["Corr_prestige_ideal_merit"].mean())
    c_local_tm = float(dc["Corr_prestige_local_merit"].mean())
    c_ideal_late = float(late["Corr_prestige_ideal_merit"].mean())
    c_local_late = float(late["Corr_prestige_local_merit"].mean())

    row = dict(
        exit_mode=mode,
        entry=float(g),
        curiosity=float(cur),
        xi=round(1.0 - float(cur), 6),
        landscape=land,
        seed=int(seed),
        horizon=horizon,
        n_final=int(n_active[-1]),
        n_founders_final=int(n_founders[-1]),
        founders_retired_step=retired_step,
        mean_age_final=float(mean_age[-1]),
        mean_tenure_final=float(mean_tenure[-1]),
        gap_flag_final=gap[-1],
        gap_flag_mean_last100=_nanmean(gap[-100:]),
        gap_recent_final=gap_recent[-1],
        gap_recent_mean_last100=_nanmean(gap_recent[-100:]),
        gap_recent_ratio_final=gap_recent_ratio[-1],
        gap_recent_ratio_mean_last100=_nanmean(gap_recent_ratio[-100:]),
        n_mean_last100=float(np.mean(n_active[-100:])),
        c_ideal_fin=c_ideal_fin,
        c_local_fin=c_local_fin,
        c_ideal_tm=c_ideal_tm,
        c_local_tm=c_local_tm,
        c_ideal_late_mean=c_ideal_late,
        c_local_late_mean=c_local_late,
        c_ideal_step399=float(prev["Corr_prestige_ideal_merit"]),
        c_local_step399=float(prev["Corr_prestige_local_merit"]),
        wedge_local_minus_ideal_fin=c_local_fin - c_ideal_fin,
        wedge_local_minus_ideal_tm=c_local_tm - c_ideal_tm,
        wedge_local_minus_ideal_late=c_local_late - c_ideal_late,
        top10_fin=float(last["top10pct_prestige_share"]),
        top10_tm=float(dc["top10pct_prestige_share"].mean()),
        top10_late_mean=float(late["top10pct_prestige_share"].mean()),
        top10_step399=float(prev["top10pct_prestige_share"]),
        harvested_fin=float(last["percentage_knowledge_harvested"]),
        ideal_loss_cum=-float(
            sum(getattr(a, "ideal_merit_cum", 0.0) for a in m._scientists)
        ),
        ideal_merit_sum=float(np.sum(finite_merits)) if finite_merits else np.nan,
        ideal_merit_mean=float(np.mean(finite_merits)) if finite_merits else np.nan,
        n_unscored_ideal=int(n_unscored),
        mean_prestige_final=float(last["mean_prestige"]),
    )
    traj = pd.DataFrame(
        {
            "step": np.arange(1, horizon + 1),
            "gap": gap,
            "gap_recent": gap_recent,
            "gap_recent_ratio": gap_recent_ratio,
            "n_active": n_active,
            "n_founders": n_founders,
            "mean_age": mean_age,
            "mean_tenure": mean_tenure,
            "c_ideal": dcs["Corr_prestige_ideal_merit"].to_numpy(),
            "c_local": dcs["Corr_prestige_local_merit"].to_numpy(),
            "top10": dcs["top10pct_prestige_share"].to_numpy(),
        }
    )
    return row, traj


# ---------------------------------------------------------------------------
# Grid, tags, and workers
# ---------------------------------------------------------------------------
def cell_tag(spec: tuple) -> str:
    mode, g, cur, land = spec
    return "cell_%s_g%02d_cur%02d_%s" % (
        mode,
        int(round(g * 100)),
        int(round(cur * 10)),
        land,
    )


def cells(include_xi_ends: bool = True) -> list[tuple]:
    grid = list(itertools.product(MODES, ENTRY, CURIOSITY, LAND))
    if include_xi_ends:
        for mode in ("age_hazard_sync", "age_hazard_staggered"):
            for cur in (0.1, 0.9):
                grid.append((mode, 0.10, cur, "mgstd5"))
    return grid


def run_cell_task(spec: tuple) -> tuple[str, str, int]:
    mode, g, cur, land = spec
    tag = cell_tag(spec)
    cpath = CELLS / f"{tag}.csv"
    tpath = TRAJ / f"{tag}.csv"
    if cpath.exists() and tpath.exists():
        with cpath.open() as fh:
            if sum(1 for _ in fh) == SEEDS + 1:
                return tag, "skipped", SEEDS

    rows, trajs = [], []
    for s in range(SEEDS):
        row, traj = run_one(mode, g, cur, land, s, HORIZON)
        rows.append(row)
        trajs.append(traj)

    df = pd.DataFrame(rows)
    assert len(df) == SEEDS, f"{tag} incomplete"
    assert df["c_ideal_fin"].notna().all(), f"{tag} has NaN c_ideal_fin"

    ctmp = CELLS / f"{tag}.csv.tmp.{os.getpid()}"
    df.to_csv(ctmp, index=False)
    back = pd.read_csv(ctmp)
    assert len(back) == SEEDS
    os.replace(ctmp, cpath)

    tj = pd.concat(trajs, ignore_index=True)
    agg = tj.groupby("step", as_index=False).mean(numeric_only=True)
    agg.insert(0, "cell_tag", tag)
    ttmp = TRAJ / f"{tag}.csv.tmp.{os.getpid()}"
    agg.to_csv(ttmp, index=False)
    os.replace(ttmp, tpath)
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


def identity_proof() -> dict:
    """exit_mode='none' must be bit-identical to unpatched MyModel."""
    base = dict(
        n_agents=N_KEEP,
        initial_curiosity=0.5,
        epsilon=0.1,
        harvest=0.02,
        grid_size=20,
        cell_init_function=LAND["mgstd5"][0],
        generation_params=dict(LAND["mgstd5"][1]),
        use_distance=True,
        agent_generation_time=0.10,
        new_questions=0,
        agent_seed=7,
        step_limit=120,
        agent_generation_function=identity,
        vanishing_factor_for_prestige=0.9,
        use_visibility_reward=True,
        knowledge_normalization="share",
        prestige_accrual="harvested",
    )
    if getattr(MyModel, "_turnover_mode", None) is not None:
        setattr(MyModel, "_MyModel__generate_new_agents", MyModel._turnover_original)
        MyModel._turnover_mode = None
    clean = MyModel(**base)
    for _ in range(120):
        clean.step(True)
    ref = clean.get_report()

    install_exit_rule("none", identity)
    patched = MyModel(**base)
    for _ in range(120):
        patched.step(True)
    got = patched.get_report()

    diffs = {}
    for k in sorted(set(ref) | set(got)):
        a, b = ref.get(k), got.get(k)
        if isinstance(a, (int, float, np.floating, np.integer)) and isinstance(
            b, (int, float, np.floating, np.integer)
        ):
            if not (np.isnan(a) and np.isnan(b)) and float(a) != float(b):
                diffs[k] = [float(a), float(b)]
        elif a != b:
            diffs[k] = [str(a), str(b)]
    return {
        "keys_compared": len(set(ref) | set(got)),
        "differing_keys": diffs,
        "bit_identical": not diffs,
    }


def run_probe(n_jobs: int) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "canary.txt").write_text(run_canary() + "\n")
    ident = identity_proof()
    assert ident["bit_identical"], f"Identity proof failed: {ident['differing_keys']}"

    # Cross-version invariance check against data/turnover_v1/ none g=0.10 cur=0.5 mgstd5 seed=0
    v1_ref = pd.read_csv(
        ROOT / "data" / "turnover_v1" / "cells" / "cell_none_g10_cur05_mgstd5.csv"
    )
    v1_row = v1_ref[v1_ref["seed"] == 0].iloc[0]

    results = {}
    timings = {}
    for mode in MODES:
        t0 = time.time()
        row, traj = run_one(mode, 0.10, 0.5, "mgstd5", 0, HORIZON)
        timings[mode] = round(time.time() - t0, 3)
        results[mode] = {
            "n_final": row["n_final"],
            "n_founders_final": row["n_founders_final"],
            "founders_retired_step": row["founders_retired_step"],
            "c_ideal_fin": row["c_ideal_fin"],
            "c_ideal_step399": row["c_ideal_step399"],
            "c_ideal_late_mean": row["c_ideal_late_mean"],
            "c_ideal_tm": row["c_ideal_tm"],
            "top10_fin": row["top10_fin"],
            "top10_step399": row["top10_step399"],
            "n_unscored_ideal": row["n_unscored_ideal"],
            "traj_n_founders_steps_10_50_100_199_200": [
                int(traj.loc[traj["step"] == st, "n_founders"].iloc[0])
                for st in (10, 50, 100, 199, 200)
            ],
        }
        if mode == "none":
            non_ideal_diffs = {
                "c_local_fin": abs(row["c_local_fin"] - float(v1_row["c_local_fin"])),
                "top10_fin": abs(row["top10_fin"] - float(v1_row["top10_fin"])),
                "harvested_fin": abs(row["harvested_fin"] - float(v1_row["harvested_fin"])),
                "mean_prestige_final": abs(row["mean_prestige_final"] - float(v1_row["mean_prestige_final"])),
            }
            for k, d in non_ideal_diffs.items():
                assert d <= 1e-12, f"turnover_v1 non-ideal mismatch on {k}: {d}"
            results["none"]["vs_turnover_v1_non_ideal_max_diff"] = max(non_ideal_diffs.values())
            results["none"]["vs_turnover_v1_c_ideal_fin"] = [
                float(v1_row["c_ideal_fin"]), row["c_ideal_fin"]
            ]

    # Verify smooth founder retirement on staggered
    nf = results["age_hazard_staggered"]["traj_n_founders_steps_10_50_100_199_200"]
    assert nf[0] < 21 and nf[1] < nf[0] and nf[2] < nf[1] and nf[4] == 0, (
        f"staggered founder retirement failed: {nf}"
    )

    mean_s = sum(timings.values()) / len(timings)
    est_wall = round((76 * SEEDS * mean_s) / n_jobs, 1)
    rep = {
        "driver": DRIVER_VERSION,
        "identity_proof": ident,
        "probe_runs": results,
        "timing_s_per_run": timings,
        "est_wall_s_for_1520_runs_at_n_jobs": {str(n_jobs): est_wall},
    }
    (OUT / "probe_results.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--no-xi-ends", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    CELLS.mkdir(parents=True, exist_ok=True)
    TRAJ.mkdir(parents=True, exist_ok=True)

    if args.probe:
        run_probe(args.n_jobs)
        return 0

    (OUT / "canary.txt").write_text(run_canary() + "\n")
    src_before = snapshot_sources()
    grid = cells(include_xi_ends=not args.no_xi_ends)

    (OUT / "design.json").write_text(
        json.dumps(
            {
                "driver_version": DRIVER_VERSION,
                "modes": list(MODES),
                "entry_rates": list(ENTRY),
                "curiosities_main": list(CURIOSITY),
                "xi_ends_added": not args.no_xi_ends,
                "landscapes": list(LAND.keys()),
                "cells_gross": len(grid),
                "runs_gross": len(grid) * SEEDS,
                "horizon": HORIZON,
                "tau_max": TAU_MAX,
                "n_keep": N_KEEP,
                "t_window": T_WINDOW,
                "sources_sha256": src_before,
                "not_calibrated": "Model-internal; NOT calibrated (owner ruling 2026-11-13)",
            },
            indent=2,
        )
    )

    t0 = time.time()
    log_path = OUT / "run_log.txt"
    done, skipped = 0, 0
    with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
        futs = {ex.submit(run_cell_task, spec): spec for spec in grid}
        for i, fut in enumerate(as_completed(futs), 1):
            tag, status, nrows = fut.result()
            if status == "ok":
                done += 1
            else:
                skipped += 1
            msg = f"[turnover_v2] {i}/{len(grid)} {tag} -> {status} ({nrows} rows)"
            print(msg, flush=True)
            with log_path.open("a") as fh:
                fh.write(msg + "\n")

    frames = [pd.read_csv(f) for f in sorted(CELLS.glob("cell_*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(OUT / "runs.csv", index=False)

    traj_files = sorted(TRAJ.glob("cell_*.csv"))
    assert len(frames) == len(grid), f"cell count mismatch: {len(frames)} vs {len(grid)}"
    assert len(traj_files) == len(grid), f"traj count mismatch: {len(traj_files)} vs {len(grid)}"
    assert len(df) == len(grid) * SEEDS, f"run count mismatch: {len(df)} vs {len(grid)*SEEDS}"

    src_after = snapshot_sources()
    assert src_before == src_after, "model source files mutated during sweep"

    manifest = {
        "driver_version": DRIVER_VERSION,
        "cells_gross": len(grid),
        "cells_net": len(frames),
        "cells_new": done,
        "cells_skipped": skipped,
        "runs_gross": len(grid) * SEEDS,
        "runs_net_ok": int(df["c_ideal_fin"].notna().sum()),
        "wall_s": round(time.time() - t0, 1),
        "n_jobs": args.n_jobs,
        "sources_moved_during_sweep": False,
        "sources_sha256": src_after,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
