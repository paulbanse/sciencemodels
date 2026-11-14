"""Career-flows sweep B on the noisy landscape (figures-v1 era, ADR-0002 driver).

Brief: rerun the career-flow sweep B of `docs/career_flows_v1.md` (its declared
extension, deviation (b): "a noisy rerun is the natural extension") with
`noisyGaussian {prop_random: 5, std: 5, std_curiosity: 0.1}` instead of the
clean `multipleGaussians {std: 5, number_gaussians: 2}` bundle, everything else
identical, so every headline finding can be tested for landscape dependence.

Grid (noisy arm): agent_generation_time {0.05, 0.1, 0.2} x curiosity
{0.3, 0.5, 0.7}, vanishing 0.9, step_limit 200, seeds 0..19 -> 9 cells x 20
seeds = 180 runs.

Control arm (comparability bridge, 1 cell): the original mgstd5 cell
curiosity 0.5, g=0.1, same seeds, run through this driver and diffed
bit-exactly against `data/career_flows_v1/careerB_cur5_van9_mgstd5_gen0p1_h200.csv`.
If the model code moved since the T2 round, that control shows it, and every
noisy-vs-clean difference is then attributable to the landscape plus a named
delta instead of a silent code change.

Model-internal; NOT calibrated (owner ruling 2026-11-13).

Measurement definitions (`run_one`, `career_measures_from_agent_vars`) are
copied verbatim from `scripts/career_flows/career_flows_driver.py` so the noisy
numbers are comparable with the clean ones by construction; the duplication is
reported as a need in the document (ADR-0002 prefers one home for a measure).

Writes only data/career_flows_noisy_v1/. Checkpointed per cell (atomic
checkpoint, atomic CSVs with a row-count gate); n_jobs=3 (concurrent sweeps may
be live in this tree).

    /opt/hermes/.venv/bin/python scripts/career_flows_noisy/driver.py --sweeps both
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone

REPO = "/workspace/git/sciencemodels"
sys.path.insert(0, REPO)

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.simplefilter(action="ignore", category=FutureWarning)

OUT = os.path.join(REPO, "data", "career_flows_noisy_v1")

MODEL_TARGETS = [
    "Corr_prestige_ideal_merit",
    "Corr_prestige_local_merit",
    "top10pct_prestige_share",
    "explored_percentage",
    "percentage_knowledge_harvested",
    "mean_age",
    "mean_prestige",
]

LANDSCAPES = {
    "mgstd5": ("multipleGaussians", {"std": 5, "number_gaussians": 2}),
    "noisypr5": ("noisyGaussian", {"prop_random": 5, "std": 5, "std_curiosity": 0.1}),
}

FIXED = {
    "n_agents": 20,
    "epsilon": 0.1,
    "harvest": 0.02,
    "grid_size": 20,
    "use_distance": True,
    "new_questions": 0,
    "use_visibility_reward": True,
    "knowledge_normalization": "share",
    "prestige_accrual": "harvested",
}

SWEEP_NOISY = {
    "prefix": "noisyB",
    "curiosity": [0.3, 0.5, 0.7],
    "vanishing": [0.9],
    "landscapes": ["noisypr5"],
    "gen_time": [0.05, 0.1, 0.2],
    "step_limit": 200,
    "seeds": list(range(20)),
}
SWEEP_CONTROL = {
    "prefix": "control",
    "curiosity": [0.5],
    "vanishing": [0.9],
    "landscapes": ["mgstd5"],
    "gen_time": [0.1],
    "step_limit": 200,
    "seeds": list(range(20)),
}


def run_one(landscape_fn, generation_params, curiosity, vanishing, gen_time,
            step_limit, seed):
    """One seeded run (verbatim from scripts/career_flows/career_flows_driver.py)."""
    from sciencemodels import MyModel

    try:
        model = MyModel(
            n_agents=FIXED["n_agents"],
            initial_curiosity=curiosity,
            epsilon=FIXED["epsilon"],
            harvest=FIXED["harvest"],
            grid_size=FIXED["grid_size"],
            cell_init_function=landscape_fn,
            generation_params=generation_params,
            use_distance=FIXED["use_distance"],
            agent_generation_time=gen_time,
            new_questions=FIXED["new_questions"],
            agent_seed=seed,
            step_limit=step_limit,
            vanishing_factor_for_prestige=vanishing,
            use_visibility_reward=FIXED["use_visibility_reward"],
            knowledge_normalization=FIXED["knowledge_normalization"],
            prestige_accrual=FIXED["prestige_accrual"],
        )
        for _ in range(step_limit):
            model.step(True)
        mv = model.datacollector.get_model_vars_dataframe()
        out = mv[["Step"] + MODEL_TARGETS].copy()
        out.insert(0, "seed", seed)
        n_final = len(model._scientists)
        av = model.datacollector.get_agent_vars_dataframe()
        return "ok", out, n_final, av
    except Exception:
        return "fail", traceback.format_exc(), None, None


def career_measures_from_agent_vars(av, model_vars):
    """Per-step population/career measures (verbatim from the T2 driver)."""
    av = av.reset_index()
    step_col = "Step" if "Step" in av.columns else av.columns[0]
    id_col = None
    for cand in ("agent_id", "AgentID", "agent"):
        if cand in av.columns:
            id_col = cand
            break
    if id_col is None:
        id_col = [c for c in av.columns if c != step_col][0]
    entry = av.groupby(id_col)[step_col].min().rename("entry_step")
    av = av.merge(entry, left_on=id_col, right_index=True)
    av["is_newcomer"] = av["entry_step"] > 0
    g = av.groupby(step_col)
    pop = g.size().rename("n_agents")
    inc = av[~av.is_newcomer].groupby(step_col)["prestige"].mean().rename("incumbent_mean_prestige")
    new = av[av.is_newcomer].groupby(step_col)["prestige"].mean().rename("newcomer_mean_prestige")
    res = pd.concat([pop, inc, new], axis=1)
    res["newcomer_mean_prestige"] = res["newcomer_mean_prestige"].fillna(0.0)
    res["prestige_gap"] = res["newcomer_mean_prestige"] - res["incumbent_mean_prestige"]
    res["prestige_gap_ratio"] = np.where(
        res["incumbent_mean_prestige"] > 0,
        res["prestige_gap"] / res["incumbent_mean_prestige"], 0.0)
    harvested = model_vars.set_index("Step")["percentage_knowledge_harvested"]
    res["harvested_cum"] = harvested.reindex(res.index).fillna(0.0)
    res["harvest_flow"] = res["harvested_cum"].diff().fillna(0.0).clip(lower=0.0)
    res["percap_harvest_cum"] = res["harvested_cum"] / res["n_agents"]
    res["percap_harvest_flow"] = res["harvest_flow"] / res["n_agents"]
    return res.reset_index().rename(columns={step_col: "Step"})


def cell_tag(spec, cur, van, land, gen):
    return "{}_cur{}_van{}_{}_gen{}_h{}".format(
        spec["prefix"], int(round(cur * 10)), int(round(van * 10)), land,
        str(gen).replace(".", "p"), spec["step_limit"])


def _write_atomic(df, path, expected_rows):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    got = len(pd.read_csv(tmp))
    if got != expected_rows:
        raise RuntimeError(f"{path}: wrote {got} rows, expected {expected_rows}")
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", choices=["noisy", "control", "both"], default="both")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    t_start = time.time()
    cp_path = os.path.join(OUT, "checkpoint.json")
    log_path = os.path.join(OUT, "run_log.txt")
    checkpoint = {"schema": "career_flows_noisy_v1", "cells": {}}
    if os.path.isfile(cp_path):
        with open(cp_path) as f:
            checkpoint = json.load(f)

    def log(msg):
        line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    from sciencemodels import landscapes as L

    # design record + process snapshot before the heavy launch (grid constants
    # that never reach the CSVs, and who else was running in this tree)
    import platform

    design = {
        "task": "career-flows sweep B on the noisy landscape; model-internal; NOT calibrated",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "grids": {"noisy": {k: v for k, v in SWEEP_NOISY.items()},
                  "control": {k: v for k, v in SWEEP_CONTROL.items()}},
        "fixed_params": FIXED,
        "landscape_bundles": LANDSCAPES,
        "seed_semantics": "one agent_seed per run (0..19); MyModel also uses it as the "
                          "landscape seed, so landscape and behaviour seeds coincide, as in T2",
        "population_rule_under_entry": "21 + floor(g * t) (T2 Q1 init-time accumulator quirk)",
        "measurement_provenance": "run_one + career_measures_from_agent_vars copied "
                                  "verbatim from scripts/career_flows/career_flows_driver.py",
        "comparison_source": "data/career_flows_v1/careerB_*_mgstd5_h200*.csv (read-only)",
        "environment": {"python": platform.python_version(),
                        "numpy": np.__version__, "pandas": pd.__version__},
        "n_jobs": 3,
    }
    with open(os.path.join(OUT, "design.json"), "w") as f:
        json.dump(design, f, indent=1)
    try:
        ps = subprocess.run(["ps", "-eo", "pid,etime,pcpu,cmd"], capture_output=True,
                            text=True).stdout
        with open(os.path.join(OUT, "process_snapshot.txt"), "w") as f:
            f.write(ps)
    except Exception as exc:  # snapshot is evidence, never a blocker
        log(f"process snapshot failed: {exc}")

    # CANARY GATE (hard abort before any cell)
    can = subprocess.run(
        ["/opt/hermes/.venv/bin/python", "tests/test_smoke.py"],
        cwd=REPO, capture_output=True, text=True,
        env={**os.environ, "MPLBACKEND": "Agg"})
    if can.returncode != 0:
        print("CANARY FAILED --- aborting before any cell:\n" + can.stdout + can.stderr)
        sys.exit(1)
    log("canary gate OK: tests/test_smoke.py exit 0 (hard launch gate)")

    fn_map = {"mgstd5": L.multipleGaussians, "noisypr5": L.noisyGaussian}

    def run_sweep(spec):
        cells = [(cur, van, land, gen)
                 for land in spec["landscapes"]
                 for gen in spec["gen_time"]
                 for van in spec["vanishing"]
                 for cur in spec["curiosity"]]
        for cur, van, land, gen in cells:
            tag = cell_tag(spec, cur, van, land, gen)
            prev = checkpoint["cells"].get(tag, {}).get("status")
            if prev in ("ok", "ok_with_failures"):
                log(f"SKIP {tag} (already {prev})")
                continue
            t0 = time.time()
            gparams = dict(LANDSCAPES[land][1])
            results = Parallel(n_jobs=3, return_as="list")(
                delayed(run_one)(fn_map[land], dict(gparams), cur, van, gen,
                                 spec["step_limit"], s)
                for s in spec["seeds"])
            frames, career_frames, errors, n_finals = [], [], {}, []
            for s, (status, payload, n_ag, av) in zip(spec["seeds"], results):
                if status == "ok":
                    payload.insert(0, "cell_tag", tag)
                    payload.insert(1, "curiosity", cur)
                    payload.insert(2, "vanishing_factor_for_prestige", van)
                    payload.insert(3, "landscape", land)
                    payload.insert(4, "agent_generation_time", gen)
                    payload.insert(5, "step_limit", spec["step_limit"])
                    frames.append(payload)
                    n_finals.append(n_ag)
                    cm = career_measures_from_agent_vars(av, payload)
                    cm.insert(0, "seed", s)
                    career_frames.append(cm)
                else:
                    errors[str(s)] = payload
            expected = len(spec["seeds"]) * (spec["step_limit"] + 1)
            if frames:
                _write_atomic(pd.concat(frames, ignore_index=True),
                              os.path.join(OUT, f"{tag}.csv"), expected)
            if career_frames:
                _write_atomic(pd.concat(career_frames, ignore_index=True),
                              os.path.join(OUT, f"{tag}_career.csv"), expected)
            status = "ok" if not errors else ("ok_with_failures" if frames else "crashed")
            checkpoint["cells"][tag] = {
                "sweep": "noisy" if spec["prefix"] == "noisyB" else "control",
                "status": status,
                "params": {"curiosity": cur, "vanishing_factor_for_prestige": van,
                           "landscape": land, "agent_generation_time": gen,
                           "step_limit": spec["step_limit"]},
                "seeds_ok": len(frames), "seeds_failed": len(errors),
                "rows": sum(len(f) for f in frames),
                "n_scientists_final_min": min(n_finals) if n_finals else None,
                "n_scientists_final_max": max(n_finals) if n_finals else None,
                "errors": {k: v[-2000:] for k, v in errors.items()},
                "csv": f"{tag}.csv" if frames else None,
                "career_csv": f"{tag}_career.csv" if career_frames else None,
                "wall_s": round(time.time() - t0, 2),
                "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            with open(cp_path + ".tmp", "w") as f:
                json.dump(checkpoint, f, indent=1)
            os.replace(cp_path + ".tmp", cp_path)
            e = checkpoint["cells"][tag]
            log(f"CELL {tag} -> {status} seeds_ok={e['seeds_ok']} "
                f"rows={e['rows']} agents_final={e['n_scientists_final_min']}-"
                f"{e['n_scientists_final_max']} wall={e['wall_s']}s"
                + (f" ERRORS={list(errors)}" if errors else ""))

    sweeps = {"noisy": SWEEP_NOISY, "control": SWEEP_CONTROL}
    todo = ["noisy", "control"] if args.sweeps == "both" else [args.sweeps]
    for name in todo:
        spec = sweeps[name]
        n = len(spec["curiosity"]) * len(spec["landscapes"]) * len(spec["gen_time"])
        log(f"sweep {name} start: {n} cells x {len(spec['seeds'])} seeds, "
            f"h{spec['step_limit']}, n_jobs=3")
        run_sweep(spec)

    log(f"driver done: {len(checkpoint['cells'])} cells checkpointed, "
        f"total_wall={time.time() - t_start:.1f}s")
    print(json.dumps({t: c["status"] for t, c in checkpoint["cells"].items()}, indent=1))


if __name__ == "__main__":
    main()
