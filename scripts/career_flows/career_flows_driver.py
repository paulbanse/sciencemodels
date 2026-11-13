"""Career-flows driver (T2): oscillation confirmation at horizon 400 + career-flow sweep.

Sweep A (confirmation): the T1 entry cells re-run at a longer horizon and more
seeds --- curiosity {0.3, 0.5, 0.7} x vanishing 0.9 x {mgstd5, noisypr5} x
agent_generation_time 0.1, step_limit 400, 40 seeds. (The brief said "4
representative cells" but enumerated the 3 x 2 crossing; the full crossing is
run and the deviation is disclosed.)

Sweep B (career flows): agent_generation_time {0.05, 0.1, 0.2} x curiosity
{0.3, 0.5, 0.7}, vanishing 0.9, landscape mgstd5 (declared pin: canonical bulk
bundle; legibility was T1's axis, demographics are this task's question),
step_limit 200, 20 seeds.

Writes only data/career_flows_v1/. Checkpointed per cell (restart-safe); a
failing seed is recorded, never dropped silently. n_jobs=3 (concurrent-sweep
hygiene: a foreign sweep is live in this tree).
"""
import json
import os
import subprocess
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone

sys.path.insert(0, "/workspace/git/sciencemodels")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.simplefilter(action="ignore", category=FutureWarning)

REPO = "/workspace/git/sciencemodels"
OUT = os.path.join(REPO, "data", "career_flows_v1")
os.makedirs(OUT, exist_ok=True)

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

SWEEP_A = {
    "prefix": "confirmA",
    "curiosity": [0.3, 0.5, 0.7],
    "vanishing": [0.9],
    "landscapes": ["mgstd5", "noisypr5"],
    "gen_time": [0.1],
    "step_limit": 400,
    "seeds": list(range(40)),
}
SWEEP_B = {
    "prefix": "careerB",
    "curiosity": [0.3, 0.5, 0.7],
    "vanishing": [0.9],
    "landscapes": ["mgstd5"],
    "gen_time": [0.05, 0.1, 0.2],
    "step_limit": 200,
    "seeds": list(range(20)),
}


def run_one(landscape_fn, generation_params, curiosity, vanishing, gen_time,
            step_limit, seed, need_agent_vars):
    """One seeded run. Returns (status, payload, n_scientists_final)."""
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
        av = None
        if need_agent_vars:
            av = model.datacollector.get_agent_vars_dataframe()
        return "ok", out, n_final, av
    except Exception:
        return "fail", traceback.format_exc(), None, None


def career_measures_from_agent_vars(av, model_vars):
    """Derive population/career measures from the agent-vars longitudinal frame.

    av: mesa agent-vars dataframe (index Step + agent id, or columns after
    reset_index). Returns a per-Step frame: n_agents, incumbent_mean_prestige,
    newcomer_mean_prestige, prestige_gap, prestige_gap_ratio, plus harvest
    flows from model_vars.
    """
    av = av.reset_index()
    # mesa 3.2: index levels are (Step, agent_id) - names may vary; find them
    step_col = "Step" if "Step" in av.columns else av.columns[0]
    id_col = None
    for cand in ("agent_id", "AgentID", "agent"):
        if cand in av.columns:
            id_col = cand
            break
    if id_col is None:
        # level name of the non-Step index level
        id_col = [c for c in av.columns if c != step_col][0]
        # fallthrough: if reset_index produced a level named 'Level_1' etc.
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


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", choices=["A", "B", "both"], default="both")
    args = ap.parse_args()

    t_start = time.time()
    cp_path = os.path.join(OUT, "checkpoint.json")
    log_path = os.path.join(OUT, "run_log.txt")
    checkpoint = {"schema": "career_flows_v1", "cells": {}}
    if os.path.isfile(cp_path):
        with open(cp_path) as f:
            checkpoint = json.load(f)

    def log(msg):
        line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    from sciencemodels import landscapes as L

    # CANARY GATE: hard abort before any cell if the seeded smoke test fails.
    # (Added after the T2 sweep completed, as a launch gate for regenerations;
    #  the T2 sweep itself was canary-gated in-session pre-launch --- disclosed
    #  in the document. Launch gate only: no dynamics code.)
    can = subprocess.run(
        ["/opt/hermes/.venv/bin/python", "tests/test_smoke.py"],
        cwd=REPO, capture_output=True, text=True,
        env={**os.environ, "MPLBACKEND": "Agg"})
    if can.returncode != 0:
        print("CANARY FAILED --- aborting before any cell:\n" + can.stdout + can.stderr)
        sys.exit(1)
    log("canary gate OK: tests/test_smoke.py exit 0 (hard launch gate)")

    fn_map = {"mgstd5": L.multipleGaussians, "noisypr5": L.noisyGaussian}

    def run_sweep(spec, sweep_name, need_agent_vars):
        cells = []
        for land in spec["landscapes"]:
            for gen in spec["gen_time"]:
                for van in spec["vanishing"]:
                    for cur in spec["curiosity"]:
                        cells.append((cur, van, land, gen))
        for cur, van, land, gen in cells:
            lname, gparams = LANDSCAPES[land]
            tag = "{}_cur{}_van{}_{}_gen{}_h{}".format(
                spec["prefix"], int(round(cur * 10)), int(round(van * 10)),
                land, str(gen).replace(".", "p"), spec["step_limit"])
            prev = checkpoint["cells"].get(tag, {}).get("status")
            if prev in ("ok", "ok_with_failures", "crashed"):
                log(f"SKIP {tag} (already {prev})")
                continue
            t0 = time.time()
            results = Parallel(n_jobs=3, return_as="list")(
                delayed(run_one)(fn_map[land], dict(gparams), cur, van, gen,
                                 spec["step_limit"], s, need_agent_vars)
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
                    if need_agent_vars and av is not None:
                        cm = career_measures_from_agent_vars(av, payload)
                        cm.insert(0, "seed", s)
                        career_frames.append(cm)
                else:
                    errors[str(s)] = payload
            rows = sum(len(f) for f in frames)
            if frames:
                pd.concat(frames, ignore_index=True).to_csv(
                    os.path.join(OUT, f"{tag}.csv"), index=False)
            if career_frames:
                pd.concat(career_frames, ignore_index=True).to_csv(
                    os.path.join(OUT, f"{tag}_career.csv"), index=False)
            status = "ok" if not errors else ("ok_with_failures" if frames else "crashed")
            entry = {
                "sweep": sweep_name, "status": status,
                "params": {"curiosity": cur, "vanishing_factor_for_prestige": van,
                           "landscape": land, "agent_generation_time": gen,
                           "step_limit": spec["step_limit"]},
                "seeds_ok": len(frames), "seeds_failed": len(errors), "rows": rows,
                "n_scientists_final_min": min(n_finals) if n_finals else None,
                "n_scientists_final_max": max(n_finals) if n_finals else None,
                "errors": {k: v[-2000:] for k, v in errors.items()},
                "csv": f"{tag}.csv" if frames else None,
                "career_csv": f"{tag}_career.csv" if career_frames else None,
                "wall_s": round(time.time() - t0, 2),
                "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            checkpoint["cells"][tag] = entry
            with open(cp_path + ".tmp", "w") as f:
                json.dump(checkpoint, f, indent=1)
            os.replace(cp_path + ".tmp", cp_path)
            log(f"CELL {tag} -> {status} seeds_ok={entry['seeds_ok']} "
                f"seeds_failed={entry['seeds_failed']} rows={rows} "
                f"agents_final={entry['n_scientists_final_min']}-{entry['n_scientists_final_max']} "
                f"wall={entry['wall_s']}s"
                + (f" ERRORS={list(errors)}" if errors else ""))

    if args.sweep in ("A", "both"):
        log(f"sweep A start (confirmation, h400, 40 seeds, n_jobs=3): "
            f"{len(SWEEP_A['curiosity'])*len(SWEEP_A['landscapes'])} cells x "
            f"{len(SWEEP_A['seeds'])} seeds")
        run_sweep(SWEEP_A, "A", need_agent_vars=False)
    if args.sweep in ("B", "both"):
        log(f"sweep B start (career flows, h200, 20 seeds, n_jobs=3): "
            f"{len(SWEEP_B['gen_time'])*len(SWEEP_B['curiosity'])} cells x "
            f"{len(SWEEP_B['seeds'])} seeds")
        run_sweep(SWEEP_B, "B", need_agent_vars=True)

    cells = checkpoint["cells"]
    log(f"driver done: {len(cells)} cells checkpointed, total_wall={time.time()-t_start:.1f}s")
    print(json.dumps({k: v["status"] for k, v in cells.items()}, indent=1))


if __name__ == "__main__":
    main()
