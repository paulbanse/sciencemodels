"""Path A sensitivity sweep driver — sciencemodels, sensitivity_v1.

Grid: initial_curiosity x vanishing_factor_for_prestige x landscape x agent_generation_time
      = 3 x 3 x 2 x 2 = 36 cells, 20 seeds per cell, step_limit=200, n_agents=20, grid 20x20.

Design notes (boundaries respected: this script lives in /tmp, writes only
data/sensitivity_v1/* inside the repo, edits no code):
- Seeds: one knob (agent_seed) drives agent RNG; the model injects the same
  seed into generation_params ("seed": self._seed, model.py:139-141), so
  landscape and agent RNG vary together. 20 seeds = 20 values of that knob.
- Landscapes mirror bulk_runner.py bundles exactly:
    mgstd5   = multipleGaussians {std:5, number_gaussians:2}   (bulk setup 2/3)
    noisypr5 = noisyGaussian   {prop_random:5, std:5, std_curiosity:0.1} (bulk setup 6/7)
- Fixed params pinned as in scripts/bulk_runner.py (self-describing runs):
  epsilon 0.1, harvest 0.02, use_distance True, new_questions 0,
  use_visibility_reward True, knowledge_normalization "share",
  prestige_accrual "harvested", agent_generation_function = model default
  (identity: curiosity == initial_curiosity).
- Per-run: drive MyModel.step() 200x, read the five target series from
  model.datacollector.get_model_vars_dataframe() (per-step collection).
- Checkpoint per completed cell (checkpoint.json, atomic replace) -> restart-safe.
- A failing seed is recorded, never dropped silently; a cell with >=1 successful
  seed still yields a mean trajectory over its successful seeds (n_seeds discloses).
"""
import hashlib
import json
import os
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
OUT = os.path.join(REPO, "data", "sensitivity_v1")
os.makedirs(OUT, exist_ok=True)

TARGETS = [
    "Corr_prestige_ideal_merit",
    "Corr_prestige_local_merit",
    "top10pct_prestige_share",
    "explored_percentage",
    "percentage_knowledge_harvested",
]

CURIOSITY = [0.3, 0.5, 0.7]
VANISHING = [0.5, 0.9, 1.0]
GEN_TIME = [0, 0.1]
SEEDS = list(range(20))
STEPS = 200

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
    "step_limit": 200,
}


def tag_of(cur, van, land, gen):
    return "cell_cur{:02d}_van{:02d}_{}_gen{:02d}".format(
        int(round(cur * 10)), int(round(van * 10)), land, int(round(gen * 10))
    )


def run_one(landscape_fn, generation_params, curiosity, vanishing, gen_time, seed):
    """One seeded run. Returns (status, payload) — never raises."""
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
            step_limit=STEPS,
            vanishing_factor_for_prestige=vanishing,
            use_visibility_reward=FIXED["use_visibility_reward"],
            knowledge_normalization=FIXED["knowledge_normalization"],
            prestige_accrual=FIXED["prestige_accrual"],
        )
        for _ in range(STEPS):
            model.step(True)
        df = model.datacollector.get_model_vars_dataframe()
        out = df[["Step"] + TARGETS].copy()
        out.insert(0, "seed", seed)
        return "ok", out, len(model._scientists)
    except Exception as exc:  # noqa: BLE001 — reported, not dropped
        return "fail", traceback.format_exc(), None


def main():
    t_start = time.time()
    cp_path = os.path.join(OUT, "checkpoint.json")
    log_path = os.path.join(OUT, "run_log.txt")

    checkpoint = {"schema": "sensitivity_v1", "cells": {}}
    if os.path.isfile(cp_path):
        with open(cp_path) as f:
            checkpoint = json.load(f)

    def log(msg):
        line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    cells = []
    for land in LANDSCAPES:
        for gen in GEN_TIME:
            for van in VANISHING:
                for cur in CURIOSITY:
                    cells.append((cur, van, land, gen))

    from sciencemodels import landscapes as L

    fn_map = {
        "mgstd5": L.multipleGaussians,
        "noisypr5": L.noisyGaussian,
    }

    gross = len(cells)
    done_at_start = sum(
        1
        for c in cells
        if checkpoint["cells"].get(tag_of(*c), {}).get("status") in ("ok", "ok_with_failures", "crashed")
    )
    log(f"sweep start: {gross} cells gross, {done_at_start} already checkpointed, "
        f"seeds/cell={len(SEEDS)}, steps={STEPS}, joblib n_jobs=8")

    for cur, van, land, gen in cells:
        tag = tag_of(cur, van, land, gen)
        prev = checkpoint["cells"].get(tag, {}).get("status")
        if prev in ("ok", "ok_with_failures", "crashed"):
            log(f"SKIP {tag} (already {prev})")
            continue
        t0 = time.time()
        lname, gparams = LANDSCAPES[land]
        results = Parallel(n_jobs=8, return_as="list")(
            delayed(run_one)(fn_map[land], dict(gparams), cur, van, gen, s)
            for s in SEEDS
        )
        frames, errors, n_agents_final = [], {}, []
        for s, (status, payload, n_ag) in zip(SEEDS, results):
            if status == "ok":
                payload.insert(0, "cell_tag", tag)
                payload.insert(1, "curiosity", cur)
                payload.insert(2, "vanishing_factor_for_prestige", van)
                payload.insert(3, "landscape", land)
                payload.insert(4, "agent_generation_time", gen)
                frames.append(payload)
                n_agents_final.append(n_ag)
            else:
                errors[str(s)] = payload
        rows = sum(len(f) for f in frames)
        if frames:
            cell_df = pd.concat(frames, ignore_index=True)
            csv_path = os.path.join(OUT, f"{tag}.csv")
            cell_df.to_csv(csv_path, index=False)
        status = (
            "ok" if not errors else ("ok_with_failures" if frames else "crashed")
        )
        entry = {
            "status": status,
            "params": {
                "initial_curiosity": cur,
                "vanishing_factor_for_prestige": van,
                "landscape": land,
                "agent_generation_time": gen,
            },
            "seeds_ok": len(frames),
            "seeds_failed": len(errors),
            "rows": rows,
            "n_scientists_final_min": min(n_agents_final) if n_agents_final else None,
            "n_scientists_final_max": max(n_agents_final) if n_agents_final else None,
            "errors": {k: v[-2000:] for k, v in errors.items()},
            "csv": os.path.basename(f"{tag}.csv") if frames else None,
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

    n_ok = sum(1 for c in checkpoint["cells"].values() if c["status"] == "ok")
    n_partial = sum(1 for c in checkpoint["cells"].values() if c["status"] == "ok_with_failures")
    n_crashed = sum(1 for c in checkpoint["cells"].values() if c["status"] == "crashed")
    log(f"sweep done: gross={gross} ok={n_ok} ok_with_failures={n_partial} crashed={n_crashed} "
        f"total_wall={time.time()-t_start:.1f}s")
    print(json.dumps({"gross": gross, "ok": n_ok, "ok_with_failures": n_partial,
                      "crashed": n_crashed}, indent=1))


if __name__ == "__main__":
    main()
