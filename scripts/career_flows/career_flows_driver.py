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

from sciencemodels.sweeps import (
    CAREER_FIXED,
    CAREER_LANDSCAPES,
    MODEL_TARGETS,
    career_measures_from_agent_vars,
    run_one,
)

# Arm-local aliases: this driver's public names are unchanged, so its sibling
# scripts (probe.py, audit_code_delta.py, remeasure_f5.py) keep importing the
# same symbols from here. The one implementation lives in sciencemodels.sweeps
# (ADR-0002: one home per measure).
FIXED = CAREER_FIXED
LANDSCAPES = CAREER_LANDSCAPES


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
