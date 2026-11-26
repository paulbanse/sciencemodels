"""Attraction-trias 2D sweep on the ADR-0003 clock (task/attraction-trias-heatmap).

Sweeps the ``prestige_weight_in_visibility`` knob (attraction trias: pure peer
proximity w = 0 -> full past-prestige-weighted proximity w = 1) against the
visibility axis xi = 1 - c, on the two canonical tiers:

    M0: g = 0,    exit_mode = "none"       (closed cohort control)
    M1: g = 0.25, exit_mode = "fixed_span" (canonical ADR-0003 turnover)

Design (contract-first; this block is the run specification):

    xi = 1 - c in {0.0, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}   (9 levels)
    w  in {0.0, 0.25, 0.5, 0.75, 1.0}                             (5 levels)
    tiers {M0, M1}  ->  90 cells x 20 seeds = 1,800 runs PER LANDSCAPE

    ADR-0003 clock: gamma = vanishing_factor_for_prestige = 0.5**(1/10)
    (~0.933, 5-year half-life), H = step_limit = 200 steps (100 academic
    years), N = n_agents = 20, grid 20x20, g = 0.25 for M1, tau_max = 80.
    Seeds: agent_seed 0..19, which drives agent AND landscape RNG jointly
    (the house seed semantics).

Landscapes (``--landscape``, F3b landscape check):

    mgstd5   multipleGaussians {std: 5, number_gaussians: 2}  ("MultG2")
    mg6      multipleGaussians {std: 5, number_gaussians: 6}  ("MultG6")
    noisypr5 noisyGaussian {prop_random: 5, std: 5, std_curiosity: 0.1}
             ("Noisy2"; the modern noisypr5 bundle, std_curiosity 0.1)

Outputs per landscape under data/attraction_heatmap[_TAG]_adr0003/ (the
mgstd5 round keeps its untagged directory for checkpoint continuity).

Headline metrics per run (self-describing full report row is written):
    percentage_knowledge_harvested        (Panel A)
    Mean_of_Corr_prestige_ideal_merit     = time-mean Corr(P_out, M*) (Panel B)

Checkpointing (ADR-0001): one CSV per completed cell
(``cells/cell_xi{tag}_w{tag}_{tier}.csv``); completed cells are skipped on
restart and a timeout loses at most one cell.
In-run canaries: (i) a two-run determinism probe, (ii) a drift probe that
re-runs seed 0 of the anchor cell and diffs it against the stored checkpoint
row exactly (terminates loudly instead of writing bad outputs), (iii) the
w-extreme identity probes (w = 0.0 == use_visibility_reward=False,
w = 1.0 == use_visibility_reward=True) on a short config.

NOT CALIBRATED: model-internal only (owner ruling 2026-11-13). The ADR-0003
values are conceptual scale anchors, not empirical fits.

Run:
    /opt/hermes/.venv/bin/python scripts/attraction_heatmap/run_attraction_heatmap.py \
        --landscape {mgstd5,mg6,noisypr5}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from sciencemodels import MyModel  # noqa: E402
from sciencemodels.landscapes import multipleGaussians, noisyGaussian  # noqa: E402
from sciencemodels.model import (  # noqa: E402
    CANONICAL_AGENT_GENERATION_TIME,
    CANONICAL_TAU_MAX,
    CANONICAL_VANISHING_FACTOR,
)

XI_LEVELS = (0.0, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
W_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
SEEDS = tuple(range(20))
STEP_LIMIT = 200  # H = 200 steps = 100 academic years (ADR-0003)
N_JOBS = 3        # declared deliberately (ADR-0001 process hygiene)

#: house landscape bundles (mgstd5 = MultG2, mg6 = MultG6, noisypr5 = Noisy2)
LANDSCAPES = {
    "mgstd5": {
        "cell_init_function": multipleGaussians,
        "generation_params": {"std": 5, "number_gaussians": 2},
    },
    "mg6": {
        "cell_init_function": multipleGaussians,
        "generation_params": {"std": 5, "number_gaussians": 6},
    },
    "noisypr5": {
        "cell_init_function": noisyGaussian,
        "generation_params": {
            "prop_random": 5, "std": 5, "std_curiosity": 0.1,
        },
    },
}
DISPLAY_NAME = {"mgstd5": "mgstd5", "mg6": "MultG6", "noisypr5": "Noisy2"}

TIERS = {
    "m0": {"agent_generation_time": 0.0, "exit_mode": "none"},
    "m1": {
        "agent_generation_time": float(CANONICAL_AGENT_GENERATION_TIME),
        "exit_mode": "fixed_span",
        "tau_max": CANONICAL_TAU_MAX,
    },
}

BASE_PARAMS = {
    "n_agents": 20,
    "epsilon": 0.1,
    "harvest": 0.02,
    "grid_size": 20,
    "use_distance": True,
    "new_questions": 0,
    "agent_generation_function": lambda cur, _rng, _params: cur,
    "vanishing_factor_for_prestige": float(CANONICAL_VANISHING_FACTOR),
    "use_visibility_reward": True,
    "knowledge_normalization": "share",
    "prestige_accrual": "harvested",
    "entry_prestige_stock": "prestige",
    "step_limit": STEP_LIMIT,
}


def _xi_tag(xi: float) -> str:
    return f"{int(round(xi * 100)):03d}"


def _w_tag(w: float) -> str:
    return f"{int(round(w * 100)):03d}"


def out_dir(landscape: str) -> str:
    """Output root per landscape; mgstd5 keeps the F3 directory name."""
    tag = "" if landscape == "mgstd5" else f"_{landscape}"
    return os.path.join("data", f"attraction_heatmap{tag}_adr0003")


def cell_path(tier: str, xi: float, w: float, landscape: str) -> str:
    return os.path.join(
        out_dir(landscape), "cells",
        f"cell_xi{_xi_tag(xi)}_w{_w_tag(w)}_{tier}.csv"
    )


def run_one(tier: str, xi: float, w: float, seed: int,
            landscape: str = "mgstd5",
            step_limit: int = STEP_LIMIT) -> dict:
    """One seeded run -> one self-describing report row (+ sweep tags)."""
    params = dict(
        BASE_PARAMS,
        initial_curiosity=np.float64(1.0 - xi),
        agent_seed=seed,
        step_limit=step_limit,
        prestige_weight_in_visibility=w,
        # landscape bundle; the model injects agent_seed as the landscape seed
        **LANDSCAPES[landscape],
        **TIERS[tier],
    )
    model = MyModel(**params)
    row = model.run_mode_for_bulk()
    return {
        "tier": tier, "xi": float(xi), "w": float(w),
        "curiosity": float(1.0 - xi), "landscape": landscape,
        **row,
    }


def run_cell(tier: str, xi: float, w: float,
             landscape: str = "mgstd5") -> pd.DataFrame:
    rows = [run_one(tier, xi, w, seed, landscape) for seed in SEEDS]
    df = pd.DataFrame(rows)
    assert len(df) == len(SEEDS), f"cell {tier}/xi{xi}/w{w}: {len(df)} rows"
    return df


def cell_is_complete(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    ok = len(df) == len(SEEDS) and sorted(df["agent_seed"].tolist()) == list(
        SEEDS
    )
    if not ok:
        print(f"INCOMPLETE checkpoint {path}: {len(df)} rows -> re-running")
    return ok


def _sig_equal(a, b) -> bool:
    """Exact comparison across a CSV round-trip.

    Numeric values compare as floats (31 == 31.0 exactly), NaN == NaN counts
    as equal (gap_recent is NaN on runs without recent arrivals); everything
    else compares as str. A plain repr/str signature false-positives on
    integer milestone columns (e.g. weighted_90_step = 80 in memory, 80.0
    after pandas reads the column that also holds the 250.0 default).
    """
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)
    if fa != fa and fb != fb:  # both NaN (float('nan') != float('nan'))
        return True
    return fa == fb


def canaries(landscape: str) -> dict:
    """In-run self-checks before any new variants are produced."""
    out = {}

    # (i) determinism: the same seeded run twice must be bit-identical
    r1 = run_one("m0", 0.0, 0.0, 0, landscape, step_limit=30)
    r2 = run_one("m0", 0.0, 0.0, 0, landscape, step_limit=30)
    diffs = [k for k in r1 if not _sig_equal(r1[k], r2[k])]
    assert not diffs, f"determinism canary FAILED, differing keys: {diffs}"
    out["determinism"] = "OK"

    # (ii) drift probe: re-run seed 0 of the anchor cell and diff against the
    #      stored checkpoint row exactly (loud failure instead of bad outputs).
    #      _sig_equal is the comparison signature: the stored row round-trips
    #      through CSV, where dtypes differ from the in-memory ones (int vs
    #      float64) while the values are identical. Keys absent from older
    #      checkpoints (e.g. the landscape tag) are skipped.
    anchor = cell_path("m0", 0.0, 0.0, landscape)
    if cell_is_complete(anchor):
        stored = pd.read_csv(anchor)
        ref = stored[stored["agent_seed"] == 0].iloc[0]
        now = run_one("m0", 0.0, 0.0, 0, landscape)
        drift = [
            k for k in now
            if k in ref.index and not _sig_equal(now[k], ref[k])
        ]
        assert not drift, (
            f"DRIFT canary FAILED against {anchor}: {drift} - code changed "
            "under a stored cell; refusing to write outputs"
        )
        out["drift_vs_checkpoint"] = "OK"
    else:
        out["drift_vs_checkpoint"] = "skipped (no checkpoint yet)"

    # (iii) w-extreme identity on a short config (mirrors smoke canary 14)
    def _mini(**kw):
        m = MyModel(**dict(
            BASE_PARAMS, initial_curiosity=np.float64(0.5), agent_seed=7,
            step_limit=30, **LANDSCAPES[landscape], **kw))
        return m.run_mode_for_bulk(), m

    r_flag_off, m_flag_off = _mini(use_visibility_reward=False)
    r_w0, m_w0 = _mini(prestige_weight_in_visibility=0.0)
    r_flag_on, m_flag_on = _mini()
    r_w1, m_w1 = _mini(prestige_weight_in_visibility=1.0)
    skip = {"use_visibility_reward", "prestige_weight_in_visibility"}
    dyn = lambda r: sorted((k, repr(v)) for k, v in r.items()
                           if k not in skip)  # noqa: E731
    assert dyn(r_flag_off) == dyn(r_w0), "w=0.0 != use_visibility_reward=False"
    assert dyn(r_flag_on) == dyn(r_w1), "w=1.0 != use_visibility_reward=True"
    assert m_flag_off.datacollector.get_model_vars_dataframe().equals(
        m_w0.datacollector.get_model_vars_dataframe()), "w=0.0 trajectory moved"
    assert m_flag_on.datacollector.get_model_vars_dataframe().equals(
        m_w1.datacollector.get_model_vars_dataframe()), "w=1.0 trajectory moved"
    out["w_extremes"] = "OK"
    return out


def main(landscape: str = "mgstd5") -> None:
    assert landscape in LANDSCAPES, f"unknown landscape {landscape!r}"
    cell_dir = os.path.join(out_dir(landscape), "cells")
    manifest_path = os.path.join(out_dir(landscape), "manifest.json")
    os.makedirs(cell_dir, exist_ok=True)
    t0 = time.time()
    canary_report = canaries(landscape)
    print(f"canaries ({landscape}):", canary_report)

    jobs = []
    for tier in TIERS:
        for xi in XI_LEVELS:
            for w in W_LEVELS:
                path = cell_path(tier, xi, w, landscape)
                if cell_is_complete(path):
                    continue
                jobs.append((tier, xi, w, path))
    gross = len(TIERS) * len(XI_LEVELS) * len(W_LEVELS)
    print(
        f"grid ({landscape}): {gross} cells x {len(SEEDS)} seeds = "
        f"{gross * len(SEEDS)} runs; {gross - len(jobs)} cells already "
        f"checkpointed, {len(jobs)} to run"
    )

    done = 0
    with ProcessPoolExecutor(max_workers=N_JOBS) as pool:
        futs = {
            pool.submit(run_cell, tier, xi, w, landscape): (tier, xi, w, path)
            for tier, xi, w, path in jobs
        }
        for fut in as_completed(futs):
            tier, xi, w, path = futs[fut]
            df = fut.result()
            tmp = path + ".tmp"
            df.to_csv(tmp, index=False)
            os.replace(tmp, path)  # atomic: a crash leaves no partial cell
            done += 1
            print(f"[{done}/{len(jobs)}] wrote {path} ({len(df)} runs)")

    ran = sum(
        1 for tier in TIERS for xi in XI_LEVELS for w in W_LEVELS
        if cell_is_complete(cell_path(tier, xi, w, landscape))
    )
    if not jobs and os.path.exists(manifest_path):
        # idempotent re-run over a complete archive: keep the recorded
        # manifest (and with it the writing round's provenance) untouched
        print(f"grid complete, manifest untouched -> {manifest_path}")
    else:
        manifest = {
            "schema": "attraction_heatmap_adr0003_v2",
            "task_id": "attraction-trias-landscapes",
            "git_head": os.popen("git rev-parse HEAD").read().strip(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "design": {
                "xi_levels": list(XI_LEVELS),
                "w_levels": list(W_LEVELS),
                "tiers": TIERS,
                "seeds": list(SEEDS),
                "step_limit": STEP_LIMIT,
                "grid_size": 20,
                "n_agents": 20,
                "landscape": landscape,
                "landscape_display": DISPLAY_NAME[landscape],
                "landscape_bundle": repr(LANDSCAPES[landscape]),
                "vanishing_factor_for_prestige":
                    float(CANONICAL_VANISHING_FACTOR),
                "tau_max": CANONICAL_TAU_MAX,
                "metrics": {
                    "panel_a": "percentage_knowledge_harvested",
                    "panel_b": ("Mean_of_Corr_prestige_ideal_merit"
                                " (= time-mean Corr(P_out, M*))"),
                },
            },
            "canaries": canary_report,
            "gross_cells": gross,
            "net_cells_complete": ran,
            "gross_runs": gross * len(SEEDS),
            "net_runs": ran * len(SEEDS),
            "n_jobs": N_JOBS,
            "wall_s": round(time.time() - t0, 1),
        }
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=1)
        print(f"net cells complete: {ran}/{gross}; manifest -> {manifest_path}")
    assert ran == gross, f"grid incomplete: {ran}/{gross} cells"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--landscape", choices=sorted(LANDSCAPES),
                    default="mgstd5",
                    help="landscape bundle to sweep (default: mgstd5)")
    args = ap.parse_args()
    main(args.landscape)
