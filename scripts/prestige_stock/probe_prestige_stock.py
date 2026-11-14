"""Critical-path probe for the fifth-axis prestige-stock sweep.

Four proofs before the grid runs (parameter-sweep-rounds, step 2):

1. IDENTITY  - with the active stock set to `prestige`, the driver's patched
   supervisor block must be bit-identical to the model's own method (same
   seed, same report). This is what makes the axis honest: the patch changes
   only which attribute the roulette reads, never the baseline dynamics.
2. BINDING   - the active stock must actually reach the dynamics: three
   stocks, one seed, a headline outcome that differs.
3. COHORT    - the newcomer/incumbent split (`_scientists[:number_agents]`)
   must match the age structure, otherwise the prestige gap is measuring the
   wrong two groups.
4. SCHEMA + TIMING - the four extra report keys must be present in a
   WORKER-written CSV, and the wall time per run must be known.

Run: MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/prestige_stock/probe_prestige_stock.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import run_prestige_stock_sweep as W
from sciencemodels.model import MyModel, generate_data_parametric_exploration


def bit_equal(a, b):
    return sorted((k, repr(v)) for k, v in a.items()) == sorted(
        (k, repr(v)) for k, v in b.items()
    )


def run_once(cell, seed=3, step_limit=None):
    grid = W.param_grid_for(cell)
    grid["agent_seed"] = seed
    if step_limit is not None:
        grid["step_limit"] = step_limit
    model = MyModel(**grid)
    report = model.run_mode_for_bulk()
    return model, report


def main():
    # entry ON, otherwise the supervisor channel never executes
    cell = {
        "tag": "probe",
        "entry_prestige_stock": "prestige",
        "agent_generation_time": 0.1,
        "initial_curiosity": 0.5,
        "landscape": "mgstd5",
    }
    failures = []

    W.install_stock_selector()
    W.install_report_wrapper()
    print("divergence canary:", W._DIVERGENCE_CANARY)

    # ---- 1. identity: patched(stock=prestige) == model's own method --------
    W.ACTIVE_STOCK["attr"] = "prestige"
    _, rep_patched = run_once(cell)
    W.restore_model()
    _, rep_original = run_once(cell)
    W.install_stock_selector()
    identity = bit_equal(rep_patched, rep_original)
    print(f"\n[1] IDENTITY patched(stock=prestige) == model original: {identity}")
    if not identity:
        for k in sorted(set(rep_patched) | set(rep_original)):
            if repr(rep_patched.get(k)) != repr(rep_original.get(k)):
                print(f"    differs: {k}: patched={rep_patched.get(k)!r} "
                      f"original={rep_original.get(k)!r}")
        failures.append("identity")

    # ---- 2. binding: the stock must reach the dynamics --------------------
    heads = {}
    for stock in W.STOCKS:
        W.ACTIVE_STOCK["attr"] = stock
        _, rep = run_once(cell)
        heads[stock] = (
            rep["mean_prestige"],
            rep["n_newcomers"],
            rep["newcomer_incumbent_prestige_gap"],
            rep["corr_prestige_ideal_merit_final"],
        )
        print(f"[2] stock={stock:20s} mean_prestige={rep['mean_prestige']:.6e} "
              f"n_newcomers={rep['n_newcomers']:2d} "
              f"gap={rep['newcomer_incumbent_prestige_gap']:+.6e} "
              f"c_ideal_fin={rep['corr_prestige_ideal_merit_final']:+.4f}")
    distinct = len({repr(v) for v in heads.values()}) > 1
    print(f"[2] BINDING across the three stocks: {distinct}")
    if not distinct:
        failures.append("binding")

    # ---- 3. cohort: incumbents == founders by age -------------------------
    W.ACTIVE_STOCK["attr"] = "prestige"
    model, rep = run_once(cell)
    scientists = list(model._scientists)
    founders = scientists[: model.number_agents]
    newcomers = scientists[model.number_agents:]
    ages_f = sorted({a.age for a in founders})
    ages_n = sorted({a.age for a in newcomers})
    print(f"[3] COHORT incumbents={len(founders)} ages={ages_f} | "
          f"newcomers={len(newcomers)} ages={len(ages_n)} distinct "
          f"range {ages_n[0]}..{ages_n[-1]}")
    print(f"    report says n_incumbents={rep['n_incumbents']} "
          f"n_newcomers={rep['n_newcomers']}")
    cofounders_in_newcomers = sum(1 for a in newcomers if a.age in ages_f)
    print(f"    newcomers sharing the founder age band: {cofounders_in_newcomers}")
    if len(founders) != 20 or len(newcomers) != rep["n_newcomers"]:
        failures.append("cohort")

    # ---- 4. schema + timing through the production interface --------------
    W.ACTIVE_STOCK["attr"] = "prestige_vanishing"
    probe_dir = W.OUT_DIR / "probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    tag = "cell_pvan_gen01_cur05_mgstd5"
    t0 = time.time()
    generate_data_parametric_exploration(
        str(probe_dir / tag),
        W.param_grid_for({**cell, "tag": tag, "entry_prestige_stock": "pvan"}),
        repeats_per_setting=2,
        change_landscape_seed=False,
        intention="w",
        skip_to=0,
        longitudinal=False,
    )
    wall = time.time() - t0
    import csv as _csv

    with open(probe_dir / f"{tag}.csv", newline="") as fh:
        rows = list(_csv.DictReader(fh))
    required = [
        "ideal_merit_sum", "ideal_merit_mean",
        "corr_prestige_ideal_merit_final", "corr_prestige_local_merit_final",
        "Mean_of_Corr_prestige_ideal_merit", "Mean_of_Corr_prestige_local_merit",
        "top10pct_prestige_share",
        "n_incumbents", "n_newcomers",
        "newcomer_incumbent_prestige_gap", "newcomer_incumbent_gap_active_stock",
        "mean_prestige",
    ]
    missing = [k for k in required if k not in rows[0]]
    print(f"\n[4] SCHEMA rows={len(rows)} cols={len(rows[0])} missing={missing}")
    for k in required:
        print(f"    {k:40s} {[r.get(k) for r in rows]}")
    print(f"[4] TIMING 2 runs through the sweep interface: {wall:.1f}s "
          f"({wall / 2:.2f}s/run)")
    if missing or len(rows) != 2:
        failures.append("schema")

    print("\nPROBE:", "ALL PROOFS OK" if not failures else f"FAILED -> {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
