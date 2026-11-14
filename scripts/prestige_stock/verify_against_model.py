"""Verify the fifth-axis sweep against the model's own implementation.

Two rounds landed in the shared tree WHILE this sweep was running (see
docs/prestige_stock_v1.md, "Concurrent rounds"):

* 00:17:23Z - `metrics.py` / `io.py`: the ideal-merit pairing audit and the
  seed columns. `ideal_Merit` is an input to `Corr_prestige_ideal_merit`, so
  the ideal-side numbers may move; local-merit quantities must NOT.
* 00:26:23Z - `model.py`: `entry_prestige_stock` was added as a real model
  parameter (`model.py:136,314`), i.e. the axis this round measured
  driver-side now exists in the model.

This script settles both questions with runs, not argument:

1. EQUIVALENCE - for each of the three stocks, the driver-side selector and
   the model's own `entry_prestige_stock` must produce bit-identical reports
   on the same cell and seed. That is the "results transfer unchanged" claim
   in the report, tested.
2. AUDIT DELTA - re-run an entry-off cell (where the selector is inert, so
   the pairing audit is the only difference) under current code and compare
   against the same seeds in the sweep's CSV. Ideal-merit quantities are the
   treatment; local-merit quantities are the control and must not move.

Run: MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/prestige_stock/verify_against_model.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_prestige_stock_sweep as W
from sciencemodels.model import MyModel

ENTRY_ON = {
    "tag": "verify",
    "entry_prestige_stock": "prestige_vanishing",
    "agent_generation_time": 0.1,
    "initial_curiosity": 0.5,
    "landscape": "mgstd5",
}
ENTRY_OFF = {
    "tag": "verify",
    "entry_prestige_stock": "prestige",
    "agent_generation_time": 0.0,
    "initial_curiosity": 0.5,
    "landscape": "mgstd5",
}
CELL_CSV = "cell_pre_gen00_cur05_mgstd5.csv"  # the entry-off sweep cell


def run(cell, seed, stock=None, driver_patch=False):
    grid = W.param_grid_for(cell)
    grid["agent_seed"] = seed
    if stock is not None:
        grid["entry_prestige_stock"] = stock
    # the wrapper measures the gap on "the stock that weights entry"; in the
    # model-own path that is the model's own parameter, in the driver path it
    # is ACTIVE_STOCK -- set both so the two runs are comparable
    W.ACTIVE_STOCK["attr"] = stock or cell["entry_prestige_stock"]
    if driver_patch:
        # Deliberately bypass install_stock_selector(): its divergence canary
        # correctly refuses this copy now that the model gained the real
        # parameter (00:26:23Z), and comparing that FROZEN copy against the
        # model's live implementation is precisely what this test does. This
        # is a verification path, never a sweep launch.
        MyModel._MyModel__generate_new_agents = W._patched_generate_new_agents
        W.ACTIVE_STOCK["attr"] = stock or cell["entry_prestige_stock"]
    else:
        W.restore_model()          # the model's own supervisor block
    W.install_report_wrapper()
    model = MyModel(**grid)
    return model.run_mode_for_bulk()


def bits(report):
    return sorted((k, repr(v)) for k, v in report.items())


def main():
    W.install_report_wrapper()
    failures = []

    print("=== 1. EQUIVALENCE: driver-side selector vs the model's own knob ===")
    for stock in W.STOCKS:
        W.restore_model()
        own = run(ENTRY_ON, seed=7, stock=stock, driver_patch=False)
        patched = run(ENTRY_ON, seed=7, stock=stock, driver_patch=True)
        same = bits(own) == bits(patched)
        print(f"  {stock:20s} model-own vs driver-side: "
              f"{'IDENTICAL' if same else 'DIFFER'} "
              f"(n_newcomers={own['n_newcomers']}, "
              f"c_ideal_fin={own['corr_prestige_ideal_merit_final']:+.4f})")
        if not same:
            for k in sorted(set(own) | set(patched)):
                if repr(own.get(k)) != repr(patched.get(k)):
                    print(f"      differs {k}: own={own.get(k)!r} "
                          f"patched={patched.get(k)!r}")
            failures.append(f"equivalence:{stock}")
    W.restore_model()

    print("\n=== 2. AUDIT DELTA: current code vs the sweep's stored rows ===")
    import statistics as st
    for label, cell, stock in (
        ("entry-OFF cell (selector inert; isolates the audit)",
         ENTRY_OFF, None),
        ("entry-ON cell, stock=prestige (baseline selector)",
         {**ENTRY_ON, "entry_prestige_stock": "prestige"}, "prestige"),
    ):
        csv_name = ("cell_pre_gen00_cur05_mgstd5.csv" if stock is None
                    else "cell_pre_gen01_cur05_mgstd5.csv")
        with open(W.OUT_DIR / csv_name, newline="") as fh:
            stored = list(csv.DictReader(fh))
        keys = list(stored[0])
        ideal_keys = [k for k in keys if "ideal_merit" in k
                      or "Corr_prestige_ideal" in k]
        control_keys = [k for k in keys if "Corr_prestige_local" in k
                        or k in ("percentage_knowledge_harvested",
                                 "explored_percentage", "mean_prestige",
                                 "top10pct_prestige_share")]
        n = 10
        cur_rows = [run(cell, seed=s, stock=stock, driver_patch=False)
                    for s in range(n)]
        print(f"\n  --- {label}: {csv_name}, {n} seeds ---")
        print(f"    {'key':40s} {'stored mean':>14s} {'current mean':>14s} "
              f"{'delta':>12s} {'max abs':>10s}")
        for k in ideal_keys + control_keys:
            if k not in cur_rows[0]:
                continue
            old = [float(r[k]) for r in stored[:n]]
            new = [float(r[k]) for r in cur_rows]
            ds = [b - a for a, b in zip(old, new)]
            kind = "IDEAL" if k in ideal_keys else "ctrl"
            print(f"    [{kind}] {k:34s} {st.fmean(old):14.6g} "
                  f"{st.fmean(new):14.6g} {st.fmean(ds):+12.3e} "
                  f"{max(abs(d) for d in ds):10.3e}")
            if kind == "ctrl" and max(abs(d) for d in ds) > 1e-12:
                failures.append(f"control-moved:{k}")

    print("\n=== 3. DOES THE AUDIT MOVE THE HEADLINE CONTRASTS? ===")
    # largest stored wedge_fin contrast: entry 0.1, cur 0.5, noisypr5
    cell_a = {**ENTRY_ON, "entry_prestige_stock": "prestige",
              "landscape": "noisypr5", "initial_curiosity": 0.5}
    csv_a = "cell_pre_gen01_cur05_noisypr5.csv"
    csv_b = "cell_pvan_gen01_cur05_noisypr5.csv"
    with open(W.OUT_DIR / csv_a, newline="") as fh:
        stored_pre = list(csv.DictReader(fh))
    with open(W.OUT_DIR / csv_b, newline="") as fh:
        stored_pvan = list(csv.DictReader(fh))
    n = 10

    def wedge_fin(row):
        return (float(row["corr_prestige_local_merit_final"])
                - float(row["corr_prestige_ideal_merit_final"]))

    cur_pre = [run(cell_a, seed=s, stock="prestige",
                   driver_patch=False) for s in range(n)]
    cur_pvan = [run(cell_a, seed=s, stock="prestige_vanishing",
                    driver_patch=False) for s in range(n)]
    stored_diff = [wedge_fin(b) - wedge_fin(a)
                   for a, b in zip(stored_pre[:n], stored_pvan[:n])]
    current_diff = [wedge_fin(b) - wedge_fin(a)
                    for a, b in zip(cur_pre, cur_pvan)]
    import statistics as st
    print(f"    cell {csv_a[5:-4]} vs pvan, {n} seeds, wedge_fin contrast")
    print(f"    stored  (pre-audit) mean={st.fmean(stored_diff):+.4f} "
          f"max|.|={max(abs(d) for d in stored_diff):.4f}")
    print(f"    current (post-audit) mean={st.fmean(current_diff):+.4f} "
          f"max|.|={max(abs(d) for d in current_diff):.4f}")
    print(f"    shift in the contrast mean="
          f"{st.fmean(current_diff) - st.fmean(stored_diff):+.4f}")

    print("\nVERIFY:", "ALL OK" if not failures else f"FAILED -> {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
