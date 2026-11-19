"""M0 dyadic-off control for the P-0011 probe (M0 + M2 without entry).

The brief's `m0_m2_probe/` holds the M0+M2 arms (g = 0, pull = 1.0,
lambda in {0, 1}). This control adds the missing reference: M0 with the
dyadic layer OFF (g = 0, mgstd5 + noisypr5, 20 seeds = 40 runs), so the
probe can answer whether M2 still fragments teams / bifurcates alignment
WITHOUT demographic entry -- a question that needs a baseline.

Params mirror ``scripts/path_b/run_baseline_off.py`` exactly (same
BASE_PARAMS, credit structural, dyadic off) except
``agent_generation_time = 0.0``. Writes
``data/path_b_postbug/m0_m2_probe/m0_off_control.csv``.

Usage:
    python scripts/bug0006_wave2/run_m0_control.py
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "scripts" / "path_b")):
    if p not in sys.path:
        sys.path.insert(0, p)

import path_b_v2_common as C  # noqa: E402
import run_path_b_sweep as PB1  # noqa: E402
from sciencemodels.model import MyModel  # noqa: E402

OUT = ROOT / "data" / "path_b_postbug" / "m0_m2_probe" / "m0_off_control.csv"
SEEDS = 20


def run_one_off_g0(land: str, seed: int) -> dict:
    f, gp = PB1.LANDSCAPE_BUNDLES[land]
    params = dict(PB1.BASE_PARAMS)
    params.update(
        cell_init_function=f,
        generation_params=dict(gp),
        agent_seed=seed,
        step_limit=PB1.STEP_LIMIT,
        agent_generation_time=0.0,
        dyadic_mode="off",
        credit_sharing="structural",
    )
    m = MyModel(**params)
    for _ in range(PB1.STEP_LIMIT):
        m.step(True)
    row = C.report_extras(m, m.get_report())
    row["land"] = land
    row["agent_generation_time"] = 0.0
    row["dyadic_mode"] = "off"
    return row


def work(task: tuple) -> dict:
    return run_one_off_g0(*task)


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tasks = [(land, seed) for land in PB1.LANDSCAPES for seed in range(SEEDS)]
    with ProcessPoolExecutor(max_workers=min(10, len(tasks))) as ex:
        rows = list(ex.map(work, tasks))
    df = pd.DataFrame(rows)
    assert len(df) == len(PB1.LANDSCAPES) * SEEDS, f"expected 40 rows, got {len(df)}"
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
