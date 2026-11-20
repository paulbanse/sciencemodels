"""Fifth-axis sweep: which prestige stock weights the supervisor roulette.

ADR-0002: driver code lives in scripts/prestige_stock/, run outputs in
data/prestige_stock_v1/. NOT CALIBRATED: model-internal exploration only
(owner ruling 2026-11-13).

THE AXIS IS APPLIED BY THIS DRIVER, NOT BY THE MODEL. As of this run the
model has no `entry_prestige_stock` parameter: the supervisor roulette in
`sciencemodels/model.py` (`_MyModel__generate_new_agents`) hard-codes the raw
`prestige` stock in two places, and the lever is catalogued but unimplemented
(`docs/conceptual_degrees_of_freedom.md:200`, `docs/way_forward.md:122`).
The owner ruled (2026-11-13) that this round applies the selector driver-side
and discloses it. Concretely:

* `_install_stock_selector()` replaces `_MyModel__generate_new_agents` with a
  body that is byte-faithful to the model's except that the two `a.prestige`
  reads become `getattr(a, ACTIVE_STOCK["attr"])`.
* `_DIVERGENCE_CANARY` asserts, from `inspect.getsource` of the ORIGINAL
  method, that it still contains exactly two `a.prestige` reads and no stock
  selector of its own -- so if the model later gains the real parameter, this
  driver fails loudly instead of silently measuring a stale copy.
* With the active stock set to `prestige`, the patched method must be
  bit-identical to the model's own (proved in the probe, see
  `probe_prestige_stock.py`); results therefore transfer unchanged if the
  parameter lands with the same two-line semantics.

Grid: 3 stocks x 2 entry rates x 3 curiosities x 2 landscapes = 36 cells,
20 seeds per cell = 720 runs, step_limit 200.

Run:
    MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/prestige_stock/run_prestige_stock_sweep.py --probe
    MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/prestige_stock/run_prestige_stock_sweep.py
"""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
from joblib import Parallel, delayed

from sciencemodels import landscapes
from sciencemodels.agents import Scientist
import sciencemodels.io as _io
from sciencemodels.model import MyModel, generate_data_parametric_exploration

# --- axis levels ------------------------------------------------------------
STOCKS = ("prestige", "prestige_vanishing", "prestige_visibility")
ENTRY_RATES = (0.0, 0.1)
CURIOSITIES = (0.3, 0.5, 0.7)
LANDSCAPES = ("mgstd5", "noisypr5")

LANDSCAPE_BUNDLES = {
    # waves 1-2 bundles, mirroring bulk_runner setups 2/3 and 6/7
    # (docs/sensitivity_map_v1.md, axis (c))
    "mgstd5": (
        landscapes.multipleGaussians,
        {"std": 5, "number_gaussians": 2},
    ),
    "noisypr5": (
        landscapes.noisyGaussian,
        {"prop_random": 5, "std": 5, "std_curiosity": 0.1},
    ),
}

SEEDS = 20
STEP_LIMIT = 200
GRID_SIZE = 20
N_AGENTS = 20

OUT_DIR = REPO / "data" / "prestige_stock_v1"
STATE_DIR = OUT_DIR / "state"
FIG_DIR = OUT_DIR / "figures"
LOG_PATH = OUT_DIR / "run_log.txt"

BASE_PARAMS = {
    "n_agents": N_AGENTS,
    "initial_curiosity": 0.5,
    "epsilon": 0.1,
    "harvest": 0.02,
    "grid_size": GRID_SIZE,
    "use_distance": True,
    "agent_generation_time": 0.0,
    "new_questions": 0,
    "agent_seed": 0,
    "step_limit": STEP_LIMIT,
    # model default (`lambda cur, rng, params: cur`); pinned explicitly so the
    # CSV is self-describing, as in docs/sensitivity_map_v1.md
    "agent_generation_function": lambda cur, _rng, _params: cur,
    "vanishing_factor_for_prestige": 0.9,  # bulk_runner / waves 1-2 base
    "use_visibility_reward": True,
    "knowledge_normalization": "share",
    "prestige_accrual": "harvested",
}

ACTIVE_STOCK = {"attr": "prestige"}

# --- the driver-side stock selector ----------------------------------------
_ORIG_SUPERVISOR_BLOCK = MyModel._MyModel__generate_new_agents
_DIVERGENCE_CANARY = None  # set by _install_stock_selector()


def _patched_generate_new_agents(self):
    """Model body with the two supervisor-weight reads redirected to the axis.

    Byte-faithful to `MyModel._MyModel__generate_new_agents` apart from the
    two `getattr` lines; `_DIVERGENCE_CANARY` asserts the original still
    matches what was copied.
    """
    if self.agent_generation_time > 0:
        agents_to_generate: int = np.floor(
            self.steps * self.agent_generation_time
        ) - np.floor((self.steps - 1) * self.agent_generation_time)
        stock = ACTIVE_STOCK["attr"]
        prestige_agent_tuples = [
            (getattr(a, stock), a) for a in self._scientists
        ]
        totalPrestige = sum(getattr(a, stock) for a in self._scientists)
        supervisors = []
        for _ in range(int(agents_to_generate)):
            val = self.rng.uniform(0, totalPrestige)
            for p, a in prestige_agent_tuples:
                if val <= p:
                    supervisors.append(a)
                    break
                val -= p
        for sup in supervisors:
            w = self.agent_generation_function(
                self.initial_curiosity, self.rng, self.generation_params
            )
            a = Scientist(self, w, self.epsilon)
            coords = sup.pos
            self.grid.place_agent(a, coords)
            self._scientists.append(a)
        for agent in self._scientists:
            self.grid.properties["explored"].data[agent.pos] = True


def install_stock_selector() -> None:
    """Install the selector once per process, guarded against model drift."""
    global _DIVERGENCE_CANARY
    if _DIVERGENCE_CANARY is None:
        original_src = inspect.getsource(_ORIG_SUPERVISOR_BLOCK)
        reads = original_src.count("a.prestige")
        has_selector = "getattr(a," in original_src or "entry_prestige" in original_src
        if reads != 2 or has_selector:
            raise RuntimeError(
                "model supervisor block no longer matches the copy this driver "
                f"patched (a.prestige reads={reads}, has_selector={has_selector}): "
                "re-derive scripts/prestige_stock/run_prestige_stock_sweep.py "
                "against the current model before sweeping."
            )
        _DIVERGENCE_CANARY = {
            "a_prestige_reads": reads,
            "model_already_has_selector": has_selector,
            "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    # assign unconditionally so the probe can restore() and re-install()
    MyModel._MyModel__generate_new_agents = _patched_generate_new_agents


# NOTE (2026-11-20, post-CDF-v5): install_stock_selector/restore_model and the
# patched body above are RETIRED for sweeping -- the model owns
# `entry_prestige_stock` and verify_model_selector() guards it. They remain
# only for probe_prestige_stock.py's legacy identity proof, which must be
# re-derived separately (it still raises loudly via the canary above).


def restore_model() -> None:
    """Put the model's own method back (used by the probe's identity proof)."""
    MyModel._MyModel__generate_new_agents = _ORIG_SUPERVISOR_BLOCK


def verify_model_selector() -> None:
    """The model owns `entry_prestige_stock` now (cdf v4/v5): verify, don't patch.

    The driver-side monkeypatch below is retired for sweeping; this guard
    fails loudly if the model's selector semantics drift again.
    """
    global _DIVERGENCE_CANARY
    params = inspect.signature(MyModel.__init__).parameters
    src = inspect.getsource(MyModel._MyModel__generate_new_agents)
    ok = (
        "entry_prestige_stock" in params
        and "entry_stock = self.entry_prestige_stock" in src
        and src.count("getattr(a, entry_stock)") == 2
    )
    if not ok:
        raise RuntimeError(
            "model no longer owns a verifiable entry_prestige_stock selector "
            f"(param={'entry_prestige_stock' in params}, "
            f"getattr reads={src.count('getattr(a, entry_stock)')}): "
            "re-derive scripts/prestige_stock/run_prestige_stock_sweep.py "
            "against the current model before sweeping."
        )
    _DIVERGENCE_CANARY = {
        "model_owns_selector": True,
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "driver-side monkeypatch retired; axis passed as the "
                "entry_prestige_stock constructor parameter",
    }


# --- report wrapper (model does not report ideal merit / final wedge / gap) --
_ORIG_BUILD_REPORT = _io.build_report


def _welfare_build_report(model):
    row = _ORIG_BUILD_REPORT(model)
    df = model.datacollector.get_model_vars_dataframe()
    # BUG-0006 convention (metrics.py): finite-part over agents -- unscored
    # step-t newborns carry ideal_Merit = NaN and must not poison the sums
    merits = [float(a.ideal_Merit) for a in model.agents]
    finite = [m for m in merits if np.isfinite(m)]
    row["ideal_merit_sum"] = float(np.sum(finite)) if finite else float("nan")
    row["ideal_merit_mean"] = float(np.mean(finite)) if finite else float("nan")
    row["n_unscored_final_step"] = len(merits) - len(finite)
    row["corr_prestige_ideal_merit_final"] = float(
        df["Corr_prestige_ideal_merit"].iloc[-1]
    )
    row["corr_prestige_local_merit_final"] = float(
        df["Corr_prestige_local_merit"].iloc[-1]
    )
    # newcomer / incumbent split: `_scientists` is created in order, so the
    # first `number_agents` entries are the founders of step 0
    scientists = list(model._scientists)
    founders = scientists[: model.number_agents]
    newcomers = scientists[model.number_agents:]
    row["n_incumbents"] = len(founders)
    row["n_newcomers"] = len(newcomers)
    if founders:
        row["incumbent_prestige_mean"] = float(
            np.mean([a.prestige for a in founders])
        )
    if newcomers:
        row["newcomer_prestige_mean"] = float(
            np.mean([a.prestige for a in newcomers])
        )
        row["newcomer_incumbent_prestige_gap"] = (
            row["newcomer_prestige_mean"] - row["incumbent_prestige_mean"]
        )
        stock = ACTIVE_STOCK["attr"]
        row["newcomer_incumbent_gap_active_stock"] = float(
            np.mean([getattr(a, stock) for a in newcomers])
            - np.mean([getattr(a, stock) for a in founders])
        )
    else:
        row["newcomer_prestige_mean"] = float("nan")
        row["newcomer_incumbent_prestige_gap"] = float("nan")
        row["newcomer_incumbent_gap_active_stock"] = float("nan")
    return row


def install_report_wrapper() -> None:
    if not getattr(_io, "_ps_patch_installed", False):
        _io.build_report = _welfare_build_report
        _io._ps_patch_installed = True


# --- grid -------------------------------------------------------------------
def tag_for(stock, entry, curiosity, landscape):
    short = {
        "prestige": "pre",
        "prestige_vanishing": "pvan",
        "prestige_visibility": "pvis",
    }[stock]
    return (
        f"cell_{short}"
        f"_gen{int(round(entry * 10)):02d}"
        f"_cur{int(round(curiosity * 10)):02d}"
        f"_{landscape}"
    )


def build_cells():
    cells = []
    for stock in STOCKS:
        for entry in ENTRY_RATES:
            for curiosity in CURIOSITIES:
                for landscape in LANDSCAPES:
                    cells.append(
                        {
                            "tag": tag_for(stock, entry, curiosity, landscape),
                            "entry_prestige_stock": stock,
                            "agent_generation_time": entry,
                            "initial_curiosity": curiosity,
                            "landscape": landscape,
                        }
                    )
    return cells


def param_grid_for(cell) -> dict:
    cell_init, generation_params = LANDSCAPE_BUNDLES[cell["landscape"]]
    grid = dict(BASE_PARAMS)
    grid.update(
        {
            "initial_curiosity": cell["initial_curiosity"],
            "agent_generation_time": cell["agent_generation_time"],
            "entry_prestige_stock": cell["entry_prestige_stock"],
            "cell_init_function": cell_init,
            "generation_params": generation_params,
        }
    )
    return grid


def count_rows(path: Path) -> int:
    with open(path, newline="") as fh:
        n = sum(1 for _ in csv.reader(fh))
    return max(n - 1, 0)


def is_done(cell, seeds: int) -> bool:
    csv_path = OUT_DIR / f"{cell['tag']}.csv"
    state_path = STATE_DIR / f"{cell['tag']}.json"
    if not (csv_path.exists() and state_path.exists()):
        return False
    try:
        state = json.loads(state_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        state.get("status") == "ok"
        and state.get("rows") == seeds
        and count_rows(csv_path) == seeds
    )


def run_cell(cell, seeds: int) -> dict:
    # joblib ships task bodies by value and does not re-execute this module in
    # the worker, so both patches are installed from inside the task.
    # The driver-side stock monkeypatch is RETIRED: the model owns
    # `entry_prestige_stock` now (passed via param_grid_for); only the report
    # wrapper is still installed here. ACTIVE_STOCK remains for the wrapper's
    # active-stock gap column.
    install_report_wrapper()
    ACTIVE_STOCK["attr"] = cell["entry_prestige_stock"]

    tag = cell["tag"]
    csv_path = OUT_DIR / f"{tag}.csv"
    state_path = STATE_DIR / f"{tag}.json"
    if is_done(cell, seeds):
        return {"tag": tag, "status": "skipped", "rows": seeds, "wall_s": 0.0}

    started = time.time()
    generate_data_parametric_exploration(
        str(OUT_DIR / tag),
        param_grid_for(cell),
        repeats_per_setting=seeds,
        change_landscape_seed=False,
        intention="w",
        skip_to=0,
        longitudinal=False,
    )
    wall = time.time() - started
    rows = count_rows(csv_path) if csv_path.exists() else -1
    status = "ok" if rows == seeds else "row_mismatch"

    state = {
        "tag": tag,
        "status": status,
        "rows": rows,
        "expected_rows": seeds,
        "wall_s": round(wall, 2),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **{k: cell[k] for k in (
            "entry_prestige_stock",
            "agent_generation_time",
            "initial_curiosity",
            "landscape",
        )},
    }
    tmp = state_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, state_path)

    with open(LOG_PATH, "a") as fh:
        fh.write(
            f"[{state['finished_at']}] {tag} -> {status} rows={rows}/{seeds} "
            f"wall={wall:.1f}s\n"
        )
    return {"tag": tag, "status": status, "rows": rows, "wall_s": round(wall, 2)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Fifth-axis prestige-stock sweep")
    parser.add_argument("--probe", action="store_true",
                        help="one cell, two seeds (critical-path check)")
    parser.add_argument("--seeds", type=int, default=SEEDS)
    parser.add_argument("--n-jobs", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None,
                        help="output dir (default: data/prestige_stock_v1)")
    args = parser.parse_args()

    global OUT_DIR, STATE_DIR, FIG_DIR, LOG_PATH
    if args.out:
        OUT_DIR = Path(args.out)
        STATE_DIR = OUT_DIR / "state"
        FIG_DIR = OUT_DIR / "figures"
        LOG_PATH = OUT_DIR / "run_log.txt"

    seeds = 2 if args.probe else args.seeds
    n_jobs = min(args.n_jobs, 3)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    verify_model_selector()
    design = {
        "run": OUT_DIR.name,
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "not_calibrated": True,
        "axis": "entry_prestige_stock (supervisor-roulette weight)",
        "axis_source": "applied by this driver; the model has no such parameter "
                       "(catalogued at docs/conceptual_degrees_of_freedom.md:200)",
        "levels": {
            "entry_prestige_stock": list(STOCKS),
            "agent_generation_time": list(ENTRY_RATES),
            "initial_curiosity": list(CURIOSITIES),
            "landscape": list(LANDSCAPES),
        },
        "landscape_bundles": {
            k: {"generator": v[0].__name__, "params": v[1]}
            for k, v in LANDSCAPE_BUNDLES.items()
        },
        "seeds_per_cell": seeds,
        "seed": "agent_seed = repeat 0..n-1; landscape seed derived from it",
        "fixed_params": {
            k: (v.__name__ if callable(v) else v)
            for k, v in BASE_PARAMS.items()
            if k not in ("cell_init_function", "agent_generation_function")
        },
        "divergence_canary": _DIVERGENCE_CANARY,
    }
    (OUT_DIR / "design.json").write_text(json.dumps(design, indent=2))

    cells = build_cells()
    if args.probe:
        cells = [c for c in cells if c["tag"] == "cell_pre_gen00_cur05_mgstd5"]
    if args.limit is not None:
        cells = cells[: args.limit]

    todo = [c for c in cells if not is_done(c, seeds)]
    print(
        f"design: {len(cells)} cells x {seeds} seeds "
        f"({len(cells) * seeds} runs); complete: {len(cells) - len(todo)}; "
        f"to run: {len(todo)}; n_jobs={n_jobs}",
        flush=True,
    )

    results = Parallel(n_jobs=n_jobs)(delayed(run_cell)(c, seeds) for c in todo)

    ok = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skipped"]
    bad = [r for r in results if r["status"] not in ("ok", "skipped")]
    gross = len(cells) * seeds
    net = sum(r["rows"] for r in results if r["rows"] > 0) + (
        len(cells) - len(todo)
    ) * seeds
    print(
        f"GROSS cells={len(cells)} runs={gross} | NET ok={len(ok)} "
        f"skipped={len(skipped)} failed={len(bad)} rows_net={net}/{gross}",
        flush=True,
    )
    for r in bad:
        print(f"  FAILED {r}", flush=True)
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
