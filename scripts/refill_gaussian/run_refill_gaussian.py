"""Refill-gaussian v1 driver -- IE-0006 Option 5 / workplan item 13.

Standalone, checkpointed experiment for harvest-conserving Gaussian neighbour
refill on a 20x20 torus.  No model file is edited (ADR-0001 owner boundary):
``GaussianRefillModel`` subclasses ``MyModel`` and overrides only ``farm``,
``_generate_new_questions``, ``step`` and ``run_mode_for_bulk``.

Design
------
- Landscape ``mgstd5`` = ``multipleGaussians`` with ``std=5`` and
  ``number_gaussians=2``.
- ``n_agents=20``, ``H=400``, ``epsilon=0.1``, ``harvest=0.02``,
  ``use_distance=True``, ``vanishing_factor_for_prestige=0.9``,
  ``use_visibility_reward=True``, share knowledge normalisation,
  ``prestige_accrual="harvested"``, ``entry_prestige_stock="prestige"``,
  ``dyadic_mode="off"``.
- Axes: refill regime x curiosity x entry rate; 20 seeds per cell.

Refill regimes
--------------
``off``          no refill (baseline).
``occupied``     the model's own ``new_questions=True`` rule: a depleted
                 occupied tile is replaced by a uniform draw in place.
``frontier``     a depleted occupied tile spawns its new question on a random
                 unoccupied tile (the interventions-v1 frontier arm).
``gaussian_s*``  harvest-conserving Option 5: each harvest of Delta k at
                 (x, y) is moved to another tile drawn from a discretised 2D
                 torus Gaussian centred at (x, y), excluding the source.

The Gaussian kernel is translation-invariant on the torus, so it is built once
per run on integer offsets, wrapped to the shortest displacement, with the
source weight set to zero and the rest renormalised.  Sampling is inverse-CDF
(``searchsorted``) off the seeded model RNG.

Metrics
-------
Because Gaussian refill conserves total stock, ``percentage_knowledge_harvested``
is not a usable harvest measure for those arms.  The driver tracks the removed
harvest flow directly (``cumulative_harvest_flow``) plus a late-window flow
(last 50 steps), explored share, and Corr(prestige, ideal_merit).  Occupied and
frontier refills are not stock-conserving; their injected stock is reported for
contrast.

Checkpointing / canaries (ADR-0001)
-----------------------------------
Each cell is written atomically (``.tmp`` -> row-count check -> ``os.replace``)
with a state JSON; a restart skips verified cells.  Before the grid,
``--probe`` proves: (a) ``off`` is bit-identical to plain ``MyModel``,
(b) Gaussian refill conserves total stock, (c) every regime moves the dynamics
against ``off``, and (d) a worker-written cell file carries the analysis
schema.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from joblib import Parallel, delayed

from sciencemodels import landscapes as L
from sciencemodels.model import MyModel

OUT = REPO / "data" / "refill_gaussian_v1"
CELLS_DIR = OUT / "cells"
STATE_DIR = OUT / "state"
PROBE_DIR = OUT / "probe"
for _d in (OUT, CELLS_DIR, STATE_DIR, PROBE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- constants
SEEDS = list(range(20))
STEPS = 400
LATE_WINDOW = 50
CURIOSITIES = (0.3, 0.5, 0.7)
GENERATION_RATES = (0.0, 0.1)
SIGMAS = (1.0, 3.0, 6.0)
REFILL_LEVELS = (
    "off",
    "occupied",
    "frontier",
    *(f"gaussian_s{sigma:g}" for sigma in SIGMAS),
)
GAUSSIAN_LEVELS = tuple(f"gaussian_s{sigma:g}" for sigma in SIGMAS)

LANDSCAPE = ("mgstd5", L.multipleGaussians, {"std": 5, "number_gaussians": 2})

FIXED = {
    "n_agents": 20,
    "epsilon": 0.1,
    "harvest": 0.02,
    "grid_size": 20,
    "use_distance": True,
    "vanishing_factor_for_prestige": 0.9,
    "use_visibility_reward": True,
    "knowledge_normalization": "share",
    "prestige_accrual": "harvested",
    "entry_prestige_stock": "prestige",
    "dyadic_mode": "off",
    "step_limit": STEPS,
}

# Columns every analysis step may rely on (the schema proof asserts them).
REQUIRED_COLUMNS = [
    "cell_tag",
    "refill_mode",
    "sigma",
    "initial_curiosity",
    "agent_generation_time",
    "agent_seed",
    "landscape",
    "cumulative_harvest_flow",
    "late_window_harvest_flow",
    "late_window_flow_rate",
    "cumulative_refill_added",
    "final_total_knowledge",
    "explored_percentage_final",
    "explored_percentage_mean",
    "corr_prestige_ideal_merit_final",
    "corr_prestige_ideal_merit_mean",
    "corr_prestige_ideal_merit_late_mean",
    "corr_prestige_local_merit_final",
    "percentage_knowledge_harvested_final",
    "top10pct_prestige_share_final",
    "mean_prestige_final",
    "n_agents_final",
]


# --------------------------------------------------------------- the model
class GaussianRefillModel(MyModel):
    """``MyModel`` plus the refill regimes of workplan item 13.

    Only the four overrides below are added; every baseline level is inert, so
    ``off`` reproduces plain ``MyModel`` bit-for-bit.
    """

    def __init__(self, *, refill_mode: str = "off", sigma: float | None = None,
                 **kwargs):
        if refill_mode not in REFILL_LEVELS:
            raise ValueError(
                f"refill_mode must be one of {REFILL_LEVELS}, got {refill_mode!r}"
            )
        if refill_mode in GAUSSIAN_LEVELS:
            sigma = float(refill_mode.removeprefix("gaussian_s"))
        if sigma is not None and (not np.isfinite(sigma) or sigma <= 0.0):
            raise ValueError(f"sigma must be finite and positive, got {sigma!r}")
        if refill_mode in GAUSSIAN_LEVELS and sigma is None:
            raise ValueError("Gaussian refill requires a positive sigma")

        self.refill_mode = refill_mode
        self.sigma = sigma
        self.cumulative_harvest_flow = np.float64(0.0)
        self.cumulative_refill_added = np.float64(0.0)
        self._harvest_flow_history: list[float] = []
        # Occupied refill is the model's own new_questions rule; frontier and
        # Gaussian are driver-side overrides and keep the flag off.
        kwargs["new_questions"] = refill_mode == "occupied"
        super().__init__(**kwargs)

        if refill_mode in GAUSSIAN_LEVELS:
            probs = self._torus_gaussian_probs(sigma)
            self._gaussian_probs = probs
            self._gaussian_cdf = np.cumsum(probs)

    # ---- Option 5: harvest-conserving Gaussian neighbour refill
    def _torus_gaussian_probs(self, sigma: float) -> np.ndarray:
        """Discretised 2D torus-Gaussian weights over all non-source offsets.

        ``w(ox, oy) = exp(-d^2 / (2 sigma^2))`` with ``d`` the shortest torus
        displacement.  The source offset (0, 0) is excluded and the remaining
        mass is renormalised, so every draw lands on another tile.
        """
        n = self.size
        weights = np.empty((n, n), dtype=np.float64)
        for ox in range(n):
            dx = ox if ox <= n // 2 else ox - n
            for oy in range(n):
                dy = oy if oy <= n // 2 else oy - n
                weights[ox, oy] = np.exp(-0.5 * (dx * dx + dy * dy) / (sigma * sigma))
        weights[0, 0] = 0.0
        total = weights.sum()
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError(f"degenerate Gaussian kernel for sigma={sigma!r}")
        return weights.ravel() / total

    def _draw_gaussian_target(self, pos: tuple[int, int]) -> tuple[int, int]:
        """Draw a non-source torus tile from the precomputed Gaussian CDF."""
        n = self.size
        idx = int(np.searchsorted(self._gaussian_cdf, self.rng.random(), side="right"))
        idx = min(idx, n * n - 1)
        ox, oy = divmod(idx, n)
        x, y = pos
        return ((x + ox) % n, (y + oy) % n)

    def farm(self, pos, agent=None):
        """Harvest, then optionally move the removed flow to a Gaussian target."""
        knowledge = self.grid.properties["knowledge"].data
        before = np.float64(knowledge[pos])
        super().farm(pos, agent)
        removed = before - np.float64(knowledge[pos])
        self.cumulative_harvest_flow += removed
        if self.refill_mode in GAUSSIAN_LEVELS and removed > 0.0:
            target = self._draw_gaussian_target(pos)
            knowledge[target] += removed
            self.cumulative_refill_added += removed

    # ---- occupied / frontier contrast arms
    def _generate_new_questions(self):
        if self.refill_mode in ("off",) + GAUSSIAN_LEVELS:
            return
        if self.refill_mode == "occupied":
            return super()._generate_new_questions()
        # frontier: a depleted occupied tile spawns on a random free tile
        knowledge = self.grid.properties["knowledge"].data
        occupied = {a.pos for a in self.agents}
        depleted = sorted(p for p in occupied if knowledge[p] < 0.025)
        if not depleted:
            return
        mask = np.zeros((self.size, self.size), dtype=bool)
        for p in occupied:
            mask[p] = True
        free = np.argwhere(~mask)
        if len(free) == 0:
            return
        n = min(len(depleted), len(free))
        for i in self.rng.choice(len(free), size=n, replace=False):
            x, y = free[i]
            old = np.float64(knowledge[x, y])
            knowledge[x, y] = self.rng.random()
            self.cumulative_refill_added += np.float64(knowledge[x, y]) - old

    # ---- flow history for the late-window measure
    def step(self, endupdate=True):
        super().step(endupdate=endupdate)
        self._harvest_flow_history.append(float(self.cumulative_harvest_flow))

    # ---- analysis payload
    def run_mode_for_bulk(self, longitudinal=False):
        row = super().run_mode_for_bulk(longitudinal=longitudinal)
        if longitudinal:
            return row
        mv = self.datacollector.get_model_vars_dataframe()
        last = mv.iloc[-1]
        corr = mv["Corr_prestige_ideal_merit"]
        hist = self._harvest_flow_history
        window = min(LATE_WINDOW, max(len(hist) - 1, 0))
        late_flow = (hist[-1] - hist[-window - 1]) if window > 0 else hist[-1]
        row["refill_mode"] = self.refill_mode
        row["sigma"] = float(self.sigma) if self.sigma is not None else float("nan")
        row["initial_curiosity"] = float(self.initial_curiosity)
        row["agent_generation_time"] = float(self.agent_generation_time)
        row["cumulative_harvest_flow"] = float(self.cumulative_harvest_flow)
        row["late_window_harvest_flow"] = float(late_flow)
        row["late_window_flow_rate"] = float(late_flow / LATE_WINDOW)
        row["cumulative_refill_added"] = float(self.cumulative_refill_added)
        row["final_total_knowledge"] = float(
            np.sum(self.grid.properties["knowledge"].data)
        )
        row["explored_percentage_final"] = float(last["explored_percentage"])
        row["explored_percentage_mean"] = float(mv["explored_percentage"].mean())
        row["corr_prestige_ideal_merit_final"] = float(
            last["Corr_prestige_ideal_merit"]
        )
        row["corr_prestige_ideal_merit_mean"] = float(corr.mean())
        row["corr_prestige_ideal_merit_late_mean"] = float(
            corr.iloc[-LATE_WINDOW:].mean()
        )
        row["corr_prestige_local_merit_final"] = float(
            last["Corr_prestige_local_merit"]
        )
        row["percentage_knowledge_harvested_final"] = float(
            last["percentage_knowledge_harvested"]
        )
        row["top10pct_prestige_share_final"] = float(
            last["top10pct_prestige_share"]
        )
        row["mean_prestige_final"] = float(last["mean_prestige"])
        row["n_agents_final"] = len(self._scientists)
        return row


# ------------------------------------------------------------------ cells
def cell_list() -> list[dict]:
    cells = []
    for mode in REFILL_LEVELS:
        for curiosity in CURIOSITIES:
            for generation in GENERATION_RATES:
                cells.append(
                    dict(
                        refill_mode=mode,
                        initial_curiosity=curiosity,
                        agent_generation_time=generation,
                    )
                )
    assert len(cells) == len(REFILL_LEVELS) * len(CURIOSITIES) * len(
        GENERATION_RATES
    ), f"unexpected cell count {len(cells)}"
    return cells


def cell_tag(cell: dict) -> str:
    mode = cell["refill_mode"]
    c = int(round(cell["initial_curiosity"] * 10))
    g = int(round(cell["agent_generation_time"] * 10))
    return f"cell_refill-{mode}_c{c:02d}_gen{g:02d}_mgstd5"


def params_for(cell: dict, seed: int) -> dict:
    _name, fn, gparams = LANDSCAPE
    p = dict(FIXED)
    p.update(
        cell_init_function=fn,
        generation_params=dict(gparams),
        initial_curiosity=cell["initial_curiosity"],
        agent_generation_time=cell["agent_generation_time"],
        agent_seed=seed,
        refill_mode=cell["refill_mode"],
    )
    return p


# ------------------------------------------------------------ one run/probe
def run_one(cell: dict, seed: int, model_cls=None) -> tuple[str, dict | str]:
    """One seeded run; returns ``("ok", row)`` or ``("fail", traceback)``."""
    cls = model_cls or GaussianRefillModel
    params = params_for(cell, seed)
    if cls is MyModel:
        params.pop("refill_mode", None)
    try:
        model = cls(**params)
        row = model.run_mode_for_bulk()
        row["cell_tag"] = cell_tag(cell)
        row["landscape"] = LANDSCAPE[0]
        return "ok", row
    except Exception:  # noqa: BLE001
        return "fail", traceback.format_exc()


def _numeric_diffs(a: dict, b: dict, keys) -> dict:
    out = {}
    for k in keys:
        va, vb = a.get(k), b.get(k)
        if isinstance(va, (int, float, np.floating, np.integer)) and isinstance(
            vb, (int, float, np.floating, np.integer)
        ):
            out[k] = float(abs(float(va) - float(vb)))
        else:
            out[k] = 0.0 if va == vb else float("inf")
    return out


# ------------------------------------------------------------------ proofs
def proof_identity() -> dict:
    """The off regime must reproduce plain ``MyModel`` bit-for-bit."""
    results = []
    for seed in (0, 1, 2):
        cell = dict(refill_mode="off", initial_curiosity=0.5,
                    agent_generation_time=0.1)
        st_plain, plain = run_one(cell, seed, model_cls=MyModel)
        st_mine, mine = run_one(cell, seed)
        assert st_plain == st_mine == "ok", (st_plain, st_mine)
        common = sorted(set(plain) & set(mine))
        diffs = _numeric_diffs(plain, mine, common)
        worst = max(diffs.values())
        results.append(dict(seed=seed, worst=worst, n_common=len(common),
                            driver_only=sorted(set(mine) - set(plain))))
        assert worst == 0.0, (
            f"off regime not bit-identical on seed {seed}: "
            f"{[k for k, v in diffs.items() if v != 0.0][:8]}"
        )
    return {"checks": len(results), "worst": 0.0,
            "driver_only_keys": results[0]["driver_only"]}


def proof_conservation() -> dict:
    """Gaussian refill must leave the total knowledge stock unchanged."""
    rows = []
    for sigma in SIGMAS:
        cell = dict(refill_mode=f"gaussian_s{sigma:g}",
                    initial_curiosity=0.5, agent_generation_time=0.1)
        for seed in (0, 1):
            st, row = run_one(cell, seed)
            assert st == "ok", (sigma, seed, st)
            total_err = abs(row["final_total_knowledge"] - 1.0)
            flow_err = abs(row["cumulative_refill_added"]
                           - row["cumulative_harvest_flow"])
            rows.append(dict(sigma=sigma, seed=seed,
                             total_error=total_err, flow_error=flow_err,
                             final_total=row["final_total_knowledge"]))
            assert total_err < 1e-9, (
                f"sigma={sigma} seed={seed}: total stock drift {total_err}"
            )
            assert flow_err < 1e-9, (
                f"sigma={sigma} seed={seed}: refill/flow mismatch {flow_err}"
            )
    return {"checks": len(rows),
            "max_total_error": max(r["total_error"] for r in rows),
            "max_flow_error": max(r["flow_error"] for r in rows)}


def proof_binding() -> dict:
    """Every non-off regime must move at least one dynamics reporter."""
    base = dict(refill_mode="off", initial_curiosity=0.5,
                agent_generation_time=0.1)
    variant_modes = [m for m in REFILL_LEVELS if m != "off"]
    st0, b = run_one(base, 0)
    assert st0 == "ok"
    model_keys = [
        k for k in b
        if isinstance(b[k], (int, float, np.floating, np.integer))
        and k not in REQUIRED_COLUMNS
        and k not in ("refill_mode", "sigma", "initial_curiosity",
                      "agent_generation_time", "cell_tag", "landscape")
    ]
    out = {}
    for mode in variant_modes:
        st, v = run_one(dict(base, refill_mode=mode), 0)
        assert st == "ok", (mode, st)
        diffs = _numeric_diffs(b, v, model_keys)
        moved = sorted(k for k, d in diffs.items() if d > 0.0)
        out[mode] = {"n_moved": len(moved), "sample": moved[:5],
                     "max": max(diffs.values())}
        assert moved, f"binding failed: {mode} moved nothing"
    return out


def proof_schema(cell: dict, n_jobs: int) -> dict:
    """A WORKER-written cell file must carry every analysis column."""
    rows = Parallel(n_jobs=n_jobs, return_as="list")(
        delayed(run_one)(cell, s) for s in SEEDS[:4]
    )
    ok = [r for st, r in rows if st == "ok"]
    assert len(ok) == 4, f"schema probe: {len(ok)}/4 runs ok"
    path = PROBE_DIR / "schema_probe.csv"
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=sorted(ok[0].keys()))
        writer.writeheader()
        writer.writerows(ok)
    os.replace(tmp, path)
    with open(path, newline="") as fh:
        back = list(csv.DictReader(fh))
    missing = [c for c in REQUIRED_COLUMNS if c not in (back[0] if back else {})]
    assert not missing, f"worker-written file missing columns: {missing}"
    assert len(back) == 4, f"expected 4 rows, got {len(back)}"
    return {"rows": len(back), "columns": len(back[0]), "missing": missing}


def probe_timing() -> dict:
    out = {}
    for mode in REFILL_LEVELS:
        cell = dict(refill_mode=mode, initial_curiosity=0.5,
                    agent_generation_time=0.1)
        t0 = time.time()
        st, _ = run_one(cell, 0)
        out[mode] = {"wall_s": round(time.time() - t0, 2), "status": st}
    return out


# --------------------------------------------------------------- the sweep
def run_cell(cell: dict, n_jobs: int, log) -> dict:
    tag = cell_tag(cell)
    csv_path = CELLS_DIR / f"{tag}.csv"
    state_path = STATE_DIR / f"{tag}.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text())
        if state.get("status") in ("ok", "ok_with_failures"):
            log(f"SKIP {tag} (already {state['status']})")
            return state
    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, return_as="list")(
        delayed(run_one)(cell, s) for s in SEEDS
    )
    rows, errors = [], {}
    for seed, (status, payload) in zip(SEEDS, results):
        if status == "ok":
            rows.append(payload)
        else:
            errors[str(seed)] = str(payload)[-2000:]
    n_expect = len(SEEDS)
    status = "ok" if len(rows) == n_expect else (
        "ok_with_failures" if rows else "crashed"
    )
    if rows:
        tmp = csv_path.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=sorted(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        with open(tmp, newline="") as fh:
            n_back = sum(1 for _ in csv.DictReader(fh))
        assert n_back == len(rows), (
            f"{tag}: wrote {len(rows)} rows, read back {n_back}"
        )
        os.replace(tmp, csv_path)
    entry = {
        "cell_tag": tag,
        "status": status,
        "params": dict(cell),
        "seeds_ok": len(rows),
        "seeds_failed": len(errors),
        "rows": len(rows),
        "rows_expected": n_expect,
        "errors": errors,
        "csv": csv_path.name if rows else None,
        "wall_s": round(time.time() - t0, 2),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    tmp_state = state_path.with_suffix(".json.tmp")
    tmp_state.write_text(json.dumps(entry, indent=1))
    os.replace(tmp_state, state_path)
    log(
        f"CELL {tag} -> {status} seeds_ok={entry['seeds_ok']} "
        f"rows={entry['rows']}/{n_expect} wall={entry['wall_s']}s"
        + (f" ERRORS={list(errors)}" if errors else "")
    )
    return entry


def preflight_snapshot(path: Path) -> None:
    try:
        import subprocess

        out = subprocess.run(["ps", "aux"], capture_output=True, text=True).stdout
        heavy = [
            ln for ln in out.splitlines()
            if any(k in ln for k in ("python", "julia")) and "grep" not in ln
        ]
        path.write_text(
            f"{len(heavy)} live python/julia processes\n" + "\n".join(heavy)
        )
    except Exception as exc:  # noqa: BLE001
        path.write_text(f"snapshot failed: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true",
                    help="run identity/conservation/binding/schema/timing proofs")
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--only-mode", default=None,
                    help="run only one refill mode (default: all)")
    args = ap.parse_args()

    log_path = OUT / "run_log.txt"

    def log(msg):
        line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as fh:
            fh.write(line + "\n")

    if args.probe:
        log("PROBE start")
        ident = proof_identity()
        log(f"PROBE identity ok: {ident['checks']} checks, worst diff 0.0, "
            f"driver-only keys={ident['driver_only_keys']}")
        cons = proof_conservation()
        log(f"PROBE conservation ok: {cons}")
        binding = proof_binding()
        log("PROBE binding ok: "
            + "; ".join(f"{k}->{v['n_moved']} keys" for k, v in binding.items()))
        schema = proof_schema(cell_list()[0], args.n_jobs)
        log(f"PROBE schema ok: {schema}")
        timing = probe_timing()
        log(f"PROBE timing: {timing}")
        (PROBE_DIR / "probe_results.json").write_text(json.dumps(
            {"identity": ident, "conservation": cons, "binding": binding,
             "schema": schema, "timing": timing},
            indent=1, default=str,
        ))
        print(json.dumps(
            {"identity": ident, "conservation": cons, "binding": binding,
             "schema": schema, "timing": timing},
            indent=1, default=str,
        ))
        return 0

    t_start = time.time()
    cells = cell_list()
    if args.only_mode:
        if args.only_mode not in REFILL_LEVELS:
            raise SystemExit(f"unknown mode {args.only_mode!r}")
        cells = [c for c in cells if c["refill_mode"] == args.only_mode]
    design = {
        "schema": "refill_gaussian_v1",
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "grid": f"cells={len(cells)} seeds={len(SEEDS)} steps={STEPS}",
        "refill_levels": list(REFILL_LEVELS),
        "sigmas": list(SIGMAS),
        "curiosities": list(CURIOSITIES),
        "generation_rates": list(GENERATION_RATES),
        "fixed": FIXED,
        "landscape": {"name": LANDSCAPE[0], "fn": LANDSCAPE[1].__name__,
                      "params": LANDSCAPE[2]},
        "n_jobs": args.n_jobs,
        "model_boundary": "GaussianRefillModel subclasses MyModel; no model edit",
        "flow_metric": "cumulative removed harvest Delta k (stock-conserving "
                       "for Gaussian arms); late window = last 50 steps",
    }
    (OUT / "design.json").write_text(json.dumps(design, indent=1, default=str))
    preflight_snapshot(OUT / "process_snapshot.txt")

    gross = len(cells) * len(SEEDS)
    log(f"sweep start: cells={len(cells)} gross_runs={gross} "
        f"seeds/cell={len(SEEDS)} steps={STEPS} n_jobs={args.n_jobs}")

    states = []
    for i, cell in enumerate(cells, 1):
        state = run_cell(cell, args.n_jobs, log)
        states.append(state)
        if i % 6 == 0:
            log(f"progress {i}/{len(cells)} cells, "
                f"elapsed={time.time() - t_start:.0f}s")

    n_ok = sum(1 for s in states if s["status"] == "ok")
    n_part = sum(1 for s in states if s["status"] == "ok_with_failures")
    n_crash = sum(1 for s in states if s["status"] == "crashed")
    runs_ok = sum(s["seeds_ok"] for s in states)
    runs_fail = sum(s["seeds_failed"] for s in states)
    wall = time.time() - t_start
    line = (
        f"GROSS cells={len(cells)} runs={gross} | NET ok_cells={n_ok} "
        f"ok_with_failures={n_part} crashed={n_crash} "
        f"runs_ok={runs_ok} runs_failed={runs_fail} wall={wall:.1f}s"
    )
    log(line)
    print(line)
    return 0 if n_crash == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
