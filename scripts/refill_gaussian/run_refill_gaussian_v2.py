"""Refill-gaussian v2 driver -- peak-centroid tracking & P-0007 sabbatical pairing.

Extends ``scripts/refill_gaussian/`` (Slot 4 / prompt D4) with two additions,
without touching ``sciencemodels/`` (ADR-0001 owner boundary):

1. **Spatial peak-centroid tracking** (reporting-only): per step, the torus
   centroid of the grid's top-5% knowledge mass, the distance the centroid
   moves, agent-to-peak distance, and the top-5% dispersion -- summarised at
   ``H = 400`` and ``H = 800`` windows to test whether ``gaussian_s6``
   generates a true traveling frontier or flat diffusion.
2. **Mid-career sabbatical shocks** (``p_reloc = 0.02``, P-0007): the
   depletion-triggered relocation mechanism of
   ``scripts/interventions/run_interventions_v1.py`` (``InterventionsModel``
   ._sabbatical_relocate) is imported READ-ONLY and called unchanged; the
   relocation count is measured by position diff around the call.

Reused unchanged from ``run_refill_gaussian`` (v1): ``GaussianRefillModel``
(refill regimes), ``LANDSCAPE``, ``SEEDS``, flow accounting.  The v1 fixed
parameters are pinned explicitly (``FIXED_V2``); only ``step_limit`` moves
(400 -> 800), and one H=400 control cell re-runs the v1 archive bit-exactly
(``proof_cross_round``) to prove the extended code is dynamics-inert where
the brief's comparisons live.

Tracking metrics are computed from pure observation (no RNG draws), so a
``p_reloc = 0.0`` run is bit-identical to v1's untracked run -- proven, not
assumed, by ``proof_cross_round``.

Design (per brief)
------------------
- Modes: ``gaussian_s3``, ``gaussian_s6``, ``frontier``.
- Sabbatical: ``p_reloc in {0.0, 0.02}`` (0.0 = baseline + control arm).
- ``c in {0.3, 0.5, 0.7}``, ``g = 0.1`` fixed, landscape ``mgstd5``, 20 seeds.
- Horizon ``H = 800`` per run; ``H = 400`` and ``H = 800`` windows reported
  from the same trajectory (``step_limit`` feeds reporting sentinels only).

Checkpointing (ADR-0001): per-cell atomic CSV + state JSON, restart-safe;
per-cell trajectory file (``traj/``) for the centroid paths.  ``--probe``
runs: identity (inert baseline vs plain ``MyModel``), conservation,
binding (p_reloc and the tracking metrics both move), schema (worker-written
file carries the analysis columns), the cross-round H=400 control cell, and
timing.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
IV_DIR = REPO / "scripts" / "interventions"
if str(IV_DIR) not in sys.path:
    sys.path.insert(0, str(IV_DIR))

from joblib import Parallel, delayed

import run_refill_gaussian as v1
from run_interventions_v1 import InterventionsModel  # sabbatical mechanism, read-only

OUT = REPO / "data" / "refill_gaussian_v2"
CELLS_DIR = OUT / "cells"
STATE_DIR = OUT / "state"
TRAJ_DIR = OUT / "traj"
PROBE_DIR = OUT / "probe"
for _d in (OUT, CELLS_DIR, STATE_DIR, TRAJ_DIR, PROBE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- constants
SEEDS = v1.SEEDS                      # 0..19, same seed identity as v1
STEPS = 800                           # full horizon; H=400 is a window
H400 = 400
LATE_WINDOW = v1.LATE_WINDOW          # 50
CURIOSITIES = (0.3, 0.5, 0.7)
GENERATION_RATES = (0.1,)             # brief: g = 0.1 fixed
RELOC_LEVELS = (0.0, 0.02)            # matches run_interventions_v1.RELOC_LEVELS
MODES_V2 = ("gaussian_s3", "gaussian_s6", "frontier")
TOP_FRACTION = 0.05                   # top-5% knowledge mass
V1_CONTROL_CELL = dict(
    refill_mode="gaussian_s3", initial_curiosity=0.5, agent_generation_time=0.1,
)

FIXED_V2 = dict(v1.FIXED, step_limit=STEPS)

TRACKING_COLUMNS = [
    "centroid_path_len_400", "centroid_path_len_800",
    "centroid_net_disp_400", "centroid_net_disp_800",
    "centroid_directedness_800",
    "mean_agent_peak_dist_400", "mean_agent_peak_dist_800",
    "final_agent_peak_dist",
    "top5_dispersion_mean_400", "top5_dispersion_mean_800",
    "top5_dispersion_last_400", "top5_dispersion_last_800",
    "centroid_resultant_mean",
]
MIRROR_COLUMNS = [
    "h400_cumulative_harvest_flow", "h400_late_window_flow_rate",
    "h400_explored_percentage_final", "h400_corr_prestige_ideal_merit_final",
]
REQUIRED_COLUMNS_V2 = v1.REQUIRED_COLUMNS + [
    "p_reloc", "sabbatical_relocations", "step_limit",
] + TRACKING_COLUMNS + MIRROR_COLUMNS

# Pre-declared verdict rule (traveling frontier vs flat diffusion), tested
# for satisfiability in the probe before the grid launches.  Units: grid
# tiles (20x20 torus; maximal minimal-image distance ~14.1).
VERDICT_RULE = {
    "directedness": "centroid_net_disp_800 / centroid_path_len_800",
    "traveling_frontier_if": {
        "directedness_at_least": 0.5,
        "net_disp_800_at_least": 2.0,
        "dispersion_growth_at_most": 1.25,
        "agent_peak_dist_growth_at_most": 0.5,
    },
    "flat_diffusion_if": {
        "directedness_at_most": 0.2,
        "or_net_disp_below_2_with_dispersion_growth_at_least": 1.5,
    },
    "otherwise": "mixed / wandering",
}


# --------------------------------------------------------------- torus math
def _torus_min_delta(a, b, n):
    d = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) % n
    return np.minimum(d, n - d)


def _torus_dist(a, b, n):
    d = _torus_min_delta(a, b, n)
    return float(np.sqrt((d * d).sum()))


def _torus_centroid(positions, weights, n):
    """Mass-weighted torus mean via per-axis angle averaging.

    Returns (centroid_xy, mean_resultant_length).  The resultant length is
    a quality flag: near zero means the mass is spread around the ring and
    the centroid position is noisy.
    """
    pos = np.asarray(positions, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    wsum = w.sum()
    if wsum <= 0:
        raise ValueError("degenerate top-5% mass")
    out, res = [], []
    for axis in (0, 1):
        theta = 2.0 * np.pi * pos[:, axis] / n
        mx = float((w * np.cos(theta)).sum() / wsum)
        my = float((w * np.sin(theta)).sum() / wsum)
        ang = np.arctan2(my, mx) % (2.0 * np.pi)
        out.append(ang * n / (2.0 * np.pi))
        res.append(float(np.hypot(mx, my)))
    return out, float(np.mean(res))


# --------------------------------------------------------------- the model
class GaussianSabbaticalModel(v1.GaussianRefillModel):
    """``GaussianRefillModel`` + P-0007 sabbatical shocks + peak tracking.

    - ``p_reloc`` (0.0 default, inert BY CONSTRUCTION: the relocation hook is
      skipped entirely at 0.0, so no RNG draw leaves the model's stream) --
      the mechanism itself is ``InterventionsModel._sabbatical_relocate``,
      called unchanged (single home: the interventions round).
    - The tracker is pure observation: it reads ``knowledge`` and agent
      positions after each step and draws nothing.
    """

    def __init__(self, *, p_reloc: float = 0.0, **kwargs):
        if p_reloc not in RELOC_LEVELS:
            raise ValueError(f"p_reloc must be one of {RELOC_LEVELS}, got {p_reloc!r}")
        if kwargs.get("dyadic_mode", "off") != "off":
            raise ValueError("refill_gaussian_v2 runs with dyadic_mode='off'")
        self.p_reloc = float(p_reloc)
        self._run_seed = int(kwargs.get("agent_seed", 0))  # not stored by MyModel
        self.sabbatical_relocations = 0
        super().__init__(**kwargs)

        n = self.size
        self._top_k = max(1, int(round(TOP_FRACTION * n * n)))
        self._traj_rows: list[dict] = []
        self._tr_path = 0.0
        self._tr_agent_sum = 0.0
        self._tr_disp_sum = 0.0
        self._tr_resultant_sum = 0.0
        self._tr_steps = 0
        self._snap400: dict | None = None
        c0 = self._track_observe(step=0)
        self._c_prev = c0[:2]

    # ---- tracking (reporting-only: no RNG) -----------------------------
    def _track_observe(self, step: int):
        n = self.size
        knowledge = self.grid.properties["knowledge"].data
        flat = np.asarray(knowledge, dtype=np.float64).ravel()
        idx = np.argsort(-flat, kind="stable")[: self._top_k]
        rows, cols = np.divmod(idx, n)
        weights = flat[idx]
        centroid, resultant = _torus_centroid(
            np.column_stack([rows, cols]), weights, n
        )
        # dispersion: mass-weighted RMS minimal-image radius of the top set
        d = _torus_min_delta(np.column_stack([rows, cols]), centroid, n)
        disp = float(np.sqrt((weights * (d * d).sum(axis=1)).sum() / weights.sum()))
        # agent-to-peak distance
        apos = [a.pos for a in self._scientists]
        if apos:
            ad = _torus_min_delta(np.asarray(apos, dtype=np.float64), centroid, n)
            agent_dist = float(np.sqrt((ad * ad).sum(axis=1)).mean())
        else:
            agent_dist = float("nan")
        self._traj_rows.append(dict(
            agent_seed=self._run_seed, step=step,
            cx=centroid[0], cy=centroid[1],
            agent_peak_dist=agent_dist, dispersion=disp, resultant=resultant,
        ))
        return centroid[0], centroid[1], agent_dist, disp, resultant

    def _track_step(self):
        # rows so far = 1 (step-0 snapshot) + completed steps -> label = len
        cx, cy, agent_dist, disp, resultant = self._track_observe(
            step=len(self._traj_rows))
        self._tr_path += _torus_dist((cx, cy), self._c_prev, self.size)
        self._c_prev = (cx, cy)
        self._tr_agent_sum += agent_dist
        self._tr_disp_sum += disp
        self._tr_resultant_sum += resultant
        self._tr_steps += 1
        if self._tr_steps == H400 and self._snap400 is None:
            self._snap400 = self._snapshot()

    def _snapshot(self) -> dict:
        steps = max(self._tr_steps, 1)
        cx0, cy0 = self._traj_rows[0]["cx"], self._traj_rows[0]["cy"]
        return dict(
            path=self._tr_path,
            net=_torus_dist(self._c_prev, (cx0, cy0), self.size),
            agent_mean=self._tr_agent_sum / steps,
            disp_mean=self._tr_disp_sum / steps,
            disp_last=self._traj_rows[-1]["dispersion"],
            resultant_mean=self._tr_resultant_sum / steps,
        )

    # ---- step: sabbatical hook (v1 interventions order) then dynamics --
    def step(self, endupdate=True):
        if self.p_reloc > 0.0:
            before = [a.pos for a in self._scientists]
            InterventionsModel._sabbatical_relocate(self)
            self.sabbatical_relocations += sum(
                1 for a, b in zip(self._scientists, before) if a.pos != b
            )
        super().step(endupdate=endupdate)
        self._track_step()

    # ---- analysis payload ---------------------------------------------
    def run_mode_for_bulk(self, longitudinal=False):
        row = super().run_mode_for_bulk(longitudinal=longitudinal)
        if longitudinal:
            return row
        assert self._snap400 is not None, "H400 snapshot missing"
        snap4, snap8 = self._snap400, self._snapshot()
        path8, net8 = snap8["path"], snap8["net"]
        row["p_reloc"] = float(self.p_reloc)
        row["sabbatical_relocations"] = int(self.sabbatical_relocations)
        row["step_limit"] = int(self.step_limit)
        row["centroid_path_len_400"] = float(snap4["path"])
        row["centroid_path_len_800"] = float(path8)
        row["centroid_net_disp_400"] = float(snap4["net"])
        row["centroid_net_disp_800"] = float(net8)
        row["centroid_directedness_800"] = float(net8 / path8) if path8 > 0 else 0.0
        row["mean_agent_peak_dist_400"] = float(snap4["agent_mean"])
        row["mean_agent_peak_dist_800"] = float(snap8["agent_mean"])
        row["final_agent_peak_dist"] = float(self._traj_rows[-1]["agent_peak_dist"])
        row["top5_dispersion_mean_400"] = float(snap4["disp_mean"])
        row["top5_dispersion_mean_800"] = float(snap8["disp_mean"])
        row["top5_dispersion_last_400"] = float(snap4["disp_last"])
        row["top5_dispersion_last_800"] = float(snap8["disp_last"])
        row["centroid_resultant_mean"] = float(snap8["resultant_mean"])

        # H=400 mirrors of the v1 flow/level channels (cross-round tables)
        hist = self._harvest_flow_history
        assert len(hist) == self.step_limit, (len(hist), self.step_limit)
        row["h400_cumulative_harvest_flow"] = float(hist[H400 - 1])
        row["h400_late_window_flow_rate"] = float(
            (hist[H400 - 1] - hist[H400 - 1 - LATE_WINDOW]) / LATE_WINDOW
        )
        mv = self.datacollector.get_model_vars_dataframe()
        n_mv = len(mv)
        assert n_mv in (self.step_limit, self.step_limit + 1), n_mv
        idx400 = H400 - 1 if n_mv == self.step_limit else H400
        row["h400_explored_percentage_final"] = float(
            mv["explored_percentage"].iloc[idx400]
        )
        row["h400_corr_prestige_ideal_merit_final"] = float(
            mv["Corr_prestige_ideal_merit"].iloc[idx400]
        )
        return row


# ------------------------------------------------------------------ cells
def cell_list() -> list[dict]:
    cells = []
    for mode in MODES_V2:
        for reloc in RELOC_LEVELS:
            for curiosity in CURIOSITIES:
                cells.append(dict(
                    refill_mode=mode,
                    p_reloc=reloc,
                    initial_curiosity=curiosity,
                    agent_generation_time=0.1,
                ))
    assert len(cells) == len(MODES_V2) * len(RELOC_LEVELS) * len(CURIOSITIES)
    return cells


def cell_tag(cell: dict) -> str:
    c = int(round(cell["initial_curiosity"] * 10))
    r = int(round(cell["p_reloc"] * 100))
    return f"cell_refill-{cell['refill_mode']}_c{c:02d}_gen01_reloc{r:02d}_mgstd5"


def params_for(cell: dict, seed: int, steps: int = STEPS) -> dict:
    _name, fn, gparams = v1.LANDSCAPE
    p = dict(FIXED_V2)
    p["step_limit"] = steps
    p.update(
        cell_init_function=fn,
        generation_params=dict(gparams),
        initial_curiosity=cell["initial_curiosity"],
        agent_generation_time=cell["agent_generation_time"],
        agent_seed=seed,
        refill_mode=cell["refill_mode"],
        p_reloc=cell["p_reloc"],
    )
    return p


# ------------------------------------------------------------ one run/probe
def run_one(cell: dict, seed: int, steps: int = STEPS,
            model_cls=None) -> tuple[str, dict | str, list | None]:
    """One seeded run: ``("ok", row, traj_rows)`` or ``("fail", tb, None)``."""
    cls = model_cls or GaussianSabbaticalModel
    params = params_for(cell, seed, steps=steps)
    if cls is v1.MyModel:
        params.pop("refill_mode", None)
        params.pop("p_reloc", None)
    try:
        model = cls(**params)
        row = model.run_mode_for_bulk()
        row["cell_tag"] = cell_tag(cell)
        row["landscape"] = v1.LANDSCAPE[0]
        traj = getattr(model, "_traj_rows", None)
        return "ok", row, traj
    except Exception:  # noqa: BLE001
        return "fail", traceback.format_exc(), None


def _numeric_diffs(a: dict, b: dict, keys) -> dict:
    out = {}
    for k in keys:
        va, vb = a.get(k), b.get(k)
        if isinstance(va, (int, float, np.floating, np.integer)) and isinstance(
            vb, (int, float, np.floating, np.integer)
        ):
            fa, fb = float(va), float(vb)
            if fa != fa and fb != fb:          # both NaN: equal by convention
                out[k] = 0.0
            elif fa != fa or fb != fb:         # one-sided NaN: never equal
                out[k] = float("inf")
            else:
                out[k] = float(abs(fa - fb))
        else:
            out[k] = 0.0 if va == vb else float("inf")
    return out


# ------------------------------------------------------------------ proofs
def proof_identity() -> dict:
    """Baseline knobs (off, p_reloc=0) must reproduce plain ``MyModel``."""
    results = []
    for seed in (0, 1, 2):
        cell = dict(refill_mode="off", p_reloc=0.0, initial_curiosity=0.5,
                    agent_generation_time=0.1)
        st_plain, plain, _ = run_one(cell, seed, model_cls=v1.MyModel)
        st_mine, mine, _ = run_one(cell, seed)
        assert st_plain == st_mine == "ok", (st_plain, st_mine)
        common = sorted(set(plain) & set(mine))
        diffs = _numeric_diffs(plain, mine, common)
        worst = max(diffs.values())
        results.append(dict(seed=seed, worst=worst, n_common=len(common),
                            driver_only=sorted(set(mine) - set(plain))))
        assert worst == 0.0, (
            f"baseline not bit-identical on seed {seed}: "
            f"{[k for k, v in diffs.items() if v != 0.0][:8]}"
        )
    return {"checks": len(results), "worst": 0.0,
            "driver_only_keys": results[0]["driver_only"]}


def proof_conservation() -> dict:
    rows = []
    for mode in ("gaussian_s3", "gaussian_s6"):
        for seed in (0, 1):
            st, row, _ = run_one(dict(refill_mode=mode, p_reloc=0.0,
                                      initial_curiosity=0.5,
                                      agent_generation_time=0.1), seed)
            assert st == "ok", (mode, seed, st)
            total_err = abs(row["final_total_knowledge"] - 1.0)
            flow_err = abs(row["cumulative_refill_added"]
                           - row["cumulative_harvest_flow"])
            rows.append(dict(mode=mode, seed=seed, total_error=total_err,
                             flow_error=flow_err))
            assert total_err < 1e-9 and flow_err < 1e-9, (mode, seed, total_err)
    return {"checks": len(rows),
            "max_total_error": max(r["total_error"] for r in rows),
            "max_flow_error": max(r["flow_error"] for r in rows)}


def proof_binding() -> dict:
    """p_reloc must move outcomes; the tracking metrics must move with the
    refill regime (sigma) -- the outcome-layer probe for both new channels."""
    out = {}
    base = dict(refill_mode="frontier", p_reloc=0.0, initial_curiosity=0.5,
                agent_generation_time=0.1)
    st0, b, _ = run_one(base, 0)
    assert st0 == "ok"

    st, v, _ = run_one(dict(base, p_reloc=0.02), 0)
    assert st == "ok"
    keys = [k for k in b if isinstance(b.get(k), (int, float, np.floating, np.integer))]
    diffs = _numeric_diffs(b, v, keys)
    moved = sorted(k for k, d in diffs.items() if d > 0.0)
    assert moved, "binding failed: p_reloc moved nothing"
    assert v["sabbatical_relocations"] > 0, "p_reloc > 0 but no relocation counted"
    out["p_reloc"] = {"n_moved": len(moved), "sample": moved[:6],
                      "relocations": v["sabbatical_relocations"]}

    st3, r3, _ = run_one(dict(base, refill_mode="gaussian_s3"), 0)
    st6, r6, _ = run_one(dict(base, refill_mode="gaussian_s6"), 0)
    assert st3 == st6 == "ok"
    tdiffs = _numeric_diffs(r3, r6, TRACKING_COLUMNS)
    tmoved = sorted(k for k, d in tdiffs.items() if d > 0.0)
    assert tmoved, "binding failed: tracking metrics identical across sigma"
    out["tracking_vs_sigma"] = {"n_moved": len(tmoved), "sample": tmoved[:6],
                                "max": max(tdiffs.values())}
    return out


def proof_cross_round() -> dict:
    """Control cell: one v1 cell re-run at H=400 with tracking ON must match
    ``data/refill_gaussian_v1`` bit-exactly on every shared numeric column.

    This proves (a) the tracker is observation-only, (b) p_reloc=0 is inert,
    and (c) the post-v1 code (ADR-0003 clock, commit 8b36a72) did not move
    these channels.  Structural columns are compared by equality; the cell
    tag differs BY CONSTRUCTION (v2 tags carry the reloc axis) and is the
    only permitted structural difference.
    """
    v1_path = (REPO / "data" / "refill_gaussian_v1" / "cells"
               / f"{v1.cell_tag(V1_CONTROL_CELL)}.csv")
    with open(v1_path, newline="") as fh:
        stored = {int(r["agent_seed"]): r for r in csv.DictReader(fh)}
    assert len(stored) == len(SEEDS), (len(stored), len(SEEDS))

    # ADR-0003 (commit 8b36a72) re-scaled the age bookkeeping (1 step =
    # 0.5 yr, age += 0.5, entry age 25).  ``mean_age`` therefore differs
    # against the pre-ADR-0003 v1 archive BY CONSTRUCTION -- measured, not
    # assumed: a full per-column ledger shows mean_age as the ONLY moved
    # column (see probe_results.json -> cross_round.drift_ledger).  Every
    # dynamics channel the v2 report rests on must stay bit-identical.
    AGE_CLOCK_DRIFT_OK = {"mean_age"}

    rows = Parallel(n_jobs=3, return_as="list")(
        delayed(run_one)(dict(V1_CONTROL_CELL, p_reloc=0.0), s, steps=H400)
        for s in SEEDS
    )
    worst = 0.0
    worst_key = None
    structural_diff = []
    drift = {}
    n_compared = 0
    for seed, (status, row, _traj) in zip(SEEDS, rows):
        assert status == "ok", (seed, row)
        ref = stored[seed]
        # internal consistency at H=400: mirrors equal full-horizon values
        assert row["h400_cumulative_harvest_flow"] == row["cumulative_harvest_flow"]
        assert row["h400_late_window_flow_rate"] == row["late_window_flow_rate"]
        for k in sorted(set(ref) & set(row)):
            va, vb = ref[k], row[k]
            if k in ("cell_tag", "landscape") or not _is_number(va) or not _is_number(vb):
                if k == "cell_tag":
                    continue                      # differs by construction
                if str(va) != str(vb):
                    structural_diff.append((seed, k, va, vb))
                continue
            fa, fb = float(va), float(vb)
            if fa != fa and fb != fb:
                d = 0.0                     # both NaN: equal by convention
            elif fa != fa or fb != fb:
                d = float("inf")            # one-sided NaN: never equal
            else:
                d = abs(fa - fb)
            if k in AGE_CLOCK_DRIFT_OK:
                drift[k] = max(drift.get(k, 0.0), d)
                continue
            n_compared += 1
            if d > worst:
                worst, worst_key = d, k
    assert not structural_diff, f"structural mismatch: {structural_diff[:5]}"
    assert set(drift) <= AGE_CLOCK_DRIFT_OK, f"unexpected drift: {set(drift)}"
    assert worst == 0.0, (
        f"cross-round control NOT bit-exact outside the age clock: "
        f"worst {worst} at {worst_key}"
    )
    return {"cell": v1.cell_tag(V1_CONTROL_CELL), "seeds": len(SEEDS),
            "numeric_columns_compared": n_compared, "worst": worst,
            "worst_key": worst_key, "structural_ok": True,
            "age_clock_drift_ok": sorted(AGE_CLOCK_DRIFT_OK),
            "drift_ledger": drift}


def _is_number(x) -> bool:
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


def proof_schema(cell: dict, n_jobs: int) -> dict:
    rows = Parallel(n_jobs=n_jobs, return_as="list")(
        delayed(run_one)(cell, s) for s in SEEDS[:4]
    )
    ok = [r for st, r, _t in rows if st == "ok"]
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
    missing = [c for c in REQUIRED_COLUMNS_V2 if c not in (back[0] if back else {})]
    assert not missing, f"worker-written file missing columns: {missing}"
    assert len(back) == 4
    return {"rows": len(back), "columns": len(back[0]), "missing": missing}


def probe_timing() -> dict:
    out = {}
    for mode in MODES_V2:
        t0 = time.time()
        st, _, _ = run_one(dict(refill_mode=mode, p_reloc=0.02,
                                initial_curiosity=0.5, agent_generation_time=0.1), 0)
        out[mode] = {"wall_s": round(time.time() - t0, 2), "status": st}
    return out


# --------------------------------------------------------------- the sweep
def run_cell(cell: dict, n_jobs: int, log) -> dict:
    tag = cell_tag(cell)
    csv_path = CELLS_DIR / f"{tag}.csv"
    traj_path = TRAJ_DIR / f"{tag}.csv"
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
    rows, traj_rows, errors = [], [], {}
    for seed, (status, payload, traj) in zip(SEEDS, results):
        if status == "ok":
            rows.append(payload)
            traj_rows.extend(traj)
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
        assert n_back == len(rows), (tag, n_back, len(rows))
        os.replace(tmp, csv_path)
    if traj_rows:
        expected_traj = len(rows) * (STEPS + 1)   # step-0 snapshot + STEPS
        tmp_t = traj_path.with_suffix(".csv.tmp")
        with open(tmp_t, "w", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["agent_seed", "step", "cx", "cy",
                                "agent_peak_dist", "dispersion", "resultant"])
            writer.writeheader()
            writer.writerows(traj_rows)
        with open(tmp_t, newline="") as fh:
            n_back = sum(1 for _ in csv.DictReader(fh))
        assert n_back == expected_traj, (tag, n_back, expected_traj)
        os.replace(tmp_t, traj_path)
    entry = {
        "cell_tag": tag,
        "status": status,
        "params": dict(cell),
        "seeds_ok": len(rows),
        "seeds_failed": len(errors),
        "rows": len(rows),
        "rows_expected": n_expect,
        "traj_rows": len(traj_rows),
        "errors": errors,
        "csv": csv_path.name if rows else None,
        "traj": traj_path.name if traj_rows else None,
        "wall_s": round(time.time() - t0, 2),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    tmp_state = state_path.with_suffix(".json.tmp")
    tmp_state.write_text(json.dumps(entry, indent=1))
    os.replace(tmp_state, state_path)
    log(f"CELL {tag} -> {status} seeds_ok={entry['seeds_ok']} "
        f"rows={entry['rows']}/{n_expect} wall={entry['wall_s']}s"
        + (f" ERRORS={list(errors)}" if errors else ""))
    return entry


def preflight_snapshot(path: Path) -> None:
    try:
        import subprocess
        out = subprocess.run(["ps", "aux"], capture_output=True, text=True).stdout
        heavy = [ln for ln in out.splitlines()
                 if any(k in ln for k in ("python", "julia")) and "grep" not in ln]
        path.write_text(f"{len(heavy)} live python/julia processes\n" + "\n".join(heavy))
    except Exception as exc:  # noqa: BLE001
        path.write_text(f"snapshot failed: {exc}")


def source_hashes() -> dict:
    files = [
        REPO / "sciencemodels" / f for f in
        sorted(p.name for p in (REPO / "sciencemodels").glob("*.py"))
    ] + [
        SCRIPT_DIR / "run_refill_gaussian.py",
        SCRIPT_DIR / "run_interventions_v1.py" if (SCRIPT_DIR / "run_interventions_v1.py").exists()
        else REPO / "scripts" / "interventions" / "run_interventions_v1.py",
        Path(__file__).resolve(),
    ]
    out = {}
    for f in files:
        out[str(f.relative_to(REPO))] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=3)   # cgroup quota = 3 cores
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
        log(f"PROBE identity ok: {ident}")
        cons = proof_conservation()
        log(f"PROBE conservation ok: {cons}")
        binding = proof_binding()
        log(f"PROBE binding ok: {binding}")
        schema = proof_schema(cell_list()[0], args.n_jobs)
        log(f"PROBE schema ok: {schema}")
        control = proof_cross_round()
        log(f"PROBE cross-round control ok: {control}")
        timing = probe_timing()
        log(f"PROBE timing: {timing}")
        results = {"identity": ident, "conservation": cons, "binding": binding,
                   "schema": schema, "cross_round": control, "timing": timing,
                   "verdict_rule": VERDICT_RULE}
        (PROBE_DIR / "probe_results.json").write_text(
            json.dumps(results, indent=1, default=str))
        print(json.dumps(results, indent=1, default=str))
        return 0

    t_start = time.time()
    cells = cell_list()
    design = {
        "schema": "refill_gaussian_v2",
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "grid": f"cells={len(cells)} seeds={len(SEEDS)} steps={STEPS}",
        "modes": list(MODES_V2),
        "reloc_levels": list(RELOC_LEVELS),
        "curiosities": list(CURIOSITIES),
        "generation_rates": list(GENERATION_RATES),
        "fixed": FIXED_V2,
        "landscape": {"name": v1.LANDSCAPE[0], "fn": v1.LANDSCAPE[1].__name__,
                      "params": v1.LANDSCAPE[2]},
        "n_jobs": args.n_jobs,
        "model_boundary": ("GaussianSabbaticalModel subclasses v1's "
                           "GaussianRefillModel; sabbatical mechanism imported "
                           "unchanged from scripts/interventions/"
                           "run_interventions_v1.py (InterventionsModel"
                           "._sabbatical_relocate); no model file edited"),
        "tracking": ("top-5% knowledge-mass torus centroid per step; path "
                     "length (sum of minimal-image step moves), net "
                     "displacement from step-0 centroid, agent-to-peak "
                     "(minimal-image) distance, top-5% weighted RMS "
                     "dispersion; windows H=400 and H=800 from the same "
                     "trajectory; pure observation, no RNG draws"),
        "horizons": {"full": STEPS, "mirror_window": H400,
                     "note": "step_limit feeds reporting sentinels only; one "
                             "trajectory per run serves both windows"},
        "verdict_rule": VERDICT_RULE,
        "source_sha256": source_hashes(),
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
    line = (f"GROSS cells={len(cells)} runs={gross} | NET ok_cells={n_ok} "
            f"ok_with_failures={n_part} crashed={n_crash} "
            f"runs_ok={runs_ok} runs_failed={runs_fail} wall={wall:.1f}s")
    log(line)
    print(line)
    return 0 if n_crash == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
