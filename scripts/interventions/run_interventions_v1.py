"""Interventions v1 driver -- sciencemodels, P-0002 / P-0003 / P-0004.

Three driver-side probes of the counterforce mechanisms CONT1 filed, all
implemented as a SUBCLASS of ``MyModel`` so no model file is touched
(owner boundary; the pattern follows ``scripts/prestige_stock/``):

  Arm A  P-0002 spawn dispersal     r_spawn   in {exact, ring1, random}
  Arm B  P-0003 institutional       alpha     in {1.0, 0.5, 0.0}
         counterforces               p_reloc  in {0.0, 0.02}
                                     visibility weight 1-c in {0.5, 0.7, 0.9}
  Arm C  P-0004 paradigm refills    refill_mode in {off, occupied, frontier}

Grid (owner brief): A = 3x3x2 = 18 cells, B = 3x2x3x2 = 36 cells,
C = 3x3x2 = 18 cells; 72 cells x 20 seeds = 1,440 runs; H = 200,
n_agents = 20, grid 20x20, g = 0.1 except Arm C which sweeps g in {0, 0.1}.

Design notes:
- Seeds: one knob (agent_seed) drives the agent RNG and the model injects it
  into generation_params, so landscape and agent draws vary together (T1
  convention). 20 seeds = 20 values of that knob; seed identity also reaches
  the CSV as the ``agent_seed`` report key.
- Landscapes mirror ``scripts/sensitivity/driver_source.py`` exactly:
  mgstd5   = multipleGaussians {std:5, number_gaussians:2}
  noisypr5 = noisyGaussian {prop_random:5, std:5, std_curiosity:0.1}
- Fixed: epsilon 0.1, harvest 0.02, use_distance True, vanishing 0.9 (the
  Path-0 unified default), use_visibility_reward True, knowledge
  normalization "share", prestige_accrual "harvested", entry_prestige_stock
  "prestige", dyadic_mode "off" (Path B gated off), agent_generation_function
  = model default (identity: curiosity == initial_curiosity).
- Checkpoint per cell into ``cells/`` with an atomic replace (tmp -> row-count
  check -> os.replace), plus a state JSON per cell; a restart skips verified
  units only.  A failing seed is recorded, never dropped silently.
- Proofs before the grid (``--probe``): bit-identity of the baseline
  configuration against plain ``MyModel``, binding of every knob, schema of a
  WORKER-written cell file, and per-run timing.
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
from sciencemodels.agents import Scientist
from sciencemodels.model import MyModel

OUT = REPO / "data" / "interventions_v1"
CELLS_DIR = OUT / "cells"
STATE_DIR = OUT / "state"
PROBE_DIR = OUT / "probe"
for d in (OUT, CELLS_DIR, STATE_DIR, PROBE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- constants
SEEDS = list(range(20))
STEPS = 200
ENTRY_G = 0.1

SPAWN_LEVELS = ("exact", "ring1", "random")
ALPHA_LEVELS = (1.0, 0.5, 0.0)
RELOC_LEVELS = (0.0, 0.02)
REFILL_LEVELS = ("off", "occupied", "frontier")
VIS_WEIGHTS = (0.5, 0.7, 0.9)          # Arm B sweeps 1 - c

LANDSCAPES = {
    "mgstd5": (L.multipleGaussians, {"std": 5, "number_gaussians": 2}),
    "noisypr5": (L.noisyGaussian, {"prop_random": 5, "std": 5,
                                   "std_curiosity": 0.1}),
}

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

# Columns every analysis step may rely on (schema proof asserts these).
REQUIRED_COLUMNS = [
    "cell_tag", "arm", "agent_seed",
    "r_spawn", "alpha", "p_reloc", "refill_mode",
    "initial_curiosity", "visibility_weight", "landscape",
    "agent_generation_time",
    "corr_prestige_ideal_merit_final", "Mean_of_Corr_prestige_ideal_merit",
    "corr_prestige_local_merit_final", "Mean_of_Corr_prestige_local_merit",
    "top10pct_prestige_share", "top10_final",
    "ideal_merit_sum_final", "ideal_merit_cum_final",
    "Mean_of_Ideal_merit_cum",
    "percentage_knowledge_harvested", "explored_percentage",
    "newcomer_gap", "n_newcomers", "n_incumbents",
]


# ------------------------------------------------------------ the subclass
class InterventionsModel(MyModel):
    """``MyModel`` plus the three intervention knobs.

    Every knob is inert at its baseline level, which is what makes the
    bit-identity proof possible: ``r_spawn="exact"`` returns ``sup.pos``
    without drawing from the RNG, ``alpha=1.0`` reproduces ``p ** 1.0 == p``
    exactly, ``p_reloc=0.0`` short-circuits the relocation loop, and
    ``refill_mode="off"`` leaves ``new_questions`` False.
    """

    def __init__(self, *, r_spawn: str = "exact", alpha: float = 1.0,
                 p_reloc: float = 0.0, refill_mode: str = "off", **kwargs):
        if r_spawn not in SPAWN_LEVELS:
            raise ValueError(f"r_spawn must be one of {SPAWN_LEVELS}, got {r_spawn!r}")
        if alpha not in ALPHA_LEVELS:
            raise ValueError(f"alpha must be one of {ALPHA_LEVELS}, got {alpha!r}")
        if p_reloc not in RELOC_LEVELS:
            raise ValueError(f"p_reloc must be one of {RELOC_LEVELS}, got {p_reloc!r}")
        if refill_mode not in REFILL_LEVELS:
            raise ValueError(f"refill_mode must be one of {REFILL_LEVELS}, got {refill_mode!r}")
        self.r_spawn = r_spawn
        self.alpha = float(alpha)
        self.p_reloc = float(p_reloc)
        self.refill_mode = refill_mode
        # Arm C owns the new_questions channel: off = model baseline,
        # occupied/frontier = the model's own flag switched on.
        kwargs["new_questions"] = refill_mode != "off"
        super().__init__(**kwargs)

    # ---- P-0003.1 supervisor roulette exponent + P-0002 spawn placement
    def _MyModel__generate_new_agents(self):
        """The model's newcomer path with two knobs.

        Copied from ``model.py`` verbatim except (a) the roulette weight is
        ``prestige ** alpha`` and (b) the birth tile comes from
        ``_spawn_tile``.  At ``alpha = 1.0`` the weights are bit-identical to
        the model's raw prestige values, so the whole RNG stream matches.
        """
        if self.agent_generation_time > 0:
            agents_to_generate = int(
                np.floor(self.steps * self.agent_generation_time)
                - np.floor((self.steps - 1) * self.agent_generation_time)
            )
            entry_stock = self.entry_prestige_stock
            weights = [getattr(a, entry_stock) ** self.alpha
                       for a in self._scientists]
            total_weight = sum(weights)
            supervisors = []
            for _ in range(agents_to_generate):
                val = self.rng.uniform(0, total_weight)
                for w, a in zip(weights, self._scientists):
                    if val <= w:
                        supervisors.append(a)
                        break
                    val -= w
            for sup in supervisors:
                w = self.agent_generation_function(
                    self.initial_curiosity, self.rng, self.generation_params
                )
                a = Scientist(self, w, self.epsilon)
                a.ideal_merit_cum = np.float64(0.0)
                a.is_newcomer = True
                self.grid.place_agent(a, self._spawn_tile(sup))
                self._scientists.append(a)
            for agent in self._scientists:
                self.grid.properties["explored"].data[agent.pos] = True

    def _spawn_tile(self, sup):
        """Birth tile: exact (baseline), ring1 neighbour, or uniform tile."""
        if self.r_spawn == "exact":
            return sup.pos                      # no RNG draw: bit-identical
        if self.r_spawn == "ring1":
            x, y = sup.pos
            dx, dy = ((1, 0), (-1, 0), (0, 1), (0, -1))[
                int(self.rng.integers(0, 4))]
            return ((x + dx) % self.size, (y + dy) % self.size)
        return (int(self.rng.integers(0, self.size)),
                int(self.rng.integers(0, self.size)))

    # ---- P-0003.2 sabbatical / transient-diversity relocation
    def step(self, endupdate=True):
        if self.p_reloc > 0.0:
            self._sabbatical_relocate()
        return super().step(endupdate=endupdate)

    def _sabbatical_relocate(self):
        """Relocate trapped agents to a random unoccupied tile.

        "Trapped" is operationalised as standing on a tile below the model's
        own depletion threshold (knowledge < 0.025, the same constant the
        refill rule uses): that is where the strict-improvement rule
        (agents.py:166) pins an agent.  Each trapped agent gets one draw at
        ``p_reloc`` per step.
        """
        knowledge = self.grid.properties["knowledge"].data
        explored = self.grid.properties["explored"].data
        for a in list(self._scientists):
            if knowledge[a.pos] >= 0.025:
                continue                        # standing on live ground
            if self.rng.random() >= self.p_reloc:
                continue
            free = np.argwhere(~explored)
            if len(free) == 0:
                continue
            idx = int(self.rng.integers(len(free)))
            self.new_place(a, (int(free[idx][0]), int(free[idx][1])))

    # ---- P-0004 paradigm refill regimes
    def _generate_new_questions(self):
        if self.refill_mode == "off":
            return                             # new_questions is False anyway
        if self.refill_mode == "occupied":
            return super()._generate_new_questions()   # the model's own rule
        # frontier: each depleted OCCUPIED tile spawns its new question on a
        # random UNOCCUPIED tile, so the herd is not bailed out in place
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
            knowledge[x, y] = self.rng.random()

    # ---- reporting: intervention keys + the newcomer-incumbent gap
    def run_mode_for_bulk(self, longitudinal=False):
        row = super().run_mode_for_bulk(longitudinal=longitudinal)
        if longitudinal:
            return row
        new = [a for a in self._scientists if getattr(a, "is_newcomer", False)]
        inc = [a for a in self._scientists
               if not getattr(a, "is_newcomer", False)]
        row["n_newcomers"] = len(new)
        row["n_incumbents"] = len(inc)
        if new and inc:
            row["newcomer_prestige_mean"] = float(
                np.mean([a.prestige for a in new]))
            row["incumbent_prestige_mean"] = float(
                np.mean([a.prestige for a in inc]))
            row["newcomer_gap"] = (row["newcomer_prestige_mean"]
                                   - row["incumbent_prestige_mean"])
        else:
            row["newcomer_prestige_mean"] = float("nan")
            row["incumbent_prestige_mean"] = float("nan")
            row["newcomer_gap"] = float("nan")
        row["r_spawn"] = self.r_spawn
        row["alpha"] = self.alpha
        row["p_reloc"] = self.p_reloc
        row["refill_mode"] = self.refill_mode
        # Terminal values: build_report only exposes run means (Mean_of_*),
        # so the final step -- the statistic every earlier round reported --
        # is read off the step-level frame here.
        frame = self.datacollector.get_model_vars_dataframe()
        last = frame.iloc[-1]
        row["corr_prestige_ideal_merit_final"] = float(
            last["Corr_prestige_ideal_merit"])
        row["corr_prestige_local_merit_final"] = float(
            last["Corr_prestige_local_merit"])
        row["top10_final"] = float(last["top10pct_prestige_share"])
        row["ideal_merit_cum_final"] = float(last["Ideal_merit_cum"])
        row["explored_final"] = float(last["explored_percentage"])
        # terminal cross-section of the yardstick (agents born on the last
        # step have no ideal_Merit yet: they are skipped, not zeroed)
        vals = [a.ideal_Merit for a in self.agents
                if hasattr(a, "ideal_Merit")]
        row["ideal_merit_sum_final"] = (float(np.sum(vals)) if vals
                                        else float("nan"))
        return row


# ------------------------------------------------------------------ cells
def cell_list() -> list[dict]:
    """The 72 cells of the owner brief (A 18 + B 36 + C 18)."""
    cells = []
    for spawn in SPAWN_LEVELS:                       # Arm A
        for c in (0.3, 0.5, 0.7):
            for land in ("mgstd5", "noisypr5"):
                cells.append(dict(arm="A", r_spawn=spawn, initial_curiosity=c,
                                  landscape=land, alpha=1.0, p_reloc=0.0,
                                  refill_mode="off",
                                  agent_generation_time=ENTRY_G))
    for alpha in ALPHA_LEVELS:                       # Arm B
        for preloc in RELOC_LEVELS:
            for w in VIS_WEIGHTS:
                for land in ("mgstd5", "noisypr5"):
                    cells.append(dict(arm="B", alpha=alpha, p_reloc=preloc,
                                      visibility_weight=w,
                                      initial_curiosity=round(1.0 - w, 10),
                                      landscape=land, r_spawn="exact",
                                      refill_mode="off",
                                      agent_generation_time=ENTRY_G))
    for mode in REFILL_LEVELS:                       # Arm C
        for c in (0.3, 0.5, 0.7):
            for g in (0.0, ENTRY_G):
                cells.append(dict(arm="C", refill_mode=mode,
                                  initial_curiosity=c, agent_generation_time=g,
                                  landscape="mgstd5", r_spawn="exact",
                                  alpha=1.0, p_reloc=0.0))
    assert len(cells) == 18 + 36 + 18, f"expected 72 cells, got {len(cells)}"
    return cells


def cell_tag(cell: dict) -> str:
    parts = [f"cell{cell['arm']}"]
    if cell["arm"] == "A":
        parts.append(f"spawn-{cell['r_spawn']}")
    if cell["arm"] == "B":
        parts.append(f"a{cell['alpha']:g}")
        parts.append(f"reloc{int(round(cell['p_reloc'] * 100)):02d}")
        parts.append(f"w{int(round(cell['visibility_weight'] * 10)):02d}")
    if cell["arm"] == "C":
        parts.append(f"refill-{cell['refill_mode']}")
    parts.append(f"c{int(round(cell['initial_curiosity'] * 10)):02d}")
    if cell["arm"] == "C":
        parts.append(f"gen{int(round(cell['agent_generation_time'] * 10)):02d}")
    parts.append(cell["landscape"])
    return "_".join(parts)


def params_for(cell: dict, seed: int) -> dict:
    fname, gparams = LANDSCAPES[cell["landscape"]]
    p = dict(FIXED)
    p.update(
        cell_init_function=fname,
        generation_params=dict(gparams),
        initial_curiosity=cell["initial_curiosity"],
        agent_generation_time=cell["agent_generation_time"],
        agent_seed=seed,
        r_spawn=cell["r_spawn"],
        alpha=cell["alpha"],
        p_reloc=cell["p_reloc"],
        refill_mode=cell["refill_mode"],
    )
    return p


# ------------------------------------------------------------ one run/probe
def run_one(cell: dict, seed: int, model_cls=None, extra: dict | None = None):
    """One seeded run. Returns (status, row) -- never raises.

    ``model_cls=MyModel`` runs the UNPATCHED model: the four intervention
    kwargs are stripped (the plain class does not accept them) and ``extra``
    supplies whatever the comparison needs -- e.g. ``new_questions=True`` for
    the occupied-refill identity.
    """
    cls = model_cls or InterventionsModel
    params = params_for(cell, seed)
    if cls is MyModel:
        for k in ("r_spawn", "alpha", "p_reloc", "refill_mode"):
            params.pop(k, None)
    if extra:
        params.update(extra)
    try:
        model = cls(**params)
        row = model.run_mode_for_bulk()
        row["cell_tag"] = cell_tag(cell)
        row["arm"] = cell["arm"]
        row["landscape"] = cell["landscape"]
        row["visibility_weight"] = round(
            1.0 - cell["initial_curiosity"], 10)
        row["seed_semantics_note"] = "agent_seed drives agent+landscape RNG"
        return "ok", row
    except Exception:                              # noqa: BLE001
        return "fail", traceback.format_exc()


# ------------------------------------------------------------------ proofs
def _numeric_diffs(a: dict, b: dict, keys) -> dict:
    out = {}
    for k in keys:
        va, vb = a.get(k), b.get(k)
        if isinstance(va, (int, float, np.floating, np.integer)) and isinstance(
                vb, (int, float, np.floating, np.integer)):
            out[k] = float(abs(float(va) - float(vb)))
        else:
            out[k] = 0.0 if va == vb else float("inf")
    return out


def proof_identity() -> dict:
    """Baseline configuration must reproduce plain MyModel bit-for-bit."""
    results = []
    for land in ("mgstd5", "noisypr5"):
        for seed in (0, 1, 2):
            cell = dict(arm="A", r_spawn="exact", initial_curiosity=0.5,
                        landscape=land, alpha=1.0, p_reloc=0.0,
                        refill_mode="off", agent_generation_time=ENTRY_G)
            st_plain, plain = run_one(cell, seed, model_cls=MyModel)
            st_mine, mine = run_one(cell, seed, model_cls=InterventionsModel)
            assert st_plain == st_mine == "ok", (st_plain, st_mine)
            common = sorted(set(plain) & set(mine))
            diffs = _numeric_diffs(plain, mine, common)
            worst = max(diffs.values())
            results.append(dict(landscape=land, seed=seed, worst=worst,
                                n_common=len(common),
                                model_only=sorted(set(plain) - set(mine)),
                                driver_only=sorted(set(mine) - set(plain))))
            assert worst == 0.0, (
                f"baseline not bit-identical on {land}/seed{seed}: "
                f"{[k for k, v in diffs.items() if v != 0.0][:8]}")
    # occupied refill must equal the model with new_questions=True
    for seed in (0, 1):
        cell_off = dict(arm="C", refill_mode="off", initial_curiosity=0.5,
                        landscape="mgstd5", r_spawn="exact", alpha=1.0,
                        p_reloc=0.0, agent_generation_time=ENTRY_G)
        cell_occ = dict(cell_off, refill_mode="occupied")
        st1, plain_on = run_one(cell_occ, seed, model_cls=MyModel,
                                extra={"new_questions": True})
        st2, mine_occ = run_one(cell_occ, seed, model_cls=InterventionsModel)
        assert st1 == st2 == "ok"
        diffs = _numeric_diffs(plain_on, mine_occ,
                               sorted(set(plain_on) & set(mine_occ)))
        worst = max(diffs.values())
        results.append(dict(check="occupied==model(new_questions=True)",
                            seed=seed, worst=worst))
        assert worst == 0.0, f"occupied refill not identical: {worst}"
    # alpha = 1.0 must leave the weight vector untouched (float check)
    vals = [0.0, 1e-12, 0.5, 1.0, 12.75]
    assert all(v ** 1.0 == v for v in vals), "p ** 1.0 is not p exactly"
    return {"checks": len(results), "worst": 0.0,
            "driver_only_keys": results[0]["driver_only"]}


def proof_binding() -> dict:
    """Each knob must move at least one dynamics reporter."""
    base = dict(arm="A", r_spawn="exact", initial_curiosity=0.5,
                landscape="mgstd5", alpha=1.0, p_reloc=0.0,
                refill_mode="off", agent_generation_time=ENTRY_G)
    variants = {
        "r_spawn": dict(base, r_spawn="random"),
        "alpha": dict(base, alpha=0.0),
        "p_reloc": dict(base, p_reloc=0.02),
        "refill": dict(base, refill_mode="frontier"),
    }
    out = {}
    st0, b = run_one(base, 0)
    assert st0 == "ok"
    model_keys = [k for k in b if k not in
                  ("r_spawn", "alpha", "p_reloc", "refill_mode",
                   "newcomer_gap", "newcomer_prestige_mean",
                   "incumbent_prestige_mean", "n_newcomers", "n_incumbents")]
    for knob, cell in variants.items():
        st, v = run_one(cell, 0)
        assert st == "ok", (knob, st)
        diffs = _numeric_diffs(b, v, model_keys)
        moved = sorted(k for k, d in diffs.items() if d > 0.0)
        out[knob] = {"n_moved": len(moved), "sample": moved[:5],
                     "max": max(diffs.values())}
        assert moved, f"binding failed: {knob} moved nothing"
    return out


def proof_schema(cell: dict, n_jobs: int) -> dict:
    """A WORKER-written cell file must carry every analysis column."""
    rows = Parallel(n_jobs=n_jobs, return_as="list")(
        delayed(run_one)(cell, s) for s in SEEDS[:4])
    ok = [r for st, r in rows if st == "ok"]
    assert len(ok) == 4, f"schema probe: {len(ok)}/4 runs ok"
    path = PROBE_DIR / "schema_probe.csv"
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted(ok[0].keys()))
        w.writeheader()
        w.writerows(ok)
    os.replace(tmp, path)
    with open(path, newline="") as fh:
        back = list(csv.DictReader(fh))
    missing = [c for c in REQUIRED_COLUMNS if c not in (back[0] if back else {})]
    assert not missing, f"worker-written file missing columns: {missing}"
    assert len(back) == 4, f"expected 4 rows, got {len(back)}"
    return {"rows": len(back), "columns": len(back[0]), "missing": missing}


def probe_timing(n_jobs: int) -> dict:
    out = {}
    for name, cell in (
        ("A_entry_on", dict(arm="A", r_spawn="exact", initial_curiosity=0.5,
                            landscape="mgstd5", alpha=1.0, p_reloc=0.0,
                            refill_mode="off", agent_generation_time=ENTRY_G)),
        ("B_reloc_on", dict(arm="B", alpha=0.0, p_reloc=0.02,
                            visibility_weight=0.7, initial_curiosity=0.3,
                            landscape="noisypr5", r_spawn="exact",
                            refill_mode="off", agent_generation_time=ENTRY_G)),
        ("C_entry_off", dict(arm="C", refill_mode="off", initial_curiosity=0.5,
                             landscape="mgstd5", r_spawn="exact", alpha=1.0,
                             p_reloc=0.0, agent_generation_time=0.0)),
    ):
        t0 = time.time()
        st, _ = run_one(cell, 0)
        out[name] = {"wall_s": round(time.time() - t0, 2), "status": st}
    return out


# --------------------------------------------------------------- the sweep
def run_cell(cell: dict, n_jobs: int, log) -> dict:
    tag = cell_tag(cell)
    csv_path = CELLS_DIR / f"{tag}.csv"
    state_path = STATE_DIR / f"{tag}.json"
    if state_path.is_file():
        st = json.loads(state_path.read_text())
        if st.get("status") in ("ok", "ok_with_failures", "crashed"):
            log(f"SKIP {tag} (already {st['status']})")
            return st
    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, return_as="list")(
        delayed(run_one)(cell, s) for s in SEEDS)
    rows, errors = [], {}
    for seed, (status, payload) in zip(SEEDS, results):
        if status == "ok":
            rows.append(payload)
        else:
            errors[str(seed)] = payload[-2000:]
    n_expect = len(SEEDS)
    status = "ok" if len(rows) == n_expect else (
        "ok_with_failures" if rows else "crashed")
    if rows:
        tmp = csv_path.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=sorted(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        with open(tmp, newline="") as fh:
            n_back = sum(1 for _ in csv.DictReader(fh))
        assert n_back == len(rows), (
            f"{tag}: wrote {len(rows)} rows, read back {n_back}")
        os.replace(tmp, csv_path)
    entry = {
        "cell_tag": tag, "arm": cell["arm"], "status": status,
        "params": {k: v for k, v in cell.items()},
        "seeds_ok": len(rows), "seeds_failed": len(errors),
        "rows": len(rows), "rows_expected": n_expect,
        "errors": errors,
        "csv": csv_path.name if rows else None,
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
                 if any(k in ln for k in ("python", "julia"))
                 and "grep" not in ln]
        path.write_text(f"{len(heavy)} live python/julia processes\n" + "\n".join(heavy))
    except Exception as exc:                       # noqa: BLE001
        path.write_text(f"snapshot failed: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true",
                    help="run identity/binding/schema/timing proofs only")
    ap.add_argument("--arm", default="all", choices=["all", "A", "B", "C"])
    ap.add_argument("--n-jobs", type=int, default=3)
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
        binding = proof_binding()
        log("PROBE binding ok: "
            + "; ".join(f"{k}->{v['n_moved']} keys" for k, v in binding.items()))
        schema = proof_schema(cell_list()[0], args.n_jobs)
        log(f"PROBE schema ok: {schema}")
        timing = probe_timing(args.n_jobs)
        log(f"PROBE timing: {timing}")
        (PROBE_DIR / "probe_results.json").write_text(json.dumps(
            {"identity": ident, "binding": binding, "schema": schema,
             "timing": timing}, indent=1, default=str))
        print(json.dumps({"identity": ident, "binding": binding,
                          "schema": schema, "timing": timing}, indent=1,
                         default=str))
        return 0

    t_start = time.time()
    cells = cell_list()
    if args.arm != "all":
        cells = [c for c in cells if c["arm"] == args.arm]
    design = {
        "schema": "interventions_v1",
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "grid": f"arms={args.arm} cells={len(cells)} seeds={len(SEEDS)} "
                f"steps={STEPS}",
        "arms": {"A": "P-0002 spawn dispersal (18 cells)",
                 "B": "P-0003 lotteries + relocations (36 cells)",
                 "C": "P-0004 refill regimes (18 cells)"},
        "fixed": FIXED,
        "levels": {"r_spawn": list(SPAWN_LEVELS), "alpha": list(ALPHA_LEVELS),
                   "p_reloc": list(RELOC_LEVELS),
                   "refill_mode": list(REFILL_LEVELS),
                   "visibility_weight": list(VIS_WEIGHTS)},
        "landscapes": {k: {"fn": v[0].__name__, "params": v[1]}
                       for k, v in LANDSCAPES.items()},
        "n_jobs": args.n_jobs,
        "model_boundary": "InterventionsModel subclasses MyModel; no model edit",
        "trapped_definition": "knowledge < 0.025 (model's depletion threshold)",
    }
    (OUT / "design.json").write_text(json.dumps(design, indent=1, default=str))
    preflight_snapshot(OUT / "process_snapshot.txt")

    gross = len(cells) * len(SEEDS)
    log(f"sweep start: arms={args.arm} cells={len(cells)} gross_runs={gross} "
        f"seeds/cell={len(SEEDS)} steps={STEPS} n_jobs={args.n_jobs}")

    states = []
    for i, cell in enumerate(cells, 1):
        st = run_cell(cell, args.n_jobs, log)
        states.append(st)
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
            f"runs_ok={runs_ok} runs_failed={runs_fail} "
            f"wall={wall:.1f}s")
    log(line)
    print(line)
    return 0 if n_crash == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
