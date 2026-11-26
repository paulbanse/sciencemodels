"""ADR-0003 missing arms: the three gap cells omitted or marginal in D6/D3.

Prompt F2 (task/adr0003-missing-arms).  A compact, checkpointed companion to
``scripts/adr0003_remeasure/run_adr0003_wave.py`` (the D6 wave), filling only
the three cells its scope reductions left out or underpowered:

  lottery      interventions PhD lottery arm: alpha in {1.0, 0.5, 0.0} x
               p_reloc in {0.0, 0.02}, mgstd5, c = 0.5, g = 0.25, fixed_span,
               H = 200, 20 seeds  ->  data/interventions_lottery_adr0003/
  career_noisy career_flows noisypr5 leg: g in {0.125, 0.25, 0.5} x
               {fixed_span, age_hazard_staggered}, c = 0.5, H = 200, 20 seeds,
               career measures  ->  data/career_flows_noisy_adr0003/
  m2_n40       M2 decomposition check at n = 40 on mgstd5: lambda in {0, 1} x
               pull in {0, 1} x credit in {structural, equal_split} (logit),
               plus the M1 baseline cell, g = 0.25, fixed_span, H = 200,
               40 seeds  ->  data/path_b_m2_n40_adr0003/

Clock (ADR-0003): 1 step = 0.5 year, gamma = 0.5**(1/10) = 0.933, g = 0.25
(canonical), tau_max = 80, entry age 25, exit_mode fixed_span unless the
cell sweeps it.  Row schema, param block and summariser are imported from
the D6 runner (ADR-0002: one home per measure), so the m2_n40 cells are
bit-comparable with data/path_b_adr0003/ on the shared seeds (checked in
the analysis).  The only model extension is ``GapModel``: the PhD-lottery
supervisor exponent alpha and the sabbatical relocation knob, copied from
``sciencemodels/model.py`` verbatim except the roulette weight becomes
``entry_stock ** alpha`` (the P-0003.1 InterventionsModel precedent).  At
``alpha = 1.0, p_reloc = 0.0`` the class is bit-identical to ``MyModel``
(probed before the grid).

Checkpointing per cell (D6/ADR-0001 field-execution hygiene): atomic CSV +
state JSON under ``<arm>/cells/`` and ``<arm>/state/``, skip verified cells
on restart, failed seeds recorded never dropped.  ``--probe`` runs the
smoke canary, the GapModel bit-identity proof and two seeds per arm.

NOT CALIBRATED: model-internal only (owner ruling 2026-11-13).

Run:
    MPLBACKEND=Agg /opt/hermes/.venv/bin/python \
        scripts/adr0003_gaps/run_adr0003_gaps.py --probe
    MPLBACKEND=Agg /opt/hermes/.venv/bin/python \
        scripts/adr0003_gaps/run_adr0003_gaps.py --n-jobs 10
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from joblib import Parallel, delayed

from sciencemodels.agents import Scientist
from sciencemodels.model import MyModel
from scripts.adr0003_remeasure.run_adr0003_wave import (
    _spec,
    _tag,
    params_for,
    sha256,
    summarize,
)

ARMS = ("lottery", "career_noisy", "m2_n40")
SEEDS_20 = list(range(20))
SEEDS_40 = list(range(40))
ALPHA_LEVELS = (1.0, 0.5, 0.0)
RELOC_LEVELS = (0.0, 0.02)
G_VALUES = (0.125, 0.25, 0.5)
EXITS = ("fixed_span", "age_hazard_staggered")

ROOT = REPO / "data"
ARM_DIRS = {
    "lottery": ROOT / "interventions_lottery_adr0003",
    "career_noisy": ROOT / "career_flows_noisy_adr0003",
    "m2_n40": ROOT / "path_b_m2_n40_adr0003",
}
CENTRAL = ROOT / "adr0003_gaps"
for _d in (CENTRAL, *ARM_DIRS.values()):
    (_d / "cells").mkdir(parents=True, exist_ok=True)
    (_d / "state").mkdir(parents=True, exist_ok=True)

SOURCE_FILES = (
    "sciencemodels/agents.py",
    "sciencemodels/model.py",
    "sciencemodels/metrics.py",
    "sciencemodels/sweeps.py",
    "scripts/adr0003_remeasure/run_adr0003_wave.py",
    "scripts/adr0003_gaps/run_adr0003_gaps.py",
)


# ---------------------------------------------------------------------------
# GapModel: PhD-lottery exponent + sabbatical relocation on the ADR-0003 core
# ---------------------------------------------------------------------------
class GapModel(MyModel):
    """``MyModel`` plus the two intervention knobs this round needs.

    ``alpha`` is the supervisor-roulette exponent (P-0003.1): newborn
    supervisors are drawn with weight ``entry_prestige_stock ** alpha``, so
    ``alpha = 1.0`` is the model's own prestige-proportional rule and
    ``alpha = 0.0`` a pure lottery.  ``p_reloc`` is the sabbatical relocation
    probability for trapped agents (the D6 ``AdrInterventionModel`` rule).

    The two roulette bodies below are copied from ``sciencemodels/model.py``
    verbatim except for the ``** self.alpha`` weight (same precedent as
    ``scripts/interventions/run_interventions_v1.py``).  At ``alpha = 1.0``
    the weights are bit-identical (``x ** 1.0 == x``), so the RNG stream and
    the trajectories match ``MyModel`` exactly; the probe asserts it.
    """

    def __init__(self, *, alpha: float = 1.0, p_reloc: float = 0.0, **kwargs):
        if alpha not in ALPHA_LEVELS:
            raise ValueError(f"alpha must be one of {ALPHA_LEVELS}, got {alpha!r}")
        self.alpha = float(alpha)
        self.p_reloc = float(p_reloc)
        super().__init__(**kwargs)

    def step(self, endupdate=True):
        if self.p_reloc > 0.0:
            self._relocate_trapped()
        return super().step(endupdate=endupdate)

    def _relocate_trapped(self):
        """D6 AdrInterventionModel rule: one draw per trapped agent per step.

        Trapped = standing on a tile below the depletion threshold 0.025;
        relocation sends the agent to a random unexplored tile.
        """
        knowledge = self.grid.properties["knowledge"].data
        explored = self.grid.properties["explored"].data
        for agent in list(self._scientists):
            if knowledge[agent.pos] >= 0.025:
                continue
            if self.rng.random() >= self.p_reloc:
                continue
            free = np.argwhere(~explored)
            if len(free) == 0:
                continue
            idx = int(self.rng.integers(len(free)))
            self.new_place(agent, (int(free[idx][0]), int(free[idx][1])))

    def _spawn_one_from_roulette(self):
        """model.py verbatim except the weight is ``entry_stock ** alpha``."""
        if not self._scientists:
            return None
        entry_stock = self.entry_prestige_stock
        tuples = [(getattr(a, entry_stock) ** self.alpha, a)
                  for a in self._scientists]
        total = sum(getattr(a, entry_stock) ** self.alpha
                    for a in self._scientists)
        val = self.rng.uniform(0, total)
        supervisor = self._scientists[0]
        for p, a in tuples:
            if val <= p:
                supervisor = a
                break
            val -= p
        w = self.agent_generation_function(
            self.initial_curiosity, self.rng, self.generation_params
        )
        a = Scientist(self, w, self.epsilon)
        a.ideal_merit_cum = np.float64(0.0)
        a.is_newcomer = True
        self.grid.place_agent(a, supervisor.pos)
        self._scientists.append(a)
        return a

    def _MyModel__generate_new_agents(self):
        """model.py verbatim (birth block + ADR-0003 turnover tail) except
        the roulette weight is ``entry_stock ** alpha`` in the birth block."""
        born = []
        if self.agent_generation_time > 0:
            agents_to_generate: int = np.floor(
                self.steps * self.agent_generation_time
            ) - np.floor((self.steps - 1) * self.agent_generation_time)
            entry_stock = self.entry_prestige_stock
            prestige_agent_tuples = [
                (getattr(a, entry_stock) ** self.alpha, a)
                for a in self._scientists
            ]
            totalPrestige = sum(
                getattr(a, entry_stock) ** self.alpha for a in self._scientists
            )
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
                a.ideal_merit_cum = np.float64(0.0)
                a.is_newcomer = True
                coords = sup.pos
                self.grid.place_agent(a, coords)
                self._scientists.append(a)
                born.append(a)

        if self.exit_mode == "fixed_span":
            for _ in born:
                if len(self._scientists) > self.number_agents:
                    victim = max(
                        self._scientists,
                        key=lambda a: (a.age, -a.unique_id),
                    )
                    self._retire_agent(victim)
        elif self.exit_mode in ("age_hazard_sync", "age_hazard_staggered"):
            due = [
                a for a in list(self._scientists)
                if a.tenure_steps >= self.tau_max
            ]
            for a in due:
                self._retire_agent(a)
                self._spawn_one_from_roulette()

        for agent in self._scientists:
            self.grid.properties["explored"].data[agent.pos] = True


# ---------------------------------------------------------------------------
# Cell grid
# ---------------------------------------------------------------------------
def cell_specs() -> list[dict]:
    specs = []

    # (a) PhD lottery arm: alpha x p_reloc at the D6 intB central settings.
    for alpha in ALPHA_LEVELS:
        for p_reloc in RELOC_LEVELS:
            specs.append(_spec(
                "lottery",
                tag=_tag("lot", alpha=alpha, reloc=p_reloc, g=0.25),
                alpha=alpha, p_reloc=p_reloc, g=0.25, n_seeds=20,
            ))

    # (b) career_flows noisypr5 leg: g x exit, career measures.
    for exit_mode in EXITS:
        for g in G_VALUES:
            specs.append(_spec(
                "career_noisy",
                tag=_tag("cnois", exit=exit_mode, g=g),
                land="noisypr5", g=g, exit=exit_mode,
                needs_career=True, n_seeds=20,
            ))

    # (c) M2 n=40 check on mgstd5: baseline + lambda x pull x credit.
    specs.append(_spec(
        "m2_n40",
        tag=_tag("m2n40", kind="m1base"),
        g=0.25, dyadic=False, n_seeds=40,
    ))
    for lam in (0.0, 1.0):
        for pull in (0.0, 1.0):
            for credit in ("structural", "equal_split"):
                specs.append(_spec(
                    "m2_n40",
                    tag=_tag("m2n40", lam=lam, pull=pull,
                             credit=("struct" if credit == "structural"
                                     else "equal")),
                    lam=lam, pull=pull, credit=credit, g=0.25,
                    dyadic=True, form="logit", n_seeds=40,
                ))

    seen = set()
    for spec in specs:
        key = (spec["arm"], spec["cell_tag"])
        if key in seen:
            raise AssertionError(f"duplicate cell {key}")
        seen.add(key)
    return specs


def seeds_for(spec: dict) -> list[int]:
    return SEEDS_40 if int(spec.get("n_seeds", 20)) == 40 else SEEDS_20


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------
def run_one(spec: dict, seed: int):
    cls = GapModel if spec["arm"] == "lottery" else MyModel
    try:
        params = params_for(spec, seed)
        if spec["arm"] == "lottery":
            params.update(alpha=float(spec["alpha"]),
                          p_reloc=float(spec["p_reloc"]))
        model = cls(**params)
        for _ in range(int(spec["horizon"])):
            model.step(True)
        row = summarize(model, spec, seed)
        row["alpha"] = float(spec.get("alpha", 1.0))
        return "ok", row
    except Exception:  # noqa: BLE001
        return "fail", traceback.format_exc()


# ---------------------------------------------------------------------------
# Checkpointing (D6 conventions)
# ---------------------------------------------------------------------------
def snapshot_sources() -> dict:
    return {f: sha256(REPO / f) for f in SOURCE_FILES}


def _state_path(spec: dict) -> Path:
    return ARM_DIRS[spec["arm"]] / "state" / f"{spec['cell_tag']}.json"


def _csv_path(spec: dict) -> Path:
    return ARM_DIRS[spec["arm"]] / "cells" / f"{spec['cell_tag']}.csv"


def run_cell(spec: dict, n_jobs: int, log) -> dict:
    tag = spec["cell_tag"]
    seeds = seeds_for(spec)
    csv_path = _csv_path(spec)
    state_path = _state_path(spec)
    if state_path.is_file() and csv_path.is_file():
        state = json.loads(state_path.read_text())
        if state.get("status") in ("ok", "ok_with_failures"):
            try:
                n = sum(1 for _ in open(csv_path)) - 1
            except Exception:  # noqa: BLE001
                n = -1
            if n == len(seeds):
                log(f"SKIP {spec['arm']}/{tag} (already {state['status']})")
                return state
    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, return_as="list")(
        delayed(run_one)(spec, s) for s in seeds
    )
    rows, errors = [], {}
    for seed, (status, payload) in zip(seeds, results):
        if status == "ok":
            rows.append(payload)
        else:
            errors[str(seed)] = str(payload)[-2000:]
    status = "ok" if len(rows) == len(seeds) else (
        "ok_with_failures" if rows else "crashed"
    )
    if rows:
        tmp = csv_path.with_suffix(".csv.tmp")
        pd.DataFrame(rows).to_csv(tmp, index=False)
        back = pd.read_csv(tmp)
        assert len(back) == len(rows), f"{tag}: wrote {len(rows)}, read {len(back)}"
        os.replace(tmp, csv_path)
    entry = {
        "arm": spec["arm"],
        "cell_tag": tag,
        "status": status,
        "spec": dict(spec),
        "seeds_ok": len(rows),
        "seeds_failed": len(errors),
        "rows": len(rows),
        "rows_expected": len(seeds),
        "errors": errors,
        "csv": csv_path.name if rows else None,
        "wall_s": round(time.time() - t0, 2),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    tmp_state = state_path.with_suffix(".json.tmp")
    tmp_state.write_text(json.dumps(entry, indent=1))
    os.replace(tmp_state, state_path)
    log(
        f"CELL {spec['arm']}/{tag} -> {status} seeds_ok={len(rows)} "
        f"rows={len(rows)}/{len(seeds)} wall={entry['wall_s']}s"
        + (f" ERRORS={list(errors)}" if errors else "")
    )
    return entry


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------
def probe(specs: list[dict]) -> dict:
    """Smoke canary, GapModel bit-identity, two seeds per arm schema check."""
    out = {}
    required = (
        "c_ideal_fin", "c_ideal_tm", "explored_fin", "harvested_fin",
        "ideal_loss_cum", "n_final",
    )

    # (1) GapModel(alpha=1, p_reloc=0) bit-identity vs plain MyModel.
    spec = next(s for s in specs if s["arm"] == "lottery"
                and s["alpha"] == 1.0 and s["p_reloc"] == 0.0)
    identity_diffs = {}
    for seed in (0, 1):
        s_gap = dict(spec, alpha=1.0, p_reloc=0.0)
        status_a, row_a = run_one(s_gap, seed)
        params = params_for(spec, seed)          # no alpha/p_reloc: plain MyModel
        model = MyModel(**params)
        for _ in range(int(spec["horizon"])):
            model.step(True)
        row_b = summarize(model, spec, seed)
        row_b["alpha"] = 1.0
        assert status_a == "ok", row_a
        for key in row_a:
            va, vb = row_a[key], row_b[key]
            if isinstance(va, (int, float, np.floating, np.integer)) \
                    and not isinstance(va, bool):
                d = abs(float(va) - float(vb))
                identity_diffs[key] = max(identity_diffs.get(key, 0.0), d)
    worst = max(identity_diffs.values()) if identity_diffs else 0.0
    assert identity_diffs, "identity check compared no numeric columns"
    assert worst == 0.0, f"GapModel(alpha=1) not bit-identical: {identity_diffs}"
    out["gapmodel_bit_identity"] = {
        "seeds": [0, 1], "columns": len(identity_diffs),
        "max_abs_diff": worst,
    }

    # (2) two seeds per arm: worker-side schema.
    for arm in ARMS:
        spec = next(s for s in specs if s["arm"] == arm)
        rows, failures = [], []
        for seed in (0, 1):
            status, payload = run_one(spec, seed)
            if status == "ok":
                rows.append(payload)
            else:
                failures.append((seed, str(payload)[-500:]))
        if failures or len(rows) != 2:
            raise AssertionError(f"probe {arm} failed: {failures}")
        missing = [k for k in required if k not in rows[0]]
        if missing:
            raise AssertionError(f"probe {arm} missing keys: {missing}")
        out[arm] = {
            "cell_tag": spec["cell_tag"],
            "rows": len(rows),
            "c_ideal_tm": float(np.mean([r["c_ideal_tm"] for r in rows])),
            "explored_fin": float(np.mean([r["explored_fin"] for r in rows])),
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="all",
                    help="comma list of arms or 'all'")
    ap.add_argument("--n-jobs", type=int, default=10)
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()

    log_path = CENTRAL / "run_log.txt"

    def log(msg):
        line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as fh:
            fh.write(line + "\n")

    all_specs = cell_specs()
    if args.arms != "all":
        selected = {a.strip() for a in args.arms.split(",") if a.strip()}
        unknown = selected - set(ARMS)
        if unknown:
            raise SystemExit(f"unknown arms {sorted(unknown)}")
        all_specs = [s for s in all_specs if s["arm"] in selected]

    if args.probe:
        canary = subprocess.run(
            [sys.executable, str(REPO / "tests/test_smoke.py")],
            cwd=REPO, capture_output=True, text=True,
            env={**os.environ, "MPLBACKEND": "Agg"},
        )
        if canary.returncode != 0:
            raise SystemExit("smoke canary failed:\n" + canary.stdout + canary.stderr)
        result = probe(all_specs)
        (CENTRAL / "probe_results.json").write_text(json.dumps(result, indent=1))
        print(json.dumps(result, indent=1))
        return 0

    canary = subprocess.run(
        [sys.executable, str(REPO / "tests/test_smoke.py")],
        cwd=REPO, capture_output=True, text=True,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )
    if canary.returncode != 0:
        raise SystemExit("smoke canary failed:\n" + canary.stdout + canary.stderr)
    log("canary gate OK: tests/test_smoke.py exit 0")

    src_before = snapshot_sources()
    design = {
        "schema": "adr0003_gaps_v1",
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "clock": {
            "step_years": 0.5,
            "gamma": 0.5 ** (1 / 10),
            "g_canonical": 0.25,
            "g_swept": list(G_VALUES),
            "exit_modes": ["fixed_span", "age_hazard_staggered"],
            "tau_max": 80,
            "horizon": 200,
        },
        "arms": {
            "lottery": "alpha {1.0,0.5,0.0} x p_reloc {0.0,0.02}, mgstd5, "
                       "c=0.5, g=0.25, fixed_span, 20 seeds",
            "career_noisy": "g {0.125,0.25,0.5} x {fixed_span, "
                            "age_hazard_staggered}, noisypr5, c=0.5, "
                            "20 seeds, career measures",
            "m2_n40": "M1 baseline + lambda {0,1} x pull {0,1} x credit "
                      "{structural, equal_split}, logit, mgstd5, g=0.25, "
                      "fixed_span, 40 seeds",
        },
        "cells": len(all_specs),
        "runs_gross": sum(len(seeds_for(s)) for s in all_specs),
        "source_snapshot_before": src_before,
    }
    (CENTRAL / "design.json").write_text(json.dumps(design, indent=1))

    gross = sum(len(seeds_for(s)) for s in all_specs)
    log(
        f"gaps start: arms={sorted({s['arm'] for s in all_specs})} "
        f"cells={len(all_specs)} runs={gross} n_jobs={args.n_jobs}"
    )

    t0 = time.time()
    states = []
    for spec in all_specs:
        states.append(run_cell(spec, args.n_jobs, log))

    n_ok = sum(1 for s in states if s["status"] == "ok")
    n_part = sum(1 for s in states if s["status"] == "ok_with_failures")
    n_crash = sum(1 for s in states if s["status"] == "crashed")
    runs_ok = sum(s["seeds_ok"] for s in states)
    runs_fail = sum(s["seeds_failed"] for s in states)
    wall = time.time() - t0
    src_after = snapshot_sources()
    moved = [f for f in SOURCE_FILES if src_before[f] != src_after[f]]
    manifest = {
        "schema": "adr0003_gaps_v1",
        "cells_gross": len(all_specs),
        "runs_gross": gross,
        "cells_ok": n_ok,
        "cells_partial": n_part,
        "cells_crashed": n_crash,
        "runs_ok": runs_ok,
        "runs_failed": runs_fail,
        "wall_s": round(wall, 1),
        "n_jobs": args.n_jobs,
        "source_snapshot_after": src_after,
        "sources_moved_during_run": moved,
        "arms": sorted({s["arm"] for s in all_specs}),
        "data_dirs": {a: str(ARM_DIRS[a].relative_to(REPO)) for a in ARMS},
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (CENTRAL / "manifest.json").write_text(json.dumps(manifest, indent=1))
    line = (
        f"GROSS cells={len(all_specs)} runs={gross} | NET ok_cells={n_ok} "
        f"partial={n_part} crashed={n_crash} runs_ok={runs_ok} "
        f"runs_failed={runs_fail} wall={wall:.1f}s sources_moved={moved or 'none'}"
    )
    log(line)
    print(line)
    return 0 if n_crash == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
