"""ADR-0003 re-run wave: core Phase-1 sweeps on the 6-month clock.

Writes NEW archives only, under ``data/<arm>_adr0003/``; no pre-ADR-0003
directory is touched.  The driver is deliberately a compact core panel, not a
full Cartesian replay of every historical cell: each historical archive keeps
its signature axis and is re-run at the ADR-0003 clock (1 step = 0.5 year,
gamma = 0.5**(1/10), g in {0.125, 0.25, 0.5}, tau_max = 80, fixed_span and
age_hazard_staggered).  The report names the retained axes and the reductions.

Arms
----
sensitivity   T1-like core: c x landscape x g, fixed_span, H=200.
welfare       W_v05-like: c x visibility-reward x landscape leg x g,
              fixed_span, H=200.
career_flows  career-flow core: c x g x {fixed_span, staggered}, mgstd5,
              H=200, with agent-vars career measures.
prestige      prestige-stock selector: stock x c x landscape x g, fixed_span,
              H=200.
turnover      turnover core: c x landscape x g x {fixed_span, staggered},
              H=400.
interventions refill / spawn / sabbatical core: three compact sub-arms,
              fixed_span, H=200.
path_b        Path B flagship: lambda x pull x g, logit + equal_split,
              mgstd5, fixed_span, H=200.
fig1_xi       Figure 1 dense xi: 11 xi values x 5 tiers (M0, M1 fixed,
              M1 staggered, M2 lambda=1, M2 lambda=0), mgstd5; M0/M2 H=200,
              M1 turnover H=400.

Every cell is 20 seeds, checkpointed atomically (CSV + state JSON), and
verified on restart.  ``--probe`` runs two seeds per arm before the grid.

Run:
    MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/adr0003_remeasure/run_adr0003_wave.py --probe
    MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/adr0003_remeasure/run_adr0003_wave.py --n-jobs 10
"""
from __future__ import annotations

import argparse
import hashlib
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

from sciencemodels import landscapes as L
from sciencemodels.model import (
    CANONICAL_AGENT_GENERATION_TIME,
    CANONICAL_TAU_MAX,
    CANONICAL_VANISHING_FACTOR,
    MyModel,
)
from sciencemodels.sweeps import career_measures_from_agent_vars

SEEDS = list(range(20))
TAU_MAX = CANONICAL_TAU_MAX
G_VALUES = (0.125, 0.25, 0.5)
EXITS = ("fixed_span", "age_hazard_staggered")
CURIOSITIES = (0.3, 0.5, 0.7)
WELFARE_CURIOSITIES = (0.1, 0.3, 0.5, 0.7, 0.9)
STOCKS = ("prestige", "prestige_vanishing", "prestige_visibility")
XIS = (0.1, 0.3, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)

LANDSCAPES = {
    "mgstd5": (L.multipleGaussians, {"std": 5, "number_gaussians": 2}),
    "noisypr5": (
        L.noisyGaussian,
        {"prop_random": 5, "std": 5, "std_curiosity": 0.1},
    ),
}
LAND_KEYS = ("mgstd5", "noisypr5")

ARMS = (
    "sensitivity",
    "welfare",
    "career_flows",
    "prestige",
    "turnover",
    "interventions",
    "path_b",
    "fig1_xi",
)
ROOT = REPO / "data"
CENTRAL = ROOT / "adr0003_remeasure"
ARM_DIRS = {arm: ROOT / f"{arm}_adr0003" for arm in ARMS}
for _d in (CENTRAL, *ARM_DIRS.values()):
    (_d / "cells").mkdir(parents=True, exist_ok=True)
    (_d / "state").mkdir(parents=True, exist_ok=True)

SOURCE_FILES = (
    "sciencemodels/agents.py",
    "sciencemodels/model.py",
    "sciencemodels/metrics.py",
    "sciencemodels/sweeps.py",
    "scripts/adr0003_remeasure/run_adr0003_wave.py",
)


def identity(cur, _rng, _params):
    """Identity curiosity generator (top-level, spawn-safe)."""
    return cur


# ---------------------------------------------------------------------------
# Interventions subclass: spawn / sabbatical / refill on top of core turnover
# ---------------------------------------------------------------------------
class AdrInterventionModel(MyModel):
    """Core ``MyModel`` plus the compact interventions panel.

    Spawn placement is applied after the core birth/exit step, so the new
    ADR-0003 turnover is preserved.  Refill ``occupied`` uses the model's own
    new-question rule; ``frontier`` moves the replacement to a free tile.
    """

    def __init__(self, *, spawn: str = "exact", p_reloc: float = 0.0,
                 refill: str = "off", **kwargs):
        if spawn not in ("exact", "ring1", "random"):
            raise ValueError(f"spawn must be exact/ring1/random, got {spawn!r}")
        if refill not in ("off", "occupied", "frontier"):
            raise ValueError(f"refill must be off/occupied/frontier, got {refill!r}")
        self.spawn = spawn
        self.p_reloc = float(p_reloc)
        self.refill = refill
        kwargs["new_questions"] = refill == "occupied"
        super().__init__(**kwargs)

    def step(self, endupdate=True):
        if self.p_reloc > 0.0:
            self._relocate_trapped()
        return super().step(endupdate=endupdate)

    def _relocate_trapped(self):
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

    def _generate_new_questions(self):
        if self.refill == "off":
            return
        if self.refill == "occupied":
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
            knowledge[x, y] = self.rng.random()

    def _MyModel__generate_new_agents(self):
        before = {a.unique_id for a in self._scientists}
        MyModel._MyModel__generate_new_agents(self)
        if self.spawn == "exact":
            return
        for agent in list(self._scientists):
            if agent.unique_id in before:
                continue
            if self.spawn == "ring1":
                dx, dy = ((1, 0), (-1, 0), (0, 1), (0, -1))[
                    int(self.rng.integers(0, 4))
                ]
                x, y = agent.pos
                coords = ((x + dx) % self.size, (y + dy) % self.size)
            else:
                coords = (
                    int(self.rng.integers(0, self.size)),
                    int(self.rng.integers(0, self.size)),
                )
            self.new_place(agent, coords)


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------
def _tag(arm: str, **parts) -> str:
    bits = [arm]
    for key in sorted(parts):
        value = parts[key]
        if isinstance(value, float):
            value = f"{value:g}".replace(".", "p")
        bits.append(f"{key}-{value}")
    return "_".join(bits).replace(" ", "")


def _spec(arm: str, *, tag: str, **kw) -> dict:
    base = dict(
        arm=arm,
        cell_tag=tag,
        horizon=200,
        c=0.5,
        xi=0.5,
        land="mgstd5",
        g=0.25,
        exit="fixed_span",
        tau_max=TAU_MAX,
        stock="prestige",
        visibility=True,
        leg=2,
        spawn="exact",
        p_reloc=0.0,
        refill="off",
        pull=0.0,
        lam=0.0,
        form="logit",
        credit="equal_split",
        dyadic=False,
        tier="core",
        needs_career=False,
    )
    base.update(kw)
    return base


def cell_specs() -> list[dict]:
    specs = []

    # T1: curiosity x landscape x entry, fixed_span.
    for land in LAND_KEYS:
        for g in G_VALUES:
            for c in CURIOSITIES:
                specs.append(_spec(
                    "sensitivity",
                    tag=_tag("sens", land=land, g=g, c=c),
                    land=land, g=g, c=c,
                ))

    # W_v05: curiosity x visibility reward x legibility x entry.
    for c in WELFARE_CURIOSITIES:
        for visibility in (True, False):
            for leg in (2, 6):
                for g in G_VALUES:
                    specs.append(_spec(
                        "welfare",
                        tag=_tag("welf", c=c, vis=int(visibility), leg=leg, g=g),
                        c=c, visibility=visibility, leg=leg, g=g,
                    ))

    # Career-flow core: both ADR-0003 exit modes, mgstd5.
    for exit_mode in EXITS:
        for g in G_VALUES:
            for c in CURIOSITIES:
                specs.append(_spec(
                    "career_flows",
                    tag=_tag("career", exit=exit_mode, g=g, c=c),
                    c=c, g=g, exit=exit_mode, needs_career=True,
                ))

    # Prestige-stock selector.
    for stock in STOCKS:
        for land in LAND_KEYS:
            for g in G_VALUES:
                for c in CURIOSITIES:
                    specs.append(_spec(
                        "prestige",
                        tag=_tag("prest", stock=stock, land=land, g=g, c=c),
                        stock=stock, land=land, g=g, c=c,
                    ))

    # Turnover core, H=400.
    for exit_mode in EXITS:
        for land in LAND_KEYS:
            for g in G_VALUES:
                for c in CURIOSITIES:
                    specs.append(_spec(
                        "turnover",
                        tag=_tag("turn", exit=exit_mode, land=land, g=g, c=c),
                        exit=exit_mode, land=land, g=g, c=c, horizon=400,
                    ))

    # Interventions: compact spawn / sabbatical / refill sub-arms.
    for spawn in ("exact", "ring1", "random"):
        for g in G_VALUES:
            for c in CURIOSITIES:
                specs.append(_spec(
                    "interventions",
                    tag=_tag("intA", spawn=spawn, g=g, c=c),
                    spawn=spawn, g=g, c=c,
                ))
    for p_reloc in (0.0, 0.02):
        for g in G_VALUES:
            specs.append(_spec(
                "interventions",
                tag=_tag("intB", reloc=p_reloc, g=g, c=0.5),
                p_reloc=p_reloc, g=g, c=0.5,
            ))
    for refill in ("off", "occupied", "frontier"):
        for g in G_VALUES:
            for c in CURIOSITIES:
                specs.append(_spec(
                    "interventions",
                    tag=_tag("intC", refill=refill, g=g, c=c),
                    refill=refill, g=g, c=c,
                ))

    # Path B flagship: lambda x pull x entry, logit + equal_split, mgstd5.
    for lam in (0.0, 0.5, 1.0):
        for pull in (0.0, 1.0):
            for g in G_VALUES:
                specs.append(_spec(
                    "path_b",
                    tag=_tag("pb", lam=lam, pull=pull, g=g),
                    lam=lam, pull=pull, g=g, dyadic=True,
                    form="logit", credit="equal_split",
                ))

    # Figure 1 dense xi, five tiers.
    tiers = (
        ("M0", dict(exit="none", g=0.0, dyadic=False, horizon=200)),
        ("M1_fixed", dict(exit="fixed_span", g=0.25, dyadic=False, horizon=400)),
        ("M1_staggered", dict(exit="age_hazard_staggered", g=0.25,
                              dyadic=False, horizon=400)),
        ("M2_l1", dict(exit="fixed_span", g=0.25, dyadic=True,
                       lam=1.0, pull=1.0, horizon=200)),
        ("M2_l0", dict(exit="fixed_span", g=0.25, dyadic=True,
                       lam=0.0, pull=1.0, horizon=200)),
    )
    for tier, tier_kw in tiers:
        for xi in XIS:
            specs.append(_spec(
                "fig1_xi",
                tag=_tag("fig1", tier=tier, xi=xi),
                xi=xi, c=round(1.0 - xi, 2), tier=tier, **tier_kw,
            ))

    # uniqueness inside each arm directory
    seen = set()
    for spec in specs:
        key = (spec["arm"], spec["cell_tag"])
        if key in seen:
            raise AssertionError(f"duplicate cell {key}")
        seen.add(key)
    return specs


def params_for(spec: dict, seed: int) -> dict:
    fn, gp = LANDSCAPES[spec["land"]]
    params = dict(
        n_agents=20,
        initial_curiosity=float(spec["c"]),
        epsilon=0.1,
        harvest=0.02,
        grid_size=20,
        cell_init_function=fn,
        generation_params=dict(gp),
        use_distance=True,
        agent_generation_time=float(spec["g"]),
        new_questions=False,
        agent_seed=int(seed),
        step_limit=int(spec["horizon"]),
        agent_generation_function=identity,
        vanishing_factor_for_prestige=CANONICAL_VANISHING_FACTOR,
        use_visibility_reward=bool(spec["visibility"]),
        knowledge_normalization="share",
        prestige_accrual="harvested",
        entry_prestige_stock=spec["stock"],
        exit_mode=spec["exit"],
        tau_max=int(spec["tau_max"]),
    )
    if spec["dyadic"]:
        params.update(
            dyadic_mode=spec["form"],
            dyadic_lambda=float(spec["lam"]),
            dyadic_move_pull=float(spec["pull"]),
            credit_sharing=spec["credit"],
            dyadic_proposal_regime="newcomer",
            closeness_mode="grid",
            dyadic_visibility_stock="prestige_visibility",
        )
    if spec["arm"] == "interventions":
        params.update(
            spawn=spec["spawn"],
            p_reloc=float(spec["p_reloc"]),
            refill=spec["refill"],
        )
    return params


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------
def _finite_mean(values) -> float:
    a = np.asarray(values, dtype=float)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


def summarize(model, spec: dict, seed: int) -> dict:
    dc = model.datacollector.get_model_vars_dataframe()
    last = dc.iloc[-1]
    merits = [float(a.ideal_Merit) for a in model.agents if np.isfinite(a.ideal_Merit)]
    row = {
        "arm": spec["arm"],
        "cell_tag": spec["cell_tag"],
        "tier": spec["tier"],
        "seed": int(seed),
        "agent_seed": int(seed),
        "g": float(spec["g"]),
        "exit_mode": spec["exit"],
        "tau_max": int(spec["tau_max"]),
        "curiosity": float(spec["c"]),
        "xi": float(spec["xi"]),
        "landscape": spec["land"],
        "horizon": int(spec["horizon"]),
        "stock": spec["stock"],
        "use_visibility_reward": bool(spec["visibility"]),
        "leg": int(spec["leg"]),
        "spawn": spec["spawn"],
        "p_reloc": float(spec["p_reloc"]),
        "refill_mode": spec["refill"],
        "dyadic_move_pull": float(spec["pull"]),
        "dyadic_lambda": float(spec["lam"]),
        "formulation": spec["form"],
        "credit_sharing": spec["credit"],
        "c_ideal_fin": float(last["Corr_prestige_ideal_merit"]),
        "c_ideal_tm": float(dc["Corr_prestige_ideal_merit"].mean()),
        "c_ideal_late": float(dc["Corr_prestige_ideal_merit"].iloc[-100:].mean()),
        "c_local_fin": float(last["Corr_prestige_local_merit"]),
        "c_local_tm": float(dc["Corr_prestige_local_merit"].mean()),
        "c_local_late": float(dc["Corr_prestige_local_merit"].iloc[-100:].mean()),
        "wedge_fin": float(last["Corr_prestige_local_merit"]
                           - last["Corr_prestige_ideal_merit"]),
        "wedge_tm": float(dc["Corr_prestige_local_merit"].mean()
                          - dc["Corr_prestige_ideal_merit"].mean()),
        "top10_fin": float(last["top10pct_prestige_share"]),
        "top10_tm": float(dc["top10pct_prestige_share"].mean()),
        "top10_late": float(dc["top10pct_prestige_share"].iloc[-100:].mean()),
        "harvested_fin": float(last["percentage_knowledge_harvested"]),
        "explored_fin": float(last["explored_percentage"]),
        "explored_tm": float(dc["explored_percentage"].mean()),
        "best_knowledge_fin": float(last["best_knowledge"]),
        "avg_knowledge_fin": float(last["avg_knowledge_on_grid"]),
        "ideal_merit_sum": float(np.sum(merits)) if merits else float("nan"),
        "ideal_merit_mean": float(np.mean(merits)) if merits else float("nan"),
        "ideal_loss_cum": -float(
            sum(getattr(a, "ideal_merit_cum", 0.0) for a in model._scientists)
        ),
        "n_final": int(len(model._scientists)),
        "n_newcomers_final": int(sum(a.is_newcomer for a in model._scientists)),
        "mean_age_final": float(last["mean_age"]),
        "mean_tenure_final": float(last["mean_tenure"]),
        "gap_recent_final": float(last["gap_recent"]),
        "gap_recent_mean_late": _finite_mean(dc["gap_recent"].iloc[-100:]),
        "team_tm": float(dc["mean_team_size"].mean()),
        "repeated_tm": float(dc["repeated_collab_share"].mean()),
        "links_tm": float(dc["n_active_links"].mean()),
        "assort_tm": float(dc["dyadic_assortativity_prestige"].mean()),
    }
    if spec["needs_career"]:
        av = model.datacollector.get_agent_vars_dataframe()
        cm = career_measures_from_agent_vars(av, dc)
        clast = cm.iloc[-1]
        row.update(
            career_pop_final=int(clast["n_agents"]),
            career_gap_final=float(clast["prestige_gap"]),
            career_gap_ratio_mean_late=float(
                cm["prestige_gap_ratio"].iloc[-50:].mean()
            ),
            career_percap_harvest_final=float(clast["percap_harvest_cum"]),
            career_harvest_flow_mean_last50=float(
                cm["harvest_flow"].iloc[-50:].mean()
            ),
            career_percap_flow_mean_last50=float(
                cm["percap_harvest_flow"].iloc[-50:].mean()
            ),
        )
    return row


def run_one(spec: dict, seed: int):
    cls = AdrInterventionModel if spec["arm"] == "interventions" else MyModel
    try:
        model = cls(**params_for(spec, seed))
        for _ in range(int(spec["horizon"])):
            model.step(True)
        return "ok", summarize(model, spec, seed)
    except Exception:  # noqa: BLE001
        return "fail", traceback.format_exc()


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_sources() -> dict:
    return {f: sha256(REPO / f) for f in SOURCE_FILES}


def _state_path(spec: dict) -> Path:
    return ARM_DIRS[spec["arm"]] / "state" / f"{spec['cell_tag']}.json"


def _csv_path(spec: dict) -> Path:
    return ARM_DIRS[spec["arm"]] / "cells" / f"{spec['cell_tag']}.csv"


def run_cell(spec: dict, n_jobs: int, log) -> dict:
    tag = spec["cell_tag"]
    csv_path = _csv_path(spec)
    state_path = _state_path(spec)
    if state_path.is_file() and csv_path.is_file():
        state = json.loads(state_path.read_text())
        if state.get("status") in ("ok", "ok_with_failures"):
            try:
                n = sum(1 for _ in open(csv_path)) - 1
            except Exception:  # noqa: BLE001
                n = -1
            if n == len(SEEDS):
                log(f"SKIP {spec['arm']}/{tag} (already {state['status']})")
                return state
    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, return_as="list")(
        delayed(run_one)(spec, s) for s in SEEDS
    )
    rows, errors = [], {}
    for seed, (status, payload) in zip(SEEDS, results):
        if status == "ok":
            rows.append(payload)
        else:
            errors[str(seed)] = str(payload)[-2000:]
    status = "ok" if len(rows) == len(SEEDS) else (
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
        "rows_expected": len(SEEDS),
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
        f"rows={len(rows)}/{len(SEEDS)} wall={entry['wall_s']}s"
        + (f" ERRORS={list(errors)}" if errors else "")
    )
    return entry


def preflight_snapshot(path: Path) -> None:
    try:
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


def probe(specs: list[dict]) -> dict:
    """Two seeds per arm; verifies worker-written schema and basic binding."""
    out = {}
    required = (
        "c_ideal_fin", "c_ideal_tm", "explored_fin", "harvested_fin",
        "ideal_loss_cum", "n_final",
    )
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
        "schema": "adr0003_remeasure_v1",
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "clock": {
            "step_years": 0.5,
            "gamma": float(CANONICAL_VANISHING_FACTOR),
            "g_values": list(G_VALUES),
            "exit_modes": list(EXITS),
            "tau_max": TAU_MAX,
            "canonical_g": float(CANONICAL_AGENT_GENERATION_TIME),
            "horizons": "H=200 core; H=400 turnover and Figure 1 M1 tiers",
        },
        "arms": {
            "sensitivity": "c x landscape x g, fixed_span, H=200",
            "welfare": "c x visibility x legibility x g, fixed_span, H=200",
            "career_flows": "c x g x exit, mgstd5, H=200, career measures",
            "prestige": "stock x c x landscape x g, fixed_span, H=200",
            "turnover": "c x landscape x g x exit, H=400",
            "interventions": "spawn / sabbatical / refill compact sub-arms, H=200",
            "path_b": "lambda x pull x g, logit+equal_split, mgstd5, H=200",
            "fig1_xi": "dense xi x five tiers, mgstd5",
        },
        "seeds": len(SEEDS),
        "cells": len(all_specs),
        "runs_gross": len(all_specs) * len(SEEDS),
        "source_snapshot_before": src_before,
        "reductions": (
            "compact core panel: interventions lotteries (alpha) and the full "
            "48-cell Path B formulation/credit cross are not re-run; welfare "
            "keeps both visibility levels and both landscape legs; prestige "
            "keeps all three stocks and both landscapes."
        ),
    }
    (CENTRAL / "design.json").write_text(json.dumps(design, indent=1))
    preflight_snapshot(CENTRAL / "process_snapshot.txt")

    gross = len(all_specs) * len(SEEDS)
    log(
        f"wave start: arms={sorted({s['arm'] for s in all_specs})} "
        f"cells={len(all_specs)} runs={gross} seeds/cell={len(SEEDS)} "
        f"n_jobs={args.n_jobs}"
    )

    t0 = time.time()
    states = []
    for i, spec in enumerate(all_specs, 1):
        states.append(run_cell(spec, args.n_jobs, log))
        if i % 20 == 0:
            log(f"progress {i}/{len(all_specs)} cells, "
                f"elapsed={time.time() - t0:.0f}s")

    n_ok = sum(1 for s in states if s["status"] == "ok")
    n_part = sum(1 for s in states if s["status"] == "ok_with_failures")
    n_crash = sum(1 for s in states if s["status"] == "crashed")
    runs_ok = sum(s["seeds_ok"] for s in states)
    runs_fail = sum(s["seeds_failed"] for s in states)
    wall = time.time() - t0
    src_after = snapshot_sources()
    moved = [f for f in SOURCE_FILES if src_before[f] != src_after[f]]
    manifest = {
        "schema": "adr0003_remeasure_v1",
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
