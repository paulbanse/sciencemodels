"""Turnover probe v1 (P-0006): stationary-population demography.

The model has no exit rule at all -- every scientist lives forever, so the
population grows as ``20 + floor(g * t)`` and the founders sit on the peaks they
depleted early in the run. P-0006 asks whether the career-flow findings survive
once generations actually turn over.

THE AXIS IS DRIVER-SIDE. ``sciencemodels/model.py`` is outside this round's
write boundary, and the model does implement a retirement rule of its own, so
each exit mode is applied by wrapping ``MyModel.__generate_new_agents`` from
this driver (see ``parameter-sweep-rounds``,
``references/driver-side-model-extensions.md``). The consequences, stated here
rather than in a footnote:

* the wrapper is installed INSIDE the task function, idempotently per process,
  because a module-level install does not survive the pickler into a joblib
  worker;
* ``exit_mode = "none"`` runs the original method and only stamps a
  ``birth_step`` attribute on the newborns, so it must be bit-identical to the
  unpatched model -- proven at runtime in ``identity_proof()`` and recorded in
  ``design.json``;
* a divergence canary asserts the source lines the wrapper relies on are still
  the ones in the tree, and fails loudly naming the file to re-derive;
* the numbers describe THIS DRIVER. A future model-side exit rule with the same
  semantics would reproduce them; one with different semantics would not.

Design: 3 exit modes x 3 entry rates x 3 curiosities x 2 landscapes = 54 cells
x 20 seeds = 1,080 runs at horizon 400. Checkpointed per cell (atomic write,
row-count verified). Model-internal; NOT calibrated (owner ruling 2026-11-13).

Outputs under ``data/turnover_v1/``:
    cells/<tag>.csv   20 rows, one per seed, terminal-state metrics
    traj/<tag>.csv    one row per step, seed means of the trajectory
    design.json       grid, constants, axis_source, proofs
    manifest.json     gross vs net counts
    run_log.txt       one line per completed cell
"""
from __future__ import annotations

import inspect
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from joblib import Parallel, delayed  # noqa: E402

from sciencemodels import landscapes as lf  # noqa: E402
from sciencemodels.model import MyModel  # noqa: E402

OUT = os.path.join(ROOT, "data", "turnover_v1")
CELLS = os.path.join(OUT, "cells")
TRAJ = os.path.join(OUT, "traj")

DRIVER_VERSION = "turnover_v1 / driver 1"
HORIZON = 400
SEEDS = 20
TAU_MAX = 200          # tenure cap for the age_hazard mode, in steps
N_KEEP = 20            # target population for fixed_span (the founder count)
T_WINDOW = 20          # "recent arrival" window: one entry generation at g=0.05


def _nanmean(values):
    """Mean that returns NaN (not a warning) when nothing is defined."""
    a = np.asarray(values, dtype=float)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")

MODES = ("none", "fixed_span", "age_hazard")
ENTRY = (0.05, 0.10, 0.20)
CURIOSITY = (0.3, 0.5, 0.7)

LAND = {
    "mgstd5": (lf.multipleGaussians, {"std": 5, "number_gaussians": 2}),
    "noisypr5": (
        lf.noisyGaussian,
        {"prop_random": 5, "std": 5, "std_curiosity": 0.1},
    ),
}

# lines the wrapper depends on. If the model moves, these change and the
# canary refuses instead of silently measuring stale code.
CANARY_TOKENS = (
    "agents_to_generate",
    "entry_stock = self.entry_prestige_stock",
    "self._scientists.append(a)",
)


def identity(cur, rng, params):
    return cur


# ---------------------------------------------------------------------------
# Driver-side exit rules
# ---------------------------------------------------------------------------
def _retire(model, agent):
    """Remove an agent from the active roster, the grid and the AgentSet."""
    if agent in model._scientists:
        model._scientists.remove(agent)
    try:
        model.grid.remove_agent(agent)
    except Exception:
        pass
    try:
        agent.remove()
    except Exception:
        pass


def _spawn_from_roulette(model, agent_generation_function):
    """One newcomer via the model's own supervisor-proportional rule.

    Byte-faithful to the body of ``__generate_new_agents`` except that it draws
    exactly one newcomer: same roulette over the same ``entry_prestige_stock``,
    same single ``rng.uniform`` per newcomer, same placement at the supervisor's
    position. Keeping the RNG call sequence identical is what lets results
    transfer to a real model-side implementation with the same semantics.
    """
    entry_stock = model.entry_prestige_stock
    tuples = [(getattr(a, entry_stock), a) for a in model._scientists]
    total = sum(getattr(a, entry_stock) for a in model._scientists)
    val = model.rng.uniform(0, total)
    sup = model._scientists[0] if model._scientists else None
    for p, a in tuples:
        if val <= p:
            sup = a
            break
        val -= p
    w = agent_generation_function(model.initial_curiosity, model.rng, model.generation_params)
    from sciencemodels.agents import Scientist

    a = Scientist(model, w, model.epsilon)
    a.ideal_merit_cum = np.float64(0.0)
    a.is_newcomer = True
    a.birth_step = model.steps
    model.grid.place_agent(a, sup.pos)
    model._scientists.append(a)
    return a


def install_exit_rule(mode, agent_generation_function):
    """Install the wrapper for ``mode``, re-installing when the mode changes.

    A joblib worker handles cells of several modes over its lifetime, so the
    wrapper is rebuilt from the pristine original each time the requested mode
    differs. ``__generate_new_agents`` is captured ONCE into
    ``MyModel._turnover_original``: wrapping the wrapper would stack exit rules,
    which is the classic way a driver-side patch silently double-counts.
    """
    if mode not in MODES:
        raise ValueError("unknown exit mode %r" % (mode,))

    # name-mangled: inside the class it is __generate_new_agents, from outside
    # the attribute is _MyModel__generate_new_agents
    attr = "_MyModel__generate_new_agents"
    original = getattr(MyModel, "_turnover_original", None)
    if original is None:
        original = getattr(MyModel, attr)
        src = inspect.getsource(original)
        missing = [t for t in CANARY_TOKENS if t not in src]
        if missing:
            raise RuntimeError(
                "divergence canary: sciencemodels/model.py no longer contains %r "
                "in __generate_new_agents; re-derive the driver-side wrapper "
                "before running (file: sciencemodels/model.py)" % (missing,)
            )
        MyModel._turnover_original = original

    if getattr(MyModel, "_turnover_mode", None) == mode:
        return

    def patched(self):
        n_before = len(self._scientists)
        original(self)
        born = self._scientists[n_before:]
        for a in born:
            a.birth_step = self.steps

        if mode == "none":
            return

        if mode == "fixed_span":
            # every arrival retires the oldest active scientist: N stays at the
            # founder count, so the population is stationary by construction.
            for _ in born:
                if len(self._scientists) > N_KEEP:
                    victim = max(
                        self._scientists, key=lambda a: (a.age, -a.unique_id)
                    )
                    _retire(self, victim)
            return

        if mode == "age_hazard":
            due = [
                a
                for a in list(self._scientists)
                if self.steps - getattr(a, "birth_step", 0) >= TAU_MAX
            ]
            for a in due:
                _retire(self, a)
                _spawn_from_roulette(self, agent_generation_function)

    patched._turnover_mode = mode
    setattr(MyModel, attr, patched)
    MyModel._turnover_mode = mode


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------
def run_one(mode, g, cur, land, seed):
    install_exit_rule(mode, identity)
    f, gp = LAND[land]
    m = MyModel(
        n_agents=N_KEEP,
        initial_curiosity=cur,
        epsilon=0.1,
        harvest=0.02,
        grid_size=20,
        cell_init_function=f,
        generation_params=dict(gp),
        use_distance=True,
        agent_generation_time=g,
        new_questions=0,
        agent_seed=seed,
        step_limit=HORIZON,
        agent_generation_function=identity,
        vanishing_factor_for_prestige=0.9,
        use_visibility_reward=True,
        knowledge_normalization="share",
        prestige_accrual="harvested",
    )
    # founders are the generation-0 cohort
    for a in m._scientists:
        a.birth_step = 0

    gap, gap_recent, n_active, n_founders, mean_age, mean_tenure = [], [], [], [], [], []
    for _ in range(HORIZON):
        m.step(True)
        sci = m._scientists
        n_active.append(len(sci))
        n_founders.append(sum(1 for a in sci if getattr(a, "birth_step", 0) == 0))
        mean_age.append(float(np.mean([a.age for a in sci])))
        ten = [m.steps - getattr(a, "birth_step", 0) for a in sci]
        mean_tenure.append(float(np.mean(ten)))

        # (i) the career-flows definition: the permanent is_newcomer flag. It is
        # only meaningful while an incumbent generation still exists -- once all
        # founders have retired every agent carries the flag, so this contrasts
        # nothing. Recorded for comparability with career_flows_v1, not as the
        # headline.
        new = [a.prestige for a in sci if getattr(a, "is_newcomer", False)]
        old = [a.prestige for a in sci if not getattr(a, "is_newcomer", False)]
        gap.append(float(np.mean(new) - np.mean(old)) if new and old else np.nan)

        # (ii) the measure that survives turnover: recent arrivals against
        # established agents, defined by tenure rather than by a permanent flag.
        recent = [a.prestige for a, t in zip(sci, ten) if t < T_WINDOW]
        est = [a.prestige for a, t in zip(sci, ten) if t >= T_WINDOW]
        gap_recent.append(
            float(np.mean(recent) - np.mean(est)) if recent and est else np.nan
        )

    dc = m.datacollector.get_model_vars_dataframe()
    last = dc.iloc[-1]
    merits = [float(a.ideal_Merit) for a in m.agents]

    # first step at which no founder is still active (None if never)
    retired_step = next((i + 1 for i, k in enumerate(n_founders) if k == 0), None)

    row = dict(
        exit_mode=mode,
        entry=g,
        curiosity=cur,
        landscape=land,
        seed=seed,
        horizon=HORIZON,
        n_final=int(n_active[-1]),
        n_founders_final=int(n_founders[-1]),
        founders_retired_step=retired_step,
        mean_age_final=float(mean_age[-1]),
        mean_tenure_final=float(mean_tenure[-1]),
        gap_flag_final=gap[-1],
        gap_flag_mean_last100=_nanmean(gap[-100:]),
        gap_recent_final=gap_recent[-1],
        gap_recent_mean_last100=_nanmean(gap_recent[-100:]),
        n_mean_last100=float(np.mean(n_active[-100:])),
        c_ideal_fin=float(last.get("Corr_prestige_ideal_merit", np.nan)),
        c_local_fin=float(last.get("Corr_prestige_local_merit", np.nan)),
        c_ideal_tm=float(dc["Corr_prestige_ideal_merit"].mean()),
        c_local_tm=float(dc["Corr_prestige_local_merit"].mean()),
        top10_fin=float(last.get("top10pct_prestige_share", np.nan)),
        harvested_fin=float(last.get("percentage_knowledge_harvested", np.nan)),
        ideal_merit_sum=float(np.sum(merits)),
        ideal_merit_mean=float(np.mean(merits)),
        mean_prestige_final=float(last.get("mean_prestige", np.nan)),
    )
    # the model collects once at init as well, so dc carries HORIZON + 1 rows;
    # drop the init row so the series line up with steps 1..HORIZON
    dcs = dc.iloc[-HORIZON:]
    traj = pd.DataFrame(
        {
            "step": np.arange(1, HORIZON + 1),
            "gap": gap,
            "gap_recent": gap_recent,
            "n_active": n_active,
            "n_founders": n_founders,
            "mean_age": mean_age,
            "mean_tenure": mean_tenure,
            "c_ideal": dcs["Corr_prestige_ideal_merit"].to_numpy(),
            "c_local": dcs["Corr_prestige_local_merit"].to_numpy(),
            "top10": dcs["top10pct_prestige_share"].to_numpy(),
        }
    )
    return row, traj


# ---------------------------------------------------------------------------
# One cell
# ---------------------------------------------------------------------------
def cell_tag(spec):
    mode, g, cur, land = spec
    return "cell_%s_g%02d_cur%02d_%s" % (
        mode,
        int(round(g * 100)),
        int(round(cur * 10)),
        land,
    )


def run_cell(spec):
    mode, g, cur, land = spec
    tag = cell_tag(spec)
    cpath = os.path.join(CELLS, tag + ".csv")
    tpath = os.path.join(TRAJ, tag + ".csv")
    if os.path.exists(cpath) and sum(1 for _ in open(cpath)) == SEEDS + 1:
        return tag, "skipped"
    rows, trajs = [], []
    for s in range(SEEDS):
        row, traj = run_one(mode, g, cur, land, s)
        rows.append(row)
        trajs.append(traj)
    df = pd.DataFrame(rows)
    assert len(df) == SEEDS, tag + " incomplete"
    tmp = cpath + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, cpath)

    # seed means of the trajectory
    tj = pd.concat(trajs, ignore_index=True)
    agg = tj.groupby("step", as_index=False).mean(numeric_only=True)
    agg.insert(0, "cell_tag", tag)
    ttmp = tpath + ".tmp"
    agg.to_csv(ttmp, index=False)
    os.replace(ttmp, tpath)

    with open(os.path.join(OUT, "run_log.txt"), "a") as fh:
        fh.write("%s ok seeds=%d\n" % (tag, len(df)))
    return tag, "ok"


# ---------------------------------------------------------------------------
# Proofs
# ---------------------------------------------------------------------------
def identity_proof():
    """exit_mode='none' must be bit-identical to the unpatched model."""
    base = dict(
        n_agents=N_KEEP, initial_curiosity=0.5, epsilon=0.1, harvest=0.02,
        grid_size=20, cell_init_function=LAND["mgstd5"][0],
        generation_params=dict(LAND["mgstd5"][1]), use_distance=True,
        agent_generation_time=0.10, new_questions=0, agent_seed=7,
        step_limit=120, agent_generation_function=identity,
        vanishing_factor_for_prestige=0.9, use_visibility_reward=True,
        knowledge_normalization="share", prestige_accrual="harvested",
    )
    # 1) unpatched (reset first, so the proof is repeatable in one process)
    if getattr(MyModel, "_turnover_mode", None) is not None:
        setattr(MyModel, "_MyModel__generate_new_agents", MyModel._turnover_original)
        MyModel._turnover_mode = None
    clean = MyModel(**base)
    for _ in range(120):
        clean.step(True)
    ref = clean.get_report()

    # 2) patched, mode none
    install_exit_rule("none", identity)
    patched = MyModel(**base)
    for _ in range(120):
        patched.step(True)
    got = patched.get_report()

    diffs = {}
    for k in sorted(set(ref) | set(got)):
        a, b = ref.get(k), got.get(k)
        if isinstance(a, (int, float, np.floating, np.integer)) and isinstance(
            b, (int, float, np.floating, np.integer)
        ):
            if not (np.isnan(a) and np.isnan(b)) and float(a) != float(b):
                diffs[k] = [float(a), float(b)]
        elif a != b:
            diffs[k] = [str(a), str(b)]
    return {
        "keys_compared": len(set(ref) | set(got)),
        "differing_keys": diffs,
        "bit_identical": not diffs,
        "source_tokens_present": {
            t: (t in inspect.getsource(MyModel._turnover_original))
            for t in CANARY_TOKENS
        },
    }


def binding_proof():
    """Mode 'none' and 'fixed_span' must differ at fixed seed and parameters."""
    out = {}
    for mode in ("none", "fixed_span"):
        row, _ = run_one(mode, 0.20, 0.5, "mgstd5", 3)
        out[mode] = {k: row[k] for k in ("n_final", "n_founders_final", "mean_age_final")}
    out["differs"] = out["none"] != out["fixed_span"]
    return out


def cells():
    return [
        (m, g, c, l)
        for m, g, c, l in itertools.product(MODES, ENTRY, CURIOSITY, LAND)
    ]


if __name__ == "__main__":
    os.makedirs(CELLS, exist_ok=True)
    os.makedirs(TRAJ, exist_ok=True)

    # canary first
    canary = os.system(
        "MPLBACKEND=Agg %s %s > %s/canary.txt 2>&1"
        % (sys.executable, os.path.join(ROOT, "tests", "test_smoke.py"), OUT)
    )
    assert canary == 0, "canary failed - aborting before any run"

    t_proof = time.time()
    ident = identity_proof()
    assert ident["bit_identical"], "IDENTITY PROOF FAILED: %r" % (ident["differing_keys"],)
    bind = binding_proof()
    assert bind["differs"], "BINDING PROOF FAILED: mode does not change the state"
    proofs_s = round(time.time() - t_proof, 1)

    # a fresh process is required after the identity proof installed mode 'none'
    grid = cells()
    json.dump(
        {
            "round": "turnover_v1 (P-0006)",
            "driver_version": DRIVER_VERSION,
            "axis_source": (
                "driver-side wrapper around MyModel.__generate_new_agents; the "
                "model implements no exit rule of its own and is outside this "
                "round's write boundary"
            ),
            "grid": "3 exit modes x 3 entry rates x 3 curiosities x 2 landscapes",
            "cells_gross": len(grid),
            "runs_gross": len(grid) * SEEDS,
            "horizon": HORIZON,
            "seeds": SEEDS,
            "tau_max": TAU_MAX,
            "n_keep": N_KEEP,
            "fixed_parameters": {
                "epsilon": 0.1, "harvest": 0.02, "grid_size": 20,
                "use_distance": True, "new_questions": 0,
                "vanishing_factor_for_prestige": 0.9,
                "use_visibility_reward": True,
                "knowledge_normalization": "share",
                "prestige_accrual": "harvested",
            },
            "identity_proof": ident,
            "binding_proof": bind,
            "proofs_s": proofs_s,
            "canary": "tests/test_smoke.py exit 0 (see canary.txt)",
        },
        open(os.path.join(OUT, "design.json"), "w"),
        indent=1,
    )
    print("identity bit-identical:", ident["bit_identical"],
          "| binding differs:", bind["differs"])

    t0 = time.time()
    # joblib with loky re-imports the module in each worker, so the identity
    # patch installed above stays in the parent only. Each worker installs the
    # mode its own cell needs, inside run_one -> run_cell.
    results = Parallel(n_jobs=3, verbose=5)(delayed(run_cell)(c) for c in grid)
    done = sum(1 for _, st in results if st == "ok")
    skipped = sum(1 for _, st in results if st == "skipped")

    frames = [
        pd.read_csv(os.path.join(CELLS, f))
        for f in sorted(os.listdir(CELLS))
        if f.endswith(".csv")
    ]
    df = pd.concat(frames, ignore_index=True)
    json.dump(
        {
            "cells_gross": len(grid),
            "cells_new": done,
            "cells_skipped": skipped,
            "runs_gross": len(grid) * SEEDS,
            "runs_net_ok": int(df["c_ideal_fin"].notna().sum()),
            "wall_s": round(time.time() - t0, 1),
            "n_jobs": 3,
            "driver": DRIVER_VERSION,
        },
        open(os.path.join(OUT, "manifest.json"), "w"),
        indent=1,
    )
    print(
        "GROSS cells=%d runs=%d | NEW %d SKIPPED %d | NET ok=%d | wall=%.0fs"
        % (len(grid), len(grid) * SEEDS, done, skipped,
           int(df["c_ideal_fin"].notna().sum()), time.time() - t0)
    )
