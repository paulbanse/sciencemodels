"""Shared sweep machinery for the career-flow experiment family (ADR-0002).

ADR-0002 requires one home per measure and one home per parameter setup. The
career-flow rounds broke that: `scripts/career_flows/career_flows_driver.py`
and `scripts/career_flows_noisy/driver.py` each carried their own copy of the
fixed configuration, the landscape presets, the model-reporter list and the two
measurement functions (`run_one`, `career_measures_from_agent_vars`). The
noisy round's own report flags the duplication as a need
(`output/reports/career_flows_noisy_v1.md`, "One home for the measurement
definitions").

This module is that home. Both drivers import from here and keep only what is
arm-specific (their grids, their output folders, their aggregation steps), so
the two arms remain comparable by construction: the measurements are not
merely equivalent, they are the same code.

The consolidation is behaviour-preserving and was verified as such rather than
argued: three seeded cells (a canonical and a noisy landscape, two curiosity
levels, three entry rates, seeds 0/3/7, 200 steps) produce bit-identical
model payloads and bit-identical derived measures through this module and
through the pre-consolidation drivers (`md5` of the CSV serialisation, both
arms, before and after).

What stays out of this module on purpose: the arm-specific cell grids and
aggregation. They differ by design (the noisy arm varies legibility, the clean
arm varies demographics) and are not duplication.

It also holds ``BULK_SETUPS``, the single source of truth for the 8-setup
parametric design that ``scripts/bulk_runner.py`` and
``scripts/figures/run_fresh_sweep.py`` run (section below).
"""
from __future__ import annotations

import traceback
from typing import Any, Callable, NamedTuple

import numpy as np
import pandas as pd

#: Model-level reporters the career-flow analysis consumes. One list for both
#: arms, so a reporter added here reaches every arm's payload.
MODEL_TARGETS: list[str] = [
    "Corr_prestige_ideal_merit",
    "Corr_prestige_local_merit",
    "top10pct_prestige_share",
    "explored_percentage",
    "percentage_knowledge_harvested",
    "mean_age",
    "mean_prestige",
]

#: Model constants held fixed across the career-flow family. Anything an arm
#: varies must be a sweep axis, not an entry here, or the two arms stop being
#: comparable.
CAREER_FIXED: dict[str, Any] = {
    "n_agents": 20,
    "epsilon": 0.1,
    "harvest": 0.02,
    "grid_size": 20,
    "use_distance": True,
    "new_questions": 0,
    "use_visibility_reward": True,
    "knowledge_normalization": "share",
    "prestige_accrual": "harvested",
}

#: Landscape presets used by the career-flow family: the canonical
#: multiple-Gaussian bundle used by the clean arm and the noisy bundle the
#: legibility arm varies against. ``(cell_init_function name, generation_params)``.
CAREER_LANDSCAPES: dict[str, tuple[str, dict[str, Any]]] = {
    "mgstd5": ("multipleGaussians", {"std": 5, "number_gaussians": 2}),
    "noisypr5": ("noisyGaussian", {"prop_random": 5, "std": 5, "std_curiosity": 0.1}),
}


# ---------------------------------------------------------------------------
# The 8-setup bulk design: one home for the parametric sweep's setup table
# ---------------------------------------------------------------------------
#
# ``BULK_SETUPS`` is the single source of truth for the 8-setup design that
# ``scripts/bulk_runner.py`` and ``scripts/figures/run_fresh_sweep.py`` run:
# four landscape families, each in a swept-curiosity arm (even index, the
# curiosity axis runs 0.0 .. 1.0) and a fixed-curiosity arm (odd index,
# curiosity pinned at 0.5 with a uniform draw per newcomer). Both callers
# import this table; they differ only in repeats and step limit, which they
# pass in.


class BulkSetup(NamedTuple):
    """One row of the 8-setup bulk design.

    ``cell_init`` is the *name* of the landscape generator in
    :mod:`sciencemodels.landscapes`, not the function: importing this module
    stays free of ``landscapes`` (and hence matplotlib). ``curiosity_swept``
    says whether the setup sweeps ``initial_curiosity`` over
    :data:`BULK_CURIOSITY_SWEEP` (newcomers take the swept value unchanged) or
    fixes it at :data:`BULK_CURIOSITY_FIXED` (newcomers draw uniformly).
    """

    index: int
    family: str
    cell_init: str
    generation_params: dict[str, Any]
    curiosity_swept: bool


#: Curiosity axis of the swept setups: 11 values, 0.0 .. 1.0 in steps of 0.1.
#: A list, not a tuple: the production grid reads list-valued entries as axes.
BULK_CURIOSITY_SWEEP: list[float] = [k / 10 for k in range(11)]

#: Curiosity value of the fixed setups; newcomers draw uniformly around it.
BULK_CURIOSITY_FIXED: float = 0.5

#: The 8 setups in index order, so ``BULK_SETUPS[k].index == k``.
BULK_SETUPS: tuple[BulkSetup, ...] = (
    BulkSetup(0, "randominit", "randominit", {}, True),
    BulkSetup(1, "randominit", "randominit", {}, False),
    BulkSetup(
        2, "MultG2", "multipleGaussians", {"std": 5, "number_gaussians": 2}, True
    ),
    BulkSetup(
        3, "MultG2", "multipleGaussians", {"std": 5, "number_gaussians": 2}, False
    ),
    BulkSetup(
        4, "MultG6", "multipleGaussians", {"std": 5, "number_gaussians": 6}, True
    ),
    BulkSetup(
        5, "MultG6", "multipleGaussians", {"std": 5, "number_gaussians": 6}, False
    ),
    BulkSetup(
        6,
        "Noisy2",
        "noisyGaussian",
        {"prop_random": 5, "std": 5, "std_curiosity": 0.1},
        True,
    ),
    BulkSetup(
        7,
        "Noisy2",
        "noisyGaussian",
        {"prop_random": 5, "std": 5, "std_curiosity": 0.1},
        False,
    ),
)


def bulk_setup(index: int) -> BulkSetup:
    """The record for setup ``index``, with the index range checked."""
    if not 0 <= index < len(BULK_SETUPS):
        raise IndexError(f"no bulk setup {index}: the design has {len(BULK_SETUPS)}")
    return BULK_SETUPS[index]


def bulk_param_grid(setup: BulkSetup, step_limit: int = 400) -> dict[str, Any]:
    """The ``param_grid`` dict for one bulk setup, keys in the inherited order.

    ``step_limit`` is the one knob the two callers set differently
    (``scripts/bulk_runner.py`` runs 400 steps, the figures-v1 sweep 200);
    every other entry comes from the setup record.
    """
    from sciencemodels import landscapes  # deferred: landscapes imports matplotlib

    generation_function = getattr(landscapes, setup.cell_init)
    curiosities: list[float] | float
    if setup.curiosity_swept:
        curiosities = list(BULK_CURIOSITY_SWEEP)
        # Identity: the swept mu is the newcomer's curiosity unchanged.
        agent_generation_function = lambda cur, _rng, _params: cur
    else:
        curiosities = BULK_CURIOSITY_FIXED
        agent_generation_function = landscapes.uniform

    return {
        "n_agents": 20,  # minimum 3
        "initial_curiosity": curiosities,  # interest for exploration
        "use_distance": True,  # if True, agents take into account distance when choosing where to go
        "epsilon": 0.1,  # noise in the measure of utility
        "harvest": 0.02,  # proportion of science harvested at each iteration
        "grid_size": 20,
        "cell_init_function": generation_function,  # generation method for the landscape, see landscapes for the definitions
        "generation_params": dict(setup.generation_params),  # parameters for the generation method
        "agent_generation_time": 0,  # number of agent generated per generation, for example if 0.1, 10 new agent per generation
        "new_questions": 0,
        "agent_seed": 1,  # seed for the random of the agents
        "step_limit": step_limit,
        "agent_generation_function": agent_generation_function,  # if this is not defined the function is just mu
        "vanishing_factor_for_prestige": 0.9,
        "use_visibility_reward": True,
        "knowledge_normalization": "share",  # explicit pin so runs are self-describing
        "prestige_accrual": "harvested",  # explicit pin so runs are self-describing
    }


def career_measures_from_agent_vars(av: pd.DataFrame, model_vars: pd.DataFrame) -> pd.DataFrame:
    """Derive population/career measures from the agent-vars longitudinal frame.

    Parameters
    ----------
    av
        Mesa agent-vars dataframe, indexed by ``(Step, agent id)`` or already
        reset so that both levels appear as columns. The index level names are
        not assumed: the step level is ``Step`` when present, and the agent-id
        level is found among the usual names or taken as the remaining column.
    model_vars
        Model-vars dataframe with a ``Step`` column and
        ``percentage_knowledge_harvested``.

    Returns
    -------
    pandas.DataFrame
        One row per step with ``n_agents``, ``incumbent_mean_prestige``,
        ``newcomer_mean_prestige`` (newcomers are agents whose first
        observation is after step 0), ``prestige_gap``,
        ``prestige_gap_ratio``, and the harvest flows ``harvested_cum``,
        ``harvest_flow``, ``percap_harvest_cum``, ``percap_harvest_flow``.

    Notes
    -----
    The body is the pre-consolidation implementation, unchanged: both arms
    were verified to produce identical output before and after the move.
    """
    av = av.reset_index()
    # mesa 3.2: index levels are (Step, agent_id) - names may vary; find them
    step_col = "Step" if "Step" in av.columns else av.columns[0]
    id_col = None
    for cand in ("agent_id", "AgentID", "agent"):
        if cand in av.columns:
            id_col = cand
            break
    if id_col is None:
        # level name of the non-Step index level
        id_col = [c for c in av.columns if c != step_col][0]
        # fallthrough: if reset_index produced a level named 'Level_1' etc.
    entry = av.groupby(id_col)[step_col].min().rename("entry_step")
    av = av.merge(entry, left_on=id_col, right_index=True)
    av["is_newcomer"] = av["entry_step"] > 0
    g = av.groupby(step_col)
    pop = g.size().rename("n_agents")
    inc = av[~av.is_newcomer].groupby(step_col)["prestige"].mean().rename("incumbent_mean_prestige")
    new = av[av.is_newcomer].groupby(step_col)["prestige"].mean().rename("newcomer_mean_prestige")
    res = pd.concat([pop, inc, new], axis=1)
    res["newcomer_mean_prestige"] = res["newcomer_mean_prestige"].fillna(0.0)
    res["prestige_gap"] = res["newcomer_mean_prestige"] - res["incumbent_mean_prestige"]
    res["prestige_gap_ratio"] = np.where(
        res["incumbent_mean_prestige"] > 0,
        res["prestige_gap"] / res["incumbent_mean_prestige"], 0.0)
    harvested = model_vars.set_index("Step")["percentage_knowledge_harvested"]
    res["harvested_cum"] = harvested.reindex(res.index).fillna(0.0)
    res["harvest_flow"] = res["harvested_cum"].diff().fillna(0.0).clip(lower=0.0)
    res["percap_harvest_cum"] = res["harvested_cum"] / res["n_agents"]
    res["percap_harvest_flow"] = res["harvest_flow"] / res["n_agents"]
    return res.reset_index().rename(columns={step_col: "Step"})


def run_one(
    landscape_fn: Callable[..., np.float64],
    generation_params: dict[str, Any],
    curiosity: float,
    vanishing: float,
    gen_time: float,
    step_limit: int,
    seed: int,
    need_agent_vars: bool = True,
) -> tuple[str, Any, int | None, pd.DataFrame | None]:
    """One seeded career-flow run.

    Returns ``(status, payload, n_scientists_final, agent_vars)``, where
    ``status`` is ``"ok"`` or ``"fail"``; on failure ``payload`` carries the
    traceback so a failing seed is recorded rather than dropped silently
    (ADR-0001's gross-versus-net accounting). ``agent_vars`` is ``None`` when
    ``need_agent_vars`` is false, which is the clean arm's cheap path: the
    agent-level frame is large and only the arms that derive career measures
    need it.
    """
    from sciencemodels import MyModel

    try:
        model = MyModel(
            n_agents=CAREER_FIXED["n_agents"],
            initial_curiosity=curiosity,
            epsilon=CAREER_FIXED["epsilon"],
            harvest=CAREER_FIXED["harvest"],
            grid_size=CAREER_FIXED["grid_size"],
            cell_init_function=landscape_fn,
            generation_params=generation_params,
            use_distance=CAREER_FIXED["use_distance"],
            agent_generation_time=gen_time,
            new_questions=CAREER_FIXED["new_questions"],
            agent_seed=seed,
            step_limit=step_limit,
            vanishing_factor_for_prestige=vanishing,
            use_visibility_reward=CAREER_FIXED["use_visibility_reward"],
            knowledge_normalization=CAREER_FIXED["knowledge_normalization"],
            prestige_accrual=CAREER_FIXED["prestige_accrual"],
        )
        for _ in range(step_limit):
            model.step(True)
        mv = model.datacollector.get_model_vars_dataframe()
        out = mv[["Step"] + MODEL_TARGETS].copy()
        out.insert(0, "seed", seed)
        n_final = len(model._scientists)
        av = None
        if need_agent_vars:
            av = model.datacollector.get_agent_vars_dataframe()
        return "ok", out, n_final, av
    except Exception:
        return "fail", traceback.format_exc(), None, None
