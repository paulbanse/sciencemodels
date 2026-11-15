"""Shared machinery for the Path B v2 round (movement coupling).

ADR-0002: one home per measure. The probe, the sweep driver and the analyser
all import from here, so the cell tag, the parameter block and the per-run
extraction cannot drift between them.

NOT CALIBRATED: model-internal only (owner ruling 2026-11-13). The Klann
moments (team size mean 2.65 / median 2, repeated-collab share 0.34) are
diagnostic reference values for the Gate 2 evaluation, never fitting targets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sciencemodels import landscapes
from sciencemodels.model import MyModel

#: landscape bundles mirroring the wave-1/2 cells and Path B v1
LANDSCAPE_BUNDLES = {
    "mgstd5": (landscapes.multipleGaussians, {"std": 5, "number_gaussians": 2}),
    "noisypr5": (
        landscapes.noisyGaussian,
        {"prop_random": 5, "std": 5, "std_curiosity": 0.1},
    ),
}

#: fixed parameters of the family (identical to Path B v1's BASE_PARAMS plus
#: the new coupling, which is passed per cell)
BASE_PARAMS = {
    "n_agents": 20,
    "initial_curiosity": 0.5,
    "epsilon": 0.1,
    "harvest": 0.02,
    "grid_size": 20,
    "use_distance": True,
    "agent_generation_time": 0.1,
    "new_questions": 0,
    "agent_generation_function": lambda cur, _rng, _params: cur,
    "vanishing_factor_for_prestige": 0.9,
    "use_visibility_reward": True,
    "knowledge_normalization": "share",
    "prestige_accrual": "harvested",
    "entry_prestige_stock": "prestige",
    "dyadic_proposal_regime": "newcomer",
    "closeness_mode": "grid",
    "dyadic_visibility_stock": "prestige_visibility",
}

STEP_LIMIT = 200
SEEDS = 20

#: constants that never reach the CSV as such
DESIGN_PINS = {
    "n_agents": 20,
    "grid_size": 20,
    "step_limit": STEP_LIMIT,
    "epsilon": 0.1,
    "harvest": 0.02,
    "initial_curiosity": 0.5,
    "agent_generation_time": 0.1,
    "vanishing_factor_for_prestige": 0.9,
    "use_distance": True,
    "use_visibility_reward": True,
    "new_questions": 0,
    "knowledge_normalization": "share",
    "prestige_accrual": "harvested",
    "entry_prestige_stock": "prestige",
    "agent_generation_function": "identity (curiosity pinned at 0.5)",
    "dyadic_proposal_regime": "newcomer",
    "closeness_mode": "grid",
    "dyadic_visibility_stock": "prestige_visibility",
    "seed_semantics": "agent_seed 0..19 drives agent and landscape RNG jointly (model injects agent_seed into generation_params)",
    "canary_anchor_mean_prestige": 0.025399760810316363,
    "team_definition": "co-harvest groups: agents sharing a tile at end of step, size >= 2; newborns count with their supervisor (they land on the supervisor's tile)",
    "repeated_definition": "per step, share of active co-harvest pairs seen active in any earlier step; 0.0 when no active pairs",
    "move_pull_definition": "reward multiplier 1 + dyadic_move_pull * mean_j(1/(torus_distance(pos, partner_j)+1)) over the agent's declared links; links are updated at end of step, so the pull reads the previous step's links (one-step lag)",
}


def cell_tag(lam, pull, form, credit, land):
    """Run-tagged cell name carrying every varied parameter."""
    return "cell_lam%03d_pull%03d_%s_%s_%s" % (
        int(round(lam * 100)),
        int(round(pull * 100)),
        form,
        "struct" if credit == "structural" else "equal",
        land,
    )


def build_params(lam, pull, form, credit, land, seed):
    """Constructor kwargs for one cell run."""
    f, gp = LANDSCAPE_BUNDLES[land]
    p = dict(BASE_PARAMS)
    p.update(
        cell_init_function=f,
        generation_params=dict(gp),
        agent_seed=seed,
        step_limit=STEP_LIMIT,
        dyadic_mode=form,
        dyadic_lambda=lam,
        dyadic_move_pull=pull,
        credit_sharing=credit,
    )
    return p


def report_extras(m, row):
    """Driver-side per-run columns from the model's own datacollector.

    One home per measure: the model reports per-step values, this function
    aggregates the trajectory (time means, mass of degenerate repeated-share
    steps, cumulative ideal loss).
    """
    dc = m.datacollector.get_model_vars_dataframe()
    row["agents_final"] = len(m._scientists)
    row["newcomers_final"] = int(sum(1 for a in m._scientists if a.is_newcomer))
    for src, dst in (
        ("mean_team_size", "team_tm"),
        ("repeated_collab_share", "rep_tm"),
        ("dyadic_assortativity_prestige", "assort_tm"),
        ("n_active_links", "links_tm"),
        ("link_repeated_share", "link_rep_tm"),
        ("link_assortativity_prestige", "link_assort_tm"),
        ("Corr_prestige_ideal_merit", "c_ideal_tm"),
        ("Corr_prestige_local_merit", "c_local_tm"),
    ):
        row[dst] = float(dc[src].mean())
    row["rep_mass0"] = float((dc["repeated_collab_share"] == 0).mean())
    row["rep_mass1"] = float((dc["repeated_collab_share"] == 1).mean())
    row["ideal_loss_cum"] = -float(
        sum(getattr(a, "ideal_merit_cum", 0.0) for a in m._scientists)
    )
    row["ideal_loss_mean"] = -float(
        np.mean([getattr(a, "ideal_merit_cum", 0.0) for a in m._scientists])
    )
    return row


def run_one(lam, pull, form, credit, land, seed):
    """One seeded run of one cell; returns the report row (a dict).

    The four varied parameters are written INTO the row, not only into the
    filename: Path B v1 carried `dyadic_lambda` in the cell tag alone, so a
    pairing key had to be reconstructed from filenames. Columns here make the
    cell identity part of the artifact.
    """
    m = MyModel(**build_params(lam, pull, form, credit, land, seed))
    for _ in range(STEP_LIMIT):
        m.step(True)
    row = report_extras(m, m.get_report())
    row["dyadic_lambda"] = float(lam)
    row["dyadic_move_pull"] = float(pull)
    row["formulation"] = form
    row["credit_sharing_axis"] = credit
    row["landscape"] = land
    return row


#: report columns the analysis consumes (asserted against the worker CSV)
ANALYSIS_KEYS = (
    "agent_seed",
    "team_tm",
    "rep_tm",
    "links_tm",
    "assort_tm",
    "link_rep_tm",
    "c_ideal_tm",
    "c_local_tm",
    "ideal_loss_cum",
    "rep_mass0",
    "rep_mass1",
)


def assert_schema(df: pd.DataFrame, where: str) -> None:
    """Schema proof: every analysis key must be in a file read from disk."""
    missing = [k for k in ANALYSIS_KEYS if k not in df.columns]
    if missing:
        raise AssertionError(f"{where}: missing analysis keys {missing}")
    if "Mean_of_Corr_prestige_ideal_merit" not in df.columns:
        raise AssertionError(f"{where}: missing the ideal-merit reporter")
