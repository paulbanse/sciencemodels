"""Seeded smoke test for the sciencemodels core (ADR-0002).

Plain asserts, no pytest dependency. Run:

    MPLBACKEND=Agg /opt/hermes/.venv/bin/python tests/test_smoke.py

Canary run (3 agents, seeded, 30 steps) asserting:
1. the run completes and the end report carries the self-describing
   parameter keys (knowledge_normalization, prestige_accrual) and the
   harvested_*_step milestone keys;
2. a same-seed rerun is bit-identical;
3. a newcomer run (agent_generation_time=0.1, step_limit=20) completes with
   more than 3 scientists.
4. regression BUG-0002 (4-arg Gaussian);
5. regression BUG-0003 (append-mode header);
6. ideal-merit pairing audit: synthetic 3-agent case against a
   hand-computed greedy optimum (sorted-to-sorted pairing) plus the
   ideal_merit_cum accumulator;
7. seed identity columns (agent_seed, landscape_seed) in the report;
8. binding probe flags a mu-ignoring constant generator;
9. entry_prestige_stock knob: validation and both alternate stocks run.
10. Path B dyadic layer: off-mode bit-identity with every knob cranked plus
    the bit-level anchor (mean_prestige), both formulations binding across
    lambda, and validation of the six new knobs.
11. regression BUG-0006: unscored step-t newborns (ideal_Merit=NaN) are
    excluded from correlation_reporter rather than treated as ideal_Merit=0.
12. ADR-0003 backward compatibility: default ``exit_mode="none"`` is
    bit-identical to an explicit none run at the legacy gamma=0.9, keeps the
    population growing, and the clock adds +0.5 year per step.
13. ADR-0003 canonical turnover canary: ``exit_mode="fixed_span"``,
    ``g=0.25``, ``gamma=0.5**(1/10)`` and ``tau_max=80`` holds N=20,
    retires the founder cohort, and reports ``gap_recent``.
14. attraction-trias knob ``prestige_weight_in_visibility``: default None is
    bit-identical to the pre-edit code path (stored digests), w=0.0 matches
    ``use_visibility_reward=False``, w=1.0 matches
    ``use_visibility_reward=True``, w=0.5 binds cleanly, and values outside
    [0, 1] are refused.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from sciencemodels import MyModel, metrics
from sciencemodels.landscapes import Gaussian, multipleGaussians, uniform
from sciencemodels.model import (
    CANONICAL_AGENT_GENERATION_TIME,
    CANONICAL_TAU_MAX,
    CANONICAL_VANISHING_FACTOR,
    generate_data_parametric_exploration,
    probe_agent_generation_binding,
)

CANARY_PARAMS = {
    "n_agents": 3,
    "initial_curiosity": 0.5,
    "epsilon": 0.1,
    "harvest": 0.02,
    "grid_size": 10,
    "cell_init_function": multipleGaussians,
    "generation_params": {"seed": 4, "std": 5, "number_gaussians": 4},
    "use_distance": True,
    "agent_generation_time": 0,
    "new_questions": 0,
    "agent_seed": 1,
    "step_limit": 30,
    "agent_generation_function": uniform,
    "vanishing_factor_for_prestige": 0.5,
    "use_visibility_reward": True,
}

REQUIRED_REPORT_KEYS = (
    "knowledge_normalization",
    "prestige_accrual",
    "harvested_10_step",
    "harvested_25_step",
)


def bits(report):
    """Exact (bit-level) snapshot of a report dict for identity comparison."""
    return sorted((k, repr(v)) for k, v in report.items())


def main():
    # 1. seeded canary run completes and reports the required keys
    model = MyModel(**CANARY_PARAMS)
    report = model.run_mode_for_bulk()
    for key in REQUIRED_REPORT_KEYS:
        assert key in report, f"report missing required key: {key}"
    assert report["knowledge_normalization"] == "share"
    assert report["prestige_accrual"] == "harvested"
    print(f"canary run OK: {len(report)} report keys, "
          f"harvested_10_step={report['harvested_10_step']}, "
          f"harvested_25_step={report['harvested_25_step']}")

    # 2. same-seed rerun is bit-identical
    rerun = MyModel(**CANARY_PARAMS)
    report_rerun = rerun.run_mode_for_bulk()
    assert bits(report) == bits(report_rerun), "same-seed rerun diverged"
    print("same-seed rerun bit-identical OK")

    # 3. newcomer run completes with more than 3 scientists
    newcomer_params = dict(CANARY_PARAMS, agent_generation_time=0.1, step_limit=20)
    newcomer_model = MyModel(**newcomer_params)
    newcomer_model.run_mode_for_bulk()
    n_scientists = len(newcomer_model._scientists)
    assert n_scientists > 3, f"expected newcomers, got {n_scientists} scientists"
    print(f"newcomer run OK: {n_scientists} scientists after 20 steps")

    # 4. regression BUG-0002: the landscape family takes rng=None, so the
    #    legacy 4-argument call site works; Gaussian rebuilds from args["seed"]
    gaussian_value = Gaussian(5, 5, 10, {"seed": 4, "std": 5})
    assert isinstance(gaussian_value, float), type(gaussian_value)
    assert np.isfinite(gaussian_value), gaussian_value
    print(f"4-arg Gaussian OK: {gaussian_value!r} ({type(gaussian_value).__name__})")

    # 5. regression BUG-0003: an append-mode run appends rows only - the
    #    header check must test the path that is actually written (.csv)
    with tempfile.TemporaryDirectory(prefix="smoke_append_", dir="/tmp") as tmp:
        target = os.path.join(tmp, "append_header_check")
        grid = dict(CANARY_PARAMS, step_limit=5)
        generate_data_parametric_exploration(
            target, grid, repeats_per_setting=1, intention="w"
        )
        generate_data_parametric_exploration(
            target, grid, repeats_per_setting=1, intention="a"
        )
        with open(target + ".csv", newline="") as fh:
            lines = fh.read().splitlines()
    header, data_rows = lines[0], lines[1:]
    assert lines.count(header) == 1, f"header appears {lines.count(header)} times"
    assert len(data_rows) == 2, f"expected 2 data rows, got {len(data_rows)}"
    print(
        f"append-mode header check OK: {len(lines)} lines = "
        f"1 header + {len(data_rows)} data rows"
    )

    # 6. ideal-merit pairing audit (synthetic, hand-computed optimum):
    #    3 agents on a 4x4 field, harvest 0.5. Peak 1.0 at (1,1) holds A
    #    and C (stacked), B sits on the 0.4 tile at (0,3). Actual harvests
    #    (position-sorted stacking): B 0.20, A 0.50, C 0.25. Greedy ideal
    #    (3 depleting slots): 0.50, 0.25, 0.25. Sorted-to-sorted losses:
    #    A 0.00, C 0.00, B -0.05; the stacked site averages to 0.00 each.
    #    (Pre-fix head-pairing gave A=C=-0.125, B=-0.30.)
    class _FakeLayer:
        def __init__(self, data):
            self.data = data

    class _FakeGrid:
        def __init__(self, prev, know):
            self.properties = {
                "previous_knowledge": _FakeLayer(prev),
                "knowledge": _FakeLayer(know),
            }

    class _FakeAgent:
        def __init__(self, pos):
            self.pos = pos
            # Scientist (agents.py) presets ideal_Merit=0 but defines no
            # ideal_merit_cum - the fake mirrors that so the accumulator's
            # lazy init is exercised on the real code path.
            self.ideal_Merit = 0

    class _FakeModel:
        def __init__(self, agents, prev, know, harvest):
            self.agents = agents
            self.number_agents = len(agents)
            self.grid = _FakeGrid(prev, know)
            self.harvest = harvest

    field = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 1.0, 0.2, 0.1],
            [0.3, 0.2, 0.4, 0.5],
            [0.1, 0.1, 0.2, 0.3],
        ],
        dtype=np.float64,
    )
    A, B, C = _FakeAgent((1, 1)), _FakeAgent((0, 3)), _FakeAgent((1, 1))
    fake = _FakeModel([A, B, C], field.copy(), field.copy(), np.float64(0.5))
    metrics.update_ideal_merit(fake)
    assert np.isclose(A.ideal_Merit, 0.0), A.ideal_Merit
    assert np.isclose(C.ideal_Merit, 0.0), C.ideal_Merit
    assert np.isclose(B.ideal_Merit, -0.05), B.ideal_Merit
    assert np.allclose(
        fake.grid.properties["previous_knowledge"].data, field
    ), "previous_knowledge not refreshed from knowledge"
    # second call on the refreshed field reproduces the same losses and
    # accumulates them (welfare-round need: cumulative accumulator)
    metrics.update_ideal_merit(fake)
    assert np.isclose(A.ideal_merit_cum, 0.0), A.ideal_merit_cum
    assert np.isclose(C.ideal_merit_cum, 0.0), C.ideal_merit_cum
    assert np.isclose(B.ideal_merit_cum, -0.1), B.ideal_merit_cum
    print(
        f"ideal-merit pairing audit OK: A={A.ideal_Merit}, C={C.ideal_Merit},"
        f" B={B.ideal_Merit} (hand-computed 0.0, 0.0, -0.05);"
        f" cum after 2 calls: A={A.ideal_merit_cum}, B={B.ideal_merit_cum}"
    )

    # 7. seed identity columns in the end-of-run report (welfare-round
    #    need): the report carries agent_seed and the landscape seed (the
    #    model injects agent_seed into generation_params, so the two
    #    columns agree by construction)
    assert "agent_seed" in report, "report missing agent_seed"
    assert report["agent_seed"] == CANARY_PARAMS["agent_seed"], report["agent_seed"]
    assert "landscape_seed" in report, "report missing landscape_seed"
    assert report["landscape_seed"] == report["agent_seed"]
    print(
        f"seed columns OK: agent_seed={report['agent_seed']},"
        f" landscape_seed={report['landscape_seed']}"
    )

    # 8. binding probe for agent_generation_function (welfare-round
    #    need): a constant generator is flagged inert; identity binds
    probe_rng = np.random.default_rng(0)
    assert (
        probe_agent_generation_binding(lambda mu, r, p: 0.7, probe_rng, {})
        is True
    ), "constant generator not flagged inert"
    assert (
        probe_agent_generation_binding(lambda mu, r, p: mu, probe_rng, {})
        is False
    ), "identity generator wrongly flagged inert"
    print("binding probe OK: constant generator flagged inert, identity binds")

    # 9. entry-prestige-stock knob: invalid values raise; both alternate
    #    stocks complete a seeded run with entry and reach the report
    try:
        MyModel(**dict(CANARY_PARAMS, entry_prestige_stock="bogus"))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid entry_prestige_stock accepted")
    for stock in ("prestige_vanishing", "prestige_visibility"):
        stock_model = MyModel(
            **dict(
                CANARY_PARAMS,
                agent_generation_time=0.1,
                step_limit=10,
                entry_prestige_stock=stock,
            )
        )
        stock_report = stock_model.run_mode_for_bulk()
        assert stock_report["entry_prestige_stock"] == stock
    print("entry_prestige_stock OK: validated, both alternate stocks run")

    # 10. Path B dyadic layer: (a) off-mode bit-identity with every knob
    #     cranked, (b) the bit-level canary anchor, (c) both formulations
    #     bind across lambda, (d) the six new knobs validate. Note: the
    #     cranked run also sets credit_sharing="equal_split", whose accrual
    #     deferral is provably dynamics-neutral at this config (the probe
    #     run showed bit-identity); with links present the split is the
    #     intended Q8 mechanics.
    dyadic_anchor = 0.025399760810316363
    assert report["mean_prestige"] == dyadic_anchor, report["mean_prestige"]
    cranked = MyModel(
        **dict(
            CANARY_PARAMS,
            dyadic_mode="off",
            dyadic_lambda=1.0,
            dyadic_proposal_regime="universal",
            credit_sharing="equal_split",
            closeness_mode="profile_overlap",
            dyadic_visibility_stock="entry",
        )
    )
    cranked_report = cranked.run_mode_for_bulk()
    ECHO_KEYS = {
        "dyadic_mode",
        "dyadic_lambda",
        "dyadic_proposal_regime",
        "credit_sharing",
        "closeness_mode",
        "dyadic_visibility_stock",
    }
    differing = [
        k for k in report
        if k not in ECHO_KEYS and repr(report[k]) != repr(cranked_report[k])
    ]
    assert not differing, f"off-mode dynamics moved: {differing}"
    assert cranked.datacollector.get_model_vars_dataframe().equals(
        model.datacollector.get_model_vars_dataframe()
    ), "off-mode model_vars trajectory moved"
    print(
        f"Path B off-mode bit-identity OK (anchor mean_prestige="
        f"{report['mean_prestige']!r}); {len(differing)} dynamics keys differ"
    )

    BIND_PARAMS = dict(
        CANARY_PARAMS,
        n_agents=8,
        grid_size=10,
        step_limit=40,
        agent_generation_time=0.1,
        agent_seed=5,
        vanishing_factor_for_prestige=0.9,
        generation_params={"seed": 9, "std": 5, "number_gaussians": 2},
    )
    for dyadic_mode in ("logit", "threshold"):
        outs = {}
        for lam in (0.0, 1.0):
            bm = MyModel(
                **dict(BIND_PARAMS, dyadic_mode=dyadic_mode, dyadic_lambda=lam)
            )
            br = bm.run_mode_for_bulk()
            bdc = bm.datacollector.get_model_vars_dataframe()
            outs[lam] = (br["n_active_links"], float(bdc["n_active_links"].mean()))
        assert outs[0.0] != outs[1.0], (
            f"{dyadic_mode} does not bind across lambda: {outs}"
        )
        assert outs[1.0][1] > 0 or outs[0.0][1] > 0, (
            f"{dyadic_mode}: no links formed at either lambda"
        )
        print(
            f"Path B binding OK: {dyadic_mode} lambda0={outs[0.0]} lambda1={outs[1.0]}"
        )

    # 10e. Path B v2 movement coupling (dyadic_move_pull).
    #      (i) pull=0.0 reproduces Path B v1 bit-for-bit: the anchors below were
    #          measured in the v1 tree (HEAD b28557b, pre-coupling code) on this
    #          exact config, so equality here is an identity claim, not a
    #          tolerance. The round's driver repeats the check against the
    #          stored v1 cell CSVs on 60 numeric columns (control arm).
    #      (ii) pull>0 opens the choice -> movement -> production channel: the
    #          co-harvest (production) moments must respond to BOTH the pull at
    #          fixed lambda and to lambda at fixed pull.
    MOVE_PARAMS = dict(
        BIND_PARAMS,
        dyadic_mode="logit",
        credit_sharing="equal_split",
        dyadic_lambda=1.0,
    )
    V1_ANCHOR = {
        "mean_team_size": 4.333333333333333,
        "repeated_collab_share": 0.78125,
        "n_active_links": 2,
        "mean_prestige": 0.02584457750589488,
        "team_tm": 4.239837398373984,
        "rep_tm": 0.8680933654246947,
        "links_tm": 1.7073170731707317,
        # Post-BUG-0006 anchor (pre-BUG-0006 was 0.19920101324636486)
        "c_ideal_tm": 0.234169150292581,
    }
    m0 = MyModel(**dict(MOVE_PARAMS, dyadic_move_pull=0.0))
    r0 = m0.run_mode_for_bulk()
    dc0 = m0.datacollector.get_model_vars_dataframe()
    got0 = {
        "mean_team_size": r0["mean_team_size"],
        "repeated_collab_share": r0["repeated_collab_share"],
        "n_active_links": int(r0["n_active_links"]),
        "mean_prestige": r0["mean_prestige"],
        "team_tm": float(dc0["mean_team_size"].mean()),
        "rep_tm": float(dc0["repeated_collab_share"].mean()),
        "links_tm": float(dc0["n_active_links"].mean()),
        "c_ideal_tm": float(dc0["Corr_prestige_ideal_merit"].mean()),
    }
    for key, want in V1_ANCHOR.items():
        if key == "c_ideal_tm":
            assert np.isclose(got0[key], want, rtol=0.0, atol=1e-13), (
                f"dyadic_move_pull=0.0 c_ideal_tm drifted: "
                f"{got0[key]!r} != {want!r}"
            )
        else:
            assert got0[key] == want, (
                f"dyadic_move_pull=0.0 is not bit-identical to Path B v1: "
                f"{key}={got0[key]!r} != {want!r}"
            )
    print(
        "Path B v2 identity OK: dyadic_move_pull=0.0 reproduces Path B v1 on "
        f"{len(V1_ANCHOR)} anchors"
    )

    prod = {}
    for lam, pull in ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)):
        mm = MyModel(
            **dict(MOVE_PARAMS, dyadic_lambda=lam, dyadic_move_pull=pull)
        )
        mm.run_mode_for_bulk()
        dcm = mm.datacollector.get_model_vars_dataframe()
        prod[(lam, pull)] = (
            float(dcm["mean_team_size"].mean()),
            float(dcm["repeated_collab_share"].mean()),
            float(dcm["n_active_links"].mean()),
        )
    base, lam_only, both = prod[(1.0, 0.0)], prod[(0.0, 1.0)], prod[(1.0, 1.0)]
    assert base != both, f"the pull does not bind at fixed lambda: {base} vs {both}"
    assert lam_only != both, f"lambda does not bind at fixed pull: {lam_only} vs {both}"
    assert base[0] != both[0] or base[1] != both[1], (
        "the pull moves the choice layer only, not the production layer: "
        f"team/rep {base[:2]} vs {both[:2]}"
    )
    print(
        "Path B v2 coupling OK: team/rep/links "
        f"pull0_lam1={base} lam0_pull1={lam_only} lam1_pull1={both}"
    )

    for bad in (
        dict(dyadic_mode="bogus"),
        dict(dyadic_lambda=1.5),
        dict(dyadic_proposal_regime="bogus"),
        dict(credit_sharing="bogus"),
        dict(closeness_mode="bogus"),
        dict(dyadic_visibility_stock="bogus"),
        dict(dyadic_move_pull=-0.5),
        dict(dyadic_mode="off", dyadic_move_pull=1.0),
    ):
        try:
            MyModel(**dict(CANARY_PARAMS, **bad))
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid dyadic knob accepted: {bad}")
    print(
        "Path B knob validation OK: seven knobs reject invalid values, and a "
        "positive pull with the layer off is refused"
    )

    # 11. regression BUG-0006: unscored step-t newborns carry ideal_Merit=NaN
    #     and are excluded from correlation_reporter until updateIdealMerit
    #     scores them on the following step.
    class _Bug0006Agent:
        def __init__(self, prestige, ideal_merit):
            self.prestige = prestige
            self.ideal_Merit = ideal_merit

    class _Bug0006Model:
        def __init__(self, agents):
            self.agents = agents

    synth = _Bug0006Model([
        _Bug0006Agent(1.0, -0.3),
        _Bug0006Agent(2.0, -0.2),
        _Bug0006Agent(3.0, -0.1),
        _Bug0006Agent(0.0, np.nan),
    ])
    synth_r = metrics.correlation_reporter("ideal_Merit", "prestige")(synth)
    assert np.isclose(synth_r, 1.0, rtol=0.0, atol=1e-12), (
        f"BUG-0006 synthetic regression failed: expected 1.0, got {synth_r}"
    )

    # Live model birth-step test (step 20 is a birth step at g=0.1)
    live_m = MyModel(**dict(CANARY_PARAMS, agent_generation_time=0.1, step_limit=20))
    live_m.run_mode_for_bulk()
    live_dc = live_m.datacollector.get_model_vars_dataframe()
    unscored = [a for a in live_m.agents if not np.isfinite(a.ideal_Merit)]
    scored = [a for a in live_m.agents if np.isfinite(a.ideal_Merit)]
    assert len(unscored) == 1, f"expected 1 unscored step-20 newborn, got {len(unscored)}"
    assert unscored[0].prestige == 0.0, unscored[0].prestige
    assert len(scored) == len(live_m.agents) - 1
    expected_r = np.corrcoef(
        [a.prestige for a in scored], [a.ideal_Merit for a in scored]
    )[0, 1]
    got_r = float(live_dc["Corr_prestige_ideal_merit"].iloc[-1])
    assert np.isclose(got_r, expected_r, rtol=0.0, atol=1e-12), (
        f"step-20 corr mismatch: {got_r} != {expected_r}"
    )
    # Next step (step 21, non-birth): every agent is scored (0 NaNs)
    live_m.step(True)
    assert all(np.isfinite(a.ideal_Merit) for a in live_m.agents), (
        "non-birth step 21 still has unscored ideal_Merit"
    )
    # Zero-entry invariance: 0 NaNs at step 0 and step 30
    zero_init = MyModel(**CANARY_PARAMS)
    assert all(np.isfinite(a.ideal_Merit) for a in zero_init.agents), (
        "g=0 step 0 has NaN ideal_Merit"
    )
    assert all(np.isfinite(a.ideal_Merit) for a in model.agents), (
        "g=0 step 30 has NaN ideal_Merit"
    )
    print(
        f"BUG-0006 regression OK: synth_r={synth_r:.3f}, "
        f"step20_unscored=1 (corr={got_r:+.4f}), step21_unscored=0, g0_nans=0"
    )

    # 12. ADR-0003 backward compatibility: the default exit mode is "none",
    #     and an explicit none run at the legacy retention factor gamma=0.9
    #     must reproduce it bit-for-bit.  The clock advances 0.5 year per step
    #     and newcomers enter at age 25.
    legacy_params = dict(
        CANARY_PARAMS,
        agent_generation_time=0.1,
        step_limit=40,
        vanishing_factor_for_prestige=0.9,
    )
    default_none = MyModel(**legacy_params)
    explicit_none = MyModel(**dict(legacy_params, exit_mode="none"))
    r_default = default_none.run_mode_for_bulk()
    r_explicit = explicit_none.run_mode_for_bulk()
    assert bits(r_default) == bits(r_explicit), (
        "default exit_mode is not bit-identical to explicit 'none'"
    )
    assert r_default["exit_mode"] == "none"
    assert r_default["tau_max"] == CANONICAL_TAU_MAX == 80
    assert len(default_none._scientists) > CANARY_PARAMS["n_agents"], (
        "exit_mode='none' unexpectedly retired agents"
    )
    assert "gap_recent" in r_default and "gap_recent_10" in r_default

    clock = MyModel(
        **dict(legacy_params, agent_generation_time=0.0, step_limit=10)
    )
    ages_before = [a.age for a in clock._scientists]
    for _ in range(10):
        clock.step(True)
    ages_after = [a.age for a in clock._scientists]
    assert all(b - a == 5.0 for a, b in zip(ages_before, ages_after)), (
        "ADR-0003 clock did not add 5 years over 10 steps"
    )

    newborn_model = MyModel(
        **dict(legacy_params, agent_generation_time=0.25, step_limit=4)
    )
    newborn_model.run_mode_for_bulk()
    newborns = [
        a for a in newborn_model._scientists
        if getattr(a, "is_newcomer", False)
    ]
    assert newborns, "expected newcomers after four steps at g=0.25"
    youngest = min(newborns, key=lambda a: a.age)
    assert youngest.age == 25.0, youngest.age
    assert youngest.tenure_steps == 0.0, youngest.tenure_steps
    print(
        "ADR-0003 backward compatibility OK: default == explicit none, "
        f"clock +{ages_after[0] - ages_before[0]:.1f}y/10 steps, "
        f"newcomer age={youngest.age:.1f}"
    )

    # 13. ADR-0003 canonical turnover canary.  fixed_span with one newcomer
    #     every 4 steps (g=0.25) must hold the population at N=20 and, after
    #     one 80-step generation, have replaced the whole founder cohort.
    assert CANONICAL_AGENT_GENERATION_TIME == 0.25
    assert np.isclose(CANONICAL_VANISHING_FACTOR, 0.933, atol=5e-4)
    adr_params = dict(
        CANARY_PARAMS,
        n_agents=20,
        grid_size=20,
        initial_curiosity=0.5,
        generation_params={"seed": 4, "std": 5, "number_gaussians": 2},
        agent_generation_time=CANONICAL_AGENT_GENERATION_TIME,
        step_limit=80,
        vanishing_factor_for_prestige=CANONICAL_VANISHING_FACTOR,
        exit_mode="fixed_span",
        tau_max=CANONICAL_TAU_MAX,
    )
    adr_model = MyModel(**adr_params)
    assert np.isclose(
        adr_model.vanishing_factor_for_prestige, CANONICAL_VANISHING_FACTOR
    )
    adr_report = adr_model.run_mode_for_bulk()
    assert adr_report["exit_mode"] == "fixed_span"
    assert adr_report["tau_max"] == 80
    assert len(adr_model._scientists) == 20, len(adr_model._scientists)
    assert all(a.age >= 25.0 for a in adr_model._scientists), (
        "an agent retired below entry age"
    )
    assert all(a.is_newcomer for a in adr_model._scientists), (
        "the founder cohort was not fully replaced over tau_max steps"
    )
    assert np.isfinite(adr_report["gap_recent"]), adr_report["gap_recent"]
    assert adr_report["mean_tenure"] > 0.0

    # The two age-hazard variants must run with the same clock and retire at
    # the tenure cap; helper reports on the staggered run exercise both.
    for mode in ("age_hazard_sync", "age_hazard_staggered"):
        hazard = MyModel(**dict(adr_params, exit_mode=mode))
        hazard_report = hazard.run_mode_for_bulk()
        assert hazard_report["exit_mode"] == mode
        assert len(hazard._scientists) >= 20
        assert all(a.age >= 25.0 for a in hazard._scientists)
        assert "gap_recent" in hazard_report

    for bad in (
        dict(exit_mode="bogus"),
        dict(tau_max=0),
        dict(tau_max=-1),
        dict(tau_max=80.5),
    ):
        try:
            MyModel(**dict(CANARY_PARAMS, **bad))
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid ADR-0003 knob accepted: {bad}")
    print(
        "ADR-0003 turnover canary OK: fixed_span N=20, founders replaced, "
        f"gap_recent={adr_report['gap_recent']:+.4f}, "
        "age-hazard modes and validation OK"
    )

    # 14. attraction-trias knob (prestige_weight_in_visibility):
    #     (a) the default None reproduces the PRE-EDIT code path bit-for-bit -
    #         the two digests below were measured on this exact config in the
    #         pre-edit tree (branch point fb797fb), so equality here is an
    #         identity claim, not a tolerance;
    #     (b) w = 0.0 is bit-identical to use_visibility_reward=False;
    #     (c) w = 1.0 is bit-identical to use_visibility_reward=True;
    #     (d) w = 0.5 binds: the trajectory moves off BOTH extremes and stays
    #         finite (no NaN creep);
    #     plus validation of values outside [0, 1].
    import hashlib

    ATTR_PARAMS = dict(CANARY_PARAMS, n_agents=5, step_limit=25, agent_seed=3)
    ATTR_PRE_EDIT_DIGEST = {
        True: "c625cd78f15c0dbba40503196aa7e33685b2789b0f26d66ee97c0233d959ea27",
        False: "f8ebc78a55a46a68b35cd24c2b96362451f09ba562cb9e8e49d540d8275c953d",
    }
    ATTR_ECHO_KEYS = {"use_visibility_reward", "prestige_weight_in_visibility"}

    def _report_digest(rep):
        return hashlib.sha256(
            "|".join(f"{k}={v!r}" for k, v in sorted(rep.items())).encode()
        ).hexdigest()

    def _dynamics(rep):
        return sorted(
            (k, repr(v)) for k, v in rep.items() if k not in ATTR_ECHO_KEYS
        )

    for flag, want in ATTR_PRE_EDIT_DIGEST.items():
        m = MyModel(**dict(ATTR_PARAMS, use_visibility_reward=flag))
        r = m.run_mode_for_bulk()
        assert _report_digest(r) == want, (
            f"prestige_weight_in_visibility=None drifted from the pre-edit "
            f"code path (use_visibility_reward={flag}): "
            f"{_report_digest(r)} != {want}"
        )
    print("attraction-trias None bit-identity OK (2 pre-edit digests)")

    base_false = MyModel(**dict(ATTR_PARAMS, use_visibility_reward=False))
    base_true = MyModel(**dict(ATTR_PARAMS))
    w0 = MyModel(**dict(ATTR_PARAMS, prestige_weight_in_visibility=0.0))
    w1 = MyModel(**dict(ATTR_PARAMS, prestige_weight_in_visibility=1.0))
    w5 = MyModel(**dict(ATTR_PARAMS, prestige_weight_in_visibility=0.5))
    rbf = base_false.run_mode_for_bulk()
    rbt = base_true.run_mode_for_bulk()
    rw0 = w0.run_mode_for_bulk()
    rw1 = w1.run_mode_for_bulk()
    rw5 = w5.run_mode_for_bulk()

    assert _dynamics(rbf) == _dynamics(rw0), (
        "w=0.0 is not bit-identical to use_visibility_reward=False"
    )
    assert base_false.datacollector.get_model_vars_dataframe().equals(
        w0.datacollector.get_model_vars_dataframe()
    ), "w=0.0 model_vars trajectory moved vs use_visibility_reward=False"
    assert _dynamics(rbt) == _dynamics(rw1), (
        "w=1.0 is not bit-identical to use_visibility_reward=True"
    )
    assert base_true.datacollector.get_model_vars_dataframe().equals(
        w1.datacollector.get_model_vars_dataframe()
    ), "w=1.0 model_vars trajectory moved vs use_visibility_reward=True"
    print("attraction-trias extremes OK: w=0.0 == flag False, w=1.0 == flag True")

    assert _dynamics(rw5) != _dynamics(rw0), "w=0.5 inert vs w=0.0"
    assert _dynamics(rw5) != _dynamics(rw1), "w=0.5 inert vs w=1.0"
    assert not w5.datacollector.get_model_vars_dataframe()[
        ["Corr_prestige_ideal_merit", "percentage_knowledge_harvested"]
    ].isna().any().any(), "w=0.5 produced NaN trajectory values"
    assert np.isfinite(rw5["mean_prestige"]), rw5["mean_prestige"]
    print(
        "attraction-trias binding OK: w=0.5 moves trajectories off both "
        "extremes, finite throughout"
    )

    for bad in (
        dict(prestige_weight_in_visibility=-0.1),
        dict(prestige_weight_in_visibility=1.5),
        dict(prestige_weight_in_visibility=float("nan")),
    ):
        try:
            MyModel(**dict(CANARY_PARAMS, **bad))
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"invalid prestige_weight_in_visibility accepted: {bad}"
            )
    print("attraction-trias knob validation OK: out-of-range and NaN refused")

    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
