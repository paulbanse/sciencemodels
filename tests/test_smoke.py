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
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from sciencemodels import MyModel, metrics
from sciencemodels.landscapes import Gaussian, multipleGaussians, uniform
from sciencemodels.model import (
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

    for bad in (
        dict(dyadic_mode="bogus"),
        dict(dyadic_lambda=1.5),
        dict(dyadic_proposal_regime="bogus"),
        dict(credit_sharing="bogus"),
        dict(closeness_mode="bogus"),
        dict(dyadic_visibility_stock="bogus"),
    ):
        try:
            MyModel(**dict(CANARY_PARAMS, **bad))
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid dyadic knob accepted: {bad}")
    print("Path B knob validation OK: all six knobs reject invalid values")

    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
