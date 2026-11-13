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
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sciencemodels import MyModel
from sciencemodels.landscapes import multipleGaussians, uniform

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

    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
