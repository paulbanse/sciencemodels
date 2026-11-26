"""Thin entry point: one interactive run with animation (ADR-0002).

Parameter setup preserved verbatim from the pre-restructure ``runner.py``;
the model code lives in the ``sciencemodels`` package. Run from anywhere:

    python scripts/runner.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sciencemodels import landscapes
from sciencemodels.model import CANONICAL_VANISHING_FACTOR, MyModel


# example of generation params for landscapes.multipleGaussians :

chosen_setup = (
    3  # 0 for randominit, 1 for 2 gaussians, 2 for 4 gaussians, 3 for noisy gaussian
)
generation_params = {}
generation_function = landscapes.randominit

match chosen_setup:
    case 0:
        generation_function = landscapes.randominit
        generation_params = {"seed": 4}
    case 1:
        generation_function = landscapes.multipleGaussians
        generation_params = {"seed": 4, "std": 2, "number_gaussians": 2}
    case 2:
        generation_function = landscapes.multipleGaussians
        generation_params = {"seed": 4, "std": 5, "number_gaussians": 4}
    case 3:
        generation_function = landscapes.noisyGaussian
        generation_params = {
                "prop_random": 5,
                "std": 5,
                "std_curiosity": 0.1,
        }
    case _:
        print("chosen setup number non recognized")
        exit()

all_params = {'n_agents':3,             # minimum 3
              "initial_curiosity":0.5, # interest for exploration
              "use_distance": True,        # if True, agents take into account distance when choosing where to go
              "epsilon": 0.1,           # noise in the measure of utility
              "harvest":0.02,            # proportion of science harvested at each iteration 
              "grid_size":20,
              "cell_init_function":generation_function, # generation method for the landscape, see landscapes for the definitions
              "generation_params": generation_params,       # parameters for the generation method
              "agent_generation_time": 0,                       # number of agent generated per generation, for example if 0.1, 10 new agent per generation
              "new_questions": 0, 
              "agent_seed": 1,                           # seed for the random of the agents
              "step_limit": 200,
              "agent_generation_function": landscapes.uniform, # if this is not defined the function is just 
              "vanishing_factor_for_prestige": CANONICAL_VANISHING_FACTOR,  # ADR-0003: 0.5**(1/10) ~= 0.933
              "use_visibility_reward": True,
              "knowledge_normalization": "share",   # explicit pin so runs are self-describing
              "prestige_accrual": "harvested",      # explicit pin so runs are self-describing
              }

model = MyModel(**all_params)

model.animate_steps(
    dynamic_plot=True, csv_name="long_NGauss", end_report_file=""
)  # csv_name="data_randominit",end_report_file='end_report.csv')
