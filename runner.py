import mesa_model
import landscapefunctions
import os
import numpy as np


# example of generation params for landscapefunctions.multipleGaussians :

chosen_setup = (
    3  # 0 for randominit, 1 for 2 gaussians, 2 for 6 gaussians, 3 for noisy gaussian
)
generation_params = {}
generation_function = landscapefunctions.randominit

match chosen_setup:
    case 0:
        generation_function = landscapefunctions.randominit
        generation_params = {"seed": 4}
    case 1:
        generation_function = landscapefunctions.multipleGaussians
        generation_params = {"seed": 4, "std": 2, "number_gaussians": 1}
    case 2:
        generation_function = landscapefunctions.multipleGaussians
        generation_params = {"seed": 4, "std": 5, "number_gaussians": 4}
    case 3:
        generation_function = landscapefunctions.noisyGaussian
        generation_params = {
            "prop_random": 5,
            "seed": 4,
            "std": 5,
            "std_curiosity": 0.1,
        }
    case _:
        print("chosen setup number non recognized")
        exit()


all_params = {
    "n_agents": 20,  # minimum 3
    "n_connection": 2,
    "initial_curiosity": 0.5,  # interest for exploration
    "use_distance": True,  # if True, agents take into account distance when choosing where to go
    "epsilon": 0.1,  # noise in the measure of utility
    "harvest": 0.02,  # proportion of science harvested at each iteration
    "grid_size": 20,
    "cell_init_function": generation_function,  # generation method for the landscape, see landscapefunctions for the definitions
    "generation_params": generation_params,  # parameters for the generation method
    "agent_generation_rate": 0,  # number of agent generated per generation, for example if 0.1, 10 new agent per generation
    "new_questions": 1,
    "agent_seed": 1,  # seed for the random of the agents
    "step_limit": 400,
    "agent_generation_function": landscapefunctions.uniform,  # if this is not defined the function is just
    "vanishing_factor_for_prestige": 0.5,
    "use_visibility_reward": True,
}

model = mesa_model.MyModel(**all_params)


model.animate_steps(
    dynamic_plot=False, csv_name="long_NGauss", end_report_file=""
)  # csv_name="data_randominit",end_report_file='end_report.csv')
