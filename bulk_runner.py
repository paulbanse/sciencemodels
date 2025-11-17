
import mesa_model
import landscapefunctions
import os

# generation_params = {'seed': 4}

# example of generation params for landscapefunctions.multipleGaussians : 

# generation_params = {'seed': 4, 
#                     "std": 5, 
#                     "number_gaussians": 6}

generation_params = {'prop_random': 5,
                     'seed': 4, 
                     "std": 5}

all_params = {'n_agents':20,             # minimum 3
              "n_connection": 2,
              "initial_curiosity":[k/10 for k in range(11)], # interest for exploration
              "use_distance": 0,        # if True, agents take into account distance when choosing where to go
              "epsilon": 0.1,           # noise in the measure of utility
              "harvest":0.02,            # proportion of science harvested at each iteration 
              "sizeGrid":20,
              "initCellFunc":landscapefunctions.noisyGaussian, # generation method for the landscape, see landscapefunctions for the definitions
              "generation_params": generation_params,       # parameters for the generation method
              "agent_generation_rate": 0,                       # number of agent generated per generation, for example if 0.1, 10 new agent per generation
              "constant_population": 0,                     # NOT YET IMPLEMENTED delete as many agent as generate by age for a constant population
              "agent_seed": 1,                           # seed for the random of the agents
              "step_limit": 400
              }

mesa_model.generate_data_parametric_exploration("end_report-noisy.csv", param_grid = all_params, repeats_per_setting = 40, change_landscape_seed = True, intention = "w", skip_to = 0)