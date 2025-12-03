
import mesa_model
import landscapefunctions
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

chosen_setup = 3 # 0 for randominit, 1 for 2 gaussians, 2 for 6 gaussians, 3 for noisy gaussian
generation_params = {}
generation_function =landscapefunctions.randominit

match chosen_setup:
    case 0:
        generation_function =landscapefunctions.randominit
        generation_params = {'seed': 4}
    case 1:
        generation_function = landscapefunctions.multipleGaussians
        generation_params = {'seed': 4, 
                            "std": 5, 
                            "number_gaussians": 2}
    case 2:
        generation_function = landscapefunctions.multipleGaussians
        generation_params = {'seed': 4, 
                            "std": 5, 
                            "number_gaussians": 6}
    case 3:
        generation_function = landscapefunctions.noisyGaussian
        generation_params = {'prop_random': 5,
                            'seed': 4, 
                            "std": 5,
                            "std_curiosity": 0.1} 
    case _:
        print("chosen setup number non recognized")
        exit()


# generation_params = {'seed': 4, 
#                     "std": 5, 
#                     "number_gaussians": 2}

# example of generation params for landscapefunctions.noisyGaussian : 

# generation_params = {'prop_random': 5,
#                      'seed': 4, 
#                      "std": 5,
#                      "std_curiosity": 0.1} 

all_params = {'n_agents':20,             # minimum 3
              "n_connection": 2,
              "initial_curiosity":[k/10 for k in range(11)], # interest for exploration
              "use_distance": True,        # if True, agents take into account distance when choosing where to go
              "epsilon": 0.1,           # noise in the measure of utility
              "harvest":0.02,            # proportion of science harvested at each iteration 
              "sizeGrid":20,
              "initCellFunc":landscapefunctions.randominit, # generation method for the landscape, see landscapefunctions for the definitions
              "generation_params": generation_params,       # parameters for the generation method
              "agent_generation_rate": 0,                       # number of agent generated per generation, for example if 0.1, 10 new agent per generation
              "constant_population": 0,                     # NOT YET IMPLEMENTED delete as many agent as generate by age for a constant population
              "agent_seed": 1,                           # seed for the random of the agents
              "step_limit": 400,
              "AgentGenerationFunc": landscapefunctions.beta, # if this is not defined the function is just 
              "vanishing_factor_for_prestige": 0.99,
              "use_visibility_reward": True,
              }

filename = "testrun_visibilityReward_" + "_".join([f"{key}{value}" for key, value in generation_params.items()]) + ".csv"
print(filename, generation_params)
mesa_model.generate_data_parametric_exploration(filename, param_grid = all_params, repeats_per_setting = 40, change_landscape_seed = True, intention = "w", skip_to = 0)



# difference with jakob: changed the noise, used geometric noise, and use geometric weight on the reward