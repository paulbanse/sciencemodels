
import mesa_model
import landscapefunctions
import os
import numpy as np

generation_params = {'seed': 4, 
                     "std": 5, 
                     "number_gaussians": 2,
                     "std_curiosity": 0.2}
"""example of generation params for landscapefunctions.multipleGaussians : 

generation_params = {'seed': 4, 
                     "std": 5, 
                     "number_gaussians": 2}
"""




all_params = {'n_agents':20,             # minimum 3
              "n_connection": 2,
              "initial_curiosity":0.5, # interest for exploration
              "use_distance": 0,        # if True, agents take into account distance when choosing where to go
              "epsilon": 0.1,           # noise in the measure of utility
              "harvest":0.02,            # proportion of science harvested at each iteration 
              "sizeGrid":20,
              "initCellFunc":landscapefunctions.randominit, #landscapefunctions.randominit, # generation method for the landscape, see landscapefunctions for the definitions
              "generation_params": generation_params,       # parameters for the generation method
              "agent_generation_rate": 0,                       # number of agent generated per generation, for example if 0.1, 10 new agent per generation
              "constant_population": 0,                     # NOT YET IMPLEMENTED delete as many agent as generate by age for a constant population
              "agent_seed": 1,                           # seed for the random of the agents
              "step_limit": 200,
              "AgentGenerationFunc": landscapefunctions.beta,
              "visibility_factor_for_prestige": 0.5,
              "vanishing_factor_for_prestige": 0.9999
              }
model = mesa_model.MyModel(**all_params)


model.animate_steps(dynamic_plot=True, csv_name="data",end_report_file='')#,end_report_file="end_report.csv")

#mesa_model.plot_normalized_values(csv_name="data", size = all_params["sizeGrid"])

#mesa_model.plot_tileknowledge_per_prestige()

#os.remove("end_report.csv")
#for curiosity in [k/10 for k in range(0,11)]:
#    print("now processing curiosity =", curiosity)
#    all_params["initial_curiosity"] = curiosity
#    all_params["agent_seed"] = 10
#    for repeat in range(40):
#        all_params["agent_seed"] += 1
#        model = mesa_model.MyModel(**all_params)
#        model.animate_steps(dynamic_plot=False, steplimit=400, csv_name="data",end_report_file="end_report.csv")
#        del model
