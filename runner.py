
import mesa_model
import landscapefunctions

generation_params = {'seed': 1, 
                     "std": 15, 
                     "number_gaussians": 5}
        # landscapefunctions.multipleGaussians
all_params = {'n_agents':3,             # minimum 3
              "n_connection": 3,
              "initial_curiosity":0.4 , # interest for exploration
              "epsilon": 0.1,           # noise in the measure of utility
              "harvest":0.1,            # proportion of science harvested at each iteration 
              "sizeGrid":10,
              "initCellFunc":landscapefunctions.randominit, # generation method for the landscape, see landscapefunctions for the definitions
              "generation_params": generation_params,       # parameters for the generation method
              "agent_generation_rate": 0,                       # number of agent generated per generation, for example if 0.1, 10 new agent per generation
              "constant_population": 0                      # NOT YET IMPLEMENTED delete as many agent as generate by age for a constant population
              }
model = mesa_model.MyModel(**all_params)


model.animate_steps()
#model.Vizualize_3d()
