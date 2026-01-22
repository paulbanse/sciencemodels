import mesa_model
import landscapefunctions
import os
import warnings
from joblib import Parallel, delayed

warnings.simplefilter(action="ignore", category=FutureWarning)

chosen_setup = (
    3  # 0 for randominit, 1 for 2 gaussians, 2 for 6 gaussians, 3 for noisy gaussian
)
generation_params = {}


def generate_on_run(chosen_setup, foldername=""):
    print(f"Starting setup {chosen_setup}")
    match chosen_setup:
        case 0 | 1:
            generation_function = landscapefunctions.randominit
            generation_params = {}
        case 2 | 3:
            generation_function = landscapefunctions.multipleGaussians
            generation_params = {"std": 5, "number_gaussians": 2}
        case 4 | 5:
            generation_function = landscapefunctions.multipleGaussians
            generation_params = {"std": 5, "number_gaussians": 6}
        case 6 | 7:
            generation_function = landscapefunctions.noisyGaussian
            generation_params = {
                "prop_random": 5,
                "std": 5,
                "std_curiosity": 0.1,
            }
        case _:
            print("chosen setup number non recognized")
            exit()

    if chosen_setup % 2 == 0:
        curiosities = [k / 10 for k in range(11)]
        agent_generation_function = lambda cur, _rng, _params: cur
    else:
        curiosities = 0.5
        agent_generation_function = landscapefunctions.uniform

    all_params = {
        "n_agents": 20,  # minimum 3
        "n_connection": 2,
        "initial_curiosity": curiosities,  # interest for exploration
        "use_distance": True,  # if True, agents take into account distance when choosing where to go
        "epsilon": 0.1,  # noise in the measure of utility
        "harvest": 0.02,  # proportion of science harvested at each iteration
        "grid_size": 20,
        "cell_init_function": generation_function,  # generation method for the landscape, see landscapefunctions for the definitions
        "generation_params": generation_params,  # parameters for the generation method
        "agent_generation_rate": 0,  # number of agent generated per generation, for example if 0.1, 10 new agent per generation
        "new_questions": 0,
        "agent_seed": 1,  # seed for the random of the agents
        "step_limit": 400,
        "agent_generation_function": agent_generation_function,  # if this is not defined the function is just mu
        "vanishing_factor_for_prestige": 0.9,
        "use_visibility_reward": True,
    }

    # "_".join([f"{key}{value}" for key, value in generation_params.items()])
    filename = "data/" + foldername + "/run_number" + str(chosen_setup)
    print(filename, generation_params)

    mesa_model.generate_data_parametric_exploration(
        filename,
        param_grid=all_params,
        repeats_per_setting=40,
        change_landscape_seed=True,
        intention="w",
        skip_to=0,
        longitudinal=False,
    )


foldername = "first_large_run"
os.makedirs("data/" + foldername, exist_ok=True)
with (
    open("bulk_runner.py") as fp,
    open("data/" + foldername + "/creationscript.txt", "w") as tp,
):
    tp.writelines(fp.readlines())


Parallel(n_jobs=10, return_as="list")(
    [delayed(generate_on_run)(i, foldername) for i in range(8)]
)

