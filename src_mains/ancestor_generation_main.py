from src.utils.all_imports import *
from src.adaptive_agent import AdaptiveAgent
from src.task import Task
from src.repertoire_optimiser import RepertoireOptimiser
from src.adaptation_algorithms.ite import ITE
from src.repertoire_loader import RepertoireLoader
from src.utils import hexapod_damage_dicts
from src.gaussian_process import GaussianProcess

task = Task(episode_length=150,
            num_iterations=500,
            grid_shape=tuple([4]*6))

# CHANGE THIS
family_name = "family_0"
damage_dict = hexapod_damage_dicts.leg_5_broken
robot_name = "harmed5"

# Load the simulated repertoire 
repertoire_loader = RepertoireLoader()
simu_arrs = repertoire_loader.load_repertoire(repertoire_path=f"results/{family_name}/repertoire", remove_empty_bds=False)

# Define an Adaptive Agent which inherits from the task and gets its mu and var set to the simulated repertoire's mu and var
agent = AdaptiveAgent(task=task,
                      name=robot_name,
                      sim_repertoire_arrays=simu_arrs,
                      damage_dictionary=damage_dict)

# The below shouldn't have to be changed from agent to agent:

# Define a GP
gp = GaussianProcess()

# Create an ITE object with previous objects as inputs
ite = ITE(agent=agent,
          gaussian_process=gp,
          alpha=0.9,
          plot_repertoires=False,
          save_res_arrs=True,
          path_to_results=f"results/{family_name}/{agent.name}/",
          verbose=True,
          )

# # Run the ITE algorithm
ite.run(counter_thresh=20)

