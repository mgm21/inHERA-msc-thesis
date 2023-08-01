from src.utils.all_imports import *
from src.adaptive_agent import AdaptiveAgent
from src.task import Task
from src.repertoire_optimiser import RepertoireOptimiser

task = Task(episode_length=150, num_iterations=500, grid_shape=tuple([4]*6))

agent = AdaptiveAgent(task=task,)

repertoire_opt = RepertoireOptimiser(task=task, env=agent.env)

# Location to store results
now = datetime.now()
dt_string = now.strftime("%m_%d-%H_%M_%S")
res_path = f"./results/{dt_string}"
os.mkdir(path=res_path)

repertoire_opt.optimise_repertoire(repertoire_path=f"{res_path}/repertoire/",
                                   plot_path=f"{res_path}/plots",
                                   html_path=f"{res_path}/best_policy.html",
                                   csv_results_path=f"{res_path}/class_mapelites_logs.csv")
