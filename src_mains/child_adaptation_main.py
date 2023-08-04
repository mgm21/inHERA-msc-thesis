from src.utils.all_imports import *
from src.adaptive_agent import AdaptiveAgent
from src.repertoire_loader import RepertoireLoader
from src.utils.hexapod_damage_dicts import *
from src.ancestors_generator import *
from src.adaptation_algorithms.ite import ITE
from src.adaptation_algorithms.gpcf import GPCF
from src.gaussian_process import GaussianProcess
from src.family import Family

path_to_family = f"results/family_3"
from results.family_3 import family_task
norm_params = jnp.load("results/family_3/norm_params.npy")
task = family_task.task

rep_loader = RepertoireLoader()
sim_repertoire_arrays = rep_loader.load_repertoire(repertoire_path=f"{path_to_family}/repertoire/")

ancest_gen = AncestorsGenerator(path_to_family=path_to_family, task=task)
damage_combination = (1, 2, 3, 4, 5)
damage_name = ancest_gen.get_name_from_combination(damage_combination)
damage_dict = ancest_gen.get_damage_dict_from_combination(damage_combination)

new_agent = AdaptiveAgent(task=task,
                          sim_repertoire_arrays=sim_repertoire_arrays,
                          damage_dictionary=damage_dict,
                          name=damage_name)

fam = Family(path_to_family=path_to_family)

# TODO: you do not have to pass a gp, you can simply create it in the ITE constructor
gpcf = GPCF(family=fam,
            agent=new_agent,
            gaussian_process=GaussianProcess(),
            path_to_results=f"{path_to_family}/new_agents/{new_agent.name}/GPCF/",
            save_res_arrs=True,
            verbose=False,
            norm_params=norm_params)

gpcf.run(num_iter=10)







