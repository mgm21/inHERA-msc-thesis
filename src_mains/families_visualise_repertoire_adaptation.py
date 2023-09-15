from src.utils.all_imports import *
from src.loaders.repertoire_loader import RepertoireLoader
from src.core.adaptation.adaptive_agent import AdaptiveAgent
from src.core.adaptation.gaussian_process import GaussianProcess
from src.utils import hexapod_damage_dicts

# Import algorithms
from src.adaptation_algorithms.ite import ITE
from src.adaptation_algorithms.gpcf_variants.gpcf_1trust import GPCF1trust

# from families.family_3 import family_task
# path_to_family = "families/family_3"
# task = family_task.task
# repertoire_loader = RepertoireLoader()
# simu_arrs = repertoire_loader.load_repertoire(repertoire_path=f"{path_to_family}/repertoire", remove_empty_bds=False)
# damage_dict = hexapod_damage_dicts.leg_1_broken
# agent = AdaptiveAgent(task=task, sim_repertoire_arrays=simu_arrs, damage_dictionary=damage_dict)
# gp = GaussianProcess()
# norm_params = jnp.load(f"{path_to_family}/norm_params.npy")

# ite = ITE(agent=agent,
#             gaussian_process=gp,
#             verbose=True,
#             norm_params=norm_params)

# ite.run(num_iter=5)