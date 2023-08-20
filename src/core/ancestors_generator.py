from src.utils.all_imports import *
from src.utils.hexapod_damage_dicts import get_damage_dict, shortform_damage_list
from src.core.adaptive_agent import AdaptiveAgent
from src.core.gaussian_process import GaussianProcess
from src.adaptation_algorithms.ite import ITE
from src.loaders.repertoire_loader import RepertoireLoader

class AncestorsGenerator:
    def __init__(self, path_to_family, task, shortform_damage_list=shortform_damage_list, ite_alpha=0.9, ite_num_iter=20, verbose=False, norm_params=(0, 40)):
        self.path_to_repertoire = f"{path_to_family}/repertoire"
        self.path_to_ancestors = f"{path_to_family}/ancestors"
        self.task = task
        self.shortform_damage_list = shortform_damage_list
        self.ite_alpha = ite_alpha
        self.verbose = verbose
        self.ite_num_iter = ite_num_iter
        self.norm_params = norm_params

        rep_loader = RepertoireLoader()
        # TODO: change the fact that RepertoireLoader requires a path without the last "/" whereas the rest of the code takes paths without
        self.simu_arrs = rep_loader.load_repertoire(repertoire_path=f"{self.path_to_repertoire}/",)
        # print(self.simu_arrs[2])


    def generate_auto_ancestors(self, num_broken_limbs=2, num_total_limbs=6,):
        # Get all the combinations of num_broken_limbs possible for robot with num_total_limbs limbs
        all_combinations = list(itertools.combinations(range(0, num_total_limbs), num_broken_limbs))
        if self.verbose: print(f"all_combinations: {all_combinations}")

        for combination in all_combinations:
            damage_dict = self.get_damage_dict_from_combination(combination)
            name = self.get_name_from_combination(combination)
            self.generate_ancestor(damage_dict, name)

    def generate_custom_ancestors(self, damage_dicts=[{}]):
        for count, damage_dict in enumerate(damage_dicts):
            self.generate_ancestor(damage_dict, name=f"custom{count}")

    def generate_ancestor(self, damage_dict, name):
        agent = AdaptiveAgent(task=self.task,
                      name=name,
                      sim_repertoire_arrays=self.simu_arrs,
                      damage_dictionary=damage_dict)
        
        # Define a GP
        gp = GaussianProcess()

        # Create an ITE object with previous objects as inputs
        ite = ITE(agent=agent,
                gaussian_process=gp,
                alpha=self.ite_alpha,
                plot_repertoires=False,
                save_res_arrs=True,
                path_to_results=f"{self.path_to_ancestors}/{agent.name}/",
                verbose=self.verbose,
                norm_params=self.norm_params
                )

        # Run the ITE algorithm which saves the arrays TODO: could later decouple responsibilities here
        ite.run(num_iter=self.ite_num_iter)

        del agent, gp, ite
        gc.collect()
        
    
    def get_damage_dict_from_combination(self, combination):
        merged_dict = {}
        for num in combination:
            merged_dict.update(self.shortform_damage_list[num])
        
        damage_dict = get_damage_dict(merged_dict)

        return damage_dict
    
    def get_name_from_combination(self, combination):
        # The name will be, for example, damaged_0_2, if legs 0 and 2 were damaged
        name = 'damaged' + (len(combination)*'_{}').format(*combination)
        return name

if __name__ == "__main__":
    from src.utils.hexapod_damage_dicts import *
    from src.utils.hexapod_damage_dicts import shortform_damage_list

    # TODO: Make the way task is loaded integrated into AncestorGenerator constructor.
    # So far, must manually create the directory with a repertoire and a family_task.py file as well as an empty ancestors folder.
    # Will automate this later. Perhaps json/pickle the task object and put it in the family so that everyone can refer to a unique task.
    
    # CHANGE THIS
    from families.family_3 import family_task

    norm_params = jnp.load("families/family_3/norm_params.npy")

    # CHANGE PATH!
    ancest_gen = AncestorsGenerator(path_to_family='families/family_3',
                                    task = family_task.task,
                                    shortform_damage_list=shortform_damage_list,
                                    ite_alpha=0.99,
                                    ite_num_iter=20,
                                    verbose=False,
                                    norm_params=norm_params)
    
    # For automatically generated ancestors
    ancest_gen.generate_auto_ancestors(num_broken_limbs=5)
