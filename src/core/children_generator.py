from src.utils.all_imports import *
from src.utils.hexapod_damage_dicts import get_damage_dict, shortform_damage_list
from src.core.adaptive_agent import AdaptiveAgent
from src.core.gaussian_process import GaussianProcess
from src.adaptation_algorithms.ite import ITE
from src.adaptation_algorithms.gpcf import GPCF
from src.adaptation_algorithms.gpcf_1trust import GPCF1Trust
from src.loaders.repertoire_loader import RepertoireLoader
from src.core.family import Family

# TODO: this is a copy of AncestorsGenerator to have a quick way to test children without re-writing logic. Ultimately
#  make a super class

class ChildrenGenerator:
    def __init__(self, algorithms_to_test, path_to_family, task, shortform_damage_list=shortform_damage_list,
                  ite_alpha=0.9, ite_num_iter=20, verbose=False, norm_params=(0, 40),
                  gpcf_kappa=0.05, children_in_ancestors=True):
        
        # Difference with AncestorsGenerator
        self.algorithms_to_test = algorithms_to_test

        self.family = Family(path_to_family=path_to_family)

        self.gpcf_kappa = gpcf_kappa

        
        self.path_to_repertoire = f"{path_to_family}/repertoire"
        self.path_to_children = f"{path_to_family}/children"
        self.task = task
        self.shortform_damage_list = shortform_damage_list
        self.ite_alpha = ite_alpha
        self.verbose = verbose
        self.ite_num_iter = ite_num_iter
        self.norm_params = norm_params
        self.children_in_ancestors = children_in_ancestors

        rep_loader = RepertoireLoader()
        # TODO: change the fact that RepertoireLoader requires a path without the last "/" whereas the rest of the code takes paths without
        self.simu_arrs = rep_loader.load_repertoire(repertoire_path=f"{self.path_to_repertoire}/",)
        # print(self.simu_arrs[2])

    def generate_auto_children(self, num_broken_limbs=2, num_total_limbs=6,):
        # Get all the combinations of num_broken_limbs possible for robot with num_total_limbs limbs
        all_combinations = list(itertools.combinations(range(0, num_total_limbs), num_broken_limbs))
        if self.verbose: print(f"all_combinations: {all_combinations}")

        for combination in all_combinations:
            damage_dict = self.get_damage_dict_from_combination(combination)
            name = self.get_name_from_combination(combination)
            self.generate_child(damage_dict, name)

    def generate_custom_children(self, damage_dicts=[{}]):
        for count, damage_dict in enumerate(damage_dicts):
            self.generate_child(damage_dict, name=f"custom{count}")

    def generate_child(self, damage_dict, name):
        # Make sure to remove agent from family before collaboration algorithms (does not affect single agent algos)
        if not self.children_in_ancestors: 
            # This means that the child being generated should not use itself as an ancestor (i.e. an agent with the exact same damage)
            if self.verbose: print(self.family.ancestors_names)
            self.family.remove_ancestor_from_ancest_arrs(ancest_name=name)
            if self.verbose: print(self.family.ancestors_names)
        
        if "ITE" in self.algorithms_to_test:
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
                    path_to_results=f"{self.path_to_children}/{agent.name}/ITE/",
                    verbose=self.verbose,
                    norm_params=self.norm_params,
                    )

            # Run the ITE algorithm which saves the arrays TODO: could later decouple responsibilities here
            ite.run(num_iter=self.ite_num_iter)
        
            # Memory concern 
            del gp, agent
        
        if "GPCF" in self.algorithms_to_test:
            agent = AdaptiveAgent(task=self.task,
                        name=name,
                        sim_repertoire_arrays=self.simu_arrs,
                        damage_dictionary=damage_dict)


            gp = GaussianProcess(kappa=self.gpcf_kappa)

            # Create an ITE object with previous objects as inputs
            gpcf = GPCF(agent=agent,
                    gaussian_process=gp,
                    alpha=self.ite_alpha,
                    save_res_arrs=True,
                    path_to_results=f"{self.path_to_children}/{agent.name}/GPCF/",
                    verbose=self.verbose,
                    norm_params=self.norm_params,
                    family=self.family,
                    )

            gpcf.run(num_iter=self.ite_num_iter)
        
            # Memory concern
            del gp, agent
        
        if "GPCF-1trust" in self.algorithms_to_test:
            agent = AdaptiveAgent(task=self.task,
                        name=name,
                        sim_repertoire_arrays=self.simu_arrs,
                        damage_dictionary=damage_dict)
            
            gp = gp = GaussianProcess(kappa=self.gpcf_kappa)

            # Create an ITE object with previous objects as inputs
            gpcf_1trust = GPCF1Trust(agent=agent,
                    gaussian_process=gp,
                    alpha=self.ite_alpha,
                    save_res_arrs=True,
                    path_to_results=f"{self.path_to_children}/{agent.name}/GPCF-1trust/",
                    verbose=self.verbose,
                    norm_params=self.norm_params,
                    family=self.family,
                    )
            
            gpcf_1trust.run(num_iter=self.ite_num_iter)

            del gp, agent

        # Make sure to reset only after all collaborative algorithms have taken place
        if not self.children_in_ancestors:
            self.family.reset_the_ancestor_arrs()


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

    # TODO: Make the way task is loaded integrated into ChildGenerator constructor.
    # So far, must manually create the directory with a repertoire and a family_task.py file as well as an empty children folder.
    # Will automate this later. Perhaps json/pickle the task object and put it in the family so that everyone can refer to a unique task.
    
    # CHANGE THIS
    from results.family_3 import family_task
    norm_params = jnp.load("results/family_3/norm_params.npy")

    # CHANGE PATH!
    ancest_gen = ChildrenGenerator(algorithms_to_test = ["ITE", "GPCF"],
                                   path_to_family='results/family_3',
                                    task = family_task.task,
                                    shortform_damage_list=shortform_damage_list,
                                    ite_alpha=0.99,
                                    ite_num_iter=20,
                                    verbose=False,
                                    norm_params=norm_params)
    
    # For automatically generated children
    ancest_gen.generate_auto_children(num_broken_limbs=5)
