from src.utils.all_imports import *
from src.utils.hexapod_damage_dicts import shortform_damage_list
from src.utils.hexapod_damage_dicts import get_damage_dict
from src.core.adaptation.adaptive_agent import AdaptiveAgent
from src.core.adaptation.gaussian_process import GaussianProcess
from src.adaptation_algorithms.ite import ITE
from src.adaptation_algorithms.gpcf_variants.gpcf import GPCF
from src.adaptation_algorithms.gpcf_variants.gpcf_1trust import GPCF1Trust
from src.adaptation_algorithms.gpcf_variants.gpcf_reg import GPCFReg
from src.adaptation_algorithms.inhera_variants.inhera import InHERA
from src.adaptation_algorithms.inhera_variants.inhera_b0 import InHERAB0
from src.adaptation_algorithms.inhera_variants.inhera_expert import InHERAExpert
from src.adaptation_algorithms.inhera_variants.inhera_b0_expert import InHERAB0Expert
from src.loaders.repertoire_loader import RepertoireLoader
from src.core.family_setup.family import Family

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

    def generate_auto_children(self, num_broken_limbs=2, num_total_limbs=6, **kwargs):
        # Get all the combinations of num_broken_limbs possible for robot with num_total_limbs limbs
        all_combinations = list(itertools.combinations(range(0, num_total_limbs), num_broken_limbs))
        if self.verbose: print(f"all_combinations: {all_combinations}")

        for combination in all_combinations:
            damage_dict = self.get_damage_dict_from_combination(combination)
            name = self.get_name_from_combination(combination)
            self.generate_child(damage_dict, name, **kwargs)

    def generate_custom_children(self, combinations, **kwargs):
        for combination in combinations:
                    damage_dict = self.get_damage_dict_from_combination(combination)
                    name = self.get_name_from_combination(combination)
                    self.generate_child(damage_dict, name, **kwargs)
    
    def generate_hyperparam_sweep_children(self, num_broken_limbs=1, num_total_limbs=6,):
        # Get all the combinations of num_broken_limbs possible for robot with num_total_limbs limbs
        all_combinations = list(itertools.combinations(range(0, num_total_limbs), num_broken_limbs))
        if self.verbose: print(f"all_combinations: {all_combinations}")

        regularisation_weights = jnp.array([1.000, 0.100, 0.010, 0.001])

        for combination in all_combinations:
            for l1_regularisation_weight in regularisation_weights:
                for invmax_regularisation_weight in regularisation_weights:
                    if self.verbose: print(f"l1_regularisation_weight: {l1_regularisation_weight}")
                    if self.verbose: print(f"invmax_regularisation_weight: {invmax_regularisation_weight}")

                    damage_dict = self.get_damage_dict_from_combination(combination)
                    name = self.get_name_from_combination(combination)

                    agent = AdaptiveAgent(task=self.task,
                                name=name,
                                sim_repertoire_arrays=self.simu_arrs,
                                damage_dictionary=damage_dict)
                    
                    gp = gp = GaussianProcess(kappa=self.gpcf_kappa, l1_regularisation_weight=l1_regularisation_weight, invmax_regularisation_weight=invmax_regularisation_weight,)

                    gpcf_reg = GPCFReg(agent=agent,
                            gaussian_process=gp,
                            alpha=self.ite_alpha,
                            save_res_arrs=True,
                            path_to_results=f"{self.path_to_children}/{agent.name}-l1-{round(l1_regularisation_weight, 3)}-invmax{round(invmax_regularisation_weight, 3)}/GPCF-reg/",
                            verbose=self.verbose,
                            norm_params=self.norm_params,
                            family=self.family,
                            )
                    
                    gpcf_reg.run(num_iter=self.ite_num_iter)

                    del gp, agent
                    del gpcf_reg
                    gc.collect()
            
    def generate_child(self, damage_dict, name, **kwargs):
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
            gp = GaussianProcess(**kwargs)

            # Create an ITE object with previous objects as inputs
            ite = ITE(agent=agent,
                    gaussian_process=gp,
                    alpha=self.ite_alpha,
                    plot_repertoires=True,
                    save_res_arrs=True,
                    path_to_results=f"{self.path_to_children}/{agent.name}/ITE/",
                    verbose=self.verbose,
                    norm_params=self.norm_params,
                    )

            # Run the ITE algorithm which saves the arrays TODO: could later decouple responsibilities here
            ite.run(num_iter=self.ite_num_iter)
        
            # Memory concern 
            del agent, gp
            del ite
            gc.collect()
        
        if "GPCF" in self.algorithms_to_test:
            agent = AdaptiveAgent(task=self.task,
                        name=name,
                        sim_repertoire_arrays=self.simu_arrs,
                        damage_dictionary=damage_dict)


            gp = GaussianProcess(**kwargs)

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
            del gpcf
            gc.collect()
        
        if "GPCF-1trust" in self.algorithms_to_test:
            agent = AdaptiveAgent(task=self.task,
                        name=name,
                        sim_repertoire_arrays=self.simu_arrs,
                        damage_dictionary=damage_dict)
            
            gp = GaussianProcess(**kwargs)

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
            del gpcf_1trust
            gc.collect()
        
        if "GPCF-reg" in self.algorithms_to_test:
            agent = AdaptiveAgent(task=self.task,
                        name=name,
                        sim_repertoire_arrays=self.simu_arrs,
                        damage_dictionary=damage_dict)
            
            gp = GaussianProcess(**kwargs)

            # Create an ITE object with previous objects as inputs
            gpcf_reg = GPCFReg(agent=agent,
                    gaussian_process=gp,
                    alpha=self.ite_alpha,
                    save_res_arrs=True,
                    path_to_results=f"{self.path_to_children}/{agent.name}/GPCF-reg/",
                    verbose=self.verbose,
                    norm_params=self.norm_params,
                    family=self.family,
                    )
            
            gpcf_reg.run(num_iter=self.ite_num_iter)

            del gp, agent
            del gpcf_reg
            gc.collect()

        if "inHERA" in self.algorithms_to_test:
            agent = AdaptiveAgent(task=self.task,
                        name=name,
                        sim_repertoire_arrays=self.simu_arrs,
                        damage_dictionary=damage_dict)
            
            gp = GaussianProcess(**kwargs)

            # Create an ITE object with previous objects as inputs
            inhera = InHERA(agent=agent,
                    gaussian_process=gp,
                    alpha=self.ite_alpha,
                    save_res_arrs=True,
                    path_to_results=f"{self.path_to_children}/{agent.name}/inHERA/",
                    verbose=self.verbose,
                    norm_params=self.norm_params,
                    family=self.family,
                    )
            
            inhera.run(num_iter=self.ite_num_iter)

            del gp, agent
            del inhera
            gc.collect()

        if "inHERA-b0" in self.algorithms_to_test:
            agent = AdaptiveAgent(task=self.task,
                        name=name,
                        sim_repertoire_arrays=self.simu_arrs,
                        damage_dictionary=damage_dict)
            
            gp = GaussianProcess(**kwargs)

            # Create an ITE object with previous objects as inputs
            inhera_b0 = InHERAB0(agent=agent,
                    gaussian_process=gp,
                    alpha=self.ite_alpha,
                    save_res_arrs=True,
                    path_to_results=f"{self.path_to_children}/{agent.name}/inHERA-b0/",
                    verbose=self.verbose,
                    norm_params=self.norm_params,
                    family=self.family,
                    )
            
            inhera_b0.run(num_iter=self.ite_num_iter)

            del gp, agent
            del inhera_b0
            gc.collect()
        
        if "inHERA-expert" in self.algorithms_to_test:
            agent = AdaptiveAgent(task=self.task,
                        name=name,
                        sim_repertoire_arrays=self.simu_arrs,
                        damage_dictionary=damage_dict)
            
            gp = GaussianProcess(**kwargs)

            # Create an ITE object with previous objects as inputs
            inhera_expert = InHERAExpert(agent=agent,
                    gaussian_process=gp,
                    alpha=self.ite_alpha,
                    save_res_arrs=True,
                    path_to_results=f"{self.path_to_children}/{agent.name}/inHERA-expert/",
                    verbose=self.verbose,
                    norm_params=self.norm_params,
                    family=self.family,
                    )
            
            inhera_expert.run(num_iter=self.ite_num_iter)

            del gp, agent
            del inhera_expert
            gc.collect()

        if "inHERA-b0-expert" in self.algorithms_to_test:
            agent = AdaptiveAgent(task=self.task,
                        name=name,
                        sim_repertoire_arrays=self.simu_arrs,
                        damage_dictionary=damage_dict)
            
            gp = GaussianProcess(**kwargs)

            # Create an ITE object with previous objects as inputs
            inhera_b0_expert = InHERAB0Expert(agent=agent,
                    gaussian_process=gp,
                    alpha=self.ite_alpha,
                    save_res_arrs=True,
                    path_to_results=f"{self.path_to_children}/{agent.name}/inHERA-b0-expert/",
                    verbose=self.verbose,
                    norm_params=self.norm_params,
                    family=self.family,
                    )
            
            inhera_b0_expert.run(num_iter=self.ite_num_iter)

            del gp, agent
            del inhera_b0_expert
            gc.collect()                        

        # Make sure to reset only after all collaborative algorithms have taken place
        if not self.children_in_ancestors:
            self.family.reset_the_ancestor_arrs()

    def get_damage_dict_from_combination(self, combination):
        merged_dict = {}
        for num in combination:
            merged_dict.update(self.shortform_damage_list[num])
        
        damage_dict = get_damage_dict(merged_dict)

        print(damage_dict)

        return damage_dict
    
    def get_name_from_combination(self, combination):
        if len(combination) == 0:
            return "intact"
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
    from dummy_families import family_task
    norm_params = jnp.load("dummy_families/family-seed_0_last_repertoire/norm_params.npy")

    # CHANGE PATH!
    children_gen = ChildrenGenerator(algorithms_to_test = ["ITE", "GPCF"],
                                   path_to_family='dummy_families/family-seed_0_last_repertoire',
                                    task = family_task.task,
                                    shortform_damage_list=shortform_damage_list,
                                    ite_alpha=0.99,
                                    ite_num_iter=20,
                                    verbose=False,
                                    norm_params=norm_params)
    
    print(children_gen.get_name_from_combination((1, 2)))
    print(children_gen.get_damage_dict_from_combination(()))
    
    # # For automatically generated children
    # children_gen.generate_auto_children(num_broken_limbs=5)


