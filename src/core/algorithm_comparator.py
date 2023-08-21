from src.utils.all_imports import *
from src.core.ancestors_generator import AncestorsGenerator
from src.core.children_generator import ChildrenGenerator
from src.loaders.children_loader import ChildrenLoader
from src.utils.hexapod_damage_dicts import shortform_damage_list, intact
from src.utils.visualiser import Visualiser


class AlgorithmComparator:
    def __init__(self,algorithms_to_test,
            path_to_family,
            task,
            norm_params,
            algo_num_iter,
            ancest_num_legs_damaged=(1,),
            children_num_legs_damaged=(1,),
            verbose=False,
            ite_alpha=0.9,
            gpcf_kappa=0.05,
            algorithms_to_plot=["ITE", "GPCF"],
            children_in_ancestors=True):
        
        self.algorithms_to_test = algorithms_to_test
        self.path_to_family = path_to_family
        self.task = task
        self.norm_params = norm_params
        self.algo_num_iter = algo_num_iter
        self.ancest_num_legs_damaged = ancest_num_legs_damaged
        self.children_num_legs_damaged = children_num_legs_damaged
        self.verbose = verbose
        self.ite_alpha = ite_alpha
        self.gpcf_kappa = gpcf_kappa
        self.algorithms_to_plot = algorithms_to_plot
        self.children_in_ancestors = children_in_ancestors

    def run(self,):
        self.generate_ancestors()
        self.generate_children()
        means, vars = self.get_algo_means_and_var()
        self.savefig_algos_comparison_results(means, vars)
        
    def generate_ancestors(self,):
        ancest_gen = AncestorsGenerator(path_to_family=self.path_to_family,
                                    task=self.task,
                                    shortform_damage_list=shortform_damage_list,
                                    ite_alpha=self.ite_alpha,
                                    ite_num_iter=self.algo_num_iter,
                                    verbose=self.verbose,
                                    norm_params=self.norm_params)
        
        for i in range(len(self.ancest_num_legs_damaged)):
            ancest_gen.generate_auto_ancestors(num_broken_limbs=self.ancest_num_legs_damaged[i])

        # Include the simulation repertoire in the ancestors
        ancest_gen.generate_ancestor(damage_dict=intact, name="intact")
    

    def generate_children(self,):
        children_gen = ChildrenGenerator(algorithms_to_test=self.algorithms_to_test,
                                         path_to_family=self.path_to_family,
                                         task=self.task,
                                         shortform_damage_list=shortform_damage_list,
                                         ite_alpha=self.ite_alpha,
                                         ite_num_iter=self.algo_num_iter,
                                         verbose=self.verbose,
                                         norm_params=self.norm_params,
                                         gpcf_kappa=self.gpcf_kappa,
                                         children_in_ancestors=self.children_in_ancestors)
        
        for i in range(len(self.children_num_legs_damaged)):
            children_gen.generate_auto_children(num_broken_limbs=self.children_num_legs_damaged[i])
        
        # Include the simulation repertoire in the children
        children_gen.generate_child(damage_dict=intact, name="intact")
    
    def get_algo_means_and_var(self):
        children_loader = ChildrenLoader()
        res_dict = children_loader.load_children(path_to_family=self.path_to_family,
                                      algos=self.algorithms_to_plot)        
        
        means = []
        vars = []
        medians = []
        quantiles1 = []
        quantiles2 = []

        for algo in algorithms_to_plot:
            means += [jnp.nanmean(res_dict[algo], axis=0)]
            vars += [jnp.nanvar(res_dict[algo], axis=0)]
            medians += [jnp.nanmedian(res_dict[algo], axis=0)]
            quantiles1 += [jnp.nanquantile(res_dict[algo], q=0.25, axis=0)]
            quantiles2 += [jnp.nanquantile(res_dict[algo], q=0.75, axis=0)]
        
        return means, vars, medians, quantiles1, quantiles2
    
    def get_damage_specific_algo_means_and_vars(self,):
        children_loader = ChildrenLoader()
        res_dict = children_loader.load_children_by_num_broken_legs(path_to_family=self.path_to_family, algos=self.algorithms_to_plot)

        means = {}
        vars = {}
        medians = {}
        quantiles1 = {}
        quantiles2 = {}

        for algo in algorithms_to_plot:
            means[algo] = {}
            vars[algo] = {}
            medians[algo] = {}
            quantiles1[algo] = {}
            quantiles2[algo] = {}

            for num_damage in res_dict[algo]:
                means[algo][num_damage] = jnp.nanmean(res_dict[algo][num_damage], axis=0)
                vars[algo][num_damage] = jnp.nanvar(res_dict[algo][num_damage], axis=0)
                medians[algo][num_damage] = jnp.nanmedian(res_dict[algo][num_damage], axis=0)
                quantiles1[algo][num_damage] = jnp.nanquantile(res_dict[algo][num_damage], q=0.25, axis=0)
                quantiles2[algo][num_damage] = jnp.nanquantile(res_dict[algo][num_damage], q=0.75, axis=0)
        
        return means, vars, medians, quantiles1, quantiles2

    
    def savefig_algos_comparison_results(self, means, vars):
        visu = Visualiser()
        visu.get_mean_and_var_plot(means=means,
                                   vars=vars,
                                   names=self.algorithms_to_plot,
                                   path_to_res=self.path_to_family)
        
    def savefig_algos_comparison_results2(self, medians, quantiles1 ,quantiles2):
        visu = Visualiser()
        visu.get_medians_and_quants_plot(medians=medians,
                                   quantiles1=quantiles1,
                                   quantiles2=quantiles2,
                                   names=self.algorithms_to_plot,
                                   path_to_res=self.path_to_family)
    
    def savefig_per_damage_algos_comparison_results(self, means_dict, vars_dict,):
        visu = Visualiser()
        visu.get_mean_and_var_plot_per_damage_level(means_dict=means_dict,
                                                    vars_dict=vars_dict,
                                                    path_to_res=self.path_to_family,)
        
    def savefig_per_damage_algos_comparison_results2(self, medians_dict, quantiles1_dict, quantiles2_dict,):
        visu = Visualiser()
        visu.get_medians_and_quants_plot_per_damage_level(medians_dict=medians_dict,
                                                    quantiles1_dict=quantiles1_dict,
                                                    quantiles2_dict=quantiles2_dict,
                                                    path_to_res=self.path_to_family,)


if __name__ == "__main__":
    # Choose the family. Change both!
    from families.family_12 import family_task
    path_to_family = "families/family_12"

    task = family_task.task
    norm_params = jnp.load(f"{path_to_family}/norm_params.npy")
    algo_num_iter = family_task.algo_num_iter
    children_in_ancestors = family_task.children_in_ancestors

    ancest_num_legs_damaged = (1,) # Must be a tuple e.g. (2,)
    children_num_legs_damaged = (1, 2, 3, 4, 5) # Must be a tuple e.g. (1, 2, 3, 4,)
    algorithms_to_test = ["ITE", "GPCF", "GPCF-1trust", "GPCF-reg", "inHERA"] # "ITE", "GPCF", "GPCF-1trust", "GPCF-reg", "inHERA"
    algorithms_to_plot = ["ITE", "GPCF", "GPCF-1trust", "GPCF-reg", "inHERA"] # "ITE", "GPCF", "GPCF-1trust", "GPCF-reg", "inHERA"
    verbose = True

    ite_alpha = 0.9
    gpcf_kappa = 0.05

    # DEFINE AN ALGORITHM COMPARATOR
    algo_comp = AlgorithmComparator(algorithms_to_test=algorithms_to_test,
                  path_to_family=path_to_family,
                  task=task,
                  norm_params=norm_params,
                  algo_num_iter=algo_num_iter,
                  ancest_num_legs_damaged=ancest_num_legs_damaged,
                  children_num_legs_damaged=children_num_legs_damaged,
                  verbose=verbose,
                  ite_alpha=ite_alpha,
                  gpcf_kappa=gpcf_kappa,
                  algorithms_to_plot=algorithms_to_plot,
                  children_in_ancestors=children_in_ancestors)
    
    start_time = time.time()

    # # # TO RUN THE FULL FLOW
    # algo_comp.run()

    # # TO ONLY GENERATE THE ANCESTORS
    # algo_comp.generate_ancestors()

    # # # # TO ONLY MAKE THE CHILDREN ADAPT
    # algo_comp.generate_children()

    # # TO ONLY PRODUCE AND SAVE THE OVERALL PLOTS
    # means, vars, medians, quant1, quant2 = algo_comp.get_algo_means_and_var()
    # algo_comp.savefig_algos_comparison_results2(medians, quant1, quant2,)

    # means, vars, _, _, _ = algo_comp.get_algo_means_and_var()
    # algo_comp.savefig_algos_comparison_results(means, vars)

    # # TO PRODUCE THE PLOTS PER DAMAGE LEVEL
    # means_dict, vars_dict, _, _, _ = algo_comp.get_damage_specific_algo_means_and_vars()
    # algo_comp.savefig_per_damage_algos_comparison_results(means_dict, vars_dict,)

    # # TO PRODUCE THE PLOTS PER DAMAGE LEVEL
    # _, _, medians_dict, quantiles1_dict, quantiles2_dict = algo_comp.get_damage_specific_algo_means_and_vars()
    # algo_comp.savefig_per_damage_algos_comparison_results2(medians_dict, quantiles1_dict, quantiles2_dict)

    # TO RUN A HYPERPARAMETER SWEEP
    from src.core.adaptive_agent import AdaptiveAgent
    from src.core.gaussian_process import GaussianProcess

    children_gen = ChildrenGenerator(algorithms_to_test=algo_comp.algorithms_to_test,
                                         path_to_family=algo_comp.path_to_family,
                                         task=algo_comp.task,
                                         shortform_damage_list=shortform_damage_list,
                                         ite_alpha=algo_comp.ite_alpha,
                                         ite_num_iter=algo_comp.algo_num_iter,
                                         verbose=algo_comp.verbose,
                                         norm_params=algo_comp.norm_params,
                                         gpcf_kappa=algo_comp.gpcf_kappa,
                                         children_in_ancestors=algo_comp.children_in_ancestors)
    
    children_gen.generate_hyperparam_sweep_children(num_broken_limbs=1)
    children_gen.generate_hyperparam_sweep_children(num_broken_limbs=2)


    end_time = time.time()
    print(f"Execution time (s): {end_time - start_time}")