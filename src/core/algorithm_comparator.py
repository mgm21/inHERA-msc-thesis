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

        for algo in algorithms_to_plot:
            means += [jnp.nanmean(res_dict[algo], axis=0)]
            vars += [jnp.nanvar(res_dict[algo], axis=0)]
        
        return means, vars
    
    def get_damage_specific_algo_means_and_vars(self,):
        children_loader = ChildrenLoader()
        res_dict = children_loader.load_children_by_num_broken_legs(path_to_family=self.path_to_family, algos=self.algorithms_to_plot)

        means = {}
        vars = {}

        for algo in algorithms_to_plot:
            means[algo] = {}
            vars[algo] = {}

            for num_damage in res_dict[algo]:
                means[algo][num_damage] = jnp.mean(res_dict[algo][num_damage], axis=0)
                vars[algo][num_damage] = jnp.var(res_dict[algo][num_damage], axis=0)
        
        return means, vars

    
    def savefig_algos_comparison_results(self, means, vars):
        visu = Visualiser()
        visu.get_mean_and_var_plot(means=means,
                                   vars=vars,
                                   names=self.algorithms_to_plot,
                                   path_to_res=self.path_to_family)
    
    def savefig_per_damage_algos_comparison_results(self, means_dict, vars_dict,):
        visu = Visualiser()
        visu.get_mean_and_var_plot_per_damage_level(means_dict=means_dict,
                                                    vars_dict=vars_dict,
                                                    path_to_res=self.path_to_family,)


if __name__ == "__main__":
    # Choose hyper parameters for comparison experiment
    from results.family_8 import family_task
    path_to_family = "results/family_8"
    task = family_task.task
    norm_params = jnp.load(f"{path_to_family}/norm_params.npy")
    algo_num_iter = 20
    ancest_num_legs_damaged = (1, 2, 3, 4, 5)
    children_num_legs_damaged = (1, 2, 3, 4, 5)
    algorithms_to_test = ["ITE", "GPCF"]
    algorithms_to_plot = ["ITE", "GPCF"]
    verbose = True
    ite_alpha = 0.9
    gpcf_kappa = 0.05
    children_in_ancestors = False

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

    # # # TO RUN THE FULL FLOW
    # algo_comp.run()

    # # TO ONLY GENERATE THE ANCESTORS
    # algo_comp.generate_ancestors()

    # # # TO ONLY MAKE THE CHILDREN ADAPT
    algo_comp.generate_children()

    # TO ONLY PRODUCE AND SAVE THE OVERALL PLOTS
    means, vars = algo_comp.get_algo_means_and_var()
    algo_comp.savefig_algos_comparison_results(means, vars,)

    # TO PRODUCE THE PLOTS PER DAMAGE LEVEL
    means_dict, vars_dict = algo_comp.get_damage_specific_algo_means_and_vars()
    algo_comp.savefig_per_damage_algos_comparison_results(means_dict, vars_dict,)

    