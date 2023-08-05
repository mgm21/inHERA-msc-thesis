from src.utils.all_imports import *
from src.core.ancestors_generator import AncestorsGenerator
from src.core.children_generator import ChildrenGenerator
from src.loaders.children_loader import ChildrenLoader
from src.utils.hexapod_damage_dicts import shortform_damage_list
from src.utils.visualiser import Visualiser

class AlgorithmComparator:
    def run(self,
            algorithms_to_test,
            path_to_family,
            task,
            norm_params,
            algo_num_iter,
            ancest_num_legs_damaged=(1),
            children_num_legs_damaged=(1),
            verbose=False,
            ite_alpha=0.9):
        
        self.generate_ancestors(path_to_family, task, ite_alpha, algo_num_iter, verbose, norm_params, ancest_num_legs_damaged)
        self.generate_children(algorithms_to_test, path_to_family, task, ite_alpha, algo_num_iter, verbose, norm_params, children_num_legs_damaged)
        means, vars = self.get_algo_means_and_var(path_to_family=path_to_family, algorithms_to_test=algorithms_to_test)
        self.savefig_algos_comparison_results(means, vars)
        
    def generate_ancestors(self,
                           path_to_family,
                           task,
                           ite_alpha,
                           algo_num_iter,
                           verbose,
                           norm_params,
                           ancest_num_legs_damaged):
        
        ancest_gen = AncestorsGenerator(path_to_family=path_to_family,
                                    task = task,
                                    shortform_damage_list=shortform_damage_list,
                                    ite_alpha=ite_alpha,
                                    ite_num_iter=algo_num_iter,
                                    verbose=verbose,
                                    norm_params=norm_params)
        
        for i in range(len(ancest_num_legs_damaged)):
            ancest_gen.generate_auto_ancestors(num_broken_limbs=ancest_num_legs_damaged[i])
    

    def generate_children(self,
                          algorithms_to_test,
                          path_to_family,
                          task,
                          ite_alpha,
                          algo_num_iter,
                          verbose,
                          norm_params,
                          children_num_legs_damaged):
        
        children_gen = ChildrenGenerator(algorithms_to_test,
                                         path_to_family=path_to_family,
                                         task = task,
                                         shortform_damage_list=shortform_damage_list,
                                         ite_alpha=ite_alpha,
                                         ite_num_iter=algo_num_iter,
                                         verbose=verbose,
                                         norm_params=norm_params)
        
        for i in range(len(children_num_legs_damaged)):
            children_gen.generate_auto_children(num_broken_limbs=children_num_legs_damaged[i])
    
    def get_algo_means_and_var(self, path_to_family, algorithms_to_test):
        children_loader = ChildrenLoader()
        res_dict = children_loader.load_children(path_to_family=path_to_family,
                                      algos_to_extract=algorithms_to_test)
        
        means = []
        vars = []

        for algo in algorithms_to_test:
            means += [jnp.nanmean(res_dict[algo], axis=0)]
            vars += [jnp.nanvar(res_dict[algo], axis=0)]
    
        
        return means, vars
    
    def savefig_algos_comparison_results(self, means, vars, algorithms_to_test, path_to_family):
        visu = Visualiser()
        visu.get_mean_and_var_plot(means=means,
                                   vars=vars,
                                   names=algorithms_to_test,
                                   path_to_res=path_to_family,
                                   rolling_max=True)


if __name__ == "__main__":
    # Choose hyper parameters for comparison experiment
    from results.family_4 import family_task
    path_to_family = "results/family_4"
    task = family_task.task
    norm_params = jnp.load(f"{path_to_family}/norm_params.npy")
    algo_num_iter = 20
    ancest_num_legs_damaged=(1, 2)
    children_num_legs_damaged=(1,)
    algorithms_to_test = ["ITE", "GPCF"]
    verbose=True
    ite_alpha=0.9

    # Run the comparison routine
    algo_comp = AlgorithmComparator()

    # # TO RUN THE FULL FLOW 
    # algo_comp.run(path_to_family=path_to_family,
    #               task=task,
    #               norm_params=norm_params,
    #               algo_num_iter=algo_num_iter,
    #               ancest_num_legs_damaged=ancest_num_legs_damaged,
    #               verbose=verbose,
    #               ite_alpha=ite_alpha)

    # TO GENERATE THE ANCESTORS ...

    # # TO MAKE THE CHILDREN ADAPT
    # algo_comp.generate_children(algorithms_to_test=algorithms_to_test,
    #                             path_to_family=path_to_family,
    #                             task=task,
    #                             ite_alpha=ite_alpha,
    #                             algo_num_iter=algo_num_iter,
    #                             verbose=verbose,
    #                             norm_params=norm_params,
    #                             children_num_legs_damaged=children_num_legs_damaged)


    # TO VIEW THE PLOTS AFTER THE CHILDREN HAVE GONE THROUGH ADAPTATION
    means, vars = algo_comp.get_algo_means_and_var(path_to_family=path_to_family, algorithms_to_test=algorithms_to_test)
    algo_comp.savefig_algos_comparison_results(means, vars, algorithms_to_test=algorithms_to_test, path_to_family=path_to_family)


    

    