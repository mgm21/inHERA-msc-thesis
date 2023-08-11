from src.utils.all_imports import *
from src.adaptation_algorithms.adaptation_algorithm import AdaptationAlgorithm

class ExperienceSharingAlgorithm(AdaptationAlgorithm):
    def __init__(self, family, agent, gaussian_process, alpha=0.9, verbose=False,
                 path_to_results="results/ite_example/", save_res_arrs=True, norm_params=(0, 40), plot_repertoires=False):
        
        super().__init__(agent, gaussian_process, alpha, verbose,
                 path_to_results, save_res_arrs, norm_params, plot_repertoires)
        
        self.family = family

        if self.verbose: print(f"These are the ExpSharing ancestors: {self.family.ancestors_names}")
        if self.verbose: print(f"This is the current child: {self.agent.name}")
    
    def run_setup(self, num_iter):
        super().run_setup(num_iter)
        num_ancestors = len(self.family.ancestor_mus)
        self.ancestor_mus_at_obs = jnp.full(shape=(num_ancestors, num_iter), fill_value=jnp.nan)
    
    def update_mean_function(self, counter):
        self.ancestor_mus_at_curr_obs = self.family.ancestor_mus[:, self.tested_indices[counter:counter+1]]
        self.ancestor_mus_at_obs = self.ancestor_mus_at_obs.at[:, counter].set(jnp.squeeze(self.ancestor_mus_at_curr_obs))
        W = self.get_ancestor_weights(counter,)
        self.mean_func = self.family.ancestor_mus.T @ W
        self.mean_func_at_obs = self.ancestor_mus_at_obs[:, :counter+1].T @ W

        if self.verbose: print(f"mean_func_at_obs: {self.mean_func_at_obs}")
        if self.verbose: print(f"mean_func: {self.mean_func}")
        if self.verbose: print(f"x observations passed to gp: {self.agent.x_observed[:counter+1]}")
        if self.verbose: print(f"y_obs - mean_func_at_obs passed to gp: {self.agent.y_observed[:counter+1] - self.mean_func_at_obs[:counter+1]}")
        if self.verbose: print(f"y priors passed to gp: {self.mean_func}")
        if self.verbose: print(f"mu @ x_observed: {self.agent.mu[self.tested_indices]}")
        if self.verbose: print(f"y_observed values: {self.agent.y_observed}")
        
    def get_ancestor_weights(self, counter):
        raise NotImplementedError


if __name__ == "__main__":
    from src.core.family import Family
    from src.core.adaptive_agent import AdaptiveAgent
    from src.core.task import Task
    from src.utils import hexapod_damage_dicts
    from src.core.gaussian_process import GaussianProcess
    from src.loaders.repertoire_loader import RepertoireLoader

    path_to_family = "results/family_4_1"
    from results.family_4_1 import family_task
    fam = Family(path_to_family=path_to_family)
    task = family_task.task
    norm_params = jnp.load(f"{path_to_family}/norm_params.npy")
    damage_dict = hexapod_damage_dicts.leg_0_broken
    rep_loader = RepertoireLoader()
    simu_arrs = rep_loader.load_repertoire(repertoire_path=f"{path_to_family}/repertoire",)
    agent = AdaptiveAgent(task=task,
                      sim_repertoire_arrays=simu_arrs,
                      damage_dictionary=damage_dict)
    gp = GaussianProcess()

    exp_sh_algo = ExperienceSharingAlgorithm(family=fam,
                                      agent=agent,
                                      gaussian_process=gp,
                                      verbose=True,
                                      norm_params=norm_params,
                                      save_res_arrs=True)        
        