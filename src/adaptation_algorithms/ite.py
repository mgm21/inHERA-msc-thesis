from src.adaptation_algorithms.abstract_adaptation_algorithms.adaptation_algorithm import AdaptationAlgorithm

class ITE(AdaptationAlgorithm):
    def __init__(self, agent, gaussian_process, alpha=0.9, verbose=False,
                 path_to_results="families/algorithm_example/", save_res_arrs=True, norm_params=(0, 40), plot_repertoires=True):
        super().__init__(agent, gaussian_process, alpha, verbose,
                 path_to_results, save_res_arrs, norm_params, plot_repertoires)
    
        self.mean_func = self.agent.sim_fitnesses
    
    def update_mean_function(self, counter):
        self.mean_func_at_obs = self.mean_func_at_obs.at[counter].set(self.mean_func[self.tested_indices[counter]])
        if self.verbose: print(f"mean_func_at_obs: {self.mean_func_at_obs}")

if __name__ == "__main__":
    from src.loaders.repertoire_loader import RepertoireLoader
    from src.core.adaptation.adaptive_agent import AdaptiveAgent
    from src.core.adaptation.gaussian_process import GaussianProcess
    from src.utils import hexapod_damage_dicts

    from families.family_3 import family_task
    path_to_family = "families/family_3"
    task = family_task.task
    repertoire_loader = RepertoireLoader()
    simu_arrs = repertoire_loader.load_repertoire(repertoire_path=f"{path_to_family}/repertoire", remove_empty_bds=False)
    damage_dict = hexapod_damage_dicts.leg_1_broken
    agent = AdaptiveAgent(task=task, sim_repertoire_arrays=simu_arrs, damage_dictionary=damage_dict)
    gp = GaussianProcess()
    norm_params = jnp.load(f"{path_to_family}/norm_params.npy")

    ite = ITE(agent=agent,
              gaussian_process=gp,
              verbose=True,
              norm_params=norm_params)
    
    ite.run(num_iter=5)