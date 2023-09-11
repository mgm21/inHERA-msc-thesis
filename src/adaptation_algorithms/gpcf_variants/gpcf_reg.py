from src.utils.all_imports import *
from src.adaptation_algorithms.gpcf_variant import GPCFVariant

class GPCFReg(GPCFVariant):
    """GPCF algorithm with a regularised likelihood"""
    def __init__(self, family, agent, gaussian_process, alpha=0.9, verbose=False,
                 path_to_results="families/ite_example/", save_res_arrs=True, norm_params=(0, 40), plot_repertoires=False):
        
        super().__init__(family, agent, gaussian_process, alpha, verbose,
                 path_to_results, save_res_arrs, norm_params, plot_repertoires)
        

        # Set best hyperparameters of algorithm 
        gaussian_process.kappa = 0.1
        
        # Details of regularisation
        self.gaussian_process.set_box_gradient_projection()
        self.gaussian_process.set_projection_hyperparameters(hyperparams=(-jnp.inf, +jnp.inf))
        self.gaussian_process.loss = self.gaussian_process.loss_regularised_l1_invmax
    
    def get_ancestor_weights(self, counter):
        W = self.gaussian_process.optimise_W(x_observed=self.agent.x_observed[:counter+1],
                                                    y_observed=self.agent.y_observed[:counter+1],
                                                    y_priors=self.ancestor_mus_at_obs[:, :counter+1])
        if self.verbose: print(f"GPCF's weights: {W}")
        return W


# TODO: another idea would be a decaying 1trust or decaying regularisation to be more and more willing to share the trust?

if __name__ == "__main__":
    from src.core.family import Family
    from src.core.adaptive_agent import AdaptiveAgent
    from src.core.task import Task
    from src.utils import hexapod_damage_dicts
    from src.core.gaussian_process import GaussianProcess
    from src.loaders.repertoire_loader import RepertoireLoader

    path_to_family = "families/family_4"
    from families.family_4 import family_task
    fam = Family(path_to_family=path_to_family)
    task = family_task.task
    norm_params = jnp.load(f"{path_to_family}/norm_params.npy")
    damage_dict = hexapod_damage_dicts.leg_1_broken
    rep_loader = RepertoireLoader()
    simu_arrs = rep_loader.load_repertoire(repertoire_path=f"{path_to_family}/repertoire",)
    agent = AdaptiveAgent(task=task,
                      sim_repertoire_arrays=simu_arrs,
                      damage_dictionary=damage_dict)
    gp = GaussianProcess()

    gpcf = GPCFReg(family=fam,
                agent=agent,
                gaussian_process=gp,
                verbose=True,
                norm_params=norm_params,
                save_res_arrs=True)
    
    gpcf.run(num_iter=5)
