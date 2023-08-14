from src.utils.all_imports import *
from src.adaptation_algorithms.experience_sharing_algorithm import ExperienceSharingAlgorithm
from src.utils.hexapod_damage_dicts import intact

class InHERA(ExperienceSharingAlgorithm):
    """The inHERA algorithm which allows for experience sharing/inheritance as well as taking into account the uncertainty of ancestors with regards to their final GP means"""
    def __init__(self, family, agent, gaussian_process, alpha=0.9, verbose=False,
                 path_to_results="families/ite_example/", save_res_arrs=True, norm_params=(0, 40), plot_repertoires=False,):
        
        super().__init__(family, agent, gaussian_process, alpha, verbose,
                 path_to_results, save_res_arrs, norm_params, plot_repertoires)

        self.add_simulation_ancestor()

        # Regularisation and constrained optimisation settings
        self.gaussian_process.set_box_gradient_projection()
        self.gaussian_process.set_projection_hyperparameters(hyperparams=(0, 1))
        self.gaussian_process.loss = self.gaussian_process.loss_regularised_l1_invmax
    
    def get_ancestor_weights(self, counter):
        W = self.gaussian_process.optimise_W(x_observed=self.agent.x_observed[:counter+1],
                                                    y_observed=self.agent.y_observed[:counter+1],
                                                    y_priors=self.ancestor_mus_at_obs[:, :counter+1])
        if self.verbose: print(f"GPCF's weights: {W}")
        return W
    
    def update_mean_function(self, counter):
        ...
    
    def add_simulation_ancestor(self, percentage_of_initial_uncertainty=0.5):
        """Please note that the simulation anceestor is not the intact ancestor"""
        simulated_mu = jnp.expand_dims(a=self.family.sim_fitnesses, axis=0)
        initial_uncertainty = self.gaussian_process.kernel(1, 1) + self.gaussian_process.obs_noise
        simulated_uncertainty = percentage_of_initial_uncertainty * initial_uncertainty
        simulated_var = jnp.full(shape=(1, self.family.ancestor_mus.shape[1]), fill_value=simulated_uncertainty)
        name = "simulated"

        self.family.ancestor_mus = jnp.append(arr=self.family.ancestor_mus, values=simulated_mu, axis=0)
        self.family.ancestors_vars = jnp.append(arr=self.family.ancestors_vars, values=simulated_var, axis=0)
        self.family.ancestors_names += [name]
        self.family.damage_dicts += [intact]

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
    damage_dict = hexapod_damage_dicts.leg_0_broken
    rep_loader = RepertoireLoader()
    simu_arrs = rep_loader.load_repertoire(repertoire_path=f"{path_to_family}/repertoire",)
    agent = AdaptiveAgent(task=task,
                      sim_repertoire_arrays=simu_arrs,
                      damage_dictionary=damage_dict)
    gp = GaussianProcess()

    algo = InHERA(family=fam,
                agent=agent,
                gaussian_process=gp,
                verbose=True,
                norm_params=norm_params,
                save_res_arrs=True)
    
    # algo.run(num_iter=5)