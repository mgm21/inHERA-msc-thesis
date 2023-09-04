from src.utils.all_imports import *
from src.adaptation_algorithms.experience_sharing_algorithm import ExperienceSharingAlgorithm
from src.utils.hexapod_damage_dicts import intact

class InHERAB0(ExperienceSharingAlgorithm):
    """The inHERA algorithm which allows for experience sharing/inheritance as well as taking into account the uncertainty of ancestors with regards to their final GP means"""
    def __init__(self, family, agent, gaussian_process, alpha=0.9, verbose=False,
                 path_to_results="families/ite_example/", save_res_arrs=True, norm_params=(0, 40), plot_repertoires=False,):
        
        super().__init__(family, agent, gaussian_process, alpha, verbose,
                 path_to_results, save_res_arrs, norm_params, plot_repertoires)

        self.create_simulated_arrays()

        # Set best hyperparameter
        gaussian_process.uncertainty_regularisation_weight = 0.00001

        # Regularisation and constrained optimisation settings
        self.gaussian_process.set_box_gradient_projection()
        self.gaussian_process.set_projection_hyperparameters(hyperparams=(-jnp.inf, jnp.inf))
        self.gaussian_process.loss = self.gaussian_process.loss_regularised_l1_invmax_uncertainty
    
    def get_ancestor_weights(self, counter):
        W = self.gaussian_process.optimise_W(x_observed=self.agent.x_observed[:counter+1],
                                                    y_observed=self.agent.y_observed[:counter+1],
                                                    y_priors=self.ancestor_mus_at_obs[:, :counter+1],
                                                    y_priors_vars=self.ancestor_vars_at_obs[:, :counter+1])
        if self.verbose: print(f"GPCF's weights: {W}")
        return W
    
    def update_mean_function(self, counter):
        self.ancestor_mus_at_curr_obs = self.family.ancestor_mus[:, self.tested_indices[counter:counter+1]]
        self.ancestor_mus_at_obs = self.ancestor_mus_at_obs.at[:, counter].set(jnp.squeeze(self.ancestor_mus_at_curr_obs))
        self.ancestor_vars_at_curr_obs = self.family.ancestors_vars[:, self.tested_indices[counter:counter+1]]
        self.ancestor_vars_at_obs = self.ancestor_vars_at_obs.at[:, counter].set(jnp.squeeze(self.ancestor_vars_at_curr_obs))

        W = self.get_ancestor_weights(counter,)

        self.mean_func = self.simulated_mu + (self.simulated_var - self.family.ancestors_vars.T) * (self.family.ancestor_mus.T - self.simulated_mu_repeated_num_ancest_times) @ W
        self.mean_func_at_obs = self.mean_func[self.tested_indices[:counter+1]]
    
    def run_setup(self, num_iter):
        super().run_setup(num_iter)
        num_ancestors = len(self.family.ancestor_mus)
        self.ancestor_vars_at_obs = jnp.full(shape=(num_ancestors, num_iter), fill_value=jnp.nan)
        print(f"self.ancestor_vars_at_obs: {self.ancestor_vars_at_obs}")

    
    def create_simulated_arrays(self):
        """Please note that the simulation anceestor is not the intact ancestor"""
        simulated_mu = self.family.sim_fitnesses
        initial_uncertainty = self.gaussian_process.kernel(1, 1) + self.gaussian_process.obs_noise
        simulated_var = jnp.full(shape=self.family.ancestor_mus.T.shape, fill_value=initial_uncertainty)
        num_of_ancestors = self.family.ancestor_mus.shape[0]

        # These are the base arrays that will be used in the inHERA equation. If they must be changed, it should be done here
        self.simulated_mu = simulated_mu
        self.simulated_mu_repeated_num_ancest_times = jnp.repeat(a=jnp.expand_dims(a=self.simulated_mu, axis=1), repeats=num_of_ancestors, axis=1)

        self.simulated_mu *= 1e-10
        self.simulated_mu_repeated_num_ancest_times *= 1e-10

        self.simulated_var = simulated_var


if __name__ == "__main__":
    from src.core.family import Family
    from src.core.adaptive_agent import AdaptiveAgent
    from src.core.task import Task
    from src.utils import hexapod_damage_dicts
    from src.core.gaussian_process import GaussianProcess
    from src.loaders.repertoire_loader import RepertoireLoader

    path_to_family = "families/numiter40k_final_families"
    from numiter40k_final_families import family_task
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
                save_res_arrs=False)
    
    algo.run(num_iter=5)