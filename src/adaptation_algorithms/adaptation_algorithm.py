from src.utils.all_imports import *

class AdaptationAlgorithm:
    def __init__(self, agent, gaussian_process, alpha=0.9, verbose=False,
                 path_to_results="results/algorithm_example/", save_res_arrs=True, norm_params=(0, 40), plot_repertoires=False):
        self.agent = agent
        self.alpha = alpha
        self.verbose = verbose
        self.gaussian_process = gaussian_process
        self.path_to_results = path_to_results
        self.plot_repertoires = plot_repertoires
        self.save_res_arrs = save_res_arrs
        self.norm_params = norm_params

    def run(self, num_iter=20):
        self.run_setup(num_iter)

        for counter in range(num_iter):
            self.observe_acquisition_point(counter)
            self.update_mean_function(counter)
            self.train_gaussian_process(counter)

            if self.plot_repertoires:
                self.agent.plot_repertoire(quantity="mu", path_to_save_to=self.path_to_results + f"mu{counter}")
                self.agent.plot_repertoire(quantity="var", path_to_save_to=self.path_to_results + f"var{counter}")

        self.save_results()
    
    
    def run_setup(self, num_iter):
        self.seed = 1
        self.random_key = jax.random.PRNGKey(self.seed)
        self.random_key, self.subkey = jax.random.split(self.random_key)
        self.mean_func_at_obs = jnp.full(shape=num_iter, fill_value=jnp.nan)
        self.agent.x_observed = jnp.full(shape=(num_iter, 6), fill_value=jnp.nan)
        self.agent.y_observed = jnp.full(shape=num_iter, fill_value=jnp.nan)
        self.jitted_gp_acquisition = jax.jit(self.gaussian_process.acquisition_function)
        self.tested_indices = jnp.full(shape=num_iter, fill_value=jnp.nan, dtype=int)
    
    def observe_acquisition_point(self, counter):
            if self.verbose: print(f"Iteration: {counter}")

            index_to_test = self.jitted_gp_acquisition(mu=self.agent.mu, var=self.agent.var)
            self.tested_indices = self.tested_indices.at[counter].set(index_to_test)
            if self.verbose: print(f"index_to_test: {index_to_test}")
            if self.verbose: print(f"tested_indices: {self.tested_indices}")

            observed_fitness, _, _, self.random_key = self.agent.test_descriptor(index=index_to_test, random_key=self.random_key)
            observed_fitness = (observed_fitness - self.norm_params[0])/(self.norm_params[1] - self.norm_params[0])
            if self.verbose: print(f"observed_fitness: {observed_fitness} compared to original fitness: {self.agent.sim_fitnesses[index_to_test]}")

            self.update_agent_arrays(counter, index_to_test, observed_fitness)
    
    def update_mean_function(self, counter):
        # Override this method if there is need to update the mean function (prior)
        self.mean_func = jnp.zeros(shape=self.agent.sim_fitnesses.shape)
        self.mean_func_at_obs = self.mean_func_at_obs.at[counter].set(0)
        if self.verbose: print(f"mean_func_at_obs: {self.mean_func_at_obs}")

    def train_gaussian_process(self, counter):
        self.agent.mu, self.agent.var = self.gaussian_process.train(x_observed=self.agent.x_observed[:counter+1],
                                                                    y_observed=self.agent.y_observed[:counter+1] - self.mean_func_at_obs[:counter+1],
                                                                    x_test=self.agent.sim_descriptors,
                                                                    y_priors=self.mean_func)
    
    def save_results(self,):
        if self.save_res_arrs:
            if not os.path.exists(path=f"{self.path_to_results}"):
                os.makedirs(name=f"{self.path_to_results}")
            
            jnp.save(file=f"{self.path_to_results}mu", arr=self.agent.mu)
            jnp.save(file=f"{self.path_to_results}var", arr=self.agent.var)
            jnp.save(file=f"{self.path_to_results}x_observed", arr=self.agent.x_observed)
            jnp.save(file=f"{self.path_to_results}y_observed", arr=self.agent.y_observed)

            with open(f"{self.path_to_results}damage_dict.txt", "w") as file_path:
                json.dump(self.agent.damage_dictionary, file_path)    

    def update_agent_arrays(self, counter, index_to_test, observed_fitness):
        self.agent.x_observed = self.agent.x_observed.at[counter].set(self.agent.sim_descriptors[index_to_test])
        self.agent.y_observed = self.agent.y_observed.at[counter].set(observed_fitness.item())
        if self.verbose: print(f"agent's x_observed: {self.agent.x_observed}")
        if self.verbose: print(f"agent's y_observed: {self.agent.y_observed}") 
    


if __name__ == "__main__":
    from src.core.task import Task
    from src.loaders.repertoire_loader import RepertoireLoader
    from src.core.adaptive_agent import AdaptiveAgent
    from src.core.gaussian_process import GaussianProcess
    from src.utils import hexapod_damage_dicts
    from src.core.repertoire_optimiser import RepertoireOptimiser
    from src.utils.repertoire_visualiser import Visualiser

    from results.family_3 import family_task
    path_to_family = "results/family_3"
    task = family_task.task
    repertoire_loader = RepertoireLoader()
    simu_arrs = repertoire_loader.load_repertoire(repertoire_path=f"{path_to_family}/repertoire", remove_empty_bds=False)
    damage_dict = hexapod_damage_dicts.leg_0_broken
    agent = AdaptiveAgent(task=task, sim_repertoire_arrays=simu_arrs, damage_dictionary=damage_dict)
    gp = GaussianProcess()
    norm_params = jnp.load(f"{path_to_family}/norm_params.npy")

    adaptation_algo = AdaptationAlgorithm(agent=agent,
                                          gaussian_process=gp,
                                          verbose=True)
    adaptation_algo.run(num_iter=5)
