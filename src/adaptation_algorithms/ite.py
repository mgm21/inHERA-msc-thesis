from src.utils.all_imports import *

class ITE():
    def __init__(self, agent, gaussian_process, alpha=0.9, path_to_results="results/ite_example/", plot_repertoires=False, save_res_arrs=True, verbose=False):
        self.agent = agent
        self.gaussian_process = gaussian_process
        self.alpha = alpha
        self.y_prior = self.agent.sim_fitnesses
        self.path_to_results = path_to_results
        self.plot_repertoires = plot_repertoires
        self.save_res_arrs = save_res_arrs
        self.verbose = verbose

    def run(self, counter_thresh=10):
        # TODO: randomness check is this how it should be used? 
        # Define random key
        seed = 1
        random_key = jax.random.PRNGKey(seed)
        random_key, subkey = jax.random.split(random_key)

        # Initialise the counter variable
        counter = 0

        # Define the array of the prior mean function at the observations
        y_prior_at_obs = jnp.full(shape=counter_thresh, fill_value=jnp.nan)

        # Define array of the BDs observed by the agent
        # TODO: left the "6" hard-coded here, but modularise with the self.agent.env at some point
        self.agent.x_observed = jnp.full(shape=(counter_thresh, 6), fill_value=jnp.nan)

        # Define array of the fitnesses observed by the agent
        self.agent.y_observed = jnp.full(shape=counter_thresh, fill_value=jnp.nan)

        # Jit functions
        jitted_acquisition = jax.jit(self.gaussian_process.acquisition_function)

        # Repeat the following while the algorithm has not terminated
        while counter < counter_thresh and jnp.max(self.agent.y_observed[:counter], initial=-jnp.inf) < self.alpha*jnp.max(self.agent.mu):
            if self.verbose: print(f"iteration: {counter}")
        
            # Query the GPs acquisition function based on the agent's mu and var
            index_to_test = jitted_acquisition(mu=self.agent.mu, var=self.agent.var)
            if self.verbose: print(f"index_to_test: {index_to_test}")

            # Agent tests the result of the acquisition function
            observed_fitness, _, _, random_key = self.agent.test_descriptor(index=index_to_test, random_key=random_key)
            if self.verbose: print(f"observed_fitness: {observed_fitness}")

            # Update agent's x_observed and y_observed
            self.agent.x_observed = self.agent.x_observed.at[counter].set(self.agent.sim_descriptors[index_to_test])
            self.agent.y_observed = self.agent.y_observed.at[counter].set(observed_fitness.item())
            if self.verbose: print(f"agent's x_observed: {self.agent.x_observed}")
            if self.verbose: print(f"agent's y_observed: {self.agent.y_observed}")

            # Define the mean function
            y_prior_at_obs = y_prior_at_obs.at[counter].set(self.y_prior[index_to_test])
            if self.verbose: print(f"y_prior_at_obs: {y_prior_at_obs}")

            # TODO: check with Antoine that the way I did the GP with prior makes sense
            self.agent.mu, self.agent.var = self.gaussian_process.train(self.agent.x_observed[:counter+1],
                                                                        self.agent.y_observed[:counter+1] - y_prior_at_obs[:counter+1],
                                                                        self.agent.sim_descriptors,
                                                                        y_prior=self.y_prior)
            if self.plot_repertoires:
                # Save the plots of the repertoires
                self.agent.plot_repertoire(quantity="mu", path_to_save_to=self.path_to_results + f"mu{counter}")
                self.agent.plot_repertoire(quantity="var", path_to_save_to=self.path_to_results + f"var{counter}")
            
            # Move onto the next iteration
            counter += 1

        if self.save_res_arrs:
            if not os.path.exists(path=f"{self.path_to_results}"):
                os.makedirs(name=f"{self.path_to_results}")
            jnp.save(file=f"{self.path_to_results}mu", arr=self.agent.mu)
            jnp.save(file=f"{self.path_to_results}var", arr=self.agent.var)
            jnp.save(file=f"{self.path_to_results}x_observed", arr=self.agent.x_observed)
            jnp.save(file=f"{self.path_to_results}y_observed", arr=self.agent.y_observed)
            
if __name__ == "__main__":
    # Import all the necessary libraries
    from src.task import Task
    from src.repertoire_loader import RepertoireLoader
    from src.adaptive_agent import AdaptiveAgent
    from src.gaussian_process import GaussianProcess
    from src.utils import hexapod_damage_dicts
    from src.repertoire_optimiser import RepertoireOptimiser
    from src.utils.visualiser import Visualiser

    # Define all the objects that are fed to ITE constructor:
    # Define an overall task (true for the whole family simulated and adaptive)
    task = Task()

    # # Define a repertoire optimiser
    # repertoire_optimiser = RepertoireOptimiser(task=task)
    # repertoire_optimiser.optimise_repertoire(repertoire_path="./results/ite_example/sim_repertoire")
        
    # Define a simulated repertoire 
    repertoire_loader = RepertoireLoader()
    simu_arrs = repertoire_loader.load_repertoire(repertoire_path="results/ite_example/sim_repertoire", remove_empty_bds=False)

    # To figure out the smallest element of the simulated fitnesses
    # print(jnp.min(simu_arrs[2][simu_arrs[2] != -jnp.inf]))

    # Define an Adaptive Agent wihch inherits from the task and gets its mu and var set to the simulated repertoire's mu and var
    damage_dict = hexapod_damage_dicts.leg_1_broken
    agent = AdaptiveAgent(task=task, sim_repertoire_arrays=simu_arrs, damage_dictionary=damage_dict)

    # Define a GP
    gp = GaussianProcess()

    # Create an ITE object with previous objects as inputs
    ite = ITE(agent=agent,
              gaussian_process=gp,
              alpha=0.99,
              plot_repertoires=False,)

    # # Run the ITE algorithm
    ite.run(counter_thresh=5)