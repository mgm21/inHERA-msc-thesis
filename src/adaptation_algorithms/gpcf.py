# TODO: There is much overlap between the adaptation algorithms, can create a super class
# TODO: When doing the above, think about how you can stop evaluating mus and vars over all of x_test, and rather pass a lambda function
#  or some function definition which is much more lightweight. It is not necessary to always pass the mus and vars around to the agent.
#  can also time how long it takes to calculate mu for the whole repertoire. Multiply that by the number of times it has to be done (once per loop I think).

from src.utils.all_imports import *

class GPCF:
    def __init__(self, family, agent, gaussian_process, alpha=0.9, verbose=False,):
        self.family = family
        self.agent = agent
        self.alpha = alpha
        self.verbose = verbose
        self.gaussian_process = gaussian_process

    def run(self, num_iter=10):
        # Pre-loop setup
        seed = 1
        random_key = jax.random.PRNGKey(seed)
        random_key, subkey = jax.random.split(random_key)

        counter = 0
        self.agent.x_observed = jnp.full(shape=(num_iter, 6), fill_value=jnp.nan)
        self.agent.y_observed = jnp.full(shape=num_iter, fill_value=jnp.nan)
        sim_vals_at_obs = jnp.full(shape=num_iter, fill_value=jnp.nan)
        num_ancestors = len(self.family.ancestor_mus)
        ancestor_mus_at_obs = jnp.full(shape=(num_ancestors, num_iter), fill_value=jnp.nan)
        
        jitted_acquisition = jax.jit(self.gaussian_process.acquisition_function)

        # Main loop
        # Repeat the following while the algorithm has not terminated
        while counter < num_iter and jnp.max(self.agent.y_observed[:counter], initial=-jnp.inf) < self.alpha*jnp.max(self.agent.mu):
            # Query the acquisition function for the next policy to test
            index_to_test = jitted_acquisition(mu=self.agent.mu, var=self.agent.var)
            if self.verbose: print(f"index_to_test: {index_to_test}")

            # Test the policy on the current agent
            observed_fitness, _, _, random_key = self.agent.test_descriptor(index=index_to_test, random_key=random_key)
            if self.verbose: print(f"observed_fitness: {observed_fitness} compared to original fitness: {self.family.sim_fitnesses[index_to_test]}")

            # Add the new observation to the agent's observation arrays
            self.agent.x_observed = self.agent.x_observed.at[counter].set(self.agent.sim_descriptors[index_to_test])
            self.agent.y_observed = self.agent.y_observed.at[counter].set(observed_fitness.item())
            if self.verbose: print(f"agent's x_observed: {self.agent.x_observed}")
            if self.verbose: print(f"agent's y_observed: {self.agent.y_observed}")

            # Append the ancestor_mus_at_obs matrix (using jax.vmap or the tree operation could be good)


            # Optimise the weights of the ancestors given the new observation
            # W = self.gaussian_process.optimise_W(x_observed=x_observed, y_observed=y_observed, y_priors=ancestor_mus_at_obs)

            # Define the new prior by taking a weighted average of the ancestor arrays
            # new_mean_func = 

            # Calculate the new mean func at the observations

            # Calculate the new mean func for all of x_test (i.e. agent.sim_descriptors)

            # Update the gaussian process of the agent

            # Increment the counter
            counter += 1
            if self.verbose: print(f"counter: {counter}")

if __name__ == "__main__":
    from src.family import Family
    from src.adaptive_agent import AdaptiveAgent
    from src.task import Task
    from src.utils import hexapod_damage_dicts
    from src.gaussian_process import GaussianProcess

    fam = Family(path_to_family="results/family_0")
    task = Task(episode_length=150, num_iterations=500, grid_shape=tuple([4]*6))
    damage_dict = hexapod_damage_dicts.intact
    simu_arrs = fam.centroids, fam.descriptors, fam.sim_fitnesses, fam.genotypes
    agent = AdaptiveAgent(task=task, sim_repertoire_arrays=simu_arrs, damage_dictionary=damage_dict)
    gp = GaussianProcess()
    gpcf = GPCF(family=fam, agent=agent, gaussian_process=gp, verbose=True)
    gpcf.run()

    # TODO: find a better way to automatically let the agent know its family's repertoire without the whole simu_arrs input above.
    # maybe a method in family to generate an agent so that it already has its arrays in the scope
    # Ultimately, should revise how the AdaptiveAgent is defined.

    # TODO: Maybe the kwargs of the Task's constructor for a whole family should also be saved in the family folder
    # to be loaded with the family and input into the agent for example
