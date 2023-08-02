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
        self.agent.y_observed = jnp.full(shape=num_iter, fill_value=-jnp.inf)
        sim_vals_at_obs = jnp.full(shape=num_iter, fill_value=jnp.nan)
        tested_indices = jnp.full(shape=num_iter, fill_value=jnp.nan, dtype=int)
        num_ancestors = len(self.family.ancestor_mus)
        ancestor_mus_at_obs = jnp.full(shape=(num_ancestors, num_iter), fill_value=jnp.nan)
        mean_func_at_obs = jnp.full(shape=num_iter, fill_value=jnp.nan)
        
        jitted_acquisition = jax.jit(self.gaussian_process.acquisition_function)

        # Main loop
        # Repeat the following while the algorithm has not terminated
        # TODO: must decide which stop condition is appropriate here (problems when second one is used because mu gets decreased and it always considers the first solution to be the best)??? Investigate!
        while counter < num_iter: #and jnp.max(self.agent.y_observed[:counter+1], initial=-jnp.inf) < self.alpha*jnp.max(self.agent.mu):
            if self.verbose: print(f"counter: {counter}")
            # Query the acquisition function for the next policy to test
            index_to_test = jitted_acquisition(mu=self.agent.mu, var=self.agent.var)

            # If index_to_test has already been tested then stop the loop
            if index_to_test in tested_indices: break

            tested_indices = tested_indices.at[counter].set(index_to_test)
            if self.verbose: print(f"index_to_test: {index_to_test}")
            if self.verbose: print(f"tested_indices: {tested_indices}")

            # debugging1 = self.agent.sim_fitnesses[199]
            # debugging2 = self.family.sim_fitnesses[199]

            # print(f"debugging1: {debugging1}")
            # print(f"debugging2: {debugging2}")

            # debugging, _, _, _ = self.agent.test_descriptor(index=199, random_key=random_key)
            # print(f"debugging: {debugging}")

            # Test the policy on the current agent
            observed_fitness, _, _, random_key = self.agent.test_descriptor(index=index_to_test, random_key=random_key)
            if self.verbose: print(f"observed_fitness: {observed_fitness} compared to repertoire fitness: {self.family.sim_fitnesses[index_to_test]}")

            # Add the new observation to the agent's observation arrays
            self.agent.x_observed = self.agent.x_observed.at[counter].set(self.agent.sim_descriptors[index_to_test])
            self.agent.y_observed = self.agent.y_observed.at[counter].set(observed_fitness.item())
            if self.verbose: print(f"agent's x_observed: {self.agent.x_observed}")
            if self.verbose: print(f"agent's y_observed: {self.agent.y_observed}")

            # Append the ancestor_mus_at_obs array
            ancestor_mus_at_curr_obs = self.family.ancestor_mus[:, tested_indices[counter:counter+1]]
            ancestor_mus_at_obs = ancestor_mus_at_obs.at[:, counter].set(jnp.squeeze(ancestor_mus_at_curr_obs))
            if self.verbose: print(f"ancestor_mus_at_curr_obs: {ancestor_mus_at_curr_obs}")
            if self.verbose: print(f"ancestor_mus_at_obs: {ancestor_mus_at_obs}")
            
            # Optimise the weights of the ancestors given the new observation
            W = self.gaussian_process.optimise_W(x_observed=self.agent.x_observed[:counter+1],
                                                  y_observed=self.agent.y_observed[:counter+1],
                                                  y_priors=ancestor_mus_at_obs[:, :counter+1])
            if self.verbose: print(f"W: {W}")

            # Calculate the new mean func at the observations
            mean_func_at_obs = mean_func_at_obs.at[counter].set(jnp.squeeze(ancestor_mus_at_obs[:, counter:counter+1].T @ W))
            print(f"mean_func_at_obs: {mean_func_at_obs}")

            # Calculate the new mean func for all of x_test (i.e. agent.sim_descriptors)
            mean_func_at_x_test = self.family.ancestor_mus.T @ W
            print(f"mean_func_at_x_test: {mean_func_at_x_test}")

            # Update the gaussian process of the agent
            self.agent.mu, self.agent.var = self.gaussian_process.train(self.agent.x_observed[:counter+1],
                                                                        self.agent.y_observed[:counter+1] - mean_func_at_obs[:counter+1],
                                                                        self.agent.sim_descriptors,
                                                                        y_priors=mean_func_at_x_test)

            # Increment the counter
            counter += 1

if __name__ == "__main__":
    from src.family import Family
    from src.adaptive_agent import AdaptiveAgent
    from src.task import Task
    from src.utils import hexapod_damage_dicts
    from src.gaussian_process import GaussianProcess

    fam = Family(path_to_family="results/family_0")
    task = Task(episode_length=150, num_iterations=500, grid_shape=tuple([4]*6))
    damage_dict = hexapod_damage_dicts.leg_0_broken
    print(fam.ancestors_names)
    simu_arrs = fam.centroids, fam.descriptors, fam.sim_fitnesses, fam.genotypes
    agent = AdaptiveAgent(task=task, sim_repertoire_arrays=simu_arrs, damage_dictionary=damage_dict)
    gp = GaussianProcess()
    gpcf = GPCF(family=fam, agent=agent, gaussian_process=gp, verbose=True)
    gpcf.run(num_iter=5)

    # TODO: find a better way to automatically let the agent know its family's repertoire without the whole simu_arrs input above.
    # maybe a method in family to generate an agent so that it already has its arrays in the scope
    #  Ultimately, should revise how the AdaptiveAgent is defined.

    # TODO: Maybe the kwargs of the Task's constructor for a whole family should also be saved in the family folder
    #  to be loaded with the family and input into the agent for example

    # TODO: strange but when running intact with GPCF the first (and only) iteration does not give W with highest on intact ancestor
    #  is this normal? Is the fact that it is not 'ranking' them but creating the best weighted sum of them a good idea?

    # TODO: problem with the ones that have UPDATING mean functions, they also update the points that had -inf fitness to start off with
    # and for some reason those points are being suggested. Investigate. This is not a problem with ITE where there will always be a -inf
    # mean value in position 0 for example. Though shouldn't there always be a very low number at those indices?

    # TODO: problem with the fact that when trying to score leg_5_broken agent for example, not getting the same results it got with ITE when its mu was created.
    # This does not make sense to me. Actually this is fine. It's because it had not observed that point which actually was not a good point to observe.
