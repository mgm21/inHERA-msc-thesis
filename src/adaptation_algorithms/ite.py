from src.utils.all_imports import *

class ITE():
    def __init__(self, agent, gaussian_process, alpha=0.9, path_to_results="results/ite_example", plot_repertoires=True):
        self.agent = agent
        self.gaussian_process = gaussian_process
        self.alpha = alpha

        self.y_prior = self.agent.sim_fitnesses
        self.path_to_results = path_to_results
        self.plot_repertoires = plot_repertoires

    def run(self):
        # TODO: randomness check is this how it should be used? 
        # Define random key
        seed = 1
        random_key = jax.random.PRNGKey(seed)
        random_key, subkey = jax.random.split(random_key)

        # TODO: add an option to save results (to a file specified in constructor?)
        counter = 0

        y_prior_at_obs = jnp.array([])

        # Repeat the following while the algorithm has not terminated
        while counter < 10 and jnp.max(self.agent.y_observed, initial=-jnp.inf) < self.alpha*jnp.max(self.agent.mu):
            counter += 1

            print(f"iteration: {counter}")
        
            # Query the GPs acquisition function based on the agent's mu and var
            # TODO: make it so that the acquisition function can only return points that are not -inf fitness
            # Though I am guessing they are not returned because the acquisition function would NEVER think they
            # are a better solution to try
            index_to_test = gp.acqusition_function(mu=self.agent.mu, var=self.agent.var)
            print(f"index_to_test: {index_to_test}")

            # TODO: question: should I update the descriptor that was trialled or the one that was observed??? I think the one that was trialled as below:
            # with self.agent.sim_descriptors[index_to_test]

            # Agent tests the result of the acquisition function
            observed_fitness, observed_descriptor, _, random_key = self.agent.test_descriptor(index=index_to_test, random_key=random_key)
            
            #  TODO: will have to change this implementation to make things jittable (no changing shapes). Could have initial arrays with -inf everywhere then not use these
            # in the gp somehow... though I do not know if any of this will work with jit (obviously the if statements won't...)
            if len(self.agent.x_observed) == 0:
                self.agent.x_observed = jnp.expand_dims(self.agent.sim_descriptors[index_to_test], axis=0)
                self.agent.y_observed = observed_fitness
            else:
                self.agent.x_observed = jnp.append(self.agent.x_observed, jnp.expand_dims(self.agent.sim_descriptors[index_to_test], 0), axis=0)
                self.agent.y_observed = jnp.append(self.agent.y_observed, observed_fitness)

            print(f"agent's x_observed: {self.agent.x_observed}")
            print(f"agent's y_observed: {self.agent.y_observed}")

            # Define the mean function
            y_prior_at_obs = jnp.append(y_prior_at_obs, self.y_prior[index_to_test]) 

            # Update the beliefs of thea agent about mu and var
            # TODO: How to best deal with the fact that the -inf fitnessed descriptors are also be updated meaning the acquisition could query one of those
            # policies which is empty? Clip them upon receiving them in the loader? E.g. as soon as the loader receives them trim all the arrays of their
            # -inf BDs. Then these clipped arrays will be passed to all the children.

            # TODO: check with Antoine that the way I did the GP with prior makes sense
            self.agent.mu, self.agent.var = self.gaussian_process.train(self.agent.x_observed,
                                                                        self.agent.y_observed - y_prior_at_obs,
                                                                        self.agent.sim_descriptors,
                                                                        y_prior=self.y_prior)
            if self.plot_repertoires:
                # Save the plots of the repertoires
                self.agent.plot_repertoire(quantity="mu", path_to_save_to=self.path_to_results + f"_mu{counter}")
                self.agent.plot_repertoire(quantity="var", path_to_save_to=self.path_to_results + f"_var{counter}")
            
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
        
    # Define a simulated repertoire # TODO could make RepertoireLoader a class called Repertoire which stores the repertoire arrays?
    repertoire_loader = RepertoireLoader()
    simu_arrs = repertoire_loader.load_repertoire(repertoire_path="results/ite_example/sim_repertoire", remove_empty_bds=False)

    # Define an Adaptive Agent wihch inherits from the task and gets its mu and var set to the simulated repertoire's mu and var
    damage_dict = hexapod_damage_dicts.all_actuators_broken
    agent = AdaptiveAgent(task=task, sim_repertoire_arrays=simu_arrs, damage_dictionary=damage_dict)

    # Define a GP
    gp = GaussianProcess()

    # Create an ITE object with previous objects as inputs
    ite = ITE(agent=agent, gaussian_process=gp, alpha=0.99)

    # Run the ITE algorithm
    ite.run()