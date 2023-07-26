from all_imports import *

from task import Task
from repertoire_loader import RepertoireLoader
from adaptive_agent import AdaptiveAgent
from gaussian_process import GaussianProcess
import hexapod_damage_dicts
from repertoire_optimiser import RepertoireOptimiser

# TODO: All problems pertaining to random keys flowing in the program. 
#  I am a little unsure about if I am passing keys correctly and globally unsure about 2 concepts:
#  1. Where random is taking effect (for this go over all qdax code and see when keys are necessary)
#  2. How resetting works and how I would implement RTE, for example

# TODO: make functions that are modular to get a quick html of a policy from an old repertoire but for a new robot

# TODO: should I as soon as the MAP-Elites grid is formed, trim all the sim_repertoire_arrays to only include the indices
#  where fitness is not -inf (like: genotypes[fitnesses != -jnp.inf]) etc...

# TODO: (PRIORITY FOR TODAY) refactorings left to do: make Task object a part of the RepertoireOptimiser and make a full flow here.
# 1st part could be as simple as inputting task to RepertoireOptimiser and extracting all its attributes AND FUNCTIONS

# DEFINE THE TASK
task = Task(episode_length=50, num_iterations=50)

# OPTIMISE A REPERTOIRE IN SIMULATION (hopefully passing the RepertoireOptimiser a Task object so that it does not repeat
# code)
# TODO: pass the task object to the RepertoireOptimiser so there is no code repeat ultimately
# TODO: I think that in the end Task should only include data (be like a data class (not functions, will have to com
# pute a new one for each agent))

# repertoire_optimsier = RepertoireOptimiser(task=task)

# repertoire_optimsier.optimise_repertoire(repertoire_path="./example_main_repertoire/",
#                                          plot_path="./example_main",
#                                          html_path="./example_main.html")

# EXTRACTED THE SIMULATED REPERTOIRE ARRAYS
repertoire_loader = RepertoireLoader()
sim_repertoire_arrays = _, descriptors, sim_fitnesses, genotypes = repertoire_loader.load_repertoire(repertoire_path="./last_repertoire/")

# DEFINE A RANDOM KEY
seed = 1
random_key = jax.random.PRNGKey(seed)
random_key, subkey = jax.random.split(random_key)

# INSTANTIATE AN ADAPTIVE AGENT

# TODO: change damage indices to a dictionary
damage_indices = [0, 0, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200]
damage_dict = hexapod_damage_dicts.get_damage_dict(damage_indices)

agent = AdaptiveAgent(task=task,
                      sim_repertoire_arrays=sim_repertoire_arrays,
                      damage_dictionary=damage_dict)

# CREATE AND USE GP (this is what the algorithm will do)
gp = GaussianProcess()

# START FLAG (put all in agent class. it tests its own policies and has access to what its simulated prior is
# I am making the choice to make it 'knowledgable' about this to make it more of an active agent. However,
# need to find a good way for it to test its policies. Doing it through fitness values is not robust (could have
# multiple BDs with the same fitness and it is trying to test a BD not a fitness value))

# TODO: Acquisition function query from Gaussian Process (using mu and var of the agent)

# TODO: Test the policy that the GP has asked for (testing a BD)

# Update the mu and var of the agent

# Continue until reaching an end condition

# Observe the best policy that had been seen in the repertoire
best_repertoire_policy = jax.tree_map(
        lambda x: x[sim_fitnesses == jnp.max(sim_fitnesses)], jax.vmap(agent.recons_fn)(genotypes)
    )

index = 201
debugging = jax.vmap(agent.recons_fn)(genotypes[index:index+1])

observed_fitness, observed_descriptor, extra_scores, random_key = agent.scoring_fn(
            best_repertoire_policy, random_key
        )

agent.y_observed = observed_fitness
agent.x_observed = observed_descriptor
x_test = descriptors

print(agent.x_observed)
print(agent.y_observed)

# END FLAG

# Run gp.train and then check whether the new mu and var have updated as expected
agent.mu, agent.var = gp.train(x_observed=agent.x_observed,
          y_observed=agent.y_observed,
          x_test=x_test)


