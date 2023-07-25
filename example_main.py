from all_imports import *

from task import Task
from repertoire_loader import RepertoireLoader
from adaptive_agent import AdaptiveAgent
from gaussian_process import GaussianProcess

# TODO: will need to make sure that the random keys are passed through the methods for the program flow to keep
# keys passing consistently

# TODO: I am a little unsure about if I am passing keys correctly and globally unsure about 2 concepts:
#  1. Where random is taking effect (for this go over all qdax code and see when keys are necessary)
#  2. How resetting works and how I would implement RTE, for example

# DEFINE THE TASK
task = Task()
task.define_task_functions()

# EXTRACTED THE SIMULATED REPERTOIRE ARRAYS
repertoire_loader = RepertoireLoader()
_, descriptors, sim_fitnesses, genotypes = repertoire_loader.load_repertoire(repertoire_path="./last_repertoire/")

# DEFINE A RANDOM KEY
seed = 1
random_key = jax.random.PRNGKey(seed)
random_key, subkey = jax.random.split(random_key)

# INSTANTIATE AN ADAPTIVE AGENT
agent = AdaptiveAgent()

# CREATE AND USE GP (this is what the algorithm will do)
gp = GaussianProcess()

# Observe the best policy that had been seen in the repertoire
best_repertoire_policy = jax.tree_map(
        lambda x: x[sim_fitnesses == jnp.max(sim_fitnesses)], jax.vmap(task.recons_fn)(genotypes)
    )

observed_fitness, observed_descriptor, extra_scores, random_key = task.scoring_fn(
            best_repertoire_policy, random_key
        )

agent.y_observed = observed_fitness
agent.x_observed = observed_descriptor
x_test = descriptors

# Run gp.train and then check whether the new mu and var have updated as expected
agent.mu, agent.var = gp.train(x_observed=agent.x_observed,
          y_observed=agent.y_observed,
          x_test=x_test)

