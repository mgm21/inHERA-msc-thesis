import os

from IPython.display import clear_output
import functools
import time

import jax
import jax.numpy as jnp

import brax
import jumanji
import qdax


from qdax.core.map_elites import MAPElites
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids, compute_euclidean_centroids, MapElitesRepertoire
from qdax import environments
from qdax.tasks.brax_envs import scoring_function_brax_envs as scoring_function
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.utils.plotting import plot_map_elites_results
from qdax.utils.plotting import plot_multidimensional_map_elites_grid

from qdax.utils.metrics import CSVLogger, default_qd_metrics

from jax.flatten_util import ravel_pytree

from IPython.display import HTML
from brax.io import html

if "COLAB_TPU_ADDR" in os.environ:
  from jax.tools import colab_tpu
  colab_tpu.setup_tpu()

clear_output()


repertoire_path = "./last_repertoire/"

# Init hyperparameters
batch_size = 128 #@param {type:"number"}
env_name = 'hexapod_uni' #@param['ant_uni', 'hopper_uni', 'walker2d_uni', 'halfcheetah_uni', 'humanoid_uni', 'ant_omni', 'humanoid_omni']
episode_length = 200 #@param {type:"integer"}
num_iterations = 50 #@param {type:"integer"}
seed = 42 #@param {type:"integer"}
policy_hidden_layer_sizes = (64, 64) #@param {type:"raw"}
iso_sigma = 0.005 #@param {type:"number"}
line_sigma = 0.05 #@param {type:"number"}
# num_init_cvt_samples = 50000 #@param {type:"integer"}
# num_centroids = 1024 #@param {type:"integer"}
min_bd = 0. #@param {type:"number"}
max_bd = 1. #@param {type:"number"}
# for higher-dimensional (>2) BD
grid_shape = (5, 5, 5, 5, 5, 5)

# Init environment
env = environments.create(env_name, episode_length=episode_length)

# Init a random key
random_key = jax.random.PRNGKey(seed)

# Init policy network
policy_layer_sizes = policy_hidden_layer_sizes + (env.action_size,)
policy_network = MLP(
    layer_sizes=policy_layer_sizes,
    kernel_init=jax.nn.initializers.lecun_uniform(),
    final_activation=jnp.tanh,
)

# Init population of policies
random_key, subkey = jax.random.split(random_key)
fake_batch = jnp.zeros(shape=(env.observation_size,))
fake_params = policy_network.init(subkey, fake_batch)

_, reconstruction_fn = ravel_pytree(fake_params)

repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=repertoire_path)

# print(f"centroids shape: {jnp.shape(repertoire.centroids)}")
# print(f"descriptors shape: {jnp.shape(repertoire.descriptors)}")
# print(f"fitnesses shape: {jnp.shape(repertoire.fitnesses)}")
# print(f"genotypes shape: {jnp.shape(repertoire.genotypes)}")

# print(f"genotypes: {repertoire.genotypes}")

best_idx = jnp.argmax(repertoire.fitnesses)
best_fitness = jnp.max(repertoire.fitnesses)
best_bd = repertoire.descriptors[best_idx]

print(
    f"Best fitness in the repertoire: {best_fitness:.2f}\n",
    f"Behavior descriptor of the best individual in the repertoire: {best_bd}\n",
    f"Index in the repertoire of this individual: {best_idx}\n"
)


# The following returns the 5 fittest BDs in the repertoire
N = 5
best_N_idx = jnp.argpartition(repertoire.fitnesses, -N)[-N:]
best_N_fitnesses = repertoire.fitnesses[best_N_idx]
best_N_bds = repertoire.descriptors[best_N_idx]
print(best_N_bds)


my_params = jax.tree_util.tree_map(
    lambda x: x[best_idx],
    repertoire.genotypes
)

# print(my_params)
# bias0 = my_params["params"]["Dense_0"]["bias"]
# kernel0 = my_params["params"]["Dense_0"]["kernel"]
# bias1 = my_params["params"]["Dense_1"]["bias"]
# kernel1 = my_params["params"]["Dense_1"]["kernel"]
# bias2 = my_params["params"]["Dense_2"]["bias"]
# kernel2 = my_params["params"]["Dense_2"]["kernel"]

# print(f"bias0: {jnp.shape(bias0)}")
# print(f"kernel0: {jnp.shape(kernel0)}")
# print(f"bias1: {jnp.shape(bias1)}")
# print(f"kernel1: {jnp.shape(kernel1)}")
# print(f"bias2: {jnp.shape(bias2)}")
# print(f"kernel2: {jnp.shape(kernel2)}")


jit_env_reset = jax.jit(env.reset)
jit_env_step = jax.jit(env.step)
jit_inference_fn = jax.jit(policy_network.apply)

rollout = []
rng = jax.random.PRNGKey(seed=1)
state = jit_env_reset(rng=rng)
while not state.done:
    rollout.append(state)
    action = jit_inference_fn(my_params, state.obs)
    # print(f"action: {action}")
    state = jit_env_step(state, action)

# print(f"The trajectory of this individual contains {len(rollout)} transitions.")

html.save_html("test-loaded.html", env.sys, [s.qp for s in rollout[:500]])


# # UNDERSTANDING HOW TO SCORE GENOTYPES

# # Create the initial environment states
# random_key, subkey = jax.random.split(random_key)
# keys = jnp.repeat(jnp.expand_dims(subkey, axis=0), repeats=batch_size, axis=0)
# reset_fn = jax.jit(jax.vmap(env.reset))
# init_states = reset_fn(keys)

# # Define the function to play a step with the policy in the environment
# def play_step_fn(env_state, policy_params, random_key,):
#     """
#     Play an environment step and return the updated state and the transition.
#     """

#     actions = policy_network.apply(policy_params, env_state.obs)    
#     state_desc = env_state.info["state_descriptor"]
#     next_state = env.step(env_state, actions)

#     transition = QDTransition(
#         obs=env_state.obs,
#         next_obs=next_state.obs,
#         rewards=next_state.reward,
#         dones=next_state.done,
#         actions=actions,
#         truncations=next_state.info["truncation"],
#         state_desc=state_desc,
#         next_state_desc=next_state.info["state_descriptor"],
#     )

#     return next_state, policy_params, random_key, transition

# bd_extraction_fn = environments.behavior_descriptor_extractor[env_name]

# scoring_fn = functools.partial(
#     scoring_function,
#     init_states=init_states,
#     episode_length=episode_length,
#     play_step_fn=play_step_fn,
#     behavior_descriptor_extractor=bd_extraction_fn,
# )

# # scores the offsprings
# fitnesses, descriptors, extra_scores, random_key = scoring_fn(
#     repertoire.genotypes, random_key
# )

# print(fitnesses)
# print(descriptors)
# print(extra_scores)
# print(random_key)