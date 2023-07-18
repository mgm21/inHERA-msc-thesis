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

print(f"centroids shape: {jnp.shape(repertoire.centroids)}")
print(f"descriptors shape: {jnp.shape(repertoire.descriptors)}")
print(f"fitnesses shape: {jnp.shape(repertoire.fitnesses)}")
print(f"genotypes shape: {jnp.shape(repertoire.genotypes)}")

print(f"genotypes: {repertoire.genotypes}")



best_idx = jnp.argmax(repertoire.fitnesses)
best_fitness = jnp.max(repertoire.fitnesses)
best_bd = repertoire.descriptors[best_idx]

# print(
#     f"Best fitness in the repertoire: {best_fitness:.2f}\n",
#     f"Behavior descriptor of the best individual in the repertoire: {best_bd}\n",
#     f"Index in the repertoire of this individual: {best_idx}\n"
# )


# my_params = jax.tree_util.tree_map(
#     lambda x: x[best_idx],
#     repertoire.genotypes
# )

# jit_env_reset = jax.jit(env.reset)
# jit_env_step = jax.jit(env.step)
# jit_inference_fn = jax.jit(policy_network.apply)

# rollout = []
# rng = jax.random.PRNGKey(seed=1)
# state = jit_env_reset(rng=rng)
# while not state.done:
#     rollout.append(state)
#     action = jit_inference_fn(my_params, state.obs)
#     # print(f"action: {action}")
#     state = jit_env_step(state, action)

# print(f"The trajectory of this individual contains {len(rollout)} transitions.")

# html.save_html("test-discard.html", env.sys, [s.qp for s in rollout[:500]])