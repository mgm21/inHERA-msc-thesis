from src.utils.all_imports import *
from src.utils.repertoire_visualiser import Visualiser as RepertoireVisualiser

# CHANGE THESE 
seed = 20
repertoire_path = f"numiter40k_final_families/family-seed_{seed}_last_repertoire/repertoire/"
plot_path = "./some_repertoire.png"

batch_size = 256
env_name = 'hexapod_uni'
episode_length = 150
num_iterations = 40000
policy_hidden_layer_sizes = (64, 64)
iso_sigma = 0.005
line_sigma = 0.05
min_bd = 0.
max_bd = 1.0
grid_shape = tuple([3])*6

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

random_key = jax.random.PRNGKey(seed)

# Init population of policies
random_key, subkey = jax.random.split(random_key)
fake_batch = jnp.zeros(shape=(env.observation_size,))
fake_params = policy_network.init(subkey, fake_batch)

_, reconstruction_fn = ravel_pytree(fake_params)

repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=repertoire_path)

fig, _ = plot_multidimensional_map_elites_grid(
        repertoire=repertoire,
        maxval=jnp.asarray([max_bd]),
        minval=jnp.asarray([min_bd]),
        grid_shape=grid_shape,)

fig.savefig("./some_repertoire.png", dpi=600)