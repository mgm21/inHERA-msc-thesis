from src.utils.all_imports import *
from src.utils.repertoire_visualiser import Visualiser as RepertoireVisualiser

# CHANGE THESE 
seed = 20
repertoire_path = f"seed_20_last_repertoire/"
best = True # If you want the rollout for the max fitness individual
descriptor_coordinates = jnp.array([0, 0]) # Careful! Will only take effect if best = False
grid_shape = tuple([6])*2
save_prefix = "halfcheetah_"

batch_size = 256
env_name = 'halfcheetah_uni'
episode_length = 150
num_iterations = 40000
policy_hidden_layer_sizes = (64, 64)
iso_sigma = 0.005
line_sigma = 0.05
min_bd = 0.
max_bd = 1.0

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

if best:
    rollout_idx = jnp.argmax(repertoire.fitnesses)
    rollout_fitness = jnp.max(repertoire.fitnesses)
    html_path = f"seed_{seed}_best_rollout.html"
else:
    desired_centroids = (descriptor_coordinates + 1)/(grid_shape[0]) - 1/(2*grid_shape[0])
    n = None
    for i, centroid in enumerate(repertoire.centroids):
        if (centroid == desired_centroids).all():
            n = i
    if n == None: print("Did not find your index.")
    rollout_idx = n
    rollout_fitness = repertoire.fitnesses[rollout_idx]
    html_path = f"seed_{seed}_idx_{rollout_idx}.html"

rollout_bd = repertoire.descriptors[rollout_idx]

print(
    f"Fitness in the repertoire: {rollout_fitness:.2f}\n",
    f"Behavior descriptor of the individual in the repertoire: {rollout_bd}\n",
    f"Index in the repertoire of this individual: {rollout_idx}\n"
)

my_params = jax.tree_util.tree_map(
    lambda x: x[rollout_idx],
    repertoire.genotypes
)

jit_env_reset = jax.jit(env.reset)
jit_env_step = jax.jit(env.step)
jit_inference_fn = jax.jit(policy_network.apply)

rollout = []
rng = jax.random.PRNGKey(seed=1)
state = jit_env_reset(rng=rng)
while not state.done:
    rollout.append(state)
    action = jit_inference_fn(my_params, state.obs)
    state = jit_env_step(state, action)

print(f"The trajectory of this individual contains {len(rollout)} transitions.")

html.save_html(save_prefix + html_path, env.sys, [s.qp for s in rollout[:500]])