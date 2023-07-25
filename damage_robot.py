from all_imports import *

# Init hyperparameters
batch_size = 128 #@param {type:"number"}
env_name = 'hexapod_uni' #@param['ant_uni', 'hopper_uni', 'walker2d_uni', 'halfcheetah_uni', 'humanoid_uni', 'ant_omni', 'humanoid_omni']
episode_length = 100 #@param {type:"integer"}
num_iterations = 150 #@param {type:"integer"}
seed = 42 #@param {type:"integer"}
policy_hidden_layer_sizes = (64, 64) #@param {type:"raw"}
iso_sigma = 0.005 #@param {type:"number"}
line_sigma = 0.05 #@param {type:"number"}
# num_init_cvt_samples = 50000 #@param {type:"integer"}
# num_centroids = 1024 #@param {type:"integer"}
min_bd = 0. #@param {type:"number"}
max_bd = 1. #@param {type:"number"}
# for higher-dimensional (>2) BD
grid_shape = (3, 3, 3, 3, 3, 3)


# This dictionary should completely disable (later maybe put a random val between 0 and 0.5?) the right side of the robot
# key: an actuator, value: strength_multiplier

multiplier = 0

actuator_strength_update_dict = {
"body_leg_0": multiplier,
"leg_0_1_2": multiplier,
"leg_0_2_3": multiplier,
"body_leg_1": multiplier,
"leg_1_1_2": multiplier,
"leg_1_2_3": multiplier,
"body_leg_2": multiplier,
"leg_2_1_2": multiplier,
"leg_2_2_3": multiplier,
"body_leg_3": multiplier,
"leg_3_1_2": multiplier,
"leg_3_2_3": multiplier,
"body_leg_4": multiplier,
"leg_4_1_2": multiplier,
"leg_4_2_3": multiplier,
"body_leg_5": multiplier,
"leg_5_1_2": multiplier,
"leg_5_2_3": multiplier,
}


env = environments.create(env_name, episode_length=episode_length)

env_actuators = env.sys.config.actuators

# Did not iterate over key, vals of actuator_strength_update_dict because the env_actuators is not a Dict and I think one can only naively iterate over them though I will leave this improvement as a TODO: though there are at most 10 limbs this is not where efficiency should be looked for.
for actuator in env_actuators:
	if actuator.name in actuator_strength_update_dict.keys():
		actuator.strength *= actuator_strength_update_dict[actuator.name]


# ALL THE CODE BELOW THIS LINE SHOULD BE REPLACED BY THE INSTANTIATION AND METHOD CALL OF CLASS: e.g. REPERTOIRECREATOR 
# Init a random key
random_key = jax.random.PRNGKey(seed)
 
# Init policy network
policy_layer_sizes = policy_hidden_layer_sizes + (env.action_size,)
policy_network = MLP(
    layer_sizes=policy_layer_sizes,
    kernel_init=jax.nn.initializers.lecun_uniform(),
    final_activation=jnp.tanh,
)

# Init population of controllers
random_key, subkey = jax.random.split(random_key)
keys = jax.random.split(subkey, num=batch_size)
fake_batch = jnp.zeros(shape=(batch_size, env.observation_size))
init_variables = jax.vmap(policy_network.init)(keys, fake_batch)


# Create the initial environment states
random_key, subkey = jax.random.split(random_key)
keys = jnp.repeat(jnp.expand_dims(subkey, axis=0), repeats=batch_size, axis=0)
reset_fn = jax.jit(jax.vmap(env.reset))
init_states = reset_fn(keys)

# Define the function to play a step with the policy in the environment
def play_step_fn(env_state, policy_params, random_key,):
    """
    Play an environment step and return the updated state and the transition.
    """

    actions = policy_network.apply(policy_params, env_state.obs)    
    state_desc = env_state.info["state_descriptor"]
    next_state = env.step(env_state, actions)

    transition = QDTransition(
        obs=env_state.obs,
        next_obs=next_state.obs,
        rewards=next_state.reward,
        dones=next_state.done,
        actions=actions,
        truncations=next_state.info["truncation"],
        state_desc=state_desc,
        next_state_desc=next_state.info["state_descriptor"],
    )

    return next_state, policy_params, random_key, transition

# Prepare the scoring function
bd_extraction_fn = environments.behavior_descriptor_extractor[env_name]
scoring_fn = functools.partial(
    scoring_function,
    init_states=init_states,
    episode_length=episode_length,
    play_step_fn=play_step_fn,
    behavior_descriptor_extractor=bd_extraction_fn,
)

# Get minimum reward value to make sure qd_score are positive
reward_offset = environments.reward_offset[env_name]

# Define a metrics function
metrics_function = functools.partial(
    default_qd_metrics,
    qd_offset=reward_offset * episode_length,
)

# Define emitter
variation_fn = functools.partial(
    isoline_variation, iso_sigma=iso_sigma, line_sigma=line_sigma
)

mixing_emitter = MixingEmitter(
    mutation_fn=None, 
    variation_fn=variation_fn, 
    variation_percentage=1.0, 
    batch_size=batch_size
)

# Instantiate MAP-Elites
map_elites = MAPElites(
    scoring_function=scoring_fn,
    emitter=mixing_emitter,
    metrics_function=metrics_function,
)

# Compute the centroids
# centroids, random_key = compute_cvt_centroids(
#     num_descriptors=env.behavior_descriptor_length,
#     num_init_cvt_samples=num_init_cvt_samples,
#     num_centroids=num_centroids,
#     minval=min_bd,
#     maxval=max_bd,
#     random_key=random_key,
# )

centroids = compute_euclidean_centroids(
    grid_shape=grid_shape,
    minval=min_bd,
    maxval=max_bd
)

# print(f"centroids: {centroids}")
# print(f"jnp.shape(centroids): {jnp.shape(centroids)}")

# Compute initial repertoire and emitter state
repertoire, emitter_state, random_key = map_elites.init(init_variables, centroids, random_key)

log_period = 10
num_loops = int(num_iterations / log_period)

csv_logger = CSVLogger(
    "./results/mapelites-logs.csv",
    header=["loop", "iteration", "qd_score", "max_fitness", "coverage", "time"]
)
all_metrics = {}

# main loop
for i in range(num_loops):
    start_time = time.time()
    # main iterations

    (repertoire, emitter_state, random_key,), metrics = jax.lax.scan(
        map_elites.scan_update,
        (repertoire, emitter_state, random_key),
        (),
        length=log_period,
    )
    timelapse = time.time() - start_time

    # log metrics
    logged_metrics = {"time": timelapse, "loop": 1+i, "iteration": 1 + i*log_period}
    for key, value in metrics.items():
        # take last value
        logged_metrics[key] = value[-1]

        # take all values
        if key in all_metrics.keys():
            all_metrics[key] = jnp.concatenate([all_metrics[key], value])
        else:
            all_metrics[key] = value

    csv_logger.log(logged_metrics)

# create the x-axis array
env_steps = jnp.arange(num_iterations) * episode_length * batch_size

# create the plots and the grid
fig, axes = plot_map_elites_results(env_steps=env_steps, metrics=all_metrics, repertoire=repertoire, min_bd=min_bd, max_bd=max_bd, grid_shape=grid_shape)
fig.savefig("example-plots-harmed")

# # create the plots and grid for the multidimensional map-elites
# fig, axes = plot_multidimensional_map_elites_grid(
#     repertoire=repertoire,
#     maxval=jnp.asarray([max_bd]),
#     minval=jnp.asarray([min_bd]),
#     grid_shape=grid_shape
# )
# fig.savefig("example-multidim-plots")

repertoire_path = "./last_repertoire/"
os.makedirs(repertoire_path, exist_ok=True)
repertoire.save(path=repertoire_path)

# print(repertoire.centroids)
# print(repertoire.descriptors)
# print(repertoire.fitnesses)
# print(repertoire.genotypes)

# Init population of policies
random_key, subkey = jax.random.split(random_key)
fake_batch = jnp.zeros(shape=(env.observation_size,))
fake_params = policy_network.init(subkey, fake_batch)

_, reconstruction_fn = ravel_pytree(fake_params)


repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=repertoire_path)

best_idx = jnp.argmax(repertoire.fitnesses)
best_fitness = jnp.max(repertoire.fitnesses)
best_bd = repertoire.descriptors[best_idx]

print(
    f"Best fitness in the repertoire: {best_fitness:.2f}\n",
    f"Behavior descriptor of the best individual in the repertoire: {best_bd}\n",
    f"Index in the repertoire of this individual: {best_idx}\n"
)


my_params = jax.tree_util.tree_map(
    lambda x: x[best_idx],
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
    # print(f"action: {action}")
    state = jit_env_step(state, action)

print(f"The trajectory of this individual contains {len(rollout)} transitions.")

# HTML(html.render(env.sys, [s.qp for s in rollout[:500]]))

html.save_html("harmed_robot.html", env.sys, [s.qp for s in rollout[:500]])

print(env.sys.config.actuators)

		