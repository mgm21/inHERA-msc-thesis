from all_imports import *
from gaussian_process import GaussianProcess

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
# print(best_N_bds)


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

# html.save_html("test-loaded.html", env.sys, [s.qp for s in rollout[:500]])


# UNDERSTANDING HOW TO SCORE GENOTYPES

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

bd_extraction_fn = environments.behavior_descriptor_extractor[env_name]

scoring_fn = functools.partial(
    scoring_function,
    init_states=init_states,
    episode_length=episode_length,
    play_step_fn=play_step_fn,
    behavior_descriptor_extractor=bd_extraction_fn,
)

# genotypes_to_test = jnp.array([my_params])

# # scores the offsprings
# fitnesses, descriptors, extra_scores, random_key = scoring_fn(
#     my_params, random_key
# )

# print(fitnesses)
# print(descriptors)
# print(extra_scores)
# print(random_key)



#### Replicating implementation from qd-skill-discovery-benchmark GitHub in hopes that it will work for scoring and will shed some light

random_key = jax.random.PRNGKey(0)

eval_env = environments.create(
   env_name=env_name,
   batch_size=1,
   episode_length=episode_length,
   auto_reset=True,
   eval_metrics=True,
)

# Init policy network
policy_layer_sizes = tuple(policy_hidden_layer_sizes + (env.action_size,))

policy_network = MLP(
    layer_sizes=policy_layer_sizes,
    kernel_init=jax.nn.initializers.lecun_uniform(),
    final_activation=jnp.tanh,
)

fake_batch = jnp.zeros(eval_env.observation_size)

random_key, subkey = jax.random.split(random_key)

init_variables = policy_network.init(subkey, fake_batch)

flat, recons_fn = jax.flatten_util.ravel_pytree(init_variables)

bd_extraction_fn = environments.behavior_descriptor_extractor[env_name]

random_key, subkey = jax.random.split(random_key)

scoring_fn, random_key = create_brax_scoring_fn(
        env=eval_env,
        policy_network=policy_network,
        bd_extraction_fn=bd_extraction_fn,
        random_key=subkey,
        episode_length=episode_length,
        deterministic=True,
    )

scoring_fn = jax.jit(scoring_fn)


genotypes = jnp.load(os.path.join(repertoire_path, "genotypes.npy"))
fitnesses = jnp.load(os.path.join(repertoire_path, "fitnesses.npy"))
descriptors = jnp.load(os.path.join(repertoire_path, "descriptors.npy"))

# UNCOMMENT ALL THE FOLLOWING LINES TO TEST/DRAW FROM REAL SIMULATOR FOR CERTAIN POLICIES

# # To score all the policies that have been populated:
# policies = jax.tree_map(
#         lambda x: x[fitnesses != -jnp.inf], jax.vmap(recons_fn)(genotypes)
#     )

# # To score the best policy that was found in the repertoire
# best_repertoire_policy = jax.tree_map(
#         lambda x: x[fitnesses == jnp.max(fitnesses)], jax.vmap(recons_fn)(genotypes)
#     )

# # To score a policy chosen at a random repertoire
# random_key, subkey = jax.random.split(random_key)
# random_policy = jax.tree_map(
#         lambda x: x[fitnesses == jnp.array(jax.random.choice(subkey, fitnesses[fitnesses != -jnp.inf]))
# ], jax.vmap(recons_fn)(genotypes)
#     )

# # Evaluate a chosen policy
# chosen_policy = random_policy # Change this line!
# eval_fitnesses, descriptors, extra_scores, random_key = scoring_fn(
#             chosen_policy, random_key
#         )

print("keep going!")

# TODO: check why descriptors are not unique. Is it on these that I should be doing all the operations?

# EVERYTHING UNDERNEATH HAS TO DO WITH GPs

# Initialise the gp
gp = GaussianProcess()

# Observe the best policy that had been seen in the repertoire
best_repertoire_policy = jax.tree_map(
        lambda x: x[fitnesses == jnp.max(fitnesses)], jax.vmap(recons_fn)(genotypes)
    )
observed_fitness, observed_descriptor, extra_scores, random_key = scoring_fn(
            best_repertoire_policy, random_key
        )

# Set y_observed to best_fitness (obtained from scoring_fn)
y_observed = observed_fitness

# Set x_observed to best_descriptor (obtained from scoring_fn)
x_observed = observed_descriptor

# Set x_test to descriptors (this is the one loaded from .npy file)
x_test = descriptors

# Run gp.train and then check whether the new mu and var have updated as expected
mu, var = gp.train(x_observed=x_observed,
          y_observed=y_observed,
          x_test=x_test)

print("end")









