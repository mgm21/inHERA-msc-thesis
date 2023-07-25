# Note: could change Task name, just gave it because Environment would lead to a confusion with brax/qdax envs
# TODO: To test if Task works, replace all the variables in repertoire_optimiser by the ones defined in a Task object
# See if you get the same results.

# TODO: maybe include everything in the init if the method is always called after the init?

from all_imports import *

class Task:
    def __init__(self,
                 batch_size=128,
                 env_name='hexapod_uni',
                 episode_length=100,
                 num_iterations=100,
                 seed=42,
                 policy_hidden_layer_sizes=(64, 64),
                 iso_sigma=0.005,
                 line_sigma=0.05,
                 min_bd=0.,
                 max_bd=1.,
                 grid_shape=tuple([3]) * 6,
                 env=None):
                
        self.batch_size = batch_size
        self.env_name = env_name
        self.episode_length = episode_length
        self.num_iterations = num_iterations
        self.seed = seed
        self.policy_hidden_layer_sizes = policy_hidden_layer_sizes
        self.iso_sigma = iso_sigma
        self.line_sigma = line_sigma
        self.min_bd = min_bd
        self.max_bd = max_bd
        self.grid_shape = grid_shape
        self.env = env

        # If a custom env has not been passed
        if self.env == None:
            self.env = environments.create(
                env_name=env_name,
                batch_size=1,
                episode_length=episode_length,
                auto_reset=True,
                eval_metrics=True,
                )

    def define_task_functions(self):
        random_key = jax.random.PRNGKey(self.seed)

        # Define the policy network (# TODO: is it okay just like this? Check its uses)
        policy_layer_sizes = tuple(self.policy_hidden_layer_sizes + (self.env.action_size,))
        policy_network = MLP(
            layer_sizes=policy_layer_sizes,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            final_activation=jnp.tanh,
        )

        # Define the reconstruction function (# TODO: is it okay just like this? Check its uses)
        fake_batch = jnp.zeros(self.env.observation_size)
        random_key, subkey = jax.random.split(random_key)
        init_variables = policy_network.init(subkey, fake_batch)
        _, recons_fn = jax.flatten_util.ravel_pytree(init_variables)

        # Define the bd_extraction function (# TODO: is it okay just like this? Check its uses)
        bd_extraction_fn = environments.behavior_descriptor_extractor[self.env_name]
        random_key, subkey = jax.random.split(random_key)
        scoring_fn, random_key = create_brax_scoring_fn(
                env=self.env,
                policy_network=policy_network,
                bd_extraction_fn=bd_extraction_fn,
                random_key=subkey,
                episode_length=self.episode_length,
                deterministic=True,
            )
        scoring_fn = jax.jit(scoring_fn)

        # Define the attributes
        self.policy_network = policy_network
        self.recons_fn = recons_fn
        self.scoring_fn = scoring_fn


if __name__ == "__main__":
    task = Task()
    task.define_task_functions()