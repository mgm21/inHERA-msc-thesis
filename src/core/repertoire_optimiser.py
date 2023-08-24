from src.utils.all_imports import *
from src.core.task import Task

class RepertoireOptimiser:
    def __init__(self,
                 task=Task(),
                 env=None):
        
        # Data coming from class dataclass
        self.batch_size = task.batch_size
        self.env_name = task.env_name
        self.episode_length = task.episode_length
        self.num_iterations = task.num_iterations
        self.seed = task.seed
        self.policy_hidden_layer_sizes = task.policy_hidden_layer_sizes
        self.iso_sigma = task.iso_sigma
        self.line_sigma = task.line_sigma
        self.min_bd = task.min_bd
        self.max_bd = task.max_bd
        self.grid_shape = task.grid_shape

        # If a custom env has not been passed, create a standard intact environment
        self.env = env
        if self.env == None:
            self.env = environments.create(task.env_name, episode_length=task.episode_length)

    def optimise_repertoire(self,
                            repertoire_path="./families/class_example_repertoire/",
                            csv_results_path="./families/class_mapelites_logs.csv",
                            plot_path="./families/class_example_plots",
                            html_path="./families/class_best_policy_in_map.html"):
        
        # Temporarily not to change the below implementation
        batch_size = self.batch_size
        env_name = self.env_name 
        episode_length = self.episode_length
        num_iterations = self.num_iterations
        seed = self.seed
        policy_hidden_layer_sizes = self.policy_hidden_layer_sizes
        iso_sigma = self.iso_sigma
        line_sigma = self.line_sigma
        min_bd = self.min_bd
        max_bd = self.max_bd
        grid_shape = self.grid_shape
        env = self.env

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

        centroids = compute_euclidean_centroids(
            grid_shape=grid_shape,
            minval=min_bd,
            maxval=max_bd
        )

        repertoire, emitter_state, random_key = map_elites.init(init_variables, centroids, random_key)

        log_period = 10
        num_loops = int(num_iterations / log_period)

        csv_logger = CSVLogger(
            csv_results_path,
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
        fig.savefig(plot_path)

        print(f"this is the plot_path: {plot_path}")
        print(f"this is the repertoire path: {repertoire_path}")

        os.makedirs(repertoire_path, exist_ok=True)
        repertoire.save(path=repertoire_path)

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
            state = jit_env_step(state, action)

        print(f"The trajectory of this individual contains {len(rollout)} transitions.")

        html.save_html(html_path, env.sys, [s.qp for s in rollout[:500]])


        
if __name__ == "__main__":
    from src.core.adaptive_agent import AdaptiveAgent
    from src.core.task import Task
    import src.utils.hexapod_damage_dicts as hexapod_damage_dicts
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=False, default="repertoires")

    args = parser.parse_args()
    save_dir = args.save_dir

    path_to_result = save_dir
    print(f"the path to result from .py source file perspective is {path_to_result}")

    for SEED in range(0, 1):
        task = Task(batch_size=256,
                    env_name="hexapod_uni",
                    episode_length=10, 
                    num_iterations=10,
                    seed=SEED,
                    policy_hidden_layer_sizes=(64, 64),
                    iso_sigma=0.005,
                    line_sigma=0.05,
                    min_bd=0.,
                    max_bd=1.,
                    grid_shape=tuple([3])*6,)
        
        repertoire_opt = RepertoireOptimiser(task=task,)

        start_time = time.time()

        print(f"these are all the paths that will be used: {path_to_result}/seed_{SEED}_result_plots \n {path_to_result}/seed_{SEED}_best_policy.html \n {path_to_result}/seed_{SEED}_repertoire/ \n {path_to_result}/seed_{SEED}_mapelites_log.csv")
        
        repertoire_opt.optimise_repertoire(plot_path=f"{path_to_result}/seed_{SEED}_result_plots",
                                        html_path=f"{path_to_result}/seed_{SEED}_best_policy.html",
                                        repertoire_path=f"{path_to_result}/seed_{SEED}_repertoire/",
                                        csv_results_path=f"{path_to_result}/seed_{SEED}_mapelites_log.csv")
        
        end_time = time.time()
        print(f"Execution time (s): {end_time - start_time}")