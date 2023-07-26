from all_imports import *

# Temporarily
from qdax.environments import hexapod

class AdaptiveAgent:
    def __init__(self,
                 task,
                 sim_repertoire_arrays=(jnp.array([]),
                                        jnp.array([]),
                                        jnp.array([]),
                                        jnp.array([])),
                 damage_dictionary={},
                 name="some_name",
                 x_observed=jnp.array([]),
                 y_observed=jnp.array([]),
                 mu=None,
                 var=None,
                 sim_noise=0.001):
        
        self.task = task
        self.sim_descriptors = sim_repertoire_arrays[1]
        self.sim_fitnesses = sim_repertoire_arrays[2]
        self.sim_genotypes = sim_repertoire_arrays[3]

        self.name = name
        self.x_observed = x_observed
        self.y_observed = y_observed

        if mu == None:
            self.mu = self.sim_fitnesses
        else:
            self.mu = mu
        
        if var == None:
            self.var = jnp.array([sim_noise] * len(self.sim_fitnesses))
        else:
            self.var = var

        self.damage_dictionary = damage_dictionary
        
        # Damage the robot
        self.damage_env()

        # TODO: Get the specific functions for this agent (only the scoring fn may be needed to be specific 
        # can change later)
        self.define_task_functions()

    
    # TODO: this fitness as an input should be changed to descriptor as described in example_main's TODO no. 2
    def test_descriptor(self, index, random_key):
        # TODO: isn't it the opposite random key and subkey (also why are we passing random key to the function below)
        random_key, subkey = jax.random.split(random_key)

        # Get the policy at index
        policy = jax.vmap(self.recons_fn)(self.sim_genotypes[index:index+1])

        # This way also works to keep correct shapes for the scoring function
        # policy = jax.vmap(self.recons_fn)(jnp.array([self.sim_genotypes[index]]))
        
        # Observe a point
        observed_fitness, observed_descriptor, extra_scores, random_key = self.scoring_fn(
                    policy, random_key
                )
        
        return observed_fitness, observed_descriptor, extra_scores, random_key

    def damage_env(self):
        # This could easily be changed for any other legged robot by changing hexapod -> ant below 
        intact_CONFIG = qdax.environments.hexapod._SYSTEM_CONFIG
        damaged_CONFIG = qdax.environments.hexapod._SYSTEM_CONFIG

        for damaged_joint_name in self.damage_dictionary.keys():
            # Note the 2 spaces after \n are important to mimic the original form of the CONFIG
            damaged_CONFIG = damaged_CONFIG.replace(f'joint: "{damaged_joint_name}"\n  strength: 200.0',
                                                    f'joint: "{damaged_joint_name}"\n  strength: {self.damage_dictionary[damaged_joint_name]}')
        
        # Change the qdax config
        qdax.environments.hexapod._SYSTEM_CONFIG = damaged_CONFIG

        self.env = environments.create(env_name=self.task.env_name,
                                       episode_length=self.task.episode_length)
        
        # # This was causing an error when running the map-elites optimisation
        # This seemed to be required to query the agent but for now will only keep the above because works for optimisatoin
        # self.env = environments.create(env_name=self.task.env_name,
        #                                batch_size=1,
        #                                episode_length=self.task.episode_length,
        #                                auto_reset=True,
        #                                eval_metrics=True,)
        
        # Revert the qdax config
        qdax.environments.hexapod._SYSTEM_CONFIG = intact_CONFIG
 
    def define_task_functions(self):
        # TODO: only the scoring function has to be redefined can fix that by putting these in the task.py later
        # Fix this seed
        random_key = jax.random.PRNGKey(1)

        # Define the policy network (# TODO: is it okay just like this? Check its uses)
        policy_layer_sizes = tuple(self.task.policy_hidden_layer_sizes + (self.env.action_size,))
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
        bd_extraction_fn = environments.behavior_descriptor_extractor[self.task.env_name]
        random_key, subkey = jax.random.split(random_key)
        scoring_fn, random_key = create_brax_scoring_fn(
                env=self.env,
                policy_network=policy_network,
                bd_extraction_fn=bd_extraction_fn,
                random_key=subkey,
                episode_length=self.task.episode_length,
                deterministic=True,
            )
        scoring_fn = jax.jit(scoring_fn)

        self.scoring_fn, self.recons_fn = scoring_fn, recons_fn

if __name__ == "__main__":
    from task import Task
    from repertoire_loader import RepertoireLoader
    import hexapod_damage_dicts

    # These lines do not change and cannot be the problem
    repertoire_loader = RepertoireLoader()
    sim_repertoire_arrays = _, descriptors, sim_fitnesses, genotypes = repertoire_loader.load_repertoire(repertoire_path="./last_repertoire/")

    # Define the agent's damage
    # These lines have been checked and do work to set all the strengths to 0
    damage_dictionary = hexapod_damage_dicts.all_actuators_broken

    task = Task()

    # Define the agent
    agent = AdaptiveAgent(task=task,
                          sim_repertoire_arrays=sim_repertoire_arrays,
                          damage_dictionary=damage_dictionary,)
    
    # Define random key
    seed = 1
    random_key = jax.random.PRNGKey(seed)
    random_key, subkey = jax.random.split(random_key)

    # Test a behaviour in the damaged environment
    observed_fitness, observed_descriptor, extra_scores, random_key = agent.test_descriptor(index=jnp.argmax(sim_fitnesses), random_key=random_key)

    print(observed_fitness)
    print(observed_descriptor)