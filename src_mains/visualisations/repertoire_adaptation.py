from src.utils.all_imports import *
from src.core.family_setup.children_generator import ChildrenGenerator


save_dir = "./trial_setting"
seed = 20
algorithms_to_test = ["ITE"]

# Change these
children_damage_combinations = [(1,),] # Careful, tuples
path_to_families = "final_families" # Careful, must be same as below
from final_families import family_task # Careful, must be same as above
verbose = True
path_to_family = f"{path_to_families}/family-seed_{seed}_last_repertoire" # Careful, individual families are not always stored with the same name

# Defined automatically
norm_params = jnp.load(f"{path_to_family}/norm_params.npy")
task = family_task.task 
num_iter = family_task.algo_num_iter
children_in_ancestors = True # TODO change if children in ancestors needed 

# Make a directory for the seed (mimic how families are named in the families folder)
path_to_results = f"{save_dir}/children/family-seed_{seed}_repertoire"
os.makedirs(name=path_to_results, exist_ok=True)

children_generator = ChildrenGenerator(algorithms_to_test=algorithms_to_test,
                                       path_to_family=path_to_family, # Not path to families!
                                       task=task,
                                       ite_alpha=0.9,
                                       ite_num_iter=num_iter,
                                       verbose=verbose,
                                       norm_params=norm_params,
                                       gpcf_kappa=0.05,
                                       children_in_ancestors=children_in_ancestors)

children_generator.path_to_children = path_to_results
children_generator.generate_custom_children(combinations=children_damage_combinations) # Generate the chosen children
# children_generator.generate_custom_children(combinations=[()]) # Generate the intact child