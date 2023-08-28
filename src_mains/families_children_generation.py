from src.utils.all_imports import *
from src.utils.hexapod_damage_dicts import shortform_damage_list
from src.core.children_generator import ChildrenGenerator


# Load/define all the hyperparameters for this run (children/damage_dict to test?, algorithms to test children on?, seed identification? fam_id, ancestors_id?, path_to_results?)
parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str, required=False, default="./trial_folder")
parser.add_argument("--job_index", type=int, required=False, default=1)

args = parser.parse_args()
save_dir = args.save_dir
seed = args.job_index

# Change these
children_damage_combinations = [(1,), (3, 4), (1, 2, 3)] # Careful, tuples
algorithms_to_test = ["ITE", "GPCF", "GPCF-1trust", "GPCF-reg", "inHERA"]
path_to_families = "numiter40k_final_families" # Careful, must be same as below
from numiter40k_final_families import family_task # Careful, must be same as above
verbose = True
path_to_family = f"{path_to_families}/family-seed_{seed}_last_repertoire" # Careful, individual families are not always stored with the same name

# Defined automatically
norm_params = jnp.load(f"{path_to_family}/norm_params.npy")
task = family_task.task 
num_iter = family_task.algo_num_iter
children_in_ancestors = family_task.children_in_ancestors

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
children_generator.generate_custom_children(combinations=[()]) # Generate the intact child