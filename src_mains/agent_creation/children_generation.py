from src.utils.all_imports import *
from src.core.family_setup.children_generator import ChildrenGenerator


# Load/define all the hyperparameters for this run (children/damage_dict to test?, algorithms to test children on?, seed identification? fam_id, ancestors_id?, path_to_results?)
parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str, required=False, default="./trial_setting")
parser.add_argument("--job_index", type=int, required=False, default=1)
parser.add_argument("--algorithm", type=str, required=False, default="GPCF")


args = parser.parse_args()
save_dir = args.save_dir
seed = args.job_index
algorithm = args.algorithm

if algorithm == "GPCF":
    algorithms_to_test = ["GPCF", "GPCF-reg", "GPCF-1trust"]
elif algorithm == "inHERA":
    algorithms_to_test = ["inHERA", "inHERA-b0"]
elif algorithm == "ITE":
    algorithms_to_test = ["ITE", "inHERA-expert", "inHERA-b0-expert"]
elif algorithm == "experts":
    algorithms_to_test = ["inHERA-expert", "inHERA-b0-expert"]
else:
    algorithms_to_test = [algorithm]

print(algorithms_to_test)

# Change these
children_damage_combinations = [(1,), (3, 4), (1, 2, 3)] # Careful, tuples
 # "ITE", "GPCF", "GPCF-1trust", "GPCF-reg", "inHERA", "inHERA-b0", "inHERA-expert", "inHERA-b0-expert"
path_to_families = "final_families" # Careful, must be same as below
from final_families import family_task # Careful, must be same as above
verbose = True
path_to_family = f"{path_to_families}/family-seed_{seed}_last_repertoire" # Careful, individual families are not always stored with the same name

# Defined automatically
norm_params = jnp.load(f"{path_to_family}/norm_params.npy")
task = family_task.task 
num_iter = family_task.algo_num_iter
children_in_ancestors = False # TODO change if children in ancestors needed 

# Make a directory for the seed (mimic how families are named in the families folder)
path_to_results = f"{save_dir}/children/family-seed_{seed}_repertoire"
os.makedirs(name=path_to_results, exist_ok=True)x

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

