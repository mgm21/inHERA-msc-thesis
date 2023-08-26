from src.utils.all_imports import *
from src.utils.hexapod_damage_dicts import shortform_damage_list
from src.core.children_generator import ChildrenGenerator


# Load/define all the hyperparameters for this run (children/damage_dict to test?, algorithms to test children on?, seed identification? fam_id, ancestors_id?, path_to_results?)
parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str, required=False, default="./")
parser.add_argument("--seed", type=int, required=False, default=0)

args = parser.parse_args()
save_dir = args.save_dir
seed = args.seed

# Change these
children_damage_combinations = [(1,), (2,),] # Careful, tuples
algorithms_to_test = ["ITE", "GPCF", "GPCF-1trust", "GPCF-reg", "inHERA"]
path_to_families = "dummy_families"
from dummy_families import family_task
verbose = True

# Defined automatically
path_to_family = f"{path_to_families}/family-seed_{seed}_repertoire"
norm_params = jnp.load(f"{path_to_family}/norm_params.npy")
task = family_task.task
num_iter = family_task.algo_num_iter
children_in_ancestors = family_task.children_in_ancestors

# Make a directory for the seed (mimic how families are named in the families folder)
path_to_results = f"{save_dir}/children/family-seed_{seed}_repertoire"
os.makedirs(name=path_to_results, exist_ok=True)

children_generator = ChildrenGenerator(algorithms_to_test=algorithms_to_test,
                                       path_to_family=path_to_family, # Not task to families!
                                       task=task,
                                       ite_alpha=0.9,
                                       ite_num_iter=num_iter,
                                       verbose=verbose,
                                       norm_params=norm_params,
                                       gpcf_kappa=0.05,
                                       children_in_ancestors=children_in_ancestors)

children_generator.path_to_children = path_to_results
children_generator.generate_custom_children(combinations=children_damage_combinations)