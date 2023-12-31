from src.utils.all_imports import *
from src.core.family_setup.children_generator import ChildrenGenerator


# TODO: could implement this script quite quickly by wrapping the families_children_generation script inside these nested for loops and making sure to store everything within a dedicated
#  hyperparameter directory whose name includes the values of the hyper parameters (whose ranges are not length 1, only include the ones that change)
now = time.time()

# Load/define all the hyperparameters for this run (different hyperparameter settings?, children/damage_dict to test?, algorithms to test children on?, seed identification? fam_id, ancestors_id?, path_to_results?)
parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str, required=False, default="lab_machine_kappa_sweep")
parser.add_argument("--job_index", type=int, required=False, default=1)
parser.add_argument("--algorithm", type=str, required=False, default="inHERA")
parser.add_argument("--hyperparameter", type=float, required=False, default=1)


args = parser.parse_args()
save_dir = args.save_dir
seed = args.job_index
algorithm = args.algorithm
hyperparameter = args.hyperparameter

if type(hyperparameter) != list:
    hyperparameter = [hyperparameter]

# Define the hyperparameter range(s)
kappa_regularisation_weight_list = hyperparameter

# Change these
children_damage_combinations = [(1,), (3, 4), (1, 2, 3)] # Careful, tuples
algorithms_to_test = [algorithm]
path_to_families = "final_families" # Careful, must be same as below
from final_families import family_task # Careful, must be same as above
verbose = True
path_to_family = f"{path_to_families}/family-seed_{seed}_last_repertoire" # Careful, individual families are not always stored with the same name

# Defined automatically
norm_params = jnp.load(f"{path_to_family}/norm_params.npy")
task = family_task.task 
num_iter = family_task.algo_num_iter
children_in_ancestors = family_task.children_in_ancestors

# Define the children_generator
children_generator = ChildrenGenerator(algorithms_to_test=algorithms_to_test,
                            path_to_family=path_to_family, # Not path to families!
                            task=task,
                            ite_alpha=0.9,
                            ite_num_iter=num_iter,
                            verbose=verbose,
                            norm_params=norm_params,
                            gpcf_kappa=0.05,
                            children_in_ancestors=children_in_ancestors)

# For all the hyperparameter settings that you would like to test the children in (add nests of for loops to test new hyper parameters)
# Add a for loop for all the parameters that you would like to test
l1 = 0.001
v = 0.0001
u = 0.0001
for kappa in kappa_regularisation_weight_list:
    print(f"Starting pass for kappa={kappa}")
    path_to_hyperparam_folder = f"{save_dir}/hyperparameter_sweep/kappa_{kappa}"
    path_to_results = f"{path_to_hyperparam_folder}/family-seed_{seed}_last_repertoire"
    os.makedirs(name=path_to_hyperparam_folder, exist_ok=True)
    os.makedirs(name=path_to_results, exist_ok=True)
    
    children_generator.path_to_children = path_to_results
    children_generator.generate_custom_children(combinations=children_damage_combinations, kappa=kappa, l1_regularisation_weight=l1, invmax_regularisation_weight=v, uncertainty_regularisation_weight=u) # Generate the chosen children
    children_generator.generate_custom_children(combinations=[()], kappa=kappa, l1_regularisation_weight=l1, invmax_regularisation_weight=v, uncertainty_regularisation_weight=u) # Generate the intact child


end = time.time()

print(f"Run took: {end} - {now} seconds")