from src.utils.all_imports import *
from src.core.algorithm_comparator import AlgorithmComparator
from src.core.ancestors_generator import AncestorsGenerator
from src.core.children_generator import ChildrenGenerator
from src.loaders.children_loader import ChildrenLoader
from src.utils.hexapod_damage_dicts import shortform_damage_list, intact
from src.utils.visualiser import Visualiser

families_path = "./final_families"
from final_families import family_task

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str, required=False, default="dummy_ancestors")

args = parser.parse_args()
save_dir = args.save_dir

path_to_result = save_dir

# Define the hyperparameters/objects common to the whole family (e.g. task-related parameters)
task = family_task.task
algo_num_iter = family_task.algo_num_iter
children_in_ancestors = family_task.children_in_ancestors
ancest_num_legs_damaged = (1,) # TODO: to test the time it takes to generate all the 1-leg damages
verbose = True
ite_alpha = 0.9
gpcf_kappa = 0.05

# For each family in families_path
for file_name in os.listdir(families_path):
    d = os.path.join(families_path, file_name)

    # For every repertoire in the repertoires folder
    if os.path.isdir(d) and file_name not in ["__pycache__",]:
        # Define the family specific hyperparameters (path and norm_params)
        path_to_family = f"{families_path}/{file_name}"
        norm_params = jnp.load(f"{path_to_family}/norm_params.npy")

        # Define an AncestorsGenerator
        ancest_gen = AncestorsGenerator(path_to_family=path_to_family,
                                        task=task,
                                        norm_params=norm_params,
                                        verbose=verbose)
        
        ancest_gen.path_to_ancestors = f"{save_dir}/{file_name}/ancestors"
        
        # Generate the ancestors for this family
        for i in range(len(ancest_num_legs_damaged)):
            ancest_gen.generate_auto_ancestors(num_broken_limbs=ancest_num_legs_damaged[i])

        del ancest_gen
        gc.collect()
