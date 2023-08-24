from src.utils.all_imports import *
from src.core.algorithm_comparator import AlgorithmComparator
from src.core.ancestors_generator import AncestorsGenerator
from src.core.children_generator import ChildrenGenerator
from src.loaders.children_loader import ChildrenLoader
from src.utils.hexapod_damage_dicts import shortform_damage_list, intact
from src.utils.visualiser import Visualiser

families_path = "./dummy_families"
from dummy_families import family_task

# Define the hyperparameters/objects common to the whole family (e.g. task-related parameters)
task = family_task.task
algo_num_iter = family_task.algo_num_iter
children_in_ancestors = family_task.children_in_ancestors
ancest_num_legs_damaged = (1, 2, 3, 4, 5) # Must be a tuple e.g. (2,)
children_num_legs_damaged = (1, 2, 3, 4, 5) # Must be a tuple e.g. (1, 2, 3, 4,)
algorithms_to_test = ["ITE", "GPCF", "GPCF-1trust", "GPCF-reg", "inHERA"] # "ITE", "GPCF", "GPCF-1trust", "GPCF-reg", "inHERA"
algorithms_to_plot = ["ITE", "GPCF", "GPCF-1trust", "GPCF-reg", "inHERA"] # "ITE", "GPCF", "GPCF-1trust", "GPCF-reg", "inHERA"
verbose = True
ite_alpha = 0.9
gpcf_kappa = 0.05

# For each family in families_path
for file_name in os.listdir(families_path):
    d = os.path.join(families_path, file_name)

    # For every repertoire in the repertoires folder
    if os.path.isdir(d) and file_name not in ["__pycache__",]:
        # Define the family specific hyperparameters (e.g. path)
        path_to_family = f"{families_path}/{file_name}"
        norm_params = jnp.load(f"{path_to_family}/norm_params.npy")

        # Define an algorithm comparator with families and family specific hyperparameters
        algo_comp = AlgorithmComparator(algorithms_to_test=algorithms_to_test,
                path_to_family=path_to_family,
                task=task,
                norm_params=norm_params,
                algo_num_iter=algo_num_iter,
                ancest_num_legs_damaged=ancest_num_legs_damaged,
                children_num_legs_damaged=children_num_legs_damaged,
                verbose=verbose,
                ite_alpha=ite_alpha,
                gpcf_kappa=gpcf_kappa,
                algorithms_to_plot=algorithms_to_plot,
                children_in_ancestors=children_in_ancestors)
        
        # Generate the ancestors for this family
        algo_comp.generate_ancestors()

        del algo_comp
        gc.collect()
