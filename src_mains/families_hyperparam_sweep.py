from src.utils.all_imports import *
from src.utils.hexapod_damage_dicts import shortform_damage_list


# Load/define all the hyperparameters for this run (different hyperparameter settings?, children/damage_dict to test?, algorithms to test children on?, seed identification? fam_id, ancestors_id?, path_to_results?)
parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str, required=False, default="dummy_ancestors")
parser.add_argument("--seed", type=int, required=False, default=0)

args = parser.parse_args()
save_dir = args.save_dir
seed = args.seed

# Tuples!
children_damage_combinations = [(1,), (2,), (3,), (4,), (5,)]

# Define the hyperparameter range(s)
l1_regularisation_weight_list = [1, 0.1, 0.01,]
invmax_regularisation_weight_list = [1, 0.1, 0.01,]
uncertainty_regularisation_weight_list = [1, 0.1, 0.01]

# For all the hyperparameter settings that you would like to test the children in (add nests of for loops to test new hyper parameters)
# Add a for loop for all the parameters that you would like to test
for l1 in l1_regularisation_weight_list:
    for v in invmax_regularisation_weight_list:
        for u in uncertainty_regularisation_weight_list:

            # Make a directory for this hyperparameter (name it with the values of the hyperparameters)
            dir_name = []

            os.makedirs(name=f"{save_dir}", exist_ok=True)
            # Make a directory for the seed (mimic how families are named in the families folder)

            # For every agent to test

                # For every algorithm to test it on

                    # Instantiate the agent

                    # Instantiate the GP

                    # Instantiate the adaptation algorithm

                    # Run the adaptation algorithm and save the results in the save_dir/hyperparam_m/family-seed_n/children/agent/algo/y_observed.npy