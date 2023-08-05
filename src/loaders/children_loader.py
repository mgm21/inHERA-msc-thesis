from src.utils.all_imports import *

class ChildrenLoader:
    def load_children(self, path_to_family="results/family_4", algos_to_extract=["ITE", "GPCF"]):
        """
        # Navigate to children in family
        # For each child
        # For each algo to test, locate the corresponding subdirectory
        # For each algo subdir, extract the y_observed.npy array
        # Store this array in an algo specific super-array

        # After the "for each child" loop:
        # For each algo
        # Get mean of y_observed for that algo (from super-array) AND STORE IN super-means
        # Get var of y_observed for that algo (from super-array) AND STORE IN super-vars

        # Call Visualiser.get_mean_and_var_plot(self, super_means, super_vars, algo_to_test, path_to_res=f"{path__to_family}"):
        """

        path_to_children = f"{path_to_family}/children"

        res_dict = {}
        for algo in algos_to_extract:
            res_dict[algo] = []

        for _, dirs, _ in os.walk(path_to_children):
            for dirname in dirs:
                if dirname not in algos_to_extract:
                    for algo in algos_to_extract:
                        algo_path = f"{path_to_children}/{dirname}/{algo}"
                        y_observed = jnp.load(f"{algo_path}/y_observed.npy")
                        res_dict[algo] += [y_observed]
        
        for algo in algos_to_extract:
            res_dict[algo] = jnp.array(res_dict[algo])
        
        return res_dict
        

if __name__ == "__main__":
    algos_to_extract = ["ITE", "GPCF"]
    path_to_family = "results/family_4"

    children_loader = ChildrenLoader()
    res_dict = children_loader.load_children(path_to_family=path_to_family, algos_to_extract=algos_to_extract)

    means = []
    vars = []

    print(res_dict)

    for algo in algos_to_extract:
        means += [jnp.nanmean(res_dict[algo], axis=0)]
        vars += [jnp.nanvar(res_dict[algo], axis=0)]
    
    # print(means)
