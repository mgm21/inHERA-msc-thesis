from src.utils.all_imports import *

class ChildrenLoader:
    all_algo_names = ["ITE", "GPCF", "GPCF-1trust", "GPCF-reg", "inHERA"]

    def load_children(self, path_to_family="families/family_4", algos=["ITE", "GPCF"]):
        path_to_children = f"{path_to_family}/children"

        res_dict = {}
        for algo in algos:
            res_dict[algo] = []

        for _, dirs, _ in os.walk(path_to_children):
            for dirname in dirs:
                if dirname not in self.all_algo_names:
                    for algo in algos:
                        algo_path = f"{path_to_children}/{dirname}/{algo}"
                        y_observed = jnp.load(f"{algo_path}/y_observed.npy")
                        res_dict[algo] += [y_observed]
        
        for algo in algos:
            res_dict[algo] = jnp.array(res_dict[algo])
        
        return res_dict
    
    def load_children_by_num_broken_legs(self, path_to_family="families/family_4", algos=["ITE", "GPCF"],):
        path_to_children = f"{path_to_family}/children"

        res_dict = {}
        for algo in algos:
            res_dict[algo] = {}
        
        for _, dirs, _ in os.walk(path_to_children):
            for dirname in dirs:
                if dirname not in self.all_algo_names:
                    for algo in algos:
                        # Figure out what type of damaged has occured
                        num_legs_damaged = len(dirname.split("_")) - 1

                        algo_path = f"{path_to_children}/{dirname}/{algo}"
                        y_observed = jnp.load(f"{algo_path}/y_observed.npy")

                        res_dict[algo][num_legs_damaged] = res_dict[algo].setdefault(num_legs_damaged, []) + [y_observed]
        
        for algo in algos:
            for key in res_dict[algo].keys():
                res_dict[algo][key] = jnp.array(res_dict[algo][key])

        return res_dict
        

if __name__ == "__main__":
    algos = ["ITE", "GPCF"]
    path_to_family = "families/family_6"

    children_loader = ChildrenLoader()
    res_dict = children_loader.load_children_by_num_broken_legs(path_to_family=path_to_family, algos=algos)