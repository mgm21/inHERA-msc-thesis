from src.utils.all_imports import *

class AncestorsLoader:
    def load_ancestors(self, path_to_ancestors="families/family_0/ancestors"):
        ancestor_mus = []
        ancestor_vars = []
        ancestor_names = []
        damage_dicts = []

        for subdir, _, _ in os.walk(path_to_ancestors):

            # For all subdirs apart from the outer dir
            if subdir != path_to_ancestors:

                # Load arrays
                mu = jnp.load(os.path.join(subdir, "mu.npy"))
                var = jnp.load(os.path.join(subdir, "var.npy"))
                name = subdir.split(f"{path_to_ancestors}/")[-1]

                # Load the damage dictionary
                with open(os.path.join(subdir, 'damage_dict.txt'), "r") as fp:
                    damage_dict = json.load(fp)

                # Append arrays
                ancestor_mus += [mu]
                ancestor_vars += [var]
                ancestor_names += [name]
                damage_dicts += [damage_dict]
        
        return jnp.array(ancestor_mus), jnp.array(ancestor_vars), ancestor_names, damage_dicts
        

if __name__ == "__main__":
    anc_load = AncestorsLoader()
    mus, vars, names, damage_dicts = anc_load.load_ancestors()
    print(mus.shape)
    print(vars.shape)
    print(names)
    print(damage_dicts)
    