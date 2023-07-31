from src.utils.all_imports import *

class AncestorsLoader:
    def load_ancestors(self, path_to_ancestors="results/family_0/ancestors"):
        ancestor_mus = []
        ancestor_vars = []
        ancestor_names = []

        for subdir, _, _ in os.walk(path_to_ancestors):

            # For all subdirs apart from the outer dir
            if subdir != path_to_ancestors:

                # Load arrays
                mu = jnp.load(os.path.join(subdir, "mu.npy"))
                var = jnp.load(os.path.join(subdir, "var.npy"))
                name = subdir.split(f"{path_to_ancestors}/")[-1]

                # Append arrays
                ancestor_mus += [mu]
                ancestor_vars += [var]
                ancestor_names += [name]
        
        return jnp.array(ancestor_mus), jnp.array(ancestor_vars), ancestor_names
        

if __name__ == "__main__":
    anc_load = AncestorsLoader()
    mus, vars, names = anc_load.load_ancestors()
    print(mus.shape)
    print(vars.shape)
    print(names)
    