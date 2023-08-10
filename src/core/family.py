from src.utils.all_imports import *
from src.loaders.repertoire_loader import RepertoireLoader
from src.loaders.ancestors_loader import AncestorsLoader

class Family:
    def __init__(self, path_to_family="results/family_0"):
        # Define the family's simulation repertoire
        rep_load = RepertoireLoader()
        rep_arrs = rep_load.load_repertoire(repertoire_path=f"{path_to_family}/repertoire")
        self.centroids, self.descriptors, self.sim_fitnesses, self.genotypes = rep_arrs

        # Define the ancestor arrays
        ancestors_loader = AncestorsLoader()
        self.ancestor_mus, self.ancestors_vars, self.ancestors_names, self.damage_dicts = ancestors_loader.load_ancestors(path_to_ancestors=f"{path_to_family}/ancestors")

        # Save a copy of the original ancestor arrays
        self._save_copy_of_init_ancest_arrs()

    # TODO: all of these may not be needed but just in case
    def _save_copy_of_init_ancest_arrs(self,):
        self.init_ancestor_mus = jnp.copy(self.ancestor_mus)
        self.init_ancestor_vars = jnp.copy(self.ancestors_vars)
        self.init_ancestor_names = jnp.copy(self.ancestors_names)
        self.init_damage_dicts = jnp.copy(self.damage_dicts)
    
    def reset_the_ancestor_arrs(self,):
        # Delete the previous ones from memory to make sure they do not occupy too much memory
        del self.ancestor_mus
        del self.ancestors_vars
        del self.ancestors_names
        del self.damage_dicts

        self.ancestor_mus = jnp.copy(self.init_ancestor_mus)
        self.ancestors_vars = jnp.copy(self.init_ancestor_vars)
        self.ancestors_names = jnp.copy(self.init_ancestor_names)
        self.damage_dicts = jnp.copy(self.init_damage_dicts)

if __name__ == "__main__":
    fam = Family(path_to_family="results/family_4_1")
    print(fam.ancestors_names)
    print(fam.ancestor_mus.shape)
    print(fam.ancestor_mus[:, 600])

