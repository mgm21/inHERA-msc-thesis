from src.utils.all_imports import *
from src.loaders.repertoire_loader import RepertoireLoader
from src.loaders.ancestors_loader import AncestorsLoader

class Family:
    def __init__(self, path_to_family="families/family_0"):
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
    # TODO: copies may not be more necessary than just setting the arrays equal (I think jax does a copy in the background anyway)
    def _save_copy_of_init_ancest_arrs(self,):
        self.init_ancestor_mus = jnp.copy(self.ancestor_mus)
        self.init_ancestor_vars = jnp.copy(self.ancestors_vars)
        self.init_ancestor_names = self.ancestors_names[:]
        self.init_damage_dicts = self.damage_dicts[:]

    # TODO: potentially useless method because of remove_ancestor_from_ancest_arrs
    def reset_the_ancestor_arrs(self,):
        # Delete the previous ones from memory to make sure they do not occupy too much memory
        del self.ancestor_mus
        del self.ancestors_vars
        del self.ancestors_names
        del self.damage_dicts
    
        self.ancestor_mus = jnp.copy(self.init_ancestor_mus)
        self.ancestors_vars = jnp.copy(self.init_ancestor_vars)
        self.ancestors_names = self.init_ancestor_names[:]
        self.damage_dicts = self.init_damage_dicts[:]
    
    def remove_ancestor_from_ancest_arrs(self, ancest_name):

        if ancest_name in self.init_ancestor_names:
            # Delete the previous ones from memory to make sure they do not occupy too much memory
            del self.ancestor_mus
            del self.ancestors_vars
            del self.ancestors_names
            del self.damage_dicts

            # Find index
            idx = self.init_ancestor_names.index(ancest_name)

            # Update the arrays
            self.ancestor_mus = jnp.delete(self.init_ancestor_mus, obj=idx, axis=0)
            self.ancestors_vars = jnp.delete(self.init_ancestor_vars, obj=idx, axis=0)
            self.ancestors_names = self.init_ancestor_names[:idx] + self.init_ancestor_names[idx+1:]
            self.damage_dicts = self.init_damage_dicts[:idx] + self.init_damage_dicts[idx+1:]

        else:
            print(f"Did not this ancest name: {ancest_name} in the list of ancestor names")


if __name__ == "__main__":
    fam = Family(path_to_family="families/family_4")
    print(fam.ancestors_names)
    print(fam.ancestor_mus)

    fam.remove_ancestor_from_ancest_arrs(ancest_name="damaged_0")
    print(fam.ancestor_mus)

    fam.remove_ancestor_from_ancest_arrs(ancest_name="damaged_5")
    print(fam.ancestor_mus)

    fam.remove_ancestor_from_ancest_arrs(ancest_name="")
    print(fam.ancestor_mus)
    print(fam.ancestors_names)

    fam.reset_the_ancestor_arrs()
    print(fam.ancestors_names)








