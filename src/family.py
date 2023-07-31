from src.utils.all_imports import *
from src.repertoire_loader import RepertoireLoader
from src.ancestors_loader import AncestorsLoader

class Family:
    def __init__(self, path_to_family="results/family_0"):
        # Define the family's simulation repertoire
        rep_load = RepertoireLoader()
        rep_arrs = rep_load.load_repertoire(repertoire_path=f"{path_to_family}/repertoire")
        self.centroids, self.descriptors, self.sim_fitnesses, self.genotypes = rep_arrs

        # Define the family ancestors' mu and var
        ancestors_loader = AncestorsLoader()
        self.ancestor_mus, self.ancestors_vars, self.ancestors_names = ancestors_loader.load_ancestors(path_to_ancestors=f"{path_to_family}/ancestors")

if __name__ == "__main__":
    fam = Family(path_to_family="results/family_0")
    print(fam.ancestors_names)
    print(fam.ancestor_mus.shape)

