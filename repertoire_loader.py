from all_imports import *

class RepertoireLoader:
    def load_repertoire(self, repertoire_path="./last_repertoire/"):
        centroids = jnp.load(os.path.join(repertoire_path, "centroids.npy"))
        descriptors = jnp.load(os.path.join(repertoire_path, "descriptors.npy"))
        fitnesses = jnp.load(os.path.join(repertoire_path, "fitnesses.npy"))
        genotypes = jnp.load(os.path.join(repertoire_path, "genotypes.npy"))
        return centroids, descriptors, fitnesses, genotypes

if __name__ == "__main__":
    repertoire_loader = RepertoireLoader()
    
    rep_arrs_1 = repertoire_loader.load_repertoire(repertoire_path="./last_repertoire/")
    rep_arrs_2 = repertoire_loader.load_repertoire(repertoire_path="./class-last_repertoire/")
    rep_arrs_3 = repertoire_loader.load_repertoire(repertoire_path="./class_example_repertoire/")    