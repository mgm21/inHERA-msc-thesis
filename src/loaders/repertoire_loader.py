from src.utils.all_imports import *

class RepertoireLoader:
    def load_repertoire(self, repertoire_path="./families/last_repertoire/", remove_empty_bds=False):
        centroids = jnp.load(os.path.join(repertoire_path, "centroids.npy"))
        descriptors = jnp.load(os.path.join(repertoire_path, "descriptors.npy"))
        fitnesses = jnp.load(os.path.join(repertoire_path, "fitnesses.npy"))
        genotypes = jnp.load(os.path.join(repertoire_path, "genotypes.npy"))

        if remove_empty_bds:
            centroids = centroids[fitnesses != -jnp.inf]
            descriptors = descriptors[fitnesses != -jnp.inf]
            genotypes = genotypes[fitnesses != -jnp.inf]
            # NB important to keep fitnesses refactor for the end because affects the ones above
            fitnesses = fitnesses[fitnesses != -jnp.inf]

            
        return centroids, descriptors, fitnesses, genotypes

if __name__ == "__main__":
    repertoire_loader = RepertoireLoader()
    
    rep_arrs_1 = repertoire_loader.load_repertoire(repertoire_path="./families//last_repertoire/")
    rep_arrs_2 = repertoire_loader.load_repertoire(repertoire_path="./families//class-last_repertoire/")
    rep_arrs_3 = repertoire_loader.load_repertoire(repertoire_path="./families//class_example_repertoire/")    