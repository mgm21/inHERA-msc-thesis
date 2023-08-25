from src.utils.all_imports import *

repertoires_path = "./final_repertoires"
families_path = "./final_families"

for file_name in os.listdir(repertoires_path):
    d = os.path.join(repertoires_path, file_name)

    # For every repertoire in the repertoires folder
    if os.path.isdir(d):
        # Creation of family folder structure
        family_path = f"{families_path}/family-{file_name}"
        shutil.copytree(f"{repertoires_path}/{file_name}", f"{family_path}/repertoire")
        os.mkdir(path=f"{family_path}/ancestors")
        os.mkdir(path=f"{family_path}/children")

        # Normalisation of fitnesses
        f = jnp.load(f"{family_path}/repertoire/fitnesses.npy")
        fmax = jnp.nanmax(f[f != -jnp.inf])
        fmin = jnp.nanmin(f[f != -jnp.inf])
        f = f.at[f != -jnp.inf].set((f[f != -jnp.inf]-fmin)/(fmax-fmin))
        norm_params = jnp.array([fmin, fmax])
        jnp.save(f"{family_path}/norm_params.npy", norm_params)
        jnp.save(f"{family_path}/repertoire/fitnesses.npy", f)

# Manually add a task file after this routine... Could think about how to save the task object and file.

