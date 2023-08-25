from src.utils.all_imports import *

f = jnp.load("final_repertoires/seed_0_last_repertoire/fitnesses.npy")

# Normalisation routine
fmax = jnp.nanmax(f[f != -jnp.inf])
fmin = jnp.nanmin(f[f != -jnp.inf])

f = f.at[f != -jnp.inf].set((f[f != -jnp.inf]-fmin)/(fmax-fmin))

norm_params = jnp.array([fmin, fmax])

# jnp.save("norm_params.npy", norm_params)
print(norm_params)

print(f)

# jnp.save("fitnesses.npy", f)
