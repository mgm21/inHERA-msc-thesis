from src.utils.all_imports import *

def plot_fitness_vs_numiter(path_to_folder, paths_to_include, path_to_result, group_names=None):
    plt.style.use("seaborn")
    fig, ax = plt.subplots()

    for path_to_include in paths_to_include:
        observation_arrays = []
        if group_names == None:
            group_name = path_to_include

        for root, dir_names, file_names in os.walk(path_to_folder):
            if path_to_include in root:
                # Add y_observed from this directory to a master list of y_observed arrays
                observation_arrays += [jnp.load(f"{root}/y_observed.npy")]
            
        # Turn list to JAX
        observation_arrays = jnp.array(observation_arrays)

        # Get the median & quantiles arrays for this path_to_include
        median_array = jnp.nanmedian(observation_arrays, axis=0)
        rolling_max_array = [jnp.nanmax(observation_arrays[:i+1]) for i in range(observation_arrays.shape[0]-1)]
        quantile1_array = jnp.nanquantile(observation_arrays, q=0.25, axis=0)
        quantile3_array = jnp.nanquantile(observation_arrays, q=0.75, axis=0)

        # Create the number of iterations array
        num_iter = jnp.array(list(range(1, observation_arrays.shape[1]+1)))  

        # Plot this median & quantiles on the general fig
        ax.plot(num_iter, median_array, label=group_name)
        ax.fill_between(num_iter, quantile1_array, quantile3_array, alpha=0.4)


    # Save the figure
    ax.legend()
    fig.savefig(path_to_result, dpi=600) 

paths_to_include = []
for i in range(5):
    paths_to_include += [f"damaged_{i}"]
plot_fitness_vs_numiter(path_to_folder="results", paths_to_include=["damaged_5"], path_to_result=".")