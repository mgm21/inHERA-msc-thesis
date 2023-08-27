from src.utils.all_imports import *

def plot_fitness_vs_numiter(path_to_folder, paths_to_include, path_to_result, show_spread=True, group_names=None,):    
    plt.style.use("seaborn")
    alpha = 0.2
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

    if len(paths_to_include) == 0: print("Please include at least 1 list with tags in paths_to_include")

    for path_to_include in paths_to_include:
        # Initialise objects to store all the y_observed arrays to be averaged to plot one group (i.e. one curve)
        observation_arrays = []
        max_observation_arrays = []

        # Generate the default group name from the tags in the path to include
        if group_names == None:
            name_tags = [tag for tag in path_to_include]
            group_name = ", ".join(name_tags).replace("/", "").replace("_", " ")

        # For all directories in the overall folder
        for root, _, _ in os.walk(path_to_folder):
            # Important to add / so that the program can differentiate damaged_0 and damaged_0_1 for example
            root += "/"

            # Only retain the ones which match the requirements in the path to include
            if all(tag in root for tag in path_to_include):
                print(f"Included in {path_to_include}'s plot: {root}")

                observation_array = jnp.load(f"{root}/y_observed.npy")
                max_observation_array = jnp.array([jnp.nanmax(observation_array[:i+1]) for i in range(observation_array.shape[0])])
                observation_arrays += [observation_array]
                max_observation_arrays += [max_observation_array]
        
        if len(observation_arrays) == 0:
            print(f"Sorry, this path_to_include {path_to_include} is not found in the {path_to_folder} folder.")
            break

        # Turn observation lists to JAX to perform jnp operations on them
        observation_arrays = jnp.array(observation_arrays)
        max_observation_arrays = jnp.array(max_observation_arrays)

        if len(observation_arrays) != 1:
            median_array = jnp.nanmedian(observation_arrays, axis=0)
            max_median_array = jnp.nanmedian(max_observation_arrays, axis=0)
            num_iter = jnp.array(list(range(1, observation_arrays.shape[1]+1))) 

            if show_spread:
                quantile1_array = jnp.nanquantile(observation_arrays, q=0.25, axis=0)
                quantile3_array = jnp.nanquantile(observation_arrays, q=0.75, axis=0)
                max_quantile1_array = jnp.nanquantile(max_observation_arrays, q=0.25, axis=0)
                max_quantile3_array = jnp.nanquantile(max_observation_arrays, q=0.75, axis=0)
                
                ax2.fill_between(num_iter, quantile1_array, quantile3_array, alpha=alpha)
                ax1.fill_between(num_iter, max_quantile1_array, max_quantile3_array, alpha=alpha)
        
        else:
            num_iter = num_iter = jnp.array(list(range(1, observation_arrays.shape[1]+1)))  
            median_array = observation_arrays[0]
            max_median_array = max_observation_arrays[0]

            # Dummy fills to ensure consistency with colour fill 
            ax2.fill_between(num_iter, jnp.zeros(shape=len(num_iter)), jnp.zeros(shape=len(num_iter)), alpha=alpha)
            ax1.fill_between(num_iter, jnp.zeros(shape=len(num_iter)), jnp.zeros(shape=len(num_iter)), alpha=alpha)

        # Plot this median & quantiles on the general fig
        ax2.plot(num_iter, median_array, label=group_name)
        ax1.plot(num_iter, max_median_array, label=group_name)

    # Configure and save the figure
    ax2.legend()
    ax1.legend()
    ax1.set_xlabel('Adaptation steps')
    ax1.set_ylabel('Maximum fitness')
    ax2.set_xlabel('Adaptation steps')
    ax2.set_ylabel('Median fitness')
    ax1.set_ylim(0, 0.5)
    ax2.set_ylim(0, 0.5)

    fig.savefig(path_to_result, dpi=600) 


now = datetime.now()
now_str = now.strftime(f"%Y-%m-%d_%H-%M-%S")

# Make sure to incude a "/" at the end of a tag to not confuse damaged_0/ with damaged_0_1/, for example
paths_to_include = []
paths_to_include += [[f"damaged_{i}_{j}/",] for i in range(0, 5) for j in range(i+1, 4)]

plot_fitness_vs_numiter(path_to_folder="numiter40k_ancestors",
                        paths_to_include=paths_to_include,
                        path_to_result=f"plot_results/result_plot-40k_ancestors-{now_str}",
                        show_spread=False,)