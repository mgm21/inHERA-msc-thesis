from src.utils.all_imports import *

def plot_fitness_vs_numiter(path_to_folder, paths_to_include, path_to_result, show_spread=True, group_names=None, include_median_plot=True, include_max_plot=True):    
    # plt.style.use("seaborn")
    sns.set()
    sns.color_palette(n_colors=10)
    maxscores = []
    medscores = []

    alpha = 0.2
    if include_median_plot and include_max_plot:
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
    elif include_median_plot and not include_max_plot:
        fig, ax2 = plt.subplots()
    elif not include_median_plot and include_max_plot:
        fig, ax1 = plt.subplots()

    if len(paths_to_include) == 0: print("Please include at least 1 list with tags in paths_to_include")

    for idx, path_to_include in enumerate(paths_to_include):
        # Initialise objects to store all the y_observed arrays to be averaged to plot one group (i.e. one curve)
        observation_arrays = []
        max_observation_arrays = []

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
            print(f"Sorry, this path_to_include {path_to_include} is not found in the {path_to_folder} folder. Or the {path_to_folder} is not recognised.")
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
                
                if include_median_plot: ax2.fill_between(num_iter, quantile1_array, quantile3_array, alpha=alpha)
                if include_max_plot: ax1.fill_between(num_iter, max_quantile1_array, max_quantile3_array, alpha=alpha)
        
        else:
            num_iter = num_iter = jnp.array(list(range(1, observation_arrays.shape[1]+1)))  
            median_array = observation_arrays[0]
            max_median_array = max_observation_arrays[0]

            # Dummy fills to ensure consistency with colour fill 
            if include_median_plot: ax2.fill_between(num_iter, jnp.zeros(shape=len(num_iter)), jnp.zeros(shape=len(num_iter)), alpha=alpha)
            if include_max_plot: ax1.fill_between(num_iter, jnp.zeros(shape=len(num_iter)), jnp.zeros(shape=len(num_iter)), alpha=alpha)

        # Calculate this curve's score
        maxscores += [round(jnp.sum(max_median_array)/20, ndigits=2)]
        medscores += [round(jnp.sum(median_array)/20, ndigits=2)]

        # Generate the default group name from the tags in the path to include
        if group_names == None:
            name_tags = [tag for tag in path_to_include]
            maxgroup_name = ", ".join(name_tags).replace("/", "").replace("_", " ").replace("-", ",") + f" , m = %.2f" % maxscores[idx]
            medgroup_name = ", ".join(name_tags).replace("/", "").replace("_", " ").replace("-", ",") + f" , m = %.2f" % medscores[idx]
        
        else:
            maxgroup_name = group_names[idx] + f" , m = %.2f" % maxscores[idx]
            medgroup_name = group_names[idx] + f" , m = %.2f" % medscores[idx]
        
        # Plot this median & quantiles on the general fig
        if include_median_plot: ax2.plot(num_iter, median_array, label=medgroup_name)
        if include_max_plot: ax1.plot(num_iter, max_median_array, label=maxgroup_name)

    # Configure and save the figure

    # min_y, max_y = 0, 0.2

    if include_max_plot:
        ax1.set_xlabel('Adaptation steps')
        ax1.set_ylabel('Maximum fitness')
        ax1.legend()
        ax1.set_ylim()

    if include_median_plot:
        ax2.set_xlabel('Adaptation steps')
        ax2.set_ylabel('Median fitness')
        ax2.legend()
        ax2.set_ylim()

    fig.savefig(path_to_result, dpi=600)
        
now = datetime.now()
now_str = now.strftime(f"%Y-%m-%d_%H-%M-%S")

# Make sure to incude a "/" at the end of a tag to not confuse damaged_0/ with damaged_0_1/, for example

# damage = "damaged_1_2_3/"
# paths_to_include = []
# paths_to_include += [["ITE/", damage]]
# paths_to_include += [["GPCF/", damage]]
# paths_to_include += [["GPCF-reg/", damage]]
# paths_to_include += [["GPCF-1trust/", damage]]
# paths_to_include += [["inHERA/", damage]]
# paths_to_include += [["inHERA-b0/", damage]]


# now = datetime.now()
# now_str = now.strftime(f"%Y-%m-%d_%H-%M-%S")

# plot_fitness_vs_numiter(path_to_folder="results/numiter40k_final_children_with_intact_ancestor_and_intact_child",
#                         paths_to_include=paths_to_include,
#                         path_to_result=f"plot_results/result_plot-adaptation-{now_str}",
#                         show_spread=False,
#                         include_median_plot=True)

# paths_to_include = []
# paths_to_include += [["ITE/", "damaged_1_2_3/"]]
# paths_to_include += [["GPCF/", "damaged_1_2_3/"]]
# paths_to_include += [["GPCF-reg/", "damaged_1_2_3/"]]
# paths_to_include += [["GPCF-1trust/", "damaged_1_2_3/"]]
# paths_to_include += [["inHERA/", "damaged_1_2_3/"]]


# paths_to_include = []
# paths_to_include += [["seed_20_", "damaged_1_2_3/"]]

# for i in [1, 0.1, 0.01]:
#     for j in [1, 0.1, 0.01]:
#         now = datetime.now()
#         now_str = now.strftime(f"%Y-%m-%d_%H-%M-%S")    
#         paths_to_include = []
#         paths_to_include += [["damaged_1/", f"l1_{i}-v_{j}", "GPCF-reg"]]

#         plot_fitness_vs_numiter(path_to_folder="numiter40k_first_hyperparameter_sweep_bad_formatting",
#                                 paths_to_include=paths_to_include,
#                                 path_to_result=f"plot_results/result_plot-adaptation-{now_str}",
#                                 show_spread=True,
#                                 include_median_plot=True)


now = datetime.now()
now_str = now.strftime(f"%Y-%m-%d_%H-%M-%S")

kappa_list = [2, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001,]
group_names = []

paths_to_include = []
for kappa in kappa_list:
    group_names += [f"k = {kappa}"]  
    paths_to_include += [["damaged_1/", f"kappa_{kappa}", "inHERA-b0"]]

plot_fitness_vs_numiter(path_to_folder="results/inhera-b0_kappa_sweep",
                        paths_to_include=paths_to_include,
                        path_to_result=f"plot_results/result_plot-adaptation-{now_str}",
                        show_spread=True,
                        include_median_plot=True,
                        group_names=group_names)


# now = datetime.now()
# now_str = now.strftime(f"%Y-%m-%d_%H-%M-%S")    
# paths_to_include = []
# paths_to_include += [["damaged_1_2_3/", f"l1_{i}-v_{j}", "GPCF-reg"] for i in [0.01] for j in [1, 0.1, 0.01]]

# plot_fitness_vs_numiter(path_to_folder="numiter40k_first_hyperparameter_sweep_bad_formatting",
#                         paths_to_include=paths_to_include,
#                         path_to_result=f"plot_results/result_plot-adaptation-{now_str}",
#                         show_spread=False,
#                         include_median_plot=True)


# paths_to_include = []
# paths_to_include += [["ITE/", "damaged_3_4/"]]
# paths_to_include += [["GPCF/", "damaged_3_4/"]]
# paths_to_include += [["GPCF-reg/", "damaged_3_4/"]]
# paths_to_include += [["GPCF-1trust/", "damaged_3_4/"]]
# paths_to_include += [["inHERA/", "damaged_3_4/"]]

# now = datetime.now()
# now_str = now.strftime(f"%Y-%m-%d_%H-%M-%S")
# plot_fitness_vs_numiter(path_to_folder="results/numiter40k_final_children_without_intact_ancestor",
#                         paths_to_include=paths_to_include,
#                         path_to_result=f"plot_results/result_plot-adaptation-{now_str}-without-intact",
#                         show_spread=False,
#                         include_median_plot=True)
# now = datetime.now()
# now_str = now.strftime(f"%Y-%m-%d_%H-%M-%S")
# plot_fitness_vs_numiter(path_to_folder="results/numiter40k_final_children_with_intact_ancestor_and_intact_child",
#                         paths_to_include=paths_to_include,
#                         path_to_result=f"plot_results/result_plot-adaptation-{now_str}-with-intact",
#                         show_spread=False,
#                         include_median_plot=True)

# paths_to_include = []
# paths_to_include += [["damaged_0/"]]

# now = datetime.now()
# now_str = now.strftime(f"%Y-%m-%d_%H-%M-%S")
# plot_fitness_vs_numiter(path_to_folder="numiter40k_final_families",
#                         paths_to_include=paths_to_include,
#                         path_to_result=f"plot_results/result_plot-adaptation-{now_str}-without-intact",
#                         show_spread=True,
#                         include_median_plot=True)

# TODO: only thing to look at is that here you have to specify the folder parent to y_observed or else it cannot find it. Is that okay?