from src.utils.all_imports import *
import matplotlib.font_manager as fm

def plot_fitness_vs_numiter(path_to_folder, paths_to_include, path_to_result, show_spread=True, group_names=None, include_median_plot=True, include_max_plot=True):    
    # plt.style.use("seaborn")
    sns.set()
    color_list = sns.color_palette(n_colors=15)

    color_dict = {"ITE/": color_list[0],
                  "GPCF-reg/": color_list[2],
                  "GPCF-1trust/": color_list[3],
                  "inHERA-expert/": color_list[7],
                  "inHERA-b0-expert/": color_list[8],
                  "inHERA-b0/": color_list[6],
                  "inHERA/": color_list[5],
                  "GPCF/": color_list[1],}


    maxscores = []
    medscores = []

    alpha = 0.2
    if include_median_plot and include_max_plot:
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
    elif include_median_plot and not include_max_plot:
        fig, ax2 = plt.subplots()
    elif not include_median_plot and include_max_plot:
        fig, ax1 = plt.subplots(figsize=(9, 6))

    if len(paths_to_include) == 0: print("Please include at least 1 list with tags in paths_to_include")

    for idx, path_to_include in enumerate(paths_to_include):

        # Set algo specific color
        for algo_name in list(color_dict.keys()):
            if algo_name in path_to_include:
                color = color_dict[algo_name]

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
                
                if include_median_plot: ax2.fill_between(num_iter, quantile1_array, quantile3_array, alpha=alpha, color=color)
                if include_max_plot: ax1.fill_between(num_iter, max_quantile1_array, max_quantile3_array, alpha=alpha, color=color)
        
        else:
            num_iter = num_iter = jnp.array(list(range(1, observation_arrays.shape[1]+1)))  
            median_array = observation_arrays[0]
            max_median_array = max_observation_arrays[0]

            # Dummy fills to ensure consistency with colour fill 
            if include_median_plot: ax2.fill_between(num_iter, jnp.zeros(shape=len(num_iter)), jnp.zeros(shape=len(num_iter)), alpha=alpha, color=color)
            if include_max_plot: ax1.fill_between(num_iter, jnp.zeros(shape=len(num_iter)), jnp.zeros(shape=len(num_iter)), alpha=alpha, color=color)

        # Calculate this curve's score
        maxscores += [round(jnp.sum(max_median_array)/20, ndigits=8)]
        medscores += [round(jnp.sum(median_array)/20, ndigits=8)]

        # Generate the default group name from the tags in the path to include
        if group_names == None:
            name_tags = [tag for tag in path_to_include]
            maxgroup_name = ", ".join(name_tags).replace("/", "").replace("_", " ").replace("-", ",") + f" , m = %.2f" % maxscores[idx]
            medgroup_name = ", ".join(name_tags).replace("/", "").replace("_", " ").replace("-", ",") + f" , m = %.2f" % medscores[idx]
        
        else:
            maxgroup_name = group_names[idx] + f" , m = %.3f" % maxscores[idx]
            medgroup_name = group_names[idx] + f" , m = %.3f" % medscores[idx]
        
        # Plot this median & quantiles on the general fig
        if include_median_plot: ax2.plot(num_iter, median_array, label=medgroup_name, color=color)
        if include_max_plot: ax1.plot(num_iter, max_median_array, label=maxgroup_name, color=color)

    # Configure and save the figure

    # min_y, max_y = 0, 0.2

    font_used = "Serif"
    font_size = 35
    font = {'fontname': font_used}
    legend_font = fm.FontProperties(family=font_used)
    legend_font._size = font_size - 3


    if include_max_plot:
        ax1.set_xlabel('Adaptation steps', fontsize=font_size, **font)
        ax1.set_ylabel('Maximum fitness', fontsize=font_size, **font)
        ax1.legend(loc="lower right", prop=legend_font,)
        ax1.tick_params(axis='x', labelsize=legend_font._size - 5)
        ax1.tick_params(axis='y', labelsize=legend_font._size - 5)
        # To put legend outside of figure
        # ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.set_ylim()

    if include_median_plot:
        ax2.set_xlabel('Adaptation steps')
        ax2.set_ylabel('Median fitness')
        ax2.legend()
        ax2.set_ylim()

    fig.savefig(path_to_result, dpi=600, bbox_inches="tight")

    return maxscores


def get_n_best_seeds(path_to_folder, paths_to_include, n=10):
    maxscores = []
    medscores = []

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

        if len(max_observation_arrays) != 1:
            max_median_array = jnp.nanmedian(max_observation_arrays, axis=0)
            num_iter = jnp.array(list(range(1, max_observation_arrays.shape[1]+1)))
        
        else:
            num_iter = num_iter = jnp.array(list(range(1, observation_arrays.shape[1]+1)))  
            median_array = observation_arrays[0]
            max_median_array = max_observation_arrays[0]

        # Calculate this curve's score
        maxscores += [round(jnp.sum(max_median_array)/20, ndigits=8)]

        # Generate the default group name from the tags in the path to include
        name_tags = [tag for tag in path_to_include]
        maxgroup_name = ", ".join(name_tags).replace("/", "").replace("_", " ").replace("-", ",") + f" , m = %.2f" % maxscores[idx]
        print(maxgroup_name)
    
    maxscores = jnp.array(maxscores)
    top_indices = jnp.argpartition(maxscores, -n)[-n:]
    top_seeds = top_indices + 1
    return top_seeds


# FLOW FOR PLOTTING ONLY THE TOP n SEEDS 
damage = "damaged_1_2_3/"
type = damage.split("damaged")[-1].strip("/")
if type == "_1_2_3":
    type = "3"
if type == "_3_4":
    type = "2"
if type == "_1":
    type = "1"
n = 15
algos = ["GPCF", "GPCF-reg", "GPCF-1trust", "inHERA", "inHERA-b0", "inHERA-expert", "inHERA-b0-expert"]
path_to_originals = "results/final_children_not_in_ancestors"

for algo in algos:
    # Get ite best seeds
    ite_all_paths = []
    for seed in range(1,21):
        ite_all_paths += [[f"seed_{seed}_", f"ITE/", damage]]
    ite_best_seeds = get_n_best_seeds(path_to_folder="results/final_children", paths_to_include=ite_all_paths, n=n)

    # Get algo best seeds
    algo_all_paths = []
    for seed in range(1,21):
        algo_all_paths += [[f"seed_{seed}_", f"{algo}/", damage]]
    algo_best_seeds = get_n_best_seeds(path_to_folder="results/final_children", paths_to_include=algo_all_paths, n=n)

    # Create the best_n folders
    overall_dir = f"results/best_{n}_{type}"
    if not os.path.exists(overall_dir):
        os.makedirs(overall_dir)

    ite_dir = f"{overall_dir}/ITE/"
    if not os.path.exists(ite_dir):
        os.makedirs(ite_dir)

    algo_dir = f"{overall_dir}/{algo}/"
    if not os.path.exists(algo_dir):
        os.makedirs(algo_dir)

    for best_seed in ite_best_seeds:
        copy_tree(f"{path_to_originals}/family-seed_{best_seed}_repertoire/{damage}/ITE", f"{ite_dir}/seed_{best_seed}/{damage}")

    for best_seed in algo_best_seeds:
        copy_tree(f"{path_to_originals}/family-seed_{best_seed}_repertoire/{damage}/{algo}/", f"{algo_dir}/seed_{best_seed}/{damage}")

    now = datetime.now()
    now_str = now.strftime(f"%Y-%m-%d_%H-%M-%S")

    paths_to_include = [["ITE/", damage],
                        [f"{algo}/", damage]]

    group_names = ["ITE", algo]
    plot_fitness_vs_numiter(path_to_folder=overall_dir,
                            paths_to_include=paths_to_include,
                            path_to_result=f"plot_results/{algo}-{type}",
                            show_spread=True,
                            include_median_plot=False,
                            group_names=group_names)

# # FINAL PLOTS PLOTTING ROUTNINE
# for algo in ["GPCF", "GPCF-reg", "GPCF-1trust", "inHERA", "inHERA-b0", "inHERA-expert", "inHERA-b0-expert"]:
#     group_names = ["ITE", algo]
#     for damage in ["damaged_1/", "damaged_3_4/", "damaged_1_2_3/"]:
#         paths_to_include = []
#         for algorithm in ["ITE/", f"{algo}/"]: # "ITE", "GPCF/", "GPCF-reg/", "GPCF-1trust/", "inHERA/", "inHERA-b0/", "inHERA-expert/", "inHERA-b0-expert/",
#             paths_to_include += [[algorithm, damage,]]

#         now = datetime.now()
#         now_str = now.strftime(f"%Y-%m-%d_%H-%M-%S")

#         plot_fitness_vs_numiter(path_to_folder="results/final_children",
#                                 paths_to_include=paths_to_include,
#                                 path_to_result=f"plot_results/{algo}-{now_str}",
#                                 show_spread=True,
#                                 include_median_plot=False,
#                                 group_names=group_names)

# # Make sure to incude a "/" at the end of a tag to not confuse damaged_0/ with damaged_0_1/, for example
# damage = "damaged_1/"
# paths_to_include = []
# paths_to_include += [["ITE/", damage]]
# paths_to_include += [["GPCF/", damage]]
# paths_to_include += [["GPCF-reg/", damage]]
# paths_to_include += [["GPCF-1trust/", damage]]
# paths_to_include += [["inHERA/", damage]]
# paths_to_include += [["inHERA-b0/", damage]]


# now = datetime.now()
# now_str = now.strftime(f"%Y-%m-%d_%H-%M-%S")

# plot_fitness_vs_numiter(path_to_folder="results/final_children",
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


# # MAXSCORING ROUTINE - l1, v
# maxscores_matrix = []
# for damage in ["damaged_1/", "damaged_3_4/", "damaged_1_2_3/"]:
#     paths_to_include = []
#     group_names = []
#     for i in [0.01, 0.001, 0.0001]:
#         for j in [1, 0.1, 0.01, 0.001, 0.0001]:
#             now = datetime.now()
#             now_str = now.strftime(f"%Y-%m-%d_%H-%M-%S")    
#             paths_to_include += [[damage, f"l1_{i}-v_{j}", "GPCF-reg"]]
#             group_names += [f"l1_{i}-v{j}"]

#     maxscores_matrix += [plot_fitness_vs_numiter(path_to_folder="results/l1_v_gpcf_sweep",
#                                     paths_to_include=paths_to_include,
#                                     path_to_result=f"plot_results/result_plot-adaptation-{now_str}",
#                                     show_spread=False,
#                                     include_median_plot=False,
#                                     group_names=group_names)]
# maxscores_matrix = jnp.array(maxscores_matrix)
# print(maxscores_matrix)
# print(jnp.argmax(jnp.mean(maxscores_matrix, axis=0)))
# print(group_names[jnp.argmax(jnp.mean(maxscores_matrix, axis=0))])

# # MAXSCORING ROUTINE - u
# maxscores_matrix = []
# for damage in ["damaged_1/", "damaged_3_4/", "damaged_1_2_3/"]:
#     paths_to_include = []
#     group_names = []
#     for u in [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]:
#             now = datetime.now()
#             now_str = now.strftime(f"%Y-%m-%d_%H-%M-%S")    
#             paths_to_include += [[damage, f"u_{u}", "inHERA/"]]
#             group_names += [f"u_{u}"]

#     maxscores_matrix += [plot_fitness_vs_numiter(path_to_folder="results/u_sweep",
#                                     paths_to_include=paths_to_include,
#                                     path_to_result=f"plot_results/result_plot-adaptation-{now_str}",
#                                     show_spread=False,
#                                     include_median_plot=False,
#                                     group_names=group_names)]
    
# maxscores_matrix = jnp.array(maxscores_matrix)
# print(maxscores_matrix)
# print(jnp.argmax(jnp.mean(maxscores_matrix, axis=0)))
# print(group_names[jnp.argmax(jnp.mean(maxscores_matrix, axis=0))])

# # MAXSCORING ROUTINE - kappa
# maxscores_matrix = []
# for damage in ["damaged_1/", "damaged_3_4/", "damaged_1_2_3/"]:
#     paths_to_include = []
#     group_names = []
#     for kappa in [4, 3, 2, 1, 0.1, 0.01, 0.001, 0.0001]:
#             now = datetime.now()
#             now_str = now.strftime(f"%Y-%m-%d_%H-%M-%S")    
#             paths_to_include += [[damage, f"kappa_{kappa}", "inHERA-b0/"]]
#             group_names += [f"k{kappa}"]

#     maxscores_matrix += [plot_fitness_vs_numiter(path_to_folder="results/kappa_sweep",
#                                     paths_to_include=paths_to_include,
#                                     path_to_result=f"plot_results/result_plot-adaptation-{now_str}",
#                                     show_spread=False,
#                                     include_median_plot=False,
#                                     group_names=group_names)]
    
# maxscores_matrix = jnp.array(maxscores_matrix)
# print(maxscores_matrix)
# print(jnp.argmax(jnp.mean(maxscores_matrix, axis=0)))
# print(group_names[jnp.argmax(jnp.mean(maxscores_matrix, axis=0))])

# # # TO PLOT JUST THE BEST HYPERPARAMETER plots for u 
# paths_to_include = []

# paths_to_include += [["damaged_1_2_3/", "inHERA/", "u_0.0001"]]
# group_names = ["u_0.0001"]
# plot_fitness_vs_numiter(path_to_folder="results/u_sweep",
#                                     paths_to_include=paths_to_include,
#                                     path_to_result=f"plot_results/result_plot-adaptation-{now_str}",
#                                     show_spread=True,
#                                     include_median_plot=False,
#                                     group_names=group_names)


# # inHERA SPECIFIC KAPPA SWEEP
# now = datetime.now()
# now_str = now.strftime(f"%Y-%m-%d_%H-%M-%S")

# kappa_list = [2, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001,]
# group_names = []

# paths_to_include = []
# for kappa in kappa_list:
#     group_names += [f"k = {kappa}"]  
#     paths_to_include += [["damaged_1/", f"kappa_{kappa}", "inHERA-b0"]]

# plot_fitness_vs_numiter(path_to_folder="results/inhera-b0_kappa_sweep",
#                         paths_to_include=paths_to_include,
#                         path_to_result=f"plot_results/result_plot-adaptation-{now_str}",
#                         show_spread=False,
#                         include_median_plot=True,
#                         group_names=group_names)


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

# paths_to_include = []
# paths_to_include += [["damaged_1/", "ITE"]]
# now = datetime.now()
# now_str = now.strftime(f"%Y-%m-%d_%H-%M-%S")
# plot_fitness_vs_numiter(path_to_folder="results/numiter40k_final_children_with_intact_ancestor_and_intact_child",
#                         paths_to_include=paths_to_include,
#                         path_to_result=f"plot_results/result_plot-adaptation-{now_str}-without-intact",
#                         show_spread=True,
#                         include_median_plot=False,
#                         group_names=["damaged 1"])





# TODO: only thing to look at is that here you have to specify the folder parent to y_observed or else it cannot find it. Is that okay?