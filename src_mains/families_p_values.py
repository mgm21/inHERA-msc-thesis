from src.utils.all_imports import *
from scipy.stats import mannwhitneyu

def p_values(path_to_folder, paths_to_include, path_to_result, adaptation_step=6, show_spread=True, group_names=None, include_median_plot=True, include_max_plot=True):    
    # plt.style.use("seaborn")
    sns.set()
    sns.color_palette(n_colors=15)    
    fig, ax = plt.subplots()

    all_arrs_to_compare = []

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
    
        max_observation_arrays = jnp.array(max_observation_arrays)
        all_arrs_to_compare += [max_observation_arrays[:, adaptation_step]]
    
    all_arrs_to_compare = jnp.array(all_arrs_to_compare)
    print(all_arrs_to_compare)

    print(all_arrs_to_compare[0])

    statistic, p_value = mannwhitneyu(all_arrs_to_compare[0], all_arrs_to_compare[1])
    print(statistic, p_value)

    # # Configure and save the figure
    # ax.set_xlabel('Adaptation steps')
    # ax.set_ylabel('Maximum fitness')
    # # ax1.legend(loc="lower right")
    # # To put legend outside of figure
    # # ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # ax.set_ylim()
    # fig.savefig(path_to_result, dpi=600, bbox_inches="tight")


damage = "damaged_1_2_3/"
paths_to_include = []
for algorithm in ["GPCF/", "inHERA/"]: #  "GPCF-reg/", "GPCF-1trust/", "inHERA/", "inHERA-b0/"
    paths_to_include += [[algorithm, damage, "kappa_0.001"]]

now = datetime.now()
now_str = now.strftime(f"%Y-%m-%d_%H-%M-%S")

p_values(path_to_folder="results/kappa_sweep",
                        paths_to_include=paths_to_include,
                        path_to_result=f"plot_results/result_plot-adaptation-{now_str}",
                        show_spread=False,
                        include_median_plot=True)