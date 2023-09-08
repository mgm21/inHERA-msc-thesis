from src.utils.all_imports import *
from scipy.stats import mannwhitneyu

def p_values(path_to_folder, paths_to_include, path_to_result, adaptation_step=3, show_spread=True, group_names=None, include_median_plot=True, include_max_plot=True):    
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

    data = jnp.array(all_arrs_to_compare)
    fig = plt.figure(figsize =(10, 7))
    ax = fig.add_axes([0, 0, 1, 1])
    bp = ax.boxplot(data)


    print(data)

    scaling = 10
    # Get the p-value between the first array and all the other arrays: https://stackoverflow.com/questions/45310254/fixed-digits-after-decimal-with-f-strings
    for i, arr in enumerate(all_arrs_to_compare):
        if i != 0:
            statistic, p_value = mannwhitneyu(all_arrs_to_compare[0], all_arrs_to_compare[i])
            print(p_value)
            x1, x2 = 0+1, i+1
            y, h, col = 1/scaling + i/scaling, 1, 'k'
            ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
            ax.text((x1+x2)*.5, y+h, f"p = %.2f" % p_value, ha='center', va='bottom', color=col)

    fig.savefig("plot_results/boxplot.png", dpi=600)


damage = "damaged_1/"
paths_to_include = []
for algorithm in ["GPCF-reg/", "GPCF-1trust/", "inHERA/", "inHERA-b0/", "GPCF/"]:
    paths_to_include += [[algorithm, damage,]]

now = datetime.now()
now_str = now.strftime(f"%Y-%m-%d_%H-%M-%S")

p_values(path_to_folder="results/data_with_only_a_few_repertoires",
                        paths_to_include=paths_to_include,
                        path_to_result=f"plot_results/result_plot-adaptation-{now_str}",
                        show_spread=False,
                        include_median_plot=True)