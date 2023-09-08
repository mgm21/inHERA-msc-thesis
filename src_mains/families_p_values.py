from src.utils.all_imports import *
from scipy.stats import mannwhitneyu
import matplotlib.patches as mpatches

def p_values(path_to_folder, paths_to_include, path_to_result, adaptation_step=2, show_spread=True, group_names=None, include_median_plot=True, include_max_plot=True):    
    sns.set()
    color_list = sns.color_palette(n_colors=15)
    colors = []

    color_dict = {"ITE/": color_list[0],
                  "GPCF-reg/": color_list[2],
                  "GPCF-1trust/": color_list[3],
                  "inHERA-expert/": color_list[7],
                  "inHERA-b0-expert/": color_list[8],
                  "inHERA-b0/": color_list[6],
                  "inHERA/": color_list[5],
                  "GPCF/": color_list[1],}

    all_arrs_to_compare = []

    if len(paths_to_include) == 0: print("Please include at least 1 list with tags in paths_to_include")

    for idx, path_to_include in enumerate(paths_to_include):
        # Initialise objects to store all the y_observed arrays to be averaged to plot one group (i.e. one curve)
        observation_arrays = []
        max_observation_arrays = []

        # Set algo specific color
        
        for algo_name in list(color_dict.keys()):
            if algo_name in path_to_include:
                colors += [color_dict[algo_name]]
                print(algo_name)

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

        # print(f"These are the max observation_arrays: {max_observation_arrays}")

        all_arrs_to_compare += [max_observation_arrays[:, adaptation_step]]

        # print(f"These are the all_arrs_to_compare: {all_arrs_to_compare}")

    data = jnp.array(all_arrs_to_compare)
    fig = plt.figure(figsize =(10, 7))
    ax = fig.add_axes([0, 0, 1, 1])
    labels = ["ITE", "GPCF", "GPCF-reg", "GPCF-1trust", "inHERA", "inHERA-b0", "inHERA-exp", "inHERA-b0-exp",]
    bp = ax.boxplot(data, labels=labels, medianprops = dict(color = "black",))

    print(colors)
    for artist, color in zip(bp['boxes'], colors):
        patch = mpatches.PathPatch(artist.get_path(), color=color)
        ax.add_artist(patch)
        artist.set_alpha(0.5)

    # print(data)

    scaling = 50
    for i, arr in enumerate(all_arrs_to_compare):
        if i != 0:
            statistic, p_value = mannwhitneyu(all_arrs_to_compare[0], all_arrs_to_compare[i])
            print(p_value)
            x1, x2 = 0+1, i+1
            y, h, col = 0.5 - 1/scaling + (i)/scaling, 1/scaling, 'k'
            ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)

            if p_value > 0.1:
                ax.text((x1+x2)*.5, y+h, f"ns" % p_value, ha='center', va='bottom', color=col)
            if p_value <= 0.05:
                ax.text((x1+x2)*.5, y+h, f"p = %.3f" % p_value, ha='center', va='bottom', color=col)
            if p_value <= 0.1:
                ax.text((x1+x2)*.5, y+h, f"p = %.3f" % p_value, ha='center', va='bottom', color=col)
    
    ax.yaxis.grid(True)
    ax.set_ylabel('Maximum fitness')

    fig.savefig("plot_results/boxplot.png", dpi=600, bbox_inches="tight")
    plt.show()


damage = "damaged_1_2_3/"
paths_to_include = []
for algorithm in ["ITE/", "GPCF/", "GPCF-reg/", "GPCF-1trust/", "inHERA/", "inHERA-b0/", "inHERA-expert/", "inHERA-b0-expert/",]:
    paths_to_include += [[algorithm, damage,]]

now = datetime.now()
now_str = now.strftime(f"%Y-%m-%d_%H-%M-%S")

p_values(path_to_folder="results/final_children_restricted",
                        paths_to_include=paths_to_include,
                        path_to_result=f"plot_results/result_plot-adaptation-{now_str}",
                        show_spread=False,
                        include_median_plot=True)