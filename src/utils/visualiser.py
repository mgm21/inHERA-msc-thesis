from src.utils.all_imports import *

class Visualiser:
    def __init__(self, plt_style="seaborn"):
        self.plt_style = plt_style

    def old_get_maxfit_vs_iter_fig(self, path_to_arrs, path_to_res="."):
        # Load the fitness
        fitness = jnp.load(f"{path_to_arrs}/y_observed.npy")

        # # Normalise the fitness
        # fitness = (fitness - (jnp.nanmin(fitness)))/(jnp.nanmax(fitness)-jnp.nanmin(fitness))

        # Transform fitnesses to rolling max
        rolling_max_fitness = [jnp.nanmax(fitness[:i+1]) for i in range(fitness.shape[0])]

        # Create the number of iterations array
        num_iter = jnp.array(list(range(1, fitness.shape[0]+1)))        

        plt.style.use(self.plt_style)
        fig, ax = plt.subplots()

        ax.set_xlabel('Adaptation steps')
        ax.set_ylabel('Maximum fitness')

        ax.plot(num_iter, rolling_max_fitness, label="ITE")


        ax.legend()
        fig.savefig(f'./maxfit_vs_iter.png', dpi=100)

    def get_maxfit_vs_iter_fig(self, observed_fitnesses, plot_names, path_to_res="."):

        plt.style.use(self.plt_style)
        fig, ax = plt.subplots()

        for i in range(len(observed_fitnesses)):
            # Extract the current observed_fitness
            fitness = observed_fitnesses[i]

            # Transform fitnesses to rolling max
            rolling_max_fitness = [jnp.nanmax(fitness[:i+1]) for i in range(fitness.shape[0])]

            # Create the number of iterations array
            num_iter = jnp.array(list(range(1, fitness.shape[0]+1)))        

            ax.set_xlabel('Adaptation steps')
            ax.set_ylabel('Maximum fitness')

            ax.plot(num_iter, rolling_max_fitness, label=f"{plot_names[i]}")

        ax.legend()
        fig.savefig(f'{path_to_res}/maxfit_vs_iter.png', dpi=100)
    
    def get_mean_and_var_plot(self, means, vars, names, path_to_res=".", save_fig=True, title="",):
        plt.style.use(self.plt_style)
        fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

        for i in range(len(means)):
            mean = means[i]
            var = vars[i]

            rolling_max_mean = jnp.array([jnp.nanmax(mean[:i+1]) for i in range(mean.shape[0])])
            num_iter = jnp.array(list(range(1, mean.shape[0]+1))) 

            ax.plot(num_iter, rolling_max_mean, label=names[i])
            ax2.plot(num_iter, mean, label=names[i])

            ax.fill_between(num_iter, rolling_max_mean-var, rolling_max_mean+var, alpha=0.4)
            ax2.fill_between(num_iter, mean-var, mean+var, alpha=0.4)
        
        ax.set_xlabel('Adaptation steps')
        ax.set_ylabel('Maximum fitness observed')

        ax2.set_xlabel('Adaptation steps')
        ax2.set_ylabel('Mean fitness')

        ax.legend()
        ax2.legend()

        ax.set_title(f"{title}")
        ax2.set_title(f"{title}")

        if save_fig: fig.savefig(f'{path_to_res}/overall_perf_vs_num_iter.png', dpi=600)

        return ax, ax2
    
    def get_mean_and_var_plot_per_damage_level(self, means_dict, vars_dict, path_to_res=".", save_fig=True):

        num_damages_in_dict = 5

        names = list(means_dict.keys())

        plt.style.use(self.plt_style)

        fig, axs = plt.subplots(5, figsize=(10, 20))
        fig2, axs2 = plt.subplots(5, figsize=(10, 20))

        for damage_num in range(1, num_damages_in_dict+1):
            means = []
            vars = []

            for algo in names:
                means += [means_dict[algo][damage_num]]
                vars += [vars_dict[algo][damage_num]]
            
            means = jnp.array(means)
            vars = jnp.array(vars)

            for i in range(len(means)):
                mean = means[i]
                var = vars[i]

                rolling_max_mean = jnp.array([jnp.nanmax(mean[:i+1]) for i in range(mean.shape[0])])
                num_iter = jnp.array(list(range(1, mean.shape[0]+1))) 

                axs[damage_num-1].plot(num_iter, rolling_max_mean, label=names[i])
                axs2[damage_num-1].plot(num_iter, mean, label=names[i])

                axs[damage_num-1].fill_between(num_iter, rolling_max_mean-var, rolling_max_mean+var, alpha=0.4)
                axs2[damage_num-1].fill_between(num_iter, mean-var, mean+var, alpha=0.4)

            axs[damage_num-1].set_title(f"N={damage_num}")
            axs2[damage_num-1].set_title(f"N={damage_num}")

            if damage_num == num_damages_in_dict:
                handles, labels = axs[damage_num-1].get_legend_handles_labels()
            
        fig.text(0.5, 0.05, 'Adaptation steps', ha='center')
        fig.text(0.05, 0.5, 'Maximum fitness observed', va='center', rotation='vertical')

        fig2.text(0.5, 0.05, 'Adaptation steps', ha='center')
        fig2.text(0.05, 0.5, 'Mean fitness', va='center', rotation='vertical')

        fig.legend(handles, labels, loc='upper right')
        fig2.legend(handles, labels, loc='upper right')
                
        if save_fig: fig.savefig(f"{path_to_res}/per_damage_max_vs_num_iter.png", dpi=600)
        if save_fig: fig2.savefig(f"{path_to_res}/per_damage_mean_vs_num_iter.png", dpi=600)

        
        

        



# TODO: you may get issues because of the nan values, make sure to only save up till counter got from now on in ITE.
#  Also don't have to impose any condition when saving values, let ITE run forever and see how far it gets.

if __name__ == "__main__":
    visu = Visualiser()

    # LOAD THE ITE RESULTS
    ite_fits = jnp.array([jnp.load("families/family_3/ancestors/damaged_0_1_2_3_4/y_observed.npy"),
                jnp.load("families/family_3/ancestors/damaged_0_1_2_3_5/y_observed.npy"),
                jnp.load("families/family_3/ancestors/damaged_0_1_2_4_5/y_observed.npy"),
                jnp.load("families/family_3/ancestors/damaged_0_1_3_4_5/y_observed.npy"),
                jnp.load("families/family_3/ancestors/damaged_1_2_3_4_5/y_observed.npy")])

    gpcf_fits = jnp.array([jnp.load("families/family_3/new_agents/damaged_0_1_2_3_4/GPCF/y_observed.npy"),
                jnp.load("families/family_3/new_agents/damaged_0_1_2_3_5/GPCF/y_observed.npy"),
                jnp.load("families/family_3/new_agents/damaged_0_1_2_4_5/GPCF/y_observed.npy"),
                jnp.load("families/family_3/new_agents/damaged_0_1_3_4_5/GPCF/y_observed.npy"),
                jnp.load("families/family_3/new_agents/damaged_1_2_3_4_5/GPCF/y_observed.npy")])
    
    # print(f"ite_fits: {ite_fits}")
    # print(f"gpcf_fits: {gpcf_fits}")
    
    ite_mean = jnp.nanmean(ite_fits, axis=0)

    gpcf_mean = jnp.nanmean(gpcf_fits, axis=0)

    # print(gpcf_mean)
    
    ite_var = jnp.nanvar(ite_fits, axis=0)
    gpcf_var = jnp.nanvar(gpcf_fits, axis=0)

    # LOAD THE GPCF RESULTS
    means = [ite_mean, gpcf_mean]
    vars = [ite_var, gpcf_var]

    plot_names = ["ITE", "GPCF"]

    visu.get_mean_and_var_plot(means=means, vars=vars, names=plot_names)

