from utils.all_imports import *

class Visualiser:
    def plot_repertoire(self, agent, quantity, path_to_base_repertoire, path_to_save_to):

        repertoire = MapElitesRepertoire.load(reconstruction_fn=agent.recons_fn, path=path_to_base_repertoire)

        if quantity == "mu":
            updated_fitness = agent.mu
        elif quantity == "var":
            updated_fitness = agent.var

        fig, _ = plot_multidimensional_map_elites_grid(
        repertoire=repertoire,
        updated_fitness=updated_fitness,
        maxval=jnp.asarray([agent.task.max_bd]),
        minval=jnp.asarray([agent.task.min_bd]),
        grid_shape=tuple(agent.task.grid_shape),)

        fig.savefig(path_to_save_to)