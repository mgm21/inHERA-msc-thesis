from src.utils.all_imports import *
from src.adaptation_algorithms.gpcf_variant import GPCFVariant

class GPCF1Trust(GPCFVariant):
    """GPCF algorithm which looks for the closest ancestor at each turn based solely on the last observation"""
    def __init__(self, family, agent, gaussian_process, alpha=0.9, verbose=False,
                 path_to_results="families/ite_example/", save_res_arrs=True, norm_params=(0, 40), plot_repertoires=False):
        
        super().__init__(family, agent, gaussian_process, alpha, verbose,
                 path_to_results, save_res_arrs, norm_params, plot_repertoires)
    
    def get_ancestor_weights(self, counter):
        # Look for the single closest ancestor
        ancestor_mus_at_curr_obs = self.ancestor_mus_at_obs
        diffs = jnp.abs(ancestor_mus_at_curr_obs[:, counter] - jnp.repeat(a=self.agent.y_observed[counter], repeats=ancestor_mus_at_curr_obs.shape[0]))
        closest_ancest_idx = jnp.nanargmin(diffs)
        W = jnp.zeros(shape=ancestor_mus_at_curr_obs.shape[0])
        W = W.at[closest_ancest_idx].set(1)

        if self.verbose: print(f"GPCF1Trust's weights: {W}")
        return W

if __name__ == "__main__":
    from src.core.family import Family
    from src.core.adaptive_agent import AdaptiveAgent
    from src.core.task import Task
    from src.utils import hexapod_damage_dicts
    from src.core.gaussian_process import GaussianProcess
    from src.loaders.repertoire_loader import RepertoireLoader

    path_to_family = "families/family_4"
    from families.family_4 import family_task
    fam = Family(path_to_family=path_to_family)
    task = family_task.task
    norm_params = jnp.load(f"{path_to_family}/norm_params.npy")
    damage_dict = hexapod_damage_dicts.leg_0_broken
    rep_loader = RepertoireLoader()
    simu_arrs = rep_loader.load_repertoire(repertoire_path=f"{path_to_family}/repertoire",)
    agent = AdaptiveAgent(task=task,
                      sim_repertoire_arrays=simu_arrs,
                      damage_dictionary=damage_dict)
    gp = GaussianProcess()

    gpcf1t = GPCF1Trust(family=fam,
                        agent=agent,
                        gaussian_process=gp,
                        verbose=True,
                        norm_params=norm_params,
                        save_res_arrs=True)
    gpcf1t.run(num_iter=5)


# TODO: only odd thing is that when running ignore-gpcf (which is gpcf_1trust basically, results have more decimals
# on the dummy leg_0_broken problem... probably nothing but if facing problems, try to reproduce this by running the
# above on both and inspecting the output of mu @ x_observed)