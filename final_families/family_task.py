from src.core.pre_adaptation.task import Task

task = Task(episode_length=150, num_iterations=40000, grid_shape=tuple([3]*6))
children_in_ancestors = True
algo_num_iter = 20