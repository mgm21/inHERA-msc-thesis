from src.utils.all_imports import *
from src.core.task import Task

task = Task(episode_length=150, num_iterations=20000, grid_shape=tuple([3]*6))
children_in_ancestors = True
algo_num_iter = 20