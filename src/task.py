# Note: could change Task name, just gave it because Environment would lead to a confusion with brax/qdax envs
from utils.all_imports import *
from dataclasses import dataclass

@dataclass
class Task:
    batch_size: int = 128
    env_name: str = "hexapod_uni"
    episode_length: int = 100
    num_iterations: int = 100
    seed: int = 42
    policy_hidden_layer_sizes: tuple = (64, 64)
    iso_sigma: float = 0.005
    line_sigma: float = 0.05
    min_bd: float = 0.
    max_bd: float = 1.
    grid_shape: tuple = tuple([3]) * 6

if __name__ == "__main__":
    task = Task()