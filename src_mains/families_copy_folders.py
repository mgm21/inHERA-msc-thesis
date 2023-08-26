from src.utils.all_imports import *

for i in range(0, 21):
    copy_tree(f"results/family-seed_{i}_last_repertoire/ancestors", f"4knumiter_final_families/family-seed_{i}_last_repertoire/ancestors")