from src.utils.all_imports import *

# Copy the content of one directory to another directory dynamically
folders = ["damaged_1_2_3", "damaged_3_4", "intact"]
for folder in folders:
    for i in range(1, 21):
        copy_tree(f"results/numiter40k_inhera_b0_children/family-seed_{i}_repertoire/{folder}/", f"results/numiter40k_final_children_with_intact_ancestor_and_intact_child/family-seed_{i}_repertoire/{folder}")

