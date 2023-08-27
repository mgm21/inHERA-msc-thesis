from src.utils.all_imports import *

# Copy the content of one directory to another directory dynamically
for i in range(0, 21):
    copy_tree(f"numiter40k_ancestors/family-seed_{i}_last_repertoire/ancestors", f"numiter40k_final_families/family-seed_{i}_last_repertoire/ancestors")

