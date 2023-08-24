from src.utils.all_imports import *

repertoires_path = "./dummy_repertoires"
families_path = "./dummy_families"

for file_name in os.listdir(repertoires_path):
    d = os.path.join(repertoires_path, file_name)
    if os.path.isdir(d):
        # family_path = f"{families_path}/family-{file_name}"
        # os.mkdir(path=family_path)
        shutil.copytree(f"{repertoires_path}/{file_name}", f"{families_path}/family-{file_name}")