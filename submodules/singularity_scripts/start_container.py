#!/usr/bin/env python3

import argparse
import glob
import os
import subprocess
import tempfile
from typing import List

import build_final_image


GIT_EXP_PATH = "git/exp/"
SFERES_EXP_PATH = "git/sferes/exp/"
GIT_PATH = "git/"

DEFAULT_BINDING_FOLDER_PARENT = GIT_EXP_PATH
ABSOLUTE_DEFAULT_BINDING_FOLDER_PARENT = os.path.join("/", GIT_EXP_PATH)

SOURCE_BINDING_FOLDER_PARENTS = [
    GIT_EXP_PATH,
    SFERES_EXP_PATH,
    GIT_PATH,
]

ABSOLUTE_SOURCE_BINDING_FOLDERS = [
    os.path.join("/", path)
    for path in SOURCE_BINDING_FOLDER_PARENTS
]


def get_default_image_name():
    return f"{build_final_image.get_project_folder_name()}.sif"


def get_default_path_project_in_container() -> str:
    return os.path.join(ABSOLUTE_DEFAULT_BINDING_FOLDER_PARENT, build_final_image.get_project_folder_name())


def build_sandbox(path_singularity_def: str,
                  image_name: str):
    # check if the sandbox has already been created
    if os.path.exists(image_name):
        return

    print(f"{image_name} does not exist, building it now from {path_singularity_def}")
    assert os.path.exists(path_singularity_def)  # exit if path_singularity_definition_file is not found

    # run commands
    command = f"singularity build --force --fakeroot --sandbox {image_name} {path_singularity_def}"
    subprocess.run(command.split())


def _get_binding_options(image_name: str, binding_folder_parent_from_inside_container: str) -> List[str]:
    path_binding_folder_from_host = os.path.join(image_name, binding_folder_parent_from_inside_container)
    if not os.path.exists(path_binding_folder_from_host):
        return []

    list_subfolders = next(os.walk(path_binding_folder_from_host))[1]
    list_possible_binding_options = [f"    --binding-folder {os.path.join('/', binding_folder_parent_from_inside_container, subfolder)}"
                                     for subfolder in list_subfolders]
    return list_possible_binding_options


def _get_binding_folder_inside_container(image_name: str):
    list_valid_binding_options = []

    for binding_folder_parent in SOURCE_BINDING_FOLDER_PARENTS:
        path_binding_folder_from_outside_image = os.path.join(image_name, binding_folder_parent, build_final_image.get_project_folder_name())
        if os.path.exists(path_binding_folder_from_outside_image):
            list_valid_binding_options.append(os.path.join("/", binding_folder_parent, build_final_image.get_project_folder_name()))
    
    if len(list_valid_binding_options) == 1:

        return list_valid_binding_options[0]
    elif len(list_valid_binding_options) == 0:
        default_path_project_in_container = get_default_path_project_in_container()

        list_possible_binding_options = []
        for binding_folder_parent in SOURCE_BINDING_FOLDER_PARENTS:
            list_possible_binding_options.extend(_get_binding_options(image_name, binding_folder_parent))

        list_possible_binding_options.append(f"    --binding-folder <other_path>")

        build_final_image.error_print(
            f"Warning: We did not find any folder to bind in the container. "
            f"The Binding between your project folder and your container is likely to be unsuccessful.\n"
            f"You may want to consider adding one of the following options to the 'start_container' command:\n"
            + '\n'.join(list_possible_binding_options))

        return default_path_project_in_container

    else:  # len(list_valid_binding_options) > 1
        list_possible_binding_options = [
            f"    --binding-folder {valid_binding_option}"
            for valid_binding_option in list_valid_binding_options]

        build_final_image.error_print(
            f"Warning: We found multiple folders to bind in the container. "
            f"The Binding between your project folder and your container may be unsuccessful.\n"
            f"You may want to consider adding one of the following options to the 'start_container' command "
            f"(the first option shown is the one that is considered):\n"
            + '\n'.join(list_possible_binding_options))

        return list_valid_binding_options[0]


def run_container(nvidia: bool,
                  use_no_home: bool,
                  use_tmp_home: bool,
                  image_name: str,
                  path_binding_folder_inside_container: str):
    additional_args = ""

    if nvidia:
        print("Nvidia runtime ON")
        additional_args += " " + "--nv"

    if use_no_home:
        print("Using --no-home")
        additional_args += " " + "--no-home"

    if use_tmp_home:
        tmp_home_folder = tempfile.mkdtemp(dir="/tmp")
        additional_args += " " + f"--home {tmp_home_folder}"
        build_final_image.error_print(f"Warning: The HOME folder is a temporary directory located in {tmp_home_folder}! "
                                      f"Do not store any result there!")

    if not path_binding_folder_inside_container:
        path_binding_folder_inside_container = _get_binding_folder_inside_container(image_name)

    print(f"Starting container {build_final_image.bold(image_name)} "
          f"with binding {build_final_image.bold(path_binding_folder_inside_container)}")

    command = f"singularity shell -w {additional_args} " \
              f"--bind {os.path.dirname(os.getcwd())}:{path_binding_folder_inside_container} " \
              f"{image_name}"
    subprocess.run(command.split())


def get_args():
    parser = argparse.ArgumentParser(description='Build a sandbox container and shell into it.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--nv', action='store_true', help='enable experimental Nvidia support')
    parser.add_argument('--no-home', action='store_true', help='apply --no-home to "singularity shell"')
    parser.add_argument('--tmp-home', action='store_true', help="binds HOME directory of the singularity container to a temporary folder")

    parser.add_argument('--path-def', required=False, type=str,
                        default=build_final_image.SINGULARITY_DEFINITION_FILE_NAME,
                        help='path to singularity definition file')

    parser.add_argument('--personal-token', required=False, type=str, default=build_final_image.get_personal_token(),
                        help='Gitlab Personal token. '
                             'If not specified, it takes the value of the environment variable PERSONAL_TOKEN, '
                             'if it exists. '
                             'If the environment variable SINGULARITYENV_PERSONAL_TOKEN is not set yet, '
                             'then it is set the value provided.')

    parser.add_argument('-b', '--binding-folder', required=False, type=str,
                        default=None,
                        help=f'If specified, it corresponds to the path of the folder from which the binding '
                             f'is performed to the current project source code. '
                             f'By default, it corresponds to the image name (without the .sif extension)')

    parser.add_argument('-i', '--image', required=False, type=str,
                        default=get_default_image_name(),
                        help='name of the sandbox image to start')

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    enable_nvidia_support = args.nv
    use_no_home = args.no_home
    use_tmp_home = args.tmp_home
    path_singularity_definition_file = args.path_def
    image_name = args.image
    binding_folder_inside_container = args.binding_folder
    personal_token = args.personal_token

    # Create environment variables for singularity
    build_final_image.generate_singularity_environment_variables(ci_job_token=None,
                                                                 personal_token=personal_token,
                                                                 project_folder=binding_folder_inside_container)

    build_sandbox(path_singularity_definition_file, image_name)
    run_container(enable_nvidia_support, use_no_home, use_tmp_home, image_name, binding_folder_inside_container)


if __name__ == "__main__":
    main()
