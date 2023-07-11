This repository contains several scripts that simplify the interactions with singularity containers for our experiments.

## Install

To install this submodules in your project:
1) Run the following commands:
```
mkdir -p submodules
git submodule add ../../../AIRL_tools/singularity_scripts ./submodules/singularity_scripts
```

The url is relative to the url of your project, so here we assumed that you project is for instance located in `/Students_projects/My_Self/My_Project`
You will now have a folder named `submodules/singularity_scripts` containing all the necessary scripts.

2) You can add in your singularity folder  symbolic links for a simplified access to the scripts of this submodules:
```
cd ./singularity/
ln -s ../submodules/singularity_scripts/start_container ./start_container
ln -s ../submodules/singularity_scripts/build_final_image ./build_final_image
```

------

## Description

There are 2 singularity scripts:
- `start_container`
- `build_final_image`

Both of them **have to be executed in your `singularity/` folder**.

### start_container

The `start_container` script builds a sandbox and shells into it.
If the sandbox already exists, we directly shell into it.

The script has several additional arguments. 
You can use the help for more information:
```bash
./start_container --help
```

### build_final_image

The `build_final_image` script builds a read-only final container in which the entire project repository is cloned.

In particular, it parses the singularity definition file, looking for those tags:
- `#NOTFORFINAL` 
- `#CLONEHERE`

#### `#NOTFORFINAL`

All the lines with `#NOTFORFINAL` are skipped and not taken into account. 
Thus, if you add the line in your `%post` section of your singularity definition file:
```bash
#====================================================================================================
exit 0 #NOTFORFINAL - the lines below this "exit" will be executed only when building the final image
#====================================================================================================
```
then:
- the script `start_container` will only take into account the lines **preceding** the tag `#NOTFORFINAL`
- the script `build_final_image` will take into account all the lines **preceding** and **after** the tag `#NOTFORFINAL`

#### `#CLONEHERE`

All the lines with the tag `#CLONEHERE` are replaced with a command that is cloning the project repository to the appropriate commit.
Technically, that resulting command executes the following operations:
```bash
git clone --recurse-submodules --shallow-submodules <repo_address_with_token> <project_name>
cd <project_name>
git checkout <commit_sha>
cd ..  
```

- If a Gitlab CI Job Token is provided, then `<repo_address_with_token>` is `http://gitlab-ci-token:<ci_job_token>@<repo_address>`.
In that case **no password needs to be entered to clone the repository**.
- If a Gitlab Personal Token is provided, then `<repo_address_with_token>` is `https://oauth:<personal_token>@<repo_address>`.
In that case **no password needs to be entered to clone the repository**.
- If no token is provided, then `<repo_address_with_token>` is simply `https://<repo_address>`.
In that case **a password will be required to clone the repository**.

The `<project_name>`, `<commit_sha>`, `<ci_job_token>` and `<personal_token>` can be specified in the arguments of the `build_final_image` script.

For more information about the arguments of the `build_final_image` script, you can use the help:
```bash
./build_final_image --help
```

#### Specify a Commit

When using the tag `#CLONEHERE` you can specify any reference to the commit you desire to clone specifically.
The `HEAD` is used by default.
**Warning: the commit mentioned should have been pushed before calling the `./build_final_image` script.**

##### Examples:
- Commit reference to clone: *last commit on branch develop*
```bash
./build_final_image --personal-token xxxxxxxx --commit-ref develop
```
- Commit reference to clone: *tag v1.2*
```bash
./build_final_image --personal-token xxxxxxxx --commit-ref v1.2
```
- Commit reference to clone: *current HEAD*
```bash
./build_final_image --personal-token xxxxxxxx
```
- Commit reference to clone: *commit preceding current HEAD*
```bash
./build_final_image --personal-token xxxxxxxx --commit-ref HEAD~1
```

------

## Quick tips with git submodules

- To update the gitlab_notebook tools in your project, run `git submodule update`
- To clone a project with the submodules already configures: `git clone --recurse-submodules https://your_url.git`