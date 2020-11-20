#!/bin/bash

_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# This script is for preparing the releases, conda environments, etc.


DEFAULT_CONDA_PATH="$HOME/opt/anaconda3/etc/profile.d/conda.sh"


function prepare_conda_env_cpu() {
        local conda_path=${1:-$DEFAULT_CONDA_PATH}; shift
        source ${conda_path}
        conda env remove --name roosterize-cpu
        conda create --name roosterize-cpu python=3.7 pip -y
        conda activate roosterize-cpu
        conda install -y pytorch-cpu==1.1.0 torchvision-cpu==0.3.0 cpuonly -c pytorch
        pip install -r requirements.txt
        conda env export --no-builds > conda-envs/cpu.yml
}


function prepare_conda_env_gpu_cuda10() {
        local conda_path=${1:-$DEFAULT_CONDA_PATH}; shift
        source ${conda_path}
        conda env remove --name roosterize-gpu
        conda create --name roosterize-gpu python=3.7 pip -y
        conda activate roosterize-gpu
        conda install -y pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
        pip install -r requirements.txt
        conda env export --no-builds > conda-envs/gpu-cuda10.yml
}


function prepare_conda_env_gpu_cuda9() {
        local conda_path=${1:-$DEFAULT_CONDA_PATH}; shift
        source ${conda_path}
        conda env remove --name roosterize-gpu
        conda create --name roosterize-gpu python=3.7 pip -y
        conda activate roosterize-gpu
        conda install -y pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0 -c pytorch
        pip install -r requirements.txt
        conda env export --no-builds > conda-envs/gpu-cuda9.yml
}


function prepare_conda_env_mac() {
        # This needs to be executed on a Mac
        local conda_path=${1:-$DEFAULT_CONDA_PATH}; shift
        source ${conda_path}
        conda env remove --name roosterize-mac
        conda create --name roosterize-mac python=3.7 pip -y
        conda activate roosterize-mac
        conda install -y pytorch==1.1.0 torchvision==0.3.0 -c pytorch
        pip install -r requirements.txt
        conda env export --no-builds > conda-envs/mac-cpu.yml
}


function package_dist() {
        # The package environment is special: we still use conda as a
        # virtual environment, but everything should be installed
        # using pip otherwise PyInstaller will not recognize.
        #
        # requirements.txt contains the right CPU-only pytorch
        # packages for installation
        local conda_path=${1:-$DEFAULT_CONDA_PATH}; shift
        source ${conda_path}
        conda env remove --name roosterize-package
        conda create --name roosterize-package python=3.7 pip -y
        conda activate roosterize-package
        pip install -r requirements.txt
        pip install pyinstaller==4.1
        make clean package
        # Package output will be at dist/roosterize.tgz
}


# ==========
# Main function -- program entry point
# This script can be executed as ./run.sh the_function_to_run

function main() {
        local action=${1:?Need Argument}; shift

        ( cd ${_DIR}
          $action "$@"
        )
}

main "$@"
