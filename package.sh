#!/bin/bash

_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# This script is for preparing the releases, conda environments, etc.


CONDA_PATH="$HOME/opt/anaconda3/etc/profile.d/conda.sh"


function prepare_conda_env_cpu() {
        source ${CONDA_PATH}
        conda env remove --name roosterize-cpu
        conda create --name roosterize-cpu python=3.7 pip -y
        conda activate roosterize-cpu
        conda install -y pytorch-cpu==1.1.0 torchvision-cpu==0.3.0 cpuonly -c pytorch
        pip install -r requirements.txt
        conda env export --no-builds > conda-envs/cpu.yml
}


function prepare_conda_env_gpu_cuda10() {
        source ${CONDA_PATH}
        conda env remove --name roosterize-gpu
        conda create --name roosterize-gpu python=3.7 pip -y
        conda activate roosterize-gpu
        conda install -y pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
        pip install -r requirements.txt
        conda env export --no-builds > conda-envs/gpu-cuda10.yml
}


function prepare_conda_env_gpu_cuda9() {
        source ${CONDA_PATH}
        conda env remove --name roosterize-gpu
        conda create --name roosterize-gpu python=3.7 pip -y
        conda activate roosterize-gpu
        conda install -y pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0 -c pytorch
        pip install -r requirements.txt
        conda env export --no-builds > conda-envs/gpu-cuda9.yml
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
