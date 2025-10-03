#!/bin/bash -i

### Shell scripting
### ---------------
### Loading conda environment thanks to interactive shell
### > See: 'dd-nm-rom/conda/README.md' file
machine="${SYS_TYPE:-toss_4_x86_64_ib}"

if [[ "${machine}" == "toss_4_x86_64_ib" ]] ;
then
    # Dane
    load_conda_env_toss
else
    # Tuolumne
    #source ddnmrom_env/bin/activate
    # todo; assumes this is launched from the commands/ folder
    venv_dir=$(cat ./../../../conda/llnl_toss/venv_path.txt)
    echo $venv_dir

    source $venv_dir/bin/activate

    export MPICH_GPU_SUPPORT_ENABLED=1
    export HSA_XNACK=1
fi

python -u main.py --inpfile inputs.json
