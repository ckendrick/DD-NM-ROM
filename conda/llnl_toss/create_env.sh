#!/bin/bash -i

# =====================================
# Deep Learning @ LLNL
# =====================================
# The present script installs PyTorch on LC TOSS systems

# Inputs
# -------------------------------------
env_file=env.yml
anaconda_dir=/collab/usr/gapps/python/${SYS_TYPE}/anaconda3-2024.02
# -------------------------------------

# Inputs for virtualenv (tuolumne)
venv_file=requirements.txt
venv_name=ddnmrom_venv
virtualenv_dir=/p/lustre5/${USER}/${venv_name}/

# check for machine type: toss_4_x86_64_ib = dane
#                         toss_4_x86_64_ib_cray = tuolumne
machine="${SYS_TYPE:-toss_4_x86_64_ib}"

if [[ "${machine}" == "toss_4_x86_64_ib" ]] ;
then
    # Dane
    if [ ! -d ${anaconda_dir} ]; then
	# Create installation directory
	mkdir -p ${anaconda_dir} && rm -rf ${anaconda_dir}
	# Install conda
	bash /collab/usr/gapps/python/${SYS_TYPE}/conda/Miniconda3-py39_4.12.0-Linux-x86_64.sh -b -f -p ${anaconda_dir}
    fi

    source ${anaconda_dir}/bin/activate
    conda env create -f ${env_file}
else
    # Tuolumne
    if [ ! -d ${virtualenv_dir} ]; then
	# Create virtualenv
	module load python/3.11.5 rocm/6.3.1

	python3 -m venv ${virtualenv_dir}

	# store directory name for later
	SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
	echo $SCRIPT_DIR
	echo ${virtualenv_dir} >> ${SCRIPT_DIR}/venv_path.txt

	source ${virtualenv_dir}/bin/activate

	pip install -r ${SCRIPT_DIR}/${venv_file}
    fi

    source ${virtualenv_dir}/bin/activate
fi
