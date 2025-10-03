"""
Launch multiple training jobs for DD-NM-ROM.
"""

import os
import sys
import json
import argparse

# Inputs
# =====================================
parser = argparse.ArgumentParser()
parser.add_argument("--inpfile", type=str, help="path to JSON input file")
args = parser.parse_args()

with open(args.inpfile) as file:
  inputs = json.load(file)

# Import 'dd_nm_rom' package
# =====================================
with open(inputs["pathfile"]) as file:
  paths = file.read().splitlines()
sys.path.extend(paths)

# Libraries
# =====================================
import copy
import subprocess
import numpy as np

from dd_nm_rom import ops

# Initialization
# =====================================
# Directories
for path in (
  inputs["paths"]["inp_dir"],
  inputs["paths"]["cmd_dir"]
):
  os.makedirs(path, exist_ok=True)

# Running
# =====================================
# Batch script function
# -------------------------------------
def generate_batch_script(tag, pyscript, inpfile):
    # Use triple quotes for a clean multiline string
    return f"""#!/bin/bash -i
#flux: -N 1
#flux: -q pbatch
#flux: -t 60
#flux: --exclusive
#flux: --setattr=thp=always
#flux: --error=train_rom_{tag}_err.txt
#flux: --job-name=train_rom_{tag}
#flux: --output=train_rom_{tag}_out.txt

### LSF syntax
### ---------------
#BSUB -nnodes 1                  #number of nodes
#BSUB -W 12:00                   #walltime in hours:minutes
#BSUB -e train_rom_{tag}_err.txt #stderr
#BSUB -o train_rom_{tag}_out.txt #stdout
#BSUB -J train_rom_{tag}         #name of job
#BSUB -q pbatch                  #queue to use
#BSUB -G sosu                    #account

### Shell scripting
### ---------------
### Loading conda environment thanks to interactive shell
### > See: 'dd-nm-rom/conda/README.md' file
machine="${{SYS_TYPE:-toss_4_x86_64_ib}}"

if [[ "${{machine}}" == "toss_4_x86_64_ib" ]] ;
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

python -u ./../../steady/scripts/{pyscript}.py --inpfile {inpfile}
"""

# Looping over trainable elements
# -------------------------------------
n_jobs = 0
for element in inputs["elements"]:
  edim = inputs["dim"]["ranges"][element]
  dims = ops.generate_combs([
    np.arange(**edim[k]) for k in ("latent_dim", "row_nonzero")
  ])
  for (ld, rnz) in dims:
    text = "Launching training for element "
    text += f"'{element}' with (ld, rnz) = ({ld}, {rnz}) ..."
    print(text)
    # Input file
    # -------------
    # > Set tag
    tag_i = f"{element}_ld_{ld}_rnz_{rnz}"
    # > Set dimensions
    dim_i = copy.deepcopy(inputs["dim"]["default"][element])
    dim_i["latent_dim"] = int(ld)
    dim_i["row_nonzero"] = int(rnz)
    # > Update file
    with open(inputs["paths"]["inpfile"][element]) as file:
      inp_i = json.load(file)
    inp_i["model"]["path"] = inputs["paths"]["save_dir"]+f"/{tag_i}/"
    inp_i["trainable"]["elements"] = [element]
    inp_i["autoencoder"]["dim"] = {element: dim_i}
    if inputs["refine"]["active"]:
      inp_i["autoencoder"]["refine"] = True
      inp_i["model"]["compile"]["lr"] = inputs["refine"]["lr"]
      tag_i += "_ref"
    # > Save file
    inpfile_i = inputs["paths"]["inp_dir"] + f'/train_rom_{tag_i}.json'
    with open(inpfile_i, 'w') as file:
      json.dump(inp_i, file, indent=2)
    # Python script
    # -------------
    pyscript_i = "train_rom"
    if (element == "port"):
      pyscript_i += "_port"
    # Batch script
    # -------------
    cmdfile_i = inputs["paths"]["cmd_dir"] + f'/train_rom_{tag_i}.sh'
    with open(cmdfile_i, 'w') as file:
      file.write(generate_batch_script(tag_i, pyscript_i, inpfile_i))
    # Launch program
    # -------------
    subprocess.run(
      f"flux batch {cmdfile_i}",
      shell=True,
      timeout=1e2,
      stdout=subprocess.DEVNULL,
      stderr=subprocess.STDOUT
    )
    n_jobs += 1

print(f"\nTotal number of jobs: {n_jobs}\n")
