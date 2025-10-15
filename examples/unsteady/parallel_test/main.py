"""
Launch multiple testing jobs for DD-NM-ROM.
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
def generate_batch_script_toss(tag, inpfile):
    return f"""#!/bin/bash -i                                                                         
                                                                         
### Slurm syntax                                                                        
### ---------------                                                                     
#SBATCH -N 1                                 #number of nodes          
#SBATCH -t 24:00:00                          #walltime in hours:minutes 
#SBATCH -e test_dd_nmrom_{tag}_err.txt       #stderr                    
#SBATCH -o test_dd_nmrom_{tag}_out.txt       #stdout                    
#SBATCH -J test_dd_nmrom_{tag}               #name of job              
#SBATCH -p pbatch                            #queue to use              
#SBATCH -A sosu                              #account                   
                                                                         
### Shell scripting                                                                     
### ---------------                                                                     
### Loading conda environment thanks to interactive shell                               
### > See: 'dd-nm-rom/conda/README.md' file                                             
load_conda_env_toss                                                                      
### Launch program                                                                      
python -u ./../scripts/test_dd_nmrom.py --inpfile {inpfile}           
"""

def generate_batch_script_coral(tag, inpfile):
    return f"""#!/bin/bash -i                                                                         
                                                                         
### LSF syntax                                                                          
### ---------------                                                                     
#BSUB -nnodes 1                              #number of nodes            
#BSUB -W 12:00                               #walltime in hours:minutes   
#BSUB -e test_dd_nmrom_{tag}_err.txt         #stderr                      
#BSUB -o test_dd_nmrom_{tag}_out.txt         #stdout                      
#BSUB -J test_dd_nmrom_{tag}                 #name of job                
#BSUB -q pbatch                              #queue to use                
#BSUB -G sosu                                #account                     
                                                                         
### Shell scripting                                                                     
### ---------------                                                                     
### Loading conda environment thanks to interactive shell                               
### > See: 'dd-nm-rom/conda/README.md' file                                             
load_conda_env_coral                                                                     
### Launch program                                                                      
python -u ./../scripts/test_dd_nmrom.py --inpfile {inpfile}           
"""

def generate_batch_script_tuo(tag, inpfile):
    return f"""#!/bin/bash
#flux: -N 1
#flux: -n 1
#flux: -c 8
#flux: -o gpu-affinity=off
#flux: -o mpibind=verbose:1
#flux: -u
#flux: --setattr=thp=always
#flux: --error=test_dd_nmrom_{tag}_err.txt
#flux: --job-name=test_dd_nmrom_{tag}
#flux: --output=test_dd_nmrom_{tag}_out.txt

### Shell scripting 
### ---------------
### Loading conda environment thanks to interactive shell
### > See: 'dd-nm-rom/conda/README.md' file
#source ddnmrom_env/bin/activate
# todo; assumes this is launched from the commands/ folder
venv_dir=$(cat ./../../../conda/llnl_toss/venv_path.txt)
echo $venv_dir

echo "Activating venv.."
source $venv_dir/bin/activate
echo "Done activating venv"

export MPICH_GPU_SUPPORT_ENABLED=1
export HSA_XNACK=1

### Launch program
python -u ./../scripts/test_dd_nmrom.py --inpfile {inpfile}
"""

if (inputs["system"] == "coral"):
  generate_batch_script = generate_batch_script_coral
  batch_cmd = lambda cmdfile: f"bsub < {cmdfile}"
elif (inputs["system"] == "toss"):
  generate_batch_script = generate_batch_script_toss
  batch_cmd = lambda cmdfile: f"sbatch {cmdfile}"
elif (inputs["system"] == "tuo"):
  generate_batch_script = generate_batch_script_tuo
  batch_cmd = lambda cmdfile: f"flux batch --flags waitable {cmdfile}"
else:
  raise ValueError("System not valid.")

# Generate all configurations
# -------------------------------------
dims, cfgs = {}, []
for element in inputs["elements"]:
  dims[element] = ops.generate_combs([
    np.arange(**inputs["dim"]["ranges"][element]["latent_dim"]),
    np.arange(**inputs["dim"]["ranges"][element]["row_nonzero"])
  ])
  cfgs.append(np.arange(len(dims[element])))
cfgs = ops.generate_combs(cfgs)

# Loop over configurations
# -------------------------------------
n_jobs = 0
for cfg in cfgs:
  # > Set tag
  tag_i = {}
  text = "\nLaunching testing for elements:"
  for (e, element) in enumerate(inputs["elements"]):
    ld, rnz = dims[element][cfg[e]]
    tag_i[element] = f"{element}_ld_{ld}_rnz_{rnz}"
    text += f"\n- '{element}' with (ld, rnz) = ({ld}, {rnz})"
  fulltag_i = "_".join(tag_i.values())
  print(text)
  # > Update input file
  with open(inputs["paths"]["inpfile"]) as file:
    inp_i = json.load(file)
  for (element, etag) in tag_i.items():
    inp_i["paths"]["nets_tag"][element] = etag
  # > Save input file
  inpfile_i = inputs["paths"]["inp_dir"] + f'/test_dd_nmrom_{fulltag_i}.json'
  with open(inpfile_i, 'w') as file:
    json.dump(inp_i, file, indent=2)
  # Batch script
  # -------------
  cmdfile_i = inputs["paths"]["cmd_dir"] + f'/test_dd_nmrom_{fulltag_i}.sh'
  with open(cmdfile_i, 'w') as file:
    file.write(generate_batch_script(fulltag_i, inpfile_i))
  # Launch program
  # -------------
  subprocess.run(
    batch_cmd(cmdfile_i),
    shell=True,
    timeout=1e2,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT
  )
  n_jobs += 1

print(f"\nTotal number of jobs: {n_jobs}\n")
