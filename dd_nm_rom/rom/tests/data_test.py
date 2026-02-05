"""
Train interior and interface autoencoders for DD-NM-ROM.
"""

import sys
import json
import argparse
import os

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

# Environment
# =====================================
from dd_nm_rom import env
env.set(**inputs["env"])

# Libraries
# =====================================
import shutil
import numpy as np

from dd_nm_rom import utils
from dd_nm_rom import fom as fom_mod
from dd_nm_rom import field as field_mod
from dd_nm_rom.elements import mesh as mesh_mod
from dd_nm_rom.rom.nonlinear import Autoencoder, Data, Model
from dd_nm_rom import backend as bkd

import torch
import torch.distributed as dist

# Initialization
# =====================================
print("\nInitialization ...")
# Mesh
mesh = utils.get_class(modules=[mesh_mod], **inputs["mesh"])
mesh.build()
# Field
field = utils.get_class(
  modules=[field_mod],
  name=inputs["field"]["name"]
)(mesh=mesh, **inputs["field"]["kwargs"])
field.set_params(mu=field.sample_design_space())
# FOM
fom = utils.get_class(
  modules=[fom_mod],
  name="Burgers2D"
)(mesh=mesh, **inputs["fom"]["kwargs"])
fom.build(field)
# DD-FOM
dd_fom = utils.get_class(
  modules=[fom_mod],
  name="DDBurgers2D"
)(monolithic=fom, constraint_type="strong")
dd_fom.build()

# Data Loading
# =====================================
print("\nLoading data ...")

test_dir = os.getcwd() + "/test_data_outputs"
print(test_dir)
os.makedirs(test_dir, exist_ok=True)

ndof = dd_fom.get_ndof()
print(" NDOF = {}".format(ndof))
ndof = 8

nsamples = 32
nepochs = 10

scale = 1.0 / ndof
shift = 1.0e-3
dataset = []
#if not bkd.distributed() or (bkd.distributed() and bkd.get_rank() == 0):

ind_start = 0
ind_end = nsamples
rank = 0

if bkd.distributed():
  rank = bkd.get_rank()
  npr = (nsamples // bkd.get_nranks())
  ind_start = npr * bkd.get_rank()
  ind_end = ind_start + npr
  print(" RANK {}: samples from {} to {}".format(bkd.get_rank(), ind_start, ind_end))

for s in range(ind_start, ind_end):
  start = s
  end = s + scale*ndof
  dataset.append(np.linspace(start, end, num=ndof))

if bkd.distributed():
  print(" RANK {}: samples = {}".format(bkd.get_rank(), dataset))
else:
  print(dataset)

if bkd.distributed():
    bkd._COMM.Barrier()
    dist.barrier()

np.savetxt("{}/test_data_init_data{}.txt".format(test_dir, rank), bkd.to_numpy(dataset))

data = Data(
  snapshots=dataset,
  validation_split=0.25,
  batch_size=1024
)

np.savetxt("{}/test_data_init_snapshots{}.txt".format(test_dir, rank), bkd.to_numpy(data.snapshots))
np.savetxt("{}/test_data_init_train{}.txt".format(test_dir, rank), bkd.to_numpy(data.train))
np.savetxt("{}/test_data_init_valid{}.txt".format(test_dir, rank), bkd.to_numpy(data.valid))

for epoch in range(nepochs):
    data.on_epoch_begin()

    np.savetxt("{}/test_data_epoch{}_snapshots{}.txt".format(test_dir, epoch, rank), bkd.to_numpy(data.snapshots))
    np.savetxt("{}/test_data_epoch{}_train{}.txt".format(test_dir, epoch, rank), bkd.to_numpy(data.train))
    np.savetxt("{}/test_data_epoch{}_valid{}.txt".format(test_dir, epoch, rank), bkd.to_numpy(data.valid))
    batches_cpu = []
    batches_valid_cpu = []
    for batch in data.batches:
       batches_cpu.append(bkd.to_numpy(batch))
    for batch in data.batches_valid:
       batches_valid_cpu.append(bkd.to_numpy(batch))
    np.savetxt("{}/test_data_epoch{}_batches{}.txt".format(test_dir, epoch, rank), np.array(batches_cpu).squeeze())
    np.savetxt("{}/test_data_epoch{}_batches_valid{}.txt".format(test_dir, epoch, rank), np.array(batches_valid_cpu).squeeze())
    
    if bkd.distributed():
        bkd._COMM.Barrier()
        dist.barrier()

if bkd.distributed():
    bkd._COMM.Barrier()
    dist.barrier()

    bkd.finalize_distributed()
