import pytest

import numpy as np
import torch
import torch.distributed as dist
import torch.testing

from dd_nm_rom import backend as bkd
from dd_nm_rom.utils import parallel_print

def setup_module(module):
    bkd.set(backend="torch", device="cuda", seed=0)

def teardown_module(module):
    bkd.finalize_distributed()

def _generate_matrix(size):
    return torch.randn(size, device=bkd.device())

@pytest.mark.mpi_skip
def test_distributed_serial():
    assert bkd.distributed() == False

@pytest.mark.mpi(min_size=2)
def test_distributed_parallel():
    assert bkd.distributed() == True
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    assert comm.Get_size() == bkd.get_nranks()
    assert comm.Get_rank() == bkd.get_rank()
    if comm.Get_rank() == 0:
        assert bkd.root() == True
    else:
        assert bkd.root() == False

def _check_sizes(sizes, expected):
    if not bkd.distributed():
        assert sizes is not None
        assert len(sizes) == 1
        assert sizes[0] == expected
    else:
        assert sizes is not None
        assert len(sizes) == bkd.get_nranks()
        for rank in range(bkd.get_nranks()):
            assert sizes[rank] == expected


def test_get_local_sizes_np():
    x = np.random.randn(2, 4)

    sizes = bkd.get_local_sizes(x)
    if bkd.root():
        _check_sizes(sizes, 2)
    else:
        assert sizes is None
    
    sizes_dim1 = bkd.get_local_sizes(x, dim=1)
    if bkd.root():
        _check_sizes(sizes_dim1, 4)
    else:
        assert sizes_dim1 is None


def test_get_local_sizes():
    x = torch.randn((2, 4), device=bkd.device())

    sizes = bkd.get_local_sizes(x)
    if bkd.root():
        _check_sizes(sizes, 2)
    else:
        assert sizes is None
    
    sizes_dim1 = bkd.get_local_sizes(x, dim=1)
    if bkd.root():
        _check_sizes(sizes_dim1, 4)
    else:
        assert sizes_dim1 is None


def test_get_local_sizes_all():
    x = torch.randn((2, 4), device=bkd.device())

    sizes = bkd.get_local_sizes_all(x)
    _check_sizes(sizes, 2)
    
    sizes_dim1 = bkd.get_local_sizes_all(x, dim=1)
    _check_sizes(sizes_dim1, 4)


def test_gather_tensor_calc_sizes():
    local_size = 2
    global_size = local_size * bkd.get_nranks()
    x_local = torch.randn((local_size, 4), device=bkd.device())

    x_global = bkd.gather_tensor(x_local)
    #parallel_print(" RANK {}: x_local = {}".format(bkd.get_rank(), x_local))
    #parallel_print(" RANK {}: x_global (after gather) = {}".format(bkd.get_rank(), x_global))

    assert x_global is not None
    if bkd.root():
        assert x_global.shape[0] == global_size
        assert x_global.shape[1] == 4
    else:
        torch.testing.assert_close(x_local, x_global)


def test_scatter_tensor_allocate():
    local_size = 2
    global_size = local_size * bkd.get_nranks()
    x = None
    if bkd.root():
        x = torch.randn((global_size, 4), device=bkd.device())
        #print(" ROOT x to scatter = {}".format(x))

    x_rank = bkd.scatter_tensor(x, x_out=None)
    #parallel_print(" RANK {}: x_rank = {}".format(bkd.get_rank(), x_rank))
    assert x_rank is not None
    assert x_rank.shape[0] == local_size
    assert x_rank.shape[1] == 4

    if not bkd.root():
        x = torch.empty((global_size, 4), device=bkd.device())
    dist.broadcast(x, src=0)
    x = torch.tensor_split(x, bkd.get_nranks())
    torch.testing.assert_close(x[bkd.get_rank()], x_rank)


@pytest.mark.mpi
def test_create_shard_dtensor():
    local_size = 2
    global_size = local_size * bkd.get_nranks()

    x_local = torch.full((local_size, 4), bkd.get_rank(), device=bkd.device(), dtype=bkd.floatx())
    
    x_global = torch.zeros((global_size, 4), device=bkd.device())
    for rank in range(bkd.get_nranks()):
        x_global[rank * local_size : (rank+1) * local_size,] = rank

    dist_x = bkd.to_sharded_dtensor(x_local)

    torch.testing.assert_close(dist_x.to_local(), x_local)
    torch.testing.assert_close(dist_x.full_tensor(), x_global)


@pytest.mark.mpi
def test_create_replica_dtensor():
    local_size = 2
    global_size = local_size * bkd.get_nranks()

    x_local = torch.full((local_size, 4), bkd.get_rank(), device=bkd.device(), dtype=bkd.floatx())
    
    x_global = torch.zeros((global_size, 4), device=bkd.device())
    for rank in range(bkd.get_nranks()):
        x_global[rank * local_size : (rank+1) * local_size,] = rank

    dist_x = bkd.to_sharded_dtensor(x_local)

    torch.testing.assert_close(dist_x.to_local(), x_local)
    torch.testing.assert_close(dist_x.full_tensor(), x_global)

    g = torch.Generator(device=bkd.device())
    g.manual_seed(0)
    i = torch.randperm(global_size, generator=g)
    di = bkd.to_replica_dtensor(i)

    xdi = dist_x[di]
    torch.testing.assert_close(xdi.full_tensor(), x_global[i])

