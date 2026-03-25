import os
import socket
from mpi4py import MPI
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard, Replicate
import random
import numpy as np
import scipy as sp

from typing import Any, Union, List


# Global
# -------------------------------------
_SEED = None
_VALID_BKD = {"numpy", "torch"}
_VALID_DEVICE = {"cpu", "cuda"}
_VALID_DTYPE = {"float32", "float64"}
_COMM = None
_RANK = None
_NRANKS = None
_DMESH = None
_DEVICE_PER_RANK = 4

# Setting
# -------------------------------------
def set(
  backend: str = "numpy",
  device: str = "cpu",
  device_idx: int = 0,
  nb_threads: int = 8,
  epsilon: Union[float, None] = 1e-10,
  floatx: str = "float64",
  seed: Union[int, None] = None
) -> None:
  """
  Configure the settings for the computational backend.

  This function sets up various parameters for the backend environment,
  including the computational backend, device, number of threads, and
  precision settings.

  :param backend: The computational backend to use (e.g., "numpy").
  :type backend: str
  :param device: The device to use (e.g., "cpu").
  :type device: str
  :param device_idx: The index of the device to use (e.g., 0 for the
                     first device).
  :type device_idx: int
  :param nb_threads: The number of threads to use.
  :type nb_threads: int
  :param epsilon: A small value to avoid numerical instability. If None,
                  a default value is used.
  :type epsilon: float or None
  :param floatx: The floating-point precision to use (e.g., "float64").
  :type floatx: str
  :param seed: The seed for random number generation. If None, the seed
               is not set.
  :type seed: int or None

  :return: None
  :rtype: None
  """
  set_backend(backend)
  set_device(device, device_idx, nb_threads)

  if is_torch_backend():
    init_distributed(device)
    # TODO: some schedulers handle binding automatically, so device_count will always=1
    if (_NRANKS > 1 and torch.accelerator.device_count() > 1):
      # TODO: fix this to work for multiple nodes!
      device_idx = _RANK % _NRANKS
      #print(" BACKEND: RANK {} reassigning device_idx to {}".format(_RANK, device_idx))


  set_seed(seed)
  #set_device(device, device_idx, nb_threads)
  set_floatx(floatx)
  set_epsilon(epsilon)

def get_backend() -> str:
  """
  Returns the current backend identifier.

  :return: The backend identifier.
  :rtype: str
  """
  return _BKD

def set_backend(
  value: str = "numpy"
) -> None:
  """
  Set the backend for the library.

  :param value: The backend to be set.
  :type value: str

  :raises ValueError: If the provided backend is not in the list of valid
                      backends.
  """
  global _BKD
  _BKD = value
  if (value not in _VALID_BKD):
    raise ValueError(
      f"Unknown backend: '{value}'. Valid options are: {_VALID_BKD}"
    )

# Conversion
# -------------------------------------
def to_numpy(x: Any) -> np.ndarray:
  """
  Convert the input to a NumPy array.

  If the input is already a NumPy array, it is returned as-is. If the input
  is a PyTorch tensor, it is converted to a NumPy array. For other types
  such as `int`, `float`, `list`, or `tuple`, the input is converted to a NumPy
  array with a `float` data type. If the input does not match any of these
  types, it is returned unchanged.

  :param x: The input to convert to a NumPy array. Can be a NumPy array,
            PyTorch tensor, int, float, list, or tuple.
  :type x: Any

  :return: The converted NumPy array or the original input if it cannot be
           converted.
  :rtype: np.ndarray or Any
  """
  if (x is not None):
    if isinstance(x, np.ndarray):
      return x
    elif (torch.is_tensor(x)):
      return x.numpy(force=True)
    elif isinstance(x, (int, float, list, tuple)):
      return np.array(x, dtype=floatx("numpy"))
    else:
      return x

def to_backend(x: Any) -> Union[np.ndarray, torch.Tensor]:
  """
  Convert input to a backend-specific format.

  If the backend is set to "torch" and the input `x` is not already a
  PyTorch tensor, it converts `x` to a PyTorch tensor. If the backend is
  not "torch", it converts `x` to a NumPy array.

  :param x: The input to be converted.
  :type x: Any

  :return: The input converted to the appropriate format based on the
           backend setting.
  :rtype: Union[np.ndarray, torch.Tensor]
  """
  if (x is not None):
    if (_BKD == "torch"):
      if torch.is_tensor(x):
        return x
      else:
        return torch.as_tensor(to_numpy(x), dtype=floatx("torch"), device=device())
    else:
      return to_numpy(x)

def to_sparse(
  x: Union[np.ndarray, sp.sparse.spmatrix]
) -> sp.sparse.spmatrix:
  """
  Convert the input array or sparse matrix to a Compressed Sparse Row (CSR)
  matrix.

  If the input `x` is already a sparse matrix, it will be converted to CSR
  format. If `x` is a dense NumPy array, it will be converted to a CSR sparse
  matrix.

  :param x: The input array or sparse matrix to convert.
  :type x: Union[np.ndarray, sp.sparse.spmatrix]

  :return: The input converted to a CSR sparse matrix.
  :rtype: sp.sparse.spmatrix
  """
  return x.tocsr() if sp.sparse.issparse(x) else sp.sparse.csr_matrix(x)

def to_sp_backend(x: sp.sparse.spmatrix) -> torch.Tensor:
    if (x is not None):
        if (_BKD == "torch"):
            if torch.is_tensor(x):
                return x.to_sparse_csr().cuda()
            else:
                #return torch.sparse_csr_tensor(x.indptr, x.indices, x.data, x.shape, device=device())
                return torch.sparse_csr_tensor(x.indptr, x.indices, x.data, x.shape)

        else:
            return x


def to_sp_coo_backend(x: sp.sparse.spmatrix) -> torch.Tensor:
    """
    Converts a scipy sparse matrix in COO to torch sparse COO
    This routine constructs the torch tensor using the direct data pointers from scipy,
    avoiding additional memory copies and object creation overhead

    If backend is not torch, then the original matrix is returned
    """
    if (x is not None):
        if (_BKD == "torch"):
            if torch.is_tensor(x):
                return x.to_sparse_coo().cuda()
            else:
                row = x.row
                col = x.col
                xcoo = torch.sparse_coo_tensor(torch.tensor(np.vstack((row, col))), x.data, size=x.shape)
                return xcoo
        else:
            return x


def torch_csr_to_scipy(x: torch.Tensor) -> sp.sparse.spmatrix:
    if torch.is_tensor(x):
        return sp.sparse.csr_matrix((x.values().cpu(), x.col_indices().cpu(), x.crow_indices().cpu()), shape=(x.shape[0], x.shape[1]))
    elif isinstance(x, List):
        for i in range(len(x)):
            x[i] = sp.sparse.csr_matrix((x[i].values().cpu(), x[i].col_indices().cpu(), x[i].crow_indices().cpu()), shape=(x[i].shape[0], x[i].shape[1]))
        return x
    else:
        return x


def torch_hstack(x: Union[List, torch.Tensor]) -> torch.Tensor:
    # helper function around torch.hstack for CSR tensors
    # this converts x to COO, since hstack does not work with CSR tensors
    # after the hstack, returns x back in a CSR tensor
    if isinstance(x, List):
        for i in range(len(x)):
            x[i] = x[i].to_sparse_coo()
        x = torch.hstack(x)
        return x.to_sparse_csr()
    else:
        return torch.hstack(x.to_sparse_coo()).to_sparse_csr()


def torch_bmat(x: List) -> torch.Tensor:
    for i in range(len(x)):
      for j in range(len(x[i])):
        if x[i][j] is None:
          continue
        x[i][j] = torch_csr_to_scipy(x[i][j].to_sparse_csr())
    x = sp.sparse.bmat(x, format="csr")
    return to_sp_backend(x)


# Device
# -------------------------------------
def device() -> str:
  """
  Returns the current device identifier.

  :return: The device identifier as a string.
  :rtype: str
  """
  return _DEVICE

def set_device(
  value: str = None,
  index: int = 0,
  nb_threads: int = 8,
) -> None:
  """
  Set the device for computations.

  This function sets the global device for PyTorch operations and configures
  the number of threads for operations.

  :param value: The device to set (e.g., "cpu", "cuda"). If None or "cuda",
                the function will select "cuda" if available, otherwise "cpu".
  :type value: str, optional
  :param index: The device index, default is 0.
  :type index: int, optional
  :param nb_threads: Number of threads to use, default is 8.
  :type nb_threads: int, optional

  :return: None
  :rtype: None

  :raises ValueError: If the device specified in `value` is not valid.
  """
  if ((value is None) or (value == "cuda")):
    value = "cuda" if torch.cuda.is_available() else "cpu"
  if (value not in _VALID_DEVICE):
    raise ValueError(
      f"Unknown device: '{value}'. Valid options are: {_VALID_DEVICE}"
    )
  if (value == "cuda"):
    value += f":{index}"
  global _DEVICE
  _DEVICE = value
  # Set default device
  try:
    torch.set_default_device(torch.device(_DEVICE))
    torch.set_num_interop_threads(nb_threads)
    torch.set_num_threads(nb_threads)
  except:
    raise RuntimeWarning("failed to set device")

# Epsilon
# -------------------------------------
def machine_eps() -> float:
  """
  Returns the machine epsilon for the floating-point precision defined
  by `_FLOATX`.

  Machine epsilon is the smallest positive number :math:`\epsilon` such that
  :math:`1.0 + \epsilon \neq 1.0`. This function returns the machine epsilon
  for the data type specified by the global variable `_FLOATX`.

  :return: Machine epsilon for the specified floating-point precision.
  :rtype: float

  :raises KeyError: If `_FLOATX` is not one of 'float16', 'float32', or 'float64'.
  """
  return float(np.finfo(
    {
      "float16": np.float16,
      "float32": np.float32,
      "float64": np.float64
    }[_FLOATX]
  ).eps)

def epsilon() -> float:
  """
  Returns the current small epsilon value used for numerical stability.

  :return: A small epsilon value.
  :rtype: float
  """
  return _EPSILON

def set_epsilon(
  value: Union[float, None] = None
) -> None:
  """
  Set the global epsilon value used for numerical precision.

  If no value is provided, the function sets epsilon to the machine epsilon.

  :param value: The epsilon value to set. If None, defaults to machine epsilon.
  :type value: float or None

  :return: None
  :rtype: None
  """
  if (value is None):
    value = machine_eps()
  global _EPSILON
  _EPSILON = value

# Float
# -------------------------------------
def floatx(
  bkd: str = "torch"
) -> Union[str, type, torch.dtype]:
  """
  Returns the floating point precision type based on the backend and global
  `_FLOATX` setting.

  :param bkd: The backend to use ("torch" or "numpy"). Default is "torch".
  :type bkd: str

  :return: The floating point precision type for the specified backend.
  :rtype: Union[str, type, torch.dtype]

  :raises ValueError: If the backend is not "torch" or "numpy".
  """
  if (bkd == "torch"):
    return {
      "float16": torch.float16,
      "float32": torch.float32,
      "float64": torch.float64
    }[_FLOATX]
  elif (bkd == "numpy"):
    return {
      "float16": np.float16,
      "float32": np.float32,
      "float64": np.float64
    }[_FLOATX]
  else:
    return _FLOATX

def set_floatx(
  value: str
) -> None:
  """
  Set the global floating-point precision type for the library.

  This function sets the global floating-point precision type (`_FLOATX`) to
  the specified value. If the value is not in the list of valid data types,
  it raises a `ValueError`. Additionally, it tries to set the default floating-
  point dtype in PyTorch.

  :param value: The desired floating-point precision type.
  :type value: str

  :raises ValueError: If the provided value is not in the list of valid dtypes.

  :return: None
  :rtype: None
  """
  global _FLOATX
  _FLOATX = value
  if (value not in _VALID_DTYPE):
    raise ValueError(
      f"Unknown dtype: '{value}'. Valid options are: {_VALID_DTYPE}"
    )
  try:
    torch.set_default_dtype(floatx())
  except:
    pass

# Seed
# -------------------------------------
def seed() -> Union[int, None]:
  """
  Retrieve the current seed value.

  :return: The current seed value if set, otherwise None.
  :rtype: Union[int, None]
  """
  return _SEED

def set_seed(
  value: Union[int, None] = None
) -> None:
  """
  Set random number generator seeds for reproducibility.

  This function sets the seed for Python"s built-in random module, NumPy,
  and PyTorch, ensuring deterministic operations. It"s essential for
  achieving reproducible results in data processing and machine learning
  tasks. If `value` is provided, all random generators will use the same seed.

  :param value: An integer seed for random number generators.
  :type value: int or None

  :return: None
  :rtype: None
  """
  global _SEED
  _SEED = value
  if (value is not None):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    # torch.use_deterministic_algorithms(True)
    os.environ["PYTHONHASHSEED"] = str(value)


def is_torch_backend():
    if _BKD == "torch":
        return True
    else:
        return False


def init_distributed(backend_type="cuda"):
  global _COMM, _RANK, _NRANKS
  _COMM = MPI.COMM_WORLD
  _RANK = _COMM.Get_rank()
  _NRANKS = _COMM.Get_size()

  print(" INIT DISTRIBUTED: rank = {}, num ranks = {}".format(_RANK, _NRANKS))

  if _NRANKS > 1:
    # Broadcast root hostname to all other ranks
    root_addr = None
    if _RANK == 0:
      root_addr = socket.gethostname()
    root_addr = _COMM.bcast(root_addr, root=0)

    os.environ["MASTER_ADDR"] = root_addr
    os.environ["MASTER_PORT"] = "23457"

    print(" -- FLUX_JOB_SIZE = {} FLUX_TASK_RANK = {}".format(int(os.environ.get('FLUX_JOB_SIZE')), int(os.environ.get('FLUX_TASK_RANK'))))

    # Use FLUX vars if available, else fallback to MPI
    world_size = int(os.environ.get('FLUX_JOB_SIZE', _NRANKS))
    rank_env = int(os.environ.get('FLUX_TASK_RANK', _RANK))
    assert rank_env == _RANK

    backend = "gloo" if backend_type == "cpu" else "nccl"

    print("  Creating torch process group: world size = {} rank = {}, backend type = '{}'".format(world_size, rank_env, backend))
    print("   RANK {} number of available devices = {}".format(_RANK, torch.accelerator.device_count()))

    # Set up process group
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        rank=rank_env,
        world_size=world_size
    )

    init_device_mesh(backend_type, use_2d=False)
  else:
    print("Serial mode; no distributed!")

  print("   RANK {}: Initialized on device '{}'".format(_RANK, torch.cuda.get_device_name()))
  print("   RANK {}:   Device properties: {}".format(_RANK, torch.cuda.get_device_properties()))


def init_device_mesh(backend_type, use_2d=True):
  if not _BKD == "torch" or not distributed():
    return

  # fallback to 1D mesh if grid is not even
  if _NRANKS % _DEVICE_PER_RANK != 0:
    use_2d = False

  if use_2d:
    mesh = (_NRANKS // _DEVICE_PER_RANK, _DEVICE_PER_RANK)
    dims = ("GLOBAL", "LOCAL")
  else:
    mesh = (_NRANKS,)
    dims = ("GLOBAL",)

  print("   RANK {}:   Initializing device mesh {} ({})".format(_RANK, mesh, dims))
  global _DMESH
  _DMESH = dist.init_device_mesh(backend_type, mesh_shape=mesh, mesh_dim_names=dims)
  print("   RANK {}:   Initialized device mesh: {}".format(_RANK, _DMESH))


def finalize_distributed():
  if not distributed():
    return
  dist.destroy_process_group()


def distributed():
  if _NRANKS is not None and _NRANKS > 1:
    return True
  return False

def get_rank():
  if _RANK is not None:
    return _RANK
  else:
    raise RuntimeError("Tried to get rank, but not using distributed!")

def get_nranks():
  if _NRANKS is not None:
    return _NRANKS
  else:
    raise RuntimeError("Tried to get number of ranks, but not using distributed!")

def barrier():
  if not distributed():
    return

  torch.accelerator.synchronize()
  _COMM.Barrier()
  dist.barrier()


def root():
  if not distributed():
    return True
  return _RANK == 0


# Parallel helpers:
def get_local_sizes(data, dim=0, root=0):
  """
  Returns a list with the size of data (at the given dim) for each rank
  Returned list is only defined on rank root, other ranks are undefined
  """
  lsize = data.shape[dim]
  if not distributed():
    return [lsize]

  rank_sizes = _COMM.gather(lsize, root=root)
  barrier()

  return rank_sizes


def get_local_sizes_all(data, dim=0):
  """
  All-node version of get_local_sizes (size of data returned for all ranks)
  """
  lsize = data.shape[dim]
  if not distributed():
    return [lsize]

  rank_sizes = _COMM.allgather(lsize)
  barrier()

  return rank_sizes


def gather_tensor(x, sizes=None, dim=0, root=0):
  """
  Gathers the local tensor x from each rank onto root rank.

  If sizes is not provided, shape of each input x is determined. Otherwise,
  sizes is a list of sizes over all ranks.

  Rank local tensors x are assumed to be split over specified dim, so all other
  dimensions are assumed the same size on all ranks.
  """
  if not distributed():
    return x

  data = None

  rank_sizes = sizes
  calc_sizes = False
  if rank_sizes is None and _RANK == root:
    # if rank sizes is undefined on root rank, then we need to determine the sizes
    calc_sizes = True
  calc_sizes = _COMM.bcast(calc_sizes, root=root)
  barrier()

  if calc_sizes:
    rank_sizes = get_local_sizes(x, dim, root)

  if _RANK == root:
    data = []
    for rank in range(_NRANKS):
      # check each rank has same size tensor (torch gather requires this)
      assert rank_sizes[rank] == rank_sizes[0]
      # NOTE: assumes data is 2D tensor, where the second dim is always constant across ranks (e.g domain size)
      data.append(torch.zeros_like(x, device=device()))

  dist.gather(x, data, root)

  barrier()

  if _RANK == root:
    # Root rank stacks all gathered tensors
    data = torch.cat(data, dim=dim)

  barrier()

  # Root rank returns combined tensor, all others just return input tensor
  if _RANK == root:
    return data
  else:
    return x


def scatter_tensor(x, x_out=None, dim=0, root=0):
  """
  Scatters a distributed tensor x across all ranks (from source rank root) and returns result
  Each rank gets 1/nranks portion of x tensor (along dim)
  NOTE: assumes x is distributed across all ranks, and each rank must have same size

  If x_out=None, then result tensor will be allocated with size according to x.shape[dim] // nranks
  NOTE: x_out will be overwritten by this operation
  """
  if not distributed():
    return x

  full_size = 0
  data = None
  if _RANK == root:
    data = list(torch.tensor_split(x, _NRANKS, dim=dim))
    full_size = x.shape[dim]
    assert len(data) == _NRANKS
    if x_out is None:
      full_size = list(x.size())
      full_size[dim] = x.shape[dim] // _NRANKS

  if x_out is None:
    full_size = _COMM.bcast(full_size, root=root)
    x_out = torch.empty(full_size, device=device())

  barrier()
  dist.scatter(x_out, data, root)
  barrier()

  return x_out


def get_mesh():
  return _DMESH


# Methods for DTensor creation
def to_sharded_dtensor(x: torch.Tensor,
                       shape: List[int] = None,
                       stride: List[int] = None) -> DTensor:
  """
  Creates a distributed tensor from each rank's local tensor x.
  The returned distributed tensor is sharded across the current device mesh on dim 1,
  so the DTensor represents a full tensor of x combined across all ranks.
  """
  if not distributed() or _DMESH is None:
    return x

  dtensor = DTensor.from_local(x,
                               device_mesh=_DMESH,
                               placements=[Shard(0)],
                               shape=shape,
                               stride=stride,
                               run_check=False)
  return dtensor


def to_replica_dtensor(x: torch.Tensor,
                       shape: List[int] = None,
                       stride: List[int] = None) -> DTensor:
  """
  Creates a distributed tensor from each rank's local tensor x.
  The returned distributed tensor is replicated across all ranks in the current device mesh on dim 1,
  so the DTensor represents a full tensor of x combined across all ranks.
  """
  if not distributed() or _DMESH is None:
    return x

  dtensor = DTensor.from_local(x,
                               device_mesh=_DMESH,
                               placements=[Replicate()],
                               shape=shape,
                               stride=stride,
                               run_check=False)
  return dtensor
