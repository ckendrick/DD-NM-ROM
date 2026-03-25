import pytest

import numpy as np
import torch
import torch.testing

from copy import deepcopy
from dd_nm_rom import backend as bkd
from dd_nm_rom.utils import parallel_print
from dd_nm_rom.rom.nonlinear.autoencoder.nn_numpy import activation

def setup_module(module):
    bkd.set(backend="torch", device="cuda", seed=0)


def teardown_module(module):
    bkd.finalize_distributed()


def test_softplus():
    x = np.random.randn(16)
    x_torch = bkd.to_backend(deepcopy(x))

    act_np = activation.Softplus()
    fun_np = act_np(deepcopy(x), with_jac=False)
    fx, fdx = act_np(deepcopy(x), with_jac=True)

    act_torch = activation.get("softplus")
    fun_torch = bkd.to_numpy(act_torch(bkd.to_backend(deepcopy(x)), with_jac=False).to("cpu"))
    fx_t, fdx_t = act_torch(bkd.to_backend(deepcopy(x)), with_jac=True)

    # with jacobian
    np.testing.assert_allclose(fun_torch, fun_np)
    np.testing.assert_allclose(bkd.to_numpy(fx_t.to("cpu")), fx)
    np.testing.assert_allclose(bkd.to_numpy(fdx_t.to("cpu").to_dense()), fdx.todense())


