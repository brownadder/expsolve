# Copyright (c) 2019-2024 Pranav Singh
# Use of this source code is governed by the MIT license that can be found in the LICENSE file.

# This file is common to all graph instance level operations
# These should not be operators acting on grids 

import numpy as np

import torch
from torch import float64, meshgrid, linspace, sum, real, is_tensor, is_complex
from torch.linalg import norm

from ...linalg import matmul
from ...utils import complexify

from ..spatial import fixrange


def dim(u):
    '''Returns the dimensions of the graph

    u             discretised function

    output        scalar int
    '''
    return len(list(u.shape))-1


def alldims(u):
    return tuple(range(1, dim(u)+1))


def graph2d(n, xrange=[-1, 1], dtype=float64, device=torch.device('cpu')):
    '''Create a simple one-dimensional grid

    n         scalar int
    xrange    2 length list of reals

    output    n x 1 float'''
    offset = (xrange[1] - xrange[0]) / (2 * n)
    return linspace(xrange[0] + offset, xrange[1] - offset, n, dtype=dtype).unsqueeze(dim=0).to(device)


def grid(n, xrange=-1, dtype=float64, device=torch.device('cpu')):
    '''Create an n-dimensional grid

    n         dim length array of int
    xrange    dim x 2 ndarray of reals (if a light of length 2 is provided it is copied in all dims)

    output    dim length list of ndarrays, each of size n_1 x n_2 x ... x n_N'''
    dims = len(n)
    xrange = fixrange(xrange, dims)
    xlist = []
    for i in range(dims):
        xlist.append(grid1d(n[i], xrange[i], dtype=dtype, device=device).flatten())
    x = meshgrid(xlist, indexing='ij')
    x = list(x)
    for i in range(dims):
        x[i] = (x[i]).unsqueeze(dim=0)
    return x


def l2norm(u, xrange=-1):
    '''Computes the L2 norm ||u||
    
    u           complex-valued discretised function discretised
                on domain described by xrange
    xrange      dims x 2

    output      scalar real'''
    xrange = fixrange(xrange, dim(u))
    s = np.prod((xrange[:, 1] - xrange[:, 0])/u.shape[1:])
    return np.sqrt(s) * norm(u.flatten(start_dim=1), dim=1)


def normalize(u, xrange=-1, keepreal=False):
    nrm = l2norm(u, xrange)
    nrm = nrm.view(u.shape[0], *([1] * dim(u)))
    if keepreal:
        return u/nrm
    else:
        return complexify(u/nrm)


def l2inner(u, v, xrange=-1):
    '''Computes the complex L2 inner product <u, v>
    which is conjugate linear in u and linear in v
    
    u,v         complex-valued discretised functions discretised
                on domain described by xrange
    xrange      dims x 2 ndarray in higher dimensions, or a list in 1d (default [-1,1])
    
    output      scalar complex'''
    xrange = fixrange(xrange, dim(u))
    s = np.prod((xrange[:, 1] - xrange[:, 0])/u.shape[1:])

    if is_complex(u) or is_complex(v):
        return s * sum(complexify(u).flatten(start_dim=1).conj() * complexify(v).flatten(start_dim=1), dim=1)
    else:
        return s * sum(u.flatten(start_dim=1) * v.flatten(start_dim=1), dim=1)


def observable(obs, u, xrange=-1):
    '''Computes the expected value of the observable O in state u, i.e. <u, O u>
    
    obs         Hermitian operator or matrix
    u           complex-valued discretised functions discretised
                on domain described by xrange
    xrange      dims x 2 ndarray in higher dimensions, or a list in 1d (default [-1,1])
    
    output      scalar real'''
    if (is_tensor(obs)):
        Ou = matmul(obs, u)
    else:
        Ou = obs(u)
    return real(l2inner(u, Ou, xrange))