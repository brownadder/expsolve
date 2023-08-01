# Copyright (c) 2019-2022 Pranav Singh
# Use of this source code is governed by the MIT license that can be found in the LICENSE file.

import numpy as np

from torch import float64, complex128, meshgrid, linspace, sum
from torch.linalg import norm


# complex
def complex(u):
    return u.type(complex128)


def dim(u):
    '''Returns the dimensions of a discretised function u

    u             discretised function

    output        scalar int
    '''
    return len(list(u.shape))-1


def alldims(u):
    return tuple(range(1, dim(u)+1))


def fixrange(xrange, dims):
    '''Create a valid xrange to describe domain

    xrange
        input xrange can be
        -1            in which case set xrange to default: [-1, 1] in all dimensions
        list          this needs to be length 2 and is repeated across all dimensions
        ndarray       of size dims x 2 and is returned as it is
    dims          dimensions

    output        dims x 2 ndarray'''
    if isinstance(xrange, int) and xrange == -1:
        return np.array([[-1, 1] for d in range(dims)])
    else:
        if isinstance(xrange, list):
            assert len(xrange) == 2
            return np.array([xrange for d in range(dims)])
        else:
            return xrange


def grid1d(n, xrange=[-1, 1]):
    '''Create a simple one-dimensional grid

    n         scalar int
    xrange    2 length list of reals

    output    n x 1 float'''
    offset = (xrange[1] - xrange[0]) / (2 * n)
    return linspace(xrange[0] + offset, xrange[1] - offset, n, dtype=float64).unsqueeze(dim=0)


def l2inner(u, v, xrange=-1):
    '''Computes the complex L2 inner product <u, v>
    which is conjugate linear in u and linear in v
    
    u,v         complex-valued discretised functions discretised
                on domain described by xrange
    xrange      dims x 2 ndarray in higher dimensions, or a list in 1d (default [-1,1])
    
    output      scalar complex'''
    xrange = fixrange(xrange, dim(u))
    s = np.prod((xrange[:, 1] - xrange[:, 0])/u.shape[1:])
    return s * sum(complex(u).flatten(start_dim=1).conj() * complex(v).flatten(start_dim=1), dim=1)


def l2norm(u, xrange=-1):
    '''Computes the L2 norm ||u||
    
    u           complex-valued discretised function discretised
                on domain described by xrange
    xrange      dims x 2

    output      scalar real'''
    xrange = fixrange(xrange, dim(u))
    s = np.prod((xrange[:, 1] - xrange[:, 0])/u.shape[1:])
    return np.sqrt(s) * norm(u.flatten(start_dim=1), dim=1)


def normalize(u, xrange=-1):
    nrm = l2norm(u, xrange)
    nrm = nrm.view(u.shape[0], *([1] * dim(u)))
    return complex(u/nrm)


def grid(n, xrange=-1):
    '''Create an n-dimensional grid

    n         dim length array of int
    xrange    dim x 2 ndarray of reals (if a light of length 2 is provided it is copied in all dims)

    output    dim length list of ndarrays, each of size n_1 x n_2 x ... x n_N'''
    dims = len(n)
    xrange = fixrange(xrange, dims)
    xlist = []
    for i in range(dims):
        xlist.append(grid1d(n[i], xrange[i]).flatten())
    x = meshgrid(xlist, indexing='xy')
    x = list(x)
    for i in range(dims):
        x[i] = (x[i].T).unsqueeze(dim=0)
    return x


