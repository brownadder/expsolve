# Copyright (c) 2019-2022 Pranav Singh
# Use of this source code is governed by the MIT license that can be found in the LICENSE file.

import numpy as np

from torch import float64, complex128, is_tensor, meshgrid, inner, linspace, real
from torch.linalg import norm


# complex
def complex(u):
    return u.type(complex128)


# batch revisit
def dim(u):
    '''Returns the dimensions of a discretised function u

    u             discretised function

    output        scalar int
    '''
    return len(list(u.shape))


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
    return linspace(xrange[0] + offset, xrange[1] - offset, n, dtype=float64)

# batch revisit
def l2inner(u, v, xrange=-1):
    '''Computes the complex L2 inner product <u, v>
    which is conjugate linear in u and linear in v
    
    u,v         complex-valued discretised functions discretised
                on domain described by xrange
    xrange      dims x 2 ndarray in higher dimensions, or a list in 1d (default [-1,1])
    
    output      scalar complex'''
    xrange = fixrange(xrange, dim(u))
    s = np.prod((xrange[:, 1] - xrange[:, 0])/u.shape)
    return s * inner(u.flatten().conj(), v.flatten())

# batch revisit
def l2norm(u, xrange=-1):
    '''Computes the L2 norm ||u||
    
    u           complex-valued discretised function discretised
                on domain described by xrange
    xrange      dims x 2

    output      scalar real'''
    xrange = fixrange(xrange, dim(u))
    s = np.prod((xrange[:, 1] - xrange[:, 0])/u.shape)
    return np.sqrt(s) * norm(u.flatten())


# batch revisit
def normalize(u, xrange=-1):
    return complex(u/l2norm(u, xrange))


def grid(n, xrange):
    '''Create an n-dimensional grid

    n         dim length array of int
    xrange    dim x 2 ndarray of reals (if a light of length 2 is provided it is copied in all dims)

    output    dim length list of ndarrays, each of size n_1 x n_2 x ... x n_N'''
    dims = len(n)
    xrange = fixrange(xrange, dims)
    xlist = []
    for i in range(dims):
        xlist.append(grid1d(n[i], xrange[i]))
    x = meshgrid(xlist, indexing='xy')
    x = list(x)
    for i in range(dims):
        x[i] = x[i].T
    return x


# batch revisit
def observable(obs, u, xrange=-1):
    '''Computes the expected value of the observable O in state u, i.e. <u, O u>
    
    obs         Hermitian operator or matrix
    u           complex-valued discretised functions discretised
                on domain described by xrange
    xrange      dims x 2 ndarray in higher dimensions, or a list in 1d (default [-1,1])
    
    output      scalar real'''
    if (is_tensor(obs)):
        Ou = complex(obs) @ complex(u)
    else:
        Ou = obs(u)
    return real(l2inner(complex(u), Ou, xrange))
