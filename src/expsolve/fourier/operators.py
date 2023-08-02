# Copyright (c) 2019-2022 Pranav Singh
# Use of this source code is governed by the MIT license that can be found in the LICENSE file.

import numpy as np
from .discretize import dim, fixrange, l2inner, complex
from .spectral import cfft, cifft, cfftmatrix, fouriersymbol

from torch import exp, real, complex128, is_tensor
import torch


def diag(v):
    return torch.diag(v.flatten())



# batch revisit : broadcast in both directions: if either v or A is longer in dim 0
def matmul(A, v):
    shp = v.shape
    u = v.reshape([shp[0], -1]).mT

    # nb = shp[0]     # number of batches
    if A.dtype == complex128 or v.dtype == complex128:
        return torch.matmul(complex(A), complex(u)).mT.reshape(shp)
    else:
        return torch.matmul(A, u).mT.reshape(shp)



def diffmatrix(k, n, xrange):
    '''one dimensional matrix'''
    xrange = fixrange(xrange, 1)[0]     # 1D
    F = cfftmatrix(n)
    symbol = fouriersymbol(n, xrange) ** k
    Dk = F.H @ diag(symbol) @ F
    if np.mod(k, 2) == 0:
        Dk = real(Dk)
    return Dk


def _preprocess1D(x, y=None):
    X = x
    if y is None:
        y = x
        n = x.shape[1]
        X = torch.tensor(range(1, n+1))
    X = X.detach().cpu().flatten()
    y = y.detach().cpu()
    return X, y


def plot(plt, x, y=None, *args, **kwargs):
    x, y = _preprocess1D(x, y)
    nb = y.shape[0]
    for i in range(nb):
        plt.plot(x, y[i].flatten(), *args, **kwargs)


def plotshaded(plt, x, y=None, *args, **kwargs):
    x, y = _preprocess1D(x, y)
    mean_data = torch.mean(y, axis=0)
    variance_data = torch.std(y, axis=0)
    plt.plot(x, mean_data, *args, **kwargs)
    plt.fill_between(x, mean_data - variance_data, mean_data + variance_data, alpha=0.2, *args, **kwargs)


def semilogy(plt, x, y=None, *args, **kwargs):
    x, y = _preprocess1D(x, y)
    nb = y.shape[0]
    for i in range(nb):
        plt.semilogy(x, y[i].flatten(), *args, **kwargs)


# improve pre-processing
def imshow(plt, xrange, y, *args, **kwargs):
    assert dim(y) == 2
    assert y.shape[0] == 1
    region = list(fixrange(xrange, 2).flatten())
    plt.imshow(y.reshape(y.shape[1:]), extent=region)


# batch revisit
def fourierproduct(fn, c, u, d):
    '''Fourier symbol c is along the (d+1)-th dimension - apply fn(c) along this direction'''
    fc = fn(c)  # this is 1-D:                           n_d
    dims = dim(u)  # this may be N-D:     n_b x n_1 x n_2 x ... x n_d x .... x n_N
    nd = len(fc)
    broadcastshape = [nd if di == d+1 else 1 for di in range(dims+1)]
    
    fc = fc.reshape(broadcastshape)        # extend symbol to shape 1 x 1 x ... 1 x n_d x 1 x ... 1
    return fc * u

    

# batch revisit
def fourierfn(fn, u, d, xrange):
    '''fn        function
    u         ndarray of complex numbers
    d         scalar int - dimension to apply fn in
    implements fn(d/dx_d) * u'''
    shape = list(u.shape)
    device = u.device
    fs = fouriersymbol(shape[d+1], xrange[d], device)
    return cifft(fourierproduct(fn, fs, cfft(u, d), d), d)


# batch revisit
def diffop(d, k, u, xrange=-1):
    xrange = fixrange(xrange, dim(u))
    return fourierfn(lambda x: x ** k, u, d, xrange)


# batch revisit
def diffopexp(d, k, s, u, xrange=-1):
    '''This evaluates exp(s d^k/dx_d^k) (u) using spectral/fourier'''
    xrange = fixrange(xrange, dim(u))
    return fourierfn(lambda x: exp(s * x**k), u, d, xrange)


# batch revisit
def laplacianopexp(lapsymb, s, u):
    esL = exp(s * lapsymb)   # in practical implementation better to compute and store this once.
    return cifft(esL * cfft(u))



# batch revisit - later - broadcast like mv
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