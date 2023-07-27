# Copyright (c) 2019-2022 Pranav Singh
# Use of this source code is governed by the MIT license that can be found in the LICENSE file.

import numpy as np
from .discretize import dim, fixrange
from .spectral import cfft, cifft, cfftmatrix, fouriersymbol

from torch import diag, exp, real


def diffmatrix(k, n, xrange):
    '''one dimensional matrix'''
    xrange = fixrange(xrange, 1)[0]     # 1D
    F = cfftmatrix(n)
    symbol = fouriersymbol(n, xrange) ** k
    Dk = F.H @ diag(symbol) @ F
    if np.mod(k, 2) == 0:
        Dk = real(Dk)
    return Dk


def fourierproduct(fn, c, u, d):
    '''Fourier symbol c is along the d-th dimension - apply fn(c) along this direction'''
    fc = fn(c)  # this is 1-D:                           n_d
    dims = dim(u)  # this may be N-D:     n_1 x n_2 x ... x n_d x .... x n_N
    if dims > 1:
        nd = len(fc)
        shp = [nd if di == d else 1 for di in range(dims)]
        fc = fc.reshape(shp)  # extend symbol to shape 1 x 1 x ... 1 x n_d x 1 x ... 1
    return fc * u


def fourierfn(fn, u, d, xrange):
    '''fn        function
    u         ndarray of complex numbers
    d         scalar int - dimension to apply fn in
    implements fn(d/dx_d) * u'''
    shape = list(u.shape)
    device = u.device
    fs = fouriersymbol(shape[d], xrange[d], device)
    return cifft(fourierproduct(fn, fs, cfft(u, d), d), d)


def diffop(d, k, u, xrange=-1):
    xrange = fixrange(xrange, dim(u))
    return fourierfn(lambda x: x ** k, u, d, xrange)


def diffopexp(d, k, s, u, xrange=-1):
    '''This evaluates exp(s d^k/dx_d^k) (u) using spectral/fourier'''
    xrange = fixrange(xrange, dim(u))
    return fourierfn(lambda x: exp(s * x**k), u, d, xrange)


def laplacianopexp(lapsymb, s, u):
    esL = exp(s * lapsymb)   # in practical implementation better to compute and store this once.
    return cifft(esL * cfft(u))

