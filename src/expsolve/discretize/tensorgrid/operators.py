# Copyright (c) 2019-2022 Pranav Singh
# Use of this source code is governed by the MIT license that can be found in the LICENSE file.

from torch import exp

from ..spatial import fixrange

from .grid import dim
from .spectral import cfft, cifft, fourierfn
from .fourier import fouriersymbol


# batch revisit
def diffop(d, k, u, xrange=-1, symbolfn=fouriersymbol):
    xrange = fixrange(xrange, dim(u))
    return fourierfn(symbolfn, lambda x: x ** k, u, d, xrange)


# batch revisit
def diffopexp(d, k, s, u, xrange=-1, symbolfn=fouriersymbol):
    '''This evaluates exp(s d^k/dx_d^k) (u) using spectral/fourier'''
    xrange = fixrange(xrange, dim(u))
    return fourierfn(symbolfn, lambda x: exp(s * x**k), u, d, xrange)


# batch revisit
def laplacianopexp(lapsymb, s, u):
    esL = exp(s * lapsymb)   # in practical implementation better to compute and store this once.
    return cifft(esL * cfft(u))