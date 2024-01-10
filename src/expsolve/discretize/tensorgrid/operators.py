# Copyright (c) 2019-2022 Pranav Singh
# Use of this source code is governed by the MIT license that can be found in the LICENSE file.

from torch import exp, is_complex, real, tensor

from ..spatial import fixrange

from .grid import dim
from .spectral import fourierfn, fourierproduct
from .fourier import fouriersymbol


# batch revisit
def diffop(d, k, u, xrange=-1, symbolfn=fouriersymbol):
    xrange = fixrange(xrange, dim(u))
    dku = fourierfn(lambda x: x ** k, u, d, xrange, symbolfn=symbolfn)
    if is_complex(u):
        return dku
    else:
        return real(dku)


# batch revisit
def diffopexp(d, k, s, u, xrange=-1, symbolfn=fouriersymbol):
    '''This evaluates exp(s d^k/dx_d^k) (u) using spectral/fourier'''
    xrange = fixrange(xrange, dim(u))
    expdku = fourierfn(lambda x: exp(s * x**k), u, d, xrange, symbolfn=symbolfn)
    if is_complex(u) or is_complex(tensor(s)):
        return expdku
    else:
        return real(expdku)


# batch revisit
def laplacianop(lapsymb, u):
    Lu = fourierproduct(lapsymb, u, d=-1)
    if is_complex(u):
        return Lu
    else:
        return real(Lu)
    

# batch revisit
def laplacianopexp(lapsymb, s, u):
    esL = exp(s * lapsymb)   # in practical implementation better to compute and store this once.
    expLu = fourierproduct(esL, u, d=-1)
    if is_complex(u) or is_complex(tensor(s)):
        return expLu
    else:
        return real(expLu)