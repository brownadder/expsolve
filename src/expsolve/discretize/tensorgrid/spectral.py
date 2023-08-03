# Copyright (c) 2019-2022 Pranav Singh
# Use of this source code is governed by the MIT license that can be found in the LICENSE file.

# This file is kept common for Fourier and finite differences 
# since it is possible to implement exp of FD diff matrices via Fourier approach
# These should be helper utilities for implementing operators

import torch

from torch import meshgrid, eye, zeros, sqrt, tensor, float64, real
from torch.fft import fft, ifft, fftn, ifftn, fftshift, ifftshift

from .grid import dim, alldims


# batch revisit
def cfft(f, d=-1):
    '''f         complex-valued data
    d         dimension in which to do fft. default (d = -1) 
              for fft in all directions
    central fft - performs fourier transform while shifting frequency to centre
    NOTE: scaled different from matlab implementation'''
    if d == -1:
        return fftshift(fftn(f, dim=alldims(f)), dim=alldims(f))
    else:
        return fftshift(fft(f, dim=d+1), d+1)


# batch revisit
def cifft(f, d=-1):
    '''inverse of cfft'''
    if d == -1:
        return ifftn(ifftshift(f, dim=alldims(f)), dim=alldims(f))
    else:
        return ifft(ifftshift(f, d+1), dim=d+1)


# batch revisit
def cfftmatrix(n):
    id = eye(n, dtype=float64)
    F = fftshift(fft(id, axis=0), dim=0) / sqrt(tensor(n))
    return F


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
def fourierfn(symbolfn, fn, u, d, xrange):
    '''fn        function
    u         ndarray of complex numbers
    d         scalar int - dimension to apply fn in
    implements fn(d/dx_d) * u'''
    shape = list(u.shape)
    device = u.device
    fs = symbolfn(shape[d+1], xrange[d], device)
    return cifft(fourierproduct(fn, fs, cfft(u, d), d), d)


# in Torch 2.0 do we need to specify device in this way?
# fouriersymbolfn needs to be passed now - this will allow direct application to finite differences
# batch revisit
def tensorizesymbol(symbolfn, n, xrange, device=torch.device('cpu')):
    '''When a full grid of the fourier symbol is required - this is helpful if
    one is using cfftn (or cfft(u,1:D)), i.e. FFT is first run in all
    directions and then there is pointwise multiplication. This symbol can
    help compute the Laplacian efficiently.
    n         dim length array int
    xrange    dim x 2 length array of reals (for 1D xrange should be [[x0,x1]])
    storefn   for gpu/cpu storage
    creates the fourier symbol in all dimensions'''
    dims = xrange.shape[0]
    if len(n) < dims:
        # best to specify n for each dim, otherwise max(n) is used
        n = n + [max(n) for i in range(dims-len(n))]  
    clist = [symbolfn(n[d], xrange[d]).to(device) for d in range(dims)]
    cgrid = meshgrid(*clist, indexing='xy')
    for i in range(dims):
        cgrid[i] = cgrid[i].unsqueeze(dim=0)
    return cgrid


# batch revisit
def laplaciansymbol(symbolfn, n, xrange, device='cpu'):
    c = tensorizesymbol(symbolfn, n, xrange, device)
    dims = xrange.shape[0]
    lapsymb = zeros(c[0].shape, dtype=float64)
    for d in range(dims):
        lapsymb = lapsymb + real(c[d]**2)
    return lapsymb

