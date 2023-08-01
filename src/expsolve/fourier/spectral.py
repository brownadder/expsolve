# Copyright (c) 2019-2022 Pranav Singh
# Use of this source code is governed by the MIT license that can be found in the LICENSE file.

import numpy as np
from torch.fft import fft, ifft, fftn, ifftn, fftshift, ifftshift
from torch import meshgrid, arange, eye, zeros
from torch import pi, sqrt, tensor, float64, real
import torch
from .discretize import alldims


# batch revisit
def cfft(f, d=-1):
    '''f         complex-valued data
    d         dimension in which to do fft. default (d = -1) 
              for fft in all directions
    central fft - performs fourier transform while shifting frequency to centre
    NOTE: scaled different from matlab implementation'''
    if d == -1:
        return fftshift(fftn(f, dim=alldims(f)))
    else:
        return fftshift(fft(f, dim=d+1), d+1)


# batch revisit
def cifft(f, d=-1):
    '''inverse of cfft'''
    if d == -1:
        return ifftn(ifftshift(f, dim=alldims(f)))
    else:
        return ifft(ifftshift(f, d+1), dim=d+1)


# batch revisit
def cfftmatrix(n):
    id = eye(n, dtype=float64)
    F = fftshift(fft(id, axis=0), dim=0) / sqrt(tensor(n))
    return F


# batch revisit
def fouriersymbol(n, xrange, device=torch.device('cpu')):
    '''n         scalar int
    xrange    2 length array of reals
    creates the fourier symbol in a single dimension'''
    lf = 2 / (xrange[1] - xrange[0])
    o = tensor(np.mod(n, 2))
    c = 1j * pi * lf * arange(-(n - o) / 2, (n - o) / 2 + o, dtype=float64).to(device)
    return c.unsqueeze(dim=0)


# batch revisit
def fouriersymbolfull(n, xrange, device=torch.device('cpu')):
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
    clist = [fouriersymbol(n[d], xrange[d]).flatten().to(device) for d in range(dims)]
    cgrid = meshgrid(*clist, indexing='xy')
    for i in range(dims):
        cgrid[i] = cgrid[i].unsqueeze(dim=0)
    return cgrid


# batch revisit
def laplaciansymbol(n, xrange, device='cpu'):
    c = fouriersymbolfull(n, xrange, device)
    dims = xrange.shape[0]
    lapsymb = zeros(c[0].shape, dtype=float64)
    for d in range(dims):
        lapsymb = lapsymb + real(c[d]**2)
    return lapsymb
