# Copyright (c) 2019-2022 Pranav Singh
# Use of this source code is governed by the MIT license that can be found in the LICENSE file.

import numpy as np
import torch

from torch import tensor, arange, float64, real, pi

from ...linalg import diag

from ..spatial import fixrange

from .spectral import cfftmatrix


def fouriersymbol(n, xrange, device=torch.device('cpu')):
    '''n         scalar int
    xrange    2 length array of reals
    creates the fourier symbol in a single dimension'''
    lf = 2 / (xrange[1] - xrange[0])
    o = tensor(np.mod(n, 2))
    c = 1j * pi * lf * arange(-(n - o) / 2, (n - o) / 2 + o, dtype=float64).to(device)
    return c


def diffmatrix(k, n, xrange):
    '''one dimensional matrix'''
    xrange = fixrange(xrange, 1)[0]     # 1D
    F = cfftmatrix(n)
    symbol = fouriersymbol(n, xrange) ** k
    Dk = F.H @ diag(symbol) @ F
    if np.mod(k, 2) == 0:
        Dk = real(Dk)
    return Dk

