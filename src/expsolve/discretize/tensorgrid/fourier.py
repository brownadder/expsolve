# Copyright (c) 2019-2022 Pranav Singh
# Use of this source code is governed by the MIT license that can be found in the LICENSE file.

import numpy as np
import torch

from torch import tensor, arange, complex128, real, pi, eye, sqrt, float64
from torch.fft import fft, fftshift

from ...linalg import diag
from ...utils import complexifytype, realifytype

from ..spatial import fixrange


def fouriersymbol(n, xrange, dtype=complex128, device=torch.device('cpu')):
    '''n         scalar int
    xrange    2 length array of reals
    creates the fourier symbol in a single dimension'''
    lf = 2 / (xrange[1] - xrange[0])
    o = tensor(np.mod(n, 2))
    c = 1j * pi * lf * arange(-(n - o) / 2, (n - o) / 2 + o, dtype=realifytype(dtype)).to(device)
    return c.to(complexifytype(dtype))


# batch revisit
def cfftmatrix(n, dtype=float64, device=torch.device('cpu')):
    id = eye(n, dtype=dtype, device=device)
    F = fftshift(fft(id, axis=0), dim=0) / sqrt(tensor(n))
    return F


def diffmatrix(k, n, xrange, dtype=complex128, device=torch.device('cpu')):
    '''one dimensional matrix'''
    xrange = fixrange(xrange, 1)[0]     # 1D
    F = cfftmatrix(n, dtype=dtype, device=device)
    symbol = fouriersymbol(n, xrange, dtype=dtype, device=device) ** k
    Dk = F.H @ diag(symbol) @ F
    if np.mod(k, 2) == 0:
        Dk = real(Dk)
    return Dk

