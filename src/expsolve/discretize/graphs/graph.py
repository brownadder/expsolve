# Copyright (c) 2019-2024 Pranav Singh
# Use of this source code is governed by the MIT license that can be found in the LICENSE file.

# This file is common to all graph instance level operations
# These should not be operators acting on grids 

import numpy as np

import torch
from torch import float64, meshgrid, linspace, sum, real, is_tensor, is_complex
from torch.linalg import norm

from ...linalg import matmul
from ...utils import complexify

from ..spatial import fixrange
from itertools import product



def honeycombgrid(h, n, active = lambda x, y: True, cx = [0., 0.]):
    cx = np.array(cx)
    c = (h/np.sqrt(3))*np.array([(col * 3/2, (row + (1/2 if col%2==1 else 0)) * np.sqrt(3)) for (col, row) in product(range(n[0]), range(n[1]))])
    ct = c.T
    cr = np.array([[np.min(ct[0]), np.max(ct[0])], [np.min(ct[1]), np.max(ct[1])]])
    cav = np.array([(cr[0][0] + cr[0][1])/2, (cr[1][0] + cr[1][1])/2])
    c  = c - cav.T + cx.T
    centers = np.array([(x,y) for (x,y) in c if active(x,y)])

    ct = centers.T
    cr = np.array([[np.min(ct[0]), np.max(ct[0])], [np.min(ct[1]), np.max(ct[1])]])
    xr =  cr + h*np.array([[-1/np.sqrt(3)-0.05, 1/np.sqrt(3)+0.05], [-0.5, 0.5]])

    nc = len(centers)
    
    centers = torch.tensor(centers)

    c = centers.T
    cx = c[0].unsqueeze(0)
    cy = c[1].unsqueeze(0)
    dist = torch.sqrt((cx - cx.T)**2 + (cy - cy.T)**2)

    return centers, torch.tensor(xr), nc, dist


