# Copyright (c) 2019-2022 Pranav Singh
# Use of this source code is governed by the MIT license that can be found in the LICENSE file.

# This .py file is a basic example. A more detailed example is available in tutorial1d.ipynb

import numpy as np
import matplotlib.pyplot as plt

import expsolve as es
import expsolve.discretize.tensorgrid as ex

from torch import matrix_exp, pi, real, sqrt, abs, sin, exp
from torch.linalg import norm

# Create a 1D grid
xrange = [0, 2*pi]
x = ex.grid1d(100, xrange)     #

# Differrentiating a function using diffop 
s = sin(x)
ds = ex.diffop(0, 1, s, xrange)   # 1st derivative using diffop
d2s = ex.diffop(0, 2, s, xrange)  # 2st derivative using diffop

ex.plot(plt, x, s)
ex.plot(plt, x, real(ds))
ex.plot(plt, x, real(d2s))
plt.show()

# Differrentiating using a (full/dense) differrentiation matrix (expensive)
D2 = ex.diffmatrix(2, 100, xrange)
ex.plot(plt, x, es.linalg.matmul(D2, s))
plt.show()

# Three differrent ways to check error in differrentiation
print(norm(es.linalg.matmul(D2, s) - d2s))
print(ex.l2norm(es.linalg.matmul(D2, s) - d2s, xrange))
print(sqrt(real(ex.l2inner(es.linalg.matmul(D2, s) - d2s, es.linalg.matmul(D2, s) - d2s, xrange))))

# The Schrodinger equation iu' = Hu, H = -L + V. Or u' = iLu -iVu.
n = 200 
xr = [-10, 10]
x = ex.grid1d(n, xr)                # discretise [-10,10] with 200 grid points

x0 = -2.0
u = ex.normalize( exp(-(x-x0)**2/(2*0.25)), xr)         # initial condition: Gaussian
V = x**4 - 10*x**2                                      # potential: double well 

# Splitting and composition
eLu = lambda h, u: ex.diffopexp(0, 2, 1j*h, u, xr)      # flow under the Laplacian (1d): iL
eVu = lambda h, u: exp(-1j*h*V)*u                       # flow under the potential: -iV
strang = lambda h, u: eVu(h/2, eLu(h, eVu(h/2, u)))     # Strang splitting of the flow under iL-iV

# Exact flow
D2 = ex.diffmatrix(2, n, xr)
H = -D2 + es.linalg.diag(V)                                    # explicitly created Hamiltonian matrix
exact = lambda h, u: es.linalg.matmul(matrix_exp(-1j*h*H), u)      # exact solution by brute force via matrix_exp

# Error in splitting for large time step
T = 0.5                                             # solve over [0,T]
uref = exact(T, u)                                  # exact solution by brute force via matrix_exp
ustrang = strang(T, u)                              # one step of Strang over [0,T], expect large error

ex.plot(plt, x, abs(uref))
ex.plot(plt, x, abs(ustrang))
plt.show()

print(ex.l2norm(uref-ustrang, xr))

# Splitting with small timestep
def runstrang(T, N, u0):
    u = u0
    h = T/N
    for i in range(N):
        u = strang(h, u)
    return u


ex.plot(plt, x, abs(runstrang(T, 1000, u)))   # 1000 steps of Strang splitting
plt.show()

# Check the rate of convergence of Strang splitting
Nlist = 2**np.arange(2, 9)
hlist = T/Nlist
err = [ex.l2norm(uref-runstrang(T, N, u))[0] for N in Nlist]

print(err)
print(Nlist)
print(hlist)

plt.loglog(hlist, err)
plt.loglog(hlist, hlist**2)     # the error is expected to be O(h^2)

plt.xlabel('time step')
plt.ylabel('L2 error')
plt.legend(['error in strang', 'O(h^2)'])
plt.show()