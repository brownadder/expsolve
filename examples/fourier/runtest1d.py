# Copyright (c) 2019-2022 Pranav Singh
# Use of this source code is governed by the MIT license that can be found in the LICENSE file.

# This .py file is a basic example. A more detailed example is available in tutorial1d.ipynb

import numpy as np
from scipy.sparse.linalg import expm
import matplotlib.pyplot as plt
import expsolve.fourier as fe

# Create a 1D grid
xrange = [0, 2*np.pi]
x = fe.grid1d(100, xrange)     #

# Differentiating a function using diffop 
s = np.sin(x)
ds = fe.diffop(0,1,s,xrange)   # 1st derivative using diffop
d2s = fe.diffop(0,2,s,xrange)  # 2st derivative using diffop


plt.plot(x, s)
plt.plot(x, np.real(ds))
plt.plot(x, np.real(d2s))
plt.show()

# Differentiating using a (full/dense) differentiation matrix (expensive)
D2 = fe.diffmatrix(2, 100, xrange)
plt.plot(np.real(D2.dot(s)))
plt.show()

# Three different ways to check error in differentiation
print(np.linalg.norm(D2.dot(s)-d2s))
print(fe.l2norm(D2.dot(s)-d2s, xrange))
print(np.sqrt(np.real(fe.l2inner(D2.dot(s)-d2s, D2.dot(s)-d2s, xrange))))

# The Schrodinger equation iu' = Hu, H = -L + V. Or u' = iLu -iVu.
n = 200 
xr = [-10, 10]
x = fe.grid1d(n, xr)                # discretise [-10,10] with 200 grid points

x0 = -2.0
u = np.exp(-(x-x0)**2/(2*0.25))     # initial condition: Gaussian
V = x**4 - 10*x**2                  # potential: double well 

# Splitting and composition
eLu = lambda h, u: fe.diffopexp(0, 2, 1j*h, u, xr)  # flow under the Laplacian (1d): iL
eVu = lambda h, u: np.exp(-1j*h*V)*u                # flow under the potential: -iV
strang = lambda h, u: eVu(h/2, eLu(h, eVu(h/2, u))) # Strang splitting of the flow under iL-iV

# Exact flow
D2 = fe.diffmatrix(2, n, xr)
H = -D2 + np.diag(V)                                # explicitly created Hamiltonian matrix
exact = lambda h, u: expm(-1j*h*H).dot(u)           # exact solution by brute force via expm

# Error in splitting for large time step
T = 0.5                                             # solve over [0,T]
uref = exact(T, u)                                  # exact solution by brute force via expm
ustrang = strang(T, u)                              # one step of Strang over [0,T], expect large error

plt.plot(x, np.abs(uref))
plt.plot(x, np.abs(ustrang))
plt.show()

print(fe.l2norm(uref-ustrang, xr))

# Splitting with small timestep
def runstrang(T, N, u0):
    u = u0
    h = T/N
    for i in range(N):
        u = strang(h, u)
    return u


plt.plot(np.abs(runstrang(T,1000,u)))   # 1000 steps of Strang splitting
plt.show()

# Check the rate of convergence of Strang splitting
Nlist = 2**np.arange(2,9)
hlist = T/Nlist
err = [fe.l2norm(uref-runstrang(T,N,u)) for N in Nlist]

print(err)
print(Nlist)
print(hlist)

plt.loglog(hlist, err)
plt.loglog(hlist, hlist**2) # the error is expected to be O(h^2)

plt.xlabel('time step')
plt.ylabel('L2 error')
plt.legend(['error in strang', 'O(h^2)'])
plt.show()