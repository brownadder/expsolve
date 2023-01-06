import numpy as np
import matplotlib.pyplot as plt
import expsolve as es

n = 200
L = 10
xr = [-L, L]
x = es.fourier.grid1d(n, xr)

x0 = -2.0
u = np.exp(-(x-x0)**2/(2*0.25))

u = u/es.fourier.l2norm(u, xr)

V = x**4 - 10*x**2

eLu = lambda h, u: es.fourier.diffopexp(0, 2, 1j*h, u, xr)
eVu = lambda h, u: np.exp(-1j*h*V)*u

trotterstep = lambda h, u0: es.splittings.classical(h, u0, eLu, eVu, [1.], [1.])
strangstep = lambda h, u0: es.splittings.classical(h, u0, eLu, eVu, [0.,1.], [0.5,0.5])

a = [0.0792036964311957, 0.353172906049774, -0.0420650803577195]
a4 = 1 - 2*sum(a)
alpha = a + [a4] + a[::-1]

b = [0.209515106613362, -0.143851773179818]
b3 = 0.5 - sum(b)
beta = b + [b3, b3] + b[::-1] + [0]
blanesmoanstep = lambda h, u0: es.splittings.classical(h, u0, eLu, eVu, alpha, beta)



observables = {'energy': lambda u: es.fourier.observable(lambda psi: -es.fourier.diffop(0, 2, psi, xr) + V*psi, u, xr), 
'position': lambda u: es.fourier.observable(lambda psi: x*psi, u, xr), 
'momentum': lambda u: es.fourier.observable(lambda psi: 1j*es.fourier.diffop(0, 1, psi, xr), u, xr), 
'kinetic': lambda u: es.fourier.observable(lambda psi: -es.fourier.diffop(0, 2, psi, xr), u, xr), 
'potential':lambda u: es.fourier.observable(lambda psi: V*psi, u, xr)}

T = 1
N = 1000
timegrid = es.timegrid(T, N)

trotterevolve = es.evolve(u, timegrid, trotterstep, observables)
strangevolve = es.evolve(u, timegrid, strangstep, observables)
blanesmoanevolve = es.evolve(u, timegrid, blanesmoanstep, observables)

obsvalues_trotter = trotterevolve[1]
obsvalues_strang = strangevolve[1]
obsvalues_blanesmoan = blanesmoanevolve[1]

plt.figure()

E0 = obsvalues_trotter['energy'][0]

plt.semilogy(timegrid, np.abs(obsvalues_trotter['energy']-E0))
plt.semilogy(timegrid, np.abs(obsvalues_strang['energy']-E0))
plt.semilogy(timegrid, np.abs(obsvalues_blanesmoan['energy']-E0))
plt.legend({'Trotter','Strang','Blanes-Moan'})
plt.ylabel('change in energy')
plt.xlabel('time')
plt.show()