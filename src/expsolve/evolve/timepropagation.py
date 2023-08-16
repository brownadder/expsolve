import numpy as np
import torch
import matplotlib.pyplot as plt

# observables should be a dictionary
def solvediffeq(u0, timegrid, stepper, observables={}, storeintermediate=False):
    obsvalues = {}
    for o in observables:
        obsvalues[o] = torch.zeros((len(timegrid), u0.shape[0]), dtype=torch.float64)

    uintermediate = []

    u = u0
    postprocess(0, 0, u, uintermediate, storeintermediate, obsvalues, observables)

    for n in range(len(timegrid)-1):
        h = timegrid[n+1]-timegrid[n]
        t = timegrid[n]
        u = stepper(t, h, u)
        postprocess(n+1, t, u, uintermediate, storeintermediate, obsvalues, observables)

    for o in observables:
        obsvalues[o] = obsvalues[o].T

    return u, obsvalues, uintermediate


def postprocess(n, t, u, uintermediate, storeintermediate, obsvalues, observables):

    if storeintermediate:
        uintermediate.append(u)

    for o in observables:
        op = observables[o]
        obsvalues[o][n] = op(u)


def timegrid(trange, ndt):
    return torch.linspace(trange[0], trange[1], int(ndt)+1, dtype=torch.float64)


def order(u, uref, normfn, trange, ndtlist, stepper, showplot=True):
    assert len(ndtlist)>2

    hlist = (trange[1]-trange[0])/ndtlist
    err = [normfn(uref, solvediffeq(u, timegrid(trange, ndt), stepper)[0])[0] for ndt in ndtlist]

    ord2 = (np.round(2.* np.log(err[-3]/err[-1])/np.log(hlist[-3]/hlist[-1]))).numpy().item()
    ord = ord2/2.

    c = (err[-1]/hlist[-1]**ord)/5.

    if showplot:
        plt.loglog(hlist, err)
        plt.loglog(hlist, c * hlist**ord, ':k')
        plt.xlabel('time step')
        plt.ylabel('L2 error')
        if np.mod(ord2,2)==1:
            ords = f'{ord:2.1f}'
        else:
            ords = f'{ord:1.0f}'

        plt.legend(['error in Strang', f'O(h^{ords})'])

    return ord
