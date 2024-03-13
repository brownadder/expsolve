import numpy as np
import torch
import matplotlib.pyplot as plt

# observables should be a dictionary
def solvediffeq(u0, timegrid, stepper, observables={}, storeintermediate=False, aux=None):
    obsvalues = {}
    for o in observables:
        obsvalues[o] = torch.zeros((len(timegrid), u0.shape[0]), dtype=torch.float64)

    uintermediate = []

    u = u0
    postprocess(0, timegrid[0], u, uintermediate, storeintermediate, obsvalues, observables)

    for n in range(len(timegrid)-1):
        h = timegrid[n+1]-timegrid[n]
        t = timegrid[n]
        if aux is None:
            u = stepper(t, h, u)
        else:
            u, aux = stepper(t, h, u, aux)
        postprocess(n+1, t+h, u, uintermediate, storeintermediate, obsvalues, observables)

    for o in observables:
        obsvalues[o] = obsvalues[o].T

    return u, obsvalues, uintermediate


def postprocess(n, t, u, uintermediate, storeintermediate, obsvalues, observables):

    if storeintermediate:
        uintermediate.append(u)

    for o in observables:
        op = observables[o]
        obsvalues[o][n] = op(u, t)


def timegrid(trange, ndt):
    return torch.linspace(trange[0], trange[1], int(ndt)+1, dtype=torch.float64)


def order(u, uref, normfn, trange, ndtlist, steppers, showplot=True, aux=None):
    assert len(ndtlist) > 2

    hlist = (trange[1]-trange[0])/ndtlist

    ord = {}
    err = {}
    c = {}
    ord2 = {}
    ords = {}
    for methodname in steppers:
        stepper = steppers[methodname]

        err[methodname] = [normfn(uref, solvediffeq(u, timegrid(trange, ndt), stepper, aux=aux)[0])[0] for ndt in ndtlist]

        ord2[methodname] = (np.round(2.* np.log(err[methodname][-3]/err[methodname][-1])/np.log(hlist[-3]/hlist[-1]))).numpy().item()
        ord[methodname] = ord2[methodname]/2.

        c[methodname] = (err[methodname][-1]/hlist[-1]**ord[methodname])/2.
        if np.mod(ord2[methodname], 2) == 1:
            ords[methodname] = f'{ord[methodname]:2.1f}'
        else:
            ords[methodname] = f'{ord[methodname]:1.0f}'

    if showplot:
        leg = []
        for methodname in steppers:
            plt.loglog(hlist, err[methodname])
            plt.loglog(hlist, c[methodname] * hlist**ord[methodname], ':k')
            leg += [f'error in {methodname}', f'O(h^{ords[methodname]})']
   
        plt.xlabel('time step')
        plt.ylabel('L2 error')
        plt.legend(leg)
        plt.grid(True)

    return ord, err
