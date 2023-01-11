import numpy as np


# observables should be a dictionary
def evolve(u0, timegrid, stepper, observables={}, storeintermediate=False):
    obsvalues = {}
    for o in observables:
        obsvalues[o] = []

    uintermediate = []

    u = u0
    postprocess(0, u, uintermediate, storeintermediate, obsvalues, observables)

    for n in range(len(timegrid)-1):
        h = timegrid[n+1]-timegrid[n]
        t = timegrid[n]
        u = stepper(t, h, u)
        postprocess(t, u, uintermediate, storeintermediate, obsvalues, observables)

    return u, obsvalues, uintermediate


def postprocess(t, u, uintermediate, storeintermediate, obsvalues, observables):

    if storeintermediate:
        uintermediate.append(u)

    for o in observables:
        op = observables[o]
        obsvalues[o].append(op(u))


def timegrid(T, N):
    return np.linspace(0, T, int(N)+1)
