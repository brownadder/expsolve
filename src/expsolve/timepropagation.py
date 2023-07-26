#import numpy as np
import torch

# observables should be a dictionary
def evolve(u0, timegrid, stepper, observables={}, storeintermediate=False):
    obsvalues = {}
    for o in observables:
        obsvalues[o] = torch.zeros(len(timegrid), dtype=torch.float64)

    uintermediate = []

    u = u0
    postprocess(0, 0, u, uintermediate, storeintermediate, obsvalues, observables)

    for n in range(len(timegrid)-1):
        h = timegrid[n+1]-timegrid[n]
        t = timegrid[n]
        u = stepper(t, h, u)
        postprocess(n+1, t, u, uintermediate, storeintermediate, obsvalues, observables)

    return u, obsvalues, uintermediate


def postprocess(n, t, u, uintermediate, storeintermediate, obsvalues, observables):

    if storeintermediate:
        uintermediate.append(u)

    for o in observables:
        op = observables[o]
        obsvalues[o][n] = op(u)


def timegrid(T, N):
    return torch.linspace(0, T, int(N)+1, dtype=torch.float64)
