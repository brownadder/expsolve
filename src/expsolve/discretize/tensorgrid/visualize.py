import torch
import numpy as np

from ..spatial import fixrange

from .grid import dim


def _preprocess1D(x, y=None):
    X = x
    if y is None:
        y = x
        n = x.shape[1]
        X = torch.tensor(range(1, n+1))
    X = X.detach().cpu().flatten()
    y = y.detach().cpu()
    return X, y


def plot(plt, x, y=None, legend=None, separatelines=False, *args, **kwargs):
    x, y = _preprocess1D(x, y)
    if separatelines or y.shape[0]==1:
        nb = y.shape[0]
        plist = [plt.plot(x, y[i].flatten(), *args, **kwargs) for i in range(nb)]
        legendhandles = [(p[0],) for p in plist] 
    else:
        mean_data = torch.mean(y, axis=0)
        variance_data = torch.std(y, axis=0)
        pmean = plt.plot(x, mean_data, *args, **kwargs)
        plt.fill_between(x, mean_data - variance_data, mean_data + variance_data,
                        alpha=0.2, *args, **kwargs)
        pvar = plt.fill(np.NaN, np.NaN, alpha=0.2, linewidth=0, *args, **kwargs)
        legendhandles = [(pmean[0], pvar[0])]
    
    return legendhandles


def obsplot(plt, x, y=None, obsnames=None):
    timegrid = True
    if not obsnames:
        # interpret as (plt, x, y)
        obsnames = y
        y = x
        timegrid = False
    elif not y:
        # interpret as (plt, y)
        y = x
        obsnames = None
        timegrid = False

    if not obsnames:
        obsnames = list(y.keys())

    legendhandles = []
    for o in obsnames:
        if timegrid:
            lh = plot(plt, x, y[o])
        else:
            lh = plot(plt, y[o])
        
        legendhandles.append(lh[0])

    plt.legend(legendhandles, obsnames)


def semilogy(plt, x, y=None, *args, **kwargs):
    x, y = _preprocess1D(x, y)
    nb = y.shape[0]
    for i in range(nb):
        plt.semilogy(x, y[i].flatten(), *args, **kwargs)


# improve pre-processing
def imshow(plt, xrange, y, *args, **kwargs):
    assert dim(y) == 2
    assert y.shape[0] == 1
    region = list(fixrange(xrange, 2).flatten())
    plt.imshow(y.reshape(y.shape[1:]), extent=region)


