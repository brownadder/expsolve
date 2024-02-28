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


def _plothelper(plt, x, y=None, separatelines=False, *args, **kwargs):
    x, y = _preprocess1D(x, y)
    if separatelines or y.shape[0]==1:
        nb = y.shape[0]
        plist = [plt.plot(x, y[i].flatten(), *args, **kwargs) for i in range(nb)]
        handles = [(p[0],) for p in plist] 
    else:
        mean_data = torch.mean(y, axis=0)
        variance_data = torch.std(y, axis=0)
        pmean = plt.plot(x, mean_data, *args, **kwargs)
        plt.fill_between(x, mean_data - variance_data, mean_data + variance_data,
                        alpha=0.2, *args, **kwargs) 
        pvar = plt.fill(np.NaN, np.NaN, alpha=0.2, linewidth=0, *args, **kwargs)
        handles = [(pmean, pvar)] 
        
    return handles


def plot(ax, linespecs, y=None, separatelines=False, 
         xlim=None, ylim=None, xlabel=None, ylabel=None, 
         legend=True, grid=True, bgcolor='#FEFBF6', 
         xscale='linear', yscale='linear',
         *args, **kwargs):
    
    if bgcolor is not None:
            ax.set_facecolor(bgcolor)
        
    if xlim is not None:
        ax.set_xlim(xlim[0],xlim[1])

    if ylim is not None:
        ax.set_ylim(ylim[0],ylim[1])

    if grid:
        ax.grid(True)

    if xlabel:
        ax.set_xlabel(xlabel)
    
    if ylabel:
        ax.set_ylabel(ylabel)

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    if isinstance(linespecs, list):
        handles = []
        for linespec in linespecs:
            line = linespec[0]

            (x, y) = line
            if len(linespec)>=2:
                lineargs = linespec[1]
                h = _plothelper(ax, x, y, separatelines=separatelines, *args, **lineargs, **kwargs)[0][0]
            else:
                h = _plothelper(ax, x, y, separatelines=separatelines, *args, **kwargs)[0][0]
            
            if isinstance(h, list):
                h = h[0]

            handles.append(h)

        if legend:
            legendentries = [linespec[2] for linespec in linespecs]
            ax.legend(handles, legendentries)
        
        return handles
    else:
        return _plothelper(ax, linespecs, y=y, separatelines=separatelines, *args, **kwargs)

def obsplot(plt, timegrid, obsvalues, obsnames=None):
    if obsnames:
        obsspecs = [((timegrid, obsvalues[key]), {}, key) for key in obsnames]
    else:
        obsspecs = [((timegrid, x[1]), {}, x[0]) for x in obsvalues.items()]

    fig, ax = plt.subplots()
    plot(ax, linespecs=obsspecs, xlabel='t')
    plt.show()


def semilogy(*args, **kwargs):
    return plot(*args, yscale='log', **kwargs)


def semilogx(*args, **kwargs):
    return plot(*args, xscale='log', **kwargs)


def loglog(*args, **kwargs):
    return plot(*args, yscale='log', xscale='log', **kwargs)


# improve pre-processing
def imshow(plt, xrange, y, *args, **kwargs):
    assert dim(y) == 2
    assert y.shape[0] == 1
    region = list(fixrange(xrange, 2).flatten())
    plt.imshow(y.reshape(y.shape[1:]), extent=region)


