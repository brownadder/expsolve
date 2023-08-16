import torch

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


def plot(plt, x, y=None, separatelines=False, *args, **kwargs):
    x, y = _preprocess1D(x, y)
    if separatelines:
        nb = y.shape[0]
        for i in range(nb):
            plt.plot(x, y[i].flatten(), *args, **kwargs)
    else:
        mean_data = torch.mean(y, axis=0)
        variance_data = torch.std(y, axis=0)
        plt.plot(x, mean_data, *args, **kwargs)
        plt.fill_between(x, mean_data - variance_data, mean_data + variance_data,
                        alpha=0.2, *args, **kwargs)


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


