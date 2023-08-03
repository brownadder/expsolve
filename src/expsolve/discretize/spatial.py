import numpy as np

def fixrange(xrange, dims):
    '''Create a valid xrange to describe domain

    xrange
        input xrange can be
        -1            in which case set xrange to default: [-1, 1] in all dimensions
        list          this needs to be length 2 and is repeated across all dimensions
        ndarray       of size dims x 2 and is returned as it is
    dims          dimensions

    output        dims x 2 ndarray'''
    if isinstance(xrange, int) and xrange == -1:
        return np.array([[-1, 1] for d in range(dims)])
    else:
        if isinstance(xrange, list):
            assert len(xrange) == 2
            return np.array([xrange for d in range(dims)])
        else:
            return xrange
