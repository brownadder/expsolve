import torch
from torch import complex128


def diag(v):
    return torch.diag(v.flatten())


# batch revisit : broadcast in both directions: if either v or A is longer in dim 0
def matmul(A, v):
    shp = v.shape
    u = v.reshape([shp[0], -1]).mT

    # nb = shp[0]     # number of batches
    if A.dtype == complex128 or v.dtype == complex128:
        return torch.matmul(A.type(complex128), u.type(complex128)).mT.reshape(shp)
    else:
        return torch.matmul(A, u).mT.reshape(shp)