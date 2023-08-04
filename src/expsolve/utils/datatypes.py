from torch import complex128, complex64, complex32, float64, float32, float16


def complexifytype(dtype):
    if dtype == float64 or dtype == complex128:
        return complex128
    elif dtype == float32 or dtype == complex64:
        return complex64
    elif dtype == float16 or dtype == complex32:
        return complex32


# renamed to complexify to avoid conflict with Python's complex
def complexify(u):
    return u.type(complexifytype(u.dtype))
