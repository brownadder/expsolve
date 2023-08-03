from torch import complex128

# renamed to complexify to avoid conflict with Python's complex
def complexify(u):
    return u.type(complex128)
