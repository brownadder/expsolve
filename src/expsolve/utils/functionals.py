from torch import imag, real, arctan2, pi

phase = lambda u: arctan2(imag(u),real(u))
phasescaled = lambda u: 0.5+phase(u)/(2*pi)