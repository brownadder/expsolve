# Copyright (c) 2019-2022 Pranav Singh
# Use of this source code is governed by the MIT license that can be found in the LICENSE file.


def stepper(t, h, u0, flowAu, flowBu, alpha, beta):
    u = u0
    tauA = t   # in time-ordered flows
    tauB = t   # time either moves with A or with B
    assert(len(alpha) == len(beta))
    for k in range(len(alpha)):
        if abs(alpha[k]) > 1e-14:
            u = flowAu(t, tauB, h, alpha[k], u)   
            tauA = tauA + h*alpha[k]
        if abs(beta[k]) > 1e-14:
            u = flowBu(t, tauA, h, beta[k], u)
            tauB = tauB + h*beta[k]
    return u


# alpha beta assumed underspecified by exactly 1 parameter
def consistent(a, b):
    x = list(a)
    y = list(b)
    alpha = x+[1.-sum(x)]
    beta = y+[1.-sum(y)]
    return alpha, beta


# expAu should be the first action
# len(a)=len(b) or len(a)=len(b)+1
# assuming that symmetric always needs to be consistent
def symmetric(a, b):
    x = list(a)
    y = list(b)
    nx = len(x)
    ny = len(y)
    assert (nx == ny) or (nx == ny + 1)

    xmid = 1.-2.*sum(x)
    ymid = 0.5-sum(y)
    X = x + [xmid] + x[::-1]
    Y = y + [ymid, ymid] + y[::-1]
    
    if nx == ny:
        alpha = Y
        beta = X + [0]
    else:
        alpha = X
        beta = Y + [0]

    return alpha, beta
