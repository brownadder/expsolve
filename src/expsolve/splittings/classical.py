# Copyright (c) 2019-2022 Pranav Singh
# Use of this source code is governed by the MIT license that can be found in the LICENSE file.


def stepper(h, u0, expAu, expBu, alpha, beta):
    u = u0
    assert(len(alpha) == len(beta))
    for k in range(len(alpha)):
        u = expAu(h*alpha[k], u)
        u = expBu(h*beta[k], u)
    return u


# alpha beta assumed underspecified by exactly 1 parameter
def consistent(a, b):
    alpha = a+[1.-sum(a)]
    beta = b+[1.-sum(b)]
    return alpha, beta


# assume expAu outside and expBu inside
# assume symmetric always needs to be consistent
def symmetric(a, b):
    amid = 1.-2.*sum(a)
    bmid = 0.5-sum(b)
    alpha = a + [amid] + a[::-1]
    beta = b + [bmid, bmid] + b[::-1] + [0]
    return alpha, beta
