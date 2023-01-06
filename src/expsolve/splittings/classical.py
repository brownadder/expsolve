# Copyright (c) 2019-2022 Pranav Singh
# Use of this source code is governed by the MIT license that can be found in the LICENSE file.


def classical(h, u0, expAu, expBu, alpha, beta):
    u = u0
    assert(len(alpha) == len(beta))
    for k in range(len(alpha)):
        u = expAu(h*alpha[k], u)
        u = expBu(h*beta[k], u)
    return u