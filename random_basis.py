#   Project: Tuners
#   Author: Ofer Dekel

import numpy as np

def randomBasis(dim):
    X = np.random.randn(dim, dim)
    Q, R = np.linalg.qr(X)
    return Q
