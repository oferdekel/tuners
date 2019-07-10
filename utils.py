#   Project: Tuners
#   Author: Ofer Dekel

import numpy as np


def factor2(num):
    """ Returns integers (a,b) such that a >= b, a and b are as close as possible, and a * b = num. """

    a = int(np.ceil(np.sqrt(num)))
    while num % a > 0:
        a += 1 
    b = int(num / a)
    return (a, b)
