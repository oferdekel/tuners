"""Simulates an oracle that takes a list of configuration vectors and 
measures the speed of each one.
"""

import random

def Oracle(configs):

    if not isinstance(configs, list):
        raise TypeError('not a list')

    return [random.randint(0,10) for x in configs]
