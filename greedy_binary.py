#   Project: Tuners
#   Author: Ofer Dekel

import numpy as np
from multivariate_functions import MultivariateSin
from caching_oracle import CachingOracle

class GreedyBinarySearch():

    def __init__(self, oracle):
        self.oracle = oracle
        self.best_config = np.zeros(oracle.get_dimension())
        queries = [self.best_config]
        self.oracle.query(queries)
        x, self.best_value = self.oracle.get_best(0).popitem()

    def step(self):
        previous_best = self.best_value

        queries = []
        for i in range(self.oracle.get_dimension()):
            copy = np.array(self.best_config)
            copy[i] = 1 - copy[i]
            queries.append(copy)

        self.oracle.query(queries)
        config, value = self.oracle.get_best(0).popitem()

        if value < self.best_value:
            self.best_value = value
            self.best_config = config

        return previous_best - self.best_value

    def get_best(self, tolerance=0):
        return self.oracle.get_best(tolerance)


def main():
    DIM = 20
    NUM_WAVES = 7
    PRE_AMPLITUDE_SCALE = 1
    POST_AMPLITUDE_SCALE = 0.45
    FREQUENCY_SCALE = 1
    STEPS = 10

    f = MultivariateSin(DIM, NUM_WAVES, PRE_AMPLITUDE_SCALE, POST_AMPLITUDE_SCALE, FREQUENCY_SCALE)
    o = CachingOracle(f)
    s = GreedyBinarySearch(o)
    
    for i in range(STEPS):
        print(s.step())

if __name__ == '__main__':
    main()
