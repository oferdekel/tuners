#   Project: Tuners
#   Author: Ofer Dekel

import numpy as np
from multivariate_functions import MultivariateSin
from caching_oracle import CachingOracle

class GreedyContinuousSearch():

    def __init__(self, oracle, increment = 0.1):
        self.oracle = oracle
        self.dimension = oracle.get_dimension() 
        self.increment = increment
        self.best_config = np.zeros(oracle.get_dimension())
        self.oracle.query([self.best_config])
        x, self.best_value = self.oracle.get_best(0).popitem()

    def step(self, basis):

        basis *= self.increment

        previous_best = self.best_value

        x = np.clip(self.best_config + basis, -1, 1)
        queries = x.tolist()

        x = np.clip(self.best_config - basis, -1, 1)
        queries += x.tolist()

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
    STEP_SIZE = 0.1

    f = MultivariateSin(DIM, NUM_WAVES, PRE_AMPLITUDE_SCALE, POST_AMPLITUDE_SCALE, FREQUENCY_SCALE)
    o = CachingOracle(f)
    s = GreedyContinuousSearch(o, STEP_SIZE)
    basis = np.eye(DIM)

    for i in range(STEPS):
        print(s.step(basis))

if __name__ == '__main__':
    main()
